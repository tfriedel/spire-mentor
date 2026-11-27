//! `CombatState` → the wire dict, plus the bridge-shape action /
//! pending-choice dicts and the hand previews.
//!
//! [`combat_state_to_value`] is the inverse of [`crate::state_from_wire`]. It
//! emits the canonical wire schema (`docs/sts2-hint-state-schema.json`, the
//! same shape [`crate::parse_state_dict`] accepts on the way in).
//!
//! Lossy by design — see the `state_equiv()` test predicate in
//! `sts2-py/tests/round_trip.rs` for the exhaustive list of fields
//! intentionally NOT round-tripped. The significant ones:
//!
//!   * `rng` (`Xoshiro256PlusPlus`) — only the wire-schema-recorded
//!     `rng.seed` survives; the in-process cursor is opaque.
//!   * `combat_log` (`TurnEvents`) — append-only buffer, not in the
//!     wire format.
//!   * Hook listeners — closure state on the relic / power buses,
//!     reconstructed lazily from the relic/power kinds on the next
//!     `apply()`.
//!   * `intent_history` — Rust's internal shape is private; the wire
//!     format only carries an i32-encoded summary which we re-emit as
//!     an empty list because `state_from_dict_impl` defaults it anyway.
//!
//! All other wire-schema fields are bit-faithful: the parsed state
//! emitted by `parse_state_dict(to_dict(s))` should match the parsed
//! state that produced `s`, modulo the dropped fields above.

use std::collections::BTreeMap;

use serde_json::Value;

use sts2_core::cards::{Card, CardType, VarKey};
use sts2_core::{
    Action, CardUid, CharacterId, ChoiceOption, CombatPhase, CombatState, CreatureUid, Intent,
    MonsterKind, Orb, PendingChoice, Power, PowerKind, PowerUid, Side, DARK_ORB_BASE_EVOKE,
};

use super::map_ext::{obj, set, val};
use super::names::{card_type_python_name, choice_kind_python_name};
use super::EmitError;
use crate::state_dict::{card_affliction_kind_wire_name, var_key_wire_name, WIRE_VERSION};

const fn combat_phase_to_str(p: CombatPhase) -> &'static str {
    match p {
        CombatPhase::Setup => "Setup",
        CombatPhase::PlayerTurnStart => "PlayerDraw",
        CombatPhase::PlayerPlayPhase
        | CombatPhase::PlayerEndTurnPhaseOne
        | CombatPhase::PlayerEndTurnPhaseTwo => "PlayerPlayPhase",
        CombatPhase::EnemyTurnStart | CombatPhase::EnemyTurnExec | CombatPhase::EnemyTurnEnd => {
            "EnemyPhase"
        }
        CombatPhase::Resolved { player_won: true } => "Victory",
        CombatPhase::Resolved { player_won: false } => "Defeat",
    }
}

const fn side_to_str(s: Side) -> &'static str {
    match s {
        Side::Player => "Player",
        Side::Enemy => "Enemy",
    }
}

const fn class_str_from_character(c: Option<CharacterId>) -> &'static str {
    match c {
        // `None` (no character gating set yet) defaults to "ironclad" so
        // callers don't see a schema rejection.
        Some(CharacterId::Ironclad) | None => "ironclad",
        Some(CharacterId::Silent) => "silent",
        Some(CharacterId::Defect) => "defect",
        Some(CharacterId::Necrobinder) => "necrobinder",
        Some(CharacterId::Regent) => "regent",
        Some(CharacterId::Fool) => "fool",
    }
}

/// Serialize a single [`Card`] runtime POD into the wire-format
/// `CardSpec` dict.
///
/// `state` is needed to resolve the card's *effective* energy cost
/// (base cost plus any active cost modifiers); the resolved value is
/// emitted as the additive `cost_override` field so display consumers
/// (e.g. the workbench combat panel) can show the real cost without
/// re-deriving it. `cost_override` is not a required CardSpec key, so
/// emitting it is backward-compatible and ignored on round-trip.
#[must_use]
pub fn card_to_value(state: &CombatState, card: &Card) -> Value {
    let mut d = obj();
    set(&mut d, "id", card.card_id.to_python_name());
    // `stable_idx` is wire-only metadata; we emit the uid as a
    // best-effort proxy since the Rust port doesn't carry a separate
    // stable index — the uid IS the stable handle within a combat.
    set(
        &mut d,
        "stable_idx",
        i32::try_from(card.uid.get()).unwrap_or(0),
    );
    let kind = match card.card_type {
        CardType::Status => "status",
        CardType::Curse => "curse",
        _ => "card",
    };
    set(&mut d, "kind", kind);
    set(&mut d, "upgraded", card.upgraded);
    // GH #1391: a card that upgrades more than once (Dingy Dagger) carries
    // its level; every other card's dict is unchanged.
    if card.max_upgrade_level() > 1 && card.upgraded {
        set(&mut d, "upgrade_level", i32::from(card.upgrade_level()));
    }
    set(
        &mut d,
        "cost_override",
        sts2_core::hooks::effective_card_cost(state, card.uid),
    );
    // The instance's own cost (C# `GetWithModifiers(Local)`, the same number
    // the C# producers send as `base_cost`), separate from the effective one
    // above. `cost_mods` carries the model behind it — the base and the
    // ordered local modifiers with their lifetimes — so a rebuild expires
    // each modifier on schedule instead of keeping a this-turn discount for
    // good. Omitted for the common unmodified card, where `base_cost` says
    // everything.
    let own_cost = if card.base_cost() < 0 {
        i32::from(card.base_cost())
    } else {
        card.local_cost().max(0)
    };
    set(&mut d, "base_cost", own_cost);
    if card.cost.has_local_modifiers() {
        let mut cm = obj();
        set(&mut cm, "base", i32::from(card.base_cost()));
        let mods: Vec<Value> = card
            .cost
            .mods()
            .iter()
            .map(|m| Value::from(vec![i32::from(m.amount()), i32::from(m.flags())]))
            .collect();
        cm.insert("mods".to_owned(), Value::Array(mods));
        d.insert("cost_mods".to_owned(), Value::Object(cm));
    }
    if !card.combat_vars.is_empty() {
        let mut cv = obj();
        for (key, value) in &card.combat_vars {
            set(&mut cv, var_key_wire_name(key), value);
        }
        d.insert("combat_vars".to_owned(), Value::Object(cv));
    }
    if card.base_replay_count != 0 {
        set(&mut d, "base_replay_count", card.base_replay_count);
    }
    // GH #1197: the star half of the cost, for the cards that have one.
    // `base_star_cost` is the printed cost in the state's build (C#
    // `CanonicalStarCost`); `star_cost_live` is what a play would spend now
    // (C# `GetStarCostWithModifiers`: the free-this-turn / this-combat
    // riders, Void Form, Brilliant Scarf). Star-X (Stardust, `HasStarCostX`)
    // reads -1 in both. Absent on every card without a star cost — read as 0.
    //
    // Deliberately NOT the existing `star_cost` key: the C# producers send
    // that as `CurrentStarCost` (no hooks) and the parse turns a 0 there into
    // a free-this-turn rider, so a Void-Form 0 under that name would survive
    // a round trip as a permanent discount. The parse ignores both keys.
    if card.card_id == sts2_core::cards::CardId::Stardust {
        set(&mut d, "star_cost_live", -1);
        set(&mut d, "base_star_cost", -1);
    } else {
        let printed = sts2_core::card_printed_star_cost(card.card_id, state.game_version);
        if printed > 0 {
            set(
                &mut d,
                "star_cost_live",
                sts2_core::effective_star_cost_in_combat(state, card),
            );
            set(&mut d, "base_star_cost", printed);
        }
    }
    // Base damage / block let display consumers fill the `{Damage}` /
    // `{Block}` template vars in card descriptions. Additive (not in
    // REQUIRED_CARDSPEC_KEYS); 0 for cards that deal/block neither.
    set(&mut d, "base_damage", card.base_damage);
    set(&mut d, "base_block", card.base_block);
    // The card face's runtime-calculated value (Ashen Strike's 6 becomes
    // 6 + 3 x <cards in the Exhaust Pile>), for the ~40 cards that state
    // one. Additive, and ABSENT for every other card: consumers must read a
    // missing key as "this card has no such number", never as 0 — `(Hits 0
    // times)` on a card face reads as a fact, and zero is never the truth
    // for these on a live board. Untargeted (`None`), which is what the
    // game itself shows until the player hovers an enemy.
    if let Some(v) = sts2_content::cards::preview::calculated_value(state, card, None) {
        set(&mut d, "calculated_value", v);
    }
    // Instance keyword flags, in the C# producers' vocabulary
    // (`VALID_CARD_FLAGS`), so the mirror's `_card_key` can compare them —
    // with neither side emitting flags, the diff was structurally BLIND to
    // every keyword divergence. ONLY the override/registry-derived subset is
    // emitted: the C# producer names flags only on HAND cards and never
    // names the free riders at all, so a model-derived emission (FREE_THIS_TURN,
    // GhostSeed's EtherealThisCombat) is knowledge only the mirror's own
    // model holds — emitting it manufactured a per-decision tuple desync on
    // every generated-free card (the v29wr hand/discard explosion: mirror
    // `('free_turn',)` vs world `()`). The vars still DO their behavioral
    // work; they are just not part of the comparison vocabulary.
    //
    // The keyword `retain` stays unemitted, as it has since this block
    // landed. That IS an asymmetry — the adapter maps the C# keyword onto a
    // `retain` flag — but it is pre-existing, orthogonal to GH #370, and
    // making it symmetric would start surfacing engine-vs-C# Retain-registry
    // divergences as fresh desyncs. Worth doing deliberately, not as a side
    // effect of this change.
    //
    // The Sly/Retain trio is emitted in `flags` only for cards IN HAND (every
    // other pile carries it under the separate `pile_flags` key, GH #1197),
    // because that is where the adapter computes flags — the C# producers
    // ship keywords on pile cards too, but `_flags_for_card` is called for
    // the hand alone. An
    // asymmetric emit here would manufacture a per-decision desync on every
    // Well-Laid-Plans / Hand-Trick / MasterPlanner card, which is exactly the
    // failure the paragraph above records for the free riders. GH #370.
    {
        let mut flags: Vec<&str> = Vec::new();
        // GH #372. Two independent sources make a card Ethereal and the emit
        // must union them, because the PARSE already does:
        //
        //   * `card_ethereal_inst` — the permanent keyword (per-instance
        //     `keyword_override`, else the static registry).
        //   * `VarKey::EtherealThisCombat` — the per-combat grant. GhostSeed
        //     stamps it on every basic Strike/Defend
        //     (`relics/shop_event.rs:1964`), and `state_from_dict` maps the
        //     wire's `ethereal` flag onto this same var (`:2290`).
        //
        // Emitting only the first made the round trip LOSSY in one direction:
        // a GhostSeed run's wire arrives with `flags: ["ethereal"]`, parses
        // into the combat var, and comes back out with no flag at all —
        // verified by round-tripping seed 1002's hand. Since `_card_key`
        // diffs the flag tuple across all five piles, that is a per-decision
        // desync on every basic Strike/Defend for the whole fight.
        //
        // Symmetric emit/parse is the invariant here; see the sibling note on
        // the free riders below.
        //   * A live Hex on a Hexed card (C# `HexPower.TryModifyKeywordsInCombat`):
        //     the game's board shows it, and the parse leaves it to the
        //     affliction (GH #1707), so the round trip holds.
        if sts2_core::card_effects::card_ethereal_inst(card)
            || card.combat_var(VarKey::EtherealThisCombat) == 1
            || (sts2_core::combat_loop::card_is_hexed(card)
                && state.player_power_amount(sts2_core::PowerKind::Hex) > 0)
        {
            flags.push("ethereal");
        }
        if sts2_core::card_effects::card_unplayable_inst(card) {
            flags.push("unplayable");
        }
        if sts2_core::card_effects::card_innate_inst(card) {
            flags.push("innate");
        }
        // Mirrors the adapter: keyword first, then the single-turn halves
        // with the keyword subtracted, so a card that is BOTH intrinsically
        // Sly and Hand-Tricked emits `sly` alone — as C# does, since
        // `IsSlyThisTurn` short-circuits on the keyword. Intrinsic Sly is
        // read in the state's build (`card_sly_in`: v0.111.0 Haze lost
        // it), the same predicate the parse uses to tell intrinsic from
        // granted; a version-blind emit wrote `sly` on every v0.111.0
        // Haze and the parse read that back as a MasterPlanner grant.
        let mut sly_retain: Vec<&str> = Vec::new();
        let sly_keyword = sts2_core::card_effects::card_sly_in(card.card_id, state.game_version)
            || state.player.sly_this_combat.contains(&card.uid);
        if sly_keyword {
            sly_retain.push("sly");
        } else if state.player.sly_this_turn.contains(&card.uid) {
            sly_retain.push("sly_turn");
        }
        if !sts2_core::card_effects::card_retain_inst(card)
            && state.player.retain_this_turn.contains(&card.uid)
        {
            sly_retain.push("retain_turn");
        }
        if state.player.hand.contains(&card.uid) {
            flags.extend(sly_retain);
        } else if !sly_retain.is_empty() {
            // GH #1197: the same trio for a card in any OTHER pile (a Master
            // Planner Sly card that was discarded, drawn or shuffled), under
            // its own key so `flags` — which the mirror's `_card_key` diffs
            // and the C# producers name on hand cards only — stays exactly
            // what it was. Read `flags` on hand cards and `pile_flags` on
            // every other card for the full picture.
            set(&mut d, "pile_flags", sly_retain);
        }
        if !flags.is_empty() {
            set(&mut d, "flags", flags);
        }
    }
    // Enchantments, in the C# producers' spelling so both dialects of this
    // field read the same. The engine allows a stack of several; the wire
    // shape (one `enchantment` + one `enchantment_amount`) is C#'s, which
    // holds at most one — emit the first and let the rest be lossy rather
    // than invent a shape no consumer parses.
    if let Some((id, amount)) = card.enchantments.first() {
        set(&mut d, "enchantment", id.wire_name());
        set(&mut d, "enchantment_amount", amount);
    }
    // Per-card affliction (Bound/Devoured/…): additive optional fields
    // so afflicted cards survive the round-trip — without them a
    // reconstructed ChainsOfBinding state believes every card playable.
    if let Some(aff) = card.affliction {
        set(
            &mut d,
            "affliction",
            card_affliction_kind_wire_name(aff.kind),
        );
        set(&mut d, "affliction_stacks", aff.stacks);
        if aff.applied_exhaust {
            set(&mut d, "affliction_applied_exhaust", true);
        }
    }
    Value::Object(d)
}

/// `cards_played_this_turn` on the wire: the historical five buckets
/// (Attack..Status), plus the Quest bucket only when it is non-zero, so every
/// existing producer, consumer and golden keeps its shape (GH #1111; the
/// parser accepts both lengths).
fn play_buckets_wire(buckets: &[u32; 6]) -> Vec<u32> {
    let quest = sts2_core::cards::CardType::Quest.index();
    if buckets[quest] == 0 {
        buckets[..quest].to_vec()
    } else {
        buckets.to_vec()
    }
}

fn pile_to_value(
    state: &CombatState,
    uids: &[CardUid],
    cards: &[Card],
    path: &str,
) -> Result<Value, EmitError> {
    let mut list = Vec::with_capacity(uids.len());
    for uid in uids {
        let card = cards
            .iter()
            .find(|c| c.uid == *uid)
            .ok_or_else(|| EmitError::at(path, &format!("uid {} not in arena", uid.get())))?;
        list.push(card_to_value(state, card));
    }
    Ok(Value::Array(list))
}

/// Per-stream RNG read counts, keyed by the `RunRngType` snake_case names.
///
/// Emitted twice, deliberately from one definition: as the top-level
/// `rng_counters` *diagnostic* (a counter mismatch in a real-vs-real replay
/// names the exact stream and read-count where two engines desynced, which a
/// value-only field diff cannot show when the desync hides behind a constant),
/// and inside `$.rng.counters`, where it is *load-bearing* — that copy is what
/// `state_from_dict` restores the streams from, putting a reconstruction on
/// the same die roll the producer is about to make.
///
/// `unknown_map_point` and `treasure_room` are omitted: `state_from_dict`
/// accepts them, but no combat-scoped producer moves them, so shipping them on
/// every combat wire would be noise.
#[must_use]
pub fn run_rng_counters(state: &CombatState) -> Value {
    let rng = &state.run_rng;
    let mut d = obj();
    set(&mut d, "up_front", rng.up_front.counter());
    set(&mut d, "shuffle", rng.shuffle.counter());
    set(
        &mut d,
        "combat_card_generation",
        rng.combat_card_generation.counter(),
    );
    set(
        &mut d,
        "combat_potion_generation",
        rng.combat_potion_generation.counter(),
    );
    set(
        &mut d,
        "combat_card_selection",
        rng.combat_card_selection.counter(),
    );
    set(
        &mut d,
        "combat_energy_costs",
        rng.combat_energy_costs.counter(),
    );
    set(&mut d, "combat_targets", rng.combat_targets.counter());
    set(&mut d, "monster_ai", rng.monster_ai.counter());
    set(&mut d, "niche", rng.niche.counter());
    set(&mut d, "combat_orbs", rng.combat_orbs.counter());
    // PlayerRng-side streams (C# `PlayerRng.{Rewards,Shops,Transformations}`),
    // surfaced so the lockstep harness can diff the shared `rewards` stream
    // read-count per step — the exact tool for the shared-Rewards
    // roll-interleaving divergence class (act-2 potion drop / deck |0|1).
    set(&mut d, "rewards", rng.rewards.counter());
    set(&mut d, "shops", rng.shops.counter());
    set(&mut d, "transformations", rng.transformations.counter());
    // v0.113.0 `PlayerRng.NonCombatGeneration`; emitted only for the builds
    // that have it so older wire goldens keep their key set.
    if rng.game_version >= sts2_core::GameVersion::V0_113_0 {
        set(
            &mut d,
            "non_combat_generation",
            rng.non_combat_generation.counter(),
        );
    }
    Value::Object(d)
}

/// One [`sts2_core::hooks::DamageModifier`] row → `{name, kind, side, add?, mul?}`.
///
/// `name` is the engine registry id (`WEAK`, `STRENGTH`, `BURNING_BLOOD`) —
/// the same vocabulary [`powers_to_value`] emits, so a consumer resolves a
/// title and an icon for a modifier exactly as it does for a power. `side`
/// says whose listener it is (`"enemy"` = the attacker's, `"player"` = the
/// target's), which is what separates "the attacker's Strength" from "your
/// Vulnerable" in a sentence.
///
/// `add` / `mul` are present only for the channel this listener actually
/// moved, so a consumer can print `+3` or `×3/4` without deciding which of
/// three identity defaults to suppress. `mul` is the raw `[num, den]`
/// rational the engine composes — NOT a float: the pipeline floors exactly
/// once at the end, and a `0.75` in the payload invites a consumer to
/// reconstruct that with float multiplication and land a different number.
#[must_use]
pub fn damage_modifier_to_value(
    state: &CombatState,
    m: &sts2_core::hooks::DamageModifier,
) -> Value {
    use sts2_core::hooks::{CreatureRef, DamageModifierSource};
    let mut d = obj();
    let (kind, name) = match m.source {
        DamageModifierSource::Power(k) => ("power", k.python_name()),
        DamageModifierSource::Relic(k) => ("relic", k.python_name()),
        DamageModifierSource::Other => ("other", "OTHER"),
    };
    set(&mut d, "kind", kind);
    set(&mut d, "name", name);
    let side = match m.owner {
        Some(CreatureRef::Player) => Some("player"),
        Some(CreatureRef::Enemy(uid)) => {
            // The player's own pet is player-side for the reader's purposes
            // even though the arena stores it as a creature.
            if state.player.pet_uid == Some(uid) {
                Some("player")
            } else {
                Some("enemy")
            }
        }
        Some(CreatureRef::Ally(_)) => Some("player"),
        _ => None,
    };
    if let Some(side) = side {
        set(&mut d, "side", side);
    }
    if m.additive != 0 {
        set(&mut d, "add", m.additive);
    }
    if m.mul != sts2_core::hooks::DamageMul::ONE {
        // A LIST, not a tuple: this dict is consumed both in-process (workbench
        // tests, the hint adapter) and as JSON by the browser, and a tuple
        // makes those two disagree on the Python side of an `==`.
        set(&mut d, "mul", vec![m.mul.num, m.mul.den]);
    }
    if m.cap != i32::MAX {
        set(&mut d, "cap", m.cap);
    }
    Value::Object(d)
}

/// Serialize a single [`Intent`] into the wire-format intent dict:
/// `{"kind": PascalCase str, "amount": Option<i32>, "times": Option<i32>}`.
///
/// `kind` uses the same PascalCase vocabulary `parse_intent` accepts
/// (`Attack`/`Defend`/`Buff`/…) so the dict round-trips, rather than the
/// SCREAMING_SNAKE [`Intent::python_name`] (which targets a different,
/// bridge-replay exporter). `amount` carries attack damage / buff-debuff
/// magnitude; `times` carries the multi-attack hit count.
#[must_use]
pub fn intent_to_value(intent: Intent, state: &CombatState, dealer: CreatureUid) -> Value {
    let Value::Object(mut d) = bare_intent_to_value(intent) else {
        unreachable!("bare_intent_to_value always emits an object");
    };
    // `amount` is the RAW base from the monster's move table, because that is
    // what `Intent` stores and what `unbake_wire_intent` reconstructs on the
    // way back in — keeping the dict round-trip exact. But the *game* reports
    // the baked number (C# `AttackIntent.GetSingleDamage`), so a consumer that
    // compares a mirror wire against a real one would be diffing raw against
    // baked and flagging every Strength / Weak / Shrink / player-Vulnerable
    // board as a divergence. Emit the baked number alongside, computed by the
    // engine's own pipeline rather than a second implementation of it.
    //
    // Additive only: `parse_intent` reads keys by name and ignores unknown
    // ones, so `amount` remains the round-trip carrier and ingest is unchanged.
    // `DeathBlowIntent : SingleAttackIntent` in C#, so it reports a baked
    // number on the same code path as a plain attack.
    if let Intent::Attack { damage }
    | Intent::MultiAttack { damage, .. }
    | Intent::DeathBlow { damage } = intent
    {
        let effective = sts2_core::combat_loop::preview_damage_to_player(
            state,
            sts2_core::hooks::CreatureRef::Enemy(dealer),
            damage,
            sts2_core::hooks::ValueProps::MOVE,
            None,
        );
        set(&mut d, "effective_amount", effective);
        // …and WHICH listeners moved it. `effective_amount` alone is a number
        // with no story: bug report `wb-4eae3bad6ebd57f8` is a player who
        // applied Weak to an enemy already carrying it, saw the telegraph not
        // budge, and reported the engine as broken — Weak is a duration
        // counter (`WeakPower.StackType = Counter`), so the ×0.75 was already
        // baked into the number and nothing on the board said so.
        //
        // Display-only and additive: `parse_intent` reads keys by name and
        // ignores unknown ones, so this changes no round-trip. The rows come
        // from the engine's own per-listener entry points (see
        // `hooks::damage_modifier_breakdown`) rather than a UI-side
        // re-derivation of the damage pipeline.
        let mods = sts2_core::combat_loop::preview_damage_to_player_modifiers(
            state,
            sts2_core::hooks::CreatureRef::Enemy(dealer),
            sts2_core::hooks::ValueProps::MOVE,
        );
        if !mods.is_empty() {
            let rows: Vec<Value> = mods
                .iter()
                .map(|m| damage_modifier_to_value(state, m))
                .collect();
            d.insert("modifiers".to_owned(), Value::Array(rows));
        }
    }
    Value::Object(d)
}

/// The state-free half of an intent's wire shape — `{kind, amount?, times?}`,
/// exactly the keys `parse_intent` needs to rebuild it — with no board to bake
/// a telegraph against. [`intent_to_value`] adds the baked `effective_amount`
/// and its `modifiers` on top; the SimLab catalog (`sts2_rs.monster_moves`)
/// emits this bare form for moves that have not been dealt yet.
#[must_use]
pub fn bare_intent_to_value(intent: Intent) -> Value {
    let mut d = obj();
    let kind = match intent {
        Intent::Attack { .. } | Intent::MultiAttack { .. } => "Attack",
        Intent::Defend { .. } => "Defend",
        Intent::Buff { .. } => "Buff",
        Intent::Debuff { .. } => "Debuff",
        Intent::DebuffStrong => "DebuffStrong",
        Intent::Sleep => "Sleep",
        Intent::Stun => "Stun",
        Intent::StatusCard => "StatusCard",
        Intent::Summon => "Summon",
        Intent::Heal => "Heal",
        // Bead `sts2-rs-f99e7`: GasBomb EXPLODE / WaterfallGiant
        // AboutToBlow — C# `IntentType.DeathBlow`.
        Intent::DeathBlow { .. } => "DeathBlow",
        // Bead `sts2-rs-8f336`: FatGremlin FLEE — C# `IntentType.Escape`.
        Intent::Escape => "Escape",
        Intent::Unknown => "Unknown",
        // `Intent` is #[non_exhaustive]; future variants degrade to a
        // telegraphed-but-unspecified intent rather than failing to build.
        _ => "Unknown",
    };
    set(&mut d, "kind", kind);
    let amount = match intent {
        Intent::Attack { damage }
        | Intent::MultiAttack { damage, .. }
        | Intent::DeathBlow { damage } => Some(damage),
        Intent::Buff { amount, .. } | Intent::Debuff { amount, .. } => Some(amount),
        _ => None,
    };
    if let Some(a) = amount {
        set(&mut d, "amount", a);
    }
    if let Intent::MultiAttack { hits, .. } = intent {
        set(&mut d, "times", i32::from(hits));
    }
    Value::Object(d)
}

/// The `powers` list for one creature: `[{name, amount, display_amount,
/// counter}, …]` in `uids` order.
///
/// # Errors
///
/// A power uid with no entry in `arena` — the state failed its own
/// `assert_invariants` before we got here.
pub fn powers_to_value(
    state: &CombatState,
    uids: &[PowerUid],
    arena: &[Power],
    seats: &[(std::num::NonZeroU32, u8)],
    path: &str,
) -> Result<Value, EmitError> {
    let mut list = Vec::with_capacity(uids.len());
    for uid in uids {
        let pow = arena.iter().find(|p| p.id == uid.0).ok_or_else(|| {
            EmitError::at(path, &format!("power uid {} not in arena", uid.0.get()))
        })?;
        let mut d = obj();
        set(&mut d, "name", pow.kind.python_name());
        set(&mut d, "amount", pow.amount);
        // Player-facing value for Counter powers. Mirrors C# `PowerModel.
        // DisplayAmount`: most powers show `amount`, but Slow shows
        // `SlowAmount * 10` (the per-instance `counter`, bumped per card
        // played and reset at the owner's turn start).
        let display_amount = match pow.kind {
            PowerKind::Slow => pow.counter.saturating_mul(10),
            // GH #363 A1: C# `OrbitPower.DisplayAmount => 4 - energySpent % 4`
            // (`OrbitPower.cs:25`) — the energy still owed before the next
            // grant, in 1..=4 (a fresh Orbit shows 4, not 0). The engine's
            // `counter` is the cumulative energy spent, so this is the same
            // complement the reconstruction arm inverts. Emitting `amount`
            // here — the energy the power GRANTS — made the mirror's own wire
            // disagree with the world's on a field the diff compares.
            PowerKind::Orbit => 4 - pow.counter.rem_euclid(4),
            // C# `ImpaledPower.DisplayAmount => HitCounter`: the hits left
            // before the next arrow dislodges (3 on a fresh impale). The
            // engine keeps `amount` at 1 and the HitCounter in `counter`, so
            // emitting `amount` showed "1" where the game shows "3".
            PowerKind::Impaled => pow.counter,
            // C# `MonologuePower.DisplayAmount => StrengthApplied`
            // (MonologuePower.cs:43, every build): the Strength granted so
            // far this turn, which the engine accumulates on `counter`.
            PowerKind::Monologue => pow.counter,
            // v0.111.0 `PaleBlueDotPower.DisplayAmount => Math.Max(0, 5 -
            // AttacksPlayedThisTurn)` (PaleBlueDotPower.cs:27), where that
            // count is EVERY finished play of the owner this turn — the
            // lambda filters on turn and player, not on card type — i.e.
            // `card_plays_finished_this_turn`. v0.107.1's Pale Blue Dot has
            // no override and shows `amount`. Player-only power.
            PowerKind::PaleBlueDot if state.game_version >= sts2_core::GameVersion::V0_111_0 => {
                let left = 5_u32.saturating_sub(state.combat_log.card_plays_finished_this_turn);
                i32::try_from(left).unwrap_or(0)
            }
            _ => pow.amount,
        };
        set(&mut d, "display_amount", display_amount);
        // The per-instance accumulator, verbatim. `amount` does not describe
        // it and `display_amount` only sometimes encodes it, so without this
        // key every reconstruction restarts CrimsonMantle's self-damage,
        // Slow's tally and Surrounded's facing at zero. See
        // `ParsedPower::counter`.
        set(&mut d, "counter", pow.counter);
        // The co-op seat that applied it (`ParsedPower::applier_seat`), when
        // one is recorded; never solo.
        if let Some((_, seat)) = seats.iter().find(|(id, _)| *id == pow.id) {
            set(&mut d, "applier_seat", *seat);
        }
        list.push(Value::Object(d));
    }
    Ok(Value::Array(list))
}

/// A seat's Osty (`$.player.osty`, `$.allies[i].osty`) in the shape
/// `ParsedOsty` reads, or `None` when the seat has none. The pet lives in
/// `state.creatures` (referenced by the seat's `pet_uid`), deliberately not in
/// `enemy_uids`, so the encounter loop skips it. The species is implied.
fn osty_to_value(state: &CombatState, seat: &sts2_core::PlayerCombatState) -> Option<Value> {
    let pet_uid = seat.pet_uid?;
    let c = state.creatures.iter().find(|c| c.id == pet_uid.0)?;
    let emitted_max_hp = c.max_hp.max(1);
    let mut osty = obj();
    set(&mut osty, "alive", c.current_hp > 0);
    set(&mut osty, "hp", c.current_hp.clamp(0, emitted_max_hp));
    set(&mut osty, "max_hp", emitted_max_hp);
    set(&mut osty, "block", c.block.max(0));
    Some(Value::Object(osty))
}

/// The v0.113.0 Wuppy pet (`$.player.wuppy`) in the `ParsedOsty` shape, or
/// `None` without one. A dead Wuppy stays in the fight, so it is emitted
/// `alive: false`.
fn wuppy_to_value(state: &CombatState) -> Option<Value> {
    let c = state
        .creatures
        .iter()
        .find(|c| c.species == Some(sts2_core::MonsterKind::Wuppy))?;
    let emitted_max_hp = c.max_hp.max(1);
    let mut wuppy = obj();
    set(&mut wuppy, "alive", c.current_hp > 0);
    set(&mut wuppy, "hp", c.current_hp.clamp(0, emitted_max_hp));
    set(&mut wuppy, "max_hp", emitted_max_hp);
    set(&mut wuppy, "block", c.block.max(0));
    Some(Value::Object(wuppy))
}

/// The v0.113.0 Fool mechanics (`$.player.fool`; `sts2_core::fool`), for the
/// e2e tokenizers' "fool" fill event. Absent before v0.113.0 and on a board no
/// Fool mechanic has touched, so every other board emits as before.
///
/// * `arrows`: one `[status, quietus]` per arrow slot in gain order; status 1
///   ready (`Retrieved`), 2 away (`Away`), 3 destroyed; quietus 1 when the
///   arrow is destroyed after impaling (`DestroyAfterImpaled`).
/// * `pending`: a Shoot card's slot not yet created (`shoot_slot_pending`),
///   which counts as one more ready arrow.
/// * `next`: the slot the next shot uses (the LAST ready one), or -1.
/// * `melody`: `[Wah, Cho, Ki, melody cards]` finished this turn.
/// * `impaled`: `[encounter index, arrows]` per impaled enemy.
fn fool_to_value(state: &CombatState) -> Option<Value> {
    use sts2_core::cards::CardTag;
    use sts2_core::fool;
    if !fool::enabled(state) {
        return None;
    }
    let pending = fool::shoot_slot_pending(state);
    let f = fool::fool(state);
    if f.is_none() && !pending {
        return None;
    }
    let mut out = obj();
    let arrows: Vec<Value> = f
        .map(|f| {
            f.arrows
                .iter()
                .map(|a| {
                    let status = match a.status {
                        fool::ArrowStatus::Retrieved => 1,
                        fool::ArrowStatus::Away => 2,
                        fool::ArrowStatus::Destroyed => 3,
                    };
                    serde_json::json!([status, i32::from(a.destroy_after_impaled)])
                })
                .collect()
        })
        .unwrap_or_default();
    out.insert("arrows".to_owned(), Value::Array(arrows));
    set(&mut out, "pending", pending);
    let next = fool::next_arrow(state).map_or(-1, |i| i as i64);
    set(&mut out, "next", next);
    let melody = [
        fool::finished_this_turn(state, CardTag::Wah),
        fool::finished_this_turn(state, CardTag::Cho),
        fool::finished_this_turn(state, CardTag::Ki),
        fool::melody_cards_played_this_turn(state),
    ];
    out.insert("melody".to_owned(), serde_json::json!(melody));
    let impaled: Vec<Value> = f
        .map(|f| {
            f.impaled
                .iter()
                .filter_map(|(uid, slots)| {
                    let i = state.enemy_uids.iter().position(|u| u == uid)?;
                    Some(serde_json::json!([i, slots.len()]))
                })
                .collect()
        })
        .unwrap_or_default();
    out.insert("impaled".to_owned(), Value::Array(impaled));
    Some(Value::Object(out))
}

/// A seat's combat-history counters (`state_dict::SeatHistory`), each only
/// when non-zero so a board without them emits the wire it always did. A
/// per-enemy count is keyed by the enemy's `$.encounter` index.
fn emit_seat_history(
    state: &CombatState,
    m: &mut serde_json::Map<String, Value>,
    log: &sts2_core::TurnEvents,
    powered_hits: &[(CreatureUid, u32)],
) {
    if log.unblocked_damage_received_this_combat > 0 {
        set(
            m,
            "unblocked_hits_this_combat",
            log.unblocked_damage_received_this_combat,
        );
    }
    if log.cards_exhausted_this_combat > 0 {
        set(
            m,
            "cards_exhausted_this_combat",
            log.cards_exhausted_this_combat,
        );
    }
    if log.cards_exhausted_this_turn > 0 {
        set(
            m,
            "cards_exhausted_this_turn",
            log.cards_exhausted_this_turn,
        );
    }
    if log.extra_turns_taken_this_combat > 0 {
        set(
            m,
            "extra_turns_taken_this_combat",
            log.extra_turns_taken_this_combat,
        );
    }
    // Earlier plays' gains only, as the mod sends them: a gain still tagged
    // with its card's uid is a resolving play's own (the tag is retired when
    // the play finishes), which C# Unmovable never counts against that play
    // and which a rebuilt board could not tell from an earlier play's.
    let earlier =
        log.earlier_card_block_gains(state.game_version >= sts2_core::GameVersion::V0_114_0);
    if earlier > 0 {
        set(m, "card_block_gains_this_turn", earlier);
    }
    let hits: Vec<Value> = powered_hits
        .iter()
        .filter_map(|&(uid, n)| {
            let i = state.enemy_uids.iter().position(|u| *u == uid)?;
            let mut e = obj();
            set(&mut e, "enemy", i);
            set(&mut e, "hits", n);
            Some(Value::Object(e))
        })
        .collect();
    if !hits.is_empty() {
        m.insert("powered_hits_this_turn".to_owned(), Value::Array(hits));
    }
}

/// A seat's channelled orbs (`$.player.orbs`, and a whole teammate seat's),
/// with `focus` folded into their values as C#'s `PassiveVal` / `EvokeVal` do.
/// `lightning_bonus` is the seat's Infused Core `ModifyOrbValue` +1 on
/// Lightning orbs (InfusedCore.cs:50-60), added after the Focus clamp
/// (`Hook.ModifyOrbValue` walks powers, then relics).
fn orbs_to_value(belt: &sts2_core::OrbSlots, focus: i32, lightning_bonus: i32) -> Value {
    let mut orbs_list = Vec::new();
    let mut idx: i32 = 0;
    // Focus-scaled passive/evoke clamp at 0 (C# `FocusPower.ModifyOrbValue`
    // → `Math.Max(value + focus, 0)`); `focus_scaled` applies the same clamp
    // the engine effect sites use, so a negative-Focus orb serializes 0 (not
    // a negative passive) to match the C# parity surface.
    use sts2_core::orbs::focus_scaled;
    for orb in &belt.slots {
        let (type_str, loc_key, passive, evoke) = match orb {
            Orb::Empty => continue,
            Orb::Lightning => (
                "Lightning",
                "LIGHTNING_ORB.title",
                focus_scaled(3, focus) + lightning_bonus,
                focus_scaled(8, focus) + lightning_bonus,
            ),
            Orb::Frost => (
                "Frost",
                "FROST_ORB.title",
                focus_scaled(2, focus),
                focus_scaled(5, focus),
            ),
            // Dark's passive shows the per-turn increment (clamped 6 + focus).
            // Its evoke is the C# `_evokeVal`, which STARTS at 6 and is grown
            // by the passive (`DarkOrb.cs:17,25,39`) — no second Focus add.
            // `stored` holds only the accumulated growth, so the base has to
            // go back on: emitting `stored` raw put every mirror-channelled
            // Dark orb 6 low on the wire (257 orb elements across the live
            // A1 Defect corpus), and the matching parse leg subtracted
            // nothing, so the two errors cancelled in any round-trip test.
            Orb::Dark { stored } => (
                "Dark",
                "DARK_ORB.title",
                focus_scaled(6, focus),
                stored.saturating_add(DARK_ORB_BASE_EVOKE),
            ),
            // Plasma is unaffected by Focus (C# literal 1m / 2m).
            Orb::Plasma => ("Plasma", "PLASMA_ORB.title", 1, 2),
            // Glass: passive = clamp(current_passive + focus); evoke =
            // passive * 2 (C# `GlassOrb.EvokeVal => PassiveVal * 2m`).
            Orb::Glass { current_passive } => {
                let passive = focus_scaled(*current_passive, focus);
                ("Glass", "GLASS_ORB.title", passive, passive * 2)
            }
        };
        let mut orb_dict = obj();
        set(&mut orb_dict, "index", idx);
        set(&mut orb_dict, "name", loc_key);
        set(&mut orb_dict, "type", type_str);
        set(&mut orb_dict, "passive", passive);
        set(&mut orb_dict, "evoke", evoke);
        orbs_list.push(Value::Object(orb_dict));
        idx += 1;
    }
    Value::Array(orbs_list)
}

/// Build the wire-format dict for `state`. The dict round-trips through
/// [`crate::state_from_wire`] modulo the documented ignored fields (see the
/// module-level comment).
///
/// # Errors
///
/// A pile contains a uid not in `state.cards`, or a power uid not in
/// `state.powers` (i.e. the state failed its own `assert_invariants`
/// before we got here). The `PyO3` host raises this as `ValueError`.
pub fn combat_state_to_value(state: &CombatState) -> Result<Value, EmitError> {
    let mut d = obj();
    set(&mut d, "wire_version", WIRE_VERSION);
    // The engine signals combat termination via `is_over`/`player_won`
    // (the authoritative flags MCTS reads — lethal_detector.rs:214,
    // value_net.rs:1398), NOT by transitioning `phase` to
    // `CombatPhase::Resolved`, which is never assigned in the combat loop.
    // So a player death during the enemy phase (or a victory) leaves
    // `phase` stuck at `EnemyTurnExec`/`PlayerPlayPhase`, and emitting the
    // raw phase here would report "EnemyPhase" forever — wire/PyO3 callers
    // (incl. the C#/Rust parity harness) could never observe combat-over
    // (sts2-rs-8ukqd: 40/100 parity-fuzz trials diverged on `combat_terminal`
    // for exactly this — C# enters Defeat/Victory while Rust reported
    // "EnemyPhase"). Report the resolved phase off the authoritative flags.
    let phase_str = if state.is_over {
        if state.player_won {
            "Victory"
        } else {
            "Defeat"
        }
    } else {
        combat_phase_to_str(state.phase)
    };
    set(&mut d, "phase", phase_str);
    set(&mut d, "round", i32::from(state.round));
    set(&mut d, "side", side_to_str(state.side));
    set(&mut d, "is_elite", state.is_elite);
    // Only off their defaults, so a fight's wire without them is unchanged.
    if state.act_index != 1 {
        set(&mut d, "act_index", state.act_index);
    }
    if state.is_boss {
        set(&mut d, "is_boss", true);
    }
    if state.total_floor != 0 {
        set(&mut d, "total_floor", i32::from(state.total_floor));
    }
    if let Some(f) = state.player.fool.as_deref() {
        set(&mut d, "fool_state", crate::fool_state::encode(state, f));
    }
    // Wire schema uses `turn`, which mirrors `round` in this Rust port
    // (Python sim splits them; the wire field name predates the merge).
    set(&mut d, "turn", i32::from(state.round));
    set(&mut d, "combat_id", 0_i64);
    // Declare the intent dialect. We emit the engine's own pre-modifier
    // amounts; the C#/adapter producers emit UI-effective ones and are
    // inverted on ingest. Saying so explicitly is what keeps a
    // `state_to_dict` → `state_from_dict` round-trip from subtracting
    // Strength/Vulnerable that were never added. See
    // `ParsedState::intent_amounts`.
    set(&mut d, "intent_amounts", "base");

    // Per-stream RNG cursor (counter = number of public `next_*` reads so
    // far on that stream). Mirrors C# `RunRngSet.ToSerializable().Counters`
    // (each `Rng.Counter`), keyed by the `RunRngType` snake_case names so
    // the two dumps line up field-for-field. This is a *diagnostic*: in a
    // real-vs-real replay (parity_fuzz / full-run) a counter mismatch is
    // the exact stream + read-count where the engines desynced — which the
    // value-only field diff cannot show when the desync is masked by a
    // constant (e.g. a fixed-HP monster). Cheap to emit, no behavior change.
    d.insert("rng_counters".to_owned(), run_rng_counters(state));

    // --- player ---
    let mut player = obj();
    set(
        &mut player,
        "class",
        class_str_from_character(state.player.character),
    );
    // Wire format rejects max_hp <= 0; clamp to 1 so an empty default
    // state still survives parse. The state_equiv predicate ignores
    // this if both sides agree on the clamp.
    let emitted_max_hp = state.player.max_hp.max(1);
    set(&mut player, "max_hp", emitted_max_hp);
    let emitted_hp = state.player.current_hp.clamp(0, emitted_max_hp);
    set(&mut player, "hp", emitted_hp);
    set(&mut player, "block", state.player.block.max(0));
    set(&mut player, "energy", state.player.energy.max(0));
    // C# `PlayerCombatState.MaxEnergy` is computed on every read; the stored
    // `max_energy` is the last turn start's, so a max-energy power or relic
    // gained this turn (Demesne, Pyre, Waste Away) would report a turn late.
    let max_energy = sts2_core::combat_loop::effective_max_energy(state);
    set(&mut player, "max_energy", max_energy);
    set(&mut player, "energy_regen_per_turn", max_energy);
    // Gold: gates SealOfGold's turn-start hook, so an engine -> dict ->
    // engine round-trip must not silently zero it.
    set(&mut player, "gold", state.gold);

    // Relics.
    let mut relics_list = Vec::with_capacity(state.relics.len());
    for r in &state.relics {
        let mut rd = obj();
        set(&mut rd, "id", r.kind.python_name());
        if r.counter != 0 {
            set(&mut rd, "counter", r.counter);
        }
        // ToyBox wax/melt state, emitted symmetrically with the parser
        // (and with the C# producer in `Sts2Headless/RunSimulator.cs`) so a
        // world-vs-mirror wire diff cannot false-diverge on a melted relic.
        // Omitted when false to keep every non-ToyBox wire byte-identical.
        if r.is_wax {
            set(&mut rd, "is_wax", true);
        }
        if r.is_melted {
            set(&mut rd, "is_melted", true);
        }
        // GH #1197: Emotion Chip / Mini Regent / Regalite say whether they
        // will fire on their next trigger. Only those three carry the key.
        if let Some(armed) = sts2_core::relic_effects::relic_armed(state, r) {
            set(&mut rd, "armed", armed);
        }
        relics_list.push(Value::Object(rd));
    }
    player.insert("relics".to_owned(), Value::Array(relics_list));

    // Player powers. The d03 player-Creature unification is mid-migration
    // (Phase 1): the player's powers are *split* across two surfaces.
    //   * Relic combat-start grants (BronzeScales Thorns, Vajra Strength,
    //     OddlySmoothStone Dexterity, …) are applied via
    //     `apply_power_to_creature(player_uid, …)` and live on the mirror
    //     Creature's `powers` (the listeners resolve the player through
    //     `state.player.creature_uid`).
    //   * Card self-power grants (Inflame Strength, Flex, Footwork, …) are
    //     applied via the per-character `apply_power_to_player` helpers,
    //     which still push onto the legacy `player.powers` SmallVec.
    // C# carries a single unified power list, so we must serialize the
    // *union* of both surfaces — Creature first (relics apply at combat
    // start, before any card play), then any legacy power not already
    // present, deduped by power-arena id. `init_player_creature` clones
    // `player.powers` into the Creature at combat start, so a power that
    // predates combat appears in both lists with the same id; dedup keeps
    // it once. (States built by `state_from_dict` leave `creature_uid`
    // None and keep all powers on the legacy field — the fallback below
    // serializes that single list unchanged.)
    let creature_powers = state
        .player
        .creature_uid
        .and_then(|uid| state.creatures.iter().find(|c| c.id == uid.0))
        .map(|c| c.powers.as_slice());
    let player_power_uids: Vec<PowerUid> = match creature_powers {
        Some(cp) => {
            let mut merged: Vec<PowerUid> = cp.to_vec();
            for uid in &state.player.powers {
                if !merged.iter().any(|u| u.0 == uid.0) {
                    merged.push(*uid);
                }
            }
            merged
        }
        None => state.player.powers.iter().copied().collect(),
    };
    player.insert(
        "powers".to_owned(),
        powers_to_value(
            state,
            &player_power_uids,
            &state.powers,
            &state.coop.power_seat,
            "$.player.powers",
        )?,
    );

    // Potions — a flat list of ids (None for empty slot). The Rust
    // PlayerCombatState carries only filled slots; emit each as a
    // string id.
    let potions_list: Vec<Value> = state
        .player
        .potions
        .iter()
        .map(|k| val(k.python_name()))
        .collect();
    player.insert("potions".to_owned(), Value::Array(potions_list));
    // The belt CAPACITY (C# `Player.MaxPotionCount`), which `potions` cannot
    // carry since it lists filled slots only. Omitted while unset (`0`, the
    // `Default` sentinel), so the parser keeps that sentinel too.
    if state.player.potion_slots != 0 {
        set(
            &mut player,
            "potion_slots",
            i32::from(state.player.potion_slots),
        );
    }

    if state.player.stars != 0 {
        set(&mut player, "stars", i32::from(state.player.stars));
    }

    // Per-turn play counters — emitted only mid-turn (any bucket non-zero)
    // so start-of-turn dicts keep the historical shape. The parser side
    // documents why dropping these stalls round-tripped searches (Ringing/
    // Normality/Sloth ShouldPlay gates).
    if state.player.cards_played_this_turn.iter().any(|&n| n > 0) {
        set(
            &mut player,
            "cards_played_this_turn",
            play_buckets_wire(&state.player.cards_played_this_turn),
        );
    }
    // ChainsOfBinding's once-per-turn Bound gate — same rationale as the
    // buckets above: dropping it re-opens vetoed plays on round-trip.
    if state.combat_log.chains_bound_card_played_this_turn {
        set(&mut player, "bound_card_played_this_turn", true);
    }
    if state.combat_log.hp_lost_this_turn > 0 {
        set(&mut player, "hp_lost_this_turn", true);
    }
    // GH #1197: per-turn counters the net reads directly. Each only when
    // non-zero, so a board without them emits the wire it always did; a
    // missing key means 0. `discards_this_turn` is C#'s `CardDiscardedEntry`
    // count (MementoMori's multiplier); the type counts are the
    // `cards_played_this_turn` Attack / Skill buckets.
    for (key, n) in [
        (
            "discards_this_turn",
            u32::from(state.combat_log.cards_discarded_this_turn),
        ),
        (
            "attacks_played_this_turn",
            state.player.cards_played_this_turn[CardType::Attack.index()],
        ),
        (
            "skills_played_this_turn",
            state.player.cards_played_this_turn[CardType::Skill.index()],
        ),
    ] {
        if n > 0 {
            set(&mut player, key, n);
        }
    }
    emit_seat_history(
        state,
        &mut player,
        &state.combat_log,
        &state.player.powered_hits_by_target_this_turn,
    );
    // Murder's multiplier (`state_dict::SeatHistory::cards_drawn_this_combat`).
    if state.cards_drawn > 0 {
        set(&mut player, "cards_drawn_this_combat", state.cards_drawn);
    }
    // Supermassive's multiplier
    // (`state_dict::SeatHistory::cards_generated_this_combat`).
    if state.player.generated_cards_this_combat > 0 {
        set(
            &mut player,
            "cards_generated_this_combat",
            state.player.generated_cards_this_combat,
        );
    }

    // Necrobinder's Osty pet. It lives in `state.creatures` (referenced by
    // `player.pet_uid`), deliberately *not* in `enemy_uids`, so the encounter
    // loop below skips it. Emit the `$.player.osty` shape the parser reads
    // (see `ParsedOsty`: alive/hp/max_hp/block) so downstream consumers — the
    // workbench Rust backend — can surface the companion the way the C#
    // backend already does. Name is not part of the wire shape; the consumer
    // knows the species is Osty.
    if let Some(osty) = osty_to_value(state, &state.player) {
        player.insert("osty".to_owned(), osty);
    }
    // v0.113.0 Wuppy pet (`$.player.wuppy`, same shape as the Osty). Like
    // the Osty it is not in `enemy_uids`; its `WuppysLoyaltyPower` is
    // implied. Absent without one, so every other board emits as before.
    if let Some(wuppy) = wuppy_to_value(state) {
        player.insert("wuppy".to_owned(), wuppy);
    }
    // v0.113.0 Fool mechanics (`$.player.fool`): arrows, melody, impaled
    // enemies. Absent unless a Fool mechanic has touched the board.
    if let Some(f) = fool_to_value(state) {
        player.insert("fool".to_owned(), f);
    }

    // Defect orbs. Mirror the C# `CombatPlayState` shape (RunSimulator.cs:
    // emits `orbs` + `orb_slots` whenever the belt has capacity) so the
    // workbench renders the orb belt — including its empty slots — for the
    // Rust backend too. `orbs` is the list of *channelled* (non-`Empty`)
    // orbs; `orb_slots` is the belt capacity the frontend pads with empties.
    // `passive`/`evoke` match C#'s `OrbModel.PassiveVal`/`EvokeVal`, which
    // fold in the player's Focus via `ModifyOrbValue` (Plasma excepted).
    // The Rust backend emits `<KIND>_ORB.title` loc keys (server.py resolves
    // them; the C# backend already ships a plain name).
    // A SIZED belt at 0 (Bulk Up emptied it) is emitted too, as
    // `orbs: []`/`orb_slots: 0`, so a re-import keeps it at 0 instead of
    // handing the Defect the lazy 3-slot base back. C# omits the keys at
    // `Capacity == 0`; the import reads that as the same unsized belt.
    if state.player.orbs.capacity > 0 || state.player.orbs.sized {
        let focus = sts2_core::orbs::focus_bonus(state);
        let lightning_bonus = sts2_core::orbs::lightning_orb_bonus(state);
        player.insert(
            "orbs".to_owned(),
            orbs_to_value(&state.player.orbs, focus, lightning_bonus),
        );
        set(
            &mut player,
            "orb_slots",
            i32::from(state.player.orbs.capacity),
        );
    }

    d.insert("player".to_owned(), Value::Object(player));

    // --- deck ---
    let mut deck = obj();
    deck.insert(
        "draw_pile".to_owned(),
        pile_to_value(state, &state.player.draw, &state.cards, "$.deck.draw_pile")?,
    );
    deck.insert(
        "discard_pile".to_owned(),
        pile_to_value(
            state,
            &state.player.discard,
            &state.cards,
            "$.deck.discard_pile",
        )?,
    );
    deck.insert(
        "hand".to_owned(),
        pile_to_value(state, &state.player.hand, &state.cards, "$.deck.hand")?,
    );
    deck.insert(
        "exhaust_pile".to_owned(),
        pile_to_value(
            state,
            &state.player.exhaust,
            &state.cards,
            "$.deck.exhaust_pile",
        )?,
    );
    // Cards in flight. Emitted only when non-empty, which is only ever
    // mid-resolution — most notably while a card-select screen the played
    // card opened is still up. Omitting it dropped that card on every
    // reconstruction: the choice then resolved with an empty play pile and
    // the card never reached the discard.
    if !state.player.play.is_empty() {
        deck.insert(
            "play_pile".to_owned(),
            pile_to_value(state, &state.player.play, &state.cards, "$.deck.play_pile")?,
        );
    }

    // master_deck: best-effort — emit as `None` when empty so the
    // receiver doesn't try to materialise zero entries.
    if !state.master_deck.is_empty() {
        let mut md = Vec::with_capacity(state.master_deck.len());
        for (i, cid) in state.master_deck.iter().enumerate() {
            let mut entry = obj();
            set(&mut entry, "id", cid.to_python_name());
            // master_deck has no per-card uid in the Rust port; emit a
            // synthetic stable_idx for schema validity.
            set(&mut entry, "stable_idx", i32::try_from(i).unwrap_or(0));
            set(&mut entry, "kind", "card");
            set(&mut entry, "upgraded", false);
            md.push(Value::Object(entry));
        }
        deck.insert("master_deck".to_owned(), Value::Array(md));
    }

    // Draw-pile knowledge, as indices into the `draw_pile` list emitted
    // above. Stale pins (uids no longer physically in `draw`) have no
    // index and are dropped — the same retain-sweep `cards::shuffle` /
    // `cards::pop_next_card` / `blindfold::canonicalize_unknown_middle`
    // perform before trusting either list, so dropping them here loses
    // nothing a consumer would have honored.
    let pin_indices = |pins: &[CardUid]| -> Vec<usize> {
        pins.iter()
            .filter_map(|uid| state.player.draw.iter().position(|d| d == uid))
            .collect()
    };
    let top_idx = pin_indices(&state.player.pinned_top);
    if !top_idx.is_empty() {
        set(&mut deck, "pinned_top", top_idx);
    }
    let bot_idx = pin_indices(&state.player.pinned_bottom);
    if !bot_idx.is_empty() {
        set(&mut deck, "pinned_bottom", bot_idx);
    }
    if state.player.deferred_draw_order {
        set(&mut deck, "deferred_draw_order", true);
    }

    d.insert("deck".to_owned(), Value::Object(deck));

    // --- encounter ---
    let mut enc = Vec::with_capacity(state.enemy_uids.len());
    for uid in &state.enemy_uids {
        let c = state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .ok_or_else(|| EmitError::at("$.encounter", "enemy uid not in creatures"))?;
        let mut m = obj();
        // Enemy uids only — `species` must be `Some(_)` here
        // (bead sts2-rs-a19m migrated off the player Cultist sentinel).
        let species = c
            .species
            .expect("enemy creature must carry a species (bead sts2-rs-a19m)");
        set(&mut m, "kind", species.python_name());
        set(&mut m, "uid", i64::from(c.id.get()));
        // Clamp shape to satisfy parse_state_dict's invariants.
        let m_max_hp = c.max_hp.max(1);
        set(&mut m, "max_hp", m_max_hp);
        set(&mut m, "hp", c.current_hp.clamp(0, m_max_hp));
        set(&mut m, "block", c.block.max(0));
        m.insert(
            "powers".to_owned(),
            powers_to_value(
                state,
                &c.powers,
                &state.powers,
                &state.coop.power_seat,
                "$.encounter.powers",
            )?,
        );
        // `intents` carries the creature's currently-telegraphed move so
        // display consumers (workbench combat panel) can render it. We emit a
        // 0-or-1 element list: `Creature.intent` is a single `Option<Intent>`.
        // `intent_history` stays empty: the parser only needs the current
        // intent, and Rust's `IntentHistory` shape is private and dropped on
        // the rebuild path (round-trip equiv ignores it).
        //
        // ── KNOWN GAP: multi-intent moves are truncated here ──────────────
        // C# `MoveState` takes `params AbstractIntent[]`
        // (`../sts2-wiki/decompiled/v0.107.1/MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine/MoveState.cs:48-53`)
        // and real movesets use it — e.g. `TerrorEel.cs:72`
        // `new MoveState("THRASH_MOVE", ThrashMove, new MultiAttackIntent(...),
        // new BuffIntent())`. The world therefore telegraphs `[Attack x3,
        // Buff]` where this emits `[Attack x3]`.
        //
        // The engine's DSL ALREADY models the list —
        // `monster_ai::MoveState.intents` is a `SmallVec<[Intent; 4]>` and 44
        // content moves declare two or three — but every consumer collapses it
        // to `intents.first()` (`combat_loop::perform_intent`, the intent
        // priming in `hooks`, `parse_enemy`). A 2026-08 census found most
        // multi-intent telegraphs on moves whose CONTENT declaration was
        // short, so the tail was held back. That gap is now closed and
        // guarded: `sts2-wire/tests/move_intents_match_decomp.rs` checks every
        // declared move against its C# `MoveState` ctor, and on 2026-10-30
        // (#351) only moves needing an intent the engine lacks
        // (`CardDebuffIntent`) still differ.
        //
        // Note the tail is NOT inert for search: `peek_intents` walks every
        // intent of a move, and the default `TexelWide` leaf reads that
        // lookahead (`eval_wide.rs`), so a tail edit moves depth-cut leaves.
        //
        // What remains: (2) emit the whole `current_move().intents` list
        // when the primed `c.intent` is still its head, (3) teach
        // `parse_enemy` to keep the tail. Simulation is NOT affected by the
        // gap — the non-head halves are performed by each move's `effect_fn`,
        // not by the intent — so this is a telegraph-fidelity item, not a
        // combat-correctness one. See `wire_emits_only_the_primed_head_intent`
        // in `sts2-py/src/state_dict.rs`, which pins the current shape so a
        // change here is deliberate.
        let mut intents = Vec::new();
        if let Some(intent) = c.intent {
            intents.push(intent_to_value(intent, state, CreatureUid(c.id)));
        }
        m.insert("intents".to_owned(), Value::Array(intents));
        m.insert("intent_history".to_owned(), Value::Array(Vec::new()));
        // Debug-only: the AI's current move-state name (for lockstep
        // tracing vs C# `move_state`). `normalize_rust` builds its own
        // monster dict and ignores this key, so it never reaches the
        // parity diff. `None` when the creature is driven by the
        // species-`roll_intent` path instead of a `MonsterAi`.
        if let Some(ai) = &c.ai {
            set(&mut m, "move_state", ai.current_state_id());
            // The move machine's per-instance state (#1181), which the next
            // roll reads (`CannotRepeat` / `UseOnlyOnce` / cooldowns read the
            // log) and `move_state` alone loses. `log` is in C#'s
            // `StateLog` convention (`MonsterAi::wire_move_log`), which the C#
            // producers send too; `forced` / `counter` / `forced_next` are
            // engine-only and keep a native round-trip exact.
            let mut a = obj();
            a.insert(
                "log".to_owned(),
                Value::Array(
                    ai.wire_move_log()
                        .iter()
                        .map(|id| Value::from(*id))
                        .collect(),
                ),
            );
            set(&mut a, "performed_first_move", ai.performed_first_move());
            set(&mut a, "performed", ai.performed_at_least_once());
            set(&mut a, "forced", ai.current_forced());
            set(&mut a, "counter", ai.counter());
            a.insert(
                "forced_next".to_owned(),
                ai.forced_next_state().map_or(Value::Null, Value::from),
            );
            m.insert("ai".to_owned(), Value::Object(a));
            // Hidden AI counters the wire dialect discloses (2026-08-02):
            // emit them so a native round-trip keeps what the C# producers
            // now ship. The Fabricator's counter IS the last-spawned bot
            // code; the KnowledgeDemon's IS the curse index (0..3).
            if species == MonsterKind::Fabricator {
                let code = ai.counter();
                let kind = [
                    MonsterKind::Zapbot,
                    MonsterKind::Stabbot,
                    MonsterKind::Guardbot,
                    MonsterKind::Noisebot,
                ]
                .into_iter()
                .find(|&k| {
                    sts2_content::monsters::act3_glory::fabricator_bot_code(k) == code && code != 0
                });
                if let Some(kind) = kind {
                    set(&mut m, "fabricator_last_spawned", kind.python_name());
                }
            }
            if species == MonsterKind::KnowledgeDemon {
                set(&mut m, "curse_of_knowledge_counter", ai.counter());
            }
        }
        // C# `Creature.IsAlive` is `CurrentHp > 0`, so a 0-HP enemy
        // serializes as dead (the parity harness derives
        // `alive = !is_dead`). Deriving this from HP — rather than the
        // old hardcoded `false` — fixes the FOGMOG IllusionPower parity
        // bug (sts2-rs-vvi6g): the player-killed EyeWithTeeth used to
        // read as alive at 0 HP. `state_from_dict` keeps an `is_dead`
        // enemy as a corpse (`enemy_uids` is parallel to every non-escaped
        // wire entry; a reviver's corpse — Decimillipede segments, Eye with
        // Teeth — must exist to revive), so a dumped corpse round-trips.
        set(&mut m, "is_dead", c.current_hp <= 0);
        set(&mut m, "is_escaped", false);
        // C# `RemoveCreature` parity (bead sts2-rs-c8why): a dead enemy
        // leaves the active `Enemies` list unless a power retains its
        // corpse (IllusionPower etc). We keep the creature in the stable
        // `enemy_uids` substrate (so NN slots don't reshuffle) and surface
        // this derived flag so the parity harness can compact the Rust
        // view down to C#'s `Enemies` shape. Alive enemies and retained
        // corpses are `false`.
        set(&mut m, "removed_after_death", state.is_removed_corpse(*uid));
        // Cards this creature is currently holding hostage via `SwipePower`
        // (`ThievingHopper`'s THIEVERY move). C# keeps the card ON the power
        // instance (`SwipePower.StolenCard`, surfaced as a hover tip and as
        // the card node pinned to the thief's `%StolenCardPos`); the Rust
        // model keeps one flat ledger on `CombatState`, so the identity has to
        // be re-attached to its thief here or a display consumer sees only a
        // bare "Swipe 1" and cannot say WHICH card left the deck.
        //
        // `parse_enemy` reads it back onto the rebuilt fight's ledger
        // (GH #1693): the student reads the thief's held card.
        //
        // Reclaimed entries (thief died → `SwipePower.BeforeDeath` gave the
        // card back) are omitted: the card is the player's again, and the
        // thief that held it is a corpse.
        let mut stolen = Vec::new();
        for s in &state.swipe_stolen {
            if s.reclaimed || s.thief.0 != c.id {
                continue;
            }
            let mut sd = obj();
            set(&mut sd, "id", s.card_id.to_python_name());
            set(&mut sd, "upgraded", s.upgraded);
            stolen.push(Value::Object(sd));
        }
        if !stolen.is_empty() {
            m.insert("stolen_cards".to_owned(), Value::Array(stolen));
        }
        enc.push(Value::Object(m));
    }
    d.insert("encounter".to_owned(), Value::Array(enc));

    // Co-op teammates, in the shape `$.allies` parses. Omitted when solo,
    // so every single-player state emits the dict it always did.
    // A whole seat (the wire carried its hand) round-trips whole.
    if !state.allies.is_empty() {
        let seats = !state.coop.stash.is_empty();
        let mut allies = Vec::with_capacity(state.allies.len());
        for (i, a) in state.allies.iter().enumerate() {
            let mut m = obj();
            set(&mut m, "hp", a.current_hp);
            set(&mut m, "max_hp", a.max_hp);
            set(&mut m, "block", a.block);
            if seats {
                let path = format!("$.allies[{i}]");
                set(&mut m, "class", class_str_from_character(a.character));
                set(&mut m, "energy", a.energy);
                set(&mut m, "max_energy", i32::from(a.max_energy));
                m.insert(
                    "powers".to_owned(),
                    powers_to_value(
                        state,
                        &state.coop.stash[i].powers,
                        &state.powers,
                        &state.coop.power_seat,
                        &format!("{path}.powers"),
                    )?,
                );
                m.insert(
                    "hand".to_owned(),
                    pile_to_value(state, &a.hand, &state.cards, &format!("{path}.hand"))?,
                );
                // The rest of a whole seat (a full co-op fight), when there is any:
                // omitted otherwise, so an advisor board's wire is unchanged.
                for (key, pile) in [
                    ("draw_pile", a.draw.as_slice()),
                    ("discard_pile", a.discard.as_slice()),
                    ("exhaust_pile", a.exhaust.as_slice()),
                    ("play_pile", a.play.as_slice()),
                ] {
                    if !pile.is_empty() {
                        m.insert(
                            key.to_owned(),
                            pile_to_value(state, pile, &state.cards, &format!("{path}.{key}"))?,
                        );
                    }
                }
                if a.stars != 0 {
                    set(&mut m, "stars", i32::from(a.stars));
                }
                if a.orbs.capacity > 0 || a.orbs.sized {
                    // Their own Focus: a teammate seat's powers are stashed.
                    let focus = state.coop.stash[i]
                        .powers
                        .iter()
                        .filter_map(|u| state.powers.iter().find(|p| p.id == u.0))
                        .filter(|p| p.kind == PowerKind::Focus)
                        .map(|p| p.amount)
                        .sum();
                    // A teammate's own Infused Core is not modelled on the seat.
                    m.insert("orbs".to_owned(), orbs_to_value(&a.orbs, focus, 0));
                    set(&mut m, "orb_slots", i32::from(a.orbs.capacity));
                }
                if state.coop.stash[i].combat_log.hp_lost_this_turn > 0 {
                    set(&mut m, "hp_lost_this_turn", true);
                }
                emit_seat_history(
                    state,
                    &mut m,
                    &state.coop.stash[i].combat_log,
                    &a.powered_hits_by_target_this_turn,
                );
                // Mid-turn only, as ours.
                if a.cards_played_this_turn.iter().any(|&n| n > 0) {
                    set(
                        &mut m,
                        "cards_played_this_turn",
                        play_buckets_wire(&a.cards_played_this_turn),
                    );
                }
                if !a.potions.is_empty() {
                    let list = a.potions.iter().map(|k| val(k.python_name())).collect();
                    m.insert("potions".to_owned(), Value::Array(list));
                }
                let relics = &state.coop.stash[i].relics;
                if !relics.is_empty() {
                    let list = relics
                        .iter()
                        .map(|r| {
                            let mut rd = obj();
                            set(&mut rd, "id", r.kind.python_name());
                            if r.counter != 0 {
                                set(&mut rd, "counter", r.counter);
                            }
                            Value::Object(rd)
                        })
                        .collect();
                    m.insert("relics".to_owned(), Value::Array(list));
                }
                // Their Osty, as ours (GH #1017).
                if let Some(osty) = osty_to_value(state, a) {
                    m.insert("osty".to_owned(), osty);
                }
                set(&mut m, "ended_turn", state.coop.to_play & (1 << i) == 0);
            }
            allies.push(Value::Object(m));
        }
        d.insert("allies".to_owned(), Value::Array(allies));
        if state.coop.full {
            set(&mut d, "coop_full", true);
            if state.coop.seat_index > 0 {
                set(&mut d, "coop_seat_index", state.coop.seat_index);
            }
        }
    }

    // --- rng / ascension ---
    let mut rng = obj();
    // We can't recover the original seed from the in-process RNG, so
    // emit `0` — the state_equiv predicate ignores this on round-trip
    // (rng cursor is in the documented-ignored list).
    set(&mut rng, "seed", 0_u64);
    set(&mut rng, "ascension", state.ascension);
    // The RUN's dice, unlike the search seed above, ARE recoverable and do
    // round-trip: the master seed every stream salts from, plus each stream's
    // read count. Emitting them makes a Rust-produced wire carry the same RNG
    // lineage the C# producers carry, so the native-engine oracle exercises
    // the identical restore path the DLL mirror uses.
    set(&mut rng, "run_seed", state.run_rng.wire_seed());
    rng.insert("counters".to_owned(), run_rng_counters(state));
    d.insert("rng".to_owned(), Value::Object(rng));
    // Game build (GH #787): emitted only when non-default, so every
    // v0.107.1 wire — and every golden built from one — is unchanged.
    if !state.game_version.is_default() {
        d.insert(
            "game_version".to_owned(),
            Value::String(state.game_version.to_string()),
        );
    }
    // Game mode and Dread dominion (GH #1346): only when set, so every
    // standard wire is unchanged.
    if !state.game_mode.is_default() {
        d.insert(
            "game_mode".to_owned(),
            Value::String(state.game_mode.as_str().to_owned()),
        );
    }
    if let Some(dominion) = state.dominion {
        d.insert(
            "dominion".to_owned(),
            Value::String(dominion.entry().to_owned()),
        );
    }
    // `WaterfallGiantDominion`'s per-combat notes (GH #1346), state a summon
    // reads: only when set.
    if state.combat_log.dread_explosive != 0 {
        d.insert(
            "dread_explosive".to_owned(),
            Value::from(state.combat_log.dread_explosive),
        );
    }
    if state.combat_log.dread_weak_encounter {
        d.insert("dread_weak_encounter".to_owned(), Value::Bool(true));
    }
    // `KnowledgeDemonDominion`'s turn-1 screen with the hand draw parked
    // behind it (GH #1346): the one open choice the wire carries, because
    // the engine derives all of it. Only while that screen is open.
    if let Some(parked) = sts2_core::dread::parked_knowledge_choice(state) {
        let mut m = serde_json::Map::new();
        m.insert("draw_count".to_owned(), Value::from(parked.draw_count));
        m.insert("side_fired".to_owned(), Value::Bool(parked.side_fired));
        d.insert("dread_knowledge_choice".to_owned(), Value::Object(m));
    }

    Ok(Value::Object(d))
}

// ---------------------------------------------------------------------------
// Bridge-shape action / pending-choice dicts
// ---------------------------------------------------------------------------

/// Render an [`Action`] as a bridge-shape dict — the element shape of
/// [`legal_actions`].
///
/// Shape contract (mirrors the `sts2-rl-agent` rollout bridge):
///   * `{"action": "play", "card_index": i, "target_index": j | None}`
///   * `{"action": "play", "card_index": i, "target_index": None, "ally_index": k}`
///     — a co-op `AnyAlly` card at teammate `allies[k]` (`PlayCardAtAlly`)
///   * `{"action": "end_turn"}`
///   * `{"action": "choose", "index": i, "selection": [..]}` — emitted iff a
///     `pending_choice` is open; in that case no other actions are legal.
///   * `{"action": "potion", "slot": i, "target_index": j | None}`
///   * `{"action": "drink_potion", "slot": i, "target_index": None, "ally_index": k}`
///     — a co-op `AnyPlayer` potion thrown at teammate `allies[k]`
///     (`DrinkPotionAtAlly`)
#[must_use]
pub fn combat_action_to_value(action: &Action) -> Value {
    let mut d = obj();
    match action {
        Action::PlayCard { hand_index, target } => {
            set(&mut d, "action", "play");
            set(&mut d, "card_index", *hand_index);
            set(&mut d, "target_index", *target);
        }
        // Co-op: an `AnyAlly` card at a chosen teammate. A `play` with no
        // enemy target and the teammate's `allies[]` index (GH #1017).
        Action::PlayCardAtAlly { hand_index, ally } => {
            set(&mut d, "action", "play");
            set(&mut d, "card_index", *hand_index);
            set(&mut d, "target_index", None::<u8>);
            set(&mut d, "ally_index", *ally);
        }
        Action::EndTurn => {
            set(&mut d, "action", "end_turn");
        }
        Action::UsePotion { slot, target } => {
            set(&mut d, "action", "potion");
            set(&mut d, "slot", *slot);
            set(&mut d, "target_index", *target);
        }
        Action::DrinkPotion {
            potion_slot,
            target,
        } => {
            set(&mut d, "action", "drink_potion");
            set(&mut d, "slot", *potion_slot);
            set(&mut d, "target_index", *target);
        }
        // Co-op: thrown at a teammate. A `drink_potion` with no enemy target
        // and the teammate's `allies[]` index, as `play`'s (GH #1017).
        Action::DrinkPotionAtAlly { potion_slot, ally } => {
            set(&mut d, "action", "drink_potion");
            set(&mut d, "slot", *potion_slot);
            set(&mut d, "target_index", None::<u8>);
            set(&mut d, "ally_index", *ally);
        }
        Action::Channel { orb, target } => {
            set(&mut d, "action", "channel");
            set(&mut d, "orb", format!("{orb:?}"));
            set(&mut d, "target_index", *target);
        }
        Action::Evoke { times } => {
            set(&mut d, "action", "evoke");
            set(&mut d, "times", *times);
        }
        Action::ResolveChoice { selection } => {
            set(&mut d, "action", "choose");
            // `index` — the leading chosen index (default 0 for the
            // empty/skip selection) — is kept for back-compat with
            // single-pick callers that read only `index`.
            set(&mut d, "index", selection.first().copied().unwrap_or(0u8));
            // `selection` — the FULL index list — lets callers drive a
            // multi-pick choice (HIDDEN_DAGGERS `DiscardFromHand`
            // min=max=2 emits `[0,1]`). Without it the harness could
            // only ever express a single discard, silently dropping the
            // second required card (bead
            // `sts2-rs-hd-multipick-discard-uhsk1`). It also disambiguates
            // the empty-selection SKIP (`selection == []`) from picking
            // option 0 — both collapse to `index == 0` above.
            let sel: Vec<u8> = selection.iter().copied().collect();
            set(&mut d, "selection", sel);
        }
    }
    Value::Object(d)
}

/// Enumerate the legal actions for `state` as a list of bridge-shape dicts
/// ([`combat_action_to_value`] per action). Empty list when the state is
/// terminal — MCTS callers treat this as the leaf signal.
#[must_use]
pub fn legal_actions(state: &CombatState) -> Value {
    Value::Array(
        sts2_core::legal_actions(state)
            .iter()
            .map(combat_action_to_value)
            .collect(),
    )
}

/// Where `uid` currently sits, as the `(pile, index)` pair
/// `PyCombatState::open_injected_choice` reads back — or `None` when the
/// card is in no pile at all (mid-flight during resolution, or already gone).
///
/// Search order matches the locator's own `match` arms. A card is in at most
/// one pile, so the order only decides which name a (bug-induced) duplicate
/// reports under.
#[must_use]
pub fn locate_card(state: &CombatState, uid: CardUid) -> Option<(&'static str, usize)> {
    let p = &state.player;
    // Each pile is a `SmallVec` with its own inline capacity, so the array
    // has to hold plain slices for the element types to unify.
    for (name, pile) in [
        ("hand", p.hand.as_slice()),
        ("draw", p.draw.as_slice()),
        ("discard", p.discard.as_slice()),
        ("exhaust", p.exhaust.as_slice()),
        ("play", p.play.as_slice()),
    ] {
        if let Some(i) = pile.iter().position(|c| *c == uid) {
            return Some((name, i));
        }
    }
    None
}

/// Build the per-option dict entry. The shape matches Python's
/// `combat_state_to_bridge_state` lines 307-311:
///
/// ```text
/// {'id': '<canonical name>', 'type': '<Attack|Skill|Power>', 'upgraded': bool}
/// ```
///
/// When the [`ChoiceOption`] has a `card_uid` we resolve the concrete
/// [`Card`] in `state.cards` to read the runtime `card_type` /
/// `upgraded` fields (so upgraded flags survive into the bridge). When
/// `card_uid` is `None` we fall back to the spec-level `card_type` via
/// the content catalog lookup, leaving `upgraded` false — this matches
/// the synthetic-options pattern used by `OrbSlot` / discovery / card
/// reward flavors at this pin.
#[must_use]
pub fn choice_option_to_value(option: &ChoiceOption, state: &CombatState) -> Value {
    let cards = state.cards.as_slice();
    let mut d = obj();
    set(&mut d, "id", option.card_id.to_python_name());

    // Where the option's card physically sits RIGHT NOW, as the
    // `pile`/`index` locator pair `open_injected_choice` reads back. Emitting
    // it makes this dict the exact inverse of that constructor, so a mirror
    // can re-open its own derived choice — and a driver can hold the live
    // option list to a concrete board position instead of a card name (which
    // duplicates make ambiguous). Current location, not `source_pile`: the
    // locator resolves against the live piles, and for an OPEN choice the two
    // agree anyway (nothing has moved the card yet). Absent for synthetic
    // options and for a card that has since left every pile.
    if let Some(uid) = option.card_uid {
        if let Some((pile, index)) = locate_card(state, uid) {
            set(&mut d, "pile", pile);
            set(&mut d, "index", index);
        }
    }

    // Resolve runtime card metadata via card_uid when present. Look up
    // by uid match in the arena (parallels `state.cards.iter().find` in
    // parity.rs:127 — the arena is small and this matches the
    // existing access pattern).
    let resolved = option
        .card_uid
        .and_then(|uid| cards.iter().find(|c| c.uid == uid));
    // When the option has no `card_uid` (synthetic options for orb
    // slot / enemy target / discovery before deck materialization),
    // we omit `type` since we can't determine it without resolving
    // through a card catalog — at this pin no production callers
    // build such options. `upgraded` falls back to `ChoiceOption.upgraded`.
    let (ctype_opt, upgraded) = match resolved {
        Some(card) => (Some(card.card_type), card.upgraded),
        None => (None, option.upgraded),
    };
    if let Some(ct) = ctype_opt {
        set(&mut d, "type", card_type_python_name(ct));
    }
    set(&mut d, "upgraded", upgraded);
    Value::Object(d)
}

/// Render a [`PendingChoice`] as the bridge-shape dict. Mirrors
/// `combat_state_to_bridge_state` (`bridge_replay.py:303-316`) with the
/// Rust-port addition of an explicit `kind` discriminant for non-
/// `CardSelect` flavors.
///
/// ```text
/// {
///   'type': 'card_select',          # always — matches Python bridge
///   'kind': 'put_back',             # Rust-only: ChoiceKind discriminant
///   'cards': [{'id': 'Headbutt', 'type': 'Attack', 'upgraded': False},
///             ...],
///   'min_select': 1,
///   'max_select': 1,
///   'can_skip': False,
/// }
/// ```
#[must_use]
pub fn pending_choice_to_value(pc: &PendingChoice, state: &CombatState) -> Value {
    let mut d = obj();
    // `type` is always `"card_select"` for byte-equality with the
    // Python bridge payload (Python has one PendingCardChoice class).
    set(&mut d, "type", "card_select");
    // `kind` is the Rust-side discriminant — see
    // [`choice_kind_python_name`] for the variant→string table.
    set(&mut d, "kind", choice_kind_python_name(pc.kind));
    let cards_list: Vec<Value> = pc
        .options
        .iter()
        .map(|o| choice_option_to_value(o, state))
        .collect();
    d.insert("cards".to_owned(), Value::Array(cards_list));
    set(&mut d, "min_select", pc.min_select);
    set(&mut d, "max_select", pc.max_select);
    set(&mut d, "can_skip", pc.can_skip);
    Value::Object(d)
}

/// The open pending-choice description ([`pending_choice_to_value`]), or
/// `None` when no choice is currently pending — `CombatState.pending_choice()`.
#[must_use]
pub fn pending_choice(state: &CombatState) -> Option<Value> {
    state
        .pending_choice
        .as_ref()
        .map(|pc| pending_choice_to_value(pc, state))
}

// ---------------------------------------------------------------------------
// Hand previews
// ---------------------------------------------------------------------------

/// A clone of `state` with `card_uid` moved from the hand to the play pile —
/// the board as the modifier chain sees it mid-dispatch — so play-gated
/// listeners fire when a preview reads the chain (GH #520).
///
/// Only the two piles move. In particular `cards_played_this_turn` is NOT
/// bumped: the real dispatch raises it only after the card body resolves
/// (`combat_loop.rs:3710`), which is why Lethality's "first Attack this turn"
/// gate reads `== 0`. Bumping it here would gate the boost back off.
#[must_use]
pub fn staged_for_play(state: &CombatState, card_uid: CardUid) -> CombatState {
    let mut staged = state.clone();
    if let Some(pos) = staged.player.hand.iter().position(|&u| u == card_uid) {
        staged.player.hand.remove(pos);
    }
    staged.player.play.push(card_uid);
    staged
}

/// Read-only damage preview for the live hand — `CombatState.card_damage_preview()`.
///
/// For each hand card with base damage, the damage it would deal to each
/// living enemy right now, with the FULL modifier pipeline folded in — the
/// player's Strength / Vigor / Weak, the target's Vulnerable / Intangible,
/// relic and card multipliers (C# `Hook.ModifyDamage`). Nothing is mutated;
/// this is a pure display aid so the workbench can show "9" on a Strike aimed
/// at a Vulnerable target instead of the base "6", matching the live number
/// the game UI shows.
///
/// The outer list is aligned with the hand (one entry per hand card, same
/// order as `state_to_dict(..)["deck"]["hand"]`). Each inner map maps an
/// enemy's FULL encounter-array index — the position in `enemy_uids` that
/// `state_to_dict` labels each enemy with, and the `target_index` the play
/// path expects — to the modified per-hit damage. Cards with no base damage
/// (skills/powers/0-damage) and dead enemy slots are omitted (empty map).
///
/// A typed return, not a `Value`: the Python surface keys the inner dict by
/// INT, which JSON cannot spell. `PyO3` converts the `usize` keys natively; a
/// JSON consumer sees them as strings.
///
/// Each card is scored against [`staged_for_play`] — the modifier
/// chain is read as if that card were mid-dispatch — so play-gated
/// listeners fire. Without it Lethality (`+Amount%` on the turn's first
/// powered Attack, gated on a non-empty Play pile) was silently skipped and
/// a Strike under Lethality 50 previewed its bare 6 instead of 9 (GH #520).
#[must_use]
pub fn card_damage_preview(state: &CombatState) -> Vec<BTreeMap<usize, i32>> {
    let mut out = Vec::with_capacity(state.player.hand.len());
    for card_uid in &state.player.hand {
        let mut entry = BTreeMap::new();
        if let Some(card) = state.cards.iter().find(|c| c.uid == *card_uid) {
            let base = card.base_damage;
            let card_id = card.card_id;
            if base > 0 {
                let staged = staged_for_play(state, *card_uid);
                for (idx, enemy_uid) in staged.enemy_uids.iter().enumerate() {
                    let alive = staged
                        .creatures
                        .iter()
                        .any(|c| c.id == enemy_uid.0 && c.current_hp > 0);
                    if !alive {
                        continue;
                    }
                    let dmg = sts2_core::damage::compute_modified_damage(
                        &staged,
                        sts2_core::hooks::CreatureRef::Player,
                        *enemy_uid,
                        base,
                        sts2_core::hooks::ValueProps::MOVE,
                        Some(card_id),
                    );
                    entry.insert(idx, dmg);
                }
            }
        }
        out.push(entry);
    }
    out
}

/// Read-only block preview for the live hand — the block half of
/// [`card_damage_preview`] (`CombatState.card_block_preview()`).
///
/// For each hand card with base block, returns the block it would actually
/// grant if played right now, with Dexterity / Frail / Fasten folded in via
/// [`sts2_core::card_effects::modify_player_block_move`] — the same
/// `Hook.ModifyBlock` stage `gain_block_player` runs. `None` for cards with
/// no base block, so the list is hand-aligned and a caller can skip a slot
/// without index arithmetic.
///
/// The workbench had no block counterpart to the damage preview, so a
/// Defend under Dexterity 5 rendered its printed "5 Block" while granting
/// 10 — the display gap that made Necrobinder's summon cards look like the
/// better mitigation (GH #520, bug report `wb-b5c453e74f1b80e3`).
#[must_use]
pub fn card_block_preview(state: &CombatState) -> Vec<Option<i32>> {
    let mut out = Vec::with_capacity(state.player.hand.len());
    for card_uid in &state.player.hand {
        let base = state
            .cards
            .iter()
            .find(|c| c.uid == *card_uid)
            .map_or(0, |c| c.base_block);
        if base > 0 {
            let staged = staged_for_play(state, *card_uid);
            // A face preview has no `cardPlay`: Stretch answers it unspent.
            let stretched = sts2_core::fool::stretch_preview_state(&staged, *card_uid);
            out.push(Some(sts2_core::card_effects::modify_player_block_move(
                stretched.as_ref().unwrap_or(&staged),
                base,
            )));
        } else {
            out.push(None);
        }
    }
    out
}

/// Live damage for the live hand, for the e2e net: per hand card, the
/// damage each living enemy would take, as `target index -> (damage per hit
/// after modifiers, hits)`.
///
/// Unlike [`card_damage_preview`] (which reads the card's stored
/// `base_damage`, so misses cards that calculate damage at play time and
/// applies the player's modifiers to Osty attacks), this reads
/// [`sts2_content::cards::live::live_attack`]: calculated attacks (Body Slam,
/// Perfected Strike, ...), Osty attacks dealt by the Osty, and hit counts.
/// The base then runs the same modifier chain as the preview, against
/// [`staged_for_play`]. Indexes are the full encounter-array indexes the
/// preview uses; unhittable enemies ([`CombatState::is_hittable_enemy`]) are
/// omitted, and an empty map means the card deals no damage now. [`card_damage_preview`] is unchanged.
#[must_use]
pub fn card_damage_live(state: &CombatState) -> Vec<BTreeMap<usize, (i32, f32)>> {
    state
        .player
        .hand
        .iter()
        .map(|card_uid| damage_live_of(state, *card_uid))
        .collect()
}

/// [`card_damage_live`] of one hand card.
fn damage_live_of(state: &CombatState, card_uid: CardUid) -> BTreeMap<usize, (i32, f32)> {
    let mut entry = BTreeMap::new();
    let Some(card_id) = state
        .cards
        .iter()
        .find(|c| c.uid == card_uid)
        .map(|c| c.card_id)
    else {
        return entry;
    };
    let staged = staged_for_play(state, card_uid);
    for (idx, enemy_uid) in staged.enemy_uids.iter().enumerate() {
        if !staged.is_hittable_enemy(*enemy_uid) {
            continue;
        }
        let Some(la) = sts2_content::cards::live::live_attack(state, card_uid, *enemy_uid) else {
            break;
        };
        let dmg = sts2_core::damage::compute_modified_damage(
            &staged,
            la.dealer,
            *enemy_uid,
            la.base,
            sts2_core::hooks::ValueProps::MOVE,
            Some(card_id),
        );
        entry.insert(idx, (dmg, la.hits));
    }
    entry
}

/// Live block for the live hand, for the e2e net: [`card_block_preview`] plus
/// the block that card has no `base_block` for. Sovereign Blade grants its
/// Parry amount as powered block (`CalculatedBlockVar` x `ParryPower`), and
/// the `CalculatedBlockVar` cards (Demonic Shield, Mirage, Sacrifice, Stack,
/// Expect a Fight) calculate theirs at play time. `None` when the card grants
/// none. [`card_block_preview`] is unchanged.
#[must_use]
pub fn card_block_live(state: &CombatState) -> Vec<Option<i32>> {
    let mut out = card_block_preview(state);
    for (slot, card_uid) in out.iter_mut().zip(&state.player.hand) {
        if slot.is_some() {
            continue;
        }
        let Some(card) = state.cards.iter().find(|c| c.uid == *card_uid) else {
            continue;
        };
        let staged = staged_for_play(state, *card_uid);
        *slot = if card.card_id == sts2_core::cards::CardId::SovereignBlade {
            let parry = state.player_power_amount(sts2_core::powers::PowerKind::Parry);
            (parry > 0).then(|| {
                let stretched = sts2_core::fool::stretch_preview_state(&staged, *card_uid);
                sts2_core::card_effects::modify_player_block_move(
                    stretched.as_ref().unwrap_or(&staged),
                    parry,
                )
            })
        } else {
            sts2_content::cards::preview::calculated_block(&staged, card, None).filter(|b| *b > 0)
        };
    }
    out
}

/// The damage a hand card's play deals beyond its own attack, for the e2e
/// net's live damage bound (GH #1197): what [`card_gen_live`] returns per card.
///
/// Indexes are the full encounter-array indexes [`card_damage_live`] uses;
/// only living enemies appear and only positive values are kept.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenLive {
    /// Target index -> damage the play adds at once: the Shivs it creates
    /// (each playable at any enemy), Knife Trap's Shivs, the orbs it evokes.
    pub flat: BTreeMap<usize, f64>,
    /// Target index -> damage per energy spent (Multi-Cast's evokes).
    pub per_energy: BTreeMap<usize, f64>,
    /// Target indexes the play's Doom kills outright (End of Days).
    pub kills: Vec<usize>,
    /// By the energy an X-cost play spends (index 0 ..= the player's energy;
    /// the stars for Stardust): target index -> the play's whole X damage at
    /// that X, when it is not linear in X (Tempest's overflow evokes; an X
    /// modifier such as Chemical X on the X attacks, Multi-Cast and Stardust;
    /// Heavenly Drill's doubling, see `x_table`). It replaces the attack x X
    /// and `per_energy` terms. Empty for every other card.
    pub by_x: Vec<BTreeMap<usize, f64>>,
}

impl GenLive {
    fn is_empty(&self) -> bool {
        self.flat.is_empty()
            && self.per_energy.is_empty()
            && self.kills.is_empty()
            && self.by_x.iter().all(BTreeMap::is_empty)
    }

    fn add(&mut self, row: &BTreeMap<usize, f64>, times: f64) {
        for (j, v) in row {
            let x = v * times;
            if x > 0.0 {
                *self.flat.entry(*j).or_insert(0.0) += x;
            }
        }
    }
}

/// The hittable enemies (C# `CombatState.HittableEnemies`,
/// [`CombatState::is_hittable_enemy`]) as `(encounter index, uid, hp)`.
fn living_enemies(state: &CombatState) -> Vec<(usize, CreatureUid, i32)> {
    state
        .enemy_uids
        .iter()
        .enumerate()
        .filter(|(_, uid)| state.is_hittable_enemy(**uid))
        .filter_map(|(i, uid)| {
            sts2_core::creature_by_id(&state.creatures, uid.0).map(|c| (i, *uid, c.current_hp))
        })
        .collect()
}

/// One Shiv's damage on each living enemy (Shiv.cs: `DamageVar(4)`, +2
/// upgraded), after modifiers. Accuracy (`AccuracyPower.ModifyDamageAdditive`,
/// Shiv-tagged powered attacks) is added to the base, where the additive
/// stage would put it: the modifier chain reads the resolving card off the
/// Play pile, and no Shiv is there yet.
fn shiv_row(staged: &CombatState, upgraded: bool) -> BTreeMap<usize, f64> {
    let base = if upgraded { 6 } else { 4 }
        + staged.player_power_amount(sts2_core::powers::PowerKind::Accuracy);
    living_enemies(staged)
        .into_iter()
        .map(|(i, uid, _)| {
            let d = sts2_core::damage::compute_modified_damage(
                staged,
                sts2_core::hooks::CreatureRef::Player,
                uid,
                base,
                sts2_core::hooks::ValueProps::MOVE,
                Some(sts2_core::cards::CardId::Shiv),
            );
            (i, f64::from(d.max(0)))
        })
        .filter(|(_, d)| *d > 0.0)
        .collect()
}

/// One evoke of `orb` per living enemy, after the target's modifiers (an
/// evoke is unpowered: no Strength, no Vulnerable). Lightning hits a random
/// enemy (`LightningOrb.ApplyLightningDamage`), counted as each enemy's share
/// like a random-target attack in [`card_damage_live`]; Dark hits the weakest
/// (`DarkOrb.Evoke`, the engine's first lowest-HP enemy); Glass hits all.
/// Values as the wire's `orbs` (Focus, Infused Core).
fn evoke_row(staged: &CombatState, orb: sts2_core::orbs::Orb) -> BTreeMap<usize, f64> {
    use sts2_core::orbs::{
        focus_bonus, focus_scaled, lightning_orb_bonus, Orb, DARK_ORB_BASE_EVOKE,
    };
    let living = living_enemies(staged);
    let focus = focus_bonus(staged);
    let hit = |uid: CreatureUid, v: i32| -> f64 {
        f64::from(
            sts2_core::damage::compute_modified_damage(
                staged,
                sts2_core::hooks::CreatureRef::Player,
                uid,
                v,
                sts2_core::hooks::ValueProps::UNPOWERED,
                None,
            )
            .max(0),
        )
    };
    let mut row = BTreeMap::new();
    match orb {
        Orb::Lightning => {
            let v = focus_scaled(8, focus) + lightning_orb_bonus(staged);
            #[allow(clippy::cast_precision_loss)]
            let n = living.len().max(1) as f64;
            for (i, uid, _) in &living {
                row.insert(*i, hit(*uid, v) / n);
            }
        }
        Orb::Dark { stored } => {
            let v = stored.saturating_add(DARK_ORB_BASE_EVOKE).max(0);
            if let Some((i, uid, _)) = living.iter().min_by_key(|(_, _, hp)| *hp) {
                row.insert(*i, hit(*uid, v));
            }
        }
        Orb::Glass { current_passive } => {
            let v = focus_scaled(current_passive, focus).saturating_mul(2);
            if v > 0 {
                for (i, uid, _) in &living {
                    row.insert(*i, hit(*uid, v));
                }
            }
        }
        Orb::Frost | Orb::Plasma | Orb::Empty => {}
    }
    row.retain(|_, v| *v > 0.0);
    row
}

/// The orbs `card`'s play channels (v0.111.0 `OnPlay`), in order, from the
/// engine's shared table ([`sts2_core::card_mechanics::resolve_channels`]):
/// `Some` freshly channelled orb, `None` for a random one (Chaos). `x` is the X
/// an X-cost play resolves (Tempest: `ResolveEnergyXValue()`, the +1 upgrade is
/// in the table); Voltaic's count comes from the combat's Lightning history
/// (Voltaic.cs:30-34), Chill's from the hittable enemies (Chill.cs).
fn channel_seq(state: &CombatState, card: &Card, x: i32) -> Vec<Option<sts2_core::orbs::Orb>> {
    use sts2_core::card_mechanics::{resolve_channels, ORB_KIND_ORDER};
    use sts2_core::orbs::{Orb, OrbKind};
    let hittable = u32::try_from(living_enemies(state).len()).unwrap_or(u32::MAX);
    let ch = resolve_channels(state, card, x, hittable);
    let mut seq = Vec::new();
    for (kind, n) in ORB_KIND_ORDER.iter().zip(ch.by_kind) {
        let orb = match kind {
            OrbKind::Lightning => Orb::lightning(),
            OrbKind::Frost => Orb::frost(),
            OrbKind::Dark => Orb::dark(),
            OrbKind::Plasma => Orb::plasma(),
            OrbKind::Glass => Orb::glass(),
            OrbKind::Empty => continue,
        };
        seq.extend(std::iter::repeat_n(Some(orb), n as usize));
    }
    seq.extend(std::iter::repeat_n(None, ch.random as usize));
    seq
}

/// What channelling `seq` one orb at a time into the seated player's belt
/// evokes, per channel (an empty row where it evokes nothing or an orb of
/// random kind). `OrbCmd.Channel` (OrbCmd.cs:68-92): a character with no base
/// slots gets one on an empty zero-slot belt (the engine's `channel_orb`: an
/// unsized Defect fixture belt gets the Defect base), a belt still at zero
/// channels nothing, and a full belt evokes its front orb before the new one
/// is enqueued at the back, so a long enough play evokes the orbs it
/// channelled itself.
fn overflow_rows(
    staged: &CombatState,
    seq: &[Option<sts2_core::orbs::Orb>],
) -> Vec<BTreeMap<usize, f64>> {
    let mut slots = staged.player.orbs.clone();
    if slots.capacity() == 0 {
        if staged.player.character == Some(sts2_core::potions::CharacterId::Defect) {
            slots.ensure_defect_base();
        } else {
            slots.add_slots(1);
        }
    }
    let cap = slots.capacity();
    let mut queue: std::collections::VecDeque<Option<sts2_core::orbs::Orb>> = slots
        .slots
        .iter()
        .filter(|o| !o.is_empty())
        .map(|o| Some(*o))
        .collect();
    let mut out = Vec::with_capacity(seq.len());
    for orb in seq {
        if cap == 0 {
            out.push(BTreeMap::new());
            continue;
        }
        let row = if queue.len() >= cap {
            queue
                .pop_front()
                .flatten()
                .map_or_else(BTreeMap::new, |front| evoke_row(staged, front))
        } else {
            BTreeMap::new()
        };
        out.push(row);
        queue.push_back(*orb);
    }
    out
}

/// Ignition's Plasma lands on the target teammate (Ignition.cs:33,
/// `OrbCmd.Channel<PlasmaOrb>(cardPlay.Target.Player)`): per enemy, the most
/// any living teammate's overflow evoke deals there, priced from that seat
/// (its belt, its Focus). Solo there is no teammate, and the card is
/// unplayable (`CardModel.CanPlay`'s `AnyAlly` gate).
fn ignition_row(staged: &CombatState) -> BTreeMap<usize, f64> {
    let mut best: BTreeMap<usize, f64> = BTreeMap::new();
    for (i, ally) in staged.allies.iter().enumerate() {
        if ally.current_hp <= 0 {
            continue;
        }
        let mut seated = staged.clone();
        sts2_core::coop::swap_seat(&mut seated, i);
        for row in overflow_rows(&seated, &[Some(sts2_core::orbs::Orb::plasma())]) {
            for (j, v) in row {
                let b = best.entry(j).or_insert(0.0);
                if v > *b {
                    *b = v;
                }
            }
        }
    }
    best.retain(|_, v| *v > 0.0);
    best
}

/// The X-cost plays whose damage is not linear in the X resource spent, as a
/// table by that amount (index 0 ..= the energy, or the stars for Stardust,
/// available now): what [`GenLive::by_x`] carries for them. X is
/// `ResolveEnergyXValue()` / `ResolveStarXValue()`, the amount spent folded
/// through `Hook.ModifyXValue` (CardModel.cs:1117-1146; Chemical X +2).
///
/// - Whirlwind, Skewer, Eradicate, Volley, Repeat Shot hit X times
///   (`WithHitCount(ResolveEnergyXValue())`); Stardust X times on stars.
///   [`card_damage_live`] reports one hit per energy for these (Stardust: X
///   hits at all the stars), exact for the knapsack's linear term unless an X
///   modifier shifts X, so the table exists only then.
/// - Heavenly Drill hits X times, doubled once X reaches 4
///   (HeavenlyDrill.cs:32-37): also a table whenever the doubling at the
///   energy available differs from the doubling at some smaller spend.
/// - Multi-Cast evokes the front orb X times, +1 upgraded (MultiCast.cs:22-29):
///   a table when an X modifier shifts X (the table then carries the upgrade
///   evoke too).
///
/// `None` when the linear term is exact or the play deals nothing.
fn x_table(
    state: &CombatState,
    card: &Card,
    staged: &CombatState,
    front: Option<sts2_core::orbs::Orb>,
) -> Option<Vec<BTreeMap<usize, f64>>> {
    use sts2_core::cards::CardId;
    let id = card.card_id;
    let on_stars = id == CardId::Stardust;
    let avail = if on_stars {
        i32::from(state.player.stars)
    } else {
        state.player.energy
    }
    .max(0);
    let x_of = |k: i32| sts2_core::relic_effects::dispatch_modify_x_value(state, k);
    let shifted = (0..=avail).any(|k| x_of(k) != k);
    let drill = |x: i32| if x >= 4 { 2 } else { 1 };
    // `rows`: the play's damage per unit of `count(k)`.
    let (rows, count): (BTreeMap<usize, f64>, Box<dyn Fn(i32) -> i32 + '_>) = match id {
        CardId::Whirlwind
        | CardId::Skewer
        | CardId::Eradicate
        | CardId::Volley
        | CardId::RepeatShot
        | CardId::Stardust => {
            if !shifted {
                return None;
            }
            // Stardust's live hits are X at all the stars: per unit of X.
            let unit = if on_stars {
                f64::from(x_of(avail))
            } else {
                1.0
            };
            if unit <= 0.0 {
                return None;
            }
            let rows = damage_live_of(state, card.uid)
                .into_iter()
                .map(|(j, (d, h))| (j, f64::from(d.max(0)) * f64::from(h) / unit))
                .collect();
            (rows, Box::new(x_of))
        }
        CardId::HeavenlyDrill => {
            let at_avail = drill(x_of(avail));
            if !shifted && (0..=avail).all(|k| drill(x_of(k)) == at_avail) {
                return None;
            }
            let unit = f64::from(at_avail);
            let rows = damage_live_of(state, card.uid)
                .into_iter()
                .map(|(j, (d, h))| (j, f64::from(d.max(0)) * f64::from(h) / unit))
                .collect();
            (rows, Box::new(move |k| x_of(k) * drill(x_of(k))))
        }
        CardId::MultiCast => {
            if !shifted {
                return None;
            }
            let up = i32::from(card.upgraded);
            (
                front.map_or_else(BTreeMap::new, |o| evoke_row(staged, o)),
                Box::new(move |k| x_of(k) + up),
            )
        }
        _ => return None,
    };
    let table: Vec<BTreeMap<usize, f64>> = (0..=avail)
        .map(|k| {
            let n = f64::from(count(k).max(0));
            rows.iter()
                .map(|(j, v)| (*j, v * n))
                .filter(|(_, v)| *v > 0.0)
                .collect()
        })
        .collect();
    (!table.iter().all(BTreeMap::is_empty)).then_some(table)
}

/// Generated damage for the live hand, for the e2e net (GH #1197): per hand
/// card, the deterministic damage its play adds beyond its own attack
/// ([`card_damage_live`]), or `None` when it adds none.
///
/// - Shivs: Blade Dance (3, +1 upgraded), Cloak and Dagger (1, +1), Up My
///   Sleeve (3, +1) each create Shivs, 0-cost attacks, so the play is worth
///   every Shiv that fits in the hand ([`sts2_core::cards::HAND_LIMIT`]; the
///   rest go to the discard pile) at the Shiv's damage after modifiers.
/// - Knife Trap autoplays every Shiv in the exhaust pile at its target
///   (upgraded: upgrades them first).
/// - Dualcast (2), Quadcast (4), Multi-Cast (X, +1 upgraded) evoke the front
///   orb.
/// - A channel into a full belt evokes the front orb first (`OrbCmd.Channel`),
///   the play's own orbs too once the belt's are gone (`overflow_rows`),
///   for every channelling card of the engine's table. Tempest's X Lightning
///   (+1 upgraded) is not linear in X: its damage goes into `by_x`, by the
///   energy spent (X through the relics' `ModifyXValue`, Chemical X). Voltaic
///   channels the combat's Lightning count so far; Ignition evokes the
///   target teammate's front orb (`ignition_row`).
/// - End of Days applies 29 Doom (37 upgraded) to every enemy, then kills the
///   doomed (`DoomPower.DoomKill`); `kills` lists the enemies that dies to.
///
/// Random effects stay out (a random orb's own value, random cards).
#[must_use]
pub fn card_gen_live(state: &CombatState) -> Vec<Option<GenLive>> {
    use sts2_core::cards::{CardId, CardTag, HAND_LIMIT};
    let mut out = Vec::with_capacity(state.player.hand.len());
    for card_uid in &state.player.hand {
        let Some(card) = state.cards.iter().find(|c| c.uid == *card_uid) else {
            out.push(None);
            continue;
        };
        let up = card.upgraded;
        let shivs = match card.card_id {
            CardId::BladeDance | CardId::UpMySleeve => 3 + usize::from(up),
            CardId::CloakAndDagger => 1 + usize::from(up),
            _ => 0,
        };
        let evokes = match card.card_id {
            CardId::Dualcast => 2,
            CardId::Quadcast => 4,
            _ => 0,
        };
        let mech = sts2_core::card_mechanics::card_mechanics(card.card_id, up);
        let per_x = matches!(mech.rule, sts2_core::card_mechanics::ChannelRule::PerX);
        let channels = if card.card_id == CardId::Ignition || per_x {
            Vec::new()
        } else {
            channel_seq(state, card, 0)
        };
        let relevant = shivs > 0
            || evokes > 0
            || !channels.is_empty()
            || per_x
            || matches!(
                card.card_id,
                CardId::KnifeTrap
                    | CardId::MultiCast
                    | CardId::EndOfDays
                    | CardId::Ignition
                    | CardId::Whirlwind
                    | CardId::Skewer
                    | CardId::Eradicate
                    | CardId::Volley
                    | CardId::RepeatShot
                    | CardId::Stardust
                    | CardId::HeavenlyDrill
            );
        if !relevant {
            out.push(None);
            continue;
        }
        let staged = staged_for_play(state, *card_uid);
        let mut g = GenLive::default();
        if shivs > 0 {
            let room = HAND_LIMIT.saturating_sub(staged.player.hand.len());
            #[allow(clippy::cast_precision_loss)]
            g.add(&shiv_row(&staged, false), shivs.min(room) as f64);
        }
        if card.card_id == CardId::KnifeTrap {
            for u in &state.player.exhaust {
                if let Some(c) = state.cards.iter().find(|c| c.uid == *u) {
                    if c.tags.contains(CardTag::Shiv) {
                        g.add(&shiv_row(&staged, up || c.upgraded), 1.0);
                    }
                }
            }
        }
        let belt: Vec<sts2_core::orbs::Orb> = state
            .player
            .orbs
            .slots
            .iter()
            .copied()
            .filter(|o| !o.is_empty())
            .collect();
        if let Some(front) = belt.first() {
            let row = evoke_row(&staged, *front);
            if evokes > 0 {
                g.add(&row, f64::from(evokes));
            }
            if card.card_id == CardId::MultiCast {
                g.per_energy = row.clone();
                if up {
                    g.add(&row, 1.0);
                }
            }
        }
        for row in overflow_rows(&staged, &channels) {
            g.add(&row, 1.0);
        }
        if card.card_id == CardId::Ignition {
            g.add(&ignition_row(&staged), 1.0);
        }
        if per_x {
            // Index k: the play spends k energy, X = `ModifyXValue(k)`.
            let energy = state.player.energy.max(0);
            let x_of = |k: i32| sts2_core::relic_effects::dispatch_modify_x_value(state, k);
            let seq = channel_seq(state, card, x_of(energy));
            let rows = overflow_rows(&staged, &seq);
            g.by_x = (0..=energy)
                .map(|k| {
                    let n = channel_seq(state, card, x_of(k)).len();
                    let mut acc = GenLive::default();
                    for row in rows.iter().take(n) {
                        acc.add(row, 1.0);
                    }
                    acc.flat
                })
                .collect();
            if g.by_x.iter().all(BTreeMap::is_empty) {
                g.by_x.clear();
            }
        }
        if card.card_id == CardId::EndOfDays {
            g.kills = doom_kills(&staged, if up { 37 } else { 29 });
        }
        if let Some(table) = x_table(state, card, &staged, belt.first().copied()) {
            // The table is the play's whole X damage: Multi-Cast's linear
            // evokes (and its upgrade evoke) move into it.
            if card.card_id == CardId::MultiCast {
                g.per_energy.clear();
                g.flat.clear();
            }
            g.by_x = table;
        }
        out.push((!g.is_empty()).then_some(g));
    }
    out
}

/// The living enemies `doom` more Doom would kill (`EndOfDays.OnPlay`): the
/// Doom applied as the play would (Artifact blocks it), then
/// `DoomPower.IsOwnerDoomed` (`CurrentHp <= Amount`).
fn doom_kills(staged: &CombatState, doom: i32) -> Vec<usize> {
    let mut after = staged.clone();
    for (_, uid, _) in living_enemies(staged) {
        sts2_core::card_effects::apply_power_to_enemy(
            &mut after,
            uid,
            sts2_core::powers::PowerKind::Doom,
            doom,
        );
    }
    living_enemies(staged)
        .into_iter()
        .filter(|(_, uid, hp)| {
            let d: i32 = after
                .creatures
                .iter()
                .find(|c| c.id == uid.0)
                .map_or(0, |c| {
                    c.powers
                        .iter()
                        .filter_map(|pu| after.powers.iter().find(|p| p.id == pu.0))
                        .filter(|p| p.kind == sts2_core::powers::PowerKind::Doom)
                        .map(|p| p.amount)
                        .sum()
                });
            d > 0 && *hp <= d
        })
        .map(|(i, _, _)| i)
        .collect()
}

/// What the end of the player's turn does before the enemies act, for the e2e
/// net's end-of-turn HP loss (GH #1197): what [`end_turn_live`] returns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EndTurnLive {
    /// The change in the player's block between ending the turn and the
    /// enemies' moves: the gains less what the turn end's self-damage ate.
    /// Negative when it ate block the player had.
    pub block: i32,
    /// Encounter indexes of the living enemies certain to die before their
    /// move resolves.
    pub dies: Vec<usize>,
    /// HP the player loses (net of heals, at least 0) before the enemies act:
    /// Burn-type cards in hand, Constrict, Disintegration.
    pub self_loss: i32,
    /// The player's Buffer stacks left for the enemies' hits.
    pub buffer: i32,
    /// Beating Remnant's HP-loss budget left for this turn
    /// (`MaxHpLoss - DamageReceivedThisTurn`); `None` without the relic.
    pub remnant: Option<i32>,
}

/// How many encounter enemies C# `CombatState.HittableEnemies` lists
/// ([`CombatState::is_hittable_enemy`]), as the engine resolves it everywhere
/// (`Effect::DealDamageToHittableEnemies`, the orbs' random target).
fn hittable_count(state: &CombatState) -> usize {
    state
        .enemy_uids
        .iter()
        .filter(|uid| state.is_hittable_enemy(**uid))
        .count()
}

/// Drop the last channelled orb without its evoke (`OrbCmd.EvokeLast`'s slot
/// handling, as [`sts2_core::orbs::evoke_last`] does it).
fn drop_last_orb(state: &mut CombatState) {
    if let Some(idx) = state.player.orbs.slots.iter().rposition(|o| !o.is_empty()) {
        state.player.orbs.slots.remove(idx);
        state.player.orbs.slots.push(Orb::Empty);
    }
}

/// The deterministic part of the end of the player's turn and of the enemies'
/// turn start, for the e2e net (GH #1197), in the v0.111.0 order
/// (`CombatManager.EndPlayerTurnPhaseOneInternal`, `DoTurnEnd`,
/// `EndPlayerTurnPhaseTwoInternal`, `StartTurn`), run by the engine's own
/// hooks on a clone:
///
/// 1. `AfterAutoPostPlayPhaseEntered`: every Howl from Beyond in the exhaust
///    pile auto-plays. Stampede and Simpleton (random hand cards) and
///    I Am Invincible (the hidden top of the draw pile) are left out.
/// 2. `BeforeSideTurnEnd`, Early then normal, powers then relics: Plating,
///    Orichalcum / Fake Orichalcum (latched on 0 block), Cloak Clasp, Ripple
///    Basin, Hailstorm, The Bomb, Screaming Flagon, Stone Calendar, the
///    player's Doom.
/// 3. `DoTurnEnd`: every orb's passive front to back, the front one twice
///    under Gold-Plated Cables (Lightning hits a random enemy, so it counts
///    only while exactly one is hittable); the Ethereal exhausts (Feel No
///    Pain); then the turn-end-in-hand cards (Burn, Decay, Infection, Toxic,
///    Wither through block; Beckon, Bad Luck, Regret unblockable).
/// 4. `AfterSideTurnEnd`: Consuming Shadow evokes the last orb `Amount` times
///    (Frost block, Glass all, Lightning only with one hittable enemy, Dark
///    the weakest unless a random zap was skipped before it), then the other
///    powers (Constrict), Parrying Shield (6 at 10+ block, only with one
///    hittable enemy), then the late powers (Disintegration).
/// 5. The enemies' turn start: Hardened Shell's budget resets
///    (`BeforeSideTurnStart`), then Poison ticks `min(Amount, 1 + Accelerant)`
///    times (`PoisonPower.Trigger`) before any enemy moves.
///
/// The walk stops where the combat ends or a choice opens (a choice already
/// open on `state` is dropped first). The damage runs through the engine's
/// pipeline, so block, Intangible, Hardened Shell, Buffer, Beating Remnant and
/// the like apply as in the game. A random hit the walk skips can only add
/// damage. When a listener on a bus the walk runs rolls a combat RNG stream
/// anyway (Juggernaut, Obsidian Vice, an exhaust or Howl listener), the walk
/// is repeated with the hidden streams reseeded (`END_TURN_RESEEDS` walks) and only
/// what every walk agrees on is kept, so `dies` lists certain deaths only.
/// Consuming Shadow runs ahead of the other `AfterSideTurnEnd` powers rather
/// than in its application order.
#[must_use]
pub fn end_turn_live(state: &CombatState) -> EndTurnLive {
    let (mut out, deterministic) = end_turn_read(state, &end_turn_walk(state, None));
    if !deterministic {
        for n in 1..END_TURN_RESEEDS {
            out = merge_end_turn(out, end_turn_read(state, &end_turn_walk(state, Some(n))).0);
        }
    }
    out
}

/// The walks [`end_turn_live`] runs when the first one read a combat RNG
/// stream: the first, then this many less one with the hidden streams reseeded
/// (`RunRngSet::reseed_combat_streams`). Two equally likely random targets
/// agree across all of them with probability `2^-(END_TURN_RESEEDS - 1)`.
const END_TURN_RESEEDS: u32 = 8;

/// The certain part of two walks' results: the deaths both list, the lesser
/// block, Buffer and Remnant budget, the greater self-damage.
fn merge_end_turn(a: EndTurnLive, b: EndTurnLive) -> EndTurnLive {
    EndTurnLive {
        block: a.block.min(b.block),
        dies: a.dies.into_iter().filter(|j| b.dies.contains(j)).collect(),
        self_loss: a.self_loss.max(b.self_loss),
        buffer: a.buffer.min(b.buffer),
        remnant: match (a.remnant, b.remnant) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        },
    }
}

/// [`EndTurnLive`] from the walked clone `sim` of `state`, and whether the walk
/// read no RNG (its outcome is then the only one).
fn end_turn_read(state: &CombatState, sim: &CombatState) -> (EndTurnLive, bool) {
    let dies = state
        .enemy_uids
        .iter()
        .enumerate()
        .filter(|(_, uid)| alive(state, **uid) && !alive(sim, **uid))
        .map(|(i, _)| i)
        .collect();
    let remnant = sim
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::BeatingRemnant && !r.is_melted)
        // `BeatingRemnant.MaxHpLoss` (20) less `DamageReceivedThisTurn`.
        .map(|r| (20 - r.counter).max(0));
    let deterministic = sim.run_rng == state.run_rng && sim.rng == state.rng;
    (
        EndTurnLive {
            block: sim.player.block - state.player.block,
            dies,
            self_loss: (state.player.current_hp - sim.player.current_hp).max(0),
            buffer: sim.player_power_amount(PowerKind::Buffer).max(0),
            remnant,
        },
        deterministic,
    )
}

/// One walk of [`end_turn_live`] on a clone of `state`; `reseed` names the
/// outcome stream the hidden combat RNG is reseeded to first.
fn end_turn_walk(state: &CombatState, reseed: Option<u32>) -> CombatState {
    use sts2_core::combat_loop::{
        auto_play_howls_from_exhaust, combat_is_ending, execute_turn_end_in_hand_effect,
        exhaust_turn_end_ethereals, has_turn_end_in_hand_effect, turn_end_ethereal_cards,
    };
    use sts2_core::hooks::{self, CreatureRef, TurnEndPass, TurnSideCtx, ValueProps};
    use sts2_core::orbs::trigger_orb_passive_at;
    use sts2_core::{relic_effects, RelicKind, Side};

    let halted = |s: &CombatState| s.is_over || s.pending_choice.is_some() || combat_is_ending(s);
    let side = TurnSideCtx { side: Side::Player };
    let mut sim = state.clone();
    // A choice already open is the state's, not one the walk opened.
    sim.pending_choice = None;
    if let Some(n) = reseed {
        let base = 0x1197_e2e0;
        sim.run_rng.reseed_combat_streams(base, n);
        sim.rng = sts2_core::dotnet_rng::Rng::from_search_seed(
            sts2_core::run_rng_set::outcome_stream_seed_u64(base, n, 0x1197),
        );
    }
    'walk: {
        // 1. AfterAutoPostPlayPhaseEntered.
        auto_play_howls_from_exhaust(&mut sim);
        if halted(&sim) {
            break 'walk;
        }
        // 2. BeforeSideTurnEnd.
        sim.combat_log.block_zero_at_turn_end = sim.player.block == 0;
        hooks::dispatch_before_turn_end_pass(&mut sim, &side, TurnEndPass::Early);
        let _ = relic_effects::dispatch_before_turn_end_early(&mut sim, Side::Player);
        hooks::dispatch_before_turn_end_pass(&mut sim, &side, TurnEndPass::Normal);
        let _ = relic_effects::dispatch_before_turn_end(&mut sim, Side::Player);
        if halted(&sim) {
            break 'walk;
        }
        let hand_count = i32::try_from(sim.player.hand.len()).unwrap_or(i32::MAX);
        // 3. DoTurnEnd: the orb passives.
        let cables = sim.relics.owns(RelicKind::GoldPlatedCables);
        let filled: Vec<usize> = sim
            .player
            .orbs
            .slots
            .iter()
            .enumerate()
            .filter(|(_, o)| !o.is_empty())
            .map(|(i, _)| i)
            .collect();
        let mut random_skipped = false;
        for (n, &idx) in filled.iter().enumerate() {
            let triggers = if cables && n == 0 { 2 } else { 1 };
            for _ in 0..triggers {
                match sim.player.orbs.slots.get(idx).copied() {
                    Some(Orb::Lightning) if hittable_count(&sim) != 1 => random_skipped = true,
                    Some(Orb::Empty) | None => {}
                    Some(_) => trigger_orb_passive_at(&mut sim, idx),
                }
            }
        }
        if halted(&sim) {
            break 'walk;
        }
        // The Ethereal exhausts, then the turn-end-in-hand cards, both sorted
        // from the hand before either runs.
        let turn_end_cards: Vec<CardUid> = sim
            .player
            .hand
            .iter()
            .copied()
            .filter(|uid| {
                sim.cards
                    .get(uid.0.get() as usize - 1)
                    .is_some_and(|c| has_turn_end_in_hand_effect(c.card_id))
            })
            .collect();
        let ethereal = turn_end_ethereal_cards(&sim);
        exhaust_turn_end_ethereals(&mut sim, &ethereal);
        for uid in turn_end_cards {
            if halted(&sim) {
                break 'walk;
            }
            if !sim.player.hand.contains(&uid) {
                continue;
            }
            let id = sim.cards[uid.0.get() as usize - 1].card_id;
            let _ = execute_turn_end_in_hand_effect(&mut sim, uid, id, hand_count);
        }
        if halted(&sim) {
            break 'walk;
        }
        // 4. AfterSideTurnEnd: Consuming Shadow here, then the engine's walk
        // with its stacks spent.
        let shadow = sim.player_power_amount(PowerKind::ConsumingShadow);
        consuming_shadow(&mut sim, shadow, random_skipped);
        let shadow_uids: Vec<PowerUid> = sim
            .player_power_uids()
            .into_iter()
            .filter(|u| {
                sim.powers
                    .get_by_id(u.0)
                    .is_some_and(|p| p.kind == PowerKind::ConsumingShadow)
            })
            .collect();
        for u in shadow_uids {
            if let Some(p) = sim.powers.get_mut_by_id(u.0) {
                p.amount = 0;
            }
        }
        if halted(&sim) {
            break 'walk;
        }
        sim.combat_log.block_before_late_turn_end = sim.player.block;
        hooks::dispatch_after_turn_end(&mut sim, &side);
        if halted(&sim) {
            break 'walk;
        }
        let parry = sim
            .relics
            .iter()
            .any(|r| r.kind == RelicKind::ParryingShield && !r.is_melted);
        if parry && sim.combat_log.block_before_late_turn_end >= 10 && hittable_count(&sim) == 1 {
            if let Some(&target) = sim.enemy_uids.iter().find(|uid| alive(&sim, **uid)) {
                let _ = sts2_core::damage::apply_damage(
                    &mut sim,
                    CreatureRef::Player,
                    target,
                    6,
                    ValueProps::UNPOWERED,
                    None,
                );
            }
        }
        if halted(&sim) {
            break 'walk;
        }
        hooks::dispatch_after_turn_end_late(&mut sim, &side);
        if halted(&sim) {
            break 'walk;
        }
        // 5. The enemies' turn start.
        reset_hardened_shells(&mut sim);
        tick_poison(&mut sim);
    }
    sim
}

/// `true` while the creature `uid` names has HP left.
fn alive(state: &CombatState, uid: CreatureUid) -> bool {
    state
        .creatures
        .iter()
        .any(|c| c.id == uid.0 && c.current_hp > 0)
}

/// `ConsumingShadowPower.AfterSideTurnEnd`: evoke the last orb `stacks` times,
/// keeping only what is certain. Frost and Glass evoke as the engine does; a
/// Lightning evoke hits a random enemy, so it runs only with one hittable
/// enemy (else the orb is dropped and later random-dependent evokes skip); a
/// Dark evoke hits the weakest enemy, unknown once a random hit was skipped.
fn consuming_shadow(sim: &mut CombatState, stacks: i32, mut random_skipped: bool) {
    use sts2_core::orbs::evoke_last;
    for _ in 0..stacks.max(0) {
        if sim.is_over {
            break;
        }
        let Some(last) = sim
            .player
            .orbs
            .slots
            .iter()
            .rev()
            .find(|o| !o.is_empty())
            .copied()
        else {
            break;
        };
        match last {
            Orb::Lightning if hittable_count(sim) != 1 => {
                random_skipped = true;
                drop_last_orb(sim);
            }
            Orb::Dark { .. } if random_skipped => drop_last_orb(sim),
            Orb::Frost | Orb::Glass { .. } | Orb::Lightning | Orb::Dark { .. } => evoke_last(sim),
            _ => drop_last_orb(sim),
        }
    }
}

/// `HardenedShellPower.BeforeSideTurnStart`: `damageReceivedThisTurn = 0` at
/// every side's turn start (the engine keeps it in the power's `counter`).
fn reset_hardened_shells(sim: &mut CombatState) {
    let shells: Vec<PowerUid> = sim
        .enemy_uids
        .iter()
        .filter_map(|uid| sim.creatures.iter().find(|c| c.id == uid.0))
        .flat_map(|c| c.powers.iter().copied())
        .filter(|u| {
            sim.powers
                .get_by_id(u.0)
                .is_some_and(|p| p.kind == PowerKind::HardenedShell)
        })
        .collect();
    for u in shells {
        if let Some(p) = sim.powers.get_mut_by_id(u.0) {
            p.counter = 0;
        }
    }
}

/// The enemies' `AfterSideTurnStart` Poison: `min(Amount, 1 + Accelerant)`
/// ticks of `Amount - i` HP loss, unblockable (`PoisonPower.Trigger`).
fn tick_poison(sim: &mut CombatState) {
    use sts2_core::hooks::{apply_effect, CreatureRef, Effect, ValueProps};
    let accel = sts2_core::coop::players_power_amount(sim, PowerKind::Accelerant).max(0);
    let uids: Vec<CreatureUid> = sim.enemy_uids.to_vec();
    for uid in uids {
        let amount = sim
            .creatures
            .iter()
            .find(|c| c.id == uid.0 && c.current_hp > 0)
            .map_or(0, |c| {
                c.powers
                    .iter()
                    .filter_map(|p| sim.powers.get_by_id(p.0))
                    .filter(|p| p.kind == PowerKind::Poison)
                    .map(|p| p.amount)
                    .sum()
            });
        for i in 0..amount.min(1 + accel) {
            apply_effect(
                sim,
                Effect::DealHpLoss {
                    dealer: CreatureRef::Nobody,
                    target: CreatureRef::Enemy(uid),
                    amount: amount - i,
                    props: ValueProps::UNBLOCKABLE.union(ValueProps::UNPOWERED),
                },
            );
            if !alive(sim, uid) {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Read-only getters beside the wire (GH #1097)
// ---------------------------------------------------------------------------
//
// The e2e campaign's tokenizer reads these from `sts2_rs.CombatState`
// without a second `combat_state_to_value` call. None of them changes the
// wire dict or `WIRE_VERSION`.

/// The enemy creature `uid` names, or the same error
/// [`combat_state_to_value`] raises for a state that lost one.
fn enemy_creature<'a>(
    state: &'a CombatState,
    uid: CreatureUid,
    path: &str,
) -> Result<&'a sts2_core::Creature, EmitError> {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .ok_or_else(|| EmitError::at(path, "enemy uid not in creatures"))
}

/// The intents creature `c` telegraphs, head first. Empty when it
/// telegraphs nothing.
///
/// The head is the creature's primed `intent`, the one the wire emits as
/// `intents[0]`. When that head is still the first intent of the current
/// move (the AI's `current_move().intents`), the rest of that move's intents
/// follow, as the game shows them (`MoveState` takes `params
/// AbstractIntent[]`). A head the engine re-primed away from the move table,
/// such as the Test Subject's escalated MULTI_CLAW hit count or a revive's
/// `Heal`, stands alone, so a tail is never attached to a head it does not
/// belong to.
#[must_use]
pub fn telegraph_intents(c: &sts2_core::Creature) -> smallvec::SmallVec<[Intent; 4]> {
    let Some(head) = c.intent else {
        return smallvec::SmallVec::new();
    };
    match c
        .ai
        .as_ref()
        .and_then(sts2_core::MonsterAi::current_move_opt)
    {
        Some(m) if m.intents.first() == Some(&head) => m.intents.clone(),
        _ => smallvec::smallvec![head],
    }
}

/// `CombatState.telegraph()`: per enemy, in the wire's `encounter` order,
/// the full telegraph from [`telegraph_intents`] as intent dicts in the
/// [`intent_to_value`] shape. Element 0 of each non-empty list equals the
/// wire's `encounter[i]["intents"][0]`; an enemy with no telegraph has `[]`.
///
/// # Errors
///
/// An enemy uid with no creature, as [`combat_state_to_value`] reports it.
pub fn enemy_telegraphs(state: &CombatState) -> Result<Value, EmitError> {
    let mut out = Vec::with_capacity(state.enemy_uids.len());
    for uid in &state.enemy_uids {
        let c = enemy_creature(state, *uid, "$.telegraph")?;
        let dealer = CreatureUid(c.id);
        out.push(Value::Array(
            telegraph_intents(c)
                .into_iter()
                .map(|intent| intent_to_value(intent, state, dealer))
                .collect(),
        ));
    }
    Ok(Value::Array(out))
}

/// The most steps [`enemy_intent_forecast`] walks: a step number is a `u8`.
pub const MAX_FORECAST_STEPS: usize = u8::MAX as usize;

/// `CombatState.peek_intents(k)`: per enemy, in the wire's `encounter`
/// order, the `k` moves its AI is predicted to telegraph after the current
/// one ([`sts2_core::MonsterAi::forecast_after_current`]), as
/// `{"step": int, "certainty": float, "move_id": str, "intents": [intent
/// dict, …]}`.
///
/// Step 1 is the move rolled at the end of the coming enemy turn, whether or
/// not the monster has moved yet, so the list never repeats the telegraph.
/// `certainty` is the product of the weight shares of the random picks on
/// the way (`1.0` along a fixed pattern). Intent dicts have the
/// [`intent_to_value`] shape, baked against the current board. The three
/// moves whose numbers C# computes at telegraph time (TestSubject
/// MULTI_CLAW's hit count, TheForgotten DREAD, WaterfallGiant EXPLODE) carry
/// the number the game will show at that step, counting the predicted moves
/// before it, not the move table's (#1181,
/// [`sts2_content::monsters::forecast`]). `move_id` is
/// the engine's move-state id: hidden state, for a privileged consumer only.
/// An enemy without an AI, or with one parked on a branch state, has `[]`.
/// Dead enemies keep their AI's forecast; consumers mask them with the
/// wire's `is_dead`.
///
/// Read-only: no RNG draw and no change to `state`.
///
/// # Errors
///
/// `k` above [`MAX_FORECAST_STEPS`], or an enemy uid with no creature.
pub fn enemy_intent_forecast(state: &CombatState, k: usize) -> Result<Value, EmitError> {
    if k > MAX_FORECAST_STEPS {
        return Err(EmitError::new(format!(
            "peek_intents: k={k} is above the {MAX_FORECAST_STEPS}-step limit"
        )));
    }
    let mut out = Vec::with_capacity(state.enemy_uids.len());
    for uid in &state.enemy_uids {
        let c = enemy_creature(state, *uid, "$.peek_intents")?;
        let dealer = CreatureUid(c.id);
        let steps =
            c.ai.as_ref()
                .map(|ai| ai.forecast_after_current(k, state))
                .unwrap_or_default();
        // The moves performed before each step: the current telegraph (the
        // coming enemy turn performs it), then the earlier steps.
        let mut prior: Vec<&str> =
            c.ai.as_ref()
                .and_then(|ai| ai.current_move_opt())
                .map(|m| m.id)
                .into_iter()
                .collect();
        let mut rows: Vec<Value> = Vec::with_capacity(steps.len());
        for s in &steps {
            let intents = sts2_content::monsters::forecast::forecast_step_intents(
                state, dealer, &prior, s.move_id, &s.intents,
            );
            let mut d = obj();
            set(&mut d, "step", s.step);
            set(&mut d, "certainty", f64::from(s.certainty));
            set(&mut d, "move_id", s.move_id);
            d.insert(
                "intents".to_owned(),
                Value::Array(
                    intents
                        .iter()
                        .map(|intent| intent_to_value(*intent, state, dealer))
                        .collect(),
                ),
            );
            rows.push(Value::Object(d));
            prior.push(s.move_id);
        }
        out.push(Value::Array(rows));
    }
    Ok(Value::Array(out))
}

/// `CombatState.true_draw_top(k)`: the top `k` cards of the player's draw
/// pile in true draw order, top first, as card dicts in the
/// [`card_to_value`] shape (the whole pile when it holds fewer). The top is
/// the card the next draw takes. Hidden state, for a privileged consumer
/// only; this is the order the wire's `deck.draw_pile` lists, without
/// building the rest of the dict.
///
/// # Errors
///
/// A state whose draw order is not known (`deferred_draw_order`: a
/// `belief_view`, whose unknown middle is drawn at random when a draw
/// happens), or a pile uid with no card. A `sample_world` re-deal has a
/// concrete order and returns it.
pub fn true_draw_top(state: &CombatState, k: usize) -> Result<Value, EmitError> {
    if state.player.deferred_draw_order {
        return Err(EmitError::new(
            "true_draw_top: this state's draw order is not known (deferred_draw_order, \
             as on a belief_view); read it from the real state",
        ));
    }
    let top = &state.player.draw[..k.min(state.player.draw.len())];
    pile_to_value(state, top, &state.cards, "$.true_draw_top")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    fn keys(v: &Value) -> Vec<&str> {
        v.as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// C# `PlayerCombatState.MaxEnergy` is computed on every read
    /// (`Hook.ModifyMaxEnergy` over the base, `PlayerCombatState.cs:107`): a
    /// max-energy power gained during the turn shows at once, where the
    /// engine's stored value is the turn start's. Ported from manoj fork
    /// 81abe212e.
    #[test]
    fn max_energy_is_computed_on_each_read() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        state.player.max_energy = 3; // stored at the turn's start
        let uid = sts2_core::PowerUid(NonZeroU32::new(1).expect("non-zero"));
        state.powers.push(sts2_core::Power {
            id: uid.0,
            kind: sts2_core::PowerKind::Demesne,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(uid);
        let v = combat_state_to_value(&state).expect("serializes");
        assert_eq!(v["player"]["max_energy"], 4);
        assert_eq!(v["player"]["energy_regen_per_turn"], 4);
    }

    #[test]
    fn default_state_emits_the_top_level_wire_shape() {
        let state = CombatState::default();
        let v = combat_state_to_value(&state).expect("default state serializes");
        assert_eq!(
            keys(&v),
            [
                "combat_id",
                "deck",
                "encounter",
                "intent_amounts",
                "is_elite",
                "phase",
                "player",
                "rng",
                "rng_counters",
                "round",
                "side",
                "turn",
                "wire_version",
            ]
        );
        assert_eq!(v["wire_version"], WIRE_VERSION);
        assert_eq!(v["intent_amounts"], "base");
        assert_eq!(v["combat_id"], 0);
        // An empty default state clamps to a parseable shape: `max_hp >= 1`
        // and the `None` character defaults to ironclad.
        assert_eq!(v["player"]["class"], "ironclad");
        assert_eq!(v["player"]["max_hp"], 1);
        assert_eq!(v["player"]["hp"], 0);
        assert_eq!(
            keys(&v["player"]),
            [
                "block",
                "class",
                "energy",
                "energy_regen_per_turn",
                "gold",
                "hp",
                "max_energy",
                "max_hp",
                "potions",
                "powers",
                "relics",
            ]
        );
        assert_eq!(
            keys(&v["deck"]),
            ["discard_pile", "draw_pile", "exhaust_pile", "hand"]
        );
        assert_eq!(v["encounter"], serde_json::json!([]));
        assert_eq!(
            keys(&v["rng"]),
            ["ascension", "counters", "run_seed", "seed"]
        );
        // The diagnostic copy and the load-bearing copy come from one
        // definition.
        assert_eq!(v["rng"]["counters"], v["rng_counters"]);
        // Omitted-when-default keys are OMITTED, not null.
        for absent in [
            "stars",
            "osty",
            "orbs",
            "orb_slots",
            "cards_played_this_turn",
        ] {
            assert!(
                v["player"].get(absent).is_none(),
                "{absent} should be absent"
            );
        }
    }

    #[test]
    fn default_state_round_trips_through_the_parser() {
        let state = CombatState::default();
        let v = combat_state_to_value(&state).expect("serializes");
        let parsed = crate::parse_json(&v).expect("the emitted dict must parse");
        assert_eq!(parsed.player.class, "ironclad");
        assert_eq!(parsed.wire_version, WIRE_VERSION);
    }

    #[test]
    fn default_state_has_no_legal_actions_or_choice_and_empty_previews() {
        let state = CombatState::default();
        // No hand → nothing to preview.
        assert!(card_damage_preview(&state).is_empty());
        assert!(card_block_preview(&state).is_empty());
        assert_eq!(pending_choice(&state), None);
        assert!(legal_actions(&state).is_array());
    }

    #[test]
    fn intent_dict_carries_amount_times_and_effective_amount() {
        let st = CombatState::default();
        let uid = CreatureUid(NonZeroU32::new(1).expect("non-zero"));
        let d = intent_to_value(Intent::Attack { damage: 12 }, &st, uid);
        assert_eq!(d["kind"], "Attack");
        assert_eq!(d["amount"], 12);
        assert!(d.get("times").is_none());
        assert_eq!(d["effective_amount"], 12);
        let m = intent_to_value(Intent::MultiAttack { damage: 5, hits: 3 }, &st, uid);
        assert_eq!(m["amount"], 5);
        assert_eq!(m["times"], 3);
        let s = intent_to_value(Intent::Sleep, &st, uid);
        assert_eq!(s, serde_json::json!({"kind": "Sleep"}));
    }

    #[test]
    fn combat_action_dicts_spell_none_targets_as_null() {
        let v = combat_action_to_value(&Action::PlayCard {
            hand_index: 1,
            target: None,
        });
        assert_eq!(
            v,
            serde_json::json!({"action": "play", "card_index": 1, "target_index": null})
        );
        let v = combat_action_to_value(&Action::ResolveChoice {
            selection: smallvec::SmallVec::from_slice(&[0, 1]),
        });
        assert_eq!(
            v,
            serde_json::json!({"action": "choose", "index": 0, "selection": [0, 1]})
        );
        let v = combat_action_to_value(&Action::Channel {
            orb: Orb::Lightning,
            target: None,
        });
        assert_eq!(v["orb"], "Lightning");
    }

    // --- live damage / block (GH #1197) -------------------------------------

    mod live {
        use super::*;
        use smallvec::SmallVec;
        use sts2_core::cards::CardId;
        use sts2_core::{
            CardUid, Creature, GameVersion, IntentHistory, MonsterId, MonsterKind, Power,
            PowerKind, PowerUid,
        };

        fn push_creature(state: &mut CombatState, hp: i32) -> CreatureUid {
            let raw = u32::try_from(state.creatures.len() + 1).expect("fits");
            let id = NonZeroU32::new(raw).expect("non-zero");
            state.creatures.push(Creature {
                id,
                current_hp: hp,
                max_hp: 50,
                block: 0,
                powers: SmallVec::new(),
                species: Some(MonsterKind::Nibbit),
                intent: None,
                intent_history: IntentHistory::default(),
                monster_id: MonsterId("Nibbit"),
                ai: None,
            });
            CreatureUid(id)
        }

        /// A v0.111.0 board with one living enemy; returns it.
        fn board() -> (CombatState, CreatureUid) {
            sts2_content::cards::dispatch::ensure_installed();
            let mut state = CombatState {
                game_version: GameVersion::V0_111_0,
                ..CombatState::default()
            };
            state.player.current_hp = 80;
            state.player.max_hp = 80;
            let p = push_creature(&mut state, 80);
            state.player.creature_uid = Some(p);
            let e = push_creature(&mut state, 50);
            state.enemy_uids.push(e);
            (state, e)
        }

        fn hand(state: &mut CombatState, id: CardId) -> CardUid {
            let mut card =
                sts2_content::cards::v0111::card_pod_for(id, false, GameVersion::V0_111_0)
                    .expect("spec exists");
            let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits"))
                .expect("non-zero");
            card.uid = uid;
            state.cards.push(card);
            state.player.hand.push(uid);
            uid
        }

        fn give_power(state: &mut CombatState, who: CreatureUid, kind: PowerKind, amount: i32) {
            let pid = PowerUid::new(u32::try_from(state.powers.len() + 1).expect("fits"))
                .expect("non-zero");
            state.powers.push(Power {
                id: pid.0,
                kind,
                amount,
                counter: 0,
                applier: None,
                skip_next_tick: false,
            });
            let c = state
                .creatures
                .iter_mut()
                .find(|c| c.id == who.0)
                .expect("creature");
            c.powers.push(pid);
        }

        fn player(state: &CombatState) -> CreatureUid {
            state.player.creature_uid.expect("player creature")
        }

        fn summon_osty(state: &mut CombatState, hp: i32) {
            let osty = push_creature(state, hp);
            state.player.pet_uid = Some(osty);
        }

        #[test]
        fn damage_live_osty_strength_not_player() {
            let (mut s, _) = board();
            let p = player(&s);
            give_power(&mut s, p, PowerKind::Strength, 3);
            summon_osty(&mut s, 10);
            hand(&mut s, CardId::Poke);
            // Poke is an Osty attack: the player's Strength does not apply.
            assert_eq!(card_damage_live(&s), vec![BTreeMap::from([(0, (6, 1.0))])]);
        }

        #[test]
        fn damage_live_calcify_applies() {
            let (mut s, _) = board();
            let p = player(&s);
            give_power(&mut s, p, PowerKind::Calcify, 4);
            summon_osty(&mut s, 10);
            hand(&mut s, CardId::Poke);
            assert_eq!(card_damage_live(&s), vec![BTreeMap::from([(0, (10, 1.0))])]);
        }

        /// The live values target what C# `Creature.IsHittable` admits
        /// (Creature.cs:286-298): a reviving Illusion corpse (0 HP, still in
        /// the encounter) is not hittable, nor is a living creature outside
        /// the encounter (the Osty).
        #[test]
        fn live_values_skip_unhittable_enemies() {
            use sts2_core::orbs::Orb;
            let (mut s, e0) = board();
            let corpse = second_enemy(&mut s, 0);
            give_power(&mut s, corpse, PowerKind::Illusion, 1);
            summon_osty(&mut s, 10);
            let osty = s.player.pet_uid.expect("osty");
            assert!(s.is_hittable_enemy(e0));
            assert!(!s.is_hittable_enemy(corpse));
            assert!(!s.is_hittable_enemy(osty));
            hand(&mut s, CardId::StrikeIronclad);
            hand(&mut s, CardId::Zap);
            assert_eq!(card_damage_live(&s)[0], BTreeMap::from([(0, (6, 1.0))]));
            // Zap into a full Lightning belt: the evoke's random target is the
            // one hittable enemy, so it takes the whole 8.
            belt(&mut s, 1, &[Orb::Lightning]);
            assert_eq!(
                card_gen_live(&s)[1].clone().expect("zap").flat,
                BTreeMap::from([(0, 8.0)])
            );
        }

        #[test]
        fn damage_live_osty_dead_empty() {
            let (mut s, _) = board();
            summon_osty(&mut s, 0);
            hand(&mut s, CardId::Poke);
            assert_eq!(card_damage_live(&s), vec![BTreeMap::new()]);
        }

        #[test]
        fn damage_live_body_slam_nonzero() {
            let (mut s, _) = board();
            s.player.block = 12;
            hand(&mut s, CardId::BodySlam);
            // The stored `base_damage` is 0, so the old preview is empty.
            assert_eq!(card_damage_preview(&s), vec![BTreeMap::new()]);
            assert_eq!(card_damage_live(&s), vec![BTreeMap::from([(0, (12, 1.0))])]);
        }

        #[test]
        fn damage_live_perfected_strike_vulnerable() {
            let (mut s, e) = board();
            hand(&mut s, CardId::PerfectedStrike);
            for _ in 0..4 {
                hand(&mut s, CardId::StrikeIronclad);
            }
            give_power(&mut s, e, PowerKind::Vulnerable, 2);
            // 6 + 2 x 5 Strike-tagged = 16, x1.5 Vulnerable = 24.
            assert_eq!(card_damage_live(&s)[0], BTreeMap::from([(0, (24, 1.0))]));
        }

        #[test]
        fn block_live_parry_blade() {
            let (mut s, _) = board();
            let p = player(&s);
            give_power(&mut s, p, PowerKind::Parry, 6);
            give_power(&mut s, p, PowerKind::Dexterity, 2);
            hand(&mut s, CardId::SovereignBlade);
            assert_eq!(card_block_preview(&s), vec![None]);
            assert_eq!(card_block_live(&s), vec![Some(8)]);
        }

        #[test]
        fn block_live_calculated_block_card() {
            let (mut s, _) = board();
            hand(&mut s, CardId::Stack);
            hand(&mut s, CardId::DefendIronclad);
            let live = card_block_live(&s);
            // Stack's block is its discard pile size: empty, so none.
            assert_eq!(live[0], None);
            // A plain Defend is the old preview's answer.
            assert_eq!(live[1], card_block_preview(&s)[1]);
            assert_eq!(live[1], Some(5));
            // Fill the discard pile: Stack then grants one Block per card.
            hand(&mut s, CardId::StrikeIronclad);
            let strike = s.player.hand.pop().expect("strike");
            s.player.discard.push(strike);
            assert_eq!(card_block_live(&s)[0], Some(1));
        }

        #[test]
        fn preview_unchanged_golden() {
            let (mut s, e) = board();
            let p = player(&s);
            give_power(&mut s, p, PowerKind::Strength, 3);
            give_power(&mut s, e, PowerKind::Vulnerable, 1);
            summon_osty(&mut s, 10);
            hand(&mut s, CardId::StrikeIronclad);
            hand(&mut s, CardId::Poke);
            hand(&mut s, CardId::DefendIronclad);
            hand(&mut s, CardId::SovereignBlade);
            give_power(&mut s, p, PowerKind::Parry, 6);
            // Strike (6 + 3) x 1.5 = 13. The preview also scores Poke with the
            // player's Strength (6 + 3) x 1.5 = 13 and Sovereign Blade
            // (10 + 3) x 1.5 = 19: values `card_damage_live` corrects.
            assert_eq!(
                card_damage_preview(&s),
                vec![
                    BTreeMap::from([(0, 13)]),
                    BTreeMap::from([(0, 13)]),
                    BTreeMap::new(),
                    BTreeMap::from([(0, 19)]),
                ]
            );
            assert_eq!(card_block_preview(&s), vec![None, None, Some(5), None]);
        }

        // --- generated damage (`card_gen_live`) ---------------------------

        fn second_enemy(state: &mut CombatState, hp: i32) -> CreatureUid {
            let e = push_creature(state, hp);
            state.enemy_uids.push(e);
            e
        }

        fn belt(state: &mut CombatState, cap: usize, orbs: &[sts2_core::orbs::Orb]) {
            let mut b = sts2_core::OrbSlots::with_capacity(cap);
            for o in orbs {
                b.channel(*o);
            }
            state.player.orbs = b;
        }

        fn upgraded(state: &mut CombatState, uid: CardUid) {
            let c = state.cards.iter_mut().find(|c| c.uid == uid).expect("card");
            c.upgraded = true;
        }

        #[test]
        fn gen_live_plain_cards_have_none() {
            let (mut s, _) = board();
            hand(&mut s, CardId::StrikeIronclad);
            hand(&mut s, CardId::DefendIronclad);
            assert_eq!(card_gen_live(&s), vec![None, None]);
        }

        /// Blade Dance: three Shivs, each 4 + Strength 2 + Accuracy 3 = 9,
        /// on every enemy (each Shiv picks its own target).
        #[test]
        fn gen_live_blade_dance_shivs() {
            let (mut s, _) = board();
            second_enemy(&mut s, 30);
            let p = player(&s);
            give_power(&mut s, p, PowerKind::Strength, 2);
            give_power(&mut s, p, PowerKind::Accuracy, 3);
            hand(&mut s, CardId::BladeDance);
            let g = card_gen_live(&s)[0].clone().expect("generates");
            assert_eq!(g.flat, BTreeMap::from([(0, 27.0), (1, 27.0)]));
            assert!(g.per_energy.is_empty() && g.kills.is_empty());
            // Upgraded: four Shivs. Cloak and Dagger: one.
            let bd = s.player.hand[0];
            upgraded(&mut s, bd);
            hand(&mut s, CardId::CloakAndDagger);
            let gl = card_gen_live(&s);
            assert_eq!(gl[0].clone().expect("bd").flat[&0], 36.0);
            assert_eq!(gl[1].clone().expect("cad").flat[&0], 9.0);
        }

        /// Shivs past the hand limit go to the discard pile, unplayable now.
        #[test]
        fn gen_live_shivs_capped_by_hand_limit() {
            let (mut s, _) = board();
            hand(&mut s, CardId::BladeDance);
            for _ in 0..8 {
                hand(&mut s, CardId::DefendSilent);
            }
            // Nine cards; Blade Dance leaves eight, so two Shivs fit.
            assert_eq!(card_gen_live(&s)[0].clone().expect("gen").flat[&0], 8.0);
        }

        /// Knife Trap replays every Shiv in the exhaust pile at its target;
        /// upgraded, it upgrades them first (6 damage).
        #[test]
        fn gen_live_knife_trap() {
            let (mut s, _) = board();
            hand(&mut s, CardId::KnifeTrap);
            assert_eq!(card_gen_live(&s), vec![None]);
            for _ in 0..2 {
                let u = hand(&mut s, CardId::Shiv);
                s.player.hand.retain(|x| *x != u);
                s.player.exhaust.push(u);
            }
            assert_eq!(card_gen_live(&s)[0].clone().expect("gen").flat[&0], 8.0);
            let kt = s.player.hand[0];
            upgraded(&mut s, kt);
            assert_eq!(card_gen_live(&s)[0].clone().expect("gen").flat[&0], 12.0);
        }

        /// Dualcast evokes the front orb twice; Quadcast four times; Multi-Cast
        /// X times (+1 upgraded). A Lightning evoke hits a random enemy (its
        /// share), a Dark one the weakest, Frost none.
        #[test]
        fn gen_live_evoke_cards() {
            use sts2_core::orbs::Orb;
            let (mut s, _) = board();
            second_enemy(&mut s, 30);
            let p = player(&s);
            give_power(&mut s, p, PowerKind::Focus, 1);
            // Unpowered: Strength does not touch an evoke.
            give_power(&mut s, p, PowerKind::Strength, 5);
            belt(&mut s, 3, &[Orb::Lightning, Orb::Dark { stored: 4 }]);
            hand(&mut s, CardId::Dualcast);
            hand(&mut s, CardId::Quadcast);
            let mc = hand(&mut s, CardId::MultiCast);
            let g = card_gen_live(&s);
            // Lightning 8 + 1 Focus = 9, split over two enemies.
            assert_eq!(
                g[0].clone().expect("dc").flat,
                BTreeMap::from([(0, 9.0), (1, 9.0)])
            );
            assert_eq!(
                g[1].clone().expect("qc").flat,
                BTreeMap::from([(0, 18.0), (1, 18.0)])
            );
            let m = g[2].clone().expect("mc");
            assert!(m.flat.is_empty());
            assert_eq!(m.per_energy, BTreeMap::from([(0, 4.5), (1, 4.5)]));
            upgraded(&mut s, mc);
            assert_eq!(card_gen_live(&s)[2].clone().expect("mc").flat[&1], 4.5);
            // Dark in front: 6 + 4 stored, all on the weakest (enemy 1, 30 HP).
            belt(&mut s, 3, &[Orb::Dark { stored: 4 }, Orb::Lightning]);
            assert_eq!(
                card_gen_live(&s)[0].clone().expect("dc").flat,
                BTreeMap::from([(1, 20.0)])
            );
            // Frost in front: no damage.
            belt(&mut s, 3, &[Orb::Frost, Orb::Lightning]);
            assert_eq!(card_gen_live(&s)[0], None);
            // No orbs.
            belt(&mut s, 3, &[]);
            assert_eq!(card_gen_live(&s)[0], None);
        }

        /// A channel into a full belt evokes the front orb first.
        #[test]
        fn gen_live_overflow_evoke() {
            use sts2_core::orbs::Orb;
            let (mut s, _) = board();
            belt(&mut s, 2, &[Orb::Lightning]);
            hand(&mut s, CardId::Zap);
            hand(&mut s, CardId::Glacier);
            // One free slot: Zap evokes nothing, Glacier's second Frost evokes
            // the Lightning.
            let g = card_gen_live(&s);
            assert_eq!(g[0], None);
            assert_eq!(
                g[1].clone().expect("glacier").flat,
                BTreeMap::from([(0, 8.0)])
            );
            // Full belt: Zap evokes the front Lightning, Glacier both orbs.
            belt(
                &mut s,
                2,
                &[Orb::Lightning, Orb::Glass { current_passive: 3 }],
            );
            let g = card_gen_live(&s);
            assert_eq!(g[0].clone().expect("zap").flat, BTreeMap::from([(0, 8.0)]));
            assert_eq!(
                g[1].clone().expect("glacier").flat,
                BTreeMap::from([(0, 14.0)])
            );
        }

        /// Channels past a full belt evoke the orbs the play itself channelled
        /// once the belt's own are gone (`OrbCmd.Channel` evokes the front, then
        /// enqueues at the back), and a non-Defect's empty zero-slot belt grows
        /// one slot first (OrbCmd.cs:74-77).
        #[test]
        fn gen_live_overflow_evokes_the_plays_own_orbs() {
            use sts2_core::orbs::Orb;
            let (mut s, _) = board();
            // Rainbow on a 0-slot non-Defect belt: Lightning gets the new slot,
            // Frost evokes it (8), Dark evokes the Frost (0).
            belt(&mut s, 0, &[]);
            hand(&mut s, CardId::Rainbow);
            assert_eq!(
                card_gen_live(&s)[0].clone().expect("rainbow").flat,
                BTreeMap::from([(0, 8.0)])
            );
            // Ice Lance (3 Frost) on a full 1-slot Lightning belt: only the
            // Lightning hits; the evoked Frosts deal nothing.
            belt(&mut s, 1, &[Orb::Lightning]);
            s.player.hand.clear();
            hand(&mut s, CardId::IceLance);
            assert_eq!(
                card_gen_live(&s)[0].clone().expect("ice lance").flat,
                BTreeMap::from([(0, 8.0)])
            );
        }

        /// Voltaic channels one Lightning per Lightning channelled earlier this
        /// combat (Voltaic.cs:30-34, :47-51, read before it channels).
        #[test]
        fn gen_live_voltaic_overflow() {
            use sts2_core::orbs::Orb;
            let (mut s, _) = board();
            belt(&mut s, 2, &[Orb::Glass { current_passive: 3 }]);
            hand(&mut s, CardId::Voltaic);
            // None channelled yet: Voltaic channels nothing.
            assert_eq!(card_gen_live(&s)[0], None);
            // 3: the first fills the free slot, the second evokes the Glass
            // (3 x 2), the third the first new Lightning (8).
            s.combat_log.lightning_orbs_channeled_this_combat = 3;
            assert_eq!(
                card_gen_live(&s)[0].clone().expect("voltaic").flat,
                BTreeMap::from([(0, 14.0)])
            );
        }

        /// Tempest channels X Lightning, +1 upgraded (Tempest.cs:29-37): not
        /// linear in X, so the play's damage is a table by the energy spent.
        #[test]
        fn gen_live_tempest_by_x() {
            use sts2_core::orbs::Orb;
            let (mut s, _) = board();
            s.player.energy = 3;
            belt(&mut s, 2, &[Orb::Lightning]);
            let t = hand(&mut s, CardId::Tempest);
            let g = card_gen_live(&s)[0].clone().expect("tempest");
            assert!(g.flat.is_empty() && g.per_energy.is_empty());
            let row = |v: f64| {
                if v > 0.0 {
                    BTreeMap::from([(0, v)])
                } else {
                    BTreeMap::new()
                }
            };
            // X=0, 1: into the free slot; X=2 evokes the belt's Lightning, X=3
            // the first one Tempest channelled.
            assert_eq!(g.by_x, vec![row(0.0), row(0.0), row(8.0), row(16.0)]);
            upgraded(&mut s, t);
            let g = card_gen_live(&s)[0].clone().expect("tempest+");
            assert_eq!(g.by_x, vec![row(0.0), row(8.0), row(16.0), row(24.0)]);
            // Free slots for every channel: nothing.
            belt(&mut s, 5, &[]);
            assert_eq!(card_gen_live(&s)[0], None);
            // Other cards carry no table.
            belt(&mut s, 1, &[Orb::Lightning]);
            hand(&mut s, CardId::Zap);
            assert!(card_gen_live(&s)[1].clone().expect("zap").by_x.is_empty());
        }

        /// Tempest's X is `ResolveEnergyXValue()`, folded through
        /// `Hook.ModifyXValue`: Chemical X channels 2 more Lightning at every
        /// energy spent (ChemicalX.cs:31-38 `ModifyXValue`: `+ Increase`, 2).
        #[test]
        fn gen_live_tempest_by_x_chemical_x() {
            use sts2_core::orbs::Orb;
            sts2_content::relics::dispatch::ensure_installed();
            sts2_core::relic_effects::ensure_installed();
            let (mut s, _) = board();
            s.player.energy = 2;
            belt(&mut s, 2, &[Orb::Lightning]);
            hand(&mut s, CardId::Tempest);
            s.relics.push(sts2_core::Relic {
                id: std::num::NonZeroU32::new(1).expect("non-zero"),
                kind: sts2_core::RelicKind::ChemicalX,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
            let g = card_gen_live(&s)[0].clone().expect("tempest");
            // X = k + 2: 2 channels fill the free slot then evoke the belt's
            // Lightning (8); each further one evokes a Lightning Tempest
            // channelled (8 more).
            let row = |v: f64| BTreeMap::from([(0, v)]);
            assert_eq!(g.by_x, vec![row(8.0), row(16.0), row(24.0)]);
        }

        fn chemical_x(s: &mut CombatState) {
            sts2_content::relics::dispatch::ensure_installed();
            sts2_core::relic_effects::ensure_installed();
            let id = u32::try_from(s.relics.len() + 1).expect("fits");
            s.relics.push(sts2_core::Relic {
                id: std::num::NonZeroU32::new(id).expect("non-zero"),
                kind: sts2_core::RelicKind::ChemicalX,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
        }

        /// `(damage per hit, hits)` of hand card `i` on enemy 0.
        fn dmg0(s: &CombatState, i: usize) -> (i32, f32) {
            card_damage_live(s)[i][&0]
        }

        /// Energy-X attacks hit `ResolveEnergyXValue()` times
        /// (Whirlwind.cs:33/42, Skewer.cs:28, Eradicate.cs:28, Volley.cs:24),
        /// folded through `Hook.ModifyXValue`: Chemical X adds 2 hits at every
        /// energy spent, so the play's damage is a table by the energy spent
        /// (`by_x`). Without an X modifier the knapsack's linear term is exact
        /// and no table is emitted.
        #[test]
        fn gen_live_x_attacks_chemical_x() {
            for id in [
                CardId::Whirlwind,
                CardId::Skewer,
                CardId::Eradicate,
                CardId::Volley,
            ] {
                let (mut s, _) = board();
                s.player.energy = 2;
                hand(&mut s, id);
                assert_eq!(card_gen_live(&s)[0], None, "{id:?} without Chemical X");
                chemical_x(&mut s);
                let (per_hit, hits) = dmg0(&s, 0);
                let one = f64::from(per_hit) * f64::from(hits);
                let gen = card_gen_live(&s)[0].clone().expect("x attack");
                let want: Vec<_> = (0..=2)
                    .map(|k| BTreeMap::from([(0, one * f64::from(k + 2))]))
                    .collect();
                assert_eq!(gen.by_x, want, "{id:?}");
                assert!(gen.flat.is_empty() && gen.per_energy.is_empty());
            }
        }

        /// Heavenly Drill: `num = ResolveEnergyXValue(); if (num >= 4) num *= 2`
        /// (HeavenlyDrill.cs:32-37). The doubling is judged at the X the play
        /// gets, not at the energy available now: a table whenever the energy
        /// available crosses the threshold (or an X modifier shifts it).
        #[test]
        fn gen_live_heavenly_drill_by_x() {
            let (mut s, _) = board();
            s.player.energy = 3;
            hand(&mut s, CardId::HeavenlyDrill);
            assert_eq!(card_gen_live(&s)[0], None, "below the threshold: linear");
            s.player.energy = 5;
            let (d, h) = dmg0(&s, 0);
            assert!((h - 2.0).abs() < f32::EPSILON, "doubled per energy at 5");
            let d = f64::from(d);
            let hits = [0.0, 1.0, 2.0, 3.0, 8.0, 10.0];
            let g = card_gen_live(&s)[0].clone().expect("drill");
            let want: Vec<BTreeMap<usize, f64>> = hits
                .iter()
                .map(|n| {
                    if *n > 0.0 {
                        BTreeMap::from([(0, d * n)])
                    } else {
                        BTreeMap::new()
                    }
                })
                .collect();
            assert_eq!(g.by_x, want);
            // Chemical X at 2 energy: X = 2, 3, 4 -> 2, 3, 8 hits.
            s.player.energy = 2;
            chemical_x(&mut s);
            let g = card_gen_live(&s)[0].clone().expect("drill");
            let want: Vec<_> = [2.0, 3.0, 8.0]
                .iter()
                .map(|n| BTreeMap::from([(0, d * n)]))
                .collect();
            assert_eq!(g.by_x, want);
        }

        /// Multi-Cast evokes `ResolveEnergyXValue()` times, +1 upgraded
        /// (MultiCast.cs:22-29): with Chemical X the evokes are a table by the
        /// energy spent, which carries the upgrade's evoke too.
        #[test]
        fn gen_live_multi_cast_chemical_x() {
            use sts2_core::orbs::Orb;
            let (mut s, _) = board();
            s.player.energy = 1;
            belt(&mut s, 3, &[Orb::Lightning]);
            let mc = hand(&mut s, CardId::MultiCast);
            chemical_x(&mut s);
            let row = |v: f64| BTreeMap::from([(0, v)]);
            let g = card_gen_live(&s)[0].clone().expect("mc");
            assert_eq!(g.by_x, vec![row(16.0), row(24.0)]);
            assert!(g.flat.is_empty() && g.per_energy.is_empty());
            upgraded(&mut s, mc);
            let g = card_gen_live(&s)[0].clone().expect("mc+");
            assert_eq!(g.by_x, vec![row(24.0), row(32.0)]);
            assert!(g.flat.is_empty() && g.per_energy.is_empty());
        }

        /// Stardust hits `ResolveStarXValue()` times (Stardust.cs:24): with
        /// Chemical X, a table by the stars spent.
        #[test]
        fn gen_live_stardust_chemical_x() {
            let (mut s, _) = board();
            s.player.stars = 2;
            hand(&mut s, CardId::Stardust);
            assert_eq!(card_gen_live(&s)[0], None, "without Chemical X");
            chemical_x(&mut s);
            let (d, h) = dmg0(&s, 0);
            // live_attack already reads X = 2 + 2 hits at all the stars.
            assert!((h - 4.0).abs() < f32::EPSILON);
            let d = f64::from(d);
            let g = card_gen_live(&s)[0].clone().expect("stardust");
            let want: Vec<_> = [2.0, 3.0, 4.0]
                .iter()
                .map(|n| BTreeMap::from([(0, d * n)]))
                .collect();
            assert_eq!(g.by_x, want);
        }

        /// Ignition channels its Plasma on the target teammate
        /// (Ignition.cs:33): a full teammate belt evokes that seat's front orb.
        /// Solo, there is no teammate (and the card is unplayable).
        #[test]
        fn gen_live_ignition_evokes_the_teammates_front_orb() {
            use sts2_core::orbs::Orb;
            let (mut s, _) = board();
            belt(&mut s, 1, &[Orb::Lightning]);
            hand(&mut s, CardId::Ignition);
            // Solo: the player's own full belt is not touched.
            assert_eq!(card_gen_live(&s)[0], None);
            let mut ally_belt = sts2_core::OrbSlots::with_capacity(1);
            ally_belt.channel(Orb::Dark { stored: 4 });
            s.allies.push(sts2_core::PlayerCombatState {
                current_hp: 70,
                max_hp: 70,
                orbs: ally_belt,
                ..sts2_core::PlayerCombatState::default()
            });
            // The teammate's Dark (6 + 4) on the weakest enemy.
            assert_eq!(
                card_gen_live(&s)[0].clone().expect("ignition").flat,
                BTreeMap::from([(0, 10.0)])
            );
            // A dead teammate is no target.
            s.allies[0].current_hp = 0;
            assert_eq!(card_gen_live(&s)[0], None);
        }

        /// End of Days' Doom kills where Doom reaches HP; Artifact blocks it.
        #[test]
        fn gen_live_end_of_days_doom_kill() {
            let (mut s, e0) = board();
            let e1 = second_enemy(&mut s, 29);
            let e2 = second_enemy(&mut s, 10);
            give_power(&mut s, e0, PowerKind::Doom, 21);
            give_power(&mut s, e2, PowerKind::Artifact, 1);
            let eod = hand(&mut s, CardId::EndOfDays);
            // e0: 50 HP, 21 + 29 = 50 Doom; e1: 29; e2: blocked.
            let g = card_gen_live(&s)[0].clone().expect("eod");
            assert_eq!(g.kills, vec![0, 1]);
            assert!(g.flat.is_empty());
            s.creatures
                .iter_mut()
                .find(|c| c.id == e1.0)
                .expect("e1")
                .current_hp = 30;
            assert_eq!(card_gen_live(&s)[0].clone().expect("eod").kills, vec![0]);
            // Upgraded: 37 Doom.
            upgraded(&mut s, eod);
            assert_eq!(card_gen_live(&s)[0].clone().expect("eod").kills, vec![0, 1]);
        }

        // --- end of turn (`end_turn_live`) ---------------------------------

        mod end_turn {
            use super::*;
            use sts2_core::orbs::Orb;
            use sts2_core::{Relic, RelicKind};

            fn end_board() -> (CombatState, CreatureUid) {
                sts2_content::relics::dispatch::ensure_installed();
                sts2_core::relic_effects::ensure_installed();
                board()
            }

            fn relic(state: &mut CombatState, kind: RelicKind, counter: i32) {
                let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).expect("fits"))
                    .expect("non-zero");
                state.relics.push(Relic {
                    id,
                    kind,
                    counter,
                    is_wax: false,
                    is_melted: false,
                });
            }

            fn set_enemy(state: &mut CombatState, e: CreatureUid, hp: i32, block: i32) {
                let c = state
                    .creatures
                    .iter_mut()
                    .find(|c| c.id == e.0)
                    .expect("enemy");
                c.current_hp = hp;
                c.block = block;
            }

            /// `PoisonPower.CalculateTotalDamageNextTurn`: Unblockable, so the
            /// enemy's block does not save it.
            #[test]
            fn end_live_poison_lethal() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 20, 10);
                give_power(&mut s, e, PowerKind::Poison, 20);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                set_enemy(&mut s, e, 21, 0);
                assert_eq!(end_turn_live(&s).dies, Vec::<usize>::new());
            }

            /// Accelerant: `min(Amount, 1 + Accelerant)` ticks of `Amount - i`.
            #[test]
            fn end_live_poison_accelerant() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 9, 0);
                give_power(&mut s, e, PowerKind::Poison, 5);
                assert!(end_turn_live(&s).dies.is_empty());
                let p = player(&s);
                give_power(&mut s, p, PowerKind::Accelerant, 1);
                // 5 + 4.
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                set_enemy(&mut s, e, 10, 0);
                assert!(end_turn_live(&s).dies.is_empty());
            }

            /// Glass hits every hittable enemy through its block; Gold-Plated
            /// Cables triggers the front orb again, 1 lower (`_passiveVal - 1`).
            #[test]
            fn end_live_glass_kills_all() {
                let (mut s, e0) = end_board();
                let e1 = second_enemy(&mut s, 5);
                set_enemy(&mut s, e0, 4, 0);
                set_enemy(&mut s, e1, 5, 2);
                belt(&mut s, 3, &[Orb::Glass { current_passive: 4 }]);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                relic(&mut s, RelicKind::GoldPlatedCables, 0);
                // 4 + 3 = 7 = 5 HP + 2 block.
                assert_eq!(end_turn_live(&s).dies, vec![0, 1]);
                // An Intangible enemy loses 1 per hit.
                give_power(&mut s, e0, PowerKind::Intangible, 1);
                assert_eq!(end_turn_live(&s).dies, vec![1]);
                // The state itself is untouched.
                assert_eq!(s.player.orbs.slots[0], Orb::Glass { current_passive: 4 });
            }

            /// Lightning's passive picks a random hittable enemy: certain only
            /// when there is one.
            #[test]
            fn end_live_lightning_single_enemy() {
                let (mut s, e0) = end_board();
                set_enemy(&mut s, e0, 3, 0);
                belt(&mut s, 3, &[Orb::Lightning]);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                let e1 = second_enemy(&mut s, 3);
                set_enemy(&mut s, e1, 3, 0);
                assert!(end_turn_live(&s).dies.is_empty());
                // A dead enemy is not hittable.
                set_enemy(&mut s, e1, 0, 0);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
            }

            /// Frost passives before the enemies act; Gold-Plated Cables gives
            /// the front orb one more trigger, whatever its type.
            #[test]
            fn end_live_gold_plated_cables_double_frost() {
                let (mut s, _) = end_board();
                let p = player(&s);
                give_power(&mut s, p, PowerKind::Focus, 1);
                belt(&mut s, 3, &[Orb::Frost, Orb::Frost]);
                assert_eq!(end_turn_live(&s).block, 6);
                relic(&mut s, RelicKind::GoldPlatedCables, 0);
                assert_eq!(end_turn_live(&s).block, 9);
                belt(&mut s, 3, &[Orb::Lightning, Orb::Frost]);
                assert_eq!(end_turn_live(&s).block, 3);
            }

            /// Plating (Early), Orichalcum / Fake Orichalcum (latched on 0
            /// block VeryEarly), Cloak Clasp (1 per hand card), Ripple Basin
            /// (no Attack this turn). Unpowered: Dexterity does not apply.
            #[test]
            fn end_live_plating_orichalcum_cloak_clasp() {
                let (mut s, _) = end_board();
                let p = player(&s);
                give_power(&mut s, p, PowerKind::Plating, 4);
                give_power(&mut s, p, PowerKind::Dexterity, 3);
                relic(&mut s, RelicKind::Orichalcum, 0);
                relic(&mut s, RelicKind::FakeOrichalcum, 0);
                relic(&mut s, RelicKind::CloakClasp, 0);
                relic(&mut s, RelicKind::RippleBasin, 0);
                for _ in 0..3 {
                    hand(&mut s, CardId::StrikeIronclad);
                }
                assert_eq!(end_turn_live(&s).block, 4 + 6 + 3 + 3 + 4);
                s.player.block = 5;
                assert_eq!(end_turn_live(&s).block, 4 + 3 + 4);
                // Ripple Basin's counter: Attacks played this turn.
                let rb = s
                    .relics
                    .iter_mut()
                    .find(|r| r.kind == RelicKind::RippleBasin)
                    .expect("rb");
                rb.counter = 1;
                assert_eq!(end_turn_live(&s).block, 4 + 3);
            }

            /// Consuming Shadow evokes the last orb `Amount` times after the
            /// passives: the Frost evoke is block, the Glass evoke (decayed by
            /// its passive) damage.
            #[test]
            fn end_live_consuming_shadow() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 4, 0);
                let p = player(&s);
                give_power(&mut s, p, PowerKind::ConsumingShadow, 2);
                belt(&mut s, 3, &[Orb::Glass { current_passive: 2 }, Orb::Frost]);
                // Glass passive 2, Frost passive 2; evokes: Frost 5, Glass 1 x 2.
                let r = end_turn_live(&s);
                assert_eq!(r.block, 7);
                assert_eq!(r.dies, vec![0]);
                set_enemy(&mut s, e, 5, 0);
                assert!(end_turn_live(&s).dies.is_empty());
            }

            /// Consuming Shadow's Dark evoke hits the weakest enemy, unless a
            /// random Lightning zap was skipped before it.
            #[test]
            fn end_live_consuming_shadow_dark_after_random_zap() {
                let (mut s, e0) = end_board();
                set_enemy(&mut s, e0, 15, 0);
                let p = player(&s);
                give_power(&mut s, p, PowerKind::ConsumingShadow, 1);
                // Dark passive: 0 + 6 stored, evoke 6 + 6 = 12.
                belt(&mut s, 3, &[Orb::Lightning, Orb::Dark { stored: 0 }]);
                // One enemy: Lightning 3, then the Dark 12 on it.
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                // Two: the zap is random, so the weakest is unknown.
                set_enemy(&mut s, e0, 12, 0);
                second_enemy(&mut s, 50);
                assert!(end_turn_live(&s).dies.is_empty());
                // No zap before it: the Dark evoke finds the weakest.
                belt(&mut s, 3, &[Orb::Dark { stored: 0 }]);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
            }

            /// The telegraph already carries Intangible's `ModifyDamageCap`:
            /// `hp_loss_live` lets block and the Osty see the capped hit.
            #[test]
            fn end_live_intangible_caps_the_telegraph() {
                let (mut s, e) = end_board();
                let p = player(&s);
                give_power(&mut s, p, PowerKind::Intangible, 1);
                let v = intent_to_value(Intent::Attack { damage: 10 }, &s, e);
                assert_eq!(v["effective_amount"], serde_json::json!(1));
            }

            /// Hailstorm needs a Frost orb when the turn ends.
            #[test]
            fn end_live_hailstorm() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 6, 0);
                let p = player(&s);
                give_power(&mut s, p, PowerKind::Hailstorm, 6);
                assert!(end_turn_live(&s).dies.is_empty());
                belt(&mut s, 3, &[Orb::Frost]);
                // Hailstorm (`BeforeSideTurnEnd`) wins the fight before the
                // Frost passive (`DoTurnEnd`) runs.
                assert_eq!(
                    end_turn_live(&s),
                    EndTurnLive {
                        block: 0,
                        dies: vec![0],
                        ..EndTurnLive::default()
                    }
                );
                second_enemy(&mut s, 50);
                assert_eq!(
                    end_turn_live(&s),
                    EndTurnLive {
                        block: 2,
                        dies: vec![0],
                        ..EndTurnLive::default()
                    }
                );
            }

            /// Glass and poison add up: 3 through the block, then 4 poison.
            #[test]
            fn end_live_damage_then_poison() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 7, 2);
                give_power(&mut s, e, PowerKind::Poison, 4);
                belt(&mut s, 3, &[Orb::Glass { current_passive: 5 }]);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                set_enemy(&mut s, e, 8, 2);
                assert!(end_turn_live(&s).dies.is_empty());
            }

            /// A power with its `counter` set (The Bomb's damage, Hardened
            /// Shell's damage taken this turn).
            fn counted(
                s: &mut CombatState,
                who: CreatureUid,
                kind: PowerKind,
                amount: i32,
                n: i32,
            ) {
                give_power(s, who, kind, amount);
                let id = NonZeroU32::new(u32::try_from(s.powers.len()).expect("fits"))
                    .expect("non-zero");
                s.powers.get_mut_by_id(id).expect("power").counter = n;
            }

            fn to_exhaust(s: &mut CombatState, id: CardId) {
                let uid = hand(s, id);
                s.player.hand.retain(|u| *u != uid);
                s.player.exhaust.push(uid);
            }

            /// `TheBombPower.BeforeSideTurnEnd`: at `Amount == 1` its Damage
            /// (Unpowered) hits every hittable enemy; above 1 it only ticks.
            #[test]
            fn end_live_the_bomb() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 40, 0);
                let p = player(&s);
                counted(&mut s, p, PowerKind::TheBomb, 2, 40);
                assert!(end_turn_live(&s).dies.is_empty());
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 40, 0);
                counted(&mut s, p, PowerKind::TheBomb, 1, 40);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                set_enemy(&mut s, e, 41, 0);
                assert!(end_turn_live(&s).dies.is_empty());
            }

            /// Screaming Flagon (20 to all on an empty hand) and Stone Calendar
            /// (52 to all on turn 7), both `BeforeSideTurnEnd`.
            #[test]
            fn end_live_flagon_and_calendar() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 20, 0);
                relic(&mut s, RelicKind::ScreamingFlagon, 0);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                hand(&mut s, CardId::StrikeIronclad);
                assert!(end_turn_live(&s).dies.is_empty());
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 52, 0);
                relic(&mut s, RelicKind::StoneCalendar, 0);
                s.round = 6;
                assert!(end_turn_live(&s).dies.is_empty());
                s.round = 7;
                assert_eq!(end_turn_live(&s).dies, vec![0]);
            }

            /// `HowlFromBeyond.AfterAutoPostPlayPhaseEntered`: a Howl in the
            /// exhaust pile auto-plays (18 to every enemy) as the turn ends.
            #[test]
            fn end_live_howl_from_exhaust() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 18, 0);
                to_exhaust(&mut s, CardId::HowlFromBeyond);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                set_enemy(&mut s, e, 19, 0);
                assert!(end_turn_live(&s).dies.is_empty());
            }

            /// `ParryingShield.AfterSideTurnEnd`: 6 to a random enemy at 10+
            /// block, certain only with one hittable enemy.
            #[test]
            fn end_live_parrying_shield() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 6, 0);
                relic(&mut s, RelicKind::ParryingShield, 0);
                s.player.block = 9;
                assert!(end_turn_live(&s).dies.is_empty());
                s.player.block = 10;
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                let e1 = second_enemy(&mut s, 6);
                set_enemy(&mut s, e1, 6, 0);
                assert!(end_turn_live(&s).dies.is_empty());
            }

            /// Turn-end-in-hand cards after the block gains (`DoTurnEnd`):
            /// Burn (Unpowered|Move) eats block, Beckon is Unblockable.
            #[test]
            fn end_live_burn_and_beckon() {
                let (mut s, _) = end_board();
                relic(&mut s, RelicKind::Orichalcum, 0);
                hand(&mut s, CardId::Burn);
                let r = end_turn_live(&s);
                assert_eq!((r.block, r.self_loss), (6 - 2, 0));
                hand(&mut s, CardId::Beckon);
                let r = end_turn_live(&s);
                assert_eq!((r.block, r.self_loss), (6 - 2, 6));
                assert_eq!(s.player.current_hp, 80);
            }

            /// `ConstrictPower.AfterSideTurnEnd`: Unpowered self-damage through
            /// the block left after the turn end.
            #[test]
            fn end_live_constrict() {
                let (mut s, _) = end_board();
                let p = player(&s);
                give_power(&mut s, p, PowerKind::Constrict, 3);
                let r = end_turn_live(&s);
                assert_eq!((r.block, r.self_loss), (0, 3));
                s.player.block = 5;
                let r = end_turn_live(&s);
                assert_eq!((r.block, r.self_loss), (-3, 0));
            }

            /// Beating Remnant's budget (20 less this turn's HP loss) and the
            /// Buffer stacks left after the self-damage.
            #[test]
            fn end_live_remnant_and_buffer() {
                let (mut s, _) = end_board();
                relic(&mut s, RelicKind::BeatingRemnant, 15);
                assert_eq!(end_turn_live(&s).remnant, Some(5));
                hand(&mut s, CardId::Beckon);
                let r = end_turn_live(&s);
                assert_eq!((r.self_loss, r.remnant), (5, Some(0)));
                let (mut s, _) = end_board();
                assert_eq!(end_turn_live(&s).remnant, None);
                let p = player(&s);
                give_power(&mut s, p, PowerKind::Buffer, 2);
                hand(&mut s, CardId::Beckon);
                let r = end_turn_live(&s);
                assert_eq!((r.self_loss, r.buffer), (0, 1));
            }

            /// `HardenedShellPower.BeforeSideTurnStart` resets its budget at the
            /// enemies' turn start, before Poison ticks.
            #[test]
            fn end_live_hardened_shell_resets_before_poison() {
                let (mut s, e) = end_board();
                set_enemy(&mut s, e, 12, 0);
                counted(&mut s, e, PowerKind::HardenedShell, 10, 10);
                give_power(&mut s, e, PowerKind::Poison, 12);
                // Capped at 10 this turn, not 0.
                assert!(end_turn_live(&s).dies.is_empty());
                set_enemy(&mut s, e, 10, 0);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
            }

            /// Block multipliers apply to the turn-end gains as the engine
            /// grants them: Shadowmeld doubles every block.
            #[test]
            fn end_live_shadowmeld_doubles_orichalcum() {
                let (mut s, _) = end_board();
                relic(&mut s, RelicKind::Orichalcum, 0);
                let p = player(&s);
                give_power(&mut s, p, PowerKind::Shadowmeld, 1);
                assert_eq!(end_turn_live(&s).block, 12);
            }

            /// The `DoTurnEnd` ethereal exhaust fires Feel No Pain.
            #[test]
            fn end_live_ethereal_feel_no_pain() {
                let (mut s, _) = end_board();
                let p = player(&s);
                give_power(&mut s, p, PowerKind::FeelNoPain, 3);
                hand(&mut s, CardId::Dazed);
                assert_eq!(end_turn_live(&s).block, 3);
            }

            /// A choice already open on the state is not one the walk opened:
            /// the turn end still counts.
            #[test]
            fn end_live_with_a_choice_open() {
                let (mut s, _) = end_board();
                relic(&mut s, RelicKind::Orichalcum, 0);
                let uid = hand(&mut s, CardId::StrikeIronclad);
                s.request_card_choice(
                    sts2_core::ChoiceKind::CardSelect,
                    vec![sts2_core::ChoiceOption {
                        card_id: CardId::StrikeIronclad,
                        upgraded: false,
                        card_uid: Some(uid),
                        source_pile: Some(sts2_core::cards::PileKind::Hand),
                    }],
                    1,
                    1,
                    false,
                );
                assert!(s.pending_choice.is_some());
                assert_eq!(end_turn_live(&s).block, 6);
            }

            /// A random listener on a bus the walk runs (Juggernaut's hit on a
            /// random enemy when the turn end grants block) counts only when
            /// every reseeded walk agrees: with one enemy it is certain.
            #[test]
            fn end_live_random_listener_not_certain() {
                let (mut s, e0) = end_board();
                set_enemy(&mut s, e0, 5, 0);
                relic(&mut s, RelicKind::Orichalcum, 0);
                let p = player(&s);
                give_power(&mut s, p, PowerKind::Juggernaut, 5);
                assert_eq!(end_turn_live(&s).dies, vec![0]);
                let e1 = second_enemy(&mut s, 5);
                set_enemy(&mut s, e1, 5, 0);
                assert!(end_turn_live(&s).dies.is_empty());
            }

            #[test]
            fn end_live_plain_board_is_empty() {
                let (mut s, _) = end_board();
                hand(&mut s, CardId::StrikeIronclad);
                assert_eq!(end_turn_live(&s), EndTurnLive::default());
            }
        }
    }

    // --- state facts for the shared net (GH #1197) --------------------------

    mod facts {
        use super::*;
        use sts2_core::cards::CardId;
        use sts2_core::{GameVersion, Power, PowerKind, PowerUid, Relic, RelicKind};

        fn board(version: GameVersion) -> CombatState {
            sts2_content::cards::dispatch::ensure_installed();
            sts2_content::relics::dispatch::ensure_installed();
            sts2_core::relic_effects::ensure_installed();
            let mut state = CombatState {
                game_version: version,
                ..CombatState::default()
            };
            state.player.current_hp = 80;
            state.player.max_hp = 80;
            state
        }

        fn add_card(state: &mut CombatState, id: CardId) -> CardUid {
            let mut card =
                sts2_content::cards::v0111::card_pod_for(id, false, GameVersion::V0_111_0)
                    .expect("spec exists");
            let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits"))
                .expect("non-zero");
            card.uid = uid;
            state.cards.push(card);
            uid
        }

        fn player_power(state: &mut CombatState, kind: PowerKind, amount: i32, counter: i32) {
            let pid = PowerUid::new(u32::try_from(state.powers.len() + 1).expect("fits"))
                .expect("non-zero");
            state.powers.push(Power {
                id: pid.0,
                kind,
                amount,
                counter,
                applier: None,
                skip_next_tick: false,
            });
            state.player.powers.push(pid);
        }

        fn relic(state: &mut CombatState, kind: RelicKind, counter: i32) {
            let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).expect("fits"))
                .expect("non-zero");
            state.relics.push(Relic {
                id,
                kind,
                counter,
                is_wax: false,
                is_melted: false,
            });
        }

        fn power_display(v: &Value, name: &str) -> Value {
            v["player"]["powers"]
                .as_array()
                .expect("powers")
                .iter()
                .find(|p| p["name"] == name)
                .expect("power present")["display_amount"]
                .clone()
        }

        fn relic_dict(v: &Value, id: &str) -> Value {
            v["player"]["relics"]
                .as_array()
                .expect("relics")
                .iter()
                .find(|r| r["id"] == id)
                .expect("relic present")
                .clone()
        }

        /// Master Planner made a Skill Sly (`sly_this_combat`), and the card
        /// was then discarded: the pile card still carries the fact, under
        /// `pile_flags` so the hand-only `flags` contract is untouched.
        #[test]
        fn pile_cards_keep_sly_flag() {
            let mut s = board(GameVersion::V0_111_0);
            let discarded = add_card(&mut s, CardId::DefendIronclad);
            s.player.sly_this_combat.push(discarded);
            s.player.discard.push(discarded);
            let drawn = add_card(&mut s, CardId::DefendIronclad);
            s.player.retain_this_turn.push(drawn);
            s.player.sly_this_turn.push(drawn);
            s.player.draw.push(drawn);
            let in_hand = add_card(&mut s, CardId::DefendIronclad);
            s.player.sly_this_combat.push(in_hand);
            s.player.hand.push(in_hand);
            let plain = add_card(&mut s, CardId::StrikeIronclad);
            s.player.exhaust.push(plain);

            let v = combat_state_to_value(&s).expect("serializes");
            let d = &v["deck"]["discard_pile"][0];
            assert_eq!(d["pile_flags"], serde_json::json!(["sly"]));
            assert!(d.get("flags").is_none(), "pile `flags` unchanged: {d}");
            let dr = &v["deck"]["draw_pile"][0];
            assert_eq!(
                dr["pile_flags"],
                serde_json::json!(["sly_turn", "retain_turn"])
            );
            // Hand cards keep `flags` exactly as before and get no `pile_flags`.
            let h = &v["deck"]["hand"][0];
            assert_eq!(h["flags"], serde_json::json!(["sly"]));
            assert!(h.get("pile_flags").is_none());
            assert!(v["deck"]["exhaust_pile"][0].get("pile_flags").is_none());
        }

        #[test]
        fn discards_and_type_counts() {
            let mut s = board(GameVersion::V0_111_0);
            let v = combat_state_to_value(&s).expect("serializes");
            for k in [
                "discards_this_turn",
                "attacks_played_this_turn",
                "skills_played_this_turn",
            ] {
                assert!(v["player"].get(k).is_none(), "{k} absent at 0");
            }
            s.combat_log.cards_discarded_this_turn = 2;
            s.player.cards_played_this_turn[CardType::Attack.index()] = 3;
            s.player.cards_played_this_turn[CardType::Skill.index()] = 1;
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(v["player"]["discards_this_turn"], 2);
            assert_eq!(v["player"]["attacks_played_this_turn"], 3);
            assert_eq!(v["player"]["skills_played_this_turn"], 1);
        }

        /// Emotion Chip is armed while a pulse is pending: a hit that cost HP
        /// this turn (or last turn, not yet consumed at turn start).
        #[test]
        fn relic_armed_emotion_chip() {
            let mut s = board(GameVersion::V0_111_0);
            relic(&mut s, RelicKind::EmotionChip, 0);
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(relic_dict(&v, "EMOTION_CHIP")["armed"], false);
            // A hit this turn (the engine's THIS_TURN bit).
            s.relics.iter_mut().next().expect("chip").counter = 0b10;
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(relic_dict(&v, "EMOTION_CHIP")["armed"], true);
            // Other relics carry no `armed` key.
            relic(&mut s, RelicKind::Anchor, 0);
            let v = combat_state_to_value(&s).expect("serializes");
            assert!(relic_dict(&v, "ANCHOR").get("armed").is_none());
        }

        /// Mini Regent fires on the first star spend of a turn: armed until
        /// then, disarmed after.
        #[test]
        fn relic_armed_mini_regent_used() {
            let mut s = board(GameVersion::V0_111_0);
            relic(&mut s, RelicKind::MiniRegent, 0);
            s.player.stars = 3;
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(relic_dict(&v, "MINI_REGENT")["armed"], true);
            let _ = s.spend_card_play_stars(1);
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(relic_dict(&v, "MINI_REGENT")["armed"], false);
        }

        /// Regalite's once-per-turn latch is a v0.111.0 thing; v0.107.1's
        /// Regalite fires on every generated card.
        #[test]
        fn relic_armed_regalite_latch() {
            let mut s = board(GameVersion::V0_111_0);
            relic(&mut s, RelicKind::Regalite, 0);
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(relic_dict(&v, "REGALITE")["armed"], true);
            s.relics.iter_mut().next().expect("regalite").counter = 1;
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(relic_dict(&v, "REGALITE")["armed"], false);
            s.game_version = GameVersion::V0_107_1;
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(relic_dict(&v, "REGALITE")["armed"], true);
        }

        /// C# `PaleBlueDotPower.DisplayAmount => Math.Max(0, 5 -
        /// AttacksPlayedThisTurn)`, where that count is every finished play
        /// of the owner this turn (no type filter despite the name).
        #[test]
        fn pale_blue_dot_display() {
            let mut s = board(GameVersion::V0_111_0);
            player_power(&mut s, PowerKind::PaleBlueDot, 2, 0);
            s.combat_log.card_plays_finished_this_turn = 2;
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(power_display(&v, "PALE_BLUE_DOT"), 3);
            s.combat_log.card_plays_finished_this_turn = 7;
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(power_display(&v, "PALE_BLUE_DOT"), 0);
            // v0.107.1's Pale Blue Dot has no DisplayAmount override.
            s.game_version = GameVersion::V0_107_1;
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(power_display(&v, "PALE_BLUE_DOT"), 2);
        }

        fn hand_card(state: &mut CombatState, id: CardId) -> CardUid {
            let uid = add_card(state, id);
            state.player.hand.push(uid);
            uid
        }

        /// Void Form zeroes the star cost of the turn's first `Amount` plays
        /// (C# `VoidFormPower.TryModifyStarCost`): `star_cost_live` follows,
        /// `base_star_cost` keeps the printed cost, and a non-star card
        /// carries neither key.
        #[test]
        fn star_cost_void_form_free() {
            let mut s = board(GameVersion::V0_111_0);
            hand_card(&mut s, CardId::Comet);
            hand_card(&mut s, CardId::StrikeIronclad);
            let v = combat_state_to_value(&s).expect("serializes");
            let comet = &v["deck"]["hand"][0];
            assert_eq!(comet["star_cost_live"], 5);
            assert_eq!(comet["base_star_cost"], 5);
            let strike = &v["deck"]["hand"][1];
            assert!(strike.get("star_cost_live").is_none());
            assert!(strike.get("base_star_cost").is_none());
            player_power(&mut s, PowerKind::VoidForm, 1, 0);
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(v["deck"]["hand"][0]["star_cost_live"], 0);
            assert_eq!(v["deck"]["hand"][0]["base_star_cost"], 5);
            // The existing `star_cost` key stays unemitted by the engine.
            assert!(v["deck"]["hand"][0].get("star_cost").is_none());
        }

        /// A v0.111.0 cost change reads in the state's build: Guiding Star
        /// prints 1 there, 2 in v0.107.1.
        #[test]
        fn star_cost_reads_the_build() {
            let mut s = board(GameVersion::V0_111_0);
            hand_card(&mut s, CardId::GuidingStar);
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(v["deck"]["hand"][0]["base_star_cost"], 1);
            assert_eq!(v["deck"]["hand"][0]["star_cost_live"], 1);
            s.game_version = GameVersion::V0_107_1;
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(v["deck"]["hand"][0]["base_star_cost"], 2);
        }

        /// Stardust is star-X (`HasStarCostX`): both keys read -1.
        #[test]
        fn star_cost_x_stardust_is_minus_one() {
            let mut s = board(GameVersion::V0_111_0);
            s.player.stars = 4;
            hand_card(&mut s, CardId::Stardust);
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(v["deck"]["hand"][0]["star_cost_live"], -1);
            assert_eq!(v["deck"]["hand"][0]["base_star_cost"], -1);
        }

        /// The parse ignores the new keys: a Void-Form-zeroed `star_cost_live`
        /// must not come back as a free-this-turn rider, and the wire with
        /// the keys rebuilds the same state as the wire without them.
        #[test]
        fn star_cost_live_is_ignored_by_the_parse() {
            let mut s = board(GameVersion::V0_111_0);
            hand_card(&mut s, CardId::Comet);
            player_power(&mut s, PowerKind::VoidForm, 1, 0);
            let with = combat_state_to_value(&s).expect("serializes");
            assert_eq!(with["deck"]["hand"][0]["star_cost_live"], 0);
            let mut without = with.clone();
            let card = without["deck"]["hand"][0]
                .as_object_mut()
                .expect("card dict");
            card.remove("star_cost_live");
            card.remove("base_star_cost");
            let a = crate::state_from_wire(&&with).expect("parses");
            let b = crate::state_from_wire(&&without).expect("parses");
            let comet = a
                .cards
                .iter()
                .find(|c| c.card_id == CardId::Comet)
                .expect("comet");
            assert_eq!(comet.combat_var(VarKey::StarCostFreeThisTurn), 0);
            assert_eq!(comet.combat_var(VarKey::StarCostFreeThisCombat), 0);
            assert_eq!(
                combat_state_to_value(&a).expect("re-emits"),
                combat_state_to_value(&b).expect("re-emits")
            );
        }

        /// Every power without a `DisplayAmount` override shows its `amount`
        /// (C# `PowerModel.DisplayAmount => Amount`), on both builds, whatever
        /// its `counter` and the turn's plays: the Monologue and Pale Blue Dot
        /// overrides above must not spread.
        #[test]
        fn plain_powers_display_their_amount() {
            let plain = [
                (PowerKind::Strength, 3, "STRENGTH"),
                (PowerKind::Dexterity, -2, "DEXTERITY"),
                (PowerKind::Vigor, 5, "VIGOR"),
                (PowerKind::Focus, 2, "FOCUS"),
                (PowerKind::Ritual, 1, "RITUAL"),
                (PowerKind::Thorns, 4, "THORNS"),
            ];
            for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
                let mut s = board(version);
                for (i, (kind, amount, _)) in plain.iter().enumerate() {
                    player_power(&mut s, *kind, *amount, 7 + i32::try_from(i).expect("small"));
                }
                s.combat_log.card_plays_finished_this_turn = 4;
                let v = combat_state_to_value(&s).expect("serializes");
                for (_, amount, name) in plain {
                    assert_eq!(power_display(&v, name), amount, "{name} on {version:?}");
                }
            }
        }

        /// C# `MonologuePower.DisplayAmount => StrengthApplied`, the engine's
        /// `counter`.
        #[test]
        fn monologue_display() {
            let mut s = board(GameVersion::V0_111_0);
            player_power(&mut s, PowerKind::Monologue, 1, 3);
            let v = combat_state_to_value(&s).expect("serializes");
            assert_eq!(power_display(&v, "MONOLOGUE"), 3);
        }
    }
}
