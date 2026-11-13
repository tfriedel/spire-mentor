//! Enchantment hook channels (bead `sts2-rs-6ynt.11`).
//!
//! Mirrors the six dispatch points Python's `sts2_env/cards/enchantments.py`
//! exposes at pin `sts2-rl-agent@01c094d`:
//!
//! | Channel                       | Python entry                                  |
//! |-------------------------------|-----------------------------------------------|
//! | [`dispatch_on_card_drawn`]    | `enchantments.py:121` `on_card_drawn`         |
//! | [`dispatch_after_player_turn_start`] | `enchantments.py:126` `after_player_turn_start` |
//! | [`dispatch_before_flush`]     | `enchantments.py:137` `before_flush`          |
//! | [`dispatch_modify_shuffle_order`] | `enchantments.py:149` `modify_shuffle_order` |
//! | [`dispatch_on_card_played`]   | `enchantments.py:158` `on_card_played`        |
//! | [`dispatch_after_card_played`] | `enchantments.py:184` `after_card_played`    |
//!
//! The table above names the Python *channel* each dispatcher mirrors, NOT
//! the site it is fired from. One site deliberately diverges from the Python
//! pin: [`dispatch_on_card_played`] now fires AFTER the card's own effect
//! resolves, per C# `CardModel.OnPlayWrapper` (`CardModel.cs:1930-1943`) —
//! see that function's docs (bead `sts2-rs-ench-onplay-order`).
//!
//! ## Surface-bead scope
//!
//! This bead lands the dispatch surface only — every per-card stub
//! returns without applying enchantment effects. Concrete listener
//! bodies (Slither cost-roll on draw, Imbued auto-play on turn 1,
//! SlumberingEssence pre-flush cost cut, PerfectFit re-shuffle order,
//! Adroit / Swift / Sown / Corrupted / Momentum on play, Vigorous /
//! Glam / Goopy used-marker after play) land in follow-up sub-beads
//! per `docs/audits/2026-05-15-ironclad-completeness/06-powers-enchantments.md`.
//!
//! ## Empty-bus early-out
//!
//! Each dispatcher walks only the relevant pile(s) and returns early
//! when no card in scope has any enchantments. This matches the
//! discipline established by bead `sts2-rs-4lk9` (powers/relics
//! empty-bus early-out) so adding the surface does not regress the
//! `transitions_per_sec` anchor with empty enchantment loadouts (the
//! common case at this pin until enchantments are populated by the
//! run-builder layer).

use crate::cards::{Card, EnchantmentId, VarKey};
use crate::hooks::{CreatureRef, ValueProps};
use crate::{CardUid, CombatState, PlayerCombatState};

// Bit-packed per-combat "used" flags stashed on the card's
// `VarKey::EnchantmentUsed` combat-var. The Vigorous + Glam bits agree
// byte-for-byte with the content-side `sts2-content::enchantments`
// encoding (helpers `mark_vigorous_used` / `vigorous_used` /
// `mark_glam_used` / `glam_used`). Swift + Sown bits are core-side
// only at this pin — their consumed state is queried directly here on
// the dispatch path, not from sts2-content helpers.
const VIGOROUS_USED_BIT: i32 = 0b0001;
const GLAM_USED_BIT: i32 = 0b0010;
const SWIFT_USED_BIT: i32 = 0b0100;
const SOWN_USED_BIT: i32 = 0b1000;
/// v0.113.0 Swift / Sown disable in `AfterCardPlayed`, not in `OnPlay`
/// (`Swift.cs:19`, `Sown.cs:17`). The effect fires with the card still
/// Normal; these PENDING bits stand for "disabled once this play iteration's
/// `AfterCardPlayed` reaches the enchantment". They turn into the USED bits at
/// the card's next enchantment `OnPlay` (always after that `AfterCardPlayed`),
/// and a Music Box clone made during the same `AfterCardPlayed` (relics come
/// before card enchantments in `IterateHookListeners`) drops them, so it keeps
/// a Normal Swift / Sown ([`clear_pending_first_play_disable`]).
const SWIFT_PENDING_BIT: i32 = 0b1_0000_0000;
const SOWN_PENDING_BIT: i32 = 0b10_0000_0000;

/// Music Box clone of the card being played (v0.113.0): the enchantment has
/// not yet disabled itself, so drop the pending Swift / Sown disables.
pub fn clear_pending_first_play_disable(card: &mut crate::cards::Card) {
    let bits = card.combat_var(VarKey::EnchantmentUsed);
    if bits & (SWIFT_PENDING_BIT | SOWN_PENDING_BIT) != 0 {
        card.set_combat_var(
            VarKey::EnchantmentUsed,
            bits & !(SWIFT_PENDING_BIT | SOWN_PENDING_BIT),
        );
    }
}

/// Additive damage bonus contributed by the *played card's* enchantments,
/// for a powered attack. Mirrors the C# `*.EnchantDamageAdditive` chain
/// (`Sharp`/`Inky` flat, `Vigorous` once-per-card, `Momentum` accumulated
/// into `VarKey::MomentumDamage`) and the Python `enchant_damage_additive`.
/// Returns 0 for unpowered hits (C# gates each enchantment on
/// `props.IsPoweredAttack()`).
///
/// This is the canonical core copy; the sts2-content helper delegates here
/// so the value the damage pipeline applies (see [`hooks::modify_damage`])
/// can never drift from the unit-tested content logic. Bead
/// `sts2-rs-enchant-damage-additive` (PARITY136: a `Sharp(+2)` STRIKE_DEFECT
/// auto-played by Uproar dealt 6 in Rust vs 8 in C# because the enchantment
/// additive was never threaded into `modify_damage`).
#[must_use]
pub fn enchant_damage_additive(card: &crate::cards::Card, is_powered: bool) -> i32 {
    enchant_damage_additive_for(card, is_powered, crate::GameVersion::V0_107_1)
}

/// [`enchant_damage_additive`] for game build `version`. v0.111.0 drops
/// Inky's `+1`: the enchantment keeps only its `PowerVar<WeakPower>(1)` and
/// no longer overrides `EnchantDamageAdditive` (`v0.111.0/…/Enchantments/
/// Inky.cs:19`; v0.107.1 `Inky.cs:43` returned `DamageVar(1m)` on a powered
/// attack). Every other term is version-independent.
#[must_use]
pub fn enchant_damage_additive_for(
    card: &crate::cards::Card,
    is_powered: bool,
    version: crate::GameVersion,
) -> i32 {
    if !is_powered {
        return 0;
    }
    let mut total = card.enchantment(EnchantmentId::Sharp);
    // Momentum stashes `+stack` per play onto its own `MomentumDamage` slot
    // (see `dispatch_on_card_played`); the running total is the bonus. It
    // used to share Rampage's `RampageDamage` slot, which made a Momentum
    // Rampage count both growths twice (GH #1021).
    if card.has_enchantment(EnchantmentId::Momentum) {
        total = total.saturating_add(card.combat_var(VarKey::MomentumDamage));
    }
    // Vigorous applies once per card per combat; the consumed bit lives on
    // `VarKey::EnchantmentUsed`.
    if card.combat_var(VarKey::EnchantmentUsed) & VIGOROUS_USED_BIT == 0 {
        total = total.saturating_add(card.enchantment(EnchantmentId::Vigorous));
    }
    // Inky (Blade of Ink shivs): fixed +1, regardless of stacks. v0.107.1
    // `Inky.cs` `EnchantDamageAdditive` returns `DynamicVars.Damage.BaseValue`
    // = `DamageVar(1m)` (was `2m` in v0.103.2) on a powered attack.
    // v0.111.0: Inky has no EnchantDamageAdditive any more (decomp Inky.cs:19)
    if card.has_enchantment(EnchantmentId::Inky) && version < crate::GameVersion::V0_111_0 {
        total = total.saturating_add(1);
    }
    // TezcatarasEmber (v0.107.1 NEW enchant, `TezcatarasEmber.cs`): a flat
    // `+3` on a powered attack. `EnchantDamageAdditive` returns
    // `base.DynamicVars.Damage.BaseValue` = `DamageVar(3m)` when
    // `props.IsPoweredAttack()`, else `0` (the powered gate is the `is_powered`
    // early-return above). The value is exposed via [`TEZCATARAS_EMBER_DAMAGE`]
    // so the `NutritiousSoup` follow-up (`sts2-rs-ajjri`) can query it without
    // re-deriving the constant.
    if card.has_enchantment(EnchantmentId::TezcatarasEmber) {
        total = total.saturating_add(TEZCATARAS_EMBER_DAMAGE);
    }
    total
}

/// Flat powered-attack damage bonus granted by the v0.107.1 `TezcatarasEmber`
/// enchantment (`TezcatarasEmber.cs` `CanonicalVars = DamageVar(3m, Move)`).
/// Exposed as a named constant so callers (e.g. the `NutritiousSoup`
/// follow-up `sts2-rs-ajjri`) can query the enchant value directly. The
/// powered gate lives at the [`enchant_damage_additive`] call site, mirroring
/// `EnchantDamageAdditive`'s `props.IsPoweredAttack()` guard.
pub const TEZCATARAS_EMBER_DAMAGE: i32 = 3;

/// Multiplicative damage scalar in fixed-point (×100) contributed by the played
/// card's enchantments, for a powered attack. Mirrors the C#
/// `*.EnchantDamageMultiplicative` chain: `Favored.cs` and `Instinct.cs` return
/// `2m` (→ 200), `Corrupted.cs` returns `1.5m` (→ 150); every other case (and
/// any unpowered hit) is `1×` (→ 100). Returns the integer ×100 form so the
/// damage pipeline can stay float-free. Canonical core copy; the sts2-content
/// helper delegates here so the unit-tested logic and the value applied in
/// [`crate::hooks::modify_damage`] share one source of truth.
#[must_use]
pub fn enchant_damage_multiplicative_x100(card: &crate::cards::Card, is_powered: bool) -> i32 {
    if !is_powered {
        return 100;
    }
    if card.has_enchantment(EnchantmentId::Favored) || card.has_enchantment(EnchantmentId::Instinct)
    {
        return 200;
    }
    if card.has_enchantment(EnchantmentId::Corrupted) {
        return 150;
    }
    100
}

/// Additive block bonus contributed by the played card's enchantments.
/// Mirrors the C# `Nimble.cs` (flat `+Amount` block) and `Goopy.cs` (block
/// grows `Amount - 1` per replay) `EnchantBlockAdditive` chain.
///
/// **v0.107.1 change (§8):** `EnchantBlockAdditive` lost its `props`
/// argument and the `IsPoweredCardOrMonsterMoveBlock` guard — the block
/// bonus now applies **unconditionally** whenever the block grant carries a
/// card source with the enchantment (`Hook.cs:1461` calls
/// `enchantment.EnchantBlockAdditive(num)` with no props). This is the patch
/// note "Entrench can now gain extra Block from enchantments like Nimble":
/// Entrench re-grants block via an unpowered Move, which the old guard
/// excluded. The powered gate was dropped here to match (was: `Nimble.cs` /
/// `Goopy.cs` `if (!props.IsPoweredCardOrMonsterMoveBlock()) return 0m;`).
///
/// Canonical core copy; the sts2-content helper delegates here so the value
/// the block pipeline applies (see [`crate::hooks::modify_block`]) can never
/// drift from the unit-tested content logic.
#[must_use]
pub fn enchant_block_additive(card: &crate::cards::Card) -> i32 {
    let mut total = card.enchantment(EnchantmentId::Nimble);
    if card.has_enchantment(EnchantmentId::Goopy) {
        // Python: `card.enchantments.get("Goopy", 1) - 1`. Goopy starts at 1 and
        // ticks up each play, so the block bonus is `Amount - 1` (≥ 0).
        total = total.saturating_add((card.enchantment(EnchantmentId::Goopy) - 1).max(0));
    }
    total
}

/// Extra plays the *played card's* enchantment grants (Spiral / Glam).
///
/// C# `CardModel.GeneratePlayCount` (v0.107.1 `CardModel.cs:2016-2021`,
/// v0.111.0 `CardModel.cs:2029-2034`) starts from
/// `GetEnchantedReplayCount() + 1`, where `GetEnchantedReplayCount` is
/// `Enchantment?.EnchantPlayCount(BaseReplayCount)` (`CardModel.cs:1130-1133`)
/// — so the enchantment bonus is folded in for EVERY play (hand play and
/// auto-play alike, both run `OnPlayWrapper`), before `Hook.ModifyCardPlayCount`.
/// `Spiral.EnchantPlayCount` adds the fixed `Times` (1) always
/// (`Spiral.cs:30-33`); `Glam.EnchantPlayCount` adds it only until its
/// `AfterCardPlayed` latches `UsedThisCombat` (`Glam.cs:34-56`), which the
/// [`dispatch_after_card_played`] Glam-used bit mirrors.
///
/// Canonical core copy; `sts2_content::enchantments::enchant_play_count_bonus`
/// delegates here. GH #806 (PARITY149: a Silken-Tress Glam SCULPTING_STRIKE
/// hit once in Rust, twice in C#).
#[must_use]
pub fn enchant_play_count_bonus(card: &crate::cards::Card) -> i32 {
    /// `DynamicVar("Times", 1m)` — the fixed bonus, NOT the enchantment Amount.
    const TIMES: i32 = 1;
    let mut bonus = 0;
    if card.has_enchantment(EnchantmentId::Spiral) {
        bonus += TIMES;
    }
    if card.has_enchantment(EnchantmentId::Glam)
        && card.combat_var(VarKey::EnchantmentUsed) & GLAM_USED_BIT == 0
    {
        bonus += TIMES;
    }
    bonus
}

/// Returns `true` if any card in `pile` has at least one enchantment
/// stack. The caller uses this for the per-channel empty-bus
/// early-out — if the pile is enchantment-free the dispatch loop is
/// skipped entirely.
#[inline]
fn pile_has_any_enchantments(state: &CombatState, pile: &[CardUid]) -> bool {
    pile.iter().any(|uid| {
        let idx = (uid.get() as usize).saturating_sub(1);
        state
            .cards
            .get(idx)
            .is_some_and(|c| !c.enchantments.is_empty())
    })
}

/// Borrow a card by uid. Returns `None` if the uid is out of range
/// (defensive — every uid in a player pile should be live in
/// `state.cards`, but the dispatch path stays panic-free).
#[inline]
fn card_for(state: &CombatState, uid: CardUid) -> Option<&Card> {
    state.cards.get((uid.get() as usize).saturating_sub(1))
}

// ---------------------------------------------------------------------------
// 1. on_card_drawn
// ---------------------------------------------------------------------------

/// Fire enchantment `on_card_drawn` for a single card just moved into
/// hand. Called from the draw path once per card landed.
///
/// Python entry: `enchantments.py:121` —
/// `on_card_drawn(card, combat, from_hand_draw)`.
///
/// `from_hand_draw` distinguishes turn-start hand draws from
/// card-effect draws. NOTE: Slither ignores the flag — C#
/// `Slither.AfterCardDrawn` rerolls on EVERY draw-pile draw (turn-start
/// AND effect draws), so all draw paths must dispatch here
/// unconditionally (census v8 seeds 1003/1006/1042/1049).
pub fn dispatch_on_card_drawn(state: &mut CombatState, card_uid: CardUid, from_hand_draw: bool) {
    // Empty-bus early-out: if the drawn card has no enchantments,
    // there is nothing to dispatch.
    let Some(card) = card_for(state, card_uid) else {
        return;
    };
    if card.enchantments.is_empty() {
        return;
    }
    on_card_drawn_stub(state, card_uid, from_hand_draw);
}

/// Per-card listener body.
///
/// **Wave 3 (`sts2-rs-g7rni`):** Slither cost-roll. Python pin:
/// `if card.has_enchantment("Slither"):
///   card.set_combat_cost(combat.combat_energy_costs_rng.next_int(0, 3))`.
///
/// **Divergence:** Python uses a dedicated `combat_energy_costs_rng`
/// sub-stream on `RunState`; the dispatch site here has access to
/// `state.rng` (the combat-wide Xoshiro RNG) only. Rolling the cost
/// off `state.rng` keeps Slither observably deterministic per fixture
/// but the resulting cost will not match Python's stream byte-for-byte.
/// Tracked in `tests/parity/known_divergences.md` alongside the other
/// RNG-stream divergences (a follow-up bead can wire the dedicated
/// stream once `RunState::rng` plumbing reaches the on_card_drawn
/// site).
///
/// **Empty-state guard:** `from_hand_draw` is unused at this pin —
/// Python's `on_card_drawn` Slither leg fires regardless of the draw
/// source, only the `Imbued` auto-play leg is round-1 + draw-pile only
/// (handled by `after_player_turn_start`, not here).
#[inline]
fn on_card_drawn_stub(state: &mut CombatState, card_uid: CardUid, _from_hand_draw: bool) {
    let idx = (card_uid.get() as usize).saturating_sub(1);
    let Some(card) = state.cards.get(idx) else {
        return;
    };
    if !card.has_enchantment(EnchantmentId::Slither) {
        return;
    }
    // C# `Slither.AfterCardDrawn`: `if (base.Card.Pile.Type != PileType.Hand)
    // return;` — no roll for a card that already left the hand.
    if !state.player.hand.contains(&card_uid) {
        return;
    }
    let new_cost = state.run_rng.combat_energy_costs.next_int(0, 3);
    if let Some(card) = state.cards.get_mut(idx) {
        // `EnergyCost.SetThisCombat(NextEnergyCost())` (Slither.cs:50).
        card.cost.set_this_combat(new_cost, false);
    }
}

// ---------------------------------------------------------------------------
// 2. after_player_turn_start
// ---------------------------------------------------------------------------

/// Fire the enchantment turn-start channel once per player turn. Walks the
/// draw pile to surface Imbued cards (auto-play on turn 1). Despite the name,
/// the caller fires it in the AutoPrePlay phase
/// (`Imbued.AfterAutoPrePlayPhaseEntered`, Imbued.cs:20), after
/// `AfterPlayerTurnStart` and `AfterSideTurnStart`.
///
/// Python entry: `enchantments.py:126` —
/// `after_player_turn_start(owner, combat)`.
///
/// Empty-bus early-out: if no card in any pile has any enchantments, skip
/// the walk entirely.
pub fn dispatch_after_player_turn_start(state: &mut CombatState) {
    let p = &state.player;
    if ![
        &p.hand[..],
        &p.draw[..],
        &p.discard[..],
        &p.exhaust[..],
        &p.play[..],
    ]
    .into_iter()
    .any(|pile| pile_has_any_enchantments(state, pile))
    {
        return;
    }
    after_player_turn_start_stub(state);
}

/// Listener body.
///
/// **Wave 3 (`sts2-rs-g7rni`):** Imbued auto-play. On the first player turn
/// only, auto-play every card carrying the Imbued enchantment, wherever it
/// sits. `Hook.AfterAutoPrePlayPhaseEntered` walks the listener list that
/// `CombatState.IterateHookListeners` snapshots up front: the cards of Hand,
/// Draw, Discard, Exhaust and Play, in that order (CombatState.cs). So an
/// Imbued card drawn into the opening hand (a deck of 5 cards or fewer) plays
/// too, and before the ones still in the draw pile (GH #1094).
///
/// **v0.107.1 retime/rewindow (§8):** C# `Imbued.cs` moved its body from
/// `BeforePlayPhaseStart` to `AfterAutoPrePlayPhaseEntered` and changed the
/// window from `CombatState.RoundNumber == 1` to
/// `PlayerCombatState.TurnNumber <= 1`. The AutoPrePlay phase follows
/// `AfterPlayerTurnStart` and `AfterSideTurnStart`, so the combat loop fires
/// this from its AutoPrePlay tail. `TurnNumber` starts at `1` and increments once per player
/// turn; with no extra-turn mechanic in the solo engine `TurnNumber <= 1`
/// ≡ `round_no() == 1` (see the MrStruggles `RoundNumber→TurnNumber` note),
/// so the existing round-1 gate below is the faithful window.
///
/// **Implementation:** snapshot the Imbued uids first (so subsequent
/// shuffles / draws don't change what we're iterating), then for each
/// uid swap it to position 0 of the draw pile and call
/// [`crate::combat_loop::auto_play_card_from_draw_top`] with
/// `force_exhaust = false`. This reuses the canonical auto-play pipeline
/// (target resolution, after_card_played dispatch, natural pile routing)
/// so Imbued-played cards land in discard (or self-exhaust) — matching
/// C# `Imbued.cs`'s `CardCmd.AutoPlay`, which does NOT force exhaust.
///
/// **EchoForm divergence inherited:** `auto_play_card_from_draw_top`
/// does not run the EchoForm replay path; per the existing divergence
/// note on that function (parity follow-up `sts2-rs-tlyk`).
#[inline]
fn after_player_turn_start_stub(state: &mut CombatState) {
    // v0.107.1 window: `PlayerCombatState.TurnNumber <= 1` (was
    // `RoundNumber == 1`). `round_no()` mirrors `TurnNumber` in the solo
    // engine (no extra-turn mechanic), and it starts at 1, so `> 1` is the
    // faithful "after the first player turn" exclusion.
    if state.turn_number() > 1 {
        return;
    }
    // Snapshot the Imbued uids in C#'s listener order: Hand, Draw, Discard,
    // Exhaust, Play.
    let p = &state.player;
    let imbued: Vec<CardUid> = p
        .hand
        .iter()
        .chain(p.draw.iter())
        .chain(p.discard.iter())
        .chain(p.exhaust.iter())
        .chain(p.play.iter())
        .copied()
        .filter(|uid| {
            let idx = (uid.get() as usize).saturating_sub(1);
            state
                .cards
                .get(idx)
                .is_some_and(|c| c.has_enchantment(EnchantmentId::Imbued))
        })
        .collect();
    imbued_run(state, &imbued);
}

/// Auto-play the snapshotted Imbued cards in order. C# awaits each
/// `CardCmd.AutoPlay` (Imbued.cs:20-27), so a select one of them opens (Hidden
/// Daggers' discard) resolves before the next card plays: when a play parks on
/// `pending_autoplay_resume`, the cards not yet played go onto it as a loop
/// continuation, the BeatDown way, and [`imbued_resume`] plays them once the
/// pick lands (Silent v0.111.0 seed 5447). A Sly card that same select
/// discards nests under it, and the Sly loop keeps this continuation as its
/// outer one ([`crate::card_effects::play_discarded_sly`]).
fn imbued_run(state: &mut CombatState, imbued: &[CardUid]) {
    for (i, &uid) in imbued.iter().enumerate() {
        // `CardCmd.AutoPlay` returns at once when the combat `IsOverOrEnding`
        // (CardCmd.cs:54-57, both builds), not only once it is over: an
        // Imbued Capture Spirit that kills the last enemy leaves the next
        // Imbued card unplayed where it sits.
        if crate::combat_loop::combat_is_ending(state) {
            break;
        }
        imbued_play_one(state, uid);
        if let Some(resume) = state.pending_autoplay_resume.as_mut() {
            // The parked card took any loop of its own onto its frame
            // (`park_auto_play_finish`), so the top slot is free for ours.
            debug_assert!(resume.continuation.is_none());
            resume.continuation = Some(crate::AutoPlayLoopCont {
                resume_fn: imbued_resume,
                iters: 0,
                uids: imbued[i + 1..].iter().copied().collect(),
                force_exhaust: false,
                then: None,
            });
            return;
        }
    }
}

/// Resume entry point stored on [`crate::AutoPlayLoopCont`]: play the Imbued
/// cards not yet played.
fn imbued_resume(state: &mut CombatState, cont: &crate::AutoPlayLoopCont) {
    imbued_run(state, &cont.uids);
}

/// One Imbued card's `CardCmd.AutoPlay`, from whatever pile it is in now.
fn imbued_play_one(state: &mut CombatState, uid: CardUid) {
    // `CardCmd.AutoPlay` plays the card from whatever pile it is in now.
    // Outside the draw pile no top-of-pile bookkeeping applies: the
    // any-pile auto-play moves it to Play and resolves it.
    let Some(pos) = state.player.draw.iter().position(|u| *u == uid) else {
        if !state.player.play.contains(&uid) {
            let _ = crate::combat_loop::auto_play_specific_card_from_discard(
                state, uid, /*force_exhaust=*/ false,
            );
        }
        return;
    };
    if state.player.deferred_draw_order {
        // GH #785: under BeliefDraw the auto-play below pops the top
        // through `cards::pop_top_card`, which only takes a KNOWN card
        // deterministically. The Imbued card is known, so move it to the
        // physical top and PIN it there (it is then popped with no RNG
        // read). A swap would drop whatever sat at `draw[0]` (a pinned
        // card, or the head of the sorted unknown middle) into the
        // middle of the pile and break the pile's segment layout.
        state.player.draw.remove(pos);
        state.player.draw.insert(0, uid);
        crate::cards::pin_top(&mut state.player, uid);
    } else if pos != 0 {
        // C# `CardCmd.AutoPlay(card)` plays the card where it sits and
        // leaves the rest of the pile in order, so lift it to the top
        // (remove + insert) rather than swapping, which dropped the old
        // top card into the Imbued card's slot. Imbued cards now start at
        // the BOTTOM (`ShouldStartAtBottomOfDrawPile`, CombatManager.cs:
        // 661-666, see `start_player_turn`), so a swap would bury the top
        // card at the bottom on every Imbued combat (act23 Ironclad:1013).
        state.player.draw.remove(pos);
        state.player.draw.insert(0, uid);
    }
    // Imbued does NOT force-exhaust — C# `Imbued.cs` calls
    // `CardCmd.AutoPlay(card, null)` which leaves `ExhaustOnNextPlay`
    // at its default (false), so the played card routes to its natural
    // pile (discard unless it self-exhausts). Bead q6bix F-F.
    let _ = crate::combat_loop::auto_play_card_from_draw_top(state, /*force_exhaust=*/ false);
}

// ---------------------------------------------------------------------------
// 3. before_flush
// ---------------------------------------------------------------------------

/// Fire enchantment `before_flush` immediately before the end-of-turn
/// hand flush, per `combat.py:1024`. Walks the player hand and lets
/// SlumberingEssence-marked cards tick their cost down before they
/// move to discard.
///
/// Python entry: `enchantments.py:137` — `before_flush(owner, combat)`.
pub fn dispatch_before_flush(state: &mut CombatState) {
    if !pile_has_any_enchantments(state, &state.player.hand) {
        return;
    }
    before_flush_stub(state);
}

/// Listener body.
///
/// **Wave 4 (`sts2-rs-uafi0`):** SlumberingEssence cost-tick. For each
/// card in the player hand carrying SlumberingEssence, take 1 energy off
/// its cost UNTIL PLAYED before the end-of-turn flush. C#
/// `SlumberingEssence.BeforeFlush` does `Card.EnergyCost.AddUntilPlayed(-1)`
/// (`SlumberingEssence.cs`), a `Relative`, `WhenPlayed` local modifier — so
/// the discounts stack while the card waits and all expire when it is played
/// (a permanent `cost -= 1` kept them through every later redraw). Banked in
/// `CardEnergyCost::add_until_played` modifier; the evaluated cost
/// floors at 0. X-cost cards are skipped: C# `GetWithModifiers` ignores local
/// modifiers on a `CostsX` card.
#[inline]
fn before_flush_stub(state: &mut CombatState) {
    let hand: Vec<CardUid> = state.player.hand.iter().copied().collect();
    for uid in hand {
        let idx = (uid.get() as usize).saturating_sub(1);
        let Some(card) = state.cards.get(idx) else {
            continue;
        };
        if !card.has_enchantment(EnchantmentId::SlumberingEssence) {
            continue;
        }
        if card.is_x_cost() {
            continue;
        }
        state.add_card_cost_reduction_until_played(uid, 1);
    }
}

// ---------------------------------------------------------------------------
// 4. modify_shuffle_order
// ---------------------------------------------------------------------------

/// Fire enchantment `modify_shuffle_order` on a freshly-shuffled card
/// list. PerfectFit-marked cards float to the top.
///
/// Python entry: `enchantments.py:149` —
/// `modify_shuffle_order(cards, *, is_initial_shuffle)`.
///
/// `is_initial_shuffle` is `true` for the combat-start shuffle (which
/// Python skips — PerfectFit only applies to mid-combat reshuffles).
/// The surface stub respects the Python early-out semantics (no-op
/// when initial) so the dispatch site can fire unconditionally.
///
/// `listener_order` is the pre-shuffle pile sequence (draw pile, then
/// discard pile): the order C# `IterateCombatHookListeners` visits the cards'
/// enchantments (`PlayerCombatState.AllPiles` = Hand, Draw, Discard, Exhaust,
/// Play, each in pile order; the hand is empty or already moved into the draw
/// pile at every shuffle site).
pub fn dispatch_modify_shuffle_order(
    player: &mut PlayerCombatState,
    cards: &mut [CardUid],
    is_initial_shuffle: bool,
    state_cards: &[Card],
    listener_order: &[CardUid],
) {
    if is_initial_shuffle {
        return;
    }
    // Empty-bus early-out: nothing to reorder if no card carries an
    // enchantment.
    let any_enchanted = cards.iter().any(|uid| {
        state_cards
            .get((uid.get() as usize).saturating_sub(1))
            .is_some_and(|c| !c.enchantments.is_empty())
    });
    if !any_enchanted {
        return;
    }
    modify_shuffle_order_stub(player, cards, state_cards, listener_order);
}

/// Listener body.
///
/// PerfectFit re-ordering, C# `PerfectFit.ModifyShuffleOrder`
/// (`Models.Enchantments/PerfectFit.cs`): each PerfectFit card in the shuffled
/// list removes itself and re-inserts at index 0 (the top), one listener at a
/// time in `listener_order`. With several PerfectFit cards the LAST one
/// visited ends on top, so they come out in reverse listener order, ahead of
/// every other card (whose shuffled order is untouched). This replaced a
/// stable partition (PerfectFit cards in shuffled order), which matched C#
/// only for a single PerfectFit card (GH #1021).
#[inline]
fn modify_shuffle_order_stub(
    _player: &mut PlayerCombatState,
    cards: &mut [CardUid],
    state_cards: &[Card],
    listener_order: &[CardUid],
) {
    let has_perfect_fit = |uid: &CardUid| {
        state_cards
            .get((uid.get() as usize).saturating_sub(1))
            .is_some_and(|c| c.has_enchantment(EnchantmentId::PerfectFit))
    };
    for uid in listener_order.iter().filter(|u| has_perfect_fit(u)) {
        if let Some(pos) = cards.iter().position(|c| c == uid) {
            // `Remove` + `Insert(0, ..)`: rotate the prefix right by one.
            cards[..=pos].rotate_right(1);
        }
    }
}

// ---------------------------------------------------------------------------
// 5. on_card_played
// ---------------------------------------------------------------------------

/// Fire enchantment `on_card_played` for a card whose own effect has just
/// finished resolving. Adroit gain block, Swift draw, Sown energy,
/// Corrupted self-damage, Momentum bonus accumulation all live here.
///
/// **Dispatch site — C#, not Python (bead `sts2-rs-ench-onplay-order`).**
/// C# `CardModel.OnPlayWrapper` (`CardModel.cs:1930-1943`) awaits the card's
/// own `OnPlay` first and only then runs `Enchantment.OnPlay`; the affliction
/// leg follows, and `Hook.AfterCardPlayed` after that. The port originally
/// fired this channel BEFORE the card-effect dispatch because it followed
/// Python's `enchantments.on_card_played` site — which put Swift's draw ahead
/// of a card body's own `CardSelectCmd.FromHand` and corrupted every
/// subsequent option index (BURNING_PACT; mirror corpus records 269/270). The
/// call now lives at the top of [`crate::combat_loop::finish_card_play`], the
/// single point both the inline and surfaced-choice-resume paths pass through.
///
/// Python entry (site NO LONGER followed): `enchantments.py:158` —
/// `on_card_played(card, combat)`.
pub fn dispatch_on_card_played(state: &mut CombatState, card_uid: CardUid) {
    let Some(card) = card_for(state, card_uid) else {
        return;
    };
    if card.enchantments.is_empty() {
        return;
    }
    // `Enchantment.OnPlay` runs for THIS card (`base.Card`), but it fires from
    // `finish_card_play`, after `dispatch_for_card` restored the enclosing
    // resolving card. Name this card for the duration so card-keyed listeners
    // see it: Corrupted's self-damage inside a Cascade auto-play is owed to
    // the auto-played card in `RupturePower.playedCards`, not to Cascade
    // (act23 Ironclad-4134 act 2 @558). A hand play has no enclosing card, so
    // the `play.last()` fallback already named it: byte-neutral there.
    let prev = state.resolving_card.replace(card_uid);
    on_card_played_stub(state, card_uid);
    state.resolving_card = prev;
}

/// Per-card listener body.
///
/// **Wave 2 (`sts2-rs-vkpir`) — dynamic damage triplet wired:**
///   - `Momentum` — stash `+N` (the enchantment stack) onto the card's
///     `VarKey::MomentumDamage` slot. Read back by
///     `enchantments::enchant_damage_additive` at damage-calc time so
///     subsequent plays of THIS card see the accumulated bonus.
///   - `Corrupted` — deal `2` self-damage with
///     `UNBLOCKABLE | UNPOWERED | MOVE` (Python pin
///     `enchantments.py:on_card_played` calls
///     `combat.deal_damage(target=owner, amount=2, props=…)`).
///   - `Sharp` / `Favored` / `Vigorous` — query-only enchantments; their
///     "real bodies" live in the damage-math helpers in
///     `sts2-content::enchantments`. No `on_card_played` wiring needed.
///
/// **Wave 3 (`sts2-rs-g7rni`) — block / draw / energy wired:**
///   - `Adroit` — gain `+N` player block (routes through
///     `card_effects::gain_block_player` so the full modify_block chain
///     fires).
///   - `Swift` — draw `+N` cards on first play this combat
///     (`SWIFT_USED_BIT` gates subsequent plays of the same card).
///   - `Sown` — gain `+N` energy on first play this combat
///     (`SOWN_USED_BIT` gates subsequent plays).
///   - `Nimble` / `Spiral` / `Glam` — query-only or after_card_played
///     wiring; inert here by design.
#[inline]
fn on_card_played_stub(state: &mut CombatState, card_uid: CardUid) {
    // Snapshot enchantment counters BEFORE any mutation — borrow
    // discipline: the immutable borrow of `card_for` dies before we
    // call `apply_damage_to_player` / `gain_block_player` (which
    // need `&mut state`).
    // v0.113.0: a disable pending from an earlier play iteration has landed
    // by now (its `AfterCardPlayed` ran).
    if state.game_version >= crate::GameVersion::V0_113_0 {
        let idx = (card_uid.get() as usize).saturating_sub(1);
        if let Some(card) = state.cards.get_mut(idx) {
            let bits = card.combat_var(VarKey::EnchantmentUsed);
            let mut landed = bits & !(SWIFT_PENDING_BIT | SOWN_PENDING_BIT);
            if bits & SWIFT_PENDING_BIT != 0 {
                landed |= SWIFT_USED_BIT;
            }
            if bits & SOWN_PENDING_BIT != 0 {
                landed |= SOWN_USED_BIT;
            }
            if landed != bits {
                card.set_combat_var(VarKey::EnchantmentUsed, landed);
            }
        }
    }
    let disable_later = state.game_version >= crate::GameVersion::V0_113_0;
    let (
        card_id,
        momentum_amount,
        has_corrupted,
        adroit_amount,
        swift_amount,
        sown_amount,
        used_bits,
    ) = {
        let Some(card) = card_for(state, card_uid) else {
            return;
        };
        (
            card.card_id,
            card.enchantment(EnchantmentId::Momentum),
            card.has_enchantment(EnchantmentId::Corrupted),
            card.enchantment(EnchantmentId::Adroit),
            card.enchantment(EnchantmentId::Swift),
            card.enchantment(EnchantmentId::Sown),
            card.combat_var(VarKey::EnchantmentUsed),
        )
    };

    if momentum_amount > 0 {
        let idx = (card_uid.get() as usize).saturating_sub(1);
        if let Some(card) = state.cards.get_mut(idx) {
            let prior = card.combat_var(VarKey::MomentumDamage);
            card.set_combat_var(
                VarKey::MomentumDamage,
                prior.saturating_add(momentum_amount),
            );
        }
    }

    // C# `Corrupted.OnPlay`: `CreatureCmd.Damage(owner, 2, Unblockable |
    // Unpowered | Move, base.Card)` — the card is the damage's `cardSource`
    // (GH #1021), so card-keyed damage listeners see it.
    if has_corrupted {
        let _ = crate::combat_loop::apply_damage_to_player(
            state,
            CreatureRef::Player,
            2,
            ValueProps::UNBLOCKABLE
                .union(ValueProps::UNPOWERED)
                .union(ValueProps::MOVE),
            Some(card_id),
        );
    }

    // Adroit: gain N block per stack. Python pin calls
    // `calculate_block(N, owner, ValueProp.MOVE, …)` then
    // `_gain_block` — our `gain_block_player` is the canonical wrapper
    // that runs the modify_block chain + fires after_block_gained.
    if adroit_amount > 0 {
        crate::card_effects::gain_block_player(state, adroit_amount);
    }

    // Swift: draw N cards on first play per combat. C# `Swift.OnPlay`
    // (Swift.cs:15-22) sets `Status = Disabled` right here, inside the play —
    // not in `AfterCardPlayed` — so a Music Box `CreateClone()` made during
    // `Hook.AfterCardPlayed` copies the spent status (EnchantmentModel.cs:
    // 309-314 keeps `_status`). v0.111.0 disables before the draw, v0.107.1
    // after it; the draw cannot reach this card (it is on the Play pile), so
    // one order serves both.
    if swift_amount > 0 && (used_bits & SWIFT_USED_BIT) == 0 {
        mark_used(
            state,
            card_uid,
            if disable_later {
                SWIFT_PENDING_BIT
            } else {
                SWIFT_USED_BIT
            },
        );
        // C# `Swift.OnPlay` → `CardPileCmd.Draw(fromHandDraw = false)` —
        // vetoed by NoDraw/Fiddle like every card-effect draw.
        let _ = state.draw_into_hand_card_effect(swift_amount);
    }

    // Sown: gain N energy on first play per combat. Same first-play gate and
    // in-play disable as Swift (Sown.cs:13-20).
    if sown_amount > 0 && (used_bits & SOWN_USED_BIT) == 0 {
        mark_used(
            state,
            card_uid,
            if disable_later {
                SOWN_PENDING_BIT
            } else {
                SOWN_USED_BIT
            },
        );
        state.gain_energy(sown_amount);
    }
}

/// Set a first-play `*_USED_BIT` on the card (C# `Status = Disabled`).
fn mark_used(state: &mut CombatState, card_uid: CardUid, bit: i32) {
    let idx = (card_uid.get() as usize).saturating_sub(1);
    if let Some(card) = state.cards.get_mut(idx) {
        let bits = card.combat_var(VarKey::EnchantmentUsed);
        card.set_combat_var(VarKey::EnchantmentUsed, bits | bit);
    }
}

// ---------------------------------------------------------------------------
// 6. after_card_played
// ---------------------------------------------------------------------------

/// Fire enchantment `after_card_played` for a card whose effect has
/// fully resolved. Marks the after-played first-play enchantments
/// (Vigorous / Glam) as consumed and ticks Goopy's amount.
///
/// Python entry: `enchantments.py:184` — `after_card_played(card)`.
pub fn dispatch_after_card_played(state: &mut CombatState, card_uid: CardUid) {
    let Some(card) = card_for(state, card_uid) else {
        return;
    };
    if card.enchantments.is_empty() {
        return;
    }
    after_card_played_stub(state, card_uid);
}

/// Per-card listener body.
///
/// **Wave 2 (`sts2-rs-vkpir`):** Vigorous-used bit.
///
/// **Wave 3 (`sts2-rs-g7rni`):** Glam consumed marker. Swift / Sown set
/// theirs in [`on_card_played_stub`], as C# disables them inside `OnPlay`.
///
/// **GH #952:** Goopy's amount grows by one per play (C# `Goopy.AfterCardPlayed`:
/// `base.Amount++`), so each later play's [`enchant_block_additive`] reads the
/// grown value. The deck copy's half (`DeckVersion.Enchantment.Amount++`) is
/// folded back at `exit_combat` (sts2-content `run_bridge`).
#[inline]
fn after_card_played_stub(state: &mut CombatState, card_uid: CardUid) {
    let idx = (card_uid.get() as usize).saturating_sub(1);
    let Some(card) = state.cards.get_mut(idx) else {
        return;
    };
    let mut bits = card.combat_var(VarKey::EnchantmentUsed);
    if card.has_enchantment(EnchantmentId::Vigorous) {
        bits |= VIGOROUS_USED_BIT;
    }
    if card.has_enchantment(EnchantmentId::Glam) {
        bits |= GLAM_USED_BIT;
    }
    // Swift / Sown disable inside their own `OnPlay` (see
    // `on_card_played_stub`); only Vigorous and Glam wait for AfterCardPlayed.
    card.set_combat_var(VarKey::EnchantmentUsed, bits);
    if card.has_enchantment(EnchantmentId::Goopy) {
        let amount = card.enchantment(EnchantmentId::Goopy);
        card.set_enchantment(EnchantmentId::Goopy, amount.saturating_add(1));
    }
}

// ---------------------------------------------------------------------------
// Tests — empty-bus early-out + dispatch reachability per channel
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{Card, CardId, CardType, EnchantmentId, TargetType};
    use crate::CardUid;

    use std::num::NonZeroU32;

    fn uid(n: u32) -> CardUid {
        CardUid(NonZeroU32::new(n).expect("non-zero"))
    }

    fn make_card(n: u32, with_enchantment: bool) -> Card {
        let mut c = Card {
            uid: uid(n),
            card_id: CardId::StrikeIronclad,
            cost: crate::CardEnergyCost::new(1),
            card_type: CardType::Attack,
            target_type: TargetType::Enemy,
            upgraded: false,
            base_damage: 6,
            base_block: 0,
            ..Card::pod_defaults()
        };
        if with_enchantment {
            c.set_enchantment(EnchantmentId::Sharp, 2);
        }
        c
    }

    /// Helper: build a minimal CombatState with `n` cards in `cards`
    /// and a hand+draw that mirrors them.
    fn state_with_cards(specs: &[(u32, bool)]) -> CombatState {
        let mut state = CombatState::default();
        for (n, ench) in specs {
            state.cards.push(make_card(*n, *ench));
            state.player.hand.push(uid(*n));
            state.player.draw.push(uid(*n));
        }
        state
    }

    /// The pre-shuffle listener order a shuffle site would pass (draw, then
    /// discard).
    fn snapshot_listeners(state: &CombatState) -> Vec<CardUid> {
        state
            .player
            .draw
            .iter()
            .chain(state.player.discard.iter())
            .copied()
            .collect()
    }

    /// Empty-bus on every channel: a state with no enchantments
    /// dispatches without panic and is observably a no-op.
    #[test]
    fn empty_bus_early_out_all_channels() {
        let mut state = state_with_cards(&[(1, false), (2, false)]);
        let snapshot_hand = state.player.hand.clone();
        let snapshot_draw = state.player.draw.clone();

        dispatch_on_card_drawn(&mut state, uid(1), true);
        dispatch_after_player_turn_start(&mut state);
        dispatch_before_flush(&mut state);
        let cards_snapshot = state.cards.clone();
        let mut shuffle_buf = state.player.draw.clone();
        let mut player = state.player.clone();
        dispatch_modify_shuffle_order(
            &mut player,
            &mut shuffle_buf,
            false,
            &cards_snapshot,
            &snapshot_listeners(&state),
        );
        dispatch_on_card_played(&mut state, uid(1));
        dispatch_after_card_played(&mut state, uid(1));

        // Empty-bus path is a no-op: hand and draw orders are unchanged.
        assert_eq!(state.player.hand, snapshot_hand);
        assert_eq!(state.player.draw, snapshot_draw);
        assert_eq!(shuffle_buf.as_slice(), snapshot_draw.as_slice());
    }

    /// Initial shuffle is always a no-op even when enchantments
    /// exist (mirrors Python's `is_initial_shuffle` early-out).
    #[test]
    fn modify_shuffle_order_initial_shuffle_skips() {
        let state = state_with_cards(&[(1, true), (2, false)]);
        let snapshot = state.player.draw.clone();
        let cards_snapshot = state.cards.clone();
        let mut shuffle_buf = state.player.draw.clone();
        let mut player = state.player.clone();
        dispatch_modify_shuffle_order(
            &mut player,
            &mut shuffle_buf,
            true, // is_initial_shuffle
            &cards_snapshot,
            &snapshot,
        );
        assert_eq!(shuffle_buf.as_slice(), snapshot.as_slice());
    }

    /// Each channel reaches its per-card stub when at least one
    /// enchantment is present (no early-out). Surface bead: stubs are
    /// no-ops, so we assert reachability via "no panic + empty-bus
    /// guard does not trip".
    #[test]
    fn populated_bus_reaches_stubs_without_panic() {
        let mut state = state_with_cards(&[(1, true)]);
        dispatch_on_card_drawn(&mut state, uid(1), true);
        dispatch_after_player_turn_start(&mut state);
        dispatch_before_flush(&mut state);
        let cards_snapshot = state.cards.clone();
        let mut shuffle_buf = state.player.draw.clone();
        let mut player = state.player.clone();
        dispatch_modify_shuffle_order(
            &mut player,
            &mut shuffle_buf,
            false,
            &cards_snapshot,
            &snapshot_listeners(&state),
        );
        dispatch_on_card_played(&mut state, uid(1));
        dispatch_after_card_played(&mut state, uid(1));
    }

    /// Out-of-range uid is a defensive no-op (does not panic).
    #[test]
    fn unknown_card_uid_is_safe() {
        let mut state = state_with_cards(&[(1, true)]);
        dispatch_on_card_drawn(&mut state, uid(99), true);
        dispatch_on_card_played(&mut state, uid(99));
        dispatch_after_card_played(&mut state, uid(99));
    }

    // -----------------------------------------------------------------
    // vkpir (vxvvs Wave 2): dynamic damage triplet per-variant pins
    // -----------------------------------------------------------------

    /// Build a single-card state with `ench` stacked at `amount`. The
    /// card lives in `state.cards[0]` and is wired into a player with a
    /// real creature_uid so `apply_damage_to_player` can route through
    /// the Creature mirror.
    fn state_with_one_enchanted_card(ench: EnchantmentId, amount: i32) -> CombatState {
        let mut state = CombatState::default();
        let mut c = Card {
            uid: uid(1),
            card_id: CardId::StrikeIronclad,
            cost: crate::CardEnergyCost::new(1),
            card_type: CardType::Attack,
            target_type: TargetType::Enemy,
            upgraded: false,
            base_damage: 6,
            base_block: 0,
            ..Card::pod_defaults()
        };
        c.set_enchantment(ench, amount);
        state.cards.push(c);
        state.player.hand.push(uid(1));
        // Mint a player Creature so apply_damage_to_player can resolve
        // the target via state.player.creature_uid.
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state
    }

    /// Bead `sts2-rs-enchant-damage-additive` (PARITY136): a `Sharp(2)`
    /// enchanted attack on top of the play pile gets +2 through the damage
    /// pipeline for a powered hit, and nothing for an unpowered hit. Before
    /// the fix the enchantment additive was computed but never threaded into
    /// `modify_damage`, so a Sharp(+2) Strike auto-played by Uproar dealt 6
    /// in Rust vs 8 in C#.
    #[test]
    fn sharp_enchant_threads_into_modify_damage_for_player_card() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Sharp, 2);
        // The damage pipeline reads the player's currently-resolving card from
        // the top of the play pile (the originating-card convention).
        state.player.hand.clear();
        state.player.play.push(uid(1));
        let powered = crate::hooks::DamageCtx {
            dealer: crate::hooks::CreatureRef::Player,
            target: crate::hooks::CreatureRef::Player,
            props: ValueProps::MOVE,
        };
        assert_eq!(
            crate::hooks::modify_damage(&state, 6, &powered),
            8,
            "Sharp(2) must add +2 to a powered player-card attack"
        );
        let unpowered = crate::hooks::DamageCtx {
            props: ValueProps::UNPOWERED,
            ..powered
        };
        assert_eq!(
            crate::hooks::modify_damage(&state, 6, &unpowered),
            6,
            "Sharp does not apply to unpowered hits"
        );
        // No player card on play → no enchant bonus (e.g. enemy attack path).
        state.player.play.clear();
        assert_eq!(crate::hooks::modify_damage(&state, 6, &powered), 6);
    }

    /// C# `Hook.ModifyDamage` applies the card's enchantment stage to the base
    /// BEFORE any power (Hook.cs:1630-1644): an Instinct (2x) Strike under -2
    /// Strength deals `6 * 2 - 2 = 10`, not `(6 - 2) * 2 = 8`. Necrobinder
    /// act23 seed 1001.
    #[test]
    fn instinct_doubles_the_base_before_strength_applies() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Instinct, 1);
        state.player.hand.clear();
        state.player.play.push(uid(1));
        let strength = std::num::NonZeroU32::new(99).expect("nz");
        state.powers.push(crate::Power {
            id: strength,
            kind: crate::PowerKind::Strength,
            amount: -2,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(strength));
        let ctx = crate::hooks::DamageCtx {
            dealer: crate::hooks::CreatureRef::Player,
            target: crate::hooks::CreatureRef::Player,
            props: ValueProps::MOVE,
        };
        assert_eq!(crate::hooks::modify_damage(&state, 6, &ctx), 10);
    }

    /// A relic's `ModifyDamageAdditive` (MiniatureCannon +3) runs in
    /// `ModifyDamageInternal`, after the enchantment stage, so Corrupted does
    /// not multiply it: `13 * 1.5 + 3 = 22.5 -> 22`, not `(13 + 3) * 1.5 = 24`.
    /// Regent 1004 (GH #1054).
    #[test]
    fn corrupted_does_not_multiply_a_relic_additive() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Corrupted, 1);
        state.player.hand.clear();
        state.player.play.push(uid(1));
        let ctx = crate::hooks::DamageCtx {
            dealer: crate::hooks::CreatureRef::Player,
            target: crate::hooks::CreatureRef::Player,
            props: ValueProps::MOVE,
        };
        assert_eq!(
            crate::hooks::modify_damage_with_relic_additive(&state, 13, 3, &ctx),
            22
        );
    }

    /// GH #64 §5: the Favored multiplicative (2×) is now threaded into
    /// `modify_damage` for a player card on the play pile. A powered hit
    /// doubles; an unpowered hit is unaffected (C# gates on
    /// `props.IsPoweredAttack()`).
    #[test]
    fn favored_enchant_doubles_damage_through_modify_damage() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Favored, 1);
        state.player.hand.clear();
        state.player.play.push(uid(1));
        let powered = crate::hooks::DamageCtx {
            dealer: crate::hooks::CreatureRef::Player,
            target: crate::hooks::CreatureRef::Player,
            props: ValueProps::MOVE,
        };
        assert_eq!(
            crate::hooks::modify_damage(&state, 6, &powered),
            12,
            "Favored doubles a powered player-card attack"
        );
        let unpowered = crate::hooks::DamageCtx {
            props: ValueProps::UNPOWERED,
            ..powered
        };
        assert_eq!(
            crate::hooks::modify_damage(&state, 6, &unpowered),
            6,
            "Favored does not multiply an unpowered hit"
        );
    }

    /// GH #64 §5: Corrupted's 1.5× multiplicative threads into `modify_damage`
    /// with the single-floor convention (`floor(6 * 1.5) = 9`).
    #[test]
    fn corrupted_enchant_one_point_five_x_through_modify_damage() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Corrupted, 1);
        state.player.hand.clear();
        state.player.play.push(uid(1));
        let powered = crate::hooks::DamageCtx {
            dealer: crate::hooks::CreatureRef::Player,
            target: crate::hooks::CreatureRef::Player,
            props: ValueProps::MOVE,
        };
        assert_eq!(crate::hooks::modify_damage(&state, 6, &powered), 9);
    }

    /// GH #64 §5: Nimble's flat block additive is now threaded into
    /// `modify_block` for a player-targeted block grant sourced from the
    /// resolving card.
    #[test]
    fn nimble_enchant_adds_block_through_modify_block() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Nimble, 4);
        state.player.hand.clear();
        state.player.play.push(uid(1));
        let ctx = crate::hooks::BlockCtx {
            target: crate::hooks::CreatureRef::Player,
            props: ValueProps::MOVE,
            card_source: Some(uid(1)),
        };
        assert_eq!(
            crate::hooks::modify_block(&state, 5, &ctx),
            9,
            "Nimble(4) adds +4 to a powered player block grant"
        );
        // GH #1021: C# reads the grant's own `cardSource`. Block a power or
        // relic grants while the Nimble card resolves (Feel No Pain inside a
        // Nimble True Grit) has none, so it gets no bonus.
        let sourceless = crate::hooks::BlockCtx {
            card_source: None,
            ..ctx
        };
        assert_eq!(crate::hooks::modify_block(&state, 5, &sourceless), 5);
        // Block landing on a non-player target gets no card enchant bonus.
        let enemy_uid = crate::CreatureUid(NonZeroU32::new(2).expect("non-zero"));
        let enemy_ctx = crate::hooks::BlockCtx {
            target: crate::hooks::CreatureRef::Enemy(enemy_uid),
            ..ctx
        };
        assert_eq!(crate::hooks::modify_block(&state, 5, &enemy_ctx), 5);
    }

    /// v0.107.1 (§8): `Nimble`/`Goopy` `EnchantBlockAdditive` lost its
    /// `IsPoweredCardOrMonsterMoveBlock` guard, so the block bonus now lands
    /// on an UNPOWERED block grant too (patch note: Entrench gains extra Block
    /// from Nimble — Entrench re-grants block via an unpowered Move). Before
    /// the change a `ValueProps::UNPOWERED` grant returned `+0`.
    #[test]
    fn nimble_enchant_adds_block_even_when_unpowered_v0107() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Nimble, 4);
        state.player.hand.clear();
        state.player.play.push(uid(1));
        let unpowered = crate::hooks::BlockCtx {
            target: crate::hooks::CreatureRef::Player,
            props: ValueProps::UNPOWERED,
            card_source: Some(uid(1)),
        };
        assert_eq!(
            crate::hooks::modify_block(&state, 5, &unpowered),
            9,
            "Nimble(4) adds +4 even on an unpowered block grant (v0.107.1 guard removed)"
        );
    }

    /// v0.107.1 (§8) NEW enchant `TezcatarasEmber`: +3 on a powered attack
    /// threads through `modify_damage` for the resolving player card; an
    /// unpowered hit is unaffected (`props.IsPoweredAttack()` gate).
    #[test]
    fn tezcataras_ember_threads_three_into_modify_damage() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::TezcatarasEmber, 1);
        state.player.hand.clear();
        state.player.play.push(uid(1));
        let powered = crate::hooks::DamageCtx {
            dealer: crate::hooks::CreatureRef::Player,
            target: crate::hooks::CreatureRef::Player,
            props: ValueProps::MOVE,
        };
        assert_eq!(
            crate::hooks::modify_damage(&state, 6, &powered),
            6 + TEZCATARAS_EMBER_DAMAGE,
            "TezcatarasEmber adds +3 to a powered player-card attack"
        );
        let unpowered = crate::hooks::DamageCtx {
            props: ValueProps::UNPOWERED,
            ..powered
        };
        assert_eq!(
            crate::hooks::modify_damage(&state, 6, &unpowered),
            6,
            "TezcatarasEmber does not apply to an unpowered hit"
        );
    }

    /// Momentum: each play stashes `+stack` onto the MomentumDamage
    /// slot so subsequent plays of this same card see the accumulated
    /// bonus. Two plays of Momentum(3) → stash = 6.
    #[test]
    fn momentum_on_card_played_accumulates_rampage_damage() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Momentum, 3);
        assert_eq!(state.cards[0].combat_var(VarKey::MomentumDamage), 0);
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.cards[0].combat_var(VarKey::MomentumDamage), 3);
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.cards[0].combat_var(VarKey::MomentumDamage), 6);
    }

    /// Corrupted: dealing the card fires 2 UNBLOCKABLE self-damage on
    /// the player. With block=99, the unblockable bypass means HP
    /// drops by exactly 2 (Python `combat.py` props mask matches).
    #[test]
    fn corrupted_on_card_played_deals_two_unblockable_self_damage() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Corrupted, 1);
        state.player.block = 99;
        let hp_before = state.player.current_hp;
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(
            state.player.current_hp,
            hp_before - 2,
            "Corrupted self-damage is UNBLOCKABLE — block should not absorb",
        );
        assert_eq!(state.player.block, 99, "block untouched by UNBLOCKABLE");
    }

    /// Vigorous: after_card_played sets the per-combat used bit on the
    /// card's VarKey::EnchantmentUsed slot. The encoding mirrors
    /// `sts2-content::enchantments::mark_vigorous_used` (bit 0).
    #[test]
    fn vigorous_after_card_played_marks_used_bit() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Vigorous, 5);
        assert_eq!(state.cards[0].combat_var(VarKey::EnchantmentUsed) & 0b01, 0);
        dispatch_after_card_played(&mut state, uid(1));
        assert_eq!(
            state.cards[0].combat_var(VarKey::EnchantmentUsed) & 0b01,
            0b01,
            "Vigorous-used bit set after first play resolves",
        );
        // Idempotent: a second after_card_played leaves the bit set.
        dispatch_after_card_played(&mut state, uid(1));
        assert_eq!(
            state.cards[0].combat_var(VarKey::EnchantmentUsed) & 0b01,
            0b01
        );
    }

    /// Sharp: pure query-side enchantment; on_card_played MUST NOT
    /// stash anything onto the card or mutate the player. The damage
    /// bonus is read by `enchantments::enchant_damage_additive` at
    /// damage-calc time.
    #[test]
    fn sharp_on_card_played_is_inert() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Sharp, 3);
        let hp_before = state.player.current_hp;
        let momentum_before = state.cards[0].combat_var(VarKey::MomentumDamage);
        let generic_before = state.cards[0].combat_var(VarKey::EnchantmentUsed);
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.current_hp, hp_before);
        assert_eq!(
            state.cards[0].combat_var(VarKey::MomentumDamage),
            momentum_before
        );
        assert_eq!(
            state.cards[0].combat_var(VarKey::EnchantmentUsed),
            generic_before
        );
    }

    /// Favored: pure query-side multiplier; on_card_played MUST NOT
    /// touch combat_vars or the player. Multiplier is consumed by
    /// `enchant_damage_multiplicative_x100`.
    #[test]
    fn favored_on_card_played_is_inert() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Favored, 1);
        let hp_before = state.player.current_hp;
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.current_hp, hp_before);
        assert_eq!(state.cards[0].combat_var(VarKey::MomentumDamage), 0);
        assert_eq!(state.cards[0].combat_var(VarKey::EnchantmentUsed), 0);
    }

    // -----------------------------------------------------------------
    // g7rni (vxvvs Wave 3): block / play-count / draw / energy
    // -----------------------------------------------------------------

    /// Adroit: gain N block on play. Goes through `gain_block_player`
    /// so the modify_block chain fires.
    #[test]
    fn adroit_on_card_played_gains_block() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Adroit, 7);
        assert_eq!(state.player.block, 0);
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.block, 7);
    }

    /// Swift: draw N on first play; subsequent plays of the same card
    /// instance are inert (consumed marker set in after_card_played).
    #[test]
    fn swift_first_play_draws_then_consumed() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Swift, 1);
        // Seed a second card into the draw pile so we can observe a draw.
        let mut filler = Card {
            uid: uid(2),
            card_id: CardId::StrikeIronclad,
            cost: crate::CardEnergyCost::new(1),
            card_type: CardType::Attack,
            target_type: TargetType::Enemy,
            upgraded: false,
            base_damage: 6,
            base_block: 0,
            ..Card::pod_defaults()
        };
        filler.set_enchantment(EnchantmentId::RoyallyApproved, 0); // empty stack — keeps `enchantments` non-empty? No.
        state.cards.push(filler);
        state.player.draw.push(uid(2));
        let hand_before = state.player.hand.len();
        let draw_before = state.player.draw.len();
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.hand.len(), hand_before + 1, "Swift drew 1");
        assert_eq!(state.player.draw.len(), draw_before - 1);
        // Set the consumed marker (Wave 3 after_card_played wiring).
        dispatch_after_card_played(&mut state, uid(1));
        assert_eq!(
            state.cards[0].combat_var(VarKey::EnchantmentUsed) & 0b0100,
            0b0100,
            "Swift-used bit set"
        );
        // Second play: no draw (gate held).
        let hand_now = state.player.hand.len();
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.hand.len(), hand_now);
    }

    /// Sown: gain N energy on first play; gated by SOWN_USED_BIT.
    #[test]
    fn sown_first_play_gains_energy_then_consumed() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Sown, 2);
        state.player.energy = 0;
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.energy, 2);
        dispatch_after_card_played(&mut state, uid(1));
        assert_eq!(
            state.cards[0].combat_var(VarKey::EnchantmentUsed) & 0b1000,
            0b1000,
            "Sown-used bit set"
        );
        // Second play is inert.
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.energy, 2);
    }

    /// C# `Sown.OnPlay` / `Swift.OnPlay` (Sown.cs:13-20, Swift.cs:15-22) set
    /// `Status = Disabled` INSIDE the play, before `Hook.AfterCardPlayed`, so a
    /// Music Box `CreateClone()` there (MusicBox.cs:73-84; the clone keeps
    /// `_status`, EnchantmentModel.cs:309-314) is already Disabled and grants
    /// nothing (ckpt_act3_8000 Ironclad-8148-a0-act3#b1 @900: energy cs=0 rs=1).
    #[test]
    fn sown_and_swift_disable_during_on_play_so_a_mid_play_clone_is_spent() {
        for (ench, bit) in [
            (EnchantmentId::Sown, SOWN_USED_BIT),
            (EnchantmentId::Swift, SWIFT_USED_BIT),
        ] {
            let mut state = state_with_one_enchanted_card(ench, 1);
            dispatch_on_card_played(&mut state, uid(1));
            assert_eq!(
                state.cards[0].combat_var(VarKey::EnchantmentUsed) & bit,
                bit,
                "{ench:?} is spent by its own OnPlay, before AfterCardPlayed"
            );
        }
        // The clone's play: energy stays put.
        let mut state = state_with_one_enchanted_card(EnchantmentId::Sown, 2);
        state.player.energy = 0;
        dispatch_on_card_played(&mut state, uid(1));
        let mut clone = state.cards[0].create_clone();
        clone.uid = uid(2);
        state.cards.push(clone);
        dispatch_on_card_played(&mut state, uid(2));
        assert_eq!(
            state.player.energy, 2,
            "the Disabled clone grants no energy"
        );
    }

    /// v0.113 `Sown`/`Swift` disable in `AfterCardPlayed`, not in `OnPlay`: a
    /// Music Box clone made earlier in that hook pass (relics before card
    /// enchantments) is still Normal and fires again; the original is spent.
    #[test]
    fn sown_v0113_disables_after_play_so_a_mid_play_clone_stays_normal() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Sown, 2);
        state.game_version = crate::GameVersion::V0_113_0;
        state.player.energy = 0;
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.energy, 2);
        assert_eq!(
            state.cards[0].combat_var(VarKey::EnchantmentUsed) & SOWN_USED_BIT,
            0,
            "not yet Disabled inside the play"
        );
        let mut clone = state.cards[0].create_clone();
        clear_pending_first_play_disable(&mut clone);
        clone.uid = uid(2);
        state.cards.push(clone);
        dispatch_after_card_played(&mut state, uid(1));
        dispatch_on_card_played(&mut state, uid(2));
        assert_eq!(state.player.energy, 4, "the Normal clone grants energy");
        // The original's next play is inert.
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.energy, 4);
    }

    /// Glam: after_card_played sets bit 1 (matches sts2-content
    /// encoding for `mark_glam_used`). On-played leg is inert.
    #[test]
    fn glam_after_card_played_marks_used_bit() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Glam, 1);
        // on_card_played is inert for Glam.
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.cards[0].combat_var(VarKey::EnchantmentUsed) & 0b10, 0);
        // after_card_played sets the bit.
        dispatch_after_card_played(&mut state, uid(1));
        assert_eq!(
            state.cards[0].combat_var(VarKey::EnchantmentUsed) & 0b10,
            0b10,
            "Glam-used bit set on after_card_played"
        );
    }

    /// Spiral: query-only play-count bonus; dispatch sites are inert.
    #[test]
    fn spiral_dispatch_is_inert() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Spiral, 2);
        let hp_before = state.player.current_hp;
        let energy_before = state.player.energy;
        let block_before = state.player.block;
        dispatch_on_card_played(&mut state, uid(1));
        dispatch_after_card_played(&mut state, uid(1));
        assert_eq!(state.player.current_hp, hp_before);
        assert_eq!(state.player.energy, energy_before);
        assert_eq!(state.player.block, block_before);
        // The used-bits combat_var must stay 0 — Spiral does not own any bit.
        assert_eq!(state.cards[0].combat_var(VarKey::EnchantmentUsed), 0);
    }

    /// Nimble: query-only block bonus; on_card_played is inert. The
    /// production wiring lives in the block-calc helper, not here.
    #[test]
    fn nimble_on_card_played_is_inert() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Nimble, 3);
        let block_before = state.player.block;
        dispatch_on_card_played(&mut state, uid(1));
        assert_eq!(state.player.block, block_before);
    }

    /// Slither: on draw, the card's cost is rerolled to a value in
    /// `[0, 3]`. The reroll is deterministic given a seeded
    /// `state.rng` (the divergence from Python's
    /// `combat_energy_costs_rng` is documented above).
    #[test]
    fn slither_on_card_drawn_rerolls_cost_in_zero_three() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Slither, 1);
        state.cards[0].cost = crate::CardEnergyCost::new(1); // starting cost
        dispatch_on_card_drawn(&mut state, uid(1), true);
        let new_cost = state.cards[0].local_cost();
        assert!(
            (0..=3).contains(&new_cost),
            "Slither cost roll out of range: {new_cost}"
        );
    }

    // -----------------------------------------------------------------
    // uafi0 (vxvvs Wave 4): stateful / boss enchantments
    // -----------------------------------------------------------------

    /// SlumberingEssence: before_flush takes 1 energy off each marked card
    /// in hand UNTIL PLAYED (C# `AddUntilPlayed(-1)`), stacking per flush;
    /// the evaluated cost floors at 0 and playing the card retires every
    /// discount, restoring the printed cost.
    #[test]
    fn slumbering_essence_before_flush_ticks_cost_down() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::SlumberingEssence, 1);
        state.cards[0].cost = crate::CardEnergyCost::new(3);
        // Card must be in hand for before_flush to consider it.
        // state_with_one_enchanted_card already pushes uid(1) to hand.
        let cost = |s: &CombatState| crate::hooks::effective_card_cost(s, uid(1));
        dispatch_before_flush(&mut state);
        assert_eq!(cost(&state), 2, "tick from 3 → 2");
        dispatch_before_flush(&mut state);
        assert_eq!(cost(&state), 1);
        dispatch_before_flush(&mut state);
        assert_eq!(cost(&state), 0, "saturates at 0");
        dispatch_before_flush(&mut state);
        assert_eq!(cost(&state), 0, "stays at 0 — no underflow");
        assert_eq!(state.cards[0].base_cost(), 3, "printed cost untouched");
        assert_eq!(
            state.cards[0].local_cost(),
            -1,
            "3 - 4, clamped only on read"
        );
        state.cards[0].cost.after_card_played_cleanup();
        assert_eq!(
            cost(&state),
            3,
            "played: every until-played discount expires"
        );
    }

    /// SlumberingEssence skips X-cost cards (cost == -1 sentinel).
    #[test]
    fn slumbering_essence_skips_x_cost_cards() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::SlumberingEssence, 1);
        state.cards[0].cost = crate::CardEnergyCost::new(-1); // X-cost sentinel
        dispatch_before_flush(&mut state);
        assert_eq!(state.cards[0].local_cost(), -1, "X-cost untouched");
    }

    /// SlumberingEssence on a non-hand-resident card is inert (the
    /// dispatch walks `state.player.hand` only).
    #[test]
    fn slumbering_essence_ignores_non_hand_cards() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::SlumberingEssence, 1);
        state.cards[0].cost = crate::CardEnergyCost::new(3);
        state.player.hand.clear();
        dispatch_before_flush(&mut state);
        assert_eq!(state.cards[0].local_cost(), 3, "non-hand card untouched");
    }

    /// PerfectFit: floats marked cards to the front while preserving
    /// relative order within each group (stable partition).
    #[test]
    fn perfect_fit_modify_shuffle_floats_marked_to_front() {
        let mut state = CombatState::default();
        // Build 4 cards: indices 1..=4. Cards 2 and 4 have PerfectFit.
        for n in 1..=4u32 {
            let mut c = Card {
                uid: uid(n),
                card_id: CardId::StrikeIronclad,
                cost: crate::CardEnergyCost::new(1),
                card_type: CardType::Attack,
                target_type: TargetType::Enemy,
                upgraded: false,
                base_damage: 6,
                base_block: 0,
                ..Card::pod_defaults()
            };
            if n == 2 || n == 4 {
                c.set_enchantment(EnchantmentId::PerfectFit, 1);
            }
            state.cards.push(c);
        }
        // C# `PerfectFit.ModifyShuffleOrder` moves each PerfectFit card to
        // index 0 in listener order (pre-shuffle pile order), so the last one
        // visited ends on top; the other cards keep their shuffled order.
        let mut player = state.player.clone();
        let mut shuffle = vec![uid(1), uid(2), uid(3), uid(4)];
        dispatch_modify_shuffle_order(
            &mut player,
            &mut shuffle,
            false,
            &state.cards,
            &[uid(2), uid(4)],
        );
        assert_eq!(shuffle, vec![uid(4), uid(2), uid(1), uid(3)]);
        let mut shuffle = vec![uid(1), uid(2), uid(3), uid(4)];
        dispatch_modify_shuffle_order(
            &mut player,
            &mut shuffle,
            false,
            &state.cards,
            &[uid(4), uid(2)],
        );
        assert_eq!(shuffle, vec![uid(2), uid(4), uid(1), uid(3)]);
        // A listener-order card that is not being shuffled does not move.
        let mut shuffle = vec![uid(1), uid(2), uid(3)];
        dispatch_modify_shuffle_order(
            &mut player,
            &mut shuffle,
            false,
            &state.cards,
            &[uid(4), uid(2)],
        );
        assert_eq!(shuffle, vec![uid(2), uid(1), uid(3)]);
    }

    /// PerfectFit with no marked cards leaves the order untouched.
    #[test]
    fn perfect_fit_modify_shuffle_no_marked_preserves_order() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Sharp, 1);
        let mut shuffle = vec![uid(1)];
        let mut player = state.player.clone();
        let original = shuffle.clone();
        dispatch_modify_shuffle_order(&mut player, &mut shuffle, false, &state.cards, &original);
        assert_eq!(shuffle, original, "no PerfectFit → no reorder");
        // Burn `state` so clippy doesn't moan about `let mut` going unused.
        state.player.hand.clear();
    }

    /// Clone (`@CLONE` token, no Python pin behavior at `01c094d`): a
    /// pure marker enchantment. Every dispatch channel that walks the
    /// card MUST be a no-op for it. Pins the documented Python-pin
    /// behavior so a future real port shows up here as a diff.
    #[test]
    fn clone_marker_is_inert_on_all_channels() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Clone, 1);
        let hp_before = state.player.current_hp;
        let energy_before = state.player.energy;
        let block_before = state.player.block;
        let cost_before = state.cards[0].cost;
        let generic_before = state.cards[0].combat_var(VarKey::EnchantmentUsed);

        dispatch_on_card_drawn(&mut state, uid(1), true);
        dispatch_after_player_turn_start(&mut state);
        dispatch_before_flush(&mut state);
        let mut shuffle = vec![uid(1)];
        let cards_snapshot = state.cards.clone();
        let mut player = state.player.clone();
        dispatch_modify_shuffle_order(&mut player, &mut shuffle, false, &cards_snapshot, &[uid(1)]);
        dispatch_on_card_played(&mut state, uid(1));
        dispatch_after_card_played(&mut state, uid(1));

        assert_eq!(state.player.current_hp, hp_before);
        assert_eq!(state.player.energy, energy_before);
        assert_eq!(state.player.block, block_before);
        assert_eq!(state.cards[0].cost, cost_before);
        assert_eq!(
            state.cards[0].combat_var(VarKey::EnchantmentUsed),
            generic_before
        );
        assert_eq!(shuffle, vec![uid(1)]);
    }

    /// Imbued: after_player_turn_start auto-plays each Imbued card
    /// from the draw pile on round 1. Round-gate verified by round 2
    /// no-op.
    #[test]
    fn imbued_after_player_turn_start_auto_plays_on_round_one_only() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Imbued, 1);
        // Move the Imbued card out of hand into draw pile only.
        state.player.hand.clear();
        // Force the round counter via internal setter equivalent: round
        // initializes to 1 in `CombatState::default()` via
        // `start_player_turn`; for this test we drive
        // `after_player_turn_start` directly, which checks `round_no()`.
        // CombatState::default() sets round=1 → first invocation runs.
        let draw_before = state.player.draw.len();
        dispatch_after_player_turn_start(&mut state);
        assert!(
            state.player.draw.len() < draw_before || !state.player.draw.contains(&uid(1)),
            "Imbued card should have been auto-played out of draw on round 1",
        );
    }

    /// C# hook listeners are every combat pile's enchantments, so an Imbued
    /// card already in the hand on turn 1 (a Power Cell pulled it there
    /// before the opening draw) is auto-played from the hand (loadout Defect
    /// 1235 v0.107.1).
    #[test]
    fn imbued_card_in_hand_is_auto_played_on_turn_one() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        let mut c = Card {
            uid: uid(1),
            card_id: CardId::DefendIronclad,
            cost: crate::CardEnergyCost::new(1),
            card_type: CardType::Skill,
            target_type: TargetType::Self_,
            base_block: 5,
            ..Card::pod_defaults()
        };
        c.set_enchantment(EnchantmentId::Imbued, 1);
        state.cards.push(c);
        state.player.hand.push(uid(1));

        dispatch_after_player_turn_start(&mut state);

        assert!(
            !state.player.hand.contains(&uid(1)),
            "auto-played out of the hand"
        );
        assert!(
            state.player.discard.contains(&uid(1)),
            "it lands in discard"
        );
    }

    /// GH #785: under BeliefDraw the auto-play pops the top through
    /// `cards::pop_top_card`, which samples an UNKNOWN top. The Imbued card
    /// is known, so it must be pinned on top first: the auto-play then takes
    /// exactly it, reads no RNG, and leaves the sorted middle intact.
    #[test]
    fn imbued_auto_play_under_belief_draw_takes_the_imbued_card_without_rng() {
        let mut state = state_with_one_enchanted_card(EnchantmentId::Imbued, 1);
        state.player.hand.clear();
        // Two plain cards (uids 2, 3) join the pile.
        for n in 2..=3 {
            state.cards.push(Card {
                uid: uid(n),
                card_id: CardId::DefendIronclad,
                cost: crate::CardEnergyCost::new(1),
                card_type: CardType::Skill,
                target_type: TargetType::Self_,
                ..Card::pod_defaults()
            });
        }
        // Move the enchantment from uid 1 to uid 3, so the Imbued card is the
        // HIGHEST uid of the canonical pile: taking the lowest-uid stand-in
        // (`draw[0]`) would play the wrong card.
        let imbued = state.cards[0].enchantments;
        state.cards[0].enchantments = crate::cards::CardEnchantments::new();
        state.cards[2].enchantments = imbued;
        state.player.draw = [uid(1), uid(2), uid(3)].into_iter().collect();
        state.player.deferred_draw_order = true;

        let before = state.run_rng.shuffle.counter();
        dispatch_after_player_turn_start(&mut state);
        assert!(
            !state.player.draw.contains(&uid(3)),
            "the Imbued card (not the lowest-uid stand-in) was auto-played"
        );
        assert_eq!(state.player.draw.as_slice(), &[uid(1), uid(2)]);
        assert!(state.player.pinned_top.is_empty(), "its pin was consumed");
        assert_eq!(
            state.run_rng.shuffle.counter(),
            before,
            "a known card on top is popped with no RNG read"
        );
    }
}
