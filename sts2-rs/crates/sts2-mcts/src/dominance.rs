//! Engine-proven move-order pruning ([`crate::MctsConfig::dominance_pruning`]).
//!
//! Two card plays `a` and `b` taken in either order within one turn spend
//! the same energy and consume the same hand. When playing `a` before `b`
//! reaches a state equal to `b`-then-`a` in everything except enemy HP,
//! and no enemy has more HP and at least one has less, the `b, a` order is
//! proven no better: the follow-up `a` under the edge `b` can be dropped.
//! This is the defect behind "Strike before Beam Cell": Root Q is a mean
//! over the subtree (GH #492), and a 0-cost debuff whose value is ordering
//! scores below its own dominated alternative in a narrow value band.
//!
//! The pruning is **path-conditional**: it removes `a` at the node reached
//! by `b` from its parent, not `a` everywhere. It looks back along the
//! turn's card plays, so `a` is also dropped after `b, c` when moving it in
//! front of both is proven better. The caller folds the pruned
//! set into the transposition key so a node reached by a different edge
//! never inherits a set computed for another parent.
//!
//! ## Why enemy HP is the only axis
//!
//! Measured over the harvested fixture pool (15 030 decision points),
//! 2 785 of 2 800 proven pairs win on enemy HP and none on player HP;
//! player HP and block each carry relic and power payoffs that break
//! "more is better" (RedSkull, MeatOnTheBone, Orichalcum, Rupture, …), so
//! they are required EQUAL instead of compared. Enemy HP is compared per
//! enemy, never summed: damage moved between enemies is not dominance.
//!
//! ## Where "lower enemy HP is better" is false (decompiled v0.107.1)
//!
//! [`enemy_hp_not_monotone`] disables pruning outright when an enemy's
//! behaviour reads its own HP or an ally's. The set of monsters and powers
//! with HP or death hooks is the same in v0.111.0.
//!
//! * HP-threshold events: `Monsters/FrogKnight.cs`, `Powers/ShriekPower.cs`
//!   (TerrorEel), `Powers/PlowPower.cs` (CeremonialBeast).
//! * Death triggers, which a lower-HP line reaches sooner:
//!   `Powers/{CrabRage,SteamEruption,Infested,Stock,Surprise,Adaptable,
//!   Reattach,Illusion,Ravenous}Power.cs`. Ravenous (CorpseSlug) stuns
//!   every surviving slug and gives it permanent Strength when an ally
//!   dies, the same shape as CrabRage.
//! * Ally-dependent AI: `Monsters/{Nibbit,LivingShield,Queen,Toadpole}.cs`,
//!   and the summoners whose next move counts living allies or a free
//!   slot: `Monsters/Fabricator.cs` (`CanFabricate`),
//!   `Monsters/Ovicopter.cs` (`CanLay`), `Monsters/TwoTailedRat.cs`
//!   (`CanSummon`). A kill that lands sooner turns an attack into a summon.
//! * `Monsters/WaterfallGiant.cs` is listed by monster as well as by its
//!   SteamEruption, because the power comes from PRESSURIZE, the giant's
//!   opening move, not from the spawn: the power entry alone misses player
//!   turn 1. The monster entry also covers the turn after ABOUT_TO_BLOW,
//!   where the power is gone and pruning would be sound; that is accepted
//!   to keep the guard a plain list.
//! * `Orbs/DarkOrb.cs` evokes on the lowest-HP enemy, so with two or more
//!   enemies a per-enemy HP change can retarget it.
//!
//! ### Known limitation: player-side kill rewards
//!
//! The guard reads enemy behaviour only. Some player content pays out when
//! a particular hit kills, and a lower-HP line can lose that payout: an
//! unrelated hit (forced damage, or a block or draw card that also deals
//! damage) kills the enemy before the reward card lands.
//!
//! * Cards that check `DamageResult.WasTargetKilled` on their own attack:
//!   `Cards/{Feed,Sunder,HandOfGreed,TheHunt,KnockoutBlow,EchoingSlash}.cs`.
//! * Relics that fire on a death: `Relics/GremlinHorn.cs` (energy and a
//!   draw on any enemy death, so the payout moves with the kill) and
//!   `Relics/BookRepairKnife.cs` (heal on a Doom kill).
//!
//! These are not listed, so a pair where the lower-HP line forfeits such a
//! reward can still be pruned. The effect has not been measured; measure
//! it before adding a player-side guard.
//!
//! ## Relaxed equality
//!
//! Whole-state equality rejects the motivating pair: the two orders leave
//! the discard pile in a different order and stamp a different
//! `last_attack_skill_played`. Both are ignored unless a reader exists
//! ([`order_is_observable`]):
//!
//! * The reshuffle sorts by `(id, upgrade)` before its Fisher–Yates
//!   (`Extensions/ListExtensions.cs` `StableShuffle`), and `a`, `b` never
//!   share an identity here, so their relative discard order cannot reach
//!   the draw pile.
//! * Readers that do see discard order: `Powers/AggressionPower.cs`,
//!   `Cards/{DrainPower,AllForOne,BeatDown}.cs`,
//!   `Monsters/ThievingHopper.cs`, the discard-to-hand hook listeners
//!   `Cards/{RightHandHand,MakeItSo,Bolas,ThrummingHatchet}.cs` and the
//!   `Enchantments/PerfectFit.cs` shuffle hook.
//! * The only reader of the last card played is `Relics/HistoryCourse.cs`.
//!
//! When any of these is present the discard order and last play are
//! compared too, which is still sound.
//!
//! The two states are compared by [`sts2_core::state_hash`]: every
//! game-observable field plus every RNG cursor, without the engine's
//! per-play scratch (`powered_hits_at_card_play_start` and friends, reset
//! before any read) that a derived `==` trips on. The transposition table
//! already treats states with equal [`sts2_core::search_hash`] — the same
//! walk minus the RNG cursors — as one node, so this is the stricter of
//! the two equivalences the search relies on.

use smallvec::SmallVec;
use sts2_core::cards::{CardId, EnchantmentId};
use sts2_core::{
    apply, legal_actions, Action, CombatPhase, CombatState, Orb, PowerKind, RelicKind, StepResult,
};

use crate::EdgeAction;

const HP_EVENT_MONSTERS: &[&str] = &[
    "FrogKnight",
    "TerrorEel",
    "CeremonialBeast",
    "Nibbit",
    "LivingShield",
    "Queen",
    "Toadpole",
    // Summon branches gated on living allies or a free slot.
    "Fabricator",   // Fabricator.cs:51 CanFabricate = alive teammates < 4
    "Ovicopter",    // Ovicopter.cs:46 CanLay = alive teammates <= 3
    "TwoTailedRat", // TwoTailedRat.cs:194 CanSummon = a free slot
    // SteamEruption arrives with PRESSURIZE (WaterfallGiant.cs:221-228), not
    // on spawn, so the power entry alone misses the turn before it.
    "WaterfallGiant",
];

const HP_EVENT_POWERS: &[PowerKind] = &[
    PowerKind::Shriek,
    PowerKind::Plow,
    PowerKind::CrabRage,
    PowerKind::SteamEruption,
    PowerKind::Infested,
    PowerKind::Stock,
    PowerKind::Surprise,
    PowerKind::Adaptable,
    PowerKind::Reattach,
    PowerKind::Illusion,
    // RavenousPower.cs:24-36: an ally's death stuns each CorpseSlug and
    // gives it Strength.
    PowerKind::Ravenous,
];

const ORDER_READER_CARDS: &[CardId] = &[
    CardId::DrainPower,
    CardId::AllForOne,
    CardId::BeatDown,
    CardId::RightHandHand,
    CardId::MakeItSo,
    CardId::Bolas,
    CardId::ThrummingHatchet,
];

pub(crate) fn enemies(s: &CombatState) -> impl Iterator<Item = &sts2_core::Creature> {
    s.enemy_uids
        .iter()
        .filter_map(|u| s.creatures.iter().find(|c| c.id == u.0))
}

/// `true` when lower enemy HP is not provably better in `s`.
pub fn enemy_hp_not_monotone(s: &CombatState) -> bool {
    if enemies(s).any(|c| HP_EVENT_MONSTERS.contains(&c.monster_id.0)) {
        return true;
    }
    if s.powers
        .as_slice()
        .iter()
        .any(|p| HP_EVENT_POWERS.contains(&p.kind))
    {
        return true;
    }
    let alive = enemies(s).filter(|c| c.current_hp > 0).count();
    alive > 1
        && s.player
            .orbs
            .slots
            .iter()
            .any(|o| matches!(o, Orb::Dark { .. }))
}

/// `true` when more player block is not provably better in `s`: Orichalcum
/// (and its fake) give block at the end of a turn that ends with none
/// (`Relics/{Orichalcum,FakeOrichalcum}.cs`), so a line with a little more
/// block can end the turn with less. Every other reader of the player's
/// block in the decompiled v0.107.1 content either scales with it (Body
/// Slam, Barricade, Juggernaut) or modifies a gain the same in either order
/// (Dexterity, Frail).
pub fn block_not_monotone(s: &CombatState) -> bool {
    s.relics.owns(RelicKind::Orichalcum) || s.relics.owns(RelicKind::FakeOrichalcum)
}

/// `true` when discard order or the last card played can change the game.
pub fn order_is_observable(s: &CombatState) -> bool {
    s.relics.owns(RelicKind::HistoryCourse)
        || enemies(s).any(|c| c.monster_id.0 == "ThievingHopper")
        || s.powers
            .as_slice()
            .iter()
            .any(|p| p.kind == PowerKind::Aggression)
        || s.cards.iter().any(|c| {
            ORDER_READER_CARDS.contains(&c.card_id)
                || c.enchantments
                    .iter()
                    .any(|(e, _)| e == EnchantmentId::PerfectFit)
        })
}

pub(crate) fn player_hp_block(s: &CombatState) -> (i32, i32) {
    s.player
        .creature_uid
        .and_then(|u| s.creatures.iter().find(|c| c.id == u.0))
        .map_or((s.player.current_hp, s.player.block), |c| {
            (c.current_hp, c.block)
        })
}

fn enemy_hps(s: &CombatState) -> SmallVec<[i32; 6]> {
    enemies(s).map(|c| c.current_hp.max(0)).collect()
}

/// Zero enemy HP (the compared axis) and, when `block`, the player's block
/// (the second one), drop the workbench's play log and, when `relax`, the
/// two order artefacts no reader observes, so the hash compares everything
/// else.
fn neutralize(s: &mut CombatState, relax: bool, block: bool) {
    let player = s.player.creature_uid.map(|u| u.0);
    for c in &mut s.creatures {
        if Some(c.id) != player {
            c.current_hp = 0;
        } else if block {
            c.block = 0;
        }
    }
    if block {
        s.player.block = 0;
    }
    // A UI/diagnostics log the workbench switches on
    // (`record_card_play_log`); no game rule reads it.
    s.combat_log.card_play_log.clear();
    if relax {
        s.player.discard.sort_unstable();
        s.combat_log.last_attack_skill_played = None;
    }
}

/// Apply `a` and require the turn to continue with no randomness drawn: a
/// line that consumed RNG is one determinization's luck, not a proof.
fn step(s: &mut CombatState, a: Action) -> bool {
    let fp = s.run_rng.combat_counter_fingerprint();
    let legacy = s.rng.counter();
    matches!(apply(s, a), StepResult::Advanced)
        && !s.is_over
        && s.phase == CombatPhase::PlayerPlayPhase
        && s.run_rng.combat_counter_fingerprint() == fp
        && s.rng.counter() == legacy
}

fn play_uid(s: &CombatState, uid: sts2_core::CardUid, target: Option<u8>) -> Option<Action> {
    let i = s.player.hand.iter().position(|u| *u == uid)?;
    let a = Action::PlayCard {
        hand_index: u8::try_from(i).ok()?,
        target,
    };
    legal_actions(s).contains(&a).then_some(a)
}

/// How many same-turn plays back [`pruned_followups`] looks for an earlier
/// slot to move `a` into. Each extra slot costs one replay of the line per
/// candidate; real turns rarely chain more plays than this.
pub const MAX_LINE: usize = 6;

/// The card plays proven at least as good played BEFORE `first` as after
/// it, at `root`: [`pruned_followups`] over the one-play line
/// `[(root, first)]`, as edges at `root · first`. Empty unless `first` is a
/// card play that keeps the turn going without drawing randomness. The root
/// order rules of `pick_root` read it (GH #1022): `TieRules::ORDER` with
/// `block = false`, and `TieRules::ORDER_BLOCK` with `block = true`, which
/// also counts more player block as better — Rage before the attacks, Feel
/// No Pain before an Exhaust card — unless [`block_not_monotone`].
#[must_use]
pub fn played_before(root: &CombatState, first: &Action, block: bool) -> Vec<EdgeAction> {
    if !matches!(first, Action::PlayCard { .. }) {
        return Vec::new();
    }
    let mut child = root.clone();
    if !step(&mut child, first.clone()) {
        return Vec::new();
    }
    followups(&[(root, first.clone())], &child, block)
}

/// Follow-up plays to drop at `child`, reached by the same-turn card plays
/// `line = [(P₁, b₁), …, (Pₖ, bₖ)]` (`Pⱼ · bⱼ = Pⱼ₊₁`, `Pₖ · bₖ = child`).
///
/// `a` is dropped when, for some `j`, playing it before the rest of the
/// line — `Pⱼ · a · bⱼ … bₖ` — is proven at least as good as playing it
/// after (`child · a`). Checking only the last play (`j = k`) misses the
/// same order with a neutral card between: `Strike, Zap, Beam Cell` is
/// `Strike, Beam Cell` with Zap moved, and is just as dominated.
///
/// Every `bⱼ` must have drawn no randomness (the caller only extends the
/// line across deterministic decision-to-decision hops).
pub fn pruned_followups(line: &[(&CombatState, Action)], child: &CombatState) -> Vec<EdgeAction> {
    followups(line, child, false)
}

/// [`pruned_followups`], with `block` also counting more player block as
/// better: the earlier line must then keep the same player HP and at least
/// the same block, and beat the later one on some enemy's HP or on block.
/// `block` falls back to the plain rule where [`block_not_monotone`].
fn followups(line: &[(&CombatState, Action)], child: &CombatState, block: bool) -> Vec<EdgeAction> {
    let block = block && !block_not_monotone(child);
    if line.is_empty()
        || child.is_over
        || child.phase != CombatPhase::PlayerPlayPhase
        || enemy_hp_not_monotone(child)
    {
        return Vec::new();
    }
    // The line as (state, card uid, target), so each replay can find the
    // same physical card after `a` has shifted the hand.
    let mut plays = Vec::with_capacity(line.len());
    for (s, b) in line {
        let Action::PlayCard { hand_index, target } = *b else {
            return Vec::new();
        };
        let Some(&uid) = s.player.hand.get(hand_index as usize) else {
            return Vec::new();
        };
        plays.push((*s, uid, target));
    }
    let relax = !order_is_observable(child);
    let mut out: Vec<EdgeAction> = Vec::new();
    for a in legal_actions(child) {
        let Action::PlayCard {
            hand_index: ai,
            target: at,
        } = a
        else {
            continue;
        };
        let edge = EdgeAction::from_action(&a, child);
        if out.contains(&edge) {
            continue;
        }
        let Some(&a_uid) = child.player.hand.get(ai as usize) else {
            continue;
        };
        // The line the edge would take: `a` last.
        let mut late = child.clone();
        if !step(&mut late, a) {
            continue;
        }
        let late_hp = player_hp_block(&late);
        let late_enemy = enemy_hps(&late);
        let mut late_key = None;
        // Nearest slot first: a short replay is the cheap common case.
        for j in (0..plays.len()).rev() {
            let mut early = plays[j].0.clone();
            let Some(a0) = play_uid(&early, a_uid, at) else {
                continue;
            };
            if !step(&mut early, a0) {
                continue;
            }
            let replayed = plays[j..]
                .iter()
                .all(|&(_, uid, t)| play_uid(&early, uid, t).is_some_and(|b| step(&mut early, b)));
            if !replayed {
                continue;
            }
            let (early_hp, early_block) = player_hp_block(&early);
            let more_block = block && early_block > late_hp.1;
            if early_hp != late_hp.0
                || early_block < late_hp.1
                || (!block && early_block != late_hp.1)
            {
                continue;
            }
            let he = enemy_hps(&early);
            if he.len() != late_enemy.len()
                || he.iter().zip(&late_enemy).any(|(x, y)| x > y)
                || (he == late_enemy && !more_block)
            {
                continue;
            }
            neutralize(&mut early, relax, block);
            let late_key = *late_key.get_or_insert_with(|| {
                let mut n = late.clone();
                neutralize(&mut n, relax, block);
                sts2_core::state_hash(&n)
            });
            if sts2_core::state_hash(&early) == late_key {
                out.push(edge);
                break;
            }
        }
    }
    out
}

/// Transposition-key salt for a non-empty pruned set; `0` when empty, so
/// unpruned nodes keep merging exactly as before.
pub fn salt(pruned: &[EdgeAction]) -> u64 {
    use std::hash::{Hash, Hasher};
    if pruned.is_empty() {
        return 0;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for e in pruned {
        format!("{e:?}").hash(&mut h);
    }
    h.finish() | 1
}
