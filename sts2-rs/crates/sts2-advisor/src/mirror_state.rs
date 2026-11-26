//! What the combat mirror does TO an engine state (GH #1426 step 1).
//!
//! The mirror keeps one `CombatState` alive per fight. On every decision it
//! rebuilds a fresh state from the game's wire and carries over what the wire
//! cannot say ([`transplant_hidden`]); on a choice screen the engine did not
//! open itself it injects the game's option list ([`open_injected_choice`]);
//! on a screen a turn-start hook raised, it parks the rest of turn setup
//! ([`arm_turn_start_resume`], [`arm_turn_start_resume_after_power`]).
//! These were methods of the Python `CombatState` class (`sts2-py`); they
//! moved here unchanged so the native advisor can run them without
//! Python, and `sts2-py`'s methods call them. Errors are the message the
//! Python `ValueError` carries.

use serde_json::Value;
use smallvec::SmallVec;
use sts2_core::cards::{PileKind, VarKey};
use sts2_core::{Action, CardUid, ChoiceKind, ChoiceOption, CombatState, PendingChoice};
use sts2_wire::emit::names::choice_kind_python_name;

/// Which hidden state [`transplant_hidden`] carries; the defaults are the
/// Python method's keyword defaults.
#[derive(Debug, Clone)]
pub struct Carry {
    /// `enemy_slots`: for each fresh enemy, the carried slot (or `None`).
    pub enemy_slots: Option<Vec<Option<usize>>>,
    /// `carry_rng`.
    pub rng: bool,
    /// `carry_relic_counters`: relic ids whose counter the wire withheld.
    pub relic_counters: Vec<String>,
    /// `carry_power_counters`: power ids whose counter the wire withheld.
    pub power_counters: Vec<String>,
    /// `carry_turn_history`.
    pub turn_history: bool,
}

impl Default for Carry {
    fn default() -> Self {
        Self {
            enemy_slots: None,
            rng: true,
            relic_counters: Vec::new(),
            power_counters: Vec::new(),
            turn_history: true,
        }
    }
}

/// The carried fight's encounter slots for the enemies it shares with the
/// fresh build. The board carries no slot: the rebuild guesses one from the
/// enemy's place in the list (a lone bot before the Fabricator reads bot1),
/// while the carried fight spawned the creature and knows (bot2, when bot1
/// was taken then and has died since). The next summon picks the first free
/// slot, so the guess puts it on the wrong side of the survivor. The carried
/// slots stand only where they agree with the board: every living enemy
/// slotted, in strictly rising order down the list, as C# keeps `Enemies`
/// (`SortEnemiesBySlotName`).
fn carry_enemy_slots(fresh: &mut CombatState, carried: &CombatState, slots: &[Option<usize>]) {
    let wanted: Vec<(sts2_core::CreatureUid, u8)> = slots
        .iter()
        .zip(fresh.enemy_uids.iter())
        .filter_map(|(slot, dst)| {
            let src = carried.enemy_uids.get((*slot)?)?;
            Some((*dst, carried.enemy_slot(*src)?))
        })
        .collect();
    if wanted.is_empty() {
        return;
    }
    let living = |st: &CombatState, uid: sts2_core::CreatureUid| {
        sts2_core::creature_by_id(&st.creatures, uid.0).is_some_and(|c| c.current_hp > 0)
    };
    let order: Vec<Option<u8>> = fresh
        .enemy_uids
        .iter()
        .filter(|u| living(fresh, **u))
        .map(|u| {
            wanted
                .iter()
                .find(|(w, _)| w == u)
                .map(|(_, s)| *s)
                .or_else(|| fresh.enemy_slot(*u))
        })
        .collect();
    let rising = order.iter().all(Option::is_some) && order.windows(2).all(|w| w[0] < w[1]);
    if rising {
        for (uid, slot) in wanted {
            fresh.set_enemy_slot(uid, slot);
        }
    }
}

/// A corpse keeps the intent it showed when it died, as in the engine the
/// student trained on (GH #1669). The game's board shows a dead enemy with
/// no telegraph, so the rebuild would read its intent as empty; the carried
/// enemy (alive at the last decision, or already a corpse) still holds it.
/// `enemy_slots` (`enemy_alignment`) names each fresh enemy's carried slot.
///
/// A corpse the board DOES telegraph keeps the board's telegraph: a reviver
/// (Test Subject's `RESPAWN_MOVE`, Heal + Buff, set by `TriggerDeadState`'s
/// `SetMoveImmediate`; Decimillipede's `REATTACH_MOVE`; Parafright's
/// `REVIVE_MOVE`). Since a missed kill aligns (`align_missed_kills`, GH
/// #1897) the carried creature is the one the mirror still had alive, and
/// its pre-death attack overwrote the respawn (Spire Bench r7 seeds
/// 180000239, 180001932: a knocked-out Test Subject read Bite / Multi Claw).
pub fn carry_corpse_intents(
    fresh: &mut CombatState,
    carried: &CombatState,
    enemy_slots: &[Option<usize>],
) {
    use sts2_core::creature_by_id;
    for (slot, dst) in enemy_slots.iter().zip(fresh.enemy_uids.clone()) {
        let Some(src) = slot.and_then(|s| carried.enemy_uids.get(s)) else {
            continue;
        };
        let Some(src) = creature_by_id(&carried.creatures, src.0) else {
            continue;
        };
        let intent = src.intent;
        if let Some(c) = fresh
            .creatures
            .iter_mut()
            .find(|c| c.id == dst.0 && c.current_hp <= 0 && c.intent.is_none())
        {
            c.intent = intent;
        }
    }
}

/// Rampage's growth the way the engine holds it (GH #1668): the game grows
/// `Damage.BaseValue` on each play (`Rampage.cs:54`), so the wire's base
/// damage carries it; the engine keeps the printed base and the growth in
/// the `RampageDamage` combat var (`play_rampage`). Both deal the same, but
/// the student reads the base (`DMG`), which in training never grew.
pub fn rampage_growth_as_var(state: &mut CombatState) {
    use sts2_core::cards::{CardId, VarKey};
    let printed = if state.game_version == sts2_core::GameVersion::V0_107_1 {
        9
    } else {
        10
    };
    for c in &mut state.cards {
        if c.card_id == CardId::Rampage && c.base_damage > printed {
            let grown = c.base_damage - printed;
            c.base_damage = printed;
            c.set_combat_var(
                VarKey::RampageDamage,
                c.combat_var(VarKey::RampageDamage) + grown,
            );
        }
    }
}

/// Store the max energy the engine would (GH #1701): the board carries C#'s
/// `PlayerCombatState.MaxEnergy`, which counts Paels Flesh's +1 from turn 3,
/// while the engine grants that energy at turn start and stores max energy
/// without it, which is what the student's features read.
pub fn engine_max_energy(state: &mut CombatState) {
    let paels = sts2_content::relics::shop_event::paels_flesh_max_energy_bonus(state);
    if paels != 0 {
        let stored = (i32::from(state.player.max_energy) - paels).max(0);
        state.player.max_energy = i8::try_from(stored).unwrap_or(i8::MAX);
    }
}

/// Give each enemy's buff or debuff telegraph the amount its current move
/// declares (GH #1657). The game shows a buff or debuff without a number,
/// so a board rebuilds it with amount 0; the engine the student trained on
/// telegraphs the move's own intent, amount included. The move is the
/// enemy's AI state (the board's `move_state`, or the carried one), so this
/// runs after the AI is in place. Other intents are left as the board shows
/// them.
pub fn reprime_unshown_amounts(state: &mut CombatState) {
    use sts2_core::Intent;
    for c in &mut state.creatures {
        let (Some(shown), Some(ai)) = (c.intent, c.ai.as_ref()) else {
            continue;
        };
        let Some(head) = ai.current_move().intents.first().copied() else {
            continue;
        };
        if matches!(
            (shown, head),
            (Intent::Buff { .. }, Intent::Buff { .. })
                | (Intent::Debuff { .. }, Intent::Debuff { .. })
        ) {
            c.intent = Some(head);
        }
    }
}

/// [`crate::mirror::enemy_alignment`] in `fresh`'s enemy order: the
/// alignment is per entry of the wire `fresh` was built from, and the rest of
/// this module reads `enemy_slots[i]` as `fresh.enemy_uids[i]`. The two
/// differ when the rebuild puts a roster back in its C# slot order (a wire
/// lists corpses last) or skips an escaped entry; zipping them handed one
/// enemy's history and AI to another (GH #1695). The rebuild numbers enemies
/// in wire order, so ascending uid is wire order over the entries it kept.
#[must_use]
pub fn alignment_in_fresh_order(
    fresh: &CombatState,
    wire: &Value,
    wire_slots: &[Option<usize>],
) -> Vec<Option<usize>> {
    let kept: Vec<usize> = crate::py::list(wire, "encounter")
        .iter()
        .enumerate()
        .filter(|(_, e)| !crate::py::truthy(crate::py::get(e, "is_escaped")))
        .map(|(i, _)| i)
        .collect();
    let mut by_uid: Vec<u32> = fresh.enemy_uids.iter().map(|u| u.0.get()).collect();
    by_uid.sort_unstable();
    if kept.len() != by_uid.len() {
        return wire_slots.to_vec();
    }
    fresh
        .enemy_uids
        .iter()
        .map(|u| {
            let k = by_uid.binary_search(&u.0.get()).ok()?;
            wire_slots.get(kept[k]).copied().flatten()
        })
        .collect()
}

/// For each of `fresh`'s enemies that `enemy_slots` aligns with a carried
/// enemy: `(carried uid, fresh uid)`. A rebuild numbers creatures by wire
/// position, and the wire drops ordinary corpses, so after a kill the
/// survivors' uids shift; this is the map back (GH #1651).
#[must_use]
pub fn enemy_uid_pairs(
    fresh: &CombatState,
    carried: &CombatState,
    enemy_slots: &[Option<usize>],
) -> Vec<(u32, u32)> {
    enemy_slots
        .iter()
        .zip(&fresh.enemy_uids)
        .filter_map(|(slot, dst)| {
            let src = carried.enemy_uids.get((*slot)?)?;
            Some((src.0.get(), dst.0.get()))
        })
        .collect()
}

/// Which fresh pile position takes which carried position's hidden card
/// vars. A pile of the same length pairs by position (a changed card drops
/// at the identity check). A pile that gained or lost cards pairs where
/// the shorter one is an in-order part of the longer, and only where that
/// placement is the one way to fit it (the greedy fits from both ends
/// agree): a card played to the discard no longer wipes a per-fight
/// Ethereal off every card already there (GH #1771).
fn pile_carry_pairs<T: PartialEq>(fresh: &[T], carried: &[T]) -> Vec<(usize, usize)> {
    if fresh.len() == carried.len() {
        return (0..fresh.len()).map(|i| (i, i)).collect();
    }
    let flip = fresh.len() < carried.len();
    let (short, long) = if flip {
        (fresh, carried)
    } else {
        (carried, fresh)
    };
    let fit = |order: &mut dyn Iterator<Item = usize>, rev: bool| -> Option<Vec<usize>> {
        let mut at: Vec<usize> = Vec::with_capacity(short.len());
        let mut li: Box<dyn Iterator<Item = usize>> = if rev {
            Box::new((0..long.len()).rev())
        } else {
            Box::new(0..long.len())
        };
        for si in order {
            let hit = li.by_ref().find(|&l| long[l] == short[si])?;
            at.push(hit);
        }
        if rev {
            at.reverse();
        }
        Some(at)
    };
    let (Some(left), Some(right)) = (
        fit(&mut (0..short.len()), false),
        fit(&mut (0..short.len()).rev(), true),
    ) else {
        return Vec::new();
    };
    left.iter()
        .zip(&right)
        .enumerate()
        .filter(|(_, (a, b))| a == b)
        .map(|(si, (&l, _))| if flip { (si, l) } else { (l, si) })
        .collect()
}

/// `(fresh uid, carried uid)` of each card the per-pile position pairing
/// ([`pile_carry_pairs`]) names as the same card in both states.
fn card_uid_pairs(fresh: &CombatState, carried: &CombatState) -> Vec<(CardUid, CardUid)> {
    let id_of =
        |st: &CombatState, u: &CardUid| st.cards.iter().find(|c| c.uid == *u).map(|c| c.card_id);
    let piles = [
        (&fresh.player.hand[..], &carried.player.hand[..]),
        (&fresh.player.draw[..], &carried.player.draw[..]),
        (&fresh.player.discard[..], &carried.player.discard[..]),
        (&fresh.player.exhaust[..], &carried.player.exhaust[..]),
        (&fresh.player.play[..], &carried.player.play[..]),
    ];
    let mut out = Vec::new();
    for (fp, cp) in piles {
        let fi: Vec<_> = fp.iter().map(|u| id_of(fresh, u)).collect();
        let ci: Vec<_> = cp.iter().map(|u| id_of(carried, u)).collect();
        for (f, c) in pile_carry_pairs(&fi, &ci) {
            if fi[f].is_some() && fi[f] == ci[c] {
                out.push((fp[f], cp[c]));
            }
        }
    }
    out
}

/// Carry the unobservable half of `carried` onto `fresh`.
///
/// `fresh` is a fresh reconstruction: every observable already came
/// from reality. What reality does not report — and what therefore
/// has to survive from the previous decision or be lost — is:
///
/// * the RNG cursors (legacy stream + every `RunRngSet` stream);
/// * each enemy's `intent_history` (the wire carries an i32 summary
///   that `state_from_dict` cannot rebuild, so a fresh state's
///   history is empty and the move AI re-rolls from scratch);
/// * the draw-pile pins and the belief flag;
/// * the per-combat latch a relic keeps in `counter`, for exactly those
///   relics the wire did not disclose one for (`carry_relic_counters`).
///
/// Pins are translated BY POSITION, not by uid: `state_from_dict`
/// reallocates every `CardUid` as it walks the piles, so the carried
/// mirror's uids are meaningless here, and the pins are dropped when the
/// two draw piles disagree (a reshuffle between decisions) rather than
/// failing the whole transplant — the RNG cursors are still valid.
///
/// `enemy_slots[i]` names the CARRIED slot holding the history for this
/// state's enemy `i`, `None` for a newly-spawned one. It exists because
/// the two worlds stop agreeing on slot numbering the moment one reaps a
/// corpse the other keeps; see `combat_mirror.enemy_slot_key`. Omitting
/// it falls back to slot-for-slot, which then requires equal rosters.
/// `carry.rng` (`carry_rng`) decides who owns the RNG cursors. Default `true` is the
/// original behaviour: the wire says nothing about the run's dice, so the
/// carried mirror's own streams are the only continuity there is. Pass
/// `false` when the reconstruction was built from a wire carrying
/// `$.rng.run_seed` — those streams ARE the world's, and overwriting them
/// with the mirror's would throw away the one thing that makes an enemy
/// intent predictable instead of guessed.
///
/// `carry_relic_counters` names the relics whose `counter` must survive
/// from the carried mirror. A relic's `counter` is not always a displayed
/// tally — it is also where a one-shot per-combat latch lives (Vambrace's
/// "the first card-granted block this combat doubles"). Both producers
/// only ever emit a counter for relics that SHOW one, so an undisclosed
/// latch reset to armed on every reconcile and the mirror doubled a second
/// card's block: 89 of the 123 `$.player.block` records on the v14 sweep
/// carried Vambrace, against an 11% base rate. Pass the relics the wire
/// left at `None`; the ones it did disclose keep reality's value.
///
/// `carry_turn_history`: carry the turn's exhaust count (Evil Eye) and
/// card block gains (Unmovable). Pass `false` when the wire disclosed them
/// (`cards_exhausted_this_turn` / `card_block_gains_this_turn`, GH #989):
/// then reality wins, as it does on `bound_card_played_this_turn`.
///
/// # Errors
///
/// A roster that cannot be aligned, or an unknown relic / power id.
#[allow(clippy::too_many_lines)]
pub fn transplant_hidden(
    fresh: &mut CombatState,
    carried: &CombatState,
    carry: Carry,
) -> Result<(), String> {
    let Carry {
        enemy_slots,
        rng: carry_rng,
        relic_counters: carry_relic_counters,
        power_counters: carry_power_counters,
        turn_history: carry_turn_history,
    } = carry;
    let (fresh_draw, carried_draw) = (fresh.player.draw.len(), carried.player.draw.len());
    let fresh_enemies = fresh.enemy_uids.len();
    let slots: Vec<Option<usize>> = if let Some(v) = enemy_slots {
        if v.len() != fresh_enemies {
            return Err(format!(
                "transplant_hidden: enemy_slots has {} entries for {fresh_enemies} enemies",
                v.len()
            ));
        }
        v
    } else {
        let carried_enemies = carried.enemy_uids.len();
        if fresh_enemies != carried_enemies {
            return Err(format!(
                "transplant_hidden: enemy roster disagrees — carried mirror has \
                 {carried_enemies}, the reconstruction has {fresh_enemies}, and no \
                 enemy_slots mapping was supplied to align them"
            ));
        }
        (0..fresh_enemies).map(Some).collect()
    };

    if carry_rng {
        fresh.rng = carried.rng.clone();
        fresh.run_rng = carried.run_rng.clone();
    } else {
        // The reconstruction's `run_rng` came from the world; only the
        // search-side legacy stream has no wire representation and still
        // needs carrying.
        fresh.rng = carried.rng.clone();
    }

    for (slot, carried_slot) in slots.iter().enumerate() {
        let Some(carried_slot) = *carried_slot else {
            continue; // a newly-spawned enemy has no history to carry
        };
        let Some(&src_uid) = carried.enemy_uids.get(carried_slot) else {
            continue;
        };
        let dst_uid = fresh.enemy_uids[slot];
        let Some(src) = carried
            .creatures
            .iter()
            .find(|c| c.id == src_uid.0)
            .map(|c| (c.intent_history, c.ai.clone()))
        else {
            continue;
        };
        let (history, src_ai) = src;
        if let Some(dst) = fresh.creatures.iter_mut().find(|c| c.id == dst_uid.0) {
            dst.intent_history = history;
            // The move machine's per-instance history — `state_log`,
            // `counter`, the `performed_*` flags, a pending
            // `SetMoveImmediate` — is not expressible on any wire, so a
            // rebuilt AI starts blank and mispredicts the very next
            // telegraph. `current` is NOT carried: it rides the wire as
            // `move_state`, so reality wins on it.
            if let (Some(dst_ai), Some(src_ai)) = (dst.ai.as_mut(), src_ai.as_ref()) {
                dst_ai.transplant_instance_state_from(src_ai);
            }
        }
    }

    carry_enemy_slots(fresh, carried, &slots);

    // Pins travel as draw-pile POSITIONS, so they are only translatable
    // when both sides agree on the pile. When they don't — a reshuffle
    // between decisions is the common case — the pins are simply
    // unreconstructable and are dropped. They are the player's own
    // knowledge of an order that no longer exists, so dropping them loses
    // nothing real; refusing the whole transplant would also have thrown
    // away the RNG cursors, which ARE still valid.
    // The board's own known positions (GH #1710) win; the carry fills in
    // only a board that sends none.
    let board_pins = !fresh.player.pinned_top.is_empty() || !fresh.player.pinned_bottom.is_empty();
    if fresh_draw == carried_draw && !board_pins {
        let translate = |pins: &[CardUid]| -> smallvec::SmallVec<[CardUid; 4]> {
            pins.iter()
                .filter_map(|uid| carried.player.draw.iter().position(|d| d == uid))
                .map(|idx| fresh.player.draw[idx])
                .collect()
        };
        fresh.player.pinned_top = translate(&carried.player.pinned_top);
        fresh.player.pinned_bottom = translate(&carried.player.pinned_bottom);
    }
    fresh.player.deferred_draw_order = carried.player.deferred_draw_order;
    // The stored max energy is the last turn start's (`combat_loop` writes
    // the effective value there once per turn); every wire carries the live
    // `PlayerCombatState.MaxEnergy`. A Pyre / Demesne played this turn lifts
    // the live value now and the stored one only next turn, which is what
    // the student's state features read (GH #1699).
    if fresh.round == carried.round {
        fresh.player.max_energy = carried.player.max_energy;
    }
    // First-in-series card plays this turn — EchoForm's `num >= Amount`
    // gate (GH #368.1). No producer emits it, so the reconstruction always
    // reads 0 and the gate re-opens on every reconcile. Carried here
    // rather than derived, because the count includes plays made BEFORE
    // the EchoForm power existed — Echo Form is itself a power card, so
    // its own play is the usual such case, and a per-power counter
    // seeded at apply time cannot see it (that residual was 9 of the 28
    // remaining positive enemy-HP desyncs on the post-fix DLL sweep).
    fresh.player.cards_first_in_series_this_turn = carried.player.cards_first_in_series_this_turn;
    // Started card plays this turn — the Sloth/Ringing LEGALITY gate
    // (GH #370). `state_from_dict` already seeded this exactly when a
    // Sloth power was on the wire (`SlothPower.DisplayAmount` is the live
    // count), and that reading is authoritative, so only fall back to the
    // mirror's own accumulation when the wire had no Sloth to disclose it.
    // Ringing needs the count just as much and has no carrier of its own:
    // C# scans `History.CardPlaysStarted` and stores nothing on the power.
    let sloth_disclosed_the_count = fresh.player.powers.iter().any(|puid| {
        fresh
            .powers
            .get_by_id(puid.0)
            .is_some_and(|p| p.kind == sts2_core::powers::PowerKind::Sloth)
    });
    if !sloth_disclosed_the_count {
        fresh.player.card_plays_started_this_turn = carried.player.card_plays_started_this_turn;
    }
    // The rest of the GH #370 accumulator class. Each is a plain per-turn
    // or per-combat tally that content reads as a gate and no producer
    // emits, so a rebuild re-zeroes it and the gate re-opens every
    // reconcile. Audited read-site by read-site against the C#; only the
    // ones with a LIVE consumer are carried.
    //
    //  * `shivs_played_this_turn` — `PhantomBladesPower` gives the FIRST
    //    Shiv each turn `+Amount` and gates on this being 0
    //    (`PhantomBladesPower.cs:53-72`, C# `CardPlaysFinished == 0`).
    //    Rebuilt at 0, EVERY Shiv reads as the first and takes the bonus —
    //    and Shivs arrive in volume.
    //  * `generated_cards_this_combat` — Regent's Supermassive scales its
    //    damage directly off it (`regent/supermassive.rs:24`), so a
    //    reconstruction hits for the floor.
    //  * `stars_gained_this_turn` — Regent's Radiate scales off it
    //    (`regent/radiate.rs:11`); same shape.
    //
    // Deliberately NOT carried, each verified rather than assumed:
    // `powered_hits_this_turn` / `powered_hits_by_target_this_turn` have no
    // live consumer left — GangUp is `MultiplayerOnly` and C# excludes the
    // owner's own hits, so the solo body is a flat 5. `skills_played_this_turn`
    // is documented-dead. The within-a-play scratch fields cannot span a
    // reconcile at all, because a graded decision never lands mid-play.
    fresh.player.shivs_played_this_turn = carried.player.shivs_played_this_turn;
    // Never falls within a combat: the wire's `cards_generated_this_combat`
    // when the host sends it (a Radiant Pearl's Luminesce is generated
    // before the fight's first screen, corpus seed 180000111), else the
    // mirror's running count.
    fresh.player.generated_cards_this_combat = fresh
        .player
        .generated_cards_this_combat
        .max(carried.player.generated_cards_this_combat);
    fresh.player.stars_gained_this_turn = carried.player.stars_gained_this_turn;
    // Murder's multiplier (C# `CardDrawnEntry` count). Never falls within a
    // combat: the wire's `cards_drawn_this_combat` when the host sends it,
    // else the mirror's running count (which at least grows with the fight
    // instead of re-zeroing on every rebuild).
    fresh.cards_drawn = fresh.cards_drawn.max(carried.cards_drawn);

    // The Sly / Retain uid lists are NOT carried. They are seeded from the
    // wire in `state_from_dict` (the `retain_turn` / `sly_turn` / `sly`
    // card flags, GH #370), and a carried uid could not be honoured
    // anyway: the rebuild re-allocates every `CardUid` from its own pile
    // walk, so a uid from the previous mirror names a different card.
    // Nothing to guard, then — unlike the Sloth counter above, there is no
    // fallback branch that could fight the derivation.
    //
    // `pending_retain_count` is likewise left alone, and here the reason is
    // that ZERO is already the right answer at every rebuild:
    //  * outside the end-turn chain it is always 0 — `invariants.rs:446`
    //    asserts exactly that at each round boundary;
    //  * inside the chain it is drained from the live `WellLaidPlansPower`
    //    amount (`combat_loop.rs:2500-2513`), so a rebuilt mirror
    //    re-derives it from the power rather than needing history;
    //  * and at the one graded decision the chain can suspend on — the
    //    Well-Laid-Plans `RetainAtTurnEnd` select — it has just been set
    //    to 0 (`combat_loop.rs:2580`) because the open choice now
    //    represents the whole budget. Carrying it would restore a value
    //    the engine deliberately spent.
    //
    // `vigor_latch_card` DOES need carrying. C# `VigorPower.BeforeAttack`
    // latches `commandToModify` on the first powered attack so that only
    // that card's later commands keep the bonus; the latch clears when
    // Vigor is spent. Lost across a rebuild, a second command in the same
    // play re-takes the additive boost. It names a card in the PLAY pile
    // (`damage.rs:973` stores `play.last()`), which the wire ships in
    // order, so translate it by position exactly as the draw-pile pins are
    // translated above — and drop it when the piles disagree, since a
    // latch pointing at the wrong card is worse than none.
    // Length equality alone is a weak guard on a pile this short, so the
    // card at the translated index must also BE the same card.
    fresh.player.vigor_latch_card = carried.player.vigor_latch_card.and_then(|c| {
        if fresh.player.play.len() != carried.player.play.len() {
            return None;
        }
        let idx = carried.player.play.iter().position(|p| *p == c)?;
        let dst = *fresh.player.play.get(idx)?;
        let same_card = |s: &CombatState, uid: sts2_core::CardUid| {
            s.cards.get(uid.0.get() as usize - 1).map(|k| k.card_id)
        };
        (same_card(fresh, dst) == same_card(carried, c)).then_some(dst)
    });

    for name in carry_relic_counters {
        let kind = sts2_wire::state_dict::relic_kind_from_python_name(&name).ok_or_else(|| {
            format!("transplant_hidden: unknown relic {name:?} in carry_relic_counters")
        })?;
        let Some(counter) = carried
            .relics
            .iter()
            .find(|r| r.kind == kind)
            .map(|r| r.counter)
        else {
            continue; // the carried mirror never had it (a mid-combat pickup)
        };
        if let Some(dst) = fresh.relics.iter_mut().find(|r| r.kind == kind) {
            dst.counter = counter;
        }
    }

    // `state_from_dict` builds every state with `TurnEvents::default()`
    // and the wire has no representation for it (state_dict.rs "not in
    // the wire format"), so each reconcile re-zeroed the history counters
    // C# reads out of `CombatManager.History`. Spite (`LostHpThisTurn`)
    // and TearAsunder (`1 + DamageReceivedEntry count`) read them as HIT
    // COUNTS, so the mirror predicted 1 hit forever — 79 of the 252
    // `$.encounter[*].hp` divergences on the v27wr sweep, with deltas
    // matching `-(repeat-1) x per_hit` in closed form. The carried
    // mirror was itself produced by (previous transplant -> exactly one
    // stepped action), so its log holds the running accumulation and
    // re-carrying keeps it continuous across the whole combat; the
    // turn-boundary `reset_per_turn` fires inside whichever step crosses
    // it, preserving the per-turn/combat-wide split for free.
    //
    // Only uid-free state travels: `newly_dead_this_turn`,
    // `last_attack_skill_played` and `card_play_log` hold uids the
    // reconstruction has just reallocated, and
    // `chains_bound_card_played_this_turn` IS on the wire, so reality
    // must keep winning on it. `unmovable_card_block_plays` is safe
    // between plays: every finished play's tag is the uid-free
    // `UNMOVABLE_EARLIER_PLAY` (only a play still resolving holds its
    // card's uid) — Unmovable rebuilt without it doubled EVERY
    // block card instead of only the turn's first (30 of the 140
    // on-play `$.player.block` divergences on the v27wr sweep).
    // The History Course slots' cards (below), paired before the log is
    // borrowed.
    let pairs = card_uid_pairs(fresh, carried);
    {
        let src = &carried.combat_log;
        let dst = &mut fresh.combat_log;
        dst.cards_drawn_by_effect_this_turn = src.cards_drawn_by_effect_this_turn;
        dst.status_cards_drawn_this_turn = src.status_cards_drawn_this_turn;
        dst.doom_applied_by_player_this_turn = src.doom_applied_by_player_this_turn;
        dst.osty_attacks_this_turn = src.osty_attacks_this_turn;
        if carry_turn_history {
            dst.cards_exhausted_this_turn = src.cards_exhausted_this_turn;
        }
        // Spite's "lost HP this turn". The host discloses it (SeatHistory),
        // and within a turn it only ever turns on, so the wire's `true` must
        // not be overwritten by a mirror that missed the loss: corpus seed
        // 180000995 lost HP before the fight's first screen (wire true,
        // mirror 0) and its Spite+ previewed one hit instead of three.
        dst.hp_lost_this_turn = dst.hp_lost_this_turn.max(src.hp_lost_this_turn);
        // Never falls within a combat, so the larger of wire and mirror is
        // right either way: a producer that discloses it (GH #989) is not
        // overwritten by a mirror that missed a hit (the host's 2 read as 1
        // on corpus seed 180001311, Tear Asunder one hit short), and an
        // older one that sends nothing keeps the mirror's running count.
        dst.unblocked_damage_received_this_combat = dst
            .unblocked_damage_received_this_combat
            .max(src.unblocked_damage_received_this_combat);
        // PaelsEye / Ambergris extra turns: `turn_number()` = round + this,
        // and the turn-number gates (BagOfPreparation, Lantern, the
        // turn-1 Innate floor) read it. Older mods do not send it, so a
        // rebuild after an extra turn read turn = round (GH #812). Newer
        // mods do (TurnNumber - RoundNumber), and the count never falls
        // within a combat, so the larger of wire and mirror is right
        // either way.
        dst.extra_turns_taken_this_combat = dst
            .extra_turns_taken_this_combat
            .max(src.extra_turns_taken_this_combat);
        dst.energy_spent_this_turn = src.energy_spent_this_turn;
        dst.zero_cost_attacks_played_this_turn = src.zero_cost_attacks_played_this_turn;
        dst.card_plays_finished_this_turn = src.card_plays_finished_this_turn;
        dst.card_plays_finished_last_player_turn = src.card_plays_finished_last_player_turn;
        // Within a turn the count only grows, so the wire's (the host
        // discloses it) must not be overwritten by a mirror that missed a
        // discard before the first screen (Whispering Earring's turn-1
        // auto-plays, corpus seed 180001050: wire 3, mirror 0).
        dst.cards_discarded_this_turn = dst
            .cards_discarded_this_turn
            .max(src.cards_discarded_this_turn);
        dst.lightning_orbs_channeled_this_combat = src.lightning_orbs_channeled_this_combat;
        // ChainsOfBinding's per-turn Bound budget. C#
        // `ChainsOfBindingPower.AfterCardDrawn` (`ChainsOfBindingPower.cs:36-45`)
        // Binds a drawn card only while fewer than `Amount` `CardAfflictedEntry`
        // Bound entries happened this turn; the engine keeps that count here,
        // no producer emits it, and the rebuild read 0, so the mirror Bound
        // the next `Amount` draws of a turn that had already spent its budget
        // (the PIMC agent had to fall back on every such decision, GH #830).
        // Its sibling `chains_bound_card_played_this_turn` is NOT carried —
        // it rides the wire (`bound_card_played_this_turn`).
        dst.bound_cards_afflicted_by_chains_this_turn =
            src.bound_cards_afflicted_by_chains_this_turn;
        dst.player_card_plays_this_combat = src.player_card_plays_this_combat;
        dst.ethereal_cards_played_this_combat = src.ethereal_cards_played_this_combat;
        dst.exhaust_sweep_ethereal = src.exhaust_sweep_ethereal;
        dst.joss_paper_ethereal_count = src.joss_paper_ethereal_count;
        dst.block_zero_at_turn_end = src.block_zero_at_turn_end;
        dst.block_before_late_turn_end = src.block_before_late_turn_end;
        if carry_turn_history {
            dst.unmovable_card_block_plays
                .clone_from(&src.unmovable_card_block_plays);
        }
        // The HistoryCourse slots. The reconstruction reallocated every
        // CardUid, so a uid is re-pointed at the paired fresh card or, with
        // no pairing, dropped:
        // `listener_history_course` already falls back to
        // `card_pod_from_id` for a stale uid, which is exactly why
        // `None` is safe. Left uncarried, the relic was permanently
        // inert in the mirror (30 v28wr events — run 1056's world kept
        // auto-playing last turn's Battle Trance dupe: +3 hand, -3
        // draw, +NO_DRAW the mirror never saw).
        //
        // The stamp's `upgraded` is the card as its play FINISHED; C#
        // `CreateDupe` clones the live card at the next turn's start, after
        // Razor Tooth's `AfterCardPlayed` upgrade (and any later one). The
        // engine clones the live card through the uid; the rebuilt slot has
        // none, so it takes the carried card's CURRENT upgrade (corpus r6
        // seed 180000093: a Havoc-played Neow's Fury upgraded by Razor Tooth
        // came back as a plain dupe, "up to 2" where the game said 3).
        //
        // Where the carried card pairs with a fresh one, the slot points at
        // the FRESH card, so the engine clones it as it is when the dupe is
        // made: frozen at the reconcile, an upgrade the enemy turn takes away
        // came back (Magi Knight's Dampen downgrades every upgraded card,
        // `DampenPower.AfterApplied`; corpus r7 seed 180000943: Razor Tooth's
        // Pact's End+ dupe hit for 24, the game's downgraded one for 18, and
        // the mirror killed a Flail Knight the game left at 2 HP).
        let live = |slot: Option<(sts2_core::cards::CardId, bool, Option<CardUid>)>| {
            slot.map(|(id, up, uid)| {
                let src = uid
                    .and_then(|u| carried.cards.iter().find(|c| c.uid == u))
                    .filter(|c| c.card_id == id);
                let mapped = src
                    .and_then(|c| pairs.iter().find(|(_, cu)| *cu == c.uid))
                    .map(|(fu, _)| *fu);
                (id, src.map_or(up, |c| c.upgraded), mapped)
            })
        };
        dst.last_attack_skill_played = live(src.last_attack_skill_played);
        dst.prev_round_last_attack_skill_played = live(src.prev_round_last_attack_skill_played);
    }

    // Per-card hidden `combat_vars`. The C# producers emit no
    // `combat_vars` at all (0 occurrences across the 1484-record v27wr
    // forensic log), so every rebuild wiped ThrummingHatchet's
    // `ReturnIfPlayedRound` stamp (33 records / 64 fields — the largest
    // named pile signature), Sown/Swift-style first-play marker bits in
    // `Generic`, `GeneratedInCombat`, `EtherealThisCombat`, and every
    // scaling accumulator (Rampage / Claw / RitualDagger). Carry them
    // by pile+position, exactly as the draw-pile pins are: a pile whose
    // length changed (or whose card at the position changed identity)
    // is unreconstructable and drops its carries. The energy cost is not
    // a var at all (`Card::cost`, rebuilt from the wire's cost fields), and
    // the one wire-owned var, `WitherFakeUpgradeLevel`, is re-derived from
    // wire `base_damage`. A var the fresh card already carries (the Rust
    // dialect round-trips `combat_vars`) keeps the wire's value: reality
    // wins.
    {
        const WIRE_OWNED: &[VarKey] = &[VarKey::WitherFakeUpgradeLevel];
        let pile_pairs = [
            (&fresh.player.hand[..], &carried.player.hand[..]),
            (&fresh.player.draw[..], &carried.player.draw[..]),
            (&fresh.player.discard[..], &carried.player.discard[..]),
            (&fresh.player.exhaust[..], &carried.player.exhaust[..]),
            (&fresh.player.play[..], &carried.player.play[..]),
        ];
        let mut copies: Vec<(CardUid, Vec<(VarKey, i32)>)> = Vec::new();
        // A Hexed card shows `ethereal` on the board whatever else it has
        // (C# `HexPower.TryModifyKeywordsInCombat`), so the parse's Ghost
        // Seed stamp on a Hexed basic Strike/Defend is a guess. It is wrong
        // for one Dampen downgraded: `DowngradeInternal` resets `_keywords`
        // (CardModel.cs:2157), and Ghost Seed's Ethereal goes with it. The
        // carried card knows; there the carry wins over the wire.
        let mut unstamp: Vec<CardUid> = Vec::new();
        for (fresh_pile, carried_pile) in pile_pairs {
            let fresh_ids: Vec<_> = fresh_pile
                .iter()
                .map(|u| fresh.cards.iter().find(|c| c.uid == *u).map(|c| c.card_id))
                .collect();
            let carried_ids: Vec<_> = carried_pile
                .iter()
                .map(|u| {
                    carried
                        .cards
                        .iter()
                        .find(|c| c.uid == *u)
                        .map(|c| c.card_id)
                })
                .collect();
            for (fi, ci) in pile_carry_pairs(&fresh_ids, &carried_ids) {
                let (fresh_uid, carried_uid) = (fresh_pile[fi], carried_pile[ci]);
                let Some(src) = carried.cards.iter().find(|c| c.uid == carried_uid) else {
                    continue;
                };
                let Some(dst) = fresh.cards.iter().find(|c| c.uid == fresh_uid) else {
                    continue;
                };
                if dst.card_id != src.card_id {
                    continue;
                }
                if sts2_core::combat_loop::card_is_hexed(dst)
                    && dst.combat_var(VarKey::EtherealThisCombat) != 0
                    && src.combat_var(VarKey::EtherealThisCombat) == 0
                {
                    unstamp.push(fresh_uid);
                }
                if src.combat_vars.is_empty() {
                    continue;
                }
                let vars: Vec<(VarKey, i32)> = src
                    .combat_vars
                    .iter()
                    .filter(|(k, v)| {
                        *v != 0
                            && !WIRE_OWNED.contains(k)
                            && (dst.combat_var(*k) == 0
                                // The board stamps a card the deck does not
                                // hold as generated (1); the carried card
                                // knows it is a clone (`is_clone`, read by
                                // Stomp's cost catch-up).
                                || (*k == VarKey::GeneratedInCombat
                                    && *v == sts2_core::cards::GENERATED_AS_CLONE))
                    })
                    // Rampage's own escalation is wire-owned: C# grows
                    // `Damage.BaseValue` itself (`Rampage.cs:54`), so the
                    // reconstructed card's `base_damage` already carries
                    // the growth and re-injecting the var double-counts —
                    // MCTS planned 37 where the game dealt 23 (v34wr seed
                    // 1020 act-1 boss, round-4 agent 2). Momentum's stash
                    // (`MomentumDamage`, GH #1021) lives in C#'s
                    // enchantment model (never on the wire), so it is
                    // carried on every card, Rampage included.
                    .filter(|(k, _)| {
                        !(*k == VarKey::RampageDamage
                            && dst.card_id == sts2_core::cards::CardId::Rampage)
                    })
                    .collect();
                if !vars.is_empty() {
                    copies.push((fresh_uid, vars));
                }
            }
        }
        for (uid, vars) in copies {
            if let Some(card) = fresh.cards.iter_mut().find(|c| c.uid == uid) {
                for (k, v) in vars {
                    card.set_combat_var(k, v);
                }
            }
        }
        for uid in unstamp {
            if let Some(card) = fresh.cards.iter_mut().find(|c| c.uid == uid) {
                card.set_combat_var(VarKey::EtherealThisCombat, 0);
            }
        }
    }

    // Power counters the wire structurally cannot disclose. The C#
    // producers emit `counter` only where a display badge exists; a
    // power whose accumulator lives in InternalData (Juggling's
    // attacks-this-turn — 26 v28wr hand records where the 3rd-attack
    // clone never fired; Possess*'s stolen-stat totals) arrives as
    // `{name, amount}` and every rebuild re-zeroed it. Same contract as
    // `carry_relic_counters`: the caller names only the powers the wire
    // left undisclosed, so a disclosed counter keeps reality's value.
    // The player's powers are split across the legacy list and the
    // creature mirror (d03 migration), so read/write the union; enemies
    // align through the same `slots` mapping as the AI carry.
    for name in carry_power_counters {
        let Some(kind) = sts2_wire::state_dict::power_kind_from_python_name(&name) else {
            return Err(format!(
                "transplant_hidden: unknown power {name:?} in carry_power_counters"
            ));
        };
        let read = |st: &CombatState, uids: &[sts2_core::PowerUid]| -> Option<i32> {
            uids.iter().find_map(|pu| {
                st.powers
                    .get_by_id(pu.0)
                    .filter(|p| p.kind == kind && p.counter != 0)
                    .map(|p| p.counter)
            })
        };
        let write = |st: &mut CombatState, uids: &[sts2_core::PowerUid], c: i32| {
            for pu in uids {
                if let Some(p) = st.powers.get_mut_by_id(pu.0) {
                    if p.kind == kind {
                        p.counter = c;
                        return;
                    }
                }
            }
        };
        // Player: legacy list + creature mirror, first non-zero wins.
        let carried_player_uids: Vec<sts2_core::PowerUid> = carried
            .player
            .powers
            .iter()
            .copied()
            .chain(
                carried
                    .player
                    .creature_uid
                    .and_then(|cu| carried.creatures.iter().find(|c| c.id == cu.0))
                    .map(|c| c.powers.clone())
                    .unwrap_or_default(),
            )
            .collect();
        if let Some(c) = read(carried, &carried_player_uids) {
            let fresh_player_uids: Vec<sts2_core::PowerUid> = fresh
                .player
                .powers
                .iter()
                .copied()
                .chain(
                    fresh
                        .player
                        .creature_uid
                        .and_then(|cu| fresh.creatures.iter().find(|cr| cr.id == cu.0))
                        .map(|cr| cr.powers.clone())
                        .unwrap_or_default(),
                )
                .collect();
            write(fresh, &fresh_player_uids, c);
        }
        // Enemies, through the slot alignment.
        for (slot, carried_slot) in slots.iter().enumerate() {
            let Some(carried_slot) = *carried_slot else {
                continue;
            };
            let (Some(&src_uid), Some(&dst_uid)) = (
                carried.enemy_uids.get(carried_slot),
                fresh.enemy_uids.get(slot),
            ) else {
                continue;
            };
            let Some(src_powers) = carried
                .creatures
                .iter()
                .find(|c| c.id == src_uid.0)
                .map(|c| c.powers.clone())
            else {
                continue;
            };
            if let Some(c) = read(carried, &src_powers) {
                let Some(dst_powers) = fresh
                    .creatures
                    .iter()
                    .find(|cr| cr.id == dst_uid.0)
                    .map(|cr| cr.powers.clone())
                else {
                    continue;
                };
                write(fresh, &dst_powers, c);
            }
        }
    }

    // `Power.applier` for the applier-gated debuffs. `state_from_dict`
    // hardcodes `applier: None` (no wire field carries it), so
    // Shrink/Hex/Dampen/Constrict's `AfterDeath` removal — gated on
    // `p.applier == Some(dead)` — could never fire in a reconstruction:
    // the mirror kept a dead ShrinkerBeetle's permanent Shrink for the
    // rest of the fight (21 v28wr `$.player.powers` events). Copy the
    // carried instance's applier onto the fresh player-side power of the
    // same kind, remapped through the slot alignment; nothing else reads
    // `applier` on these kinds, so a missing mapping is inert.
    carry_power_appliers(fresh, carried, &slots);
    carry_orbit_energy(fresh, carried);
    carry_juggling_count(fresh, carried);
    Ok(())
}

/// `Power.applier` of the player-side applier-gated debuffs
/// (Shrink/Hex/Dampen/Constrict), carried from `carried` onto `fresh`
/// through the enemy slot alignment `slots` (for each fresh enemy, its
/// carried slot). Shared by [`transplant_hidden`] and the OBSERVE reseed:
/// the reseed drops every enemy-side carry, but the applier is PLAYER-side
/// state that only names an enemy, and losing it makes the debuff permanent
/// (GH dtype triage: a reseed after an unrelated HP desync left Magi
/// Knight's Dampen on after the Knight died, so the world's re-upgraded
/// cards could not be located on the next card_select).
pub fn carry_power_appliers(
    fresh: &mut CombatState,
    carried: &CombatState,
    slots: &[Option<usize>],
) {
    use sts2_core::PowerKind as PK;
    const APPLIER_KINDS: [PK; 4] = [PK::Shrink, PK::Hex, PK::Dampen, PK::Constrict];
    let mut uid_map: Vec<(sts2_core::CreatureUid, sts2_core::CreatureUid)> = Vec::new();
    for (slot, carried_slot) in slots.iter().enumerate() {
        if let Some(cs) = carried_slot {
            if let (Some(&src), Some(&dst)) =
                (carried.enemy_uids.get(*cs), fresh.enemy_uids.get(slot))
            {
                uid_map.push((src, dst));
            }
        }
    }
    let carried_player_uids: Vec<sts2_core::PowerUid> = carried
        .player
        .powers
        .iter()
        .copied()
        .chain(
            carried
                .player
                .creature_uid
                .and_then(|cu| carried.creatures.iter().find(|c| c.id == cu.0))
                .map(|c| c.powers.clone())
                .unwrap_or_default(),
        )
        .collect();
    let fresh_player_uids: Vec<sts2_core::PowerUid> = fresh
        .player
        .powers
        .iter()
        .copied()
        .chain(
            fresh
                .player
                .creature_uid
                .and_then(|cu| fresh.creatures.iter().find(|cr| cr.id == cu.0))
                .map(|cr| cr.powers.clone())
                .unwrap_or_default(),
        )
        .collect();
    for kind in APPLIER_KINDS {
        let Some(src_applier) = carried_player_uids.iter().find_map(|pu| {
            carried
                .powers
                .get_by_id(pu.0)
                .filter(|p| p.kind == kind)
                .and_then(|p| p.applier)
        }) else {
            continue;
        };
        let Some(mapped) = uid_map
            .iter()
            .find(|(src, _)| *src == src_applier)
            .map(|(_, dst)| *dst)
        else {
            continue;
        };
        for pu in &fresh_player_uids {
            if let Some(p) = fresh.powers.get_mut_by_id(pu.0) {
                if p.kind == kind && p.applier.is_none() {
                    p.applier = Some(mapped);
                    break;
                }
            }
        }
    }
}

/// Juggling's count of this turn's Attacks, where the carried fight saw
/// more than the board's History does.
///
/// The rebuild derives the count from the wire's Attack bucket of
/// `cards_played_this_turn`, which the host counts from C#
/// `History.CardPlaysStarted` filtered on `HappenedThisTurn` (round AND
/// side). C# `JugglingPower.BeforeCardPlayed` counts every Attack the owner
/// plays from its `AfterSideTurnEnd` reset on, so a Strike Hellraiser plays
/// as the turn's hand is drawn counts for Juggling but not as this turn's in
/// History (Spire Bench seed 180000794, total floor 45 round 3: the board's
/// bucket said 0 Attacks, and the game cloned the turn's SECOND hand Attack).
/// The carried fight ran the same turn and counted it; within one round it
/// stands when it is higher.
fn carry_juggling_count(fresh: &mut CombatState, carried: &CombatState) {
    use sts2_core::powers::PowerKind;
    if fresh.round != carried.round {
        return;
    }
    let count = |st: &CombatState| -> Option<(sts2_core::PowerUid, i32)> {
        st.player_power_uids().into_iter().find_map(|pu| {
            st.powers
                .get_by_id(pu.0)
                .filter(|p| p.kind == PowerKind::Juggling)
                .map(|p| (pu, p.counter))
        })
    };
    let (Some((_, seen)), Some((pu, derived))) = (count(carried), count(fresh)) else {
        return;
    };
    if seen > derived {
        if let Some(p) = fresh.powers.get_mut_by_id(pu.0) {
            p.counter = seen;
        }
    }
}

/// Copy the carried counter of each named player power onto the fresh
/// player's power of the same kind: [`transplant_hidden`]'s player-side power
/// carry alone, for a reconcile that diffs the board before it transplants.
///
/// Some observables are derived from a counter the wire withholds. Void
/// Form's free plays are its `cardsPlayedThisTurn` (C# `VoidFormPower.cs:81`
/// `TryModifyEnergyCostInCombatLate`), which no board shows, so a bare rebuild
/// read every hand card free past the free plays and `$.deck.hand` diffed on
/// cost at every reconcile of such a turn (Spire Bench seeds 180000322,
/// 180000516, 180000590, 180001456, 180000358). Only `names` (the powers the
/// wire left undisclosed) are touched, and a carried 0 is not copied, as in
/// the full transplant.
pub fn carry_player_power_counters(
    fresh: &mut CombatState,
    carried: &CombatState,
    names: &[String],
) {
    for name in names {
        let Some(kind) = sts2_wire::state_dict::power_kind_from_python_name(name) else {
            continue;
        };
        let Some(c) = carried.player_power_uids().into_iter().find_map(|pu| {
            carried
                .powers
                .get_by_id(pu.0)
                .filter(|p| p.kind == kind && p.counter != 0)
                .map(|p| p.counter)
        }) else {
            continue;
        };
        let target = fresh
            .player_power_uids()
            .into_iter()
            .find(|pu| fresh.powers.get_by_id(pu.0).is_some_and(|p| p.kind == kind));
        if let Some(p) = target.and_then(|pu| fresh.powers.get_mut_by_id(pu.0)) {
            p.counter = c;
        }
    }
}

/// Orbit's engine counter is the energy spent since it was applied; the
/// wire shows only how much is left to the next trigger, from which the
/// rebuild derives the total mod 4. A carried total that agrees mod 4 is the
/// true one, so it stands (GH #1741).
///
/// Orbit is instanced (C# `PowerInstanceType.Instanced`): a second Orbit
/// played is a second power with its own `energySpent`. Each fresh instance
/// takes the carried total of the instance in the same place, in order;
/// carrying only the first left every later one on its residue (Spire Bench
/// seed 180001317: two Orbits, the second read 0/1/3 where the game's was
/// 4/5/7).
fn carry_orbit_energy(fresh: &mut CombatState, carried: &CombatState) {
    use sts2_core::powers::PowerKind;
    let orbits = |st: &CombatState| -> Vec<(sts2_core::PowerUid, i32)> {
        st.player_power_uids()
            .into_iter()
            .filter_map(|pu| {
                st.powers
                    .get_by_id(pu.0)
                    .filter(|p| p.kind == PowerKind::Orbit)
                    .map(|p| (pu, p.counter))
            })
            .collect()
    };
    for ((_, total), (pu, derived)) in orbits(carried).into_iter().zip(orbits(fresh)) {
        if total > derived && total.rem_euclid(4) == derived.rem_euclid(4) {
            if let Some(p) = fresh.powers.get_mut_by_id(pu.0) {
                p.counter = total;
            }
        }
    }
}

/// Inverse of `choice_kind_python_name`, scanning the forward table so the
/// two cannot drift as `ChoiceKind` grows.
#[must_use]
pub fn choice_kind_from_python_name(s: &str) -> Option<ChoiceKind> {
    ChoiceKind::ALL
        .iter()
        .copied()
        .find(|k| choice_kind_python_name(*k) == s)
}

/// Open the world's choice screen on `state`, its options given as the
/// world's option dicts (`{"id", "upgraded", "pile"?, "index"?}`).
///
/// With `replace_parked`, a choice the engine already parked on is swapped
/// for the world's list, keeping its `post_effect` when the kind matches:
/// the post-effect is what runs AFTER the pick (Brand's GainStrength,
/// BurningPact's Draw, Scavenge's EnergyNextTurn), so swapping the option
/// list must not amputate it. Dropping it was 60 of the 188
/// `$.player.powers` divergences on the v27wr sweep (a Brand that exhausted
/// its card and never granted its Strength). `pending_play`, the suspended
/// rest of the card, survives either way: that is why the mirror is parked
/// here at all.
///
/// # Errors
///
/// A choice already open (without `replace_parked`), an unknown kind, an
/// arity the options cannot satisfy, or a bad option (see
/// [`parse_injected_option`]).
pub fn open_injected_choice(
    state: &mut CombatState,
    kind: &str,
    options: &[Value],
    min_select: u8,
    max_select: u8,
    can_skip: bool,
    replace_parked: bool,
) -> Result<(), String> {
    let mut carried_post = None;
    if let Some(parked) = &state.pending_choice {
        if !replace_parked {
            return Err(
                "open_injected_choice: a choice is already open — resolve or drop it first".into(),
            );
        }
        carried_post = parked.post_effect.map(|pe| (parked.kind, pe));
        state.pending_choice = None;
    }
    let kind = choice_kind_from_python_name(kind)
        .ok_or_else(|| format!("open_injected_choice: unknown choice kind {kind:?}"))?;
    // Only the SAME screen keeps the tail: a genuinely different injected
    // kind must not inherit a foreign card's post-effect.
    let carried_post = carried_post.and_then(|(pk, pe)| (pk == kind).then_some(pe));
    if min_select > max_select {
        return Err(format!(
            "open_injected_choice: min_select ({min_select}) > max_select ({max_select})"
        ));
    }
    let parsed = options
        .iter()
        .enumerate()
        .map(|(i, o)| parse_injected_option(state, o, i))
        .collect::<Result<Vec<_>, _>>()?;
    if usize::from(min_select) > parsed.len() {
        return Err(format!(
            "open_injected_choice: min_select ({min_select}) exceeds the {} options offered",
            parsed.len()
        ));
    }
    // `max_select` gets the same bound. Both drivers clamp before calling,
    // which is exactly why this has to be checked HERE: this is the only
    // thing standing between an unclamped caller and a choice whose arity
    // the option list cannot satisfy.
    if usize::from(max_select) > parsed.len() {
        return Err(format!(
            "open_injected_choice: max_select ({max_select}) exceeds the {} options offered",
            parsed.len()
        ));
    }
    state.pending_choice = Some(PendingChoice {
        kind,
        options: parsed,
        min_select,
        max_select,
        can_skip,
        post_effect: carried_post,
        picked: SmallVec::new(),
    });
    Ok(())
}

/// Decode one [`open_injected_choice`] option.
///
/// `pile` + `index` are the concrete-card locator; they must appear together
/// or not at all. A half-specified locator is a driver bug that would
/// otherwise silently produce a synthetic option and send a pile-select
/// resolver looking for a card that isn't there. A key whose value is
/// `null` counts as absent.
///
/// # Errors
///
/// A missing or mistyped field, an unknown card, a locator out of range, or
/// a locator naming a different card than the option claims.
pub fn parse_injected_option(
    state: &CombatState,
    option: &Value,
    i: usize,
) -> Result<ChoiceOption, String> {
    if !option.is_object() {
        return Err(format!("open_injected_choice: options[{i}] must be a dict"));
    }
    let present = |key: &str| option.get(key).filter(|v| !v.is_null());
    let raw_id = option
        .get("id")
        .ok_or_else(|| format!("open_injected_choice: options[{i}] is missing 'id'"))?
        .as_str()
        .ok_or_else(|| format!("open_injected_choice: options[{i}].id must be a string"))?;
    let card_id = sts2_wire::state_dict::card_id_from_python_name(raw_id).ok_or_else(|| {
        format!("open_injected_choice: options[{i}].id {raw_id:?} is not a known card")
    })?;
    let upgraded = match present("upgraded") {
        Some(v) => v
            .as_bool()
            .ok_or_else(|| format!("open_injected_choice: options[{i}].upgraded must be a bool"))?,
        None => false,
    };
    let pile_name = present("pile")
        .map(|v| {
            v.as_str()
                .ok_or_else(|| format!("open_injected_choice: options[{i}].pile must be a string"))
        })
        .transpose()?;
    let index = present("index")
        .map(|v| {
            // Python's `int` accepted `True`/`False` as 1/0; keep that.
            v.as_u64()
                .or_else(|| v.as_bool().map(u64::from))
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    format!(
                        "open_injected_choice: options[{i}].index must be a non-negative integer"
                    )
                })
        })
        .transpose()?;

    let (pile, index) = match (pile_name, index) {
        (None, None) => {
            return Ok(ChoiceOption {
                card_id,
                upgraded,
                card_uid: None,
                source_pile: None,
            })
        }
        (Some(p), Some(idx)) => (p, idx),
        (p, idx) => {
            return Err(format!(
                "open_injected_choice: options[{i}] gives {} without {} — a card locator \
                 needs both",
                if p.is_some() { "'pile'" } else { "'index'" },
                if idx.is_some() { "'pile'" } else { "'index'" },
            ))
        }
    };
    let (kind, uids) = match pile {
        "hand" => (PileKind::Hand, state.player.hand.as_slice()),
        "draw" => (PileKind::Draw, state.player.draw.as_slice()),
        "discard" => (PileKind::Discard, state.player.discard.as_slice()),
        "exhaust" => (PileKind::Exhaust, state.player.exhaust.as_slice()),
        "play" => (PileKind::Play, state.player.play.as_slice()),
        other => {
            return Err(format!(
                "open_injected_choice: options[{i}].pile {other:?} must be one of \
                 hand|draw|discard|exhaust|play"
            ))
        }
    };
    let uid: CardUid = *uids.get(index).ok_or_else(|| {
        format!(
            "open_injected_choice: options[{i}].index {index} is out of range for a \
             {}-card {pile} pile",
            uids.len()
        )
    })?;
    // The locator names a card; that card had better be the card the option
    // claims to be, or the mirror and the world are pointing at different
    // boards. A uid in a pile but absent from `state.cards` is itself a
    // broken state (the pile is a list of handles INTO that arena), so it is
    // an error rather than a silent pass.
    let Some(card) = state.cards.iter().find(|c| c.uid == uid) else {
        return Err(format!(
            "open_injected_choice: options[{i}] locates {pile}[{index}] = {uid:?}, which is \
             not in the card arena — the mirror's piles reference a card it does not have"
        ));
    };
    if card.card_id != card_id || card.upgraded != upgraded {
        return Err(format!(
            "open_injected_choice: options[{i}] says {raw_id:?} (upgraded={upgraded}) \
             at {pile}[{index}], but the mirror has {:?} (upgraded={}) there",
            card.card_id.to_python_name(),
            card.upgraded,
        ));
    }
    Ok(ChoiceOption {
        card_id,
        upgraded,
        card_uid: Some(uid),
        source_pile: Some(kind),
    })
}

/// Mark the OPEN choice as a turn-start screen the world raised from a
/// relic's `AfterPlayerTurnStart`, so resolving it also runs the rest of
/// turn setup.
///
/// C# `Hook.AfterPlayerTurnStart` **awaits** GamblingChip's
/// `CardSelectCmd.FromHandForDiscard`, so the world's board at that decision
/// is a MID-turn-setup snapshot: the hand is drawn and energy reset, but
/// every later relic on that pass (MercuryHourglass), the whole
/// `AfterPlayerTurnStartLate` pass (BloodVial) and `Hook.AfterSideTurnStart`
/// (Akabeko, Lantern) have NOT run. A mirror seeded from that wire and handed
/// a bare discard never ran that batch, and stayed short a Vigor 8 / +1
/// energy / −3-to-all / heal 2 for the rest of the combat.
///
/// This parks the `PendingClusterResume` the engine parks itself when its
/// OWN GamblingChip raises the screen: `relic_start` just past
/// `after_relic`'s equip slot. Resolving the choice then runs the remaining
/// listeners and the turn-start tail in C# order.
///
/// Returns `false` (a no-op) when no choice is open or the player does not
/// own the relic: the screen recognition was wrong, and parking a resume
/// against a board that never suspended would fire the batch twice.
///
/// # Errors
///
/// `after_relic` is not a known relic id.
pub fn arm_turn_start_resume(state: &mut CombatState, after_relic: &str) -> Result<bool, String> {
    let kind = sts2_wire::state_dict::relic_kind_from_python_name(after_relic)
        .ok_or_else(|| format!("arm_turn_start_resume: unknown relic id {after_relic:?}"))?;
    if state.pending_choice.is_none() {
        return Ok(false);
    }
    let Some(idx) = state.relics.iter().position(|r| r.kind == kind) else {
        return Ok(false);
    };
    // v0.111.0 Toasty Mittens awaits its exhaust pick, then applies its
    // Strength (ToastyMittens.cs): the world's screen, injected bare, keeps
    // that tail as the engine's own pick carries it (GH #1696).
    if kind == sts2_core::relics::RelicKind::ToastyMittens {
        if let Some(pc) = state.pending_choice.as_mut() {
            if pc.kind == ChoiceKind::ExhaustFromHand && pc.post_effect.is_none() {
                pc.post_effect = Some(sts2_core::PostChoiceEffect::GainStrength(
                    sts2_content::relics::shop_event::TOASTY_MITTENS_STRENGTH,
                ));
            }
        }
    }
    let next = u8::try_from(idx.saturating_add(1)).unwrap_or(u8::MAX);
    state.pending_cluster_resume = Some(sts2_core::PendingClusterResume {
        snapshot: state.player_power_uids(),
        idx: 0,
        autoplay: sts2_core::PendingAutoplay::default(),
        // C# StartTurn runs Hook.AfterSideTurnStart (and the orb queue) as
        // soon as SetupPlayerTurn pauses on this screen, so the wire the
        // mirror was seeded from already carries Akabeko / Lantern / Rampart.
        side_turn_start_fired: true,
        mayhem_phase: false,
        relic_start: Some(next),
    });
    Ok(true)
}

/// The POWER twin of [`arm_turn_start_resume`] (GH #355).
///
/// Both v0.107.1 relics whose `AfterPlayerTurnStart` awaits a card select
/// are gated to turn 1; the any-turn case is the three POWERS that await one
/// with no turn gate: `TyrannyPower` (exhaust N), `ToolsOfTheTradePower`
/// (discard N) and `EntropyPower` (transform N). A mirror seeded at one of
/// those screens is a mid-setup snapshot exactly as the Gambling Chip one
/// was, and answering the bare selection left the rest of the cluster, the
/// relic `AfterPlayerTurnStart` bus and `Hook.AfterSideTurnStart` unrun: the
/// mirror then read a full round behind the world.
///
/// This parks the resume the engine parks for its OWN Tyranny/Tools
/// (`combat_loop::run_after_player_turn_start_cluster`): a no-op auto-play
/// at the suspending power's index, so the resume runs from `idx + 1` plus
/// the turn-start tail. Returns `false` (a no-op) when no choice is open or
/// the player does not carry the power.
///
/// # Errors
///
/// `after_power` is not a known power id.
pub fn arm_turn_start_resume_after_power(
    state: &mut CombatState,
    after_power: &str,
) -> Result<bool, String> {
    let kind =
        sts2_wire::state_dict::power_kind_from_python_name(after_power).ok_or_else(|| {
            format!("arm_turn_start_resume_after_power: unknown power id {after_power:?}")
        })?;
    if state.pending_choice.is_none() {
        return Ok(false);
    }
    let snapshot = state.player_power_uids();
    let Some(idx) = snapshot.iter().position(|uid| {
        state
            .powers
            .get_by_id(uid.0)
            .is_some_and(|p| p.kind == kind)
    }) else {
        return Ok(false);
    };
    state.pending_cluster_resume = Some(sts2_core::PendingClusterResume {
        snapshot,
        idx,
        autoplay: sts2_core::PendingAutoplay::default(),
        // As in `arm_turn_start_resume`: the wire already carries
        // AfterSideTurnStart's effects.
        side_turn_start_fired: true,
        mayhem_phase: false,
        relic_start: None,
    });
    Ok(true)
}

/// Decode a bridge-shape action (`{"action": "play", "card_index": 0, ...}`)
/// into a core [`Action`]: `sts2-py`'s `parse_bridge_action`, for the JSON
/// the mirror is handed, with its error messages.
///
/// # Errors
///
/// A missing or unknown `action`, a missing or out-of-range index, or a
/// potion `discard` (the engine has no discard action, and decoding it as a
/// drink would diverge from the world that emitted it).
pub fn parse_bridge_action(action: &Value) -> Result<Action, String> {
    let kind = match action.get("action") {
        None => return Err("apply: missing required key 'action'".into()),
        Some(Value::String(s)) => s.as_str(),
        Some(other) => return Err(format!("apply: 'action' must be a string ({other})")),
    };
    let index = |name: &str, v: &Value| -> Result<u8, String> {
        let raw = match v {
            Value::Bool(b) => u64::from(*b),
            Value::Number(n) => n
                .as_u64()
                .ok_or_else(|| format!("apply: '{name}' must be a non-negative integer ({n})"))?,
            other => {
                return Err(format!(
                    "apply: '{name}' must be a non-negative integer ({other})"
                ))
            }
        };
        u8::try_from(raw).map_err(|_| format!("apply: '{name}'={raw} out of range for u8"))
    };
    let opt_u8 = |name: &str| -> Result<Option<u8>, String> {
        match action.get(name) {
            Some(v) if !v.is_null() => index(name, v).map(Some),
            _ => Ok(None),
        }
    };
    let req_u8 = |name: &str| -> Result<u8, String> {
        match action.get(name) {
            Some(v) => index(name, v),
            None => Err(format!("apply: missing required key '{name}'")),
        }
    };
    match kind {
        "play" => Ok(match opt_u8("ally_index")? {
            Some(ally) => Action::PlayCardAtAlly {
                hand_index: req_u8("card_index")?,
                ally,
            },
            None => Action::PlayCard {
                hand_index: req_u8("card_index")?,
                target: opt_u8("target_index")?,
            },
        }),
        "end_turn" => Ok(Action::EndTurn),
        "potion" | "drink_potion" => {
            if action.get("discard").is_some_and(crate::py::truthy) {
                return Err("apply: 'discard' is not supported — the engine has no \
                            potion-discard action, and decoding it as a drink would \
                            diverge from the world that emitted it"
                    .into());
            }
            Ok(match opt_u8("ally_index")? {
                Some(ally) => Action::DrinkPotionAtAlly {
                    potion_slot: req_u8("slot")?,
                    ally,
                },
                None => Action::DrinkPotion {
                    potion_slot: req_u8("slot")?,
                    target: opt_u8("target_index")?,
                },
            })
        }
        "choose" => {
            let mut selection: SmallVec<[u8; 4]> = SmallVec::new();
            match action.get("selection") {
                Some(Value::Array(items)) => {
                    for item in items {
                        let raw = match item {
                            Value::Bool(b) => Some(u64::from(*b)),
                            Value::Number(n) => n.as_u64(),
                            _ => None,
                        }
                        .ok_or_else(|| {
                            format!(
                                "apply: 'selection' must be a list of non-negative integers ({item})"
                            )
                        })?;
                        selection.push(u8::try_from(raw).map_err(|_| {
                            format!("apply: 'selection' entry {raw} out of range for u8")
                        })?);
                    }
                }
                Some(v) if !v.is_null() => {
                    return Err(format!(
                        "apply: 'selection' must be a list of non-negative integers ({v})"
                    ))
                }
                _ => selection.push(req_u8("index")?),
            }
            Ok(Action::ResolveChoice { selection })
        }
        other => Err(format!("apply: unknown action kind '{other}'")),
    }
}

/// [`sts2_core::StepResult`] as the lowercase string the mirror reports.
#[must_use]
pub const fn step_result_name(result: sts2_core::StepResult) -> &'static str {
    match result {
        sts2_core::StepResult::Advanced => "advanced",
        sts2_core::StepResult::PendingChoice => "pending_choice",
        sts2_core::StepResult::CombatWon => "combat_won",
        sts2_core::StepResult::CombatLost => "combat_lost",
        sts2_core::StepResult::InvalidTarget => "invalid_target",
        sts2_core::StepResult::IllegalPlay => "illegal_play",
    }
}

/// `choice_kind_is_generative`: the kinds whose options the effect rolled
/// itself, so they sit in no pile and grading them says nothing about pile
/// fidelity. `None` for an unknown kind.
#[must_use]
pub fn choice_kind_is_generative(kind: &str) -> Option<bool> {
    choice_kind_from_python_name(kind).map(|k| {
        matches!(
            k,
            ChoiceKind::AddGeneratedToHand
                | ChoiceKind::AddGeneratedRetainedToHand
                | ChoiceKind::AddGeneratedFreeToHand
                // The KnowledgeDemon curse screen's options are TOKENS the
                // boss conjured: they live in no pile, so a locator can never
                // resolve them.
                | ChoiceKind::MonsterCurseSelect
        )
    })
}

/// A turn-transition walk SUSPENDED on a player choice: Well-Laid Plans'
/// retain select inside the end-of-turn flush, a Stampede auto-play screen,
/// a suspended turn setup / cluster / hand draw, or the KnowledgeDemon
/// curse walk. No resume marker is expressible on the wire, so a mirror that
/// rebuilds here deletes the rest of the transition and lands a full round
/// behind; it keeps the carried state instead.
#[must_use]
pub const fn has_suspended_walk(state: &CombatState) -> bool {
    state.pending_wlp_retain_resume
        || state.pending_end_turn_resume.is_some()
        || state.pending_turn_setup_resume
        || state.pending_cluster_resume.is_some()
        || state.pending_autoplay_resume.is_some()
        || state.pending_draw_resume.is_some()
        || state.pending_hand_draw_resume.is_some()
        || state.pending_relic_hand_draw_resume.is_some()
        || state.pending_enemy_exec_resume.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// GH #1771: a card played to the discard keeps the per-fight Ethereal
    /// of the cards already there.
    #[test]
    fn a_grown_discard_keeps_its_cards_hidden_vars() {
        use sts2_core::cards::VarKey;
        let mut carried = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &["BULLY", "BULLY", "DISMANTLE", "BASH", "BASH", "BASH"].map(str::to_owned),
            80,
            80,
            &[],
            &[],
            "AXEBOTS_NORMAL",
            10,
            None,
            None,
        )
        .expect("scenario");
        let pile = |st: &CombatState, id: sts2_core::cards::CardId| {
            st.cards.iter().find(|c| c.card_id == id).map(|c| c.uid)
        };
        let mut all: Vec<CardUid> = carried.player.hand.drain(..).collect();
        all.extend(carried.player.draw.drain(..));
        let bullies: Vec<CardUid> = all
            .iter()
            .copied()
            .filter(|u| {
                carried
                    .cards
                    .iter()
                    .any(|c| c.uid == *u && c.card_id == sts2_core::cards::CardId::Bully)
            })
            .collect();
        let dismantle = pile(&carried, sts2_core::cards::CardId::Dismantle).expect("dismantle");
        carried.player.draw = all
            .iter()
            .copied()
            .filter(|u| !bullies.contains(u) && *u != dismantle)
            .collect();
        carried.player.discard = bullies.iter().copied().collect();
        carried.player.hand = [dismantle].into_iter().collect();
        let mut fresh = carried.clone();
        if let Some(c) = carried.cards.iter_mut().find(|c| c.uid == bullies[0]) {
            c.set_combat_var(VarKey::EtherealThisCombat, 1);
        }
        fresh.player.hand.clear();
        fresh.player.discard = [bullies[0], dismantle, bullies[1]].into_iter().collect();
        transplant_hidden(
            &mut fresh,
            &carried,
            Carry {
                enemy_slots: None,
                rng: false,
                relic_counters: Vec::new(),
                power_counters: Vec::new(),
                turn_history: false,
            },
        )
        .expect("transplant");
        let eth = |u: CardUid| {
            fresh
                .cards
                .iter()
                .find(|c| c.uid == u)
                .map(|c| c.combat_var(VarKey::EtherealThisCombat))
        };
        assert_eq!(eth(bullies[0]), Some(1));
        assert_eq!(eth(bullies[1]), Some(0));
    }

    /// Corpus seeds 180001232 / 180000028 / 180001939 / 180001014 (Ghost
    /// Seed, the Knights): Dampen downgraded the upgraded Strike, which
    /// strips Ghost Seed's Ethereal (`CardModel.DowngradeInternal`), while
    /// Hex keeps every card showing `ethereal`. The parse stamps Ghost
    /// Seed's var on both Hexed Strikes; the carried mirror knows which one
    /// lost it. Without Hex the wire's flag is evidence and still wins.
    #[test]
    fn a_hexed_card_keeps_the_carried_lack_of_ghost_seed_ethereal() {
        use sts2_core::card_afflictions::{CardAffliction, CardAfflictionKind};
        use sts2_core::cards::{CardId, VarKey};
        let mut carried = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "BASH", "BASH", "BASH"].map(str::to_owned),
            80,
            80,
            &[],
            &[],
            "AXEBOTS_NORMAL",
            10,
            None,
            None,
        )
        .expect("scenario");
        let mut all: Vec<CardUid> = carried.player.hand.drain(..).collect();
        all.extend(carried.player.draw.drain(..));
        let strikes: Vec<CardUid> = all
            .iter()
            .copied()
            .filter(|u| {
                carried
                    .cards
                    .iter()
                    .any(|c| c.uid == *u && c.card_id == CardId::StrikeIronclad)
            })
            .collect();
        carried.player.draw = all
            .iter()
            .copied()
            .filter(|u| !strikes.contains(u))
            .collect();
        carried.player.hand = strikes.iter().copied().collect();
        let run = |hexed: bool| {
            let mut carried = carried.clone();
            for c in carried
                .cards
                .iter_mut()
                .filter(|c| strikes.contains(&c.uid))
            {
                if hexed {
                    let hex = CardAffliction {
                        kind: CardAfflictionKind::Hexed,
                        stacks: 2,
                        applied_exhaust: false,
                    };
                    assert!(c.set_affliction(hex));
                }
                // The board's Ghost Seed stamp, on both.
                c.set_combat_var(VarKey::EtherealThisCombat, 1);
            }
            let mut fresh = carried.clone();
            // The mirror's own fight: Dampen downgraded the second Strike.
            if let Some(c) = carried.cards.iter_mut().find(|c| c.uid == strikes[1]) {
                c.set_combat_var(VarKey::EtherealThisCombat, 0);
                c.set_combat_var(VarKey::DampenDowngradedMarker, 1);
            }
            transplant_hidden(
                &mut fresh,
                &carried,
                Carry {
                    enemy_slots: None,
                    rng: false,
                    relic_counters: Vec::new(),
                    power_counters: Vec::new(),
                    turn_history: false,
                },
            )
            .expect("transplant");
            strikes
                .iter()
                .map(|u| {
                    fresh
                        .cards
                        .iter()
                        .find(|c| c.uid == *u)
                        .map(|c| c.combat_var(VarKey::EtherealThisCombat))
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(true), vec![Some(1), Some(0)], "Hexed: the carry wins");
        assert_eq!(run(false), vec![Some(1), Some(1)], "no Hex: the board wins");
    }

    #[test]
    fn paels_flesh_leaves_the_stored_max_energy_from_turn_three() {
        let mut s = CombatState::default();
        s.relics.push(sts2_core::Relic {
            id: std::num::NonZeroU32::new(1).expect("nz"),
            kind: sts2_core::RelicKind::PaelsFlesh,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        s.player.max_energy = 4; // C#'s MaxEnergy
        s.round = 2;
        engine_max_energy(&mut s);
        assert_eq!(s.player.max_energy, 4, "no bonus before turn 3");
        s.round = 3;
        engine_max_energy(&mut s);
        assert_eq!(s.player.max_energy, 3);
    }

    #[test]
    fn the_turn_starts_max_energy_survives_a_same_turn_rebuild() {
        let mut carried = CombatState {
            round: 3,
            ..CombatState::default()
        };
        carried.player.max_energy = 3;
        let mut fresh = carried.clone();
        fresh.player.max_energy = 4; // the wire's live value after a Pyre
        let carry = Carry {
            enemy_slots: Some(Vec::new()),
            rng: false,
            relic_counters: Vec::new(),
            power_counters: Vec::new(),
            turn_history: false,
        };
        transplant_hidden(&mut fresh, &carried, carry).expect("transplants");
        assert_eq!(fresh.player.max_energy, 3);
        let mut next = carried.clone();
        next.round = 4;
        next.player.max_energy = 4;
        let carry = Carry {
            enemy_slots: Some(Vec::new()),
            rng: false,
            relic_counters: Vec::new(),
            power_counters: Vec::new(),
            turn_history: false,
        };
        transplant_hidden(&mut next, &carried, carry).expect("transplants");
        assert_eq!(next.player.max_energy, 4, "a new turn reads the wire");
    }

    /// Corpus r6 seed 180000093: History Course replays a dupe of the live
    /// card, so a card upgraded after its play (Razor Tooth) comes back
    /// upgraded. The rebuilt slot loses the uid, so the carry must take the
    /// carried card's current upgrade rather than the stamp's.
    #[test]
    fn history_courses_slot_takes_the_cards_upgrade_since_the_play() {
        use sts2_core::cards::CardId;
        let carried = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &["NEOWS_FURY", "BASH", "BASH", "BASH", "BASH"].map(str::to_owned),
            80,
            80,
            &[],
            &[],
            "AXEBOTS_NORMAL",
            10,
            None,
            None,
        )
        .expect("scenario");
        let mut carried = carried;
        let fury = carried
            .cards
            .iter()
            .find(|c| c.card_id == CardId::NeowsFury)
            .map(|c| c.uid)
            .expect("fury");
        // Stamped as the play finished (not upgraded), upgraded after it.
        carried.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::NeowsFury, false, Some(fury)));
        carried.combat_log.last_attack_skill_played = Some((CardId::NeowsFury, false, Some(fury)));
        carried
            .cards
            .iter_mut()
            .find(|c| c.uid == fury)
            .expect("fury")
            .upgraded = true;
        let mut fresh = carried.clone();
        fresh.combat_log.prev_round_last_attack_skill_played = None;
        fresh.combat_log.last_attack_skill_played = None;
        transplant_hidden(&mut fresh, &carried, Carry::default()).expect("transplants");
        assert_eq!(
            fresh.combat_log.prev_round_last_attack_skill_played,
            Some((CardId::NeowsFury, true, Some(fury)))
        );
        assert_eq!(
            fresh.combat_log.last_attack_skill_played,
            Some((CardId::NeowsFury, true, Some(fury)))
        );
    }

    /// Corpus r7 seed 180000943: Razor Tooth upgraded the played Pact's End,
    /// then the enemy turn's Dampen downgraded it; History Course's dupe is
    /// the downgraded card (C# `CreateDupe` at the next turn's start). The
    /// slot must follow the fresh card, not freeze the upgrade it had at the
    /// reconcile.
    #[test]
    fn history_courses_slot_follows_a_downgrade_after_the_reconcile() {
        use sts2_core::cards::CardId;
        let mut carried = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &["PACTS_END", "BASH", "BASH", "BASH", "BASH"].map(str::to_owned),
            80,
            80,
            &[],
            &[],
            "AXEBOTS_NORMAL",
            10,
            None,
            None,
        )
        .expect("scenario");
        let pe = carried
            .cards
            .iter()
            .find(|c| c.card_id == CardId::PactsEnd)
            .map(|c| c.uid)
            .expect("pact's end");
        carried.combat_log.last_attack_skill_played = Some((CardId::PactsEnd, false, Some(pe)));
        carried
            .cards
            .iter_mut()
            .find(|c| c.uid == pe)
            .expect("pe")
            .upgraded = true;
        // The rebuild numbers the cards in reverse: no uid means the same card.
        let mut fresh = carried.clone();
        fresh.combat_log.last_attack_skill_played = None;
        let n = u32::try_from(fresh.cards.len()).expect("fits");
        let renumber =
            |u: CardUid| CardUid(std::num::NonZeroU32::new(n + 1 - u.0.get()).expect("nz"));
        for c in &mut fresh.cards {
            c.uid = renumber(c.uid);
        }
        fresh.cards.reverse();
        let p = &mut fresh.player;
        for u in p
            .hand
            .iter_mut()
            .chain(p.draw.iter_mut())
            .chain(p.discard.iter_mut())
            .chain(p.exhaust.iter_mut())
            .chain(p.play.iter_mut())
        {
            *u = renumber(*u);
        }
        transplant_hidden(&mut fresh, &carried, Carry::default()).expect("transplants");
        let slot = fresh.combat_log.last_attack_skill_played.expect("carried");
        assert_eq!(slot.2, Some(renumber(pe)), "the slot names the fresh card");
        // Dampen downgrades it during the enemy turn; the dupe is that card.
        let card = fresh
            .cards
            .get(slot.2.expect("uid").0.get() as usize - 1)
            .expect("indexed by uid");
        assert_eq!(card.card_id, CardId::PactsEnd);
    }

    /// Spire Bench r7 seeds 180000239 / 180001932: a knocked-out Test Subject
    /// telegraphs its respawn on the board; the carried creature (alive in a
    /// mirror that missed the kill) must not put its attack back.
    #[test]
    fn a_reviving_corpse_keeps_the_telegraph_the_board_shows() {
        use sts2_core::Intent;
        let carried = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &["StrikeIronclad".to_owned()],
            80,
            80,
            &[],
            &[],
            "TWO_TAILED_RATS_NORMAL",
            10,
            None,
            None,
        )
        .expect("scenario");
        let mut carried = carried;
        let (u0, u1) = (carried.enemy_uids[0], carried.enemy_uids[1]);
        for c in &mut carried.creatures {
            if c.id == u0.0 || c.id == u1.0 {
                c.intent = Some(Intent::Attack { damage: 22 });
            }
        }
        let mut fresh = carried.clone();
        for c in &mut fresh.creatures {
            if c.id == u0.0 {
                c.current_hp = 0;
                c.intent = Some(Intent::Heal); // the board's respawn telegraph
            } else if c.id == u1.0 {
                c.current_hp = 0;
                c.intent = None; // an ordinary corpse: no telegraph
            }
        }
        let slots: Vec<Option<usize>> = (0..fresh.enemy_uids.len()).map(Some).collect();
        carry_corpse_intents(&mut fresh, &carried, &slots);
        let intent = |u: sts2_core::CreatureUid| {
            fresh
                .creatures
                .iter()
                .find(|c| c.id == u.0)
                .and_then(|c| c.intent)
        };
        assert_eq!(
            intent(u0),
            Some(Intent::Heal),
            "the board's telegraph stands"
        );
        assert_eq!(
            intent(u1),
            Some(Intent::Attack { damage: 22 }),
            "a silent corpse keeps the one it died with (GH #1669)"
        );
    }

    /// Corpus seeds 180000995 (Spite) and 180000111 (Supermassive): the
    /// host saw an HP loss and a generated card before the mirror existed.
    /// The wire's values must survive a carried mirror that counts 0, and a
    /// carried count the wire lacks must still travel.
    #[test]
    fn the_wires_hp_loss_and_generated_cards_survive_a_mirror_that_missed_them() {
        let carry = || Carry {
            enemy_slots: Some(Vec::new()),
            rng: false,
            relic_counters: Vec::new(),
            power_counters: Vec::new(),
            turn_history: false,
        };
        let carried = CombatState::default();
        let mut fresh = carried.clone();
        fresh.combat_log.hp_lost_this_turn = 1;
        fresh.player.generated_cards_this_combat = 1;
        transplant_hidden(&mut fresh, &carried, carry()).expect("transplants");
        assert_eq!(fresh.combat_log.hp_lost_this_turn, 1, "Spite+ hits 3 times");
        assert_eq!(
            fresh.player.generated_cards_this_combat, 1,
            "Supermassive counts the Luminesce"
        );
        let mut seen = CombatState::default();
        seen.player.generated_cards_this_combat = 3;
        let mut blind = CombatState::default();
        transplant_hidden(&mut blind, &seen, carry()).expect("transplants");
        assert_eq!(
            blind.player.generated_cards_this_combat, 3,
            "an older host sends nothing"
        );
    }

    /// GH #1741: Orbit's cumulative energy survives a rebuild that can
    /// only derive it mod 4 from the badge.
    #[test]
    fn orbits_carried_energy_total_survives_a_rebuild() {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../sts2-wire/tests/fixtures/combat_action_ironclad.json"),
        )
        .expect("fixture");
        let mut wire: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
        // Four energy spent: one trigger done, four to the next.
        wire["player"]["powers"] = json!([{"name": "ORBIT", "amount": 1, "display_amount": 4}]);
        let counter = |st: &CombatState| {
            st.player_power_uids()
                .into_iter()
                .find_map(|pu| {
                    st.powers
                        .get_by_id(pu.0)
                        .filter(|p| p.kind == sts2_core::powers::PowerKind::Orbit)
                        .map(|p| p.counter)
                })
                .expect("orbit")
        };
        let mut fresh = sts2_wire::state_from_json(&wire).expect("parses");
        assert_eq!(counter(&fresh), 0, "the badge alone says 0 mod 4");
        let mut carried = fresh.clone();
        for pu in carried.player_power_uids() {
            if let Some(p) = carried.powers.get_mut_by_id(pu.0) {
                p.counter = 4;
            }
        }
        let carry = Carry {
            enemy_slots: None,
            rng: false,
            relic_counters: Vec::new(),
            power_counters: Vec::new(),
            turn_history: false,
        };
        transplant_hidden(&mut fresh, &carried, carry).expect("transplants");
        assert_eq!(counter(&fresh), 4);
    }

    /// Two Orbits are two instances, each with its own energy total: both
    /// survive the rebuild, not just the first.
    #[test]
    fn every_orbits_carried_energy_total_survives_a_rebuild() {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../sts2-wire/tests/fixtures/combat_action_ironclad.json"),
        )
        .expect("fixture");
        let mut wire: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
        wire["player"]["powers"] = json!([
            {"name": "ORBIT", "amount": 1, "display_amount": 4},
            {"name": "ORBIT", "amount": 1, "display_amount": 4},
        ]);
        let counters = |st: &CombatState| -> Vec<i32> {
            st.player_power_uids()
                .into_iter()
                .filter_map(|pu| {
                    st.powers
                        .get_by_id(pu.0)
                        .filter(|p| p.kind == sts2_core::powers::PowerKind::Orbit)
                        .map(|p| p.counter)
                })
                .collect()
        };
        let mut fresh = sts2_wire::state_from_json(&wire).expect("parses");
        assert_eq!(counters(&fresh).len(), 2, "two Orbit instances");
        assert!(
            counters(&fresh).iter().all(|c| *c < 4),
            "the badges say the total mod 4"
        );
        let mut carried = fresh.clone();
        for pu in carried.player_power_uids() {
            if let Some(p) = carried.powers.get_mut_by_id(pu.0) {
                p.counter = 4;
            }
        }
        let carry = Carry {
            enemy_slots: None,
            rng: false,
            relic_counters: Vec::new(),
            power_counters: Vec::new(),
            turn_history: false,
        };
        transplant_hidden(&mut fresh, &carried, carry).expect("transplants");
        assert_eq!(counters(&fresh), vec![4, 4]);
    }

    /// GH corpus-compare seed 180000794: a Strike Hellraiser plays as the
    /// hand is drawn counts for Juggling, not in the board's History bucket.
    /// The carried count stands within the round; a new round's board wins.
    #[test]
    fn jugglings_carried_count_stands_within_the_round() {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../sts2-wire/tests/fixtures/combat_action_ironclad.json"),
        )
        .expect("fixture");
        let mut wire: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
        wire["player"]["powers"] = json!([{"name": "JUGGLING", "amount": 1}]);
        wire["player"]["cards_played_this_turn"] = json!([0, 1, 0, 0, 0, 0]);
        let count = |st: &CombatState| -> Option<i32> {
            st.player_power_uids().into_iter().find_map(|pu| {
                st.powers
                    .get_by_id(pu.0)
                    .filter(|p| p.kind == sts2_core::powers::PowerKind::Juggling)
                    .map(|p| p.counter)
            })
        };
        let fresh = sts2_wire::state_from_json(&wire).expect("parses");
        assert_eq!(count(&fresh), Some(0), "the board's bucket: no Attack");
        let mut carried = fresh.clone();
        for pu in carried.player_power_uids() {
            if let Some(p) = carried.powers.get_mut_by_id(pu.0) {
                if p.kind == sts2_core::powers::PowerKind::Juggling {
                    p.counter = 1;
                }
            }
        }
        let carry = || Carry {
            enemy_slots: None,
            rng: false,
            relic_counters: Vec::new(),
            power_counters: Vec::new(),
            turn_history: false,
        };
        let mut same_round = fresh.clone();
        transplant_hidden(&mut same_round, &carried, carry()).expect("transplants");
        assert_eq!(count(&same_round), Some(1), "the Hellraiser Strike counts");
        let mut next_round = fresh.clone();
        next_round.round += 1;
        transplant_hidden(&mut next_round, &carried, carry()).expect("transplants");
        assert_eq!(
            count(&next_round),
            Some(0),
            "a new round starts from the board"
        );
    }

    /// The board lists a lone bot before the Fabricator, and the rebuild
    /// can only guess it holds bot1. The carried fight knows: it spawned
    /// that bot into bot2 while another held bot1. Its slot survives, so the
    /// next bot takes bot1, in front, as in the game (Spire Bench seed
    /// 170000159: two Noisebots in swapped places).
    #[test]
    fn a_carried_enemy_keeps_the_slot_the_mirror_gave_it() {
        sts2_content::cards::dispatch::ensure_installed();
        let enemy = |kind: &str, uid: i32, mv: &str| {
            json!({"kind": kind, "uid": uid, "hp": 20, "max_hp": 20, "block": 0,
                "powers": [], "intents": [], "intent_history": [], "move_state": mv,
                "is_dead": false, "is_escaped": false})
        };
        let card =
            json!({"id": "STRIKE_IRONCLAD", "stable_idx": 1, "kind": "card", "upgraded": false});
        let wire = json!({
            "wire_version": 1, "phase": "PlayerPlayPhase", "round": 3, "side": "Player",
            "is_elite": false, "turn": 3, "combat_id": 0, "rng": {"seed": 1, "ascension": 0},
            "player": {"class": "ironclad", "hp": 80, "max_hp": 80, "block": 0, "energy": 3,
                       "max_energy": 3, "energy_regen_per_turn": 3, "gold": 0, "relics": [],
                       "powers": [], "potions": []},
            "deck": {"hand": [card], "draw_pile": [], "discard_pile": [], "exhaust_pile": []},
            "encounter": [enemy("Noisebot", 1, "NOISE_MOVE"), enemy("Fabricator", 2, "FABRICATE_MOVE")],
        });
        let carry = || Carry {
            enemy_slots: None,
            rng: false,
            relic_counters: Vec::new(),
            power_counters: Vec::new(),
            turn_history: false,
        };
        let fresh = sts2_wire::state_from_json(&wire).expect("parses");
        let bot = fresh.enemy_uids[0];
        assert_eq!(fresh.enemy_slot(bot), Some(0), "the rebuild's guess");
        let mut carried = fresh.clone();
        carried.set_enemy_slot(bot, 1);
        let mut next = fresh.clone();
        transplant_hidden(&mut next, &carried, carry()).expect("transplants");
        assert_eq!(next.enemy_slot(bot), Some(1));
        assert_eq!(next.first_free_enemy_slot(5), Some(0));
        // A carried slot the board's order contradicts does not survive: the
        // bot is listed before the Fabricator, so it cannot hold bot3.
        carried.set_enemy_slot(bot, 3);
        let mut next = fresh.clone();
        transplant_hidden(&mut next, &carried, carry()).expect("transplants");
        assert_eq!(next.enemy_slot(bot), Some(0));
    }

    fn with_enemies(uids: &[u32]) -> CombatState {
        CombatState {
            enemy_uids: uids
                .iter()
                .map(|&u| sts2_core::CreatureUid(std::num::NonZeroU32::new(u).expect("non-zero")))
                .collect(),
            ..CombatState::default()
        }
    }

    /// GH #1695: a wire that lists the corpse last ([Obscura, dead
    /// Parafright]) rebuilds as uids 2, 3 but in C# slot order [3, 2]; each
    /// fresh enemy keeps its own carried slot.
    #[test]
    fn the_alignment_follows_the_rebuilds_slot_order() {
        let wire =
            json!({"encounter": [{"kind": "TheObscura"}, {"kind": "Parafright", "is_dead": true}]});
        let fresh = with_enemies(&[3, 2]);
        let slots = alignment_in_fresh_order(&fresh, &wire, &[Some(1), Some(0)]);
        assert_eq!(slots, vec![Some(0), Some(1)]);
    }

    #[test]
    fn an_escaped_entry_takes_no_uid() {
        let wire =
            json!({"encounter": [{"kind": "A", "is_escaped": true}, {"kind": "B"}, {"kind": "C"}]});
        let fresh = with_enemies(&[2, 3]);
        let slots = alignment_in_fresh_order(&fresh, &wire, &[Some(0), Some(2), None]);
        assert_eq!(slots, vec![Some(2), None]);
    }
}
