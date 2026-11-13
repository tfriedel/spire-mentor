//! Fool melody cards (`MelodyCmd`, core lane's `sts2_core::fool`). Values
//! from the v0.113.0 decomp `Models.Cards/<Name>.cs`. Coda's and ChoLimbs'
//! cost reductions live in the core cost pipeline (`hooks::card_self_cost`).
use smallvec::SmallVec;
use sts2_core::card_effects::{self, deal_damage_player_to_enemy, EffectResult};
use sts2_core::cards::{CardTag, PileKind};
use sts2_core::fool;
use sts2_core::{CardUid, CombatState, CreatureUid, DrawThen, PowerKind};

use super::helpers::{apply_to_player, current_block, current_damage, damage_all_enemies, var};

fn target(t: Option<CreatureUid>, card: &str) -> CreatureUid {
    t.unwrap_or_else(|| panic!("{card}: AnyEnemy-targeted"))
}

/// `Coda.cs`: one attack (40, +15).
pub(crate) fn play_coda(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "Coda");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    EffectResult::Resolved
}

/// `ChoLimbs.cs`: `GainBlock` (5, +2) twice.
pub(crate) fn play_cho_limbs(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    for _ in 0..2 {
        let b = current_block(state);
        let _ = card_effects::gain_block_player(state, b);
    }
    EffectResult::Resolved
}

/// `HarmonicShield.cs`: `GainBlock(0 + 5 (+2) x MelodiesThisTurn)`, Move.
pub(crate) fn play_harmonic_shield(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    let b = var(state, 5, 7) * fool::melodies_this_turn(state);
    let _ = card_effects::gain_block_player(state, b);
    EffectResult::Resolved
}

/// `RhythmStrike.cs`: `1 + MelodiesThisTurn` hits of 4 (+1).
pub(crate) fn play_rhythm_strike(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "RhythmStrike");
    let hits = 1 + fool::melodies_this_turn(state);
    for _ in 0..hits {
        if !state
            .creatures
            .iter()
            .any(|c| c.id == t.0 && c.current_hp > 0)
        {
            break;
        }
        let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    }
    EffectResult::Resolved
}

/// `SteadyCadence.cs`: attack (10, +3), then `Draw(MelodiesThisTurn)`.
pub(crate) fn play_steady_cadence(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "SteadyCadence");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    let n = fool::melodies_this_turn(state);
    if n > 0 {
        let was_suspended = state.pending_draw_resume.is_some();
        let _ = sts2_core::cards::draw_via_state(state, n);
        crate::cards::draw_then::after_draw(state, was_suspended, DrawThen::None);
    }
    EffectResult::Resolved
}

/// `KiRest.cs`: `EnergyNextTurn` 5 (+1); `RetainHand` 1 once Wah or Cho was
/// played this turn.
pub(crate) fn play_ki_rest(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let e = var(state, 5, 6);
    apply_to_player(state, PowerKind::EnergyNextTurn, e);
    if fool::played_this_turn(state, CardTag::Wah) || fool::played_this_turn(state, CardTag::Cho) {
        apply_to_player(state, PowerKind::RetainHand, 1);
    }
    EffectResult::Resolved
}

/// `WahBlast.cs`: all enemies, (14, +4) x 2 hits once Cho or Ki was played
/// this turn, else 1.
pub(crate) fn play_wah_blast(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let twice =
        fool::played_this_turn(state, CardTag::Cho) || fool::played_this_turn(state, CardTag::Ki);
    let hits = if twice { 2 } else { 1 };
    for _ in 0..hits {
        if sts2_core::combat_loop::combat_is_ending(state) {
            break;
        }
        damage_all_enemies(state, current_damage(state));
    }
    EffectResult::Resolved
}

/// `WahSymphony.cs`: all enemies, `6 + 4 (+2) x` the player's cards (every
/// combat pile) carrying Wah, Cho or Ki.
pub(crate) fn play_wah_symphony(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let p = &state.player;
    let notes = p
        .hand
        .iter()
        .chain(p.draw.iter())
        .chain(p.discard.iter())
        .chain(p.exhaust.iter())
        .chain(p.play.iter())
        .filter(|u| state.cards[u.0.get() as usize - 1].tags.has_melody())
        .count();
    let n = i32::try_from(notes).unwrap_or(i32::MAX);
    let dmg = 6 + var(state, 4, 6).saturating_mul(n);
    damage_all_enemies(state, dmg);
    EffectResult::Resolved
}

/// `ChoChorus.cs`: `GainBlock` (12, +3), then `CardCmd.AutoPlay` each hand
/// card carrying Wah or Ki (captured before the first play), in hand order,
/// suspending on a surfaced select as BeatDown does.
pub(crate) fn play_cho_chorus(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let b = current_block(state);
    let _ = card_effects::gain_block_player(state, b);
    let list: SmallVec<[CardUid; 8]> = state
        .player
        .hand
        .iter()
        .copied()
        .filter(|u| {
            let tags = state.cards[u.0.get() as usize - 1].tags;
            tags.contains(CardTag::Wah) || tags.contains(CardTag::Ki)
        })
        .collect();
    autoplay_list(state, &list);
    EffectResult::Resolved
}

/// `CardCmd.AutoPlay(item, null)` over a fixed list, parking the rest of the
/// list on a surfaced select (resumed by [`autoplay_list_resume`]).
pub(crate) fn autoplay_list(state: &mut CombatState, cards: &[CardUid]) {
    for (i, &uid) in cards.iter().enumerate() {
        sts2_core::combat_loop::auto_play_specific_card_from_discard(state, uid, false);
        if let Some(resume) = state.pending_autoplay_resume.as_mut() {
            resume.continuation = Some(sts2_core::AutoPlayLoopCont {
                resume_fn: autoplay_list_resume,
                iters: 0,
                uids: cards[i + 1..].iter().copied().collect(),
                force_exhaust: false,
                then: None,
            });
            return;
        }
    }
}

fn autoplay_list_resume(state: &mut CombatState, cont: &sts2_core::AutoPlayLoopCont) {
    autoplay_list(state, &cont.uids);
}

/// `ChoKi.cs`: read Stretch, `GainBlock` (8, +3), then Stretch again by the
/// amount read (doubling it) when it was positive.
pub(crate) fn play_cho_ki(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let stretch = state.player_power_amount(PowerKind::Stretch);
    let b = current_block(state);
    let _ = card_effects::gain_block_player(state, b);
    apply_to_player(state, PowerKind::Stretch, stretch);
    EffectResult::Resolved
}

/// `KiChant.cs`: with an empty draw pile `ShuffleIfNecessary`; with fewer
/// than 3 (+1) cards and a non-empty discard, `Shuffle` and put the cards
/// that were in the draw pile back on top in their order. Then
/// `FromCombatPile(Draw, 1, top 3 (+1))` to hand, and `CardCmd.Discard` the
/// other top cards still in the draw pile.
///
/// A Stratagem pick opened by the shuffle ends the play there (approximation:
/// C# awaits that pick, then goes on).
pub(crate) fn play_ki_chant(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let count = usize::try_from(var(state, 3, 4)).unwrap_or(0);
    let had_choice = state.pending_choice.is_some();
    if state.player.draw.is_empty() {
        if !state.player.discard.is_empty() {
            state.shuffle_draw_pile();
        }
    } else if state.player.draw.len() < count && !state.player.discard.is_empty() {
        let already: SmallVec<[CardUid; 4]> = state.player.draw.iter().copied().collect();
        state.shuffle_draw_pile();
        for (i, &uid) in already.iter().enumerate() {
            if let Some(pos) = state.player.draw.iter().position(|&u| u == uid) {
                state.player.draw.remove(pos);
                state.player.draw.insert(i, uid);
            }
        }
        for &uid in already.iter().rev() {
            sts2_core::cards::pin_top(&mut state.player, uid);
        }
    }
    if !had_choice && state.pending_choice.is_some() {
        return EffectResult::Resolved;
    }
    let mut top: SmallVec<[CardUid; 4]> = state.player.draw.iter().take(count).copied().collect();
    match top.len() {
        0 => {}
        1 => {
            let uid = top[0];
            if let Some(pos) = state.player.draw.iter().position(|&u| u == uid) {
                state.player.draw.remove(pos);
                sts2_core::cards::unpin(&mut state.player, uid);
                let _ = sts2_core::cards::try_add_to_hand(&mut state.player, uid);
            }
        }
        _ => {
            crate::cards::colorless::helpers::sort_draw_select(state, &mut top);
            let options: Vec<sts2_core::ChoiceOption> = top
                .iter()
                .map(|&uid| {
                    let c = &state.cards[uid.0.get() as usize - 1];
                    sts2_core::ChoiceOption {
                        card_id: c.card_id,
                        upgraded: c.upgraded,
                        card_uid: Some(uid),
                        source_pile: Some(PileKind::Draw),
                    }
                })
                .collect();
            state.open_hand_choice(sts2_core::PendingChoice {
                kind: sts2_core::ChoiceKind::AddFromDrawByType,
                options,
                min_select: 1,
                max_select: 1,
                can_skip: false,
                post_effect: Some(sts2_core::PostChoiceEffect::DiscardUnpickedFromDraw),
                picked: SmallVec::new(),
            });
        }
    }
    EffectResult::Resolved
}
