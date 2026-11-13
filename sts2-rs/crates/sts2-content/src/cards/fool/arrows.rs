//! Fool cards built on the arrow mechanic (`sts2_core::fool`, core lane):
//! the Shoot attacks, the impale readers and the dislodgers. Values from
//! the v0.113.0 decomp `Models.Cards/<Name>.cs`.
use smallvec::SmallVec;
use sts2_core::card_effects::{self, deal_damage_player_to_enemy, EffectResult};
use sts2_core::cards::{CardTag, PileKind};
use sts2_core::fool;
use sts2_core::hooks::{CreatureRef, ValueProps};
use sts2_core::{CardUid, CombatState, CreatureUid, PowerKind};

use super::helpers::{apply_to_enemy, apply_to_player, current_block, current_damage, var};

fn target(t: Option<CreatureUid>, card: &str) -> CreatureUid {
    t.unwrap_or_else(|| panic!("{card}: AnyEnemy-targeted"))
}

/// `ArrowCmd.ShootArrowAtTarget` with one `FromArrow` attack of the card's
/// damage: FoolishShot, QuickShot, HeavyShot (its doubled Aim is in the
/// core `AimPower` arm), SkyShot, DivineShot.
pub(crate) fn play_shot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "shot");
    let dmg = current_damage(state);
    let _ = fool::shoot_arrow_at_target(state, t, dmg, 1);
    EffectResult::Resolved
}

/// `RapidShot.cs`: one shot, 3 (+1) hits of 3.
pub(crate) fn play_rapid_shot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "RapidShot");
    let dmg = current_damage(state);
    let hits = u32::try_from(var(state, 3, 4)).unwrap_or(0);
    let _ = fool::shoot_arrow_at_target(state, t, dmg, hits);
    EffectResult::Resolved
}

/// `RepeatShot.cs`: X-cost; when X > 0, one shot of X hits.
pub(crate) fn play_repeat_shot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "RepeatShot");
    let x = card_effects::resolving_card_x_value(state);
    if x > 0 {
        let dmg = current_damage(state);
        let _ = fool::shoot_arrow_at_target(state, t, dmg, u32::try_from(x).unwrap_or(0));
    } else {
        // No AttackCommand at X == 0: the latched Vigor/Gigantification stay.
        fool::attack_skipped(state);
    }
    EffectResult::Resolved
}

/// `HobblingShot.cs`: shoot; only if an arrow was shot, Weak then
/// Vulnerable 1 (+1) on the target.
pub(crate) fn play_hobbling_shot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "HobblingShot");
    let dmg = current_damage(state);
    if fool::shoot_arrow_at_target(state, t, dmg, 1) {
        let n = var(state, 1, 2);
        apply_to_enemy(state, t, PowerKind::Weak, n);
        apply_to_enemy(state, t, PowerKind::Vulnerable, n);
    }
    EffectResult::Resolved
}

/// `MelodicShot.cs`: one shot of `1 + MelodiesThisTurn` hits.
pub(crate) fn play_melodic_shot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "MelodicShot");
    let dmg = current_damage(state);
    let hits = 1 + u32::try_from(fool::melodies_this_turn(state)).unwrap_or(0);
    let _ = fool::shoot_arrow_at_target(state, t, dmg, hits);
    EffectResult::Resolved
}

/// `TheShot.cs`: one shot of `10 (+3) x CardsExhaustedByFlorescence`
/// (`CalculationBase` 0), the count kept in the card's `Generic` var.
pub(crate) fn play_the_shot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "TheShot");
    let count = card_effects::resolving_card_inst(state)
        .map_or(0, |c| c.combat_var(sts2_core::cards::VarKey::Generic));
    let dmg = var(state, 10, 13).saturating_mul(count);
    let _ = fool::shoot_arrow_at_target(state, t, dmg, 1);
    EffectResult::Resolved
}

/// `BlossomShot.cs`: on a living target, shoot; the action is one
/// `FromArrow` attack, then `Apply<StretchPower>(TotalDamage +
/// OverkillDamage)`.
pub(crate) fn play_blossom_shot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "BlossomShot");
    let dmg = current_damage(state);
    let _ = fool::shoot_arrow_at_target_with(state, t, |state| {
        fool::begin_arrow_attack(state);
        let card = card_effects::originating_card_id(state);
        let out = sts2_core::damage::apply_damage_full(
            state,
            CreatureRef::Player,
            t,
            dmg,
            ValueProps::MOVE.union(ValueProps::FROM_ARROW),
            card,
        );
        fool::end_arrow_attack(state);
        apply_to_player(state, PowerKind::Stretch, out.total_damage + out.overkill);
    });
    EffectResult::Resolved
}

/// `BowSmack.cs` / `KiHop.cs`: read `CreatureIsImpaledByArrow` before the
/// attack, attack, then energy 1 / `Draw(2 (+1))` if it was impaled.
pub(crate) fn play_bow_smack(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "BowSmack");
    let was = fool::creature_is_impaled(state, t);
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    if was {
        state.gain_energy(1);
    }
    EffectResult::Resolved
}

/// See [`play_bow_smack`].
pub(crate) fn play_ki_hop(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "KiHop");
    let was = fool::creature_is_impaled(state, t);
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    if was {
        let n = var(state, 2, 3);
        let was_suspended = state.pending_draw_resume.is_some();
        let _ = sts2_core::cards::draw_via_state(state, n);
        crate::cards::draw_then::after_draw(state, was_suspended, sts2_core::DrawThen::None);
    }
    EffectResult::Resolved
}

/// `Nail.cs`: only against an impaled target: attack, then Weak and
/// Vulnerable 2 (+1).
pub(crate) fn play_nail(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "Nail");
    if fool::creature_is_impaled(state, t) {
        let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
        let n = var(state, 2, 3);
        apply_to_enemy(state, t, PowerKind::Weak, n);
        apply_to_enemy(state, t, PowerKind::Vulnerable, n);
    } else {
        fool::attack_skipped(state);
    }
    EffectResult::Resolved
}

/// `while (CreatureIsImpaledByPlayersArrow) DislodgeArrow(target, player)`.
/// Returns whether the target was impaled at the start.
fn dislodge_until_free(state: &mut CombatState, t: CreatureUid) -> bool {
    if !fool::creature_is_impaled(state, t) {
        return false;
    }
    while fool::creature_is_impaled(state, t) {
        if fool::arrows_in(state, t) == 0 {
            // An arrowless Impaled: the card-form dislodge drops it.
            fool::dislodge_all(state, t);
            break;
        }
        fool::dislodge_arrow(state, t);
    }
    true
}

/// `Uproot.cs`: dislodge every arrow, then Vulnerable 3 (+1).
pub(crate) fn play_uproot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "Uproot");
    if dislodge_until_free(state, t) {
        let n = var(state, 3, 4);
        apply_to_enemy(state, t, PowerKind::Vulnerable, n);
    }
    EffectResult::Resolved
}

/// `Wrest.cs`: dislodge every arrow, then `Draw(2 (+1))`.
pub(crate) fn play_wrest(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "Wrest");
    if dislodge_until_free(state, t) {
        let n = var(state, 2, 3);
        let was_suspended = state.pending_draw_resume.is_some();
        let _ = sts2_core::cards::draw_via_state(state, n);
        crate::cards::draw_then::after_draw(state, was_suspended, sts2_core::DrawThen::None);
    }
    EffectResult::Resolved
}

/// The discard-pile cards passing `filter`, in pile order.
fn discard_where(
    state: &CombatState,
    filter: impl Fn(&sts2_core::cards::Card) -> bool,
) -> SmallVec<[CardUid; 16]> {
    state
        .player
        .discard
        .iter()
        .copied()
        .filter(|u| filter(&state.cards[u.0.get() as usize - 1]))
        .collect()
}

/// `CardSelectCmd.FromCombatPile(Discard, prefs(1), filter)`: none -> nothing;
/// one -> taken without a prompt; more -> a 1-pick select of `kind` (the
/// discard pile is not re-sorted). `take` moves an auto-taken card.
fn select_from_discard(
    state: &mut CombatState,
    candidates: &[CardUid],
    kind: sts2_core::ChoiceKind,
    take: fn(&mut CombatState, CardUid),
) {
    if sts2_core::combat_loop::combat_is_ending(state) {
        return;
    }
    match candidates {
        [] => {}
        [uid] => take(state, *uid),
        _ => {
            let options: Vec<sts2_core::ChoiceOption> = candidates
                .iter()
                .map(|&uid| {
                    let c = &state.cards[uid.0.get() as usize - 1];
                    sts2_core::ChoiceOption {
                        card_id: c.card_id,
                        upgraded: c.upgraded,
                        card_uid: Some(uid),
                        source_pile: Some(PileKind::Discard),
                    }
                })
                .collect();
            state.request_card_choice(kind, options, 1, 1, false);
        }
    }
}

/// `CardPileCmd.Add(card, Hand)` from the discard pile (to discard again
/// when the hand is full).
fn discard_to_hand(state: &mut CombatState, uid: CardUid) {
    if let Some(pos) = state.player.discard.iter().position(|u| *u == uid) {
        let card = state.player.discard.remove(pos);
        let _ = sts2_core::cards::try_add_to_hand(&mut state.player, card);
    }
}

/// `CardPileCmd.Add(card, Draw, Top)` from the discard pile.
fn discard_to_draw_top(state: &mut CombatState, uid: CardUid) {
    if let Some(pos) = state.player.discard.iter().position(|u| *u == uid) {
        state.player.discard.remove(pos);
        state.player.draw.insert(0, uid);
        sts2_core::cards::pin_top(&mut state.player, uid);
    }
}

/// `ArrowStab.cs`: attack; then, with an arrow ready and room in hand, a
/// Shoot card from the discard pile to hand.
pub(crate) fn play_arrow_stab(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "ArrowStab");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    if fool::player_has_any_arrows(state) && state.player.hand.len() < sts2_core::cards::HAND_LIMIT
    {
        let shoot = discard_where(state, |c| c.has_tag(CardTag::Shoot));
        select_from_discard(
            state,
            &shoot,
            sts2_core::ChoiceKind::AddFromDiscardToHand,
            discard_to_hand,
        );
    }
    EffectResult::Resolved
}

/// `Recall.cs`: against a target impaled by the player's arrow, dislodge
/// every arrow, then any discard card to the top of the draw pile.
pub(crate) fn play_recall(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "Recall");
    if dislodge_until_free(state, t) {
        let all = discard_where(state, |_| true);
        select_from_discard(
            state,
            &all,
            sts2_core::ChoiceKind::PutBack,
            discard_to_draw_top,
        );
    }
    EffectResult::Resolved
}

/// `ChoFirm.cs`: `GainBlock` (6, +2) once, twice with an arrow ready.
pub(crate) fn play_cho_firm(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let gains = if fool::player_has_any_arrows(state) {
        2
    } else {
        1
    };
    for _ in 0..gains {
        let b = current_block(state);
        let _ = card_effects::gain_block_player(state, b);
    }
    EffectResult::Resolved
}

/// `Quiver.cs`: `ArrowCmd.GainArrowSlot` 1 (+1) times.
pub(crate) fn play_quiver(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    for _ in 0..var(state, 1, 2) {
        fool::gain_arrow_slot(state);
    }
    EffectResult::Resolved
}

/// `Florescence.cs`: exhaust every Shoot card outside the Exhaust pile
/// (`AllCards` order: Hand, Draw, Discard, Play), then a TheShot (upgraded
/// with Florescence) carrying that count to hand.
pub(crate) fn play_florescence(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let p = &state.player;
    let shoot: SmallVec<[CardUid; 16]> = p
        .hand
        .iter()
        .chain(p.draw.iter())
        .chain(p.discard.iter())
        .chain(p.play.iter())
        .copied()
        .filter(|u| state.cards[u.0.get() as usize - 1].has_tag(CardTag::Shoot))
        .collect();
    for &uid in &shoot {
        let _ = card_effects::exhaust_card_uid(state, uid);
    }
    let mut shot = super::helpers::fool_card(sts2_core::cards::CardId::TheShot);
    let count = i32::try_from(shoot.len()).unwrap_or(i32::MAX);
    shot.set_combat_var(sts2_core::cards::VarKey::Generic, count);
    let upgraded = super::helpers::upgraded(state);
    let uid = card_effects::spawn_card_into_hand(state, shot);
    if upgraded {
        let _ = card_effects::upgrade_card_uid(state, uid);
    }
    EffectResult::Resolved
}

/// `WhistlingArrows.cs` play body: `GainBlock` (3, +1). Its
/// `AfterArrowShot` discard -> hand hook is the core lane's (`fool.rs`).
pub(crate) fn play_whistling_arrows(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    let b = current_block(state);
    let _ = card_effects::gain_block_player(state, b);
    EffectResult::Resolved
}

/// `BlotOutTheSky.cs`: `GetDistinctForCombat(own character pool where
/// Shoot, 1, Rng.CombatCardGeneration)` (the shuffle reads the stream even
/// when nothing qualifies), the card free this turn, to hand.
pub(crate) fn play_blot_out_the_sky(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    let Some(own) = state.player.character else {
        return EffectResult::Resolved;
    };
    let mut pool: SmallVec<[sts2_core::cards::CardId; 16]> =
        crate::cards::colorless::helpers::fight_pool(state, own)
            .iter()
            .filter(|s| crate::cards::status::tags_for_card_id(s.id).contains(CardTag::Shoot))
            .map(|s| s.id)
            .collect();
    state
        .run_rng
        .combat_card_generation
        .shuffle_for(state.game_version, pool.as_mut_slice());
    if let Some(&id) = pool.first() {
        if let Some(mut card) = crate::cards::spec_index::card_pod_from_id(id, false) {
            card.set_to_free_this_turn();
            let _ = card_effects::spawn_card_into_hand(state, card);
        }
    }
    EffectResult::Resolved
}

/// `TrickShot.cs`: one shot. `GetResultLocationForCardPlay` (read before
/// `OnPlay`) sends a Discard-bound TrickShot back to the hand when the
/// player had an arrow; the flag for that read is kept in the card's
/// `Generic` var while it resolves (core `discard_result_goes_to_hand`).
/// A replay keeps the first play's reading.
pub(crate) fn play_trick_shot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "TrickShot");
    if fool::player_has_any_arrows(state) {
        if let Some(uid) = card_effects::resolving_card_uid(state) {
            state.cards[uid.0.get() as usize - 1]
                .set_combat_var(sts2_core::cards::VarKey::Generic, 1);
        }
    }
    let dmg = current_damage(state);
    let _ = fool::shoot_arrow_at_target(state, t, dmg, 1);
    EffectResult::Resolved
}

/// `RainShot.cs`: on a living target, shoot; the action is one `Move |
/// FromArrow` `CreatureCmd.Damage` (inside an attack context), then
/// `PowerCmd.Remove<AimPower>` and `Remove<QuietusPower>`, then
/// `TotalDamage + OverkillDamage` to every other hittable enemy, `Move`
/// with `ModifyDamageHookType.Cap` (only damage caps apply).
pub(crate) fn play_rain_shot(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "RainShot");
    let dmg = current_damage(state);
    let _ = fool::shoot_arrow_at_target_with(state, t, |state| {
        fool::begin_arrow_attack(state);
        let card = card_effects::originating_card_id(state);
        let out = sts2_core::damage::apply_damage_full(
            state,
            CreatureRef::Player,
            t,
            dmg,
            ValueProps::MOVE.union(ValueProps::FROM_ARROW),
            card,
        );
        fool::remove_player_power(state, PowerKind::Aim);
        fool::remove_player_power(state, PowerKind::Quietus);
        let splash = out.total_damage + out.overkill;
        let others: SmallVec<[CreatureUid; 4]> = state
            .enemy_uids
            .iter()
            .copied()
            .filter(|&e| {
                e != t
                    && state
                        .creatures
                        .iter()
                        .any(|c| c.id == e.0 && c.current_hp > 0)
            })
            .collect();
        // `CreatureCmd.Damage` over a list resolves every target's amount
        // before the first hit lands.
        let finals: SmallVec<[(CreatureUid, i32); 4]> = others
            .iter()
            .map(|&e| {
                let ctx = sts2_core::hooks::DamageCtx {
                    dealer: CreatureRef::Player,
                    target: CreatureRef::Enemy(e),
                    props: ValueProps::MOVE,
                };
                let cap = sts2_core::hooks::dispatch_modify_damage_cap(state, &ctx);
                (e, splash.min(cap).max(0))
            })
            .collect();
        for (e, amount) in finals {
            let _ = sts2_core::damage::apply_damage_with_final(
                state,
                CreatureRef::Player,
                e,
                amount,
                ValueProps::MOVE,
            );
        }
        fool::end_arrow_attack(state);
    });
    EffectResult::Resolved
}
