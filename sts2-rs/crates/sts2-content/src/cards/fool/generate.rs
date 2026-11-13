//! Fool cards that move, generate or transform cards with pre-v0.113
//! machinery: TheWorld, TheHierophant, TheLovers, Judgement, Growth and
//! PunchOfJoy. Values from the v0.113.0 decomp `Models.Cards/<Name>.cs`.
use smallvec::SmallVec;
use sts2_core::card_effects::{self, deal_damage_player_to_enemy, EffectResult};
use sts2_core::cards::CardId;
use sts2_core::energy_cost::CostExpiry;
use sts2_core::potions::CharacterId;
use sts2_core::{CardUid, CombatState, CreatureUid, PowerKind};

use super::helpers::{apply_to_player, current_damage, fool_card, upgraded, var};

/// C# `CardPile.MaxCardsInHand - hand.Count`, floored at 0. The played card
/// is already on the Play pile, so it does not count.
fn free_hand_slots(state: &CombatState) -> usize {
    sts2_core::cards::HAND_LIMIT.saturating_sub(state.player.hand.len())
}

/// `TheWorld.cs`: `Discard.TakeRandom(min(5 (+2), free slots),
/// Rng.CombatCardSelection)` moved to hand. `TakeRandom` shuffles the whole
/// discard pile (even when nothing is taken) and keeps the first `count`.
pub(crate) fn play_the_world(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let count = usize::try_from(var(state, 5, 7))
        .unwrap_or(0)
        .min(free_hand_slots(state));
    let mut pool: SmallVec<[CardUid; 32]> = state.player.discard.iter().copied().collect();
    state
        .run_rng
        .combat_card_selection
        .shuffle_for(state.game_version, pool.as_mut_slice());
    pool.truncate(count);
    for uid in pool {
        state.move_card_to_hand(uid);
    }
    EffectResult::Resolved
}

/// `TheHierophant.cs`: `GetDistinctForCombat(ColorlessCardPool, 3 (+1),
/// Rng.CombatCardGeneration)`; each card costs 1 less this turn or until
/// played (`EnergyCost.AddThisTurnOrUntilPlayed(-1)`), then goes to hand.
pub(crate) fn play_the_hierophant(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    let count = usize::try_from(var(state, 3, 4)).unwrap_or(0);
    let ids = crate::cards::colorless::distinct_colorless_modifier_ids(state, count);
    for id in ids {
        if let Some(mut card) = crate::cards::spec_index::card_pod_from_id(id, false) {
            card.cost.add(-1, CostExpiry::TurnOrPlayed);
            let _ = card_effects::spawn_card_into_hand(state, card);
        }
    }
    EffectResult::Resolved
}

/// `ModelDb.AllCharacters` order; v0.113.0 adds The Fool last.
const ALL_CHARACTERS: [CharacterId; 6] = [
    CharacterId::Ironclad,
    CharacterId::Silent,
    CharacterId::Regent,
    CharacterId::Necrobinder,
    CharacterId::Defect,
    CharacterId::Fool,
];

/// `TheLovers.cs`: every unlocked character pool except the owner's (kept
/// when it is the only one), `GetForCombat(.., MaxCardsInHand - hand,
/// Rng.CombatCardGeneration)` = one `NextItem` per card over the filtered
/// concatenation, each upgraded when TheLovers is, then all to hand.
pub(crate) fn play_the_lovers(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let own = state.player.character;
    let mut pool: SmallVec<[CardId; 512]> = SmallVec::new();
    for ch in ALL_CHARACTERS {
        if Some(ch) == own {
            continue;
        }
        pool.extend(
            crate::cards::colorless::helpers::fight_pool(state, ch)
                .iter()
                .map(|s| s.id),
        );
    }
    let count = free_hand_slots(state);
    if pool.is_empty() {
        return EffectResult::Resolved;
    }
    let up = upgraded(state);
    let len = i32::try_from(pool.len()).expect("pool fits in i32");
    let picks: SmallVec<[CardId; 10]> = (0..count)
        .map(|_| {
            let i = state
                .run_rng
                .combat_card_generation
                .next_int_exclusive(0, len);
            pool[usize::try_from(i).expect("non-negative index")]
        })
        .collect();
    for id in picks {
        if let Some(card) = crate::cards::spec_index::card_pod_from_id(id, up) {
            let _ = card_effects::spawn_card_into_hand(state, card);
        }
    }
    EffectResult::Resolved
}

/// `Judgement.cs`: when the target is alive at or below 30 (+10) HP,
/// `CreatureCmd.SetCurrentHp(target, 0)`, which kills it through
/// `CreatureCmd.Kill` (no damage pipeline: the same direct HP zero as
/// Doom's kill).
pub(crate) fn play_judgement(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = t.expect("Judgement: AnyEnemy-targeted");
    let threshold = var(state, 30, 40);
    let hp = state
        .creatures
        .iter()
        .find(|c| c.id == t.0)
        .map_or(0, |c| c.current_hp);
    if hp > 0 && hp <= threshold {
        let target = sts2_core::hooks::CreatureRef::Enemy(t);
        let props = sts2_core::hooks::ValueProps::UNBLOCKABLE
            .union(sts2_core::hooks::ValueProps::UNPOWERED);
        let _ = sts2_core::damage::apply_damage_direct_kill(state, target, t, hp, props);
    }
    EffectResult::Resolved
}

/// The hand's Sadness cards, in hand order.
fn sadness_in_hand(state: &CombatState) -> SmallVec<[CardUid; 10]> {
    state
        .player
        .hand
        .iter()
        .copied()
        .filter(|uid| state.cards[uid.0.get() as usize - 1].card_id == CardId::Sadness)
        .collect()
}

/// `Growth.cs`: exhaust every Sadness in hand, then
/// `Apply<StretchPower>(4 (+2) x count)` (a 0 amount applies nothing).
pub(crate) fn play_growth(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let sadness = sadness_in_hand(state);
    for &uid in &sadness {
        let _ = card_effects::exhaust_card_uid(state, uid);
    }
    let per = var(state, 4, 6);
    let n = i32::try_from(sadness.len()).unwrap_or(i32::MAX);
    apply_to_player(state, PowerKind::Stretch, per.saturating_mul(n));
    EffectResult::Resolved
}

/// `PunchOfJoy.cs`: attack (13, +4); then, when the hand holds Sadness,
/// `StableShuffle(Rng.CombatCardSelection)` them and transform the first
/// into Joy.
pub(crate) fn play_punch_of_joy(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = t.expect("PunchOfJoy: AnyEnemy-targeted");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    let mut sadness = sadness_in_hand(state);
    if sadness.is_empty() {
        return EffectResult::Resolved;
    }
    let version = state.game_version;
    let cards = &state.cards;
    let cmp = |a: CardUid, b: CardUid| {
        sts2_core::cards::card_model_compare_for(
            &cards[a.0.get() as usize - 1],
            &cards[b.0.get() as usize - 1],
            version,
        )
    };
    state.run_rng.combat_card_selection.stable_shuffle_by_for(
        version,
        sadness.as_mut_slice(),
        &cmp,
    );
    let _ = card_effects::transform_card_in_place(state, sadness[0], fool_card(CardId::Joy));
    EffectResult::Resolved
}

/// `LockedIn.cs`: `CardSelectCmd.FromCombatPile(Draw, 1, filter: Shoot tag)`,
/// the pick to hand (the SecretWeapon shape).
pub(crate) fn play_locked_in(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    crate::cards::colorless::helpers::move_card_from_draw_to_hand_where(state, |c| {
        c.has_tag(sts2_core::cards::CardTag::Shoot)
    });
    EffectResult::Resolved
}

/// `TheHermit.cs`: `CardSelectCmd.FromHandForDiscard(1)`; a Curse pick is
/// exhausted, anything else discarded; then `Draw(2 (+1))`. A hand of at most
/// one card resolves without a prompt; a combat already ending opens nothing
/// (`FromHand` returns empty) and goes straight to the draw.
pub(crate) fn play_the_hermit(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let draw = var(state, 2, 3);
    let hand: SmallVec<[CardUid; 16]> = state.player.hand.iter().copied().collect();
    if sts2_core::combat_loop::combat_is_ending(state) {
        card_effects::discard_or_exhaust_curses_then_draw(state, &[], draw);
    } else if hand.len() <= 1 {
        card_effects::discard_or_exhaust_curses_then_draw(state, &hand, draw);
    } else {
        let options: Vec<sts2_core::ChoiceOption> = hand
            .iter()
            .map(|&uid| {
                let c = &state.cards[uid.0.get() as usize - 1];
                sts2_core::ChoiceOption {
                    card_id: c.card_id,
                    upgraded: c.upgraded,
                    card_uid: Some(uid),
                    source_pile: Some(sts2_core::cards::PileKind::Hand),
                }
            })
            .collect();
        state.open_hand_choice(sts2_core::PendingChoice {
            kind: sts2_core::ChoiceKind::DiscardFromHand,
            options,
            min_select: 1,
            max_select: 1,
            can_skip: false,
            post_effect: Some(sts2_core::PostChoiceEffect::CurseExhaustThenDraw(draw)),
            picked: SmallVec::new(),
        });
    }
    EffectResult::Resolved
}
