//! The frozen card tables (M06) on the two cards that carry a late effect:
//! Regent's `MakeItSo` and Necrobinder's `RightHandHand`.
//!
//! After the content install, `card_effects::lookup` and `fire_late_effects`
//! read dense `CardId`-indexed copies of the registries instead of the locked
//! maps, and the late-effect sweep skips its pile walk when no card in the
//! arena has a late effect. An Ironclad combat never holds either card, so an
//! Ironclad-only check would pass even if the frozen sweep never fired. These
//! tests play the two cards through the engine and compare whole random
//! combats on the frozen path against the locked path.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::card_effects::{
    card_tables_frozen, freeze_card_tables, lookup_late_effect, register_late_effect, LateEffectFn,
};
use sts2_core::cards::CardId;
use sts2_core::combat_loop::{play_card, start_combat, start_player_turn};
use sts2_core::hooks::CardCtx;
use sts2_core::potions::CharacterId;
use sts2_core::{apply, legal_actions, CardUid, CombatState, MonsterKind};

fn install() {
    content_dispatch::install_default_registry();
    // ChemicalX's `modify_x_value` listener lives in the relic registry.
    sts2_content::relics::dispatch::install_default_registry();
    // The card POD factory is registered by the relic installer, not the card
    // installer. Idempotent.
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
}

/// A player state with one 300-HP Nibbit and no cards yet.
fn bare_state(character: CharacterId, seed: i32) -> CombatState {
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(seed),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(seed),
        ..Default::default()
    };
    state.player.character = Some(character);
    state.player.current_hp = 70;
    state.player.max_hp = 70;
    state.player.max_energy = 3;
    state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 300;
        c.current_hp = 300;
    }
    state
}

/// Add a fresh `card_id` to the arena (not to any pile) and return its uid.
fn add_card(state: &mut CombatState, card_id: CardId) -> CardUid {
    let mut card = sts2_core::lookup_card_factory()
        .and_then(|f| f(card_id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

/// A mid-turn state: `hand` in hand, `watched` in discard, `energy` to spend.
fn play_ready(
    character: CharacterId,
    hand: &[CardId],
    watched: CardId,
    energy: i32,
) -> (CombatState, CardUid) {
    let mut state = bare_state(character, 4242);
    state.player.energy = energy;
    state.phase = sts2_core::CombatPhase::PlayerPlayPhase;
    for &id in hand {
        let uid = add_card(&mut state, id);
        state.player.hand.push(uid);
    }
    let watched = add_card(&mut state, watched);
    state.player.discard.push(watched);
    (state, watched)
}

/// `MakeItSo` returns from discard on the third Skill played this turn
/// (C# `MakeItSo.AfterCardPlayedLate`, interval 3), through `play_card` and
/// the frozen late-effect table.
#[test]
fn make_it_so_returns_on_the_third_skill_through_the_frozen_table() {
    install();
    assert!(card_tables_frozen());
    let (mut state, make_it_so) = play_ready(
        CharacterId::Regent,
        &[CardId::DefendRegent; 3],
        CardId::MakeItSo,
        3,
    );
    for _ in 0..3 {
        assert!(state.player.discard.contains(&make_it_so));
        let _ = play_card(&mut state, 0, None);
    }
    assert!(
        state.player.hand.contains(&make_it_so),
        "hand={:?} discard={:?}",
        state.player.hand,
        state.player.discard
    );
}

/// `RightHandHand` returns from discard after a play that spends 2 or more
/// energy (C# `RightHandHand.AfterCardPlayedLate`), and not after one that
/// spends less. The trigger here is an X-cost card (Whirlwind).
#[test]
fn right_hand_hand_returns_after_a_two_energy_play_through_the_frozen_table() {
    install();
    assert!(card_tables_frozen());
    for (energy, returns) in [(3, true), (1, false)] {
        let (mut state, right_hand_hand) = play_ready(
            CharacterId::Necrobinder,
            &[CardId::Whirlwind],
            CardId::RightHandHand,
            energy,
        );
        let _ = play_card(&mut state, 0, None);
        assert_eq!(
            state.player.hand.contains(&right_hand_hand),
            returns,
            "energy {energy}: hand={:?} discard={:?}",
            state.player.hand,
            state.player.discard
        );
    }
}

/// GH #741: a NON-X play that spends 2+ energy returns `RightHandHand` too.
/// C# `PlayCardAction.ExecuteAction` sets `EnergyValue = EnergySpent = item`
/// (the energy `SpendResources` paid) for every play, not just X-cost ones.
/// Bash costs 2 (returns); Strike costs 1 (stays in discard).
#[test]
fn right_hand_hand_returns_after_a_two_cost_non_x_play() {
    install();
    for (card, returns) in [(CardId::Bash, true), (CardId::StrikeIronclad, false)] {
        let (mut state, right_hand_hand) =
            play_ready(CharacterId::Necrobinder, &[card], CardId::RightHandHand, 3);
        let _ = play_card(&mut state, 0, Some(0));
        assert_eq!(
            state.player.hand.contains(&right_hand_hand),
            returns,
            "{card:?}: hand={:?} discard={:?}",
            state.player.hand,
            state.player.discard
        );
    }
}

/// GH #741: the energy value is per play. A 2-cost Bash returns
/// `RightHandHand`; a 1-cost Strike played next must not, whatever the earlier
/// play left behind.
#[test]
fn right_hand_hand_reads_the_current_play_not_a_stale_one() {
    install();
    let (mut state, right_hand_hand) = play_ready(
        CharacterId::Necrobinder,
        &[CardId::Bash, CardId::StrikeIronclad],
        CardId::RightHandHand,
        3,
    );
    let _ = play_card(&mut state, 0, Some(0));
    assert!(state.player.hand.contains(&right_hand_hand));
    // Put it back in discard, then play the 1-cost Strike.
    state.player.hand.retain(|u| *u != right_hand_hand);
    state.player.discard.push(right_hand_hand);
    let strike_pos = state
        .player
        .hand
        .iter()
        .position(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::StrikeIronclad)
        .expect("strike in hand");
    let _ = play_card(&mut state, u8::try_from(strike_pos).expect("fits"), Some(0));
    assert!(
        state.player.discard.contains(&right_hand_hand),
        "a 1-energy play must not return RightHandHand"
    );
}

/// C# `Hook.AfterCardPlayed` runs every listener's `AfterCardPlayedLate` for
/// AUTO-plays too (`CardCmd.AutoPlay` → `OnPlayWrapper`), with `EnergyValue =
/// EnergyCost.GetAmountToSpend()`. Havoc auto-plays the draw-pile top: a 2-cost
/// Bash returns `RightHandHand`, a 1-cost Strike does not.
#[test]
fn right_hand_hand_returns_after_an_auto_played_two_cost_card() {
    install();
    for (top, returns) in [(CardId::Bash, true), (CardId::StrikeIronclad, false)] {
        let (mut state, right_hand_hand) = play_ready(
            CharacterId::Necrobinder,
            &[CardId::Havoc],
            CardId::RightHandHand,
            3,
        );
        let top_uid = add_card(&mut state, top);
        state.player.draw.push(top_uid);
        let _ = play_card(&mut state, 0, None);
        assert!(
            !state.player.draw.contains(&top_uid),
            "Havoc auto-played {top:?}"
        );
        assert_eq!(
            state.player.hand.contains(&right_hand_hand),
            returns,
            "{top:?}: hand={:?} discard={:?}",
            state.player.hand,
            state.player.discard
        );
    }
}

/// ChemicalX's +2 is an X-VALUE modifier (`ChemicalX.ModifyXValue`, read
/// through `CardModel.ResolveEnergyXValue`), not energy: the `EnergyValue`
/// RightHandHand reads is the raw energy spent. A 0-energy Whirlwind with
/// ChemicalX resolves X = 2 but spends 0, so RightHandHand stays in discard.
#[test]
fn chemical_x_does_not_count_as_energy_for_right_hand_hand() {
    install();
    let (mut state, right_hand_hand) = play_ready(
        CharacterId::Necrobinder,
        &[CardId::Whirlwind],
        CardId::RightHandHand,
        0,
    );
    state.relics.push(sts2_core::Relic {
        id: std::num::NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::ChemicalX,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    let hp_before = state.creatures.iter().map(|c| c.current_hp).sum::<i32>();
    let _ = play_card(&mut state, 0, None);
    let hp_after = state.creatures.iter().map(|c| c.current_hp).sum::<i32>();
    assert!(
        hp_after < hp_before,
        "ChemicalX still gives Whirlwind X = 2 hits"
    );
    assert!(
        state.player.discard.contains(&right_hand_hand),
        "0 energy spent must not return RightHandHand: hand={:?}",
        state.player.hand
    );
}

static MAKE_IT_SO: OnceLock<LateEffectFn> = OnceLock::new();
static RIGHT_HAND_HAND: OnceLock<LateEffectFn> = OnceLock::new();
static MAKE_IT_SO_RETURNS: AtomicU32 = AtomicU32::new(0);
static RIGHT_HAND_HAND_RETURNS: AtomicU32 = AtomicU32::new(0);

/// Run the installed late effect and count the times it moved the watched
/// card into hand. Registering this in place of the installed handler sends
/// `fire_late_effects` down the locked path without changing what it does.
fn counted(
    installed: &OnceLock<LateEffectFn>,
    returns: &AtomicU32,
    state: &mut CombatState,
    watched: CardUid,
    played: &CardCtx,
) {
    let was_in_hand = state.player.hand.contains(&watched);
    (installed.get().expect("installed handler saved"))(state, watched, played);
    if !was_in_hand && state.player.hand.contains(&watched) {
        returns.fetch_add(1, Ordering::Relaxed);
    }
}

fn make_it_so_counted(state: &mut CombatState, watched: CardUid, played: &CardCtx) {
    counted(&MAKE_IT_SO, &MAKE_IT_SO_RETURNS, state, watched, played);
}

fn right_hand_hand_counted(state: &mut CombatState, watched: CardUid, played: &CardCtx) {
    counted(
        &RIGHT_HAND_HAND,
        &RIGHT_HAND_HAND_RETURNS,
        state,
        watched,
        played,
    );
}

const REGENT_DECK: &[CardId] = &[
    CardId::MakeItSo,
    CardId::MakeItSo,
    CardId::DefendRegent,
    CardId::DefendRegent,
    CardId::DefendRegent,
    CardId::DefendRegent,
    CardId::DefendRegent,
    CardId::DefendRegent,
    CardId::StrikeRegent,
    CardId::StrikeRegent,
];

const NECROBINDER_DECK: &[CardId] = &[
    CardId::RightHandHand,
    CardId::RightHandHand,
    CardId::Whirlwind,
    CardId::Whirlwind,
    CardId::Whirlwind,
    CardId::StrikeNecrobinder,
    CardId::StrikeNecrobinder,
    CardId::DefendNecrobinder,
    CardId::DefendNecrobinder,
    CardId::DefendNecrobinder,
];

/// Per-step hashes of one seeded uniform-random combat: the action, its
/// result, the parity-bridge state export and the combat RNG cursor.
fn random_combat(character: CharacterId, deck: &[CardId], seed: u64) -> Vec<u64> {
    let mut state = bare_state(character, i32::try_from(seed).expect("small seed"));
    for &id in deck {
        let uid = add_card(&mut state, id);
        state.player.draw.push(uid);
    }
    let _ = start_combat(&mut state);
    let _ = start_player_turn(&mut state);
    let mut agent = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut steps = Vec::new();
    while !state.is_over && steps.len() < 400 {
        let actions = legal_actions(&state);
        if actions.is_empty() {
            break;
        }
        agent ^= agent << 13;
        agent ^= agent >> 7;
        agent ^= agent << 17;
        let action = actions[(agent % actions.len() as u64) as usize].clone();
        let result = apply(&mut state, action.clone());
        let mut h = DefaultHasher::new();
        format!("{action:?}|{result:?}").hash(&mut h);
        sts2_core::parity::combat_state_to_bridge_state(&state)
            .to_string()
            .hash(&mut h);
        state.rng.counter().hash(&mut h);
        steps.push(h.finish());
    }
    steps
}

fn all_combats() -> Vec<Vec<u64>> {
    let mut out = Vec::new();
    for seed in 1..=12 {
        out.push(random_combat(CharacterId::Regent, REGENT_DECK, seed));
        out.push(random_combat(
            CharacterId::Necrobinder,
            NECROBINDER_DECK,
            seed,
        ));
    }
    out
}

/// Random Regent and Necrobinder combats whose decks hold the late-effect
/// cards play out step for step the same on the frozen tables as on the
/// locked registries, and the late effects really fire in them.
#[test]
fn frozen_and_locked_late_sweeps_agree_on_random_combats() {
    install();
    assert!(card_tables_frozen());
    let frozen = all_combats();

    let make_it_so = lookup_late_effect(CardId::MakeItSo).expect("MakeItSo late effect");
    let right_hand_hand =
        lookup_late_effect(CardId::RightHandHand).expect("RightHandHand late effect");
    MAKE_IT_SO.set(make_it_so).expect("set once");
    RIGHT_HAND_HAND.set(right_hand_hand).expect("set once");
    register_late_effect(CardId::MakeItSo, make_it_so_counted as LateEffectFn);
    register_late_effect(
        CardId::RightHandHand,
        right_hand_hand_counted as LateEffectFn,
    );
    freeze_card_tables();
    assert!(
        !card_tables_frozen(),
        "a changed handler keeps the locked path"
    );
    let locked = all_combats();

    assert_eq!(frozen.len(), locked.len());
    for (i, (f, l)) in frozen.iter().zip(&locked).enumerate() {
        assert_eq!(
            f, l,
            "combat {i} diverged between the frozen and locked sweeps"
        );
    }
    assert!(
        MAKE_IT_SO_RETURNS.load(Ordering::Relaxed) > 0,
        "MakeItSo never returned"
    );
    assert!(
        RIGHT_HAND_HAND_RETURNS.load(Ordering::Relaxed) > 0,
        "RightHandHand never returned"
    );

    // Putting the installed handlers back makes the registry match the frozen
    // table again, so the next freeze re-validates it.
    register_late_effect(CardId::MakeItSo, make_it_so);
    register_late_effect(CardId::RightHandHand, right_hand_hand);
    freeze_card_tables();
    assert!(card_tables_frozen());
}

/// `MakeItSo` / `RightHandHand` return with `CardPileCmd.Add(this, Hand)`
/// (MakeItSo.cs:43, RightHandHand.cs:51), which refuses a combat pile once the
/// combat is ending (CardPileCmd.cs:312): a lethal trigger leaves the watched
/// card in the discard.
#[test]
fn late_effect_returns_are_refused_once_the_combat_is_ending() {
    install();
    let (mut state, make_it_so) = play_ready(
        CharacterId::Regent,
        &[CardId::DefendRegent; 3],
        CardId::MakeItSo,
        3,
    );
    for _ in 0..2 {
        let _ = play_card(&mut state, 0, None);
    }
    set_enemy_hp(&mut state, 0);
    assert!(sts2_core::combat_loop::combat_is_ending(&state));
    let _ = play_card(&mut state, 0, None);
    assert!(
        state.player.discard.contains(&make_it_so),
        "Make It So stays"
    );

    let (mut state, right_hand_hand) = play_ready(
        CharacterId::Necrobinder,
        &[CardId::Bash],
        CardId::RightHandHand,
        3,
    );
    set_enemy_hp(&mut state, 1);
    let _ = play_card(&mut state, 0, Some(0));
    assert!(sts2_core::combat_loop::combat_is_ending(&state));
    assert!(
        state.player.discard.contains(&right_hand_hand),
        "Right Hand Hand stays"
    );
}

fn set_enemy_hp(state: &mut CombatState, hp: i32) {
    let enemies = state.enemy_uids.clone();
    for c in state
        .creatures
        .iter_mut()
        .filter(|c| enemies.iter().any(|e| e.0 == c.id))
    {
        c.current_hp = hp;
    }
}
