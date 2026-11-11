//! Auto-play / HistoryCourse-dupe play-body gaps (Ironclad v0.111.0 deep
//! parity round 3). C# `CardModel.OnPlayWrapper` (v0.111.0
//! `CardModel.cs:1881-1999`, same shape in v0.107.1) is shared by hand plays,
//! auto-plays and the HistoryCourse dupe:
//!
//! * every play iteration fires `Hook.BeforeCardPlayed` (`:1926`), so
//!   `FreeAttackPower.BeforeCardPlayed` decrements once per iteration — the
//!   replays of an auto-played card included (Ironclad seed 4126: Stampede
//!   auto-plays Unrelenting under OneTwoPunch, FREE_ATTACK cs=1 rs=2);
//! * every iteration runs `Enchantment.OnPlay` (`:1940-1947`), and the dupe
//!   carries a copy of the enchantment (`DeepCloneFields`, `:1205-1210`), so a
//!   Corrupted dupe hurts its owner (Ironclad seed 4147/2118: hp off by 2);
//! * `GetResultLocationForCardPlay` sends a dupe to `PileType.None` before the
//!   Exhaust check (`:2084-2096`), auto-plays included — only Corruption's
//!   Skill→Exhaust override beats it (Ironclad seeds 4165/4167/4125: a Havoc'd
//!   dupe Anger exhausted, feeding Feel No Pain).

use smallvec::SmallVec;

use sts2_content::monster_dispatch::{build_ai, MonsterSpawnArgs, SpawnPosition};
use sts2_core::cards::{Card, CardId, CardType, EnchantmentId, TargetType, VarKey};
use sts2_core::combat_loop::{auto_play_card_from_draw_top, start_combat, start_player_turn};
use sts2_core::{
    CardUid, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind,
    PlayerCombatState, PowerKind, Relic, RelicKind,
};

const ENEMY_HP: i32 = 400;

fn lone_fat_enemy() -> (Vec<Creature>, CreatureUid) {
    let enemy_uid = CreatureUid::new(1).expect("non-zero");
    let creatures = vec![Creature {
        id: enemy_uid.0,
        current_hp: ENEMY_HP,
        max_hp: ENEMY_HP,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::ShrinkerBeetle),
        monster_id: MonsterId(MonsterKind::ShrinkerBeetle.python_name()),
        intent: None,
        intent_history: IntentHistory::default(),
        ai: Some(build_ai(
            MonsterKind::ShrinkerBeetle,
            MonsterSpawnArgs::None,
            SpawnPosition {
                alone: true,
                front: true,
                back: true,
                index: 0,
            },
        )),
    }];
    (creatures, enemy_uid)
}

fn push(cards: &mut sts2_core::CardStore, card: Card) -> CardUid {
    let uid = CardUid::new(u32::try_from(cards.len() + 1).expect("fits")).expect("non-zero");
    cards.push(Card { uid, ..card });
    uid
}

fn strike() -> Card {
    Card {
        card_id: CardId::StrikeIronclad,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        base_damage: 6,
        ..Card::pod_defaults()
    }
}

fn defend() -> Card {
    Card {
        card_id: CardId::DefendIronclad,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        base_block: 5,
        ..Card::pod_defaults()
    }
}

/// A started combat whose draw pile is five Defends (the turn-start draw) and
/// nothing else; `relics` installed. The caller adds the cards under test.
fn setup(relics: Vec<Relic>) -> CombatState {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();

    let mut cards = sts2_core::CardStore::new();
    let filler: Vec<CardUid> = (0..5).map(|_| push(&mut cards, defend())).collect();
    let (creatures, enemy_uid) = lone_fat_enemy();
    let player = PlayerCombatState {
        max_energy: 3,
        current_hp: 80,
        max_hp: 80,
        character: Some(sts2_core::potions::CharacterId::Ironclad),
        ..PlayerCombatState::default()
    };
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        creatures,
        cards,
        relics: relics.into(),
        player,
        enemy_uids: SmallVec::from_slice(&[enemy_uid]),
        ..CombatState::default()
    };
    let _ = start_combat(&mut state);
    state.player.hand.clear();
    state.player.discard.clear();
    state.player.exhaust.clear();
    state.player.draw = filler.into_iter().collect();
    state
}

fn add_power(state: &mut CombatState, kind: PowerKind, amount: i32) {
    let puid = state.player.creature_uid.expect("player creature mirror");
    sts2_core::card_effects::apply_power_to_creature(state, puid, kind, amount);
}

fn history_course() -> Vec<Relic> {
    vec![Relic {
        id: std::num::NonZeroU32::new(1).expect("non-zero"),
        kind: RelicKind::HistoryCourse,
        counter: 0,
        is_wax: false,
        is_melted: false,
    }]
}

/// Make `src` last round's last Attack/Skill and run the round-2 turn start,
/// so HistoryCourse auto-plays a dupe of it.
fn run_history_course_dupe(state: &mut CombatState, src: CardUid) {
    let id = state.cards[src.0.get() as usize - 1].card_id;
    state.combat_log.last_attack_skill_played = Some((id, false, Some(src)));
    state.round = 2;
    let _ = start_player_turn(state);
}

fn enemy_hp(state: &CombatState) -> i32 {
    state.creatures[0].current_hp
}

// --- C-autoplay-replay-skips-freeattack-consume ----------------------------

/// An auto-played Attack replayed by OneTwoPunch consumes FreeAttack on BOTH
/// plays: `FreeAttackPower.BeforeCardPlayed` runs per OnPlayWrapper iteration
/// and the replayed card sits in Play (FreeAttackPower.cs BeforeCardPlayed).
#[test]
fn auto_play_replay_consumes_free_attack_per_play() {
    let mut state = setup(Vec::new());
    let s = push(&mut state.cards, strike());
    state.player.draw.insert(0, s);
    add_power(&mut state, PowerKind::OneTwoPunch, 1);
    add_power(&mut state, PowerKind::FreeAttack, 2);

    assert!(auto_play_card_from_draw_top(&mut state, false));
    assert_eq!(enemy_hp(&state), ENEMY_HP - 12, "OneTwoPunch: hit twice");
    assert_eq!(
        state.player_power_amount(PowerKind::FreeAttack),
        0,
        "FreeAttack(2) is decremented by the first play AND the replay"
    );
}

/// The Stampede/Unrelenting shape of seed 4126: no FreeAttack up front, the
/// first play applies one, the replay consumes it, the replay's OnPlay applies
/// a fresh one — C# ends at 1, not 2.
#[test]
fn auto_played_unrelenting_under_one_two_punch_ends_with_one_free_attack() {
    let mut state = setup(Vec::new());
    let mut card = sts2_content::cards::spec_index::card_pod_from_id(CardId::Unrelenting, false)
        .expect("Unrelenting");
    card.cost = sts2_core::CardEnergyCost::new(2);
    let u = push(&mut state.cards, card);
    state.player.draw.insert(0, u);
    add_power(&mut state, PowerKind::OneTwoPunch, 1);

    assert!(auto_play_card_from_draw_top(&mut state, false));
    assert_eq!(state.player_power_amount(PowerKind::FreeAttack), 1);
}

/// The HistoryCourse dupe's replay loop consumes FreeAttack per play too.
#[test]
fn history_course_dupe_replay_consumes_free_attack_per_play() {
    let mut state = setup(history_course());
    let src = push(&mut state.cards, strike());
    add_power(&mut state, PowerKind::OneTwoPunch, 1);
    add_power(&mut state, PowerKind::FreeAttack, 2);

    run_history_course_dupe(&mut state, src);
    assert_eq!(enemy_hp(&state), ENEMY_HP - 12, "dupe hit twice");
    assert_eq!(state.player_power_amount(PowerKind::FreeAttack), 0);
}

// --- C-historycourse-dupe-skips-enchantment-onplay -------------------------

/// A HistoryCourse dupe of a Corrupted Strike runs `Corrupted.OnPlay` (2
/// self-damage) on its play.
#[test]
fn history_course_dupe_runs_its_enchantment_on_play() {
    let mut state = setup(history_course());
    let mut card = strike();
    card.set_enchantment(EnchantmentId::Corrupted, 1);
    let src = push(&mut state.cards, card);
    let hp0 = state.player.current_hp;

    run_history_course_dupe(&mut state, src);
    assert_eq!(enemy_hp(&state), ENEMY_HP - 9, "Corrupted: 6 x 1.5");
    assert_eq!(
        state.player.current_hp,
        hp0 - 2,
        "Corrupted.OnPlay self-damage"
    );
}

/// ... on every replay iteration, and the self-HP-loss feeds Rupture.
#[test]
fn history_course_dupe_replay_runs_its_enchantment_per_play() {
    let mut state = setup(history_course());
    let mut card = strike();
    card.set_enchantment(EnchantmentId::Corrupted, 1);
    let src = push(&mut state.cards, card);
    add_power(&mut state, PowerKind::OneTwoPunch, 1);
    add_power(&mut state, PowerKind::Rupture, 1);
    let hp0 = state.player.current_hp;

    run_history_course_dupe(&mut state, src);
    assert_eq!(state.player.current_hp, hp0 - 4, "2 self-damage x 2 plays");
    assert_eq!(
        state.player_power_amount(PowerKind::Strength),
        2,
        "Rupture: +1 Strength per self-HP-loss"
    );
}

// --- D3-autoplay-result-pile-ignores-IsDupe --------------------------------

fn dupe_strike() -> Card {
    let mut c = strike();
    c.set_combat_var(VarKey::IsDupe, 1);
    c
}

/// Havoc force-exhausts what it auto-plays, but a dupe's result pile is None
/// before the ExhaustOnNextPlay check: it leaves combat, nothing exhausts and
/// Feel No Pain stays silent.
#[test]
fn havoc_auto_played_dupe_leaves_combat_without_exhausting() {
    let mut state = setup(Vec::new());
    let d = push(&mut state.cards, dupe_strike());
    state.player.draw.insert(0, d);
    add_power(&mut state, PowerKind::FeelNoPain, 3);
    let block0 = state.player.block;

    assert!(auto_play_card_from_draw_top(&mut state, true));
    assert_eq!(enemy_hp(&state), ENEMY_HP - 6, "the dupe still plays");
    assert!(!state.player.exhaust.contains(&d), "a dupe never exhausts");
    assert!(!state.player.discard.contains(&d));
    assert!(!state.player.play.contains(&d));
    assert!(!state.player.draw.contains(&d));
    assert!(!state.player.hand.contains(&d));
    assert_eq!(state.player.block, block0, "no Feel No Pain block");
}

/// A Stampede/Cascade-style (non-forced) auto-play of a dupe leaves combat
/// instead of discarding.
#[test]
fn auto_played_dupe_is_not_discarded() {
    let mut state = setup(Vec::new());
    let d = push(&mut state.cards, dupe_strike());
    state.player.draw.insert(0, d);

    assert!(auto_play_card_from_draw_top(&mut state, false));
    assert!(!state.player.discard.contains(&d));
    assert!(!state.player.play.contains(&d));
}

/// Corruption's Skill→Exhaust override beats the dupe's None
/// (CorruptionPower.cs ModifyCardPlayResultLocation).
#[test]
fn auto_played_dupe_skill_under_corruption_still_exhausts() {
    let mut state = setup(Vec::new());
    let mut c = defend();
    c.set_combat_var(VarKey::IsDupe, 1);
    let d = push(&mut state.cards, c);
    state.player.draw.insert(0, d);
    add_power(&mut state, PowerKind::Corruption, 1);

    assert!(auto_play_card_from_draw_top(&mut state, false));
    assert!(state.player.exhaust.contains(&d));
}
