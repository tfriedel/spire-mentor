//! GH #795 (S8c): v0.111.0 relic behaviour changes that need a played combat
//! to observe (card plays, turn-start ordering, block breaks).
//!
//! Each test pairs the v0.111.0 behaviour (`*_v0111`) with the v0.107.1 one
//! it replaces, on the same fixture, so the v0.107.1 arm stays pinned.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Relic, RelicKind,
};

/// One Nibbit at 40 HP, player on 3 energy in the play phase, holding the
/// given relics.
fn fresh(version: GameVersion, relics: &[RelicKind]) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 40,
        max_hp: 40,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.round = 1;
    for (i, &kind) in relics.iter().enumerate() {
        state.relics.push(Relic {
            id: NonZeroU32::new(u32::try_from(i + 1).expect("u32")).expect("nz"),
            kind,
            counter: kind.initial_counter(),
            is_wax: false,
            is_melted: false,
        });
    }
    state
}

fn push_hand_card(state: &mut CombatState, card_id: CardId, card_type: CardType) {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let (target_type, base_damage, base_block) = match card_type {
        CardType::Attack => (TargetType::Enemy, 6, 0),
        _ => (TargetType::Self_, 0, 5),
    };
    state.cards.push(Card {
        uid,
        card_id,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type,
        target_type,
        upgraded: false,
        base_damage,
        base_block,
        ..Card::pod_defaults()
    });
    state.player.hand.push(uid);
}

/// Play Strike then Defend; return the card HistoryCourse would replay.
fn last_replayable(version: GameVersion) -> Option<CardId> {
    let mut state = fresh(version, &[RelicKind::HistoryCourse]);
    push_hand_card(&mut state, CardId::StrikeIronclad, CardType::Attack);
    push_hand_card(&mut state, CardId::DefendIronclad, CardType::Skill);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    state
        .combat_log
        .last_attack_skill_played
        .map(|(id, _, _)| id)
}

/// v0.107.1: HistoryCourse replays the last Attack OR Skill — the Defend.
#[test]
fn history_course_tracks_last_attack_or_skill_v0107() {
    assert_eq!(
        last_replayable(GameVersion::V0_107_1),
        Some(CardId::DefendIronclad)
    );
}

/// v0.111.0: only the last Attack (decomp HistoryCourse.cs:21) — the Strike.
#[test]
fn history_course_tracks_last_attack_or_skill_v0111() {
    assert_eq!(
        last_replayable(GameVersion::V0_111_0),
        Some(CardId::StrikeIronclad)
    );
}

/// HistoryCourse replays last round's Shiv while the player holds
/// FanOfKnives. Returns the `combat_targets` draws the replay spent.
fn history_course_fan_of_knives_shiv_targets(version: GameVersion) -> u32 {
    let mut state = fresh(version, &[RelicKind::HistoryCourse]);
    let player = state.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(
        &mut state,
        player,
        sts2_core::PowerKind::FanOfKnives,
        1,
    );
    state.round = 2;
    state.combat_log.prev_round_last_attack_skill_played = Some((CardId::Shiv, false, None));
    let before = state.run_rng.combat_targets.counter();
    let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
    state.run_rng.combat_targets.counter() - before
}

/// Under FanOfKnives the Shiv is `AllEnemies` (Shiv.cs:23-32), so
/// `CardCmd.AutoPlay` rolls no target (CardCmd.cs:74). Same C# in both builds
/// (GH #806).
#[test]
fn history_course_fan_of_knives_shiv_rolls_no_target_both_versions() {
    for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
        assert_eq!(
            history_course_fan_of_knives_shiv_targets(version),
            0,
            "{version:?}"
        );
    }
}

/// Hand draws over `turns` turn starts, folded the way `combat_loop` does:
/// `modify_hand_draw` first, then the relic `before_hand_draw` walk.
fn draws_over_turns(version: GameVersion, relic: RelicKind, turns: usize) -> Vec<i32> {
    let mut state = fresh(version, &[relic]);
    (0..turns)
        .map(|_| {
            let draw = sts2_core::relic_effects::dispatch_modify_hand_draw(&state, 5);
            let _ = sts2_core::relic_effects::dispatch_before_hand_draw(&mut state);
            draw
        })
        .collect()
}

/// v0.111.0 Pendulum: +1 to the 3rd, 6th, ... hand draw (decomp
/// Pendulum.cs:76-102). v0.107.1 draws the card after turn start instead,
/// so its hand draw never changes.
#[test]
fn pendulum_adds_to_every_third_hand_draw_v0111() {
    assert_eq!(
        draws_over_turns(GameVersion::V0_111_0, RelicKind::Pendulum, 6),
        vec![5, 5, 6, 5, 5, 6]
    );
    assert_eq!(
        draws_over_turns(GameVersion::V0_107_1, RelicKind::Pendulum, 6),
        vec![5; 6]
    );
}

/// v0.111.0 PollinousCore: counted in `BeforeHandDraw`, +2 on every 4th hand
/// draw, then reset (decomp PollinousCore.cs:88-125). The v0.107.1 tick runs
/// on `BeforeSideTurnStart`, so this fold alone never sees it.
#[test]
fn pollinous_core_adds_to_every_fourth_hand_draw_v0111() {
    assert_eq!(
        draws_over_turns(GameVersion::V0_111_0, RelicKind::PollinousCore, 8),
        vec![5, 5, 5, 7, 5, 5, 5, 7]
    );
    assert_eq!(
        draws_over_turns(GameVersion::V0_107_1, RelicKind::PollinousCore, 8),
        vec![5; 8]
    );
}

fn push_spec_card(state: &mut CombatState, card_id: CardId) {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = sts2_content::cards::spec_index::card_pod_from_id(card_id, false).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    state.player.hand.push(uid);
}

fn enemy_vulnerable(state: &CombatState) -> i32 {
    state.creatures[0]
        .powers
        .iter()
        .filter_map(|p| state.powers.get_by_id(p.0))
        .find(|p| p.kind == sts2_core::PowerKind::Vulnerable)
        .map_or(0, |p| p.amount)
}

/// Strike (6) through 3 enemy Block with HandDrill held.
fn hand_drill_after_strike(version: GameVersion) -> i32 {
    let mut state = fresh(version, &[RelicKind::HandDrill]);
    state.creatures[0].block = 3;
    push_spec_card(&mut state, CardId::StrikeIronclad);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    enemy_vulnerable(&state)
}

/// Both builds: breaking block with an attack applies Vulnerable 2 (v0.107.1
/// via `AfterDamageGiven` + `WasBlockBroken`, v0.111.0 via the new
/// `AfterBlockBroken(target, breaker)` relic arm, decomp HandDrill.cs:21).
#[test]
fn hand_drill_vulnerable_on_attack_block_break_v0111() {
    assert_eq!(hand_drill_after_strike(GameVersion::V0_107_1), 2);
    assert_eq!(hand_drill_after_strike(GameVersion::V0_111_0), 2);
}

/// Expose (0 cost) on a Nibbit holding 3 Block with HandDrill held; the
/// enemy's Vulnerable afterwards.
fn hand_drill_after_expose(version: GameVersion) -> i32 {
    let mut state = fresh(version, &[RelicKind::HandDrill]);
    state.creatures[0].block = 3;
    push_spec_card(&mut state, CardId::Expose);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(state.creatures[0].block, 0, "Expose strips the Block");
    enemy_vulnerable(&state)
}

/// The case that separates the two arms (GH #812): Expose strips Block
/// without dealing damage. v0.107.1 HandDrill listens only to
/// `AfterDamageGiven`, so the enemy keeps just Expose's own Vulnerable 2.
/// v0.111.0's `CreatureCmd.LoseBlock(…, remover: Owner)` fires
/// `AfterBlockBroken` with the player as breaker (decomp Expose.cs:39,
/// CreatureCmd.cs:722), so HandDrill adds 2 more.
#[test]
fn hand_drill_vulnerable_on_expose_block_strip_only_v0111() {
    assert_eq!(hand_drill_after_expose(GameVersion::V0_107_1), 2);
    assert_eq!(hand_drill_after_expose(GameVersion::V0_111_0), 4);
}

/// Card counts of successive combat card rewards with LastingCandy held,
/// dispatching `BeforeCombatRewardOffered` after each set like `RunManager`.
fn lasting_candy_reward_sizes(version: GameVersion, from_combat: bool) -> Vec<(usize, bool)> {
    use sts2_core::run_effects::{dispatch_before_combat_reward_offered, CombatRewardsSet};
    use sts2_core::runstate::{CardReward, CardRewardGenerationOptions, Reward, RoomType};
    sts2_content::relics::dispatch::ensure_installed();
    content_dispatch::ensure_installed();
    let mut rs = sts2_core::runstate::RunState::new();
    rs.game_version = version;
    rs.relics.push(sts2_core::RelicId::LastingCandy);
    let set = CombatRewardsSet {
        room_type: RoomType::Monster,
        rewards: vec![Reward::Card(CardReward::empty())],
        hook_relics: Vec::new(),
    };
    (0..4)
        .map(|_| {
            let opts = CardRewardGenerationOptions {
                is_from_combat: from_combat,
                ..CardRewardGenerationOptions::default()
            };
            let cards = sts2_content::cards::spec_index::roll_card_reward_cards(
                &mut rs,
                opts,
                sts2_core::CharacterId::Ironclad,
            );
            dispatch_before_combat_reward_offered(&mut rs, &set);
            let last_is_power = cards.last().is_some_and(|c| c.card_type == CardType::Power);
            (cards.len(), last_is_power)
        })
        .collect()
}

/// v0.111.0 LastingCandy: the count of card-bearing combat reward sets is
/// bumped after each set is populated, and an odd count adds a Power, so the
/// 2nd and 4th combat card rewards get a 4th card (decomp
/// LastingCandy.cs:74-76,153-170). Only the combat room's own card reward
/// qualifies (`IsFromCombat`, :122).
#[test]
fn lasting_candy_extra_power_on_every_second_combat_reward_v0111() {
    let sizes: Vec<usize> = lasting_candy_reward_sizes(GameVersion::V0_111_0, true)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(sizes, vec![3, 4, 3, 4]);
    assert!(lasting_candy_reward_sizes(GameVersion::V0_111_0, true)[1].1);
    assert!(lasting_candy_reward_sizes(GameVersion::V0_111_0, false)
        .iter()
        .all(|&(n, _)| n == 3));
    // v0.107.1 counts combat ends on the combat side; the run-side hook
    // never fires, so the bare reward roll never adds a card here.
    assert!(lasting_candy_reward_sizes(GameVersion::V0_107_1, true)
        .iter()
        .all(|&(n, _)| n == 3));
}

/// The 4th card LastingCandy adds to the 2nd combat card reward of a v0.111.0
/// Ironclad run seeded `seed` holding `extra` too, or `None` if none was added.
fn lasting_candy_fourth_card(seed: i32, extra: &[sts2_core::RelicId]) -> Option<sts2_core::Card> {
    use sts2_core::run_effects::{dispatch_before_combat_reward_offered, CombatRewardsSet};
    use sts2_core::runstate::{CardReward, CardRewardGenerationOptions, Reward, RoomType};
    sts2_content::relics::dispatch::ensure_installed();
    content_dispatch::ensure_installed();
    let mut rs = sts2_core::runstate::RunState::new();
    rs.game_version = GameVersion::V0_111_0;
    rs.character_id = sts2_core::CharacterId::Ironclad;
    rs.rng = sts2_core::RunRngSet::new(seed);
    rs.relics.push(sts2_core::RelicId::LastingCandy);
    rs.relics.extend_from_slice(extra);
    let set = CombatRewardsSet {
        room_type: RoomType::Monster,
        rewards: vec![Reward::Card(CardReward::empty())],
        hook_relics: Vec::new(),
    };
    let mut fourth = None;
    for _ in 0..2 {
        let opts = CardRewardGenerationOptions {
            is_from_combat: true,
            ..CardRewardGenerationOptions::default()
        };
        let cards = sts2_content::cards::spec_index::roll_card_reward_cards(
            &mut rs,
            opts,
            sts2_core::CharacterId::Ironclad,
        );
        dispatch_before_combat_reward_offered(&mut rs, &set);
        fourth = cards.get(3).copied();
    }
    fourth
}

/// v0.111.0 LastingCandy filters Powers out of the reward's pools AFTER the
/// `ModifyCardRewardCreationOptions` fold: C#'s option mutators change the
/// instance in place (Runs/CardCreationOptions.cs:145-151; DingyRug.cs:27
/// widens it that way), CardReward.cs:115/149/159 passes that instance to
/// `TryModifyCardRewardOptions`, and LastingCandy.cs:108,119 reads its
/// `CardPools`. With DingyRug the extra Power can be a colorless one.
#[test]
fn lasting_candy_reads_dingy_rug_widened_pool_v0111() {
    let mut colorless = 0;
    for seed in 0..64 {
        let card = lasting_candy_fourth_card(seed, &[sts2_core::RelicId::DingyRug])
            .expect("the 2nd combat reward gets a 4th card");
        assert_eq!(card.card_type, CardType::Power, "seed {seed}: {card:?}");
        if sts2_content::cards::spec_index::character_for_card_id(card.card_id).is_none() {
            colorless += 1;
        }
    }
    assert!(colorless > 0, "DingyRug's colorless Powers are candidates");
    // Without DingyRug the candidates are Ironclad Powers only.
    for seed in 0..64 {
        let card = lasting_candy_fourth_card(seed, &[]).expect("4th card");
        assert_eq!(
            sts2_content::cards::spec_index::character_for_card_id(card.card_id),
            Some(sts2_core::CharacterId::Ironclad)
        );
    }
}

/// Expose strips the target's Block with the player as remover. v0.111.0
/// HandDrill fires on that non-damage break too (2) before Expose's own
/// Vulnerable (2); v0.107.1 HandDrill only saw damage.
#[test]
fn hand_drill_fires_on_expose_strip_v0111() {
    for (version, expect) in [(GameVersion::V0_107_1, 2), (GameVersion::V0_111_0, 4)] {
        let mut state = fresh(version, &[RelicKind::HandDrill]);
        state.creatures[0].block = 5;
        push_spec_card(&mut state, CardId::Expose);
        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
        assert_eq!(enemy_vulnerable(&state), expect, "{version}");
    }
}

/// v0.107.1 Pendulum ticks on the `AfterPlayerTurnStart` relic walk
/// (decomp v0.107.1 Pendulum.cs:75-86). `Hook.cs` checks `IsOverOrEnding`
/// only when that walk starts, so a Rolling Boulder (an earlier listener in
/// the same walk) that kills the last enemy still ticks the counter; the draw
/// is skipped because `CardPileCmd.Draw` returns early once the combat is
/// ending (CardPileCmd.cs:822). Rust used to tick on `AfterSideTurnStart`,
/// which never runs on an ending combat, and fell a turn behind.
#[test]
fn pendulum_ticks_when_an_earlier_turn_start_listener_wins_v0107() {
    let mut state = fresh(GameVersion::V0_107_1, &[RelicKind::Pendulum]);
    state.relics[0].counter = 2;
    state.creatures[0].current_hp = 1;
    state.phase = CombatPhase::PlayerTurnStart;
    for _ in 0..8 {
        push_hand_card(&mut state, CardId::StrikeIronclad, CardType::Attack);
    }
    state.player.draw = std::mem::take(&mut state.player.hand).into_iter().collect();
    let puid = sts2_core::PowerUid(NonZeroU32::new(1).expect("nz"));
    state.powers.push(sts2_core::Power {
        id: puid.0,
        kind: sts2_core::PowerKind::RollingBoulder,
        amount: 10,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(puid);

    let _ = sts2_core::combat_loop::start_player_turn(&mut state);

    assert_eq!(
        state.creatures[0].current_hp, 0,
        "the boulder won the fight"
    );
    assert_eq!(
        state.relics[0].counter, 0,
        "TurnsSeen 2 -> 0: the walk that started before the kill still ticks"
    );
    assert_eq!(
        state.player.hand.len(),
        5,
        "no Pendulum card: the draw is a no-op once the combat is ending"
    );
}

/// v0.107.1 Pendulum on a live turn: 3rd turn start draws one card after the
/// hand draw, from the `AfterPlayerTurnStart` relic walk.
#[test]
fn pendulum_draws_on_third_turn_start_v0107() {
    let mut state = fresh(GameVersion::V0_107_1, &[RelicKind::Pendulum]);
    state.relics[0].counter = 2;
    state.phase = CombatPhase::PlayerTurnStart;
    for _ in 0..8 {
        push_hand_card(&mut state, CardId::StrikeIronclad, CardType::Attack);
    }
    state.player.draw = std::mem::take(&mut state.player.hand).into_iter().collect();

    let _ = sts2_core::combat_loop::start_player_turn(&mut state);

    assert_eq!(state.relics[0].counter, 0);
    assert_eq!(state.player.hand.len(), 6, "5 hand draw + 1 Pendulum card");
}

/// A HistoryCourse dupe ends in `OnPlayWrapper`'s `CheckForEmptyHand`
/// (CardModel.cs:1993 v0.107.1, :2006 v0.111.0) like any top-level play: an
/// Eidolon dupe that exhausts the hand lets Unceasing Top draw (loadout
/// Necrobinder 3373 v0.107.1). v0.107.1 only: v0.111.0 HistoryCourse replays
/// Attacks alone, and Eidolon is a Skill.
#[test]
fn history_course_eidolon_dupe_emptying_the_hand_fires_unceasing_top_v0107() {
    let version = GameVersion::V0_107_1;
    let mut state = fresh(
        version,
        &[RelicKind::HistoryCourse, RelicKind::UnceasingTop],
    );
    push_hand_card(&mut state, CardId::DefendIronclad, CardType::Skill);
    push_hand_card(&mut state, CardId::StrikeIronclad, CardType::Attack);
    let top = state.player.hand.pop().expect("strike");
    state.player.draw.push(top);
    state.round = 2;
    state.combat_log.prev_round_last_attack_skill_played = Some((CardId::Eidolon, false, None));
    let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
    let hand: Vec<CardId> = state
        .player
        .hand
        .iter()
        .map(|&u| {
            sts2_core::card_by_uid(&state.cards, u)
                .expect("card")
                .card_id
        })
        .collect();
    assert_eq!(hand, vec![CardId::StrikeIronclad], "{version:?}");
    assert!(state.player.draw.is_empty(), "{version:?}");
}
