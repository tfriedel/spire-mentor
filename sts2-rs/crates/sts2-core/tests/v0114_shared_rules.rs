//! v0.114.0 changes to rules every character meets (GH #1924), each checked
//! against the build before it.

use sts2_core::{CombatState, GameVersion, MonsterKind, Power, PowerKind, PowerUid};

/// v0.114.0 `PainfulStabsPower` no longer overrides
/// `ShouldCreatureBeRemovedFromCombatAfterDeath`: a dead owner leaves combat.
#[test]
fn a_painful_stabs_corpse_leaves_combat_from_v0114() {
    for (version, removed) in [
        (GameVersion::V0_113_0, false),
        (GameVersion::V0_114_0, true),
    ] {
        let mut state = CombatState {
            game_version: version,
            ..CombatState::default()
        };
        let enemy = state.spawn_enemy(MonsterKind::Nibbit);
        let pid = PowerUid::new(1).expect("non-zero");
        state.powers.push(Power {
            id: pid.0,
            kind: PowerKind::PainfulStabs,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        let c = state
            .creatures
            .iter_mut()
            .find(|c| c.id == enemy.0)
            .expect("enemy");
        c.powers.push(pid);
        c.current_hp = 0;
        assert_eq!(
            state.should_remove_from_combat_on_death(enemy),
            removed,
            "{version:?}"
        );
    }
}

/// v0.114.0 Unmovable counts earlier card PLAYS that gained block this turn
/// (`DistinctBy(e => e.CardPlay)`), not the gains: a two-gain card is one.
#[test]
fn unmovable_counts_earlier_plays_from_v0114() {
    let card = |n: u32| sts2_core::CardUid::new(n).expect("non-zero");
    let mut log = sts2_core::TurnEvents::default();
    // Card 3 gains block twice in one play, then card 4 once; both finish.
    log.unmovable_card_block_plays.extend([3, 3]);
    log.retire_unmovable_play(card(3));
    log.unmovable_card_block_plays.push(4);
    log.retire_unmovable_play(card(4));
    // A third play (card 5) is resolving its first gain.
    log.unmovable_card_block_plays.push(5);
    assert_eq!(
        log.unmovable_other_plays(5, false),
        3,
        "v0.113: three gains"
    );
    assert_eq!(log.unmovable_other_plays(5, true), 2, "v0.114: two plays");
    assert_eq!(log.earlier_card_block_gains(false), 3);
    assert_eq!(log.earlier_card_block_gains(true), 2);
    // Replaying card 3 later is another play: its new gain is not the old one.
    log.retire_unmovable_play(card(5));
    log.unmovable_card_block_plays.push(3);
    assert_eq!(log.unmovable_other_plays(3, true), 3);
}

/// v0.114.0 `Wuppy.SummonPet` sets the pet's HP without `CreatureCmd`, so a
/// 0-HP Wuppy is no death: Melancholy keeps its cost.
#[test]
fn a_dead_wuppy_summon_is_no_death_from_v0114() {
    use sts2_core::cards::{Card, CardId, CardType, TargetType};
    use sts2_core::CardUid;
    for (version, cost) in [(GameVersion::V0_113_0, 2), (GameVersion::V0_114_0, 3)] {
        let mut state = CombatState {
            game_version: version,
            ..CombatState::default()
        };
        let uid = CardUid::new(1).expect("non-zero");
        state.cards.push(Card {
            uid,
            card_id: CardId::Melancholy,
            cost: sts2_core::CardEnergyCost::new(3),
            card_type: CardType::Skill,
            target_type: TargetType::Self_,
            ..Card::pod_defaults()
        });
        state.player.hand.push(uid);
        state.summon_wuppy(10, 0);
        let got = state.cards[0].local_cost().max(0);
        assert_eq!(got, cost, "{version:?}");
    }
}
