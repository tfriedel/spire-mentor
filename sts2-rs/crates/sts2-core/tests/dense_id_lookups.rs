//! The dense id lookups (J07): [`creature_by_id`], [`creature_by_id_mut`],
//! [`creature_pos`], [`card_by_uid`] and [`card_by_uid_mut`] test the dense
//! `arena[id - 1]` slot and fall back to the linear scan the engine used to do.
//! They must answer exactly what the first-match scan answered, on both
//! branches:
//!
//!   * hand-built arenas: dense, sparse (skipped ids, as a fixture arena may
//!     have), permuted, and the player-mirror teardown;
//!   * every state of random combats whose creature arena grows or changes
//!     mid-fight: Infested wrigglers from a dying PhrogParasite, the
//!     Fabricator's bots, LivingFog's gas bombs, Fogmog's eye, Ovicopter's
//!     eggs, TestSubject's revives and the Necrobinder's Osty summons. Each
//!     state is checked as it is (production arenas stay dense, so the slot
//!     answers) and with its arenas reversed (so the fallback answers).
//!
//! Debug builds also assert at every lookup that the id is unique, which is
//! what makes the dense answer equal the first-match scan.

use std::num::NonZeroU32;

use smallvec::SmallVec;

use sts2_content::monster_dispatch::MonsterSpawnArgs;
use sts2_core::cards::{Card, CardId};
use sts2_core::invariants::check_combat_invariants;
use sts2_core::potions::CharacterId;
use sts2_core::runstate::{RoomType, RunState};
use sts2_core::{
    apply, card_by_uid, card_by_uid_mut, creature_by_id, creature_by_id_mut, creature_pos,
    legal_actions, CardUid, CombatState, Creature, IntentHistory, MonsterId, MonsterKind,
};

fn nz(raw: usize) -> NonZeroU32 {
    NonZeroU32::new(u32::try_from(raw).expect("fits")).expect("non-zero")
}

fn creature(raw: usize) -> Creature {
    Creature {
        id: nz(raw),
        current_hp: 10,
        max_hp: 10,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        monster_id: MonsterId("Nibbit"),
        intent: None,
        intent_history: IntentHistory::default(),
        ai: None,
    }
}

fn card(raw: usize) -> Card {
    Card {
        uid: CardUid(nz(raw)),
        card_id: CardId::StrikeIronclad,
        ..Card::pod_defaults()
    }
}

/// How the helper answers were reached across a run of checks.
#[derive(Default, Debug)]
struct Tally {
    /// The id sat in its dense slot.
    dense: usize,
    /// The id was present but not in its dense slot (fallback scan found it).
    fallback: usize,
    /// The id was absent (both branches return `None`).
    absent: usize,
}

/// Every creature lookup for ids `1..=len + 2` on `arena` equals the
/// first-match scan, for all three creature helpers.
fn check_creature_arena(arena: &[Creature], tally: &mut Tally) {
    for raw in 1..=arena.len() + 2 {
        let id = nz(raw);
        let scan = arena.iter().position(|c| c.id == id);
        match scan {
            Some(i) if i + 1 == raw => tally.dense += 1,
            Some(_) => tally.fallback += 1,
            None => tally.absent += 1,
        }
        assert_eq!(creature_pos(arena, id), scan, "creature_pos({raw})");
        assert_eq!(
            creature_by_id(arena, id).map(std::ptr::from_ref),
            scan.map(|i| std::ptr::from_ref(&arena[i])),
            "creature_by_id({raw})"
        );
        let mut probe = arena.to_vec();
        if let Some(c) = creature_by_id_mut(&mut probe, id) {
            c.block = -1;
        }
        let marked: Vec<usize> = (0..probe.len()).filter(|&i| probe[i].block == -1).collect();
        assert_eq!(
            marked,
            scan.into_iter().collect::<Vec<_>>(),
            "creature_by_id_mut({raw})"
        );
    }
}

/// Every card lookup for uids `1..=len + 2` on `arena` equals the first-match
/// scan, for both card helpers.
fn check_card_arena(arena: &[Card], tally: &mut Tally) {
    for raw in 1..=arena.len() + 2 {
        let uid = CardUid(nz(raw));
        let scan = arena.iter().position(|c| c.uid == uid);
        match scan {
            Some(i) if i + 1 == raw => tally.dense += 1,
            Some(_) => tally.fallback += 1,
            None => tally.absent += 1,
        }
        assert_eq!(
            card_by_uid(arena, uid).map(std::ptr::from_ref),
            scan.map(|i| std::ptr::from_ref(&arena[i])),
            "card_by_uid({raw})"
        );
        let mut probe = arena.to_vec();
        if let Some(c) = card_by_uid_mut(&mut probe, uid) {
            c.cost = sts2_core::CardEnergyCost::new(-7);
        }
        let marked: Vec<usize> = (0..probe.len())
            .filter(|&i| probe[i].local_cost() == -7)
            .collect();
        assert_eq!(
            marked,
            scan.into_iter().collect::<Vec<_>>(),
            "card_by_uid_mut({raw})"
        );
    }
}

#[test]
fn lookups_match_the_scan_on_dense_sparse_and_permuted_arenas() {
    let mut creatures = Tally::default();
    let mut cards = Tally::default();
    let layouts: [&[usize]; 5] = [&[1, 2, 3, 4], &[2, 5, 9], &[4, 3, 2, 1], &[1, 3, 2], &[]];
    for ids in layouts {
        let arena: Vec<Creature> = ids.iter().map(|&r| creature(r)).collect();
        check_creature_arena(&arena, &mut creatures);
        let arena: Vec<Card> = ids.iter().map(|&r| card(r)).collect();
        check_card_arena(&arena, &mut cards);
    }
    for t in [&creatures, &cards] {
        assert!(t.dense > 0 && t.fallback > 0 && t.absent > 0, "{t:?}");
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "duplicate creature id 1")]
fn a_duplicate_creature_id_trips_the_debug_assert() {
    // The dense slot holds the first copy, so only the uniqueness assert can
    // notice that a `find(id == 1 && pred)` site might have meant the second.
    let arena = vec![creature(1), creature(1)];
    let _ = creature_by_id(&arena, nz(1));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "duplicate card uid")]
fn a_duplicate_card_uid_trips_the_debug_assert() {
    let arena = vec![card(2), card(1), card(2)];
    let _ = card_by_uid(&arena, CardUid(nz(2)));
}

// ── Real combats whose arenas change mid-fight ─────────────────────────

fn install() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
}

fn fight(
    character: CharacterId,
    extra: &[CardId],
    roster: &[(MonsterKind, i32)],
    seed: i32,
) -> CombatState {
    let mut rs = RunState {
        current_hp: 80,
        max_hp: 80,
        rng: sts2_core::run_rng_set::RunRngSet::new(seed),
        character_id: character,
        ..RunState::default()
    };
    sts2_content::run_callbacks::populate_starter_deck(&mut rs, character);
    for &id in extra {
        rs.master_deck.push(
            sts2_content::cards::spec_index::card_pod_from_id(id, false)
                .expect("card in the registry"),
        );
    }
    let roster: Vec<(MonsterKind, MonsterSpawnArgs, i32)> = roster
        .iter()
        .map(|&(kind, hp)| (kind, MonsterSpawnArgs::None, hp))
        .collect();
    sts2_content::run_bridge::enter_combat_from_roster(&rs, &roster, RoomType::Monster)
}

/// What the combats did to their arenas.
#[derive(Default, Debug)]
struct Events {
    /// Steps after which the creature arena held more rows than before.
    arena_grew: usize,
    /// Steps after which a 0-HP creature (enemy corpse or Osty) was alive.
    revived: usize,
    /// Steps that ended with the player's Osty alive.
    osty_alive: usize,
}

/// Check one combat state: invariants, dense production arenas, and every
/// lookup on the arenas as they are and reversed.
fn check_state(state: &CombatState, slot: &mut Tally, fallback: &mut Tally) {
    check_combat_invariants(state, "dense-id-lookups").expect("invariants hold");
    for (i, c) in state.creatures.iter().enumerate() {
        assert_eq!(
            c.id.get() as usize,
            i + 1,
            "creature arena left dense order"
        );
    }
    for (i, c) in state.cards.iter().enumerate() {
        assert_eq!(c.uid.get() as usize, i + 1, "card arena left dense order");
    }
    check_creature_arena(&state.creatures, slot);
    check_card_arena(&state.cards, slot);
    let mut creatures = state.creatures.clone();
    creatures.reverse();
    check_creature_arena(&creatures, fallback);
    let mut cards = state.cards.to_vec();
    cards.reverse();
    check_card_arena(&cards, fallback);
}

fn play(
    mut state: CombatState,
    seed: u64,
    slot: &mut Tally,
    fallback: &mut Tally,
    ev: &mut Events,
) {
    let mut agent = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut steps = 0;
    check_state(&state, slot, fallback);
    while !state.is_over && steps < 400 {
        let actions = legal_actions(&state);
        if actions.is_empty() {
            break;
        }
        agent ^= agent << 13;
        agent ^= agent >> 7;
        agent ^= agent << 17;
        let action = actions[usize::try_from(agent % actions.len() as u64).expect("fits")].clone();
        let before_len = state.creatures.len();
        let dead_before: Vec<NonZeroU32> = state
            .creatures
            .iter()
            .filter(|c| c.current_hp <= 0)
            .map(|c| c.id)
            .collect();
        let _ = apply(&mut state, action);
        steps += 1;
        check_state(&state, slot, fallback);
        ev.arena_grew += usize::from(state.creatures.len() > before_len);
        ev.revived += usize::from(
            dead_before
                .iter()
                .any(|&id| creature_by_id(&state.creatures, id).is_some_and(|c| c.current_hp > 0)),
        );
        ev.osty_alive += usize::from(
            state
                .player
                .pet_uid
                .and_then(|u| creature_by_id(&state.creatures, u.0))
                .is_some_and(|c| c.current_hp > 0),
        );
    }
}

#[test]
fn lookups_match_the_scan_through_spawns_revives_and_osty() {
    install();
    let ironclad_extra = [
        CardId::Inflame,
        CardId::Whirlwind,
        CardId::PommelStrike,
        CardId::Bash,
    ];
    let encounters: [&[(MonsterKind, i32)]; 6] = [
        &[(MonsterKind::PhrogParasite, 12)],
        &[(MonsterKind::Fabricator, 60)],
        &[(MonsterKind::LivingFog, 60)],
        &[(MonsterKind::Fogmog, 40)],
        &[(MonsterKind::Ovicopter, 60)],
        &[(MonsterKind::TestSubject, 14)],
    ];
    let mut slot = Tally::default();
    let mut fallback = Tally::default();
    let mut ev = Events::default();
    for roster in encounters {
        let mut per = Events::default();
        for seed in 1..=6 {
            let state = fight(CharacterId::Ironclad, &ironclad_extra, roster, seed);
            play(
                state,
                u64::from(seed.unsigned_abs()),
                &mut slot,
                &mut fallback,
                &mut per,
            );
        }
        eprintln!("{:?}: {per:?}", roster[0].0);
        // Every one of these monsters adds creatures mid-fight, except
        // TestSubject, which revives its own corpse instead.
        if roster[0].0 == MonsterKind::TestSubject {
            assert!(per.revived > 0, "{:?}: {per:?}", roster[0].0);
        } else {
            assert!(per.arena_grew > 0, "{:?}: {per:?}", roster[0].0);
        }
        ev.arena_grew += per.arena_grew;
        ev.revived += per.revived;
        ev.osty_alive += per.osty_alive;
    }
    let necro_extra = [
        CardId::Afterlife,
        CardId::Reanimate,
        CardId::LegionOfBone,
        CardId::PullAggro,
    ];
    for seed in 1..=8 {
        let state = fight(
            CharacterId::Necrobinder,
            &necro_extra,
            &[(MonsterKind::Nibbit, 60), (MonsterKind::Nibbit, 60)],
            seed,
        );
        play(
            state,
            u64::from(seed.unsigned_abs()),
            &mut slot,
            &mut fallback,
            &mut ev,
        );
    }
    eprintln!("slot {slot:?}\nreversed {fallback:?}\n{ev:?}");
    // Production arenas stay dense: every present id answers from its slot.
    assert_eq!(slot.fallback, 0, "{slot:?}");
    assert!(slot.dense > 0, "{slot:?}");
    // The reversed copies answer from the fallback scan.
    assert!(fallback.fallback > 0, "{fallback:?}");
    // The combats did change their arenas mid-fight.
    assert!(ev.arena_grew > 0, "{ev:?}");
    assert!(ev.revived > 0, "{ev:?}");
    assert!(ev.osty_alive > 0, "{ev:?}");
}

#[test]
fn lookups_match_the_scan_after_the_player_mirror_teardown() {
    install();
    let mut state = fight(
        CharacterId::Necrobinder,
        &[],
        &[(MonsterKind::Nibbit, 60), (MonsterKind::Nibbit, 60)],
        3,
    );
    let _ = state.summon_osty(5);
    assert!(state.creatures.len() >= 4, "player, two Nibbits and Osty");
    // Removing the player's mirror row shifts every later creature one slot
    // down, so no present id sits in its dense slot any more.
    state.teardown_player_creature();
    let mut t = Tally::default();
    check_creature_arena(&state.creatures, &mut t);
    assert_eq!(t.dense, 0, "{t:?}");
    assert_eq!(t.fallback, state.creatures.len(), "{t:?}");
    for uid in state.enemy_uids.clone() {
        assert_eq!(
            creature_by_id(&state.creatures, uid.0).map(|c| c.id),
            Some(uid.0),
            "enemy {uid:?} found through the fallback"
        );
    }
}

// ── The Free* gate in `consume_free_card_power` ────────────────────────

/// `consume_free_card_power` now returns early when its Free* kind is absent
/// from `present_kinds`. Each kind must still be consumed by its own card
/// type and only by it, alone and next to the other two kinds.
#[test]
fn free_card_powers_are_consumed_only_by_their_own_card_type() {
    use sts2_core::card_effects::apply_power_to_creature;
    use sts2_core::cards::CardType;
    use sts2_core::combat_loop::consume_free_card_power;
    use sts2_core::PowerKind;

    let pairs = [
        (PowerKind::FreeAttack, CardType::Attack),
        (PowerKind::FreeSkill, CardType::Skill),
        (PowerKind::FreePower, CardType::Power),
    ];
    let types = [
        CardType::Attack,
        CardType::Skill,
        CardType::Power,
        CardType::Curse,
        CardType::Status,
    ];
    for held in 0..8u8 {
        for &played in &types {
            let mut state = CombatState::default();
            state.player.current_hp = 80;
            state.player.max_hp = 80;
            let player = state.init_player_creature();
            for (bit, &(kind, _)) in pairs.iter().enumerate() {
                if held & (1 << bit) != 0 {
                    apply_power_to_creature(&mut state, player, kind, 2);
                }
            }
            let before: Vec<i32> = pairs
                .iter()
                .map(|&(k, _)| state.player_power_amount(k))
                .collect();
            let want = pairs
                .iter()
                .enumerate()
                .find(|&(bit, &(_, t))| t == played && held & (1 << bit) != 0)
                .map(|(_, &(k, _))| k);
            assert_eq!(
                consume_free_card_power(&mut state, played),
                want,
                "held {held:03b}, played {played:?}"
            );
            for (i, &(k, _)) in pairs.iter().enumerate() {
                let expect = before[i] - i32::from(Some(k) == want);
                assert_eq!(
                    state.player_power_amount(k),
                    expect,
                    "{k:?} after {played:?}"
                );
            }
        }
    }
}
