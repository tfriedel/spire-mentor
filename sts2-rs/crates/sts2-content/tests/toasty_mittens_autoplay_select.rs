//! Toasty Mittens' turn-start exhaust fires Dark Embrace, whose draw takes a
//! Strike that Hellraiser auto-plays, and that Strike opens a select (Seeker
//! Strike). C# awaits all of it inside `ToastyMittens.BeforeHandDraw`: the
//! select, the Strike's move to the discard, the drawn card's
//! `AfterCardDrawn` (Chains of Binding's Bound), then the relic's Strength.
//! Only then does the hand draw run (random-loadout Ironclad 9072 v0.107.1:
//! Rust drew the hand first, so the Strike stayed in Play, the Strength was
//! never granted and Bound went on a third hand card).

use std::num::NonZeroU32;

use smallvec::SmallVec;

use sts2_content::cards::v0111::card_pod_for;
use sts2_content::monster_dispatch::{build_ai, MonsterSpawnArgs, SpawnPosition};
use sts2_core::card_afflictions::CardAfflictionKind;
use sts2_core::card_effects::apply_power_to_creature;
use sts2_core::cards::CardId;
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::{
    apply, Action, CardUid, CombatState, Creature, CreatureUid, GameVersion, IntentHistory,
    MonsterId, MonsterKind, PlayerCombatState, PowerKind, Relic, RelicKind,
};

#[test]
#[serial_test::serial(global_registry)]
fn mittens_waits_for_the_auto_played_strikes_select_before_the_hand_draw() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let version = GameVersion::V0_107_1;

    let mut cards = Vec::new();
    let mut push = |id: CardId| -> CardUid {
        let uid = CardUid::new(u32::try_from(cards.len() + 1).expect("fits")).expect("nz");
        let mut card = card_pod_for(id, false, version).expect("spec");
        card.uid = uid;
        cards.push(card);
        uid
    };
    let havoc = push(CardId::Havoc);
    let seeker = push(CardId::SeekerStrike);
    let rest: Vec<CardUid> = [
        CardId::Finesse,
        CardId::Headbutt,
        CardId::MayhemCard,
        CardId::BodySlam,
        CardId::SwordBoomerang,
        CardId::BloodWall,
        CardId::DefendIronclad,
        CardId::DefendIronclad,
        CardId::DefendIronclad,
        CardId::DefendIronclad,
    ]
    .into_iter()
    .map(&mut push)
    .collect();
    let mut draw: SmallVec<[CardUid; 32]> = SmallVec::from_slice(&[havoc, seeker]);
    draw.extend(rest.iter().copied());

    let enemy_uid = CreatureUid::new(1).expect("nz");
    let creatures = vec![Creature {
        id: enemy_uid.0,
        current_hp: 400,
        max_hp: 400,
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
    let player = PlayerCombatState {
        draw: draw.clone(),
        max_energy: 3,
        current_hp: 400,
        max_hp: 400,
        character: Some(sts2_core::potions::CharacterId::Ironclad),
        ..PlayerCombatState::default()
    };
    let mut state = CombatState {
        game_version: version,
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        creatures,
        cards: cards.into(),
        player,
        enemy_uids: SmallVec::from_slice(&[enemy_uid]),
        ..CombatState::default()
    };
    state.relics.push(Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: RelicKind::ToastyMittens,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    let _ = start_combat(&mut state);
    let _ = start_player_turn(&mut state);
    // Turn 1 is over; set the piles for turn 2's start.
    state.player.hand.clear();
    state.player.discard.clear();
    state.player.exhaust.clear();
    state.player.draw = draw;
    let puid = state.player.creature_uid.expect("player creature");
    for kind in [PowerKind::DarkEmbrace, PowerKind::Hellraiser] {
        apply_power_to_creature(&mut state, puid, kind, 1);
    }
    // The Queen's PUPPET_STRINGS puts Chains of Binding on the legacy player
    // power surface.
    let next_id = state.powers.iter().map(|p| p.id.get()).max().unwrap_or(0) + 1;
    let id = NonZeroU32::new(next_id).expect("nz");
    state.powers.push(sts2_core::Power {
        id,
        kind: PowerKind::ChainsOfBinding,
        amount: 3,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(sts2_core::PowerUid(id));
    let strength = state.player_power_amount(PowerKind::Strength);

    let _ = apply(&mut state, Action::EndTurn);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("Seeker Strike's select is open");
    let pick = pc.options[0].card_uid.expect("a draw-pile card");
    assert!(
        state.player.exhaust.contains(&havoc),
        "Mittens exhausted Havoc"
    );
    assert!(
        state.player.hand.is_empty(),
        "no hand draw under the select"
    );

    let _ = apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    assert!(state.pending_choice.is_none());
    assert!(
        state.player.discard.contains(&seeker),
        "Seeker Strike finished before the hand draw"
    );
    assert_eq!(
        state.player_power_amount(PowerKind::Strength),
        strength + 1,
        "Mittens' Strength"
    );
    assert_eq!(state.player.hand[0], pick, "the pick, then the hand draw");
    assert_eq!(state.player.hand.len(), 6);
    let bound: Vec<bool> = state
        .player
        .hand
        .iter()
        .map(|uid| {
            state.cards[uid.0.get() as usize - 1]
                .affliction
                .is_some_and(|a| a.kind == CardAfflictionKind::Bound)
        })
        .collect();
    assert_eq!(
        bound,
        [false, true, true, false, false, false],
        "Seeker Strike took the first of the 3 Bounds"
    );
}
