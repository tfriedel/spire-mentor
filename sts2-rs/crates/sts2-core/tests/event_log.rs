//! Integration tests for the game-style combat log (`sts2_core::event_log`).
//!
//! Two things are being proven here, and the second is the reason the log
//! lives on the state at all:
//!
//! 1. A real combat records the rows the game's own log shows — the card play,
//!    the energy, the hit (with dealer, receiver, amount and the card it came
//!    from), the block.
//! 2. It records them on ANY thread. [`sts2_core::damage_trace`] arms in a
//!    thread-local, so a caller that arms on one thread and runs the step on a
//!    worker gets an empty trace back; the state-resident log cannot have that
//!    failure mode, and `records_on_a_worker_thread` is the regression test
//!    that says so.

use smallvec::SmallVec;

use sts2_core::cards::{Card, CardId, CardTags, CardType, TargetType};
use sts2_core::event_log::{EventLogEntry, LogCreature, LogSource};
use sts2_core::potions::CharacterId;
use sts2_core::PowerKind;
use sts2_core::{
    apply, Action, CardUid, CombatPhase, CombatState, Creature, CreatureUid, IntentHistory,
    MonsterId, MonsterKind, Side, StepResult,
};

fn make_card(
    uid_raw: u32,
    card_id: CardId,
    card_type: CardType,
    target_type: TargetType,
    cost: i8,
    base_damage: i32,
    base_block: i32,
) -> Card {
    Card {
        uid: CardUid::new(uid_raw).expect("non-zero uid"),
        card_id,
        cost: sts2_core::CardEnergyCost::new(cost),
        card_type,
        target_type,
        upgraded: false,
        base_damage,
        base_block,
        combat_vars: sts2_core::cards::CombatVars::new(),
        enchantments: sts2_core::cards::CardEnchantments::new(),
        tags: CardTags::NONE,
        energy_spent: 0,
        base_replay_count: 0,
        keyword_override: None,
        affliction: None,
    }
}

/// Ironclad in the play phase with `Strike` + `Defend` in hand and one fat
/// Nibbit to hit. Recording is armed iff `record`.
fn fixture(record: bool) -> CombatState {
    sts2_content::cards::dispatch::ensure_installed();

    let mut s = CombatState {
        phase: CombatPhase::PlayerPlayPhase,
        round: 1,
        side: Side::Player,
        ..CombatState::default()
    };

    s.cards = vec![
        make_card(
            1,
            CardId::StrikeIronclad,
            CardType::Attack,
            TargetType::Enemy,
            1,
            6,
            0,
        ),
        make_card(
            2,
            CardId::DefendIronclad,
            CardType::Skill,
            TargetType::Self_,
            1,
            0,
            5,
        ),
    ]
    .into();
    let mut hand: SmallVec<[CardUid; 16]> = SmallVec::new();
    hand.push(CardUid::new(1).expect("non-zero"));
    hand.push(CardUid::new(2).expect("non-zero"));
    s.player.hand = hand;
    s.player.current_hp = 80;
    s.player.max_hp = 80;
    s.player.energy = 3;
    s.player.max_energy = 3;
    s.player.character = Some(CharacterId::Ironclad);

    let enemy_uid = CreatureUid::new(1).expect("non-zero");
    s.creatures = vec![Creature {
        id: enemy_uid.0,
        current_hp: 40,
        max_hp: 40,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    }];
    let mut enemy_uids: SmallVec<[CreatureUid; 8]> = SmallVec::new();
    enemy_uids.push(enemy_uid);
    s.enemy_uids = enemy_uids;

    s.combat_log.event_log.set_recording(record);
    s
}

/// Play `hand_index` at `target`, asserting the engine actually advanced.
fn play(s: &mut CombatState, hand_index: u8, target: Option<u8>) {
    assert_eq!(
        apply(s, Action::PlayCard { hand_index, target }),
        StepResult::Advanced,
        "play must advance"
    );
}

#[test]
fn a_strike_records_the_play_the_energy_and_the_hit() {
    let mut s = fixture(true);
    play(&mut s, 0, Some(0));

    let rows = s.combat_log.event_log.rows();
    // Energy FIRST, then the play. C# `CardModel.PlayInternal` awaits
    // `SpendResources()` (CardModel.cs:1817) before it records
    // `History.CardPlayStarted` (CardModel.cs:1931), so a log that opened with
    // the play would be the one that disagreed with the game.
    assert!(
        matches!(
            rows.first().map(|r| r.entry),
            Some(EventLogEntry::EnergySpent { amount: 1 })
        ),
        "Strike's energy is spent before the play opens: {rows:?}"
    );
    assert!(
        matches!(
            rows.get(1).map(|r| r.entry),
            Some(EventLogEntry::CardPlayStarted {
                card: CardId::StrikeIronclad,
                auto_played: false,
                ..
            })
        ),
        "the play follows the spend: {rows:?}"
    );
    assert!(
        rows.iter().any(|r| matches!(
            r.entry,
            EventLogEntry::CardPlayFinished {
                card: CardId::StrikeIronclad,
                ..
            }
        )),
        "the play closes: {rows:?}"
    );

    let hit = rows
        .iter()
        .find(|r| matches!(r.entry, EventLogEntry::DamageReceived { .. }))
        .expect("Strike deals damage");
    let EventLogEntry::DamageReceived {
        dealer,
        receiver,
        unblocked,
        hp_before,
        hp_after,
        ..
    } = hit.entry
    else {
        unreachable!()
    };
    assert_eq!(
        dealer,
        Some(LogCreature::Player(Some(CharacterId::Ironclad)))
    );
    assert!(
        matches!(receiver, LogCreature::Monster { id, .. } if id == MonsterId("Nibbit")),
        "the hit names the enemy it landed on: {receiver:?}"
    );
    assert_eq!(unblocked, 6, "base Strike");
    assert_eq!(
        (hp_before, hp_after),
        (40, 34),
        "HP pair brackets the write"
    );
}

#[test]
fn the_hit_names_the_card_that_dealt_it() {
    let mut s = fixture(true);
    play(&mut s, 0, Some(0));

    let hit = s
        .combat_log
        .event_log
        .rows()
        .iter()
        .find(|r| matches!(r.entry, EventLogEntry::DamageReceived { .. }))
        .expect("Strike deals damage");
    assert_eq!(
        hit.source,
        Some(LogSource::Card {
            card: CardId::StrikeIronclad,
            upgraded: false
        }),
        "\"using Strike\" — the attribution the game's log shows"
    );
    assert_eq!(
        hit.human_readable(Some(CharacterId::Ironclad)),
        "Rd 1 (Player turn): IRONCLAD dealt 6 damage to Nibbit.",
        "C# CombatHistoryEntry.HumanReadableString, verbatim"
    );
}

#[test]
fn defend_records_a_block_gain() {
    let mut s = fixture(true);
    play(&mut s, 1, None);

    let block = s
        .combat_log
        .event_log
        .rows()
        .iter()
        .find_map(|r| match r.entry {
            EventLogEntry::BlockGained { amount, .. } => Some(amount),
            _ => None,
        })
        .expect("Defend gains block");
    assert_eq!(block, 5);
}

#[test]
fn a_disarmed_combat_records_nothing() {
    let mut s = fixture(false);
    play(&mut s, 0, Some(0));
    assert!(
        s.combat_log.event_log.rows().is_empty(),
        "recording is opt-in; a headless run must pay nothing"
    );
    assert_eq!(s.creatures[0].current_hp, 34, "…but the combat still ran");
}

#[test]
fn records_on_a_worker_thread() {
    // The whole point of state residency. `damage_trace` would come back empty
    // here: it arms in a thread-local, and the work happens on a thread that
    // was never armed. The log travels WITH the state, so it does not care.
    let s = fixture(true);
    let played = std::thread::spawn(move || {
        let mut s = s;
        play(&mut s, 0, Some(0));
        s
    })
    .join()
    .expect("worker thread panicked");

    let rows = played.combat_log.event_log.rows();
    assert!(
        rows.iter()
            .any(|r| matches!(r.entry, EventLogEntry::DamageReceived { .. })),
        "a combat driven on another thread records exactly as it does here: {rows:?}"
    );
}

#[test]
fn recording_survives_a_clone() {
    // Search, replay and the workbench's branch tree all clone states. A clone
    // that lost the flag would silently stop logging mid-fight.
    let s = fixture(true);
    let mut copy = s.clone();
    play(&mut copy, 0, Some(0));
    assert!(
        !copy.combat_log.event_log.rows().is_empty(),
        "a cloned state keeps recording"
    );
    assert!(
        s.combat_log.event_log.rows().is_empty(),
        "…and the original is untouched by the clone's rows"
    );
}

// ---------------------------------------------------------------------------
// `PowerFlashed` — a power TRIGGERING, which the C# combat history does not
// record at all. See the variant's own doc for why it lives in this log
// anyway.
// ---------------------------------------------------------------------------

/// How many `PowerFlashed` rows name `power`.
fn flash_rows(s: &CombatState, power: PowerKind) -> usize {
    s.combat_log
        .event_log
        .rows()
        .iter()
        .filter(|r| matches!(r.entry, EventLogEntry::PowerFlashed { power: p, .. } if p == power))
        .count()
}

#[test]
fn power_flash_records_only_when_armed() {
    let mut s = fixture(false);
    let uid = CreatureUid::new(1).expect("non-zero");

    sts2_core::event_log::flash_power(&mut s, uid, PowerKind::Thorns);
    assert_eq!(
        s.combat_log.event_log.rows().len(),
        0,
        "a disarmed log must record nothing"
    );

    s.combat_log.event_log.set_recording(true);
    sts2_core::event_log::flash_power(&mut s, uid, PowerKind::Thorns);
    assert_eq!(flash_rows(&s, PowerKind::Thorns), 1);
}

#[test]
fn power_flash_names_the_creature_whose_power_fired() {
    let mut s = fixture(true);
    let uid = CreatureUid::new(1).expect("non-zero");

    sts2_core::event_log::flash_power(&mut s, uid, PowerKind::Thorns);
    sts2_core::event_log::flash_player_power(&mut s, PowerKind::Vulnerable);

    let rows = s.combat_log.event_log.rows();
    // The monster's flash resolves to the monster, not to whoever is acting:
    // a flash is about the power's OWNER, and getting that wrong would draw
    // the effect on the wrong side of the board.
    assert!(matches!(
        rows[0].entry,
        EventLogEntry::PowerFlashed {
            creature: LogCreature::Monster { .. },
            power: PowerKind::Thorns
        }
    ));
    assert!(matches!(
        rows[1].entry,
        EventLogEntry::PowerFlashed {
            creature: LogCreature::Player(_),
            power: PowerKind::Vulnerable
        }
    ));
}

#[test]
fn power_flash_is_not_a_combat_history_entry() {
    // C# has no `PowerFlashedEntry`: the flash is a VIEW signal, not a history
    // row. `kind()` must not end in "Entry", so the `[HISTTRACE]` diff against
    // the world side stays line-for-line.
    let e = EventLogEntry::PowerFlashed {
        creature: LogCreature::Player(Some(CharacterId::Ironclad)),
        power: PowerKind::Thorns,
    };
    assert_eq!(e.kind(), "PowerFlashed");
    assert!(!e.kind().ends_with("Entry"));
    assert!(
        e.description(Some(CharacterId::Ironclad)).is_empty(),
        "a row C# never prints must render no description"
    );
}
