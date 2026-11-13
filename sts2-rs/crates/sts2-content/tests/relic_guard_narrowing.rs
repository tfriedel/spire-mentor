//! J03: the all-registered relic channels skip the handlers whose declared
//! guard says they cannot act (`register_owned_*`, `RelicGuard::Owned`).
//!
//! That is exact only if every `Owned` declaration is true: without a
//! non-melted relic of the declared kind, the handler returns its channel's
//! identity and leaves the whole state (RNG cursors, relic counters, event
//! log) untouched. These tests check that for every production registration,
//! on states from all five characters, and check that whole random combats
//! play the same on the narrowed walk as on the full walk.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};

use sts2_core::card_effects::EffectResult;
use sts2_core::cards::{CardId, CardType};
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::hooks::{
    AfterDeathCtx, BlockClearedCtx, CardCtx, CardPlayCountCtx, CombatLifecycleCtx, CreatureRef,
    DamageCtx, OrbChanneledCtx, PowerAmountCtx, StarsSpentCtx, ValueProps,
};
use sts2_core::potions::{CharacterId, PotionKind};
use sts2_core::relic_effects::{self as re, CardEventCtx, DiedToDoomCtx, RelicGuard, ShuffleCtx};
use sts2_core::relics::RelicId;
use sts2_core::{apply, legal_actions, CardUid, CombatState, CreatureUid, MonsterKind, Relic};
use sts2_core::{CombatPhase, PowerKind, RelicKind as K, Side, StepResult};

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
}

const CHARACTERS: [CharacterId; 5] = [
    CharacterId::Ironclad,
    CharacterId::Silent,
    CharacterId::Defect,
    CharacterId::Regent,
    CharacterId::Necrobinder,
];

/// Calls one handler with context `i` of its channel's battery for the given
/// state: `Some(result is the channel identity)`, or `None` past the end.
type Probe = Box<dyn Fn(&mut CombatState, usize) -> Option<bool>>;

/// One registration on a guarded channel.
struct Registration {
    channel: &'static str,
    id: RelicId,
    guard: RelicGuard,
    call: Probe,
}

fn first_enemy(state: &CombatState) -> CreatureUid {
    *state.enemy_uids.first().expect("an enemy")
}

fn player_uid(state: &CombatState) -> CreatureUid {
    state.player.creature_uid.expect("player creature")
}

/// The first arena card of `card_type`.
fn card_of(state: &CombatState, card_type: CardType) -> Option<CardUid> {
    state
        .cards
        .iter()
        .find(|c| c.card_type == card_type)
        .map(|c| c.uid)
}

fn card_event_ctx(state: &CombatState, uid: Option<CardUid>) -> CardEventCtx {
    let card = uid.and_then(|u| state.cards.iter().find(|c| c.uid == u));
    CardEventCtx {
        card_id: card.map_or(CardId::StrikeIronclad, |c| c.card_id),
        card_uid: uid,
        target: None,
        card_tags: card.map(|c| c.tags).unwrap_or_default(),
        energy_spent: 0,
    }
}

/// The power-amount contexts: a player Poison on an enemy from a card, an
/// applier-less Weak on an enemy from a card, enemy and applier-less
/// Strength on the player.
fn power_ctx(state: &CombatState, i: usize) -> Option<PowerAmountCtx> {
    let enemy = CreatureRef::Enemy(first_enemy(state));
    let card = card_of(state, CardType::Attack);
    let (applier, target, kind, amount, source_card) = match i {
        0 => (Some(CreatureRef::Player), enemy, PowerKind::Poison, 3, card),
        1 => (None, enemy, PowerKind::Weak, 2, card),
        2 => (
            Some(enemy),
            CreatureRef::Player,
            PowerKind::Strength,
            2,
            None,
        ),
        3 => (None, CreatureRef::Player, PowerKind::Strength, 2, None),
        _ => return None,
    };
    Some(PowerAmountCtx {
        source_card,
        ..PowerAmountCtx::new(applier, target, kind, amount)
    })
}

/// Every registration on the 28 guarded channels, each with a probe over a
/// small context battery. The identity a skipped handler must return:
/// `NotImplemented` on the event channels (`after_combat_end` excepted, see
/// below), `false`/`true` on the two vetoes, `0` on the sums, the input on
/// the folds.
fn registrations() -> Vec<Registration> {
    const NI: EffectResult = EffectResult::NotImplemented;
    let mut out: Vec<Registration> = Vec::new();
    macro_rules! collect {
        ($listeners:ident, $guards:ident, |$f:ident, $s:ident, $i:ident| $body:expr) => {{
            let guards = re::$guards();
            let listeners = re::$listeners();
            assert_eq!(guards.len(), listeners.len(), stringify!($guards));
            for (&(id, $f), &(guard_id, guard)) in listeners.iter().zip(&guards) {
                assert_eq!(id, guard_id, stringify!($guards));
                out.push(Registration {
                    channel: stringify!($listeners),
                    id,
                    guard,
                    call: Box::new(move |$s: &mut CombatState, $i: usize| $body),
                });
            }
        }};
    }
    collect!(
        listeners_before_hand_draw,
        listener_guards_before_hand_draw,
        |f, s, i| { (i == 0).then(|| f(s) == NI) }
    );
    collect!(
        listeners_after_card_exhausted,
        listener_guards_after_card_exhausted,
        |f, s, i| {
            let uid = match i {
                0 => card_of(s, CardType::Attack),
                1 => card_of(s, CardType::Skill),
                2 => card_of(s, CardType::Power),
                3 => None,
                _ => return None,
            };
            let ctx = card_event_ctx(s, uid);
            Some(f(s, &ctx) == NI)
        }
    );
    collect!(
        listeners_after_card_entered_combat,
        listener_guards_after_card_entered_combat,
        |f, s, i| {
            let uid = card_of(s, CardType::Attack).expect("an attack");
            (i < 2).then(|| f(s, uid, i == 0) == NI)
        }
    );
    collect!(
        listeners_after_potion_used,
        listener_guards_after_potion_used,
        |f, s, i| { (i == 0).then(|| f(s, PotionKind::BlockPotion, None) == NI) }
    );
    collect!(
        listeners_should_die_late,
        listener_guards_should_die_late,
        |f, s, i| { (i == 0).then(|| f(s) == NI) }
    );
    collect!(
        listeners_after_combat_victory,
        listener_guards_after_combat_victory,
        |f, s, i| { (i == 0).then(|| f(s) == NI) }
    );
    collect!(
        listeners_after_combat_end,
        listener_guards_after_combat_end,
        |f, s, i| { (i == 0).then(|| f(s) == NI) }
    );
    collect!(
        listeners_after_turn_end,
        listener_guards_after_turn_end,
        |f, s, i| {
            let side = [Side::Player, Side::Enemy].get(i).copied()?;
            Some(f(s, side) == NI)
        }
    );
    collect!(
        listeners_after_died_to_doom,
        listener_guards_after_died_to_doom,
        |f, s, i| {
            let victim = [first_enemy(s), player_uid(s)].get(i).copied()?;
            Some(f(s, &DiedToDoomCtx { victim }) == NI)
        }
    );
    collect!(
        listeners_after_card_discarded,
        listener_guards_after_card_discarded,
        |f, s, i| {
            let ctx = match i {
                0 => CardCtx::default(),
                1 => {
                    let uid = card_of(s, CardType::Attack);
                    CardCtx {
                        card_id: Some(CardId::StrikeIronclad),
                        card_uid: uid,
                        card_type: Some(CardType::Attack),
                        ..CardCtx::default()
                    }
                }
                _ => return None,
            };
            Some(f(s, &ctx) == NI)
        }
    );
    collect!(
        listeners_should_clear_block,
        listener_guards_should_clear_block,
        |f, s, i| {
            let ctx = BlockClearedCtx {
                cleared: Some(CreatureRef::Player),
            };
            (i == 0).then(|| f(s, &ctx))
        }
    );
    collect!(
        listeners_after_hand_emptied,
        listener_guards_after_hand_emptied,
        |f, s, i| { (i == 0).then(|| f(s, &CombatLifecycleCtx::default()) == NI) }
    );
    collect!(
        listeners_on_orb_channeled,
        listener_guards_on_orb_channeled,
        |f, s, i| { (i == 0).then(|| f(s, &OrbChanneledCtx::default()) == NI) }
    );
    collect!(
        listeners_on_stars_spent,
        listener_guards_on_stars_spent,
        |f, s, i| {
            let amount = [1, 10].get(i).copied()?;
            Some(f(s, &StarsSpentCtx { amount }) == NI)
        }
    );
    collect!(
        listeners_after_energy_reset,
        listener_guards_after_energy_reset,
        |f, s, i| { (i == 0).then(|| f(s) == NI) }
    );
    collect!(
        listeners_after_death,
        listener_guards_after_death,
        |f, s, i| {
            let ctx = match i {
                0 => AfterDeathCtx {
                    dead: first_enemy(s),
                    dead_side: Side::Enemy,
                },
                1 => AfterDeathCtx {
                    dead: player_uid(s),
                    dead_side: Side::Player,
                },
                _ => return None,
            };
            Some(f(s, &ctx) == NI)
        }
    );
    collect!(
        listeners_modify_hand_draw,
        listener_guards_modify_hand_draw,
        |f, s, i| { (i == 0).then(|| f(s, 5) == 5) }
    );
    collect!(
        listeners_modify_x_value,
        listener_guards_modify_x_value,
        |f, s, i| { (i == 0).then(|| f(s, 3) == 3) }
    );
    collect!(
        listeners_relic_modify_card_play_count,
        listener_guards_relic_modify_card_play_count,
        |f, s, i| {
            let ctx = CardPlayCountCtx {
                owner: CreatureRef::Player,
                card_id: Some(CardId::StrikeIronclad),
                card_uid: card_of(s, CardType::Attack),
                card_type: Some(CardType::Attack),
                card_tags: sts2_core::cards::CardTags::default(),
                target: Some(first_enemy(s)),
            };
            (i == 0).then(|| f(s, &ctx) == 0)
        }
    );
    collect!(
        listeners_modify_power_amount_given,
        listener_guards_modify_power_amount_given,
        |f, s, i| {
            let ctx = power_ctx(s, i)?;
            Some(f(s, &ctx) == ctx.amount)
        }
    );
    collect!(
        listeners_modify_power_amount_received,
        listener_guards_modify_power_amount_received,
        |f, s, i| {
            let ctx = power_ctx(s, i)?;
            Some(f(s, &ctx) == ctx.amount)
        }
    );
    collect!(
        listeners_before_power_amount_changed,
        listener_guards_before_power_amount_changed,
        |f, s, i| {
            let ctx = power_ctx(s, i)?;
            Some(f(s, &ctx) == NI)
        }
    );
    collect!(
        listeners_after_modifying_power_amount_given,
        listener_guards_after_modifying_power_amount_given,
        |f, s, i| {
            let ctx = power_ctx(s, i)?;
            Some(f(s, &ctx) == NI)
        }
    );
    collect!(
        listeners_after_modifying_power_amount_received,
        listener_guards_after_modifying_power_amount_received,
        |f, s, i| {
            let ctx = power_ctx(s, i)?;
            Some(f(s, &ctx) == NI)
        }
    );
    collect!(
        listeners_count_pets,
        listener_guards_count_pets,
        |f, s, i| { (i == 0).then(|| f(s) == 0) }
    );
    collect!(
        listeners_modify_vulnerable_multiplier,
        listener_guards_modify_vulnerable_multiplier,
        |f, s, i| {
            let enemy = CreatureRef::Enemy(first_enemy(s));
            let ctx = match i {
                0 => DamageCtx {
                    dealer: CreatureRef::Player,
                    target: enemy,
                    props: ValueProps::MOVE,
                },
                1 => DamageCtx {
                    dealer: enemy,
                    target: CreatureRef::Player,
                    props: ValueProps::MOVE,
                },
                _ => return None,
            };
            Some(f(s, &ctx) == 0)
        }
    );
    collect!(
        listeners_before_play_phase_start,
        listener_guards_before_play_phase_start,
        |f, s, i| { (i == 0).then(|| f(s) == NI) }
    );
    collect!(
        listeners_after_shuffle_bus,
        listener_guards_after_shuffle_bus,
        |f, s, i| {
            let from_reshuffle = [true, false].get(i).copied()?;
            Some(f(s, &ShuffleCtx { from_reshuffle }) == NI)
        }
    );
    out
}

/// A combat for `character`: its starter deck plus an Inflame (a Power card
/// for JeweledMask and Power-typed contexts), its starter relic, two Nibbits.
fn combat(character: CharacterId, relics: Vec<Relic>, seed: u64) -> CombatState {
    let seed_i32 = i32::try_from(seed).expect("small seed");
    let starter = sts2_core::characters::character_starter(character);
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(seed_i32),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(seed_i32),
        ..Default::default()
    };
    state.player.character = Some(character);
    state.player.current_hp = starter.hp;
    state.player.max_hp = starter.hp;
    state.player.max_energy = 3;
    state.init_player_creature();
    for hp in [300, 250] {
        let enemy = state.spawn_enemy(MonsterKind::Nibbit);
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
            c.max_hp = hp;
            c.current_hp = hp;
        }
    }
    state.relics = relics.into();
    let mut deck: Vec<CardId> = sts2_content::starters::starter_deck_for(character)
        .iter()
        .map(|spec| spec.id)
        .collect();
    deck.push(CardId::Inflame);
    for id in deck {
        let mut card = sts2_core::lookup_card_factory()
            .and_then(|f| f(id, false))
            .expect("card factory has the card");
        let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
        card.uid = uid;
        state.cards.push(card);
        state.player.draw.push(uid);
    }
    state
}

fn relic(i: usize, kind: K, counter: i32, melted: bool) -> Relic {
    Relic {
        id: std::num::NonZeroU32::new(u32::try_from(i + 1).expect("fits")).expect("nz"),
        kind,
        counter,
        is_wax: melted,
        is_melted: melted,
    }
}

/// Mid-combat states for every character: after the first turn's draw, and
/// after a few random plays.
fn base_states() -> Vec<(String, CombatState)> {
    let mut out = Vec::new();
    for character in CHARACTERS {
        let starter = sts2_core::characters::character_starter(character)
            .starter_relic
            .to_relic_kind()
            .expect("starter relic kind");
        let mut state = combat(character, vec![relic(0, starter, 0, false)], 7);
        let _ = start_combat(&mut state);
        let _ = start_player_turn(&mut state);
        if character == CharacterId::Necrobinder && state.creatures.len() < 4 {
            let _ = state.summon_osty(5);
        }
        out.push((format!("{character:?} turn 1"), state.clone()));
        let mut agent: u64 = 0x2545_F491_4F6C_DD1D;
        for _ in 0..4 {
            let actions: Vec<_> = legal_actions(&state)
                .into_iter()
                .filter(|a| !matches!(a, sts2_core::Action::EndTurn))
                .collect();
            if actions.is_empty() || state.is_over {
                break;
            }
            agent ^= agent << 13;
            agent ^= agent >> 7;
            agent ^= agent << 17;
            let action =
                actions[usize::try_from(agent % actions.len() as u64).expect("fits")].clone();
            let _ = apply(&mut state, action);
        }
        out.push((format!("{character:?} after plays"), state));
    }
    // GH #795: v0.111.0 copies, so the handlers gated on the game build
    // (Pendulum, PollinousCore, Regalite, ...) act in the battery too and
    // every handler's guard is checked in both builds.
    // v0.113.0 copies likewise reach the Fool relics, which act only there.
    let later: Vec<_> = out
        .iter()
        .flat_map(|(label, state)| {
            [
                (sts2_core::GameVersion::V0_111_0, "v0.111.0"),
                (sts2_core::GameVersion::V0_113_0, "v0.113.0"),
            ]
            .map(|(version, name)| {
                let mut state = state.clone();
                state.game_version = version;
                (format!("{label} {name}"), state)
            })
        })
        .collect();
    out.extend(later);
    out
}

/// Pin the state fields the guarded handlers trigger on, so the battery
/// reaches every handler's effect: low player HP (heals, LizardTail), Block
/// banked before the late turn end (ParryingShield), a banked ethereal
/// exhaust (JossPaper), last round's Attack (HistoryCourse), no potions
/// (BeltBuckle), the play phase on the player's side.
fn prime(state: &mut CombatState, round: u16, elite: bool) {
    state.round = round;
    state.is_elite = elite;
    state.side = Side::Player;
    state.phase = CombatPhase::PlayerPlayPhase;
    let hp = state.player.max_hp * 2 / 5;
    state.player.current_hp = hp;
    let uid = player_uid(state);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == uid.0) {
        c.current_hp = hp;
    }
    state.combat_log.block_before_late_turn_end = 20;
    state.combat_log.joss_paper_ethereal_count = 1;
    let attack = card_of(state, CardType::Attack).expect("an attack");
    let attack_id = state.cards[attack.0.get() as usize - 1].card_id;
    state.combat_log.prev_round_last_attack_skill_played = Some((attack_id, false, Some(attack)));
    state.player.potions.clear();
}

/// The relic kinds the battery holds: every kind declared by an `Owned`
/// guard, and the kind of every registered `RelicId`.
fn battery_kinds(regs: &[Registration]) -> Vec<K> {
    let mut kinds: BTreeSet<usize> = BTreeSet::new();
    for r in regs {
        if let RelicGuard::Owned(kind) = r.guard {
            kinds.insert(kind as usize);
        }
        if let Some(kind) = r.id.to_relic_kind() {
            kinds.insert(kind as usize);
        }
    }
    kinds.into_iter().map(|i| K::ALL[i]).collect()
}

/// Census of the guarded channels after the content install: each fits the
/// per-combat plan (at most 32 handlers, so none silently falls back to the
/// full walk), every `Owned` guard declares its own `RelicId`'s kind, and
/// the only production handler left `Always` is JossPaper's combat-end
/// reset, which writes `combat_log` whatever the relics.
#[test]
fn guarded_channels_are_declared_and_planned() {
    install();
    let regs = registrations();
    assert!(regs.len() >= 70, "{} registrations", regs.len());
    let mut per_channel: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &regs {
        *per_channel.entry(r.channel).or_default() += 1;
        match r.guard {
            RelicGuard::Owned(kind) => assert_eq!(
                r.id.to_relic_kind(),
                Some(kind),
                "{} {:?} is guarded on another relic's kind",
                r.channel,
                r.id
            ),
            // JossPaper writes state whatever is owned; PaperPhrog acts while
            // melted (C# `VulnerablePower` reads it via `GetRelic`), which the
            // `Owned` guard would skip.
            RelicGuard::Always => assert!(
                matches!(
                    (r.channel, r.id),
                    ("listeners_after_combat_end", RelicId::JossPaper)
                        | (
                            "listeners_modify_vulnerable_multiplier",
                            RelicId::PaperPhrog
                        )
                ),
                "a production handler without a declared guard (register_owned_*): {} {:?}",
                r.channel,
                r.id
            ),
        }
    }
    for (channel, n) in per_channel {
        assert!(
            n <= 32,
            "{channel} has {n} handlers: past the plan's 32-bit cap"
        );
    }
    assert!(re::all_channels_frozen());
}

/// Every `Owned` registration, on mid-combat states of all five characters
/// in both game builds (rounds 1 and 2, elite or not, relic counters 0, 1, 2
/// and 4), holding every
/// battery relic BUT its own kind, and again holding its own kind melted:
/// each context of its channel's battery returns the identity and leaves
/// the state `==` (RNG cursors, counters and event log included). The same
/// battery with its kind held must make every handler act at least once, or
/// the check above proves nothing for it. Registering a handler under the
/// wrong kind fails here: its real relic is in the battery and it acts.
#[test]
fn every_owned_handler_is_inert_without_its_relic() {
    install();
    let regs = registrations();
    let kinds = battery_kinds(&regs);
    let stores = |counter: i32| {
        let present: sts2_core::RelicStore = kinds
            .iter()
            .enumerate()
            .map(|(i, &k)| relic(i, k, counter, false))
            .collect();
        let absent: Vec<sts2_core::RelicStore> = kinds
            .iter()
            .map(|&without| {
                kinds
                    .iter()
                    .filter(|&&k| k != without)
                    .enumerate()
                    .map(|(i, &k)| relic(i, k, counter, false))
                    .collect()
            })
            .collect();
        let melted: Vec<sts2_core::RelicStore> = kinds
            .iter()
            .map(|&m| {
                kinds
                    .iter()
                    .enumerate()
                    .map(|(i, &k)| relic(i, k, counter, k == m))
                    .collect()
            })
            .collect();
        (present, absent, melted)
    };
    // Counter 2 reaches v0.111.0 Pendulum's every-3rd-turn arm (GH #795).
    let batteries: Vec<_> = [0, 1, 2, 4].into_iter().map(|c| (c, stores(c))).collect();
    let bases = base_states();
    let mut acted: BTreeSet<(&str, RelicId)> = BTreeSet::new();
    let mut inert_checks = 0usize;
    for (label, base) in &bases {
        for (round, elite, armed) in [
            (1u16, false, false),
            (1, true, false),
            (2, false, false),
            (2, true, false),
            (1, true, true),
            (2, false, true),
        ] {
            let mut primed = base.clone();
            prime(&mut primed, round, elite);
            primed.combat_log.event_log.set_recording(armed);
            for (counter, (present, absent, melted)) in &batteries {
                for r in &regs {
                    let RelicGuard::Owned(kind) = r.guard else {
                        continue;
                    };
                    let k = kinds.iter().position(|&x| x == kind).expect("battery kind");
                    let where_ = |what: &str, i: usize| {
                        format!(
                            "{} {:?} ({what}): {label}, round {round}, elite {elite}, \
                                 event log armed {armed}, counters {counter}, context {i}",
                            r.channel, r.id
                        )
                    };
                    for (what, store) in [("absent", &absent[k]), ("melted", &melted[k])] {
                        let mut before = primed.clone();
                        before.relics = store.clone();
                        for i in 0.. {
                            let mut probe = before.clone();
                            let Some(identity) = (r.call)(&mut probe, i) else {
                                break;
                            };
                            assert!(identity, "{}: not the identity", where_(what, i));
                            assert!(probe == before, "{}: changed the state", where_(what, i));
                            // `==` ignores the event log; the armed contexts
                            // exist to catch a handler that only writes it.
                            assert!(
                                probe
                                    .combat_log
                                    .event_log
                                    .same_contents(&before.combat_log.event_log),
                                "{}: wrote the event log",
                                where_(what, i)
                            );
                            inert_checks += 1;
                        }
                    }
                    let mut before = primed.clone();
                    before.relics = present.clone();
                    for i in 0.. {
                        let mut probe = before.clone();
                        let Some(identity) = (r.call)(&mut probe, i) else {
                            break;
                        };
                        if !identity || probe != before {
                            acted.insert((r.channel, r.id));
                        }
                    }
                }
            }
        }
    }
    let silent: Vec<(&str, RelicId)> = regs
        .iter()
        .filter(|r| matches!(r.guard, RelicGuard::Owned(_)))
        .map(|r| (r.channel, r.id))
        .filter(|key| !acted.contains(key))
        .collect();
    assert!(
        silent.is_empty(),
        "the battery never made these handlers act, so their guard went untested: {silent:?}"
    );
    assert!(inert_checks > 10_000, "{inert_checks} inert checks");
}

/// `(kind, melted)` relic lists for the replayed combats: guarded-channel
/// relics of every character, some melted, plus a duplicate.
fn layouts() -> Vec<(CharacterId, Vec<(K, bool)>)> {
    vec![
        (
            CharacterId::Ironclad,
            vec![
                (K::BurningBlood, false),
                (K::ThrowingAxe, false),
                (K::VenerableTeaSet, false),
                (K::JossPaper, false),
                (K::CharonsAshes, false),
                (K::ToastyMittens, true),
                (K::ArtOfWar, false),
                (K::Pocketwatch, false),
                (K::RuinedHelmet, false),
                (K::PaperPhrog, false),
                (K::BurningSticks, false),
                (K::Kunai, false),
                (K::LizardTail, false),
                (K::MeatOnTheBone, false),
                (K::SturdyClamp, true),
                (K::Fiddle, false),
            ],
        ),
        (
            CharacterId::Silent,
            vec![
                (K::RingOfTheSnake, false),
                (K::Tingsha, false),
                (K::ToughBandages, false),
                (K::SneckoSkull, false),
                (K::UnsettlingLamp, false),
                (K::TheAbacus, false),
                (K::BiiigHug, false),
                (K::BagOfPreparation, true),
                (K::PollinousCore, false),
                (K::ParryingShield, false),
                (K::Bookmark, false),
                (K::ForgottenSoul, false),
                (K::Toolbox, false),
                (K::ChemicalX, false),
            ],
        ),
        (
            CharacterId::Defect,
            vec![
                (K::CrackedCore, false),
                (K::Metronome, false),
                (K::BoomingConch, false),
                (K::BigMushroom, false),
                (K::FakeVenerableTeaSet, false),
                (K::HistoryCourse, false),
                (K::ReptileTrinket, false),
                (K::BeltBuckle, false),
                (K::JeweledMask, false),
                (K::BlessedAntler, false),
                (K::UnceasingTop, false),
                (K::SneckoEye, true),
                (K::GremlinHorn, false),
            ],
        ),
        (
            CharacterId::Regent,
            vec![
                (K::DivineRight, false),
                (K::GalacticDust, false),
                (K::MiniRegent, false),
                (K::LunarPastry, false),
                (K::RingOfTheDrake, false),
                (K::PaelsBlood, false),
                (K::Regalite, false),
                (K::FuneraryMask, true),
                (K::ThrowingAxe, false),
                (K::ThrowingAxe, true),
                (K::RedSkull, false),
                (K::BlackBlood, false),
            ],
        ),
        (
            CharacterId::Necrobinder,
            vec![
                (K::BoundPhylactery, false),
                (K::PhylacteryUnbound, false),
                (K::BookRepairKnife, false),
                (K::SwordOfStone, false),
                (K::LastingCandy, false),
                (K::EmotionChip, false),
                (K::Shuriken, false),
                (K::OrnamentalFan, true),
                (K::CentennialPuzzle, false),
                (K::LetterOpener, false),
            ],
        ),
    ]
}

/// Per-step hashes of one seeded uniform-random combat (the action, its
/// result, and the whole state's `Debug` form: every relic counter, RNG
/// cursor and, recorded on odd seeds, the event log).
fn random_combat(character: CharacterId, layout: &[(K, bool)], seed: u64) -> Vec<u64> {
    let relics: Vec<Relic> = layout
        .iter()
        .enumerate()
        .map(|(i, &(kind, melted))| relic(i, kind, kind.initial_counter(), melted))
        .collect();
    let mut state = combat(character, relics, seed);
    let _ = start_combat(&mut state);
    state.combat_log.event_log.set_recording(seed % 2 == 1);
    let _ = start_player_turn(&mut state);
    let mut agent = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut steps = Vec::new();
    while !state.is_over && steps.len() < 300 {
        let actions = legal_actions(&state);
        if actions.is_empty() {
            break;
        }
        agent ^= agent << 13;
        agent ^= agent >> 7;
        agent ^= agent << 17;
        let action = actions[usize::try_from(agent % actions.len() as u64).expect("fits")].clone();
        let result: StepResult = apply(&mut state, action.clone());
        let mut h = DefaultHasher::new();
        format!("{action:?}|{result:?}|{state:?}").hash(&mut h);
        steps.push(h.finish());
    }
    steps
}

fn all_combats() -> Vec<Vec<u64>> {
    let mut out = Vec::new();
    for (character, layout) in layouts() {
        for seed in 1..=6 {
            out.push(random_combat(character, &layout, seed));
        }
    }
    out
}

/// Re-register every handler of the 28 guarded channels with itself through
/// the plain `register_many_*`: the handlers stay, every guard becomes
/// `Always`, and each channel leaves its frozen list, so every dispatch walks
/// every handler from here on.
fn force_full_walks() {
    re::register_many_before_hand_draw(&re::listeners_before_hand_draw());
    re::register_many_after_card_exhausted(&re::listeners_after_card_exhausted());
    re::register_many_after_card_entered_combat(&re::listeners_after_card_entered_combat());
    re::register_many_after_potion_used(&re::listeners_after_potion_used());
    re::register_many_should_die_late(&re::listeners_should_die_late());
    re::register_many_after_combat_victory(&re::listeners_after_combat_victory());
    re::register_many_after_combat_end(&re::listeners_after_combat_end());
    re::register_many_after_turn_end(&re::listeners_after_turn_end());
    re::register_many_after_died_to_doom(&re::listeners_after_died_to_doom());
    re::register_many_after_card_discarded(&re::listeners_after_card_discarded());
    re::register_many_should_clear_block(&re::listeners_should_clear_block());
    re::register_many_after_hand_emptied(&re::listeners_after_hand_emptied());
    re::register_many_on_orb_channeled(&re::listeners_on_orb_channeled());
    re::register_many_on_stars_spent(&re::listeners_on_stars_spent());
    re::register_many_after_energy_reset(&re::listeners_after_energy_reset());
    re::register_many_after_death(&re::listeners_after_death());
    re::register_many_modify_hand_draw(&re::listeners_modify_hand_draw());
    re::register_many_modify_x_value(&re::listeners_modify_x_value());
    re::register_many_relic_modify_card_play_count(&re::listeners_relic_modify_card_play_count());
    re::register_many_modify_power_amount_given(&re::listeners_modify_power_amount_given());
    re::register_many_modify_power_amount_received(&re::listeners_modify_power_amount_received());
    re::register_many_before_power_amount_changed(&re::listeners_before_power_amount_changed());
    re::register_many_after_modifying_power_amount_given(
        &re::listeners_after_modifying_power_amount_given(),
    );
    re::register_many_after_modifying_power_amount_received(
        &re::listeners_after_modifying_power_amount_received(),
    );
    re::register_many_count_pets(&re::listeners_count_pets());
    re::register_many_modify_vulnerable_multiplier(&re::listeners_modify_vulnerable_multiplier());
    re::register_many_before_play_phase_start(&re::listeners_before_play_phase_start());
    re::register_many_after_shuffle_bus(&re::listeners_after_shuffle_bus());
}

/// Whole random combats of every character with guarded-channel relics
/// (melted copies and a duplicate included) play step for step the same on
/// the narrowed walks (and the call sites that skip dead guarded channels)
/// as on the full walks. Debug builds also shadow every skipped handler.
#[test]
fn narrowed_combats_match_the_full_walk() {
    install();
    assert!(re::all_channels_frozen());
    let narrowed = all_combats();
    force_full_walks();
    assert!(
        !re::all_channels_frozen(),
        "the guarded channels left their frozen lists"
    );
    assert!(re::listener_guards_after_turn_end()
        .iter()
        .all(|&(_, g)| g == RelicGuard::Always));
    let full = all_combats();
    assert_eq!(narrowed.len(), full.len());
    let mut steps = 0usize;
    for (i, (a, b)) in narrowed.iter().zip(&full).enumerate() {
        assert_eq!(a, b, "combat {i} diverged");
        steps += a.len();
    }
    assert!(steps > 1000, "{steps} steps");
}
