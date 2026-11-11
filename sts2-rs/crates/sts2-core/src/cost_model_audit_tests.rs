//! Differential audit of the engine's card energy-cost model
//! ([`crate::CardEnergyCost`]) against a deliberately naive port of C#
//! `CardEnergyCost` (cluster C parity audit).
//!
//! C# keeps a base cost plus an ORDERED list of `LocalCostModifier`s
//! (`CardEnergyCost.cs` / `LocalCostModifier.cs`, identical in v0.107.1 and
//! v0.111.0): `GetWithModifiers(Local)` folds them in insertion order with no
//! intermediate clamp, `EndOfTurnCleanup` / `AfterCardPlayedCleanup` drop the
//! modifiers carrying that expiry bit, and `UpgradeBy` moves the base and
//! clamps Absolute amounts above a LOWERED base.
//!
//! [`CsCost`] is that C# model with an unbounded, never-compacted list; every
//! op below drives the REAL engine entry point a card / relic / potion uses
//! and the matching C# call, then compares the local cost
//! ([`crate::cards::Card::local_cost`], C# `GetWithModifiers(Local)`). The
//! enumeration walks every op sequence up to a depth and reports the
//! shortest diverging ones, so it also checks that the engine's list
//! compaction never changes a value.

use crate::cards::{Card, CardId, CardType};
use crate::{CardUid, CombatState};

#[derive(Clone, Copy, Debug)]
struct Mod {
    amount: i32,
    absolute: bool,
    end_of_turn: bool,
    when_played: bool,
    reduce_only: bool,
}

impl Mod {
    /// `LocalCostModifier.Modify`.
    fn modify(self, cur: i32) -> i32 {
        match (self.absolute, self.reduce_only) {
            (true, true) => cur.min(self.amount),
            (true, false) => self.amount,
            (false, true) => cur.min(cur + self.amount),
            (false, false) => cur + self.amount,
        }
    }
}

/// C# `CardEnergyCost` for a non-X, non-canonical card.
#[derive(Clone, Debug)]
struct CsCost {
    canonical: i32,
    base: i32,
    mods: Vec<Mod>,
}

impl CsCost {
    fn new(canonical: i32) -> Self {
        Self {
            canonical,
            base: canonical,
            mods: Vec::new(),
        }
    }

    /// `GetWithModifiers(CostModifiers.Local)` before its final clamp.
    fn local(&self) -> i32 {
        if self.base < 0 {
            return self.base;
        }
        self.mods.iter().fold(self.base, |c, m| m.modify(c))
    }

    /// C# `Set…(amount, reduceOnly)`: an Absolute local modifier.
    fn set(&mut self, amount: i32, expiry: Expiry, reduce_only: bool) {
        self.mods.push(Mod {
            amount,
            absolute: true,
            end_of_turn: expiry.end_of_turn(),
            when_played: expiry.when_played(),
            reduce_only,
        });
    }

    /// C# `Add…(amount)`: a Relative local modifier (skipped when 0).
    fn add(&mut self, amount: i32, expiry: Expiry) {
        if amount == 0 {
            return;
        }
        self.mods.push(Mod {
            amount,
            absolute: false,
            end_of_turn: expiry.end_of_turn(),
            when_played: expiry.when_played(),
            reduce_only: false,
        });
    }

    /// `UpgradeBy` (CardEnergyCost.cs:351-370).
    fn upgrade_by(&mut self, addend: i32) {
        if addend == 0 {
            return;
        }
        let old = self.base;
        let num = (self.base + addend).max(0);
        if num < old {
            for m in &mut self.mods {
                if m.absolute && m.amount > num {
                    m.amount = num;
                }
            }
        }
        self.base = num;
    }

    fn end_of_turn(&mut self) {
        self.mods.retain(|m| !m.end_of_turn);
    }

    fn played(&mut self) {
        self.mods.retain(|m| !m.when_played);
    }
}

/// C# `LocalCostModifierExpiration` combinations the setters use.
#[derive(Clone, Copy, Debug)]
enum Expiry {
    /// `EndOfTurn | WhenPlayed` (`…ThisTurnOrUntilPlayed`).
    TurnOrPlayed,
    /// `EndOfTurn` (`…ThisTurn`).
    Turn,
    /// `WhenPlayed` (`…UntilPlayed`).
    Played,
    /// `EndOfCombat` (`…ThisCombat`).
    Combat,
}

impl Expiry {
    fn end_of_turn(self) -> bool {
        matches!(self, Self::TurnOrPlayed | Self::Turn)
    }

    fn when_played(self) -> bool {
        matches!(self, Self::TurnOrPlayed | Self::Played)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// `SetToFreeThisTurn` (Bullet Time, Infernal Blade, Discovery, …).
    FreeThisTurn,
    /// `SetUntilPlayed(0)` (v0.107.1 Rocket Punch).
    FreeUntilPlayed,
    /// `SetToFreeThisCombat` (Touch of Insanity, Metamorphosis).
    FreeThisCombat,
    /// Confused / Snecko Eye's per-draw `SetThisCombat(roll)`.
    Confused(i8),
    /// Snecko Oil's `SetThisTurnOrUntilPlayed(roll)`.
    Snecko(i8),
    /// Transfigure's `AddThisCombat(1)`.
    Transfigure,
    /// Enlightenment: `SetThisTurnOrUntilPlayed(1, reduceOnly)`.
    EnlightenmentTurn,
    /// Enlightenment+: `SetThisCombat(1, reduceOnly)`.
    EnlightenmentCombat,
    /// Bookmark / Slumbering Essence: `AddUntilPlayed(-1)`.
    Bookmark,
    Upgrade,
    Downgrade,
    /// `AfterCardPlayedCleanup` of this card.
    Played,
    /// `EndOfTurnCleanup`.
    EndTurn,
    /// Stomp: another Attack played (`AddThisTurn(-1)`).
    AttackPlayed,
    /// Pinpoint: a Skill played (`AddThisTurn(-1)`).
    SkillPlayed,
    /// Banshee's Cry: an Ethereal card played (`AddThisCombat(-2)`).
    EtherealPlayed,
    /// Melancholy: a creature died (`AddThisCombat(-1)`).
    Death,
    /// Midnight: a card exhausted (`AddThisCombat(-1)`).
    Exhaust,
    /// Kingly Kick: drawn (`AddThisCombat(-1)`).
    Drawn,
}

#[derive(Clone, Copy, Debug)]
struct Subject {
    id: CardId,
    card_type: CardType,
    canonical: i8,
    upgrade_delta: i8,
}

fn factory(subject: Subject) -> impl Fn(CardId, bool) -> Option<Card> {
    move |id, up| {
        Some(Card {
            card_id: id,
            card_type: subject.card_type,
            cost: crate::CardEnergyCost::new(if up {
                (subject.canonical + subject.upgrade_delta).max(0)
            } else {
                subject.canonical
            }),
            ..Card::pod_defaults()
        })
    }
}

/// A `combat_energy_costs` seed whose first `next_int(0, 3)` is `roll`.
fn seed_for_roll(roll: i8) -> crate::dotnet_rng::Rng {
    for s in 0..10_000 {
        let mut r = crate::dotnet_rng::Rng::new(s);
        if r.next_int(0, 3) == i32::from(roll) {
            return crate::dotnet_rng::Rng::new(s);
        }
    }
    unreachable!("some seed rolls {roll}")
}

fn ops_for(subject: Subject) -> Vec<Op> {
    let mut ops = vec![
        Op::FreeThisTurn,
        Op::FreeThisCombat,
        Op::Confused(0),
        Op::Confused(2),
        Op::Snecko(0),
        Op::Snecko(3),
        Op::Transfigure,
        Op::EnlightenmentTurn,
        Op::EnlightenmentCombat,
        Op::Bookmark,
        Op::Upgrade,
        Op::Downgrade,
        Op::Played,
        Op::EndTurn,
    ];
    match subject.id {
        CardId::Stomp => ops.push(Op::AttackPlayed),
        CardId::Pinpoint => ops.push(Op::SkillPlayed),
        CardId::BansheesCry => ops.push(Op::EtherealPlayed),
        CardId::Melancholy => ops.push(Op::Death),
        CardId::Midnight => ops.push(Op::Exhaust),
        CardId::KinglyKick => ops.push(Op::Drawn),
        // `SetUntilPlayed(0)` has one caller in either build: v0.107.1
        // Rocket Punch on itself (RocketPunch.cs:49).
        CardId::RocketPunch => ops.push(Op::FreeUntilPlayed),
        _ => {}
    }
    ops
}

struct World {
    state: CombatState,
    cs: CsCost,
}

const UID: u32 = 1;

fn uid() -> CardUid {
    CardUid::new(UID).expect("nz")
}

fn new_world(subject: Subject) -> World {
    let mut state = CombatState {
        game_version: crate::GameVersion::V0_111_0,
        ..CombatState::default()
    };
    let card = Card {
        uid: uid(),
        card_id: subject.id,
        card_type: subject.card_type,
        cost: crate::CardEnergyCost::new(subject.canonical),
        ..Card::pod_defaults()
    };
    state.cards.push(card);
    state.player.hand.push(uid());
    World {
        state,
        cs: CsCost::new(i32::from(subject.canonical)),
    }
}

/// Apply `op`; `false` when it does not apply in this state (skip it).
fn apply(world: &mut World, subject: Subject, op: Op) -> bool {
    let World { state, cs } = world;
    match op {
        Op::FreeThisTurn => {
            state.cards[0].set_to_free_this_turn();
            cs.set(0, Expiry::TurnOrPlayed, false);
        }
        Op::FreeUntilPlayed => {
            state.cards[0].cost.set_until_played(0, false);
            cs.set(0, Expiry::Played, false);
        }
        Op::FreeThisCombat => {
            state.cards[0].set_to_free_this_combat();
            cs.set(0, Expiry::Combat, false);
        }
        Op::Confused(r) => {
            // The draw path (`lib.rs` Confused `AfterCardDrawn`).
            state.cards[0].cost.set_this_combat(i32::from(r), false);
            cs.set(i32::from(r), Expiry::Combat, false);
        }
        Op::Snecko(r) => {
            state.run_rng.combat_energy_costs = seed_for_roll(r);
            state.snecko_randomize_hand_costs();
            cs.set(i32::from(r), Expiry::TurnOrPlayed, false);
        }
        Op::Transfigure => {
            crate::apply_transfigure(state, uid());
            cs.add(1, Expiry::Combat);
        }
        Op::EnlightenmentTurn => {
            state.cards[0].cost.set_this_turn_or_until_played(1, true);
            cs.set(1, Expiry::TurnOrPlayed, true);
        }
        Op::EnlightenmentCombat => {
            state.cards[0].cost.set_this_combat(1, true);
            cs.set(1, Expiry::Combat, true);
        }
        Op::Bookmark => {
            // Bookmark's eligibility: `GetWithModifiers(Local) > 0`.
            if cs.local() <= 0 {
                return false;
            }
            state.add_card_cost_reduction_until_played(uid(), 1);
            cs.add(-1, Expiry::Played);
        }
        Op::Upgrade => {
            if state.cards[0].upgraded {
                return false;
            }
            state.cards[0].upgrade_with(factory(subject));
            cs.upgrade_by(i32::from(subject.upgrade_delta));
        }
        Op::Downgrade => {
            if !state.cards[0].upgraded {
                return false;
            }
            state.cards[0].downgrade_with(factory(subject));
            cs.base = cs.canonical;
        }
        Op::Played => {
            crate::combat_loop::clear_when_played_costs(state, 0);
            cs.played();
        }
        Op::EndTurn => {
            crate::combat_loop::clear_per_turn_card_vars(state);
            cs.end_of_turn();
        }
        Op::AttackPlayed => {
            state.discount_stomp_on_attack_played();
            cs.add(-1, Expiry::Turn);
        }
        Op::SkillPlayed => {
            state.discount_pinpoint_on_skill_played();
            cs.add(-1, Expiry::Turn);
        }
        Op::EtherealPlayed => {
            state.combat_log.ethereal_cards_played_this_combat += 1;
            state.banshees_cry_after_card_played(true);
            cs.add(-2, Expiry::Combat);
        }
        Op::Death => {
            state.reduce_melancholy_cost_on_death();
            cs.add(-1, Expiry::Combat);
        }
        Op::Exhaust => {
            crate::hooks::reduce_midnight_costs_v0111(state);
            cs.add(-1, Expiry::Combat);
        }
        Op::Drawn => {
            let before = state.player.hand.len() - 1;
            state.invoke_card_drawn_for_newly_drawn(before, false);
            cs.add(-1, Expiry::Combat);
        }
    }
    true
}

fn rust_local(state: &CombatState) -> i32 {
    state.cards[0].local_cost()
}

/// Walk every op sequence up to `depth`; return the shortest diverging ones
/// (a diverging prefix is not extended).
fn diverging_sequences(subject: Subject, depth: usize) -> Vec<(Vec<Op>, i32, i32)> {
    let ops = ops_for(subject);
    let mut out = Vec::new();
    let mut stack: Vec<(World, Vec<Op>)> = vec![(new_world(subject), Vec::new())];
    while let Some((world, seq)) = stack.pop() {
        if seq.len() == depth {
            continue;
        }
        for &op in &ops {
            let mut next = World {
                state: world.state.clone(),
                cs: world.cs.clone(),
            };
            if !apply(&mut next, subject, op) {
                continue;
            }
            let mut s = seq.clone();
            s.push(op);
            let (rs, cs) = (rust_local(&next.state), next.cs.local());
            if rs == cs {
                stack.push((next, s));
            } else {
                out.push((s, cs, rs));
            }
        }
    }
    out.sort_by_key(|(s, _, _)| s.len());
    out
}

const SUBJECTS: [Subject; 9] = [
    Subject {
        id: CardId::KinglyKick,
        card_type: CardType::Attack,
        canonical: 4,
        upgrade_delta: 0,
    },
    Subject {
        id: CardId::RocketPunch,
        card_type: CardType::Attack,
        canonical: 2,
        upgrade_delta: 0,
    },
    Subject {
        id: CardId::Bludgeon,
        card_type: CardType::Attack,
        canonical: 3,
        upgrade_delta: -1,
    },
    Subject {
        id: CardId::StrikeIronclad,
        card_type: CardType::Attack,
        canonical: 1,
        upgrade_delta: -1,
    },
    Subject {
        id: CardId::Stomp,
        card_type: CardType::Attack,
        canonical: 3,
        upgrade_delta: 0,
    },
    Subject {
        id: CardId::Pinpoint,
        card_type: CardType::Attack,
        canonical: 3,
        upgrade_delta: 0,
    },
    Subject {
        id: CardId::BansheesCry,
        card_type: CardType::Attack,
        canonical: 9,
        upgrade_delta: -2,
    },
    Subject {
        id: CardId::Melancholy,
        card_type: CardType::Skill,
        canonical: 3,
        upgrade_delta: 0,
    },
    Subject {
        id: CardId::Midnight,
        card_type: CardType::Attack,
        canonical: 12,
        upgrade_delta: 0,
    },
];

/// Exploration entry point: prints every shortest diverging sequence.
#[test]
#[ignore = "exploration: run with --run-ignored to list divergences"]
fn explore_cost_model_divergences() {
    let depth: usize = std::env::var("COST_AUDIT_DEPTH")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(3);
    let mut total = 0;
    for subject in SUBJECTS {
        let found = diverging_sequences(subject, depth);
        total += found.len();
        for (seq, cs, rs) in found.iter().take(60) {
            println!("{:?}: {seq:?} -> C# {cs}, Rust {rs}", subject.id);
        }
        println!("{:?}: {} diverging sequences", subject.id, found.len());
    }
    println!("total {total}");
}

/// `ops` (each must apply) from a fresh card; returns the world.
fn run(subject: Subject, ops: &[Op]) -> World {
    let mut world = new_world(subject);
    for &op in ops {
        assert!(apply(&mut world, subject, op), "{op:?} applies");
    }
    world
}

/// Run `ops` and assert the engine's local cost is C#'s `expected`.
fn assert_local(subject: Subject, ops: &[Op], expected: i32) -> World {
    let world = run(subject, ops);
    assert_eq!(world.cs.local(), expected, "C# model for {ops:?}");
    assert_eq!(
        rust_local(&world.state),
        expected,
        "{:?} {ops:?}: GetWithModifiers(Local)",
        subject.id
    );
    world
}

fn subject(id: CardId) -> Subject {
    *SUBJECTS.iter().find(|s| s.id == id).expect("subject")
}

fn push_player_power(state: &mut CombatState, raw: u32, kind: crate::PowerKind, amount: i32) {
    let id = std::num::NonZeroU32::new(raw).expect("nz");
    state.powers.push(crate::Power {
        id,
        kind,
        amount,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(crate::PowerUid(id));
}

/// An `AddThisCombat` / `AddUntilPlayed` appended while a local zero is live
/// folds after it (`CardEnergyCost.cs:110-115`): a free-this-turn Melancholy
/// shaved by a death is at -1, so Borrowed Time's +1 (skipped on a negative
/// cost, Hook.cs:1729-1732) leaves it free. The engine wrote the shave only
/// into the base, so the card read 0 and Borrowed Time made it cost 1. Same
/// for Midnight's exhaust shave, Kingly Kick's draw shave and Bookmark's
/// until-played cut on a Transfigured free card.
#[test]
fn relative_cost_mods_after_a_local_zero_fold_after_it() {
    let mut world = assert_local(
        subject(CardId::Melancholy),
        &[Op::FreeThisTurn, Op::Death],
        -1,
    );
    push_player_power(&mut world.state, 1, crate::PowerKind::BorrowedTime, 1);
    assert_eq!(crate::hooks::effective_card_cost(&world.state, uid()), 0);
    // Once the zero expires the shave stays on the base: 3 - 1.
    let _ = assert_local(
        subject(CardId::Melancholy),
        &[Op::FreeThisTurn, Op::Death, Op::EndTurn],
        2,
    );
    let _ = assert_local(
        subject(CardId::Midnight),
        &[Op::FreeThisTurn, Op::Exhaust],
        -1,
    );
    let _ = assert_local(
        subject(CardId::KinglyKick),
        &[Op::FreeThisTurn, Op::Drawn],
        -1,
    );
    let _ = assert_local(
        subject(CardId::Bludgeon),
        &[Op::FreeThisTurn, Op::Transfigure, Op::Bookmark],
        0,
    );
}

/// The Midnight half of the entered-combat catch-up (the sibling of Banshee's
/// Cry's): a generated free-this-turn Midnight (`SetToFreeThisTurn` runs
/// before `AddGeneratedCardToCombat`) entering after 2 exhausts reads
/// `0 - 2` (Midnight.cs:37-61), so a surcharge leaves it free.
#[test]
fn free_midnight_entering_combat_shaves_after_the_zero() {
    let mut world = new_world(subject(CardId::Midnight));
    world.state.combat_log.cards_exhausted_this_combat = 2;
    world.state.cards[0].set_to_free_this_turn();
    crate::card_effects::stamp_entered_combat_cost_backfill(&mut world.state, uid());
    assert_eq!(rust_local(&world.state), -2);
    push_player_power(&mut world.state, 1, crate::PowerKind::BorrowedTime, 1);
    assert_eq!(crate::hooks::effective_card_cost(&world.state, uid()), 0);
    world.state.cards[0].cost.after_card_played_cleanup();
    assert_eq!(
        world.state.cards[0].local_cost(),
        10,
        "played: the -2 stays"
    );
}

/// A `SetUntilPlayed(0)` (v0.107.1 Rocket Punch) is `WhenPlayed` only, so it
/// outlives `EndOfTurnCleanup` and so does the Transfigure `+1` appended
/// after it: `0 + 1` next turn, not 0.
#[test]
fn adds_after_a_free_until_played_zero_survive_turn_end() {
    let _ = assert_local(
        subject(CardId::RocketPunch),
        &[Op::FreeUntilPlayed, Op::Transfigure, Op::EndTurn],
        1,
    );
    let _ = assert_local(
        subject(CardId::RocketPunch),
        &[
            Op::FreeUntilPlayed,
            Op::Transfigure,
            Op::EndTurn,
            Op::Played,
        ],
        3,
    );
}

/// C# `ResetForDowngrade` (CardEnergyCost.cs:387-392) moves only `_base`; an
/// Absolute modifier keeps owning the cost. Dampen downgrading a Confused
/// card rolled to 2 left it at 2 in C#, 3 here; a Snecko roll's expiry then
/// reveals the canonical base.
#[test]
fn downgrade_keeps_an_absolute_cost() {
    let strike = subject(CardId::StrikeIronclad);
    let _ = assert_local(strike, &[Op::Upgrade, Op::Confused(2), Op::Downgrade], 2);
    let _ = assert_local(strike, &[Op::Upgrade, Op::FreeThisCombat, Op::Downgrade], 0);
    let banshee = subject(CardId::BansheesCry);
    let _ = assert_local(banshee, &[Op::Upgrade, Op::Snecko(0), Op::Downgrade], 0);
    let _ = assert_local(
        banshee,
        &[Op::Upgrade, Op::Snecko(0), Op::Downgrade, Op::Played],
        9,
    );
    let _ = assert_local(
        banshee,
        &[Op::Upgrade, Op::EnlightenmentCombat, Op::Downgrade],
        1,
    );
}

/// `UpgradeBy` moves `_base` UNDER a live Snecko roll and keeps the Relative
/// modifiers folded before it: a Banshee's Cry shaved to 7 by an Ethereal
/// play, rolled to 0 and upgraded reveals `7 - 2 = 5` once the roll drops,
/// not the bare upgraded base 7.
#[test]
fn upgrade_under_a_snecko_roll_keeps_earlier_relative_mods() {
    let banshee = subject(CardId::BansheesCry);
    for roll in [0, 3] {
        let _ = assert_local(
            banshee,
            &[Op::EtherealPlayed, Op::Snecko(roll), Op::Upgrade],
            roll.into(),
        );
        let _ = assert_local(
            banshee,
            &[
                Op::EtherealPlayed,
                Op::Snecko(roll),
                Op::Upgrade,
                Op::Played,
            ],
            5,
        );
        let _ = assert_local(
            banshee,
            &[
                Op::EtherealPlayed,
                Op::Snecko(roll),
                Op::Upgrade,
                Op::EndTurn,
            ],
            5,
        );
    }
}

/// A Relative shave that lands while a roll / cap holds the card at 0 (Stomp's
/// per-Attack `AddThisTurn(-1)` on a Snecko 0, Banshee's Cry's `-2` under
/// Enlightenment's 1) becomes a below-zero debt. When the roll drops, the
/// revealed base must absorb it, or a later reduce-only cap (Enlightenment+,
/// `min(current, 1)`) reads the undebted base and the debt comes off the
/// capped value: C# `min(3 - 1, 1) = 1`, engine 0.
#[test]
fn roll_expiry_absorbs_debts_taken_under_it() {
    let stomp = subject(CardId::Stomp);
    let ops = [Op::Snecko(0), Op::AttackPlayed, Op::Played];
    let _ = assert_local(stomp, &ops, 2);
    let _ = assert_local(stomp, &[&ops[..], &[Op::EnlightenmentCombat]].concat(), 1);
    let _ = assert_local(
        stomp,
        &[&ops[..], &[Op::EnlightenmentCombat, Op::EndTurn]].concat(),
        1,
    );
    // Without the cap the per-turn shave still comes back next turn.
    let _ = assert_local(stomp, &[&ops[..], &[Op::EndTurn]].concat(), 3);
    let _ = assert_local(
        subject(CardId::Pinpoint),
        &[
            Op::Snecko(0),
            Op::SkillPlayed,
            Op::Played,
            Op::EnlightenmentCombat,
        ],
        1,
    );
    let _ = assert_local(
        subject(CardId::BansheesCry),
        &[
            Op::EnlightenmentTurn,
            Op::EtherealPlayed,
            Op::Played,
            Op::EnlightenmentCombat,
        ],
        1,
    );
    let _ = assert_local(
        subject(CardId::Melancholy),
        &[
            Op::Snecko(0),
            Op::Death,
            Op::Played,
            Op::EnlightenmentCombat,
        ],
        1,
    );
}

/// C# threads one running cost through the early `TryModifyEnergyCostInCombat`
/// pass (Hook.cs:1727-1743); Curious reads it (`originalCost <= 0` → skip)
/// and floors its own result (CuriousPower.cs:12-33). So the player's power
/// ORDER matters: Borrowed Time then Curious on a 0-cost Power is
/// `0 + 1 - 1 = 0`; Curious 2 then Borrowed Time on a 1-cost Power is
/// `max(0, 1 - 2) + 1 = 1`. The engine summed the deltas against the local
/// cost (1 and 0).
#[test]
fn curious_reads_the_running_cost_of_the_early_cost_pass() {
    let power_card = |cost| {
        let mut world = new_world(Subject {
            id: CardId::Inflame,
            card_type: CardType::Power,
            canonical: cost,
            upgrade_delta: 0,
        });
        world.state.cards[0].card_type = CardType::Power;
        world
    };
    let mut world = power_card(0);
    push_player_power(&mut world.state, 1, crate::PowerKind::BorrowedTime, 1);
    push_player_power(&mut world.state, 2, crate::PowerKind::Curious, 1);
    assert_eq!(crate::hooks::effective_card_cost(&world.state, uid()), 0);

    let mut world = power_card(1);
    push_player_power(&mut world.state, 1, crate::PowerKind::Curious, 2);
    push_player_power(&mut world.state, 2, crate::PowerKind::BorrowedTime, 1);
    assert_eq!(crate::hooks::effective_card_cost(&world.state, uid()), 1);

    // Curious first on a 0-cost Power: skipped, then +1.
    let mut world = power_card(0);
    push_player_power(&mut world.state, 1, crate::PowerKind::Curious, 1);
    push_player_power(&mut world.state, 2, crate::PowerKind::BorrowedTime, 1);
    assert_eq!(crate::hooks::effective_card_cost(&world.state, uid()), 1);
}

/// The three orderings the old collapsed model (`Card::cost` plus cost
/// `VarKey`s) could not represent, now pinned against the C# model.
///
/// 1. A Relative modifier appended AFTER an Absolute, then a base-lowering
///    upgrade: C# clamps only the Absolute's amount and keeps the Relative
///    on top.
#[test]
fn relative_after_an_absolute_survives_a_base_lowering_upgrade() {
    let strike = subject(CardId::StrikeIronclad);
    // Confused 2, Transfigure +1, upgrade 1 -> 0: [=min(2,0)=0, +1] = 1.
    let _ = assert_local(strike, &[Op::Confused(2), Op::Transfigure, Op::Upgrade], 1);
    let _ = assert_local(strike, &[Op::Snecko(3), Op::Transfigure, Op::Upgrade], 1);
    let _ = assert_local(
        strike,
        &[Op::FreeThisCombat, Op::Transfigure, Op::Upgrade],
        1,
    );
    let bludgeon = subject(CardId::Bludgeon);
    // Confused 2 under a 3 -> 2 upgrade is not clamped: 2 + 1.
    let _ = assert_local(
        bludgeon,
        &[Op::Confused(2), Op::Transfigure, Op::Upgrade],
        3,
    );
    // Snecko 3 clamps to the new base 2, plus 1; the turn end reveals 2 + 1.
    let _ = assert_local(bludgeon, &[Op::Snecko(3), Op::Transfigure, Op::Upgrade], 3);
    let _ = assert_local(
        bludgeon,
        &[Op::Snecko(3), Op::Transfigure, Op::Upgrade, Op::EndTurn],
        3,
    );
}

/// 2. An until-played cut folded BEFORE a Snecko roll, then a base-lowering
///    upgrade that clamps the roll: the cut stays under the roll, and comes
///    back only if the roll expires without a play.
#[test]
fn until_played_cut_before_a_roll_stays_under_it_through_an_upgrade() {
    let bludgeon = subject(CardId::Bludgeon);
    // 3, -1 (played), roll 3 -> clamped to 2 by the upgrade: 2.
    let _ = assert_local(bludgeon, &[Op::Bookmark, Op::Snecko(3), Op::Upgrade], 2);
    // Turn end drops the roll, not the cut: 2 - 1.
    let _ = assert_local(
        bludgeon,
        &[Op::Bookmark, Op::Snecko(3), Op::Upgrade, Op::EndTurn],
        1,
    );
    // A play drops both.
    let _ = assert_local(
        bludgeon,
        &[Op::Bookmark, Op::Snecko(3), Op::Upgrade, Op::Played],
        2,
    );
    let strike = subject(CardId::StrikeIronclad);
    let _ = assert_local(
        strike,
        &[Op::Bookmark, Op::Snecko(3), Op::Upgrade, Op::EndTurn],
        -1,
    );
}

/// 3. An Enlightenment cap that does not lower the card is still a
///    modifier: `UpgradeBy` clamps its amount, and a later downgrade
///    (canonical base back) meets the clamped cap.
#[test]
fn an_unlowering_enlightenment_cap_is_clamped_by_upgrade_and_kept_by_downgrade() {
    let strike = subject(CardId::StrikeIronclad);
    // Cap 1 on a 1-cost Strike changes nothing; the upgrade to 0 clamps it
    // to 0; the downgrade restores base 1 under the cap 0: min(1, 0) = 0.
    let _ = assert_local(strike, &[Op::EnlightenmentCombat], 1);
    let _ = assert_local(strike, &[Op::EnlightenmentCombat, Op::Upgrade], 0);
    let _ = assert_local(
        strike,
        &[Op::EnlightenmentCombat, Op::Upgrade, Op::Downgrade],
        0,
    );
    let _ = assert_local(
        strike,
        &[
            Op::EnlightenmentTurn,
            Op::Upgrade,
            Op::Downgrade,
            Op::EndTurn,
        ],
        1,
    );
    let banshee = subject(CardId::BansheesCry);
    let _ = assert_local(
        banshee,
        &[Op::EnlightenmentCombat, Op::Upgrade, Op::Downgrade],
        1,
    );
}

fn assert_no_divergence(depth: usize) {
    let mut unexpected = Vec::new();
    for subject in SUBJECTS {
        for (seq, cs, rs) in diverging_sequences(subject, depth) {
            unexpected.push(format!("{:?}: {seq:?} -> C# {cs}, Rust {rs}", subject.id));
        }
    }
    assert!(unexpected.is_empty(), "{}", unexpected.join("\n"));
}

/// Every op sequence up to depth 3, on every subject, matches the C# model.
#[test]
fn cost_model_matches_csharp_to_depth_3() {
    assert_no_divergence(3);
}

/// Depth 4: the same, one op deeper (about 0.5 s at the test profile).
#[test]
fn cost_model_matches_csharp_to_depth_4() {
    assert_no_divergence(4);
}
