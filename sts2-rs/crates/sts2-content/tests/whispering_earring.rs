//! GH #946: Whispering Earring's turn-1 auto-play.
//!
//! C# `WhisperingEarring.AfterAutoPrePlayPhaseEnteredLate` (identical in
//! v0.107.1 and v0.111.0): on the owner's first turn, under a pushed
//! `VakuuCardSelector`, play up to 13 cards from hand — each time the FIRST
//! hand card whose `CanPlay()` holds, aimed at the leftmost hittable enemy,
//! paid through `SpendResources()` and then `CardCmd.AutoPlay(...,
//! skipXCapture: true)`. Rust had only the relic's +1 max energy.

use smallvec::SmallVec;

use sts2_content::monster_dispatch::{build_ai, MonsterSpawnArgs, SpawnPosition};
use sts2_core::cards::CardId;
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::{
    CardUid, CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId,
    MonsterKind, PlayerCombatState, Relic, RelicKind, StepResult,
};

fn enemy(id: u32, hp: i32, index: u8, alone: bool) -> Creature {
    Creature {
        id: CreatureUid::new(id).expect("non-zero").0,
        current_hp: hp,
        max_hp: hp,
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
                alone,
                front: true,
                back: true,
                index,
            },
        )),
    }
}

fn earring(melted: bool) -> Relic {
    Relic {
        id: std::num::NonZeroU32::new(1).expect("non-zero"),
        kind: RelicKind::WhisperingEarring,
        counter: 0,
        is_wax: false,
        is_melted: melted,
    }
}

struct Setup {
    draw: Vec<CardId>,
    enemy_hps: Vec<i32>,
    relics: Vec<Relic>,
    stars: i8,
    player_powers: Vec<(sts2_core::PowerKind, i32)>,
    version: sts2_core::GameVersion,
    /// The first enemy is a mid-revive corpse: 0 HP with `Adaptable`.
    adaptable_corpse: bool,
}

impl Setup {
    fn new(draw: &[CardId]) -> Self {
        Self {
            draw: draw.to_vec(),
            enemy_hps: vec![400],
            relics: vec![earring(false)],
            stars: 0,
            player_powers: Vec::new(),
            version: sts2_core::GameVersion::default(),
            adaptable_corpse: false,
        }
    }
}

/// Build a combat whose turn-1 draw deals `draw` (top first) and run turn 1's
/// start. Returns the state and the card uids in `draw` order.
fn run_turn_one(setup: Setup) -> (CombatState, Vec<CardUid>, StepResult) {
    run_turn(setup, None)
}

fn run_turn(setup: Setup, round: Option<u16>) -> (CombatState, Vec<CardUid>, StepResult) {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);

    let mut cards = sts2_core::CardStore::new();
    let mut uids = Vec::new();
    for &id in &setup.draw {
        let uid = CardUid::new(u32::try_from(cards.len() + 1).expect("fits")).expect("non-zero");
        let mut card = sts2_content::cards::spec_index::card_pod_from_id(id, false).expect("spec");
        card.uid = uid;
        cards.push(card);
        uids.push(uid);
    }
    let alone = setup.enemy_hps.len() == 1;
    let creatures: Vec<Creature> = setup
        .enemy_hps
        .iter()
        .enumerate()
        .map(|(i, &hp)| {
            enemy(
                u32::try_from(i + 1).expect("fits"),
                hp,
                u8::try_from(i).expect("fits"),
                alone,
            )
        })
        .collect();
    let enemy_uids: SmallVec<[CreatureUid; 8]> =
        creatures.iter().map(|c| CreatureUid(c.id)).collect();
    let player = PlayerCombatState {
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
        relics: setup.relics.into(),
        player,
        enemy_uids,
        game_version: setup.version,
        ..CombatState::default()
    };
    let _ = start_combat(&mut state);
    state.player.hand.clear();
    state.player.discard.clear();
    state.player.exhaust.clear();
    state.player.draw = uids.iter().copied().collect();
    if let Some(r) = round {
        state.round = r;
    }
    state.player.stars = setup.stars;
    let player_uid = state.player.creature_uid.expect("player mirror");
    for &(kind, amount) in &setup.player_powers {
        sts2_core::card_effects::apply_power_to_creature(&mut state, player_uid, kind, amount);
    }
    if setup.adaptable_corpse {
        let first = state.enemy_uids[0];
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            first,
            sts2_core::PowerKind::Adaptable,
            1,
        );
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == first.0) {
            c.current_hp = 0;
        }
    }
    let res = start_player_turn(&mut state);
    (state, uids, res)
}

fn ids(state: &CombatState, pile: &[CardUid]) -> Vec<CardId> {
    pile.iter()
        .map(|u| state.cards[u.0.get() as usize - 1].card_id)
        .collect()
}

fn hp(state: &CombatState, idx: usize) -> i32 {
    let uid = state.enemy_uids[idx];
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("enemy")
        .current_hp
}

const STRIKE: CardId = CardId::StrikeIronclad;
const DEFEND: CardId = CardId::DefendIronclad;

/// Repeatedly plays the FIRST playable card in hand, paying its cost, until
/// none is playable: 4 energy (3 + the Earring) buys Bash (2), Strike (1),
/// Defend (1); the second Strike / Defend are left in hand.
#[test]
fn plays_first_playable_card_repeatedly_and_pays_cost() {
    let (state, _uids, res) =
        run_turn_one(Setup::new(&[CardId::Bash, STRIKE, DEFEND, STRIKE, DEFEND]));
    assert_eq!(res, StepResult::Advanced);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert!(state.pending_choice.is_none());
    assert_eq!(state.player.energy, 0, "every auto-play paid its cost");
    assert_eq!(ids(&state, &state.player.hand), vec![STRIKE, DEFEND]);
    assert_eq!(
        ids(&state, &state.player.discard),
        vec![CardId::Bash, STRIKE, DEFEND],
        "played in hand order"
    );
    // Bash 8, then the Strike's 6 under Bash's Vulnerable = 9.
    assert_eq!(hp(&state, 0), 400 - 8 - 9);
    assert_eq!(state.player.block, 5);
    assert_eq!(state.combat_log.energy_spent_this_turn, 4);
}

/// A card the player cannot afford is skipped, not a stop: the first card
/// whose `CanPlay()` holds is taken, wherever it sits in hand.
#[test]
fn skips_unaffordable_cards_for_the_first_playable_one() {
    let (state, _uids, _) = run_turn_one(Setup::new(&[
        CardId::Bash,
        STRIKE,
        CardId::Bash,
        CardId::Bash,
        DEFEND,
    ]));
    // Bash (2) → 2 left, Strike (1) → 1 left; both remaining Bashes are now
    // unaffordable, so the Defend behind them is played.
    assert_eq!(state.player.energy, 0);
    assert_eq!(
        ids(&state, &state.player.hand),
        vec![CardId::Bash, CardId::Bash]
    );
    assert_eq!(
        ids(&state, &state.player.discard),
        vec![CardId::Bash, STRIKE, DEFEND]
    );
}

/// `GetTarget`: AnyEnemy → `HittableEnemies.FirstOrDefault()`, the leftmost
/// one, with no `CombatTargets` read.
#[test]
fn aims_at_the_leftmost_enemy_without_an_rng_read() {
    let mut setup = Setup::new(&[STRIKE, STRIKE, STRIKE, STRIKE, STRIKE]);
    setup.enemy_hps = vec![100, 100];
    let targets_before = {
        let (s, _, _) = {
            let mut probe = Setup::new(&[DEFEND, DEFEND, DEFEND, DEFEND, DEFEND]);
            probe.enemy_hps = vec![100, 100];
            probe.relics = Vec::new();
            run_turn_one(probe)
        };
        s.run_rng.combat_targets.counter()
    };
    let (state, _, _) = run_turn_one(setup);
    assert_eq!(hp(&state, 0), 100 - 4 * 6, "four Strikes, all on the left");
    assert_eq!(hp(&state, 1), 100);
    assert_eq!(
        state.run_rng.combat_targets.counter(),
        targets_before,
        "the leftmost pick reads no combat_targets"
    );
}

/// An X-cost card spends ALL energy through `SpendResources`, and that is
/// its X (`skipXCapture`): Whirlwind first with 4 energy hits 4 times.
#[test]
fn x_cost_card_captures_the_energy_it_spent() {
    let (state, _, _) = run_turn_one(Setup::new(&[
        CardId::Whirlwind,
        STRIKE,
        DEFEND,
        DEFEND,
        DEFEND,
    ]));
    assert_eq!(state.player.energy, 0);
    assert_eq!(hp(&state, 0), 400 - 4 * 5, "Whirlwind X=4");
    assert_eq!(
        ids(&state, &state.player.hand),
        vec![STRIKE, DEFEND, DEFEND, DEFEND]
    );
}

/// The Earring play's `Resources.EnergyValue` is `GetAmountToSpend()` AFTER
/// `SpendResources()` (CardCmd.cs:126): for an X-cost card the energy left,
/// 0 — not the X. IntimidatingHelmet (EnergyValue >= 2) does not grant on an
/// Earring-played Whirlwind (loadout Ironclad v0.107.1 2009: C# block 7, Rust
/// 11), and IvoryTile (>= 3) gives no energy.
#[test]
fn x_cost_card_energy_value_is_the_energy_left() {
    let mut setup = Setup::new(&[CardId::Whirlwind, DEFEND, DEFEND, DEFEND, DEFEND]);
    for (i, kind) in [RelicKind::IntimidatingHelmet, RelicKind::IvoryTile]
        .into_iter()
        .enumerate()
    {
        setup.relics.push(Relic {
            id: std::num::NonZeroU32::new(u32::try_from(i + 2).expect("fits")).expect("non-zero"),
            kind,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
    }
    let (state, uids, _) = run_turn_one(setup);
    assert_eq!(hp(&state, 0), 400 - 4 * 5, "Whirlwind X=4");
    assert_eq!(state.player.block, 0, "no IntimidatingHelmet block");
    assert_eq!(state.player.energy, 0, "no IvoryTile energy");
    let whirlwind = &state.cards[uids[0].0.get() as usize - 1];
    assert_eq!(whirlwind.energy_spent, 4, "the X stays the spent energy");
    assert_eq!(sts2_core::cards::play_energy_value(whirlwind), 0);
}

/// Stardust under the Earring: `SpendResources()` spends the stars and sets
/// `LastStarsSpent`, then `AutoPlay(skipXCapture: true)` keeps it — 3 stars
/// hit 3 times (loadout Regent v0.111.0 2020: Rust hit 0 times).
#[test]
fn stardust_hits_once_per_star_it_spent() {
    let mut setup = Setup::new(&[CardId::Stardust, DEFEND, DEFEND, DEFEND, DEFEND]);
    setup.stars = 3;
    let targets_before = {
        let mut probe = Setup::new(&[DEFEND, DEFEND, DEFEND, DEFEND, DEFEND]);
        probe.relics = Vec::new();
        run_turn_one(probe).0.run_rng.combat_targets.counter()
    };
    let (state, _, _) = run_turn_one(setup);
    assert_eq!(state.player.stars, 0, "the stars were spent");
    assert_eq!(hp(&state, 0), 400 - 3 * 5, "one 5-damage hit per star");
    assert_eq!(
        state.run_rng.combat_targets.counter(),
        targets_before + 3,
        "each random hit reads combat_targets"
    );
}

/// The loop is capped at 13 plays (`maxCardsToPlay`).
#[test]
fn stops_after_thirteen_plays() {
    // FlashOfSteel costs 0 and draws one, so the hand never runs dry; only
    // the cap stops the loop.
    let (state, _, _) = run_turn_one(Setup::new(&[CardId::FlashOfSteel; 25]));
    assert_eq!(state.player.discard.len(), 13);
    assert_eq!(state.player.hand.len(), 5);
    assert_eq!(state.player.draw.len(), 25 - 5 - 13);
    assert_eq!(state.player.energy, 4);
    assert_eq!(
        hp(&state, 0),
        400 - 13 * 5,
        "thirteen 5-damage FlashOfSteels"
    );
}

/// It stops as soon as the combat is over: the remaining cards stay in hand.
#[test]
fn stops_when_combat_ends() {
    let mut setup = Setup::new(&[STRIKE, STRIKE, STRIKE, STRIKE, STRIKE]);
    setup.enemy_hps = vec![10];
    let (state, _, res) = run_turn_one(setup);
    assert_eq!(res, StepResult::CombatWon);
    assert!(state.is_over);
    assert_eq!(state.player.hand.len(), 3, "two Strikes killed it");
    assert_eq!(state.player.energy, 2);
}

/// Only the first turn: on turn 2 nothing is auto-played.
#[test]
fn turn_two_does_nothing() {
    let (state, _, _) = run_turn(
        Setup::new(&[STRIKE, STRIKE, STRIKE, DEFEND, DEFEND]),
        Some(2),
    );
    assert_eq!(state.player.hand.len(), 5);
    assert_eq!(state.player.energy, 4);
    assert_eq!(hp(&state, 0), 400);
}

/// A melted Earring does nothing at all (no auto-play, no +1 energy).
#[test]
fn melted_earring_does_nothing() {
    let mut setup = Setup::new(&[STRIKE, STRIKE, STRIKE, DEFEND, DEFEND]);
    setup.relics = vec![earring(true)];
    let (state, _, _) = run_turn_one(setup);
    assert_eq!(state.player.hand.len(), 5);
    assert_eq!(state.player.energy, 3);
}

/// A selection opened by an auto-played card is answered by the
/// `VakuuCardSelector` — `options.Take(maxSelect)`, the first max_select
/// options — inline, without surfacing. Purity is "exhaust up to 3"
/// (min 0 / max 3): Vakuu exhausts the first THREE hand cards (the parity
/// harness's min-prefix drain would exhaust one).
#[test]
fn selection_inside_a_played_card_resolves_vakuu_style() {
    let (state, uids, res) = run_turn_one(Setup::new(&[
        CardId::Purity,
        STRIKE,
        DEFEND,
        STRIKE,
        DEFEND,
    ]));
    assert_eq!(res, StepResult::Advanced);
    assert!(state.pending_choice.is_none(), "nothing surfaced");
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    // Purity (exhausts itself) + the first three remaining hand cards.
    let exhausted: Vec<CardUid> = state.player.exhaust.iter().copied().collect();
    for u in &uids[1..4] {
        assert!(exhausted.contains(u), "{u:?} exhausted: {exhausted:?}");
    }
    // The last Defend is then auto-played with the energy left.
    assert!(state.player.hand.is_empty());
    assert_eq!(state.player.block, 5);
    assert_eq!(state.player.energy, 3);
}

/// The Earring pays a Guiding Star's 2 stars in `SpendResources()`, but its
/// `CardCmd.AutoPlay` passes `ResourceInfo.StarsSpent = 0` (CardCmd.cs:126,
/// both builds), so `BlackHolePower.AfterCardPlayed` (`StarsSpent > 0`) stays
/// quiet: only the card's 12 land. Rust queued the hand play's Black Hole AoE
/// as well (loadout Regent 1094 v0.107.1: Mecha Knight cs=200 rs=197).
#[test]
fn earring_star_spend_does_not_fire_black_hole() {
    let mut setup = Setup::new(&[CardId::GuidingStar]);
    setup.stars = 2;
    setup.player_powers = vec![(sts2_core::PowerKind::BlackHole, 3)];
    let (state, _uids, _) = run_turn_one(setup);
    assert_eq!(state.player.stars, 0, "the stars were spent");
    assert_eq!(
        ids(&state, &state.player.discard),
        vec![CardId::GuidingStar],
        "the Earring played Guiding Star"
    );
    assert_eq!(hp(&state, 0), 400 - 12, "Guiding Star only, no Black Hole");
}

/// Not the Earring, but the same turn-1 harness: Mayhem auto-plays the draw
/// top after the hand draw. C# `CardPileCmd.AutoPlayFromDrawPile` runs under
/// a power hook, outside any card or potion effect, so the auto-play's
/// `OnPlayWrapper` ends with `CheckForEmptyHand` (CardModel.cs:1993 v0.107.1,
/// :2006 v0.111.0). A Shadow Step that discards the whole hand lets Unceasing
/// Top draw (loadout Silent 1095 v0.111.0: C# drew Gold Axe, Rust nothing).
#[test]
fn mayhem_auto_play_that_empties_the_hand_triggers_unceasing_top() {
    for version in [
        sts2_core::GameVersion::V0_107_1,
        sts2_core::GameVersion::V0_111_0,
    ] {
        let mut setup = Setup::new(&[
            STRIKE,
            STRIKE,
            STRIKE,
            STRIKE,
            STRIKE,
            CardId::ShadowStep,
            DEFEND,
        ]);
        setup.relics = vec![Relic {
            id: std::num::NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::UnceasingTop,
            counter: 0,
            is_wax: false,
            is_melted: false,
        }];
        setup.player_powers = vec![(sts2_core::PowerKind::Mayhem, 1)];
        setup.version = version;
        let (state, _uids, _) = run_turn_one(setup);
        assert_eq!(
            ids(&state, &state.player.hand),
            vec![DEFEND],
            "{version:?}: Unceasing Top draws after Shadow Step empties the hand"
        );
    }
}

/// `CardModel.CanPlay()` asks for no target, so with the only enemy a
/// mid-revive corpse the Earring still takes the Strike first: it pays 1 and
/// `CardCmd.AutoPlay` (no hittable enemy to roll) moves it to the discard
/// pile unplayed, then the Defend plays (loadout Silent 1104 v0.107.1: Rust
/// skipped the attacks and kept their energy).
#[test]
fn pays_for_an_enemy_card_with_no_living_enemy() {
    let mut setup = Setup::new(&[STRIKE, DEFEND]);
    setup.adaptable_corpse = true;
    let (state, _uids, _) = run_turn_one(setup);
    assert!(state.player.hand.is_empty(), "both cards left the hand");
    assert_eq!(ids(&state, &state.player.discard), vec![STRIKE, DEFEND]);
    assert_eq!(state.player.energy, 2, "4 - Strike 1 - Defend 1");
    assert_eq!(state.player.block, 5);
}

/// Not the Earring, same harness: Ninja Scroll's turn-1 Shivs are real Shivs
/// (`Shiv.CreateInHand`, NinjaScroll.cs:28). A hand-built card lacked the
/// `Shiv` tag, so Accuracy did not add to them (loadout Silent 1157 v0.107.1:
/// a Ninja Scroll Shiv under Accuracy 4 hit for 8 in C#, 4 in Rust).
#[test]
fn ninja_scroll_shivs_carry_the_shiv_tag() {
    let mut setup = Setup::new(&[DEFEND, DEFEND, DEFEND, DEFEND, DEFEND]);
    setup.relics = vec![Relic {
        id: std::num::NonZeroU32::new(1).expect("non-zero"),
        kind: RelicKind::NinjaScroll,
        counter: 0,
        is_wax: false,
        is_melted: false,
    }];
    let (state, _uids, _) = run_turn_one(setup);
    let shivs: Vec<_> = state
        .player
        .hand
        .iter()
        .map(|u| &state.cards[u.0.get() as usize - 1])
        .filter(|c| c.card_id == CardId::Shiv)
        .collect();
    assert_eq!(shivs.len(), 3);
    for c in shivs {
        assert!(c.tags.contains(sts2_core::cards::CardTag::Shiv));
    }
}
