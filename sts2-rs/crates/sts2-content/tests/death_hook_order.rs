//! Order of the `Hook.AfterDeath` listener walk on an enemy kill.
//!
//! C# `CreatureCmd.KillWithoutCheckingWinCondition` awaits `Hook.AfterDeath`
//! (v0.111.0 `CreatureCmd.cs:552`, v0.107.1 `:519`), one sequential walk over
//! `CombatState.IterateHookListeners` (`Hook.cs:606-619`) in creature order:
//! allies first (player powers, relics, ...), then the enemies (their powers,
//! then the monster) (`CombatState.cs:115-170`). So a player relic's
//! `AfterDeath` and everything it awaits inline runs before the dying
//! enemy's own power reacts.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, PowerKind,
};

const BOTH: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn fresh(version: GameVersion) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(CharacterId::Ironclad);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn new_card(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn add_relic(state: &mut CombatState, kind: sts2_core::RelicKind) {
    let raw = u32::try_from(state.relics.len() + 1).expect("u32");
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(raw).expect("nz"),
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

/// A Strike kills the only Axebot. Gremlin Horn's `AfterDeath` draws a Strike
/// (`GremlinHorn.cs:24-31`), which Hellraiser auto-plays inline. The Axebot's
/// own `StockPower.AfterDeath` (`StockPower.cs:17-28`) has not spawned the
/// replacement yet, so `HittableEnemies` is empty: `Rng.NextItem` returns
/// null without rolling and the Strike hits nothing (`CardCmd.cs:74-84`).
/// The replacement then spawns untouched. act23 Ironclad:2169 step 864.
#[test]
fn gremlin_horn_draw_resolves_before_stock_respawns_the_axebot() {
    for v in BOTH {
        let mut state = fresh(v);
        let id = NonZeroU32::new(2).expect("nz");
        state.creatures.push(Creature {
            id,
            current_hp: 6,
            max_hp: 40,
            block: 0,
            powers: SmallVec::new(),
            species: Some(MonsterKind::Axebot),
            intent: None,
            intent_history: IntentHistory::default(),
            monster_id: MonsterId("Axebot"),
            ai: None,
        });
        let bot = CreatureUid(id);
        state.enemy_uids.push(bot);
        sts2_core::card_effects::apply_power_to_creature(&mut state, bot, PowerKind::Stock, 2);
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Hellraiser, 1);
        let drawn = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(drawn);
        let strike = new_card(&mut state, CardId::StrikeIronclad);
        state.player.hand.push(strike);
        let targets_before = state.run_rng.combat_targets.counter();

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

        let spawned: Vec<&Creature> = state
            .creatures
            .iter()
            .filter(|c| c.id != id && state.enemy_uids.contains(&CreatureUid(c.id)))
            .collect();
        assert_eq!(spawned.len(), 1, "{v:?}: Stock respawned one Axebot");
        assert_eq!(
            spawned[0].current_hp, spawned[0].max_hp,
            "{v:?}: the auto-played Strike found no target"
        );
        assert!(
            !state.player.draw.contains(&drawn) && !state.player.hand.contains(&drawn),
            "{v:?}: Gremlin Horn drew the Strike and Hellraiser moved it on"
        );
        assert_eq!(
            state.run_rng.combat_targets.counter(),
            targets_before,
            "{v:?}: NextItem over no enemies does not roll"
        );
    }
}

fn enemy(id: u32, hp: i32, kind: MonsterKind, name: &'static str) -> Creature {
    Creature {
        id: NonZeroU32::new(id).expect("nz"),
        current_hp: hp,
        max_hp: 100,
        block: 0,
        powers: SmallVec::new(),
        species: Some(kind),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId(name),
        ai: None,
    }
}

/// A Strike kills Rocket while Crusher, its Kaiser Crab sibling, still holds
/// `CrabRagePower`. The player's listeners come before the enemies' in the
/// `AfterDeath` walk (`CombatState.cs:115-160`), so Gremlin Horn draws a
/// Strike that Hellraiser auto-plays into Crusher's bare HP before Crusher's
/// `CrabRagePower.AfterDeath` (`CrabRagePower.cs:27-38`) grants +6 Strength
/// and 99 block. act23 Ironclad:4187 step 690.
#[test]
fn gremlin_horn_draw_resolves_before_a_siblings_crab_rage() {
    for v in BOTH {
        let mut state = fresh(v);
        state
            .creatures
            .push(enemy(2, 40, MonsterKind::Crusher, "Crusher"));
        state
            .creatures
            .push(enemy(3, 6, MonsterKind::Rocket, "Rocket"));
        let crusher = CreatureUid(NonZeroU32::new(2).expect("nz"));
        let rocket = CreatureUid(NonZeroU32::new(3).expect("nz"));
        state.enemy_uids.push(crusher);
        state.enemy_uids.push(rocket);
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            crusher,
            PowerKind::CrabRage,
            1,
        );
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Hellraiser, 1);
        let drawn = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(drawn);
        let strike = new_card(&mut state, CardId::StrikeIronclad);
        state.player.hand.push(strike);

        // Target index 1 = Rocket.
        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(1));

        let hp_block = |uid: CreatureUid| {
            let c = state.creatures.iter().find(|c| c.id == uid.0).expect("c");
            (c.current_hp, c.block)
        };
        assert_eq!(hp_block(rocket).0, 0, "{v:?}: the Strike killed Rocket");
        assert_eq!(
            hp_block(crusher),
            (34, 99),
            "{v:?}: the auto-played Strike hit Crusher before CrabRage's block"
        );
    }
}

/// The Bomb detonates on Crusher (lethal) and Rocket, which holds
/// `CrabRagePower`. `TheBombPower.BeforeSideTurnEnd` is ONE multi-target
/// `CreatureCmd.Damage(HittableEnemies, ..)` (TheBombPower.cs:56, both
/// builds), which applies every hit before `Kill(killedCreatures)`, so Rocket
/// takes the full 40 before Crusher's death gives it CrabRage's 99 block.
/// Rust killed Crusher before hitting Rocket, whose new block ate the bomb.
/// Random-loadout Ironclad v0.111.0 seed 1050.
#[test]
fn the_bomb_hits_every_enemy_before_a_kill_triggers_crab_rage() {
    for v in BOTH {
        let mut state = fresh(v);
        state
            .creatures
            .push(enemy(2, 5, MonsterKind::Crusher, "Crusher"));
        state
            .creatures
            .push(enemy(3, 60, MonsterKind::Rocket, "Rocket"));
        let crusher = CreatureUid(NonZeroU32::new(2).expect("nz"));
        let rocket = CreatureUid(NonZeroU32::new(3).expect("nz"));
        state.enemy_uids.push(crusher);
        state.enemy_uids.push(rocket);
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            rocket,
            PowerKind::CrabRage,
            1,
        );
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::TheBomb, 1);
        for pw in &mut state.powers {
            if pw.kind == PowerKind::TheBomb {
                pw.counter = 40;
            }
        }

        sts2_core::hooks::dispatch_before_turn_end(
            &mut state,
            &sts2_core::hooks::TurnSideCtx {
                side: sts2_core::Side::Player,
            },
        );

        let hp_block = |uid: CreatureUid| {
            let c = state.creatures.iter().find(|c| c.id == uid.0).expect("c");
            (c.current_hp, c.block)
        };
        assert_eq!(hp_block(crusher).0, 0, "{v:?}: the bomb killed Crusher");
        assert_eq!(
            hp_block(rocket),
            (20, 99),
            "{v:?}: the bomb hit Rocket before CrabRage's block"
        );
    }
}

// ── A select inside `Hook.AfterDeath` does not block ─────────────────────

fn resolve(state: &mut CombatState, idx: u8) {
    let _ = sts2_core::apply(
        state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![idx],
        },
    );
}

/// Two Nibbits, Gremlin Horn, Stratagem, an empty draw pile. Returns the
/// state, the dying Nibbit and the discard it reshuffles.
fn horn_stratagem_fixture(
    v: GameVersion,
    dying_hp: i32,
    discard: &[CardId],
) -> (CombatState, CreatureUid, Vec<sts2_core::CardUid>) {
    let mut state = fresh(v);
    state
        .creatures
        .push(enemy(2, dying_hp, MonsterKind::Nibbit, "Nibbit"));
    state
        .creatures
        .push(enemy(3, 90, MonsterKind::Nibbit, "Nibbit"));
    let dying = CreatureUid(NonZeroU32::new(2).expect("nz"));
    state.enemy_uids.push(dying);
    state
        .enemy_uids
        .push(CreatureUid(NonZeroU32::new(3).expect("nz")));
    add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
    let p = state.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Stratagem, 1);
    let cards: Vec<_> = discard
        .iter()
        .map(|&id| {
            let uid = new_card(&mut state, id);
            state.player.discard.push(uid);
            uid
        })
        .collect();
    (state, dying, cards)
}

/// A Strike kills one of two Nibbits. Gremlin Horn's `AfterDeath` draw
/// reshuffles the discard and Stratagem opens its select, but `Hook.AfterDeath`
/// runs each listener under a `HookPlayerChoiceContext` (Hook.cs:594-606):
/// the select is ENQUEUED as a `GenericHookGameAction` behind the running
/// `PlayCardAction` (HookPlayerChoiceContext.SignalPlayerChoiceBegun), so the
/// whole play finishes first: the Strike is already in the discard when the
/// grid shows. The pick then lands and the horn's own draw takes the next card.
/// Random-loadout Defect v0.107.1 seed 1038 step 165 (Volley).
#[test]
fn after_death_select_waits_for_the_play_to_finish() {
    for v in BOTH {
        let (mut state, dying, reshuffled) =
            horn_stratagem_fixture(v, 5, &[CardId::DefendIronclad, CardId::Bash, CardId::Anger]);
        let strike = new_card(&mut state, CardId::StrikeIronclad);
        state.player.hand.push(strike);

        let r = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

        let hp = state.creatures.iter().find(|c| c.id == dying.0).expect("c");
        assert_eq!(hp.current_hp, 0, "{v:?}: the Strike killed the Nibbit");
        assert_eq!(r, sts2_core::StepResult::PendingChoice, "{v:?}");
        let pc = state.pending_choice.as_ref().expect("the grid surfaced");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand, "{v:?}");
        assert_eq!(
            state.player.discard.as_slice(),
            &[strike],
            "{v:?}: the play finished (Strike discarded) before the select"
        );
        assert_eq!(state.player.draw.len(), 3, "{v:?}: discard reshuffled");
        assert!(state.player.hand.is_empty(), "{v:?}: nothing drawn yet");

        resolve(&mut state, 0);
        assert!(state.pending_choice.is_none(), "{v:?}");
        assert!(state.pending_draw_resume.is_none(), "{v:?}");
        assert!(state.combat_log.parked_hook_choices.is_none(), "{v:?}");
        assert_eq!(
            state.player.hand.len(),
            2,
            "{v:?}: the pick, then the horn's Draw(1)"
        );
        assert_eq!(state.player.draw.len(), 1, "{v:?}");
        assert!(
            state.player.hand.iter().all(|c| reshuffled.contains(c)),
            "{v:?}"
        );
    }
}

/// Poison kills one Nibbit at the enemy-side turn start. Gremlin Horn's draw
/// reshuffles the five-card discard into a Stratagem select that is enqueued
/// behind the running enemy-turn action (HookPlayerChoiceContext), so the
/// enemy turn and the next player turn start run first: the turn draw takes
/// all five cards. Then the grid shows its snapshot, and the answer's
/// `CardPileCmd.Add(pick, Hand)` (CardPileCmd.cs:306-520) moves the pick from
/// wherever it is now (the hand) to the hand's end; the horn's remaining draw
/// finds nothing to draw. Random-loadout Regent v0.107.1 seed 1028 (GLOW).
#[test]
fn after_death_select_on_enemy_turn_waits_for_the_next_player_turn() {
    for v in BOTH {
        let (mut state, dying, reshuffled) = horn_stratagem_fixture(
            v,
            3,
            &[
                CardId::DefendIronclad,
                CardId::Bash,
                CardId::Anger,
                CardId::IronWave,
                CardId::TwinStrike,
            ],
        );
        sts2_core::card_effects::apply_power_to_creature(&mut state, dying, PowerKind::Poison, 5);

        let r = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);

        let hp = state.creatures.iter().find(|c| c.id == dying.0).expect("c");
        assert_eq!(hp.current_hp, 0, "{v:?}: Poison killed the Nibbit");
        assert_eq!(r, sts2_core::StepResult::PendingChoice, "{v:?}");
        assert_eq!(state.phase, CombatPhase::PlayerPlayPhase, "{v:?}");
        let pc = state.pending_choice.as_ref().expect("the grid surfaced");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand, "{v:?}");
        assert_eq!(
            state.player.hand.len(),
            5,
            "{v:?}: the turn draw ran before the select"
        );
        assert!(state.player.draw.is_empty(), "{v:?}");

        let before = state.player.hand.clone();
        resolve(&mut state, 0);
        assert!(state.pending_choice.is_none(), "{v:?}");
        assert!(state.pending_draw_resume.is_none(), "{v:?}");
        assert!(state.combat_log.parked_hook_choices.is_none(), "{v:?}");
        assert_eq!(state.player.hand.len(), 5, "{v:?}: nothing left to draw");
        let pick = *state.player.hand.last().expect("hand");
        assert!(reshuffled.contains(&pick), "{v:?}");
        let mut rest = before.clone();
        rest.retain(|c| *c != pick);
        assert_eq!(
            &state.player.hand[..4],
            rest.as_slice(),
            "{v:?}: the pick, already in hand, moved to the end"
        );
        assert_ne!(before.last(), Some(&pick), "{v:?}: the pick really moved");
    }
}

// ── A per-card draw listener can end the draw mid-way ────────────────────

/// Speedster's per-card `AfterCardDrawn` damage (SpeedsterPower.cs:15-23)
/// kills the last enemy on the first card of a draw 2. C# `CardPileCmd.Draw`
/// re-checks `IsOverOrEnding` each iteration (CardPileCmd.cs:852) and stops
/// before `ShuffleIfNecessary`, so the discard is never reshuffled; the batch
/// draw dealt the damage after the whole draw and over-read the shuffle
/// stream. Random-loadout Silent v0.107.1 seed 2004.
#[test]
fn speedster_kill_mid_draw_stops_before_the_reshuffle() {
    for v in BOTH {
        let mut state = fresh(v);
        state
            .creatures
            .push(enemy(2, 2, MonsterKind::Nibbit, "Nibbit"));
        state
            .enemy_uids
            .push(CreatureUid(NonZeroU32::new(2).expect("nz")));
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Speedster, 2);
        let top = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(top);
        for _ in 0..3 {
            let uid = new_card(&mut state, CardId::DefendIronclad);
            state.player.discard.push(uid);
        }

        let _ = state.draw_into_hand_card_effect(2);

        let hp = state.creatures.iter().find(|c| c.id.get() == 2).expect("c");
        assert_eq!(hp.current_hp, 0, "{v:?}: Speedster killed the Nibbit");
        assert_eq!(state.player.hand.as_slice(), &[top], "{v:?}");
        assert_eq!(state.player.discard.len(), 3, "{v:?}: no reshuffle");
    }
}

/// v0.111.0 CacophonyPower's 33rd-draw hit (CacophonyPower.cs:22-37) kills
/// the last enemy on the first card of a draw 3 — C# `CardPileCmd.Draw` stops
/// on `IsOverOrEnding` (v0.111.0 CardPileCmd.cs:1042) before reshuffling the
/// discard, on the turn-start hand draw and on an effect draw alike. The batch
/// draw dealt the hit after the whole draw and over-read the run's `shuffle`
/// stream, so the next combat opened on another hand. Random-loadout
/// Necrobinder v0.111.0 seeds 1288 (turn-start draw) and 1269 (an
/// Eidolon-played Parse).
#[test]
fn cacophony_kill_mid_draw_stops_before_the_reshuffle() {
    for hand_draw in [true, false] {
        let mut state = fresh(GameVersion::V0_111_0);
        state
            .creatures
            .push(enemy(2, 14, MonsterKind::Nibbit, "Nibbit"));
        state
            .enemy_uids
            .push(CreatureUid(NonZeroU32::new(2).expect("nz")));
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Cacophony, 66);
        for pw in &mut state.powers {
            if pw.kind == PowerKind::Cacophony {
                pw.counter = 32;
            }
        }
        let top = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(top);
        for _ in 0..3 {
            let uid = new_card(&mut state, CardId::DefendIronclad);
            state.player.discard.push(uid);
        }

        let _ = if hand_draw {
            state.draw_into_hand(3)
        } else {
            state.draw_into_hand_card_effect(3)
        };

        let hp = state.creatures.iter().find(|c| c.id.get() == 2).expect("c");
        assert_eq!(hp.current_hp, 0, "hand_draw={hand_draw}: Cacophony killed");
        assert_eq!(
            state.player.hand.as_slice(),
            &[top],
            "hand_draw={hand_draw}"
        );
        assert_eq!(
            state.player.discard.len(),
            3,
            "hand_draw={hand_draw}: no reshuffle"
        );
    }
}

/// Centennial Puzzle draws three single cards (CentennialPuzzle.cs:47-50),
/// awaiting each. The second one reshuffles the discard and Stratagem opens
/// its select: the third draw waits for the pick instead of being drawn over
/// (or dropped past) the open select. Hand: 1 + pick + 1 + 1.
/// Random-loadout Ironclad v0.111.0 seed 2006.
#[test]
fn centennial_puzzle_draws_resume_after_a_stratagem_pick() {
    for v in BOTH {
        let (mut state, _, _) = horn_stratagem_fixture(
            v,
            50,
            &[
                CardId::DefendIronclad,
                CardId::Bash,
                CardId::Anger,
                CardId::Inflame,
                CardId::ShrugItOff,
            ],
        );
        add_relic(&mut state, sts2_core::RelicKind::CentennialPuzzle);
        let top = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(top);

        let _ = sts2_core::combat_loop::apply_damage_to_player(
            &mut state,
            sts2_core::hooks::CreatureRef::Nobody,
            5,
            sts2_core::hooks::ValueProps::NONE,
            None,
        );
        assert_eq!(state.player.hand.as_slice(), &[top], "{v:?}");
        let pc = state.pending_choice.as_ref().expect("the grid surfaced");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand, "{v:?}");

        resolve(&mut state, 0);
        assert!(state.pending_choice.is_none(), "{v:?}");
        assert!(state.pending_draw_resume.is_none(), "{v:?}");
        assert_eq!(state.player.hand.len(), 4, "{v:?}: 1 + pick + 1 + 1");
        assert_eq!(state.player.draw.len(), 2, "{v:?}");
    }
}

/// A draw 2 takes an Ethereal Dazed; Pagestorm's `AfterCardDrawn` draw 1
/// (PagestormPower.cs) reshuffles and Stratagem opens its select. C#'s outer
/// `CardPileCmd.Draw` awaits the rider, so after the pick the rider draws its
/// card and only then does the outer draw take its second: Dazed, pick, the
/// rider's card, the outer card. Rust's outer loop drew on over the open
/// select. Random-loadout Necrobinder v0.111.0 seed 2028.
#[test]
fn outer_draw_waits_behind_a_pagestorm_draws_stratagem_pick() {
    for v in BOTH {
        let (mut state, _, _) = horn_stratagem_fixture(
            v,
            50,
            &[CardId::DefendIronclad, CardId::Bash, CardId::Anger],
        );
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Pagestorm, 1);
        let dazed = new_card(&mut state, CardId::Dazed);
        state.player.draw.push(dazed);

        let _ = state.draw_into_hand_card_effect(2);
        assert_eq!(state.player.hand.as_slice(), &[dazed], "{v:?}");
        let pc = state.pending_choice.as_ref().expect("the grid surfaced");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand, "{v:?}");

        resolve(&mut state, 0);
        assert!(state.pending_choice.is_none(), "{v:?}");
        assert!(state.pending_draw_resume.is_none(), "{v:?}");
        assert_eq!(state.player.hand.len(), 4, "{v:?}: Dazed + pick + 1 + 1");
        assert!(state.player.draw.is_empty(), "{v:?}");
    }
}

/// A turn-end Lightning passive kills one of two Nibbits, and Gremlin Horn's
/// `AfterDeath` draw reshuffles into Stratagem's grid. The hook's action is
/// queued behind the end-turn action, and C# flushes the hand
/// (`EndPlayerTurnPhaseTwoInternal`) only once the queue is empty, so the
/// pick opens before the flush: the Strike is still in hand. Rust parked it
/// until the next player turn (loadout Defect 1119 v0.107.1).
#[test]
fn turn_end_orb_kill_select_opens_before_the_hand_flush() {
    for v in BOTH {
        let (mut state, _dying, _reshuffled) =
            horn_stratagem_fixture(v, 1, &[CardId::DefendIronclad, CardId::Bash, CardId::Anger]);
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id.get() == 3) {
            c.current_hp = 1;
        }
        state.player.character = Some(sts2_core::potions::CharacterId::Defect);
        state.player.orbs.ensure_defect_base();
        sts2_core::orbs::channel_orb(&mut state, sts2_core::orbs::Orb::Lightning);
        let strike = new_card(&mut state, CardId::StrikeIronclad);
        state.player.hand.push(strike);

        let r = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);

        assert_eq!(r, sts2_core::StepResult::PendingChoice, "{v:?}");
        let pc = state.pending_choice.as_ref().expect("the grid surfaced");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand, "{v:?}");
        assert!(
            state.player.hand.contains(&strike),
            "{v:?}: the hand is not flushed yet"
        );
        resolve(&mut state, 0);
        assert!(
            !state.player.hand.contains(&strike),
            "{v:?}: the flush ran after the pick"
        );
    }
}
