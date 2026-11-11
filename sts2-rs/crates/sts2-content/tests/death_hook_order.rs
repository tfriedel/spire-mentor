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

/// Breakthrough kills one of two Nibbits. Gremlin Horn's `AfterDeath` draws a
/// Seeker Strike, which Hellraiser auto-plays inline
/// (`HellraiserPower.AfterCardDrawnEarly`): its hit lands, then its
/// `CardSelectCmd.FromCombatPile` signals a player choice. That pauses the
/// horn's listener under its `HookPlayerChoiceContext`. The comment on
/// `_gameActionType` names exactly this chain: "Hellraiser triggers Seeker
/// Strike / Seeker Strike causes a player choice". Breakthrough's play
/// finishes and goes to the discard before the grid shows. Only then does the
/// pick land and Seeker Strike follow it into the discard. Random-loadout
/// Ironclad v0.111.0 seed 3249 step 383.
#[test]
fn hellraiser_auto_play_select_in_after_death_waits_for_the_play() {
    for v in BOTH {
        let (mut state, dying, _) = horn_stratagem_fixture(v, 5, &[]);
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Hellraiser, 1);
        let seeker = new_card(&mut state, CardId::SeekerStrike);
        state.player.draw.push(seeker);
        for id in [CardId::DefendIronclad, CardId::Bash, CardId::Anger] {
            let uid = new_card(&mut state, id);
            state.player.draw.push(uid);
        }
        let breakthrough = new_card(&mut state, CardId::Breakthrough);
        state.player.hand.push(breakthrough);

        let r = sts2_core::combat_loop::play_card(&mut state, 0, None);

        let hp = state.creatures.iter().find(|c| c.id == dying.0).expect("c");
        assert_eq!(hp.current_hp, 0, "{v:?}: Breakthrough killed the Nibbit");
        assert_eq!(r, sts2_core::StepResult::PendingChoice, "{v:?}");
        assert!(
            state.pending_choice.is_some(),
            "{v:?}: Seeker Strike's pick"
        );
        assert_eq!(
            state.player.discard.as_slice(),
            &[breakthrough],
            "{v:?}: Breakthrough's play finished before the pick opened"
        );

        resolve(&mut state, 0);
        assert!(state.pending_choice.is_none(), "{v:?}");
        assert!(state.pending_autoplay_resume.is_none(), "{v:?}");
        assert!(state.combat_log.parked_hook_choices.is_none(), "{v:?}");
        assert_eq!(
            state.player.discard.as_slice(),
            &[breakthrough, seeker],
            "{v:?}: Seeker Strike follows once its pick resolves"
        );
        assert_eq!(state.player.hand.len(), 1, "{v:?}: the pick");
        assert_eq!(state.player.draw.len(), 2, "{v:?}");
    }
}

/// Thunderclap applies Vulnerable to both Nibbits, and Vicious draws on each
/// (`ViciousPower.AfterPowerAmountChanged`, under the card's own context). The
/// first draw finds the draw pile empty: it reshuffles and Stratagem opens
/// its grid. `PowerCmd.Apply` awaits that draw, pick included, before the
/// second Nibbit's Vulnerable, so the second draw comes after the pick. Hand:
/// pick, the first draw, the second draw. Random-loadout Ironclad v0.107.1
/// seed 3290 step 708 (Rust drew ahead of the open grid).
#[test]
fn a_draw_behind_an_open_reshuffle_pick_waits_for_it() {
    for v in BOTH {
        let (mut state, _, reshuffled) = horn_stratagem_fixture(
            v,
            50,
            &[
                CardId::DefendIronclad,
                CardId::Bash,
                CardId::Anger,
                CardId::Inflame,
            ],
        );
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Vicious, 1);
        let clap = new_card(&mut state, CardId::Thunderclap);
        state.player.hand.push(clap);

        let r = sts2_core::combat_loop::play_card(&mut state, 0, None);

        assert_eq!(r, sts2_core::StepResult::PendingChoice, "{v:?}");
        assert!(state.player.hand.is_empty(), "{v:?}: nothing drawn yet");
        let pick = state.pending_choice.as_ref().expect("the grid").options[0]
            .card_uid
            .expect("a card");

        resolve(&mut state, 0);
        assert!(state.pending_choice.is_none(), "{v:?}");
        assert!(state.pending_draw_resume.is_none(), "{v:?}");
        assert_eq!(state.player.hand.len(), 3, "{v:?}: pick + two draws");
        assert_eq!(state.player.hand[0], pick, "{v:?}: the pick comes first");
        assert!(
            state.player.hand.iter().all(|c| reshuffled.contains(c)),
            "{v:?}"
        );
        assert_eq!(state.player.discard.as_slice(), &[clap], "{v:?}");
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

/// Scrape kills one Nibbit; Gremlin Horn's `AfterDeath` draw reshuffles the
/// three-card discard into a Stratagem select that is parked behind the play.
/// Scrape then draws those three cards and discards them (all cost 1), so the
/// draw pile is empty when the grid finally shows. C# `CardPileCmd.Draw`
/// resumes right after `ShuffleIfNecessary`: `drawPile.FirstOrDefault()` is
/// null and the horn's draw ends (CardPileCmd.cs:857-866 v0.107.1). Rust
/// re-entered the draw loop from the top and reshuffled the discard a second
/// time, opening a second Stratagem select. Random-loadout Defect v0.107.1
/// seed 5393.
#[test]
fn parked_reshuffle_select_resumes_without_a_second_shuffle() {
    for v in BOTH {
        let (mut state, dying, reshuffled) = horn_stratagem_fixture(
            v,
            5,
            &[CardId::DefendIronclad, CardId::Bash, CardId::IronWave],
        );
        let scrape = new_card(&mut state, CardId::Scrape);
        state.player.hand.push(scrape);

        let r = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

        let hp = state.creatures.iter().find(|c| c.id == dying.0).expect("c");
        assert_eq!(hp.current_hp, 0, "{v:?}: Scrape killed the Nibbit");
        assert_eq!(r, sts2_core::StepResult::PendingChoice, "{v:?}");
        let pc = state.pending_choice.as_ref().expect("the grid surfaced");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand, "{v:?}");
        assert!(state.player.draw.is_empty(), "{v:?}: Scrape drew the pile");
        assert_eq!(state.player.discard.len(), 4, "{v:?}: 3 discarded + Scrape");

        resolve(&mut state, 0);
        assert!(
            state.pending_choice.is_none(),
            "{v:?}: no second Stratagem select"
        );
        assert!(state.pending_draw_resume.is_none(), "{v:?}");
        assert!(state.combat_log.parked_hook_choices.is_none(), "{v:?}");
        assert!(state.player.draw.is_empty(), "{v:?}: no second reshuffle");
        assert_eq!(state.player.hand.len(), 1, "{v:?}: only the pick");
        assert!(reshuffled.contains(&state.player.hand[0]), "{v:?}");
        assert_eq!(state.player.discard.len(), 3, "{v:?}");
    }
}

/// The blocking twin: Scrape's own draw reshuffles into the Stratagem select,
/// which C# awaits inside the draw. After the pick the draw takes the rest of
/// the reshuffled pile and stops at the empty pile with nothing to reshuffle.
#[test]
fn blocking_reshuffle_select_resumes_on_the_reshuffled_pile() {
    for v in BOTH {
        let (mut state, _dying, reshuffled) = horn_stratagem_fixture(
            v,
            50,
            &[CardId::DefendIronclad, CardId::Bash, CardId::IronWave],
        );
        let scrape = new_card(&mut state, CardId::Scrape);
        state.player.hand.push(scrape);

        let r = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
        assert_eq!(r, sts2_core::StepResult::PendingChoice, "{v:?}");
        assert_eq!(state.player.draw.len(), 3, "{v:?}: reshuffled");

        resolve(&mut state, 0);
        assert!(state.pending_choice.is_none(), "{v:?}");
        assert!(state.pending_draw_resume.is_none(), "{v:?}");
        assert!(state.player.draw.is_empty(), "{v:?}");
        // The pick is Stratagem's, not part of Scrape's draw result, so it
        // stays; Scrape discards the two cards its draw took (cost 1).
        assert_eq!(state.player.hand.len(), 1, "{v:?}: the pick");
        assert!(reshuffled.contains(&state.player.hand[0]), "{v:?}");
        let mut gone = state.player.discard.clone();
        gone.retain(|c| reshuffled.contains(c));
        assert_eq!(gone.len(), 2, "{v:?}: both drawn cards discarded");
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

/// `Hook.AfterCardDrawn` awaits the player's powers in list order. With
/// Pagestorm applied before Cacophony, the Ethereal Dazed that would be
/// Cacophony's 33rd card first runs Pagestorm's nested draw: that reshuffles
/// the discard and the drawn card is the 33rd, whose hit kills the last
/// enemy; only then does Cacophony count the Dazed. Rust counted the Dazed
/// first, killed on it and skipped Pagestorm's draw (combat ending), so the
/// reshuffle never ran and the next combat opened on another hand
/// (random-loadout Necrobinder v0.111.0 seed 8638). Applied the other way
/// round, Cacophony does fire on the Dazed first.
#[test]
fn pagestorm_before_cacophony_draws_before_the_ethereal_card_counts() {
    for pagestorm_first in [true, false] {
        let mut state = fresh(GameVersion::V0_111_0);
        state
            .creatures
            .push(enemy(2, 14, MonsterKind::Nibbit, "Nibbit"));
        state
            .enemy_uids
            .push(CreatureUid(NonZeroU32::new(2).expect("nz")));
        let p = state.player.creature_uid.expect("player");
        let mut kinds = [PowerKind::Pagestorm, PowerKind::Cacophony];
        if !pagestorm_first {
            kinds.reverse();
        }
        for kind in kinds {
            let amount = if kind == PowerKind::Cacophony { 66 } else { 1 };
            sts2_core::card_effects::apply_power_to_creature(&mut state, p, kind, amount);
        }
        for pw in &mut state.powers {
            if pw.kind == PowerKind::Cacophony {
                pw.counter = 32;
            }
        }
        let dazed = new_card(&mut state, CardId::Dazed);
        state.player.draw.push(dazed);
        for _ in 0..3 {
            let uid = new_card(&mut state, CardId::DefendIronclad);
            state.player.discard.push(uid);
        }

        let _ = state.draw_into_hand(2);

        let hp = state.creatures.iter().find(|c| c.id.get() == 2).expect("c");
        assert_eq!(hp.current_hp, 0, "pagestorm_first={pagestorm_first}");
        if pagestorm_first {
            assert_eq!(state.player.hand.len(), 2, "Pagestorm drew before the hit");
            assert!(
                state.player.discard.is_empty(),
                "the discard was reshuffled"
            );
        } else {
            assert_eq!(state.player.hand.as_slice(), &[dazed]);
            assert_eq!(state.player.discard.len(), 3, "no reshuffle");
        }
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

/// The same turn-end Lightning kill, with an Ethereal Dazed in hand. C#'s
/// queued hook action runs only after the whole of `EndPlayerTurnPhaseOne`:
/// `DoTurnEnd` exhausts the Ethereal hand first (CombatManager.cs, after the
/// orb passives), then `BeforeFlush`, and only then does the pick open. The
/// picked Dazed and the one the horn draws after it miss the Ethereal sweep and
/// are discarded with the hand. Rust opened the pick before the sweep and
/// exhausted them too (random-loadout Ironclad 7756 v0.111.0).
#[test]
fn turn_end_orb_kill_select_opens_after_the_ethereal_sweep() {
    for v in BOTH {
        let (mut state, _dying, _reshuffled) =
            horn_stratagem_fixture(v, 1, &[CardId::Dazed, CardId::Dazed, CardId::Dazed]);
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id.get() == 3) {
            c.current_hp = 1;
        }
        state.player.character = Some(sts2_core::potions::CharacterId::Defect);
        state.player.orbs.ensure_defect_base();
        sts2_core::orbs::channel_orb(&mut state, sts2_core::orbs::Orb::Lightning);
        let dazed = new_card(&mut state, CardId::Dazed);
        state.player.hand.push(dazed);

        let r = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);

        assert_eq!(r, sts2_core::StepResult::PendingChoice, "{v:?}");
        let pc = state.pending_choice.as_ref().expect("the grid surfaced");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand, "{v:?}");
        assert_eq!(
            state.player.exhaust.as_slice(),
            &[dazed],
            "{v:?}: the Ethereal sweep ran before the pick"
        );
        resolve(&mut state, 0);
        assert_eq!(
            state.player.exhaust.as_slice(),
            &[dazed],
            "{v:?}: the picked and drawn Dazed are flushed, not exhausted"
        );
    }
}

// ── A nested damage call kills its own targets ───────────────────────────

/// Flick-Flack (8 with Strength 2) hits four enemies at [21, 8, 19, 17]; the
/// second dies. The first target's Envenom Poison is Outbreak's third, so
/// Outbreak's own `CreatureCmd.Damage` (OutbreakPower.cs:47) deals 11 to the
/// living [13, -, 11, 9] and, at the end of THAT call, kills the third and
/// fourth (CreatureCmd.cs:409 v0.107.1), which `RemoveCreature` takes out of
/// the combat. The outer tails then poison the second (killed by the outer
/// call, not yet `Kill`ed, so it can still receive powers): Outbreak counts 1.
/// The removed third and fourth receive nothing (`CanReceivePowers`,
/// Creature.cs:308-321), so Outbreak does not fire again and the first enemy
/// lives at 2. Rust queued the nested kills on the outer batch, poisoned all
/// three corpses and fired Outbreak a second time, ending the combat (loadout
/// Silent 5093 v0.107.1). Outbreak is gone in v0.111.0.
#[test]
fn a_nested_outbreak_kills_its_own_targets_before_the_outer_tails() {
    let mut state = fresh(GameVersion::V0_107_1);
    state.player.character = Some(CharacterId::Silent);
    for (i, hp) in [21, 8, 19, 17].into_iter().enumerate() {
        let id = u32::try_from(i).expect("u32") + 2;
        state
            .creatures
            .push(enemy(id, hp, MonsterKind::Nibbit, "Nibbit"));
        state
            .enemy_uids
            .push(CreatureUid(NonZeroU32::new(id).expect("nz")));
    }
    let p = state.player.creature_uid.expect("player");
    for (kind, amount) in [
        (PowerKind::Envenom, 1),
        (PowerKind::Outbreak, 11),
        (PowerKind::Strength, 2),
    ] {
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, kind, amount);
    }
    for pw in &mut state.powers {
        if pw.kind == PowerKind::Outbreak {
            pw.counter = 2;
        }
    }
    let ff = new_card(&mut state, CardId::FlickFlack);
    state.player.hand.push(ff);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);

    let poison = |state: &CombatState, i: usize| {
        state.creatures[i]
            .powers
            .iter()
            .filter_map(|puid| {
                state
                    .powers
                    .iter()
                    .find(|p| p.id == puid.0 && p.kind == PowerKind::Poison)
            })
            .map(|p| p.amount)
            .sum::<i32>()
    };
    let hp: Vec<i32> = state.creatures[1..].iter().map(|c| c.current_hp).collect();
    assert!(!state.is_over, "the first enemy survives");
    assert_eq!(hp, vec![2, 0, 0, 0]);
    assert_eq!(poison(&state, 1), 1, "the first enemy's Envenom Poison");
    for i in [3, 4] {
        assert_eq!(poison(&state, i), 0, "removed enemy {i} takes no Poison");
    }
    let outbreak = state
        .powers
        .iter()
        .find(|p| p.kind == PowerKind::Outbreak)
        .expect("outbreak");
    assert_eq!(
        outbreak.counter, 1,
        "only the second enemy's Poison counted"
    );
}

/// Outbreak's counter stays at 3 while its own damage resolves: C# does
/// `timesPoisoned %= 3` after the awaited `CreatureCmd.Damage`
/// (OutbreakPower.cs:42-48 v0.107.1). Deadly Poison is Outbreak's third; its
/// 11 kills the minion, Gremlin Horn draws, and Corrosive Wave's Poison on
/// that draw is the fourth, which fires Outbreak again before the outer `%=`.
/// Rust reset the counter before the damage, so the nested Poison counted 1
/// (Silent 6003 v0.107.1 @703: monsters[0].hp cs=73 rs=88). Outbreak is gone in
/// v0.111.0.
#[test]
fn a_poison_inside_outbreaks_damage_counts_on_top_of_its_third() {
    let mut state = fresh(GameVersion::V0_107_1);
    state.player.character = Some(CharacterId::Silent);
    for (id, hp) in [(2, 100), (3, 5)] {
        state
            .creatures
            .push(enemy(id, hp, MonsterKind::Nibbit, "Nibbit"));
        state
            .enemy_uids
            .push(CreatureUid(NonZeroU32::new(id).expect("nz")));
    }
    add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
    let p = state.player.creature_uid.expect("player");
    for (kind, amount) in [(PowerKind::Outbreak, 11), (PowerKind::CorrosiveWave, 1)] {
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, kind, amount);
    }
    for pw in &mut state.powers {
        if pw.kind == PowerKind::Outbreak {
            pw.counter = 2;
        }
    }
    let drawn = new_card(&mut state, CardId::DefendIronclad);
    state.player.draw.push(drawn);
    let dp = new_card(&mut state, CardId::DeadlyPoison);
    state.player.hand.push(dp);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

    assert!(state.player.hand.contains(&drawn), "Gremlin Horn drew");
    assert_eq!(state.creatures[2].current_hp, 0, "the 5-HP enemy died");
    assert_eq!(
        state.creatures[1].current_hp,
        100 - 11 - 11,
        "Outbreak fired twice"
    );
    let outbreak = state
        .powers
        .iter()
        .find(|p| p.kind == PowerKind::Outbreak)
        .expect("outbreak");
    assert_eq!(outbreak.counter, 1, "4 % 3 after the nested proc");
}

// ── The Hunt's extra card reward on the last enemy ───────────────────────

/// The Hunt's Fatal kill of the LAST enemy still adds its extra card reward:
/// C# calls `combatRoom.AddExtraReward(.., new CardReward(..))` before
/// `PowerCmd.Apply<TheHuntPower>` (TheHunt.cs:45-48, both builds), and only
/// the power apply no-ops once the combat `IsEnding` (PowerCmd.cs:69-72).
/// Rust counted the reward by the power's stacks, which the ending-combat
/// guard drops, so the room offered one card reward short (random-loadout
/// Silent 5307 v0.107.1, 5345 v0.111.0).
#[test]
fn the_hunt_killing_the_last_enemy_still_adds_its_card_reward() {
    for v in BOTH {
        let mut state = fresh(v);
        state.player.character = Some(CharacterId::Silent);
        state
            .creatures
            .push(enemy(2, 5, MonsterKind::Nibbit, "Nibbit"));
        let target = CreatureUid(NonZeroU32::new(2).expect("nz"));
        state.enemy_uids.push(target);
        let hunt = new_card(&mut state, CardId::TheHunt);
        state.player.hand.push(hunt);
        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
        assert_eq!(state.creatures[1].current_hp, 0, "{v:?}: the kill");
        assert_eq!(
            state.combat_log.extra_card_rewards_this_combat, 1,
            "{v:?}: the room's extra card reward"
        );
    }
}

// ── BeforeSideTurnEnd runs its Early walk before the normal one ──────────

/// `Hook.BeforeSideTurnEnd` is three full listener walks — VeryEarly, Early,
/// then normal (Hook.cs:1388-1417, both builds). Plating's block is Early
/// (`PlatingPower.BeforeSideTurnEndEarly`), The Bomb's detonation normal, so
/// even with The Bomb listed first Plating's 9 block lands, Juggernaut's
/// `AfterBlockGained` hit picks its random target (one `combat_targets`
/// draw), and only then does the Bomb kill the enemy. Rust walked once in
/// listener order: the Bomb killed the last enemy first and Juggernaut drew
/// nothing (random-loadout Ironclad 5429 v0.111.0).
#[test]
fn plating_blocks_before_the_bomb_detonates() {
    for v in BOTH {
        let mut state = fresh(v);
        state
            .creatures
            .push(enemy(2, 9, MonsterKind::Nibbit, "Nibbit"));
        let target = CreatureUid(NonZeroU32::new(2).expect("nz"));
        state.enemy_uids.push(target);
        let p = state.player.creature_uid.expect("player");
        for (kind, amount) in [
            (PowerKind::TheBomb, 1),
            (PowerKind::Plating, 9),
            (PowerKind::Juggernaut, 6),
        ] {
            sts2_core::card_effects::apply_power_to_creature(&mut state, p, kind, amount);
        }
        for pw in &mut state.powers {
            if pw.kind == PowerKind::TheBomb {
                pw.counter = 40;
            }
        }
        let targets_before = state.run_rng.combat_targets.counter();
        let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
        assert_eq!(
            state.creatures[1].current_hp, 0,
            "{v:?}: the Bomb killed it"
        );
        assert_eq!(state.player.block, 9, "{v:?}: Plating's block");
        assert_eq!(
            state.run_rng.combat_targets.counter(),
            targets_before + 1,
            "{v:?}: Juggernaut's hit picked a target"
        );
    }
}

// ── A null CombatState throws out of the card play ──────────────────────

/// Echoing Slash (10) hits A [40], B [11, block 10] and C [5]. Loop 2 of its
/// `CreatureCmd.Damage` runs A's tail first: A's Envenom Poison is Outbreak's
/// third, and Outbreak's own damage call kills B and `RemoveCreature` nulls
/// its `CombatState` (CombatState.cs:559). B's tail then reads it:
/// `Hook.AfterBlockBroken(originalTarget.CombatState, ...)`
/// (CreatureCmd.cs:378 v0.107.1) throws a NullReferenceException in the
/// listener walk (Hook.cs:253). The throw skips C's tail and the call's
/// `Kill` (C stays at 0 HP, unkilled), Echoing Slash's rewave for C, and the
/// rest of `OnPlayWrapper`: `AfterCardPlayed` (no Rage block) and the move to
/// Discard. The energy stays spent. Loadout Silent 1037 v0.107.1 (C#: the card
/// stayed in Play, discard 15; Rust 16). Outbreak is gone in v0.111.0.
fn echoing_slash_into_a_nested_outbreak(b_block: i32) -> (CombatState, sts2_core::CardUid) {
    let mut state = fresh(GameVersion::V0_107_1);
    state.player.character = Some(CharacterId::Silent);
    for (id, hp) in [(2_u32, 40), (3, 11), (4, 5)] {
        state
            .creatures
            .push(enemy(id, hp, MonsterKind::Nibbit, "Nibbit"));
        state
            .enemy_uids
            .push(CreatureUid(NonZeroU32::new(id).expect("nz")));
    }
    state.creatures[2].block = b_block;
    let p = state.player.creature_uid.expect("player");
    for (kind, amount) in [
        (PowerKind::Envenom, 1),
        (PowerKind::Outbreak, 11),
        (PowerKind::Rage, 3),
    ] {
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, kind, amount);
    }
    for pw in &mut state.powers {
        if pw.kind == PowerKind::Outbreak {
            pw.counter = 2;
        }
    }
    let slash = new_card(&mut state, CardId::EchoingSlash);
    state.player.hand.push(slash);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    (state, slash)
}

#[test]
fn a_block_broken_target_removed_by_a_nested_outbreak_throws_out_of_the_play() {
    let (state, slash) = echoing_slash_into_a_nested_outbreak(10);
    let hp: Vec<i32> = state.creatures[1..].iter().map(|c| c.current_hp).collect();
    assert_eq!(
        hp,
        vec![19, 0, 0],
        "A took the slash and Outbreak; no rewave"
    );
    assert!(!state.is_over, "A lives");
    assert_eq!(state.player.energy, 4, "the play's energy stays spent");
    assert!(
        !state.player.discard.contains(&slash)
            && !state.player.hand.contains(&slash)
            && !state.player.draw.contains(&slash)
            && !state.player.exhaust.contains(&slash),
        "the card never reaches its result pile"
    );
    assert!(state.player.play.is_empty(), "Rust drops it from Play");
    assert_eq!(state.player.block, 0, "no AfterCardPlayed: no Rage block");
    assert_eq!(state.combat_log.player_card_plays_this_combat, 0);
    assert!(
        !state.combat_log.card_play_faulted,
        "the play took the latch"
    );
}

/// Control: B has no block to break, so its tail runs, C's kill rewaves
/// (A 19 -> 9), and the card finishes into Discard with Rage's block.
#[test]
fn an_unbroken_target_removed_by_a_nested_outbreak_finishes_the_play() {
    let (state, slash) = echoing_slash_into_a_nested_outbreak(0);
    let hp: Vec<i32> = state.creatures[1..].iter().map(|c| c.current_hp).collect();
    assert_eq!(hp, vec![9, 0, 0]);
    assert!(state.player.discard.contains(&slash));
    assert!(state.player.play.is_empty());
    assert_eq!(state.player.block, 3, "Rage's AfterCardPlayed block");
    assert_eq!(state.combat_log.player_card_plays_this_combat, 1);
}

/// Omnislice's splash is ONE multi-target `CreatureCmd.Damage(list2, ..)`
/// (Omnislice.cs, both builds): both teammates lose their HP before
/// `Kill(killedCreatures)` runs. When the primary hit already killed its
/// target, the splash fells the last two enemies, so by their `AfterDeath`
/// every primary enemy is dead and `CombatManager.IsEnding` holds: Gremlin
/// Horn's `PlayerCmd.GainEnergy` and `CardPileCmd.Draw` both no-op
/// (PlayerCmd.cs, CardPileCmd.cs). Only the primary kill gains energy and
/// draws. Rust killed each splash target as it was hit, so the first of them
/// still drew a card and gained energy. Random-loadout Regent v0.107.1 seed
/// 8183 (an auto-played Omnislice under Catastrophe).
#[test]
fn omnislice_splash_kills_resolve_after_every_hit() {
    for v in BOTH {
        let mut state = fresh(v);
        for id in 2..=4 {
            state
                .creatures
                .push(enemy(id, 1, MonsterKind::Inklet, "Inklet"));
            state
                .enemy_uids
                .push(CreatureUid(NonZeroU32::new(id).expect("nz")));
        }
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        for _ in 0..3 {
            let c = new_card(&mut state, CardId::StrikeIronclad);
            state.player.draw.push(c);
        }
        let omni = new_card(&mut state, CardId::Omnislice);
        state.player.hand.push(omni);
        let energy_before = state.player.energy;
        let cost = 0; // Omnislice costs 0.

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

        assert!(
            state.creatures.iter().skip(1).all(|c| c.current_hp <= 0),
            "{v:?}: Omnislice killed all three Inklets"
        );
        assert_eq!(
            state.player.draw.len(),
            2,
            "{v:?}: only the primary kill's Gremlin Horn drew a card"
        );
        assert_eq!(
            state.player.energy,
            energy_before - cost + 1,
            "{v:?}: only the primary kill's Gremlin Horn gained energy"
        );
    }
}

/// `Hook.AfterCardDrawn` walks the player's powers in application order and
/// awaits each (CacophonyPower.cs:22-37, ChainsOfBindingPower.cs:20-30). With
/// Cacophony applied before Chains of Binding, the 33rd card's Cacophony hit
/// kills an enemy, Gremlin Horn draws the next card (GremlinHorn.cs) and that
/// nested card takes the turn's last `Bound` before Chains checks the card
/// that set the hit off, which then stays unafflicted. Rust checked Chains
/// first and bound the outer card. Random-loadout Necrobinder v0.111.0 seed
/// 8178 (the Queen's Chains of Binding).
#[test]
fn cacophony_kill_draw_takes_the_bound_before_its_trigger() {
    let mut state = fresh(GameVersion::V0_111_0);
    for id in 2..=3 {
        state
            .creatures
            .push(enemy(id, 1, MonsterKind::Nibbit, "Nibbit"));
        state
            .enemy_uids
            .push(CreatureUid(NonZeroU32::new(id).expect("nz")));
    }
    let p = state.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Cacophony, 66);
    for pw in &mut state.powers {
        if pw.kind == PowerKind::Cacophony {
            pw.counter = 32;
        }
    }
    // The Queen's Chains of Binding lands on the legacy player list (the
    // monster's `apply_power_to_player_via_legacy_surface`), applied later.
    let chains = NonZeroU32::new(50).expect("nz");
    state
        .powers
        .push(sts2_core::Power::new(chains, PowerKind::ChainsOfBinding, 1));
    state.player.powers.push(sts2_core::PowerUid(chains));
    add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
    let outer = new_card(&mut state, CardId::StrikeIronclad);
    let nested = new_card(&mut state, CardId::DefendIronclad);
    state.player.draw.push(outer);
    state.player.draw.push(nested);

    let _ = state.draw_into_hand(1);

    assert_eq!(state.player.hand.as_slice(), &[outer, nested]);
    let bound = |uid: sts2_core::CardUid| state.cards[uid.get() as usize - 1].affliction.is_some();
    assert!(bound(nested), "the Gremlin Horn draw took the Bound");
    assert!(!bound(outer), "Chains reached the outer card at its cap");
}
