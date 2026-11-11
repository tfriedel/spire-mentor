//! Fight recorder. Drives MCTS through every player decision in a
//! combat and captures enough state, search statistics, and deltas
//! per decision to render a scrollable HTML replay.
//!
//! Loop shape mirrors `crates/sts2-cli/src/sim/mod.rs::CombatSim::
//! play` — same `is_over` exit, same per-round / max-rounds caps —
//! except every `MctsAgent::pick` call is inlined so we can keep the
//! full `root_outcomes` vector and `get_pv` result alongside the
//! recommended action. Each MCTS search starts fresh (no
//! `reroot`) by default so the recorded `wins / losses / visits`
//! reflect that decision's search budget exactly, matching how the
//! vbzji headline numbers were collected. `--tree-reuse` opts in to
//! the alternative.

use std::io;
use std::path::{Path, PathBuf};

use smallvec::SmallVec;
use sts2_core::cards::{CardId, CardType};
use sts2_core::monsters::{Intent, MonsterKind};
use sts2_core::{
    apply, combat_loop, is_terminal, legal_actions, Action, CombatPhase, CombatState, Creature,
    PowerKind, StepResult,
};
use sts2_mcts::{Mcts, MctsConfig, RolloutPolicyKind};

use crate::scenario_to_combat_state;
use crate::sim::scenario::toml_file::{load_scenario_file, ScenarioFile};

/// Caps copied from `CombatSim::play` (bead `sts2-rs-2zkw`):
/// stall + cycle guards so a recorder run can't hang on a
/// pathological combat.
const MAX_COMBAT_ROUNDS: u16 = 100;
const PER_ROUND_ACTION_CAP: u32 = 200;
const DEFAULT_TOP_K_RECORDED: usize = 32;

/// Knobs for the recorder. Independent of `MctsAgentConfig` so the
/// recorder can dial in per-decision search depth / sims without
/// dragging the REPL's tree-reuse default along.
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub sims: u32,
    pub c_puct: f32,
    pub rollout_depth: u32,
    pub seed: u64,
    pub tree_reuse: bool,
    /// Maximum number of root_outcomes to retain per decision. The
    /// HTML renderer shows the top 8 by default and tucks the rest
    /// behind a `<details>`; 32 is enough to capture the long tail
    /// without blowing up the artifact.
    pub max_root_outcomes_recorded: usize,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            sims: 50_000,
            c_puct: 1.0,
            rollout_depth: 20,
            seed: 1,
            tree_reuse: false,
            max_root_outcomes_recorded: DEFAULT_TOP_K_RECORDED,
        }
    }
}

impl RecorderConfig {
    fn build_mcts_config(&self) -> MctsConfig {
        MctsConfig {
            c_puct: self.c_puct,
            max_simulations: self.sims,
            max_rollout_depth: self.rollout_depth,
            seed: self.seed,
            rollout_policy: RolloutPolicyKind::Heuristic,
            ..MctsConfig::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct CardInHand {
    pub hand_index: u8,
    pub card_id: CardId,
    pub upgraded: bool,
    pub cost: i8,
    pub card_type: CardType,
}

#[derive(Debug, Clone)]
pub struct MonsterSnapshot {
    pub slot: usize,
    pub kind: Option<MonsterKind>,
    pub display_name: String,
    pub hp: i32,
    pub max_hp: i32,
    pub block: i32,
    pub alive: bool,
    pub intent: Option<Intent>,
    pub powers: Vec<(PowerKind, i32)>,
}

#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub turn: u16,
    pub phase: CombatPhase,
    pub player_hp: i32,
    pub player_max_hp: i32,
    pub player_block: i32,
    pub player_energy: i32,
    pub player_max_energy: i8,
    pub draw_n: usize,
    pub discard_n: usize,
    pub exhaust_n: usize,
    pub hand: Vec<CardInHand>,
    pub monsters: Vec<MonsterSnapshot>,
    pub player_powers: Vec<(PowerKind, i32)>,
    pub potions_held: u8,
}

#[derive(Debug, Clone)]
pub struct RootOutcomeRow {
    pub action: Action,
    pub action_label: String,
    pub visits: u32,
    pub wins: u32,
    pub losses: u32,
    pub ties: u32,
    pub mean_value: f32,
}

#[derive(Debug, Clone)]
pub struct DecisionRecord {
    pub decision_index: u32,
    pub pre: StateSnapshot,
    pub root_outcomes: Vec<RootOutcomeRow>,
    pub total_visits: u32,
    pub recommended_idx: Option<usize>,
    pub recommended_label: String,
    pub pv: Vec<String>,
    pub player_hp_after: i32,
    pub monster_hp_after: Vec<(usize, i32)>,
    pub step_result: String,
}

#[derive(Debug, Clone)]
pub struct FightRecord {
    pub fixture_path: PathBuf,
    pub fixture_stem: String,
    pub encounter: Option<String>,
    pub floor: Option<u32>,
    pub player_outcome_label: Option<String>,
    pub config: RecorderConfig,
    pub decisions: Vec<DecisionRecord>,
    pub final_won: bool,
    pub final_round: u16,
    pub final_player_hp: i32,
    pub final_player_max_hp: i32,
    pub aborted_reason: Option<String>,
}

pub fn record_from_fixture(path: &Path, cfg: &RecorderConfig) -> io::Result<FightRecord> {
    let scenario = load_scenario_file(path).map_err(io::Error::other)?;
    let outcome_label = scenario
        .provenance
        .as_ref()
        .and_then(|p| p.player_outcome.clone());
    let floor = scenario.provenance.as_ref().and_then(|p| p.floor);
    let encounter = scenario
        .provenance
        .as_ref()
        .and_then(|p| p.encounter.clone());
    let state = scenario_to_combat_state(&scenario, cfg.seed)
        .map_err(|e| io::Error::other(format!("scenario_to_combat_state: {e}")))?;
    let fixture_stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("fixture")
        .to_string();
    Ok(drive_combat(
        path.to_path_buf(),
        fixture_stem,
        encounter,
        floor,
        outcome_label,
        state,
        &scenario,
        cfg,
    ))
}

#[allow(clippy::too_many_arguments)]
fn drive_combat(
    fixture_path: PathBuf,
    fixture_stem: String,
    encounter: Option<String>,
    floor: Option<u32>,
    player_outcome_label: Option<String>,
    mut state: CombatState,
    _scenario: &ScenarioFile,
    cfg: &RecorderConfig,
) -> FightRecord {
    let _ = combat_loop::start_combat(&mut state);
    let _ = combat_loop::start_player_turn(&mut state);

    let mut decisions: Vec<DecisionRecord> = Vec::new();
    let mut decision_index: u32 = 0;
    let mut last_round = state.round;
    let mut round_action_count: u32 = 0;
    let mut aborted_reason: Option<String> = None;

    // Persistent tree, only consulted when cfg.tree_reuse is set.
    let mut persistent: Option<Mcts> = None;
    let mut pending_action: Option<Action> = None;

    loop {
        if state.is_over || is_terminal(&state) {
            break;
        }
        if state.round != last_round {
            last_round = state.round;
            round_action_count = 0;
        }
        if state.round > MAX_COMBAT_ROUNDS {
            aborted_reason = Some(format!(
                "exceeded {MAX_COMBAT_ROUNDS} rounds — forcing loss"
            ));
            state.is_over = true;
            state.player_won = false;
            break;
        }
        if round_action_count >= PER_ROUND_ACTION_CAP {
            // Inject EndTurn to break out of a no-progress cycle.
            let res = apply(&mut state, Action::EndTurn);
            propagate_terminal(&mut state, res);
            round_action_count = 0;
            continue;
        }

        let is_decision_phase = state.phase == CombatPhase::PlayerPlayPhase
            || (state.phase == CombatPhase::PlayerTurnStart && state.pending_choice.is_some());
        if !is_decision_phase {
            // Engine left us in an intermediate phase that the public
            // play loop also bails out of. Capture the reason and
            // exit cleanly.
            aborted_reason = Some(format!("engine paused in phase {:?}", state.phase));
            break;
        }

        let actions = legal_actions(&state);
        if actions.is_empty() {
            aborted_reason = Some("no legal actions".to_string());
            break;
        }

        let pre = snapshot_state(&state);

        // Build / reuse the search tree.
        let mcts_config = cfg.build_mcts_config();
        let mut mcts: Mcts = if cfg.tree_reuse {
            match (persistent.take(), pending_action.take()) {
                (Some(mut prev), Some(action)) => {
                    if prev.reroot(&action, state.clone()) {
                        prev
                    } else {
                        Mcts::new(state.clone(), mcts_config)
                    }
                }
                _ => Mcts::new(state.clone(), mcts_config),
            }
        } else {
            Mcts::new(state.clone(), mcts_config)
        };
        let _ = mcts.search_n(cfg.sims);

        let mut outcomes = mcts.root_outcomes();
        // Sort visits desc (the renderer-facing canonical order). Q
        // ranking is shown in the table as a separate column.
        outcomes.sort_by(|a, b| {
            b.visits.cmp(&a.visits).then(
                b.mean_value
                    .partial_cmp(&a.mean_value)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        // The engine's own pick (`Mcts::explain_root_pick`: the GH #32
        // lost-position override, proven wins, the tie band and its rules),
        // so a replay record recommends what the run agent plays rather
        // than a Q-then-visits re-rank of its own (GH #1022).
        let recommended_action = mcts.explain_root_pick().ok().map(|p| p.action);
        // Bead `sts2-rs-h9qge`: format each PV step against a state
        // advanced by all prior PV steps, not the outer decision state.
        // `format_action` reads `state.player.hand[hand_index]` to label
        // the card; if every step uses the unadvanced outer state, a PV
        // chain like `PlayCard{h0=Inflame} -> PlayCard{h0=Breakthrough}`
        // mislabels both as "Play Inflame (h0)" — the engine-level PV
        // is correct, but the rendered string reads as if a single
        // Power card was played twice (observed in
        // `target/vantom-lost-replays/*_vantom_boss.html`).
        let mcts_pv = mcts.get_pv();
        let pv = {
            let mut pv_state = state.clone();
            let mut out = Vec::with_capacity(mcts_pv.len());
            for a in &mcts_pv {
                out.push(format_action(&pv_state, a));
                let _ = apply(&mut pv_state, a.clone());
            }
            out
        };

        let truncated: Vec<RootOutcomeRow> = outcomes
            .iter()
            .take(cfg.max_root_outcomes_recorded)
            .map(|o| RootOutcomeRow {
                action: o.action.clone(),
                action_label: format_action(&state, &o.action),
                visits: o.visits,
                wins: o.wins,
                losses: o.losses,
                ties: o.ties,
                mean_value: o.mean_value,
            })
            .collect();
        let total_visits = outcomes.iter().map(|o| o.visits).sum::<u32>();
        let recommended_label = recommended_action
            .as_ref()
            .map_or_else(|| "(none)".to_string(), |a| format_action(&state, a));
        let recommended_idx = recommended_action
            .as_ref()
            .and_then(|rec| truncated.iter().position(|row| row.action == *rec));

        // Apply.
        let Some(rec) = recommended_action.clone() else {
            aborted_reason = Some("MCTS produced no recommended action".to_string());
            break;
        };
        let step = apply(&mut state, rec.clone());
        round_action_count += 1;
        if cfg.tree_reuse {
            persistent = Some(mcts);
            pending_action = Some(rec);
        }
        propagate_terminal(&mut state, step);

        let player_hp_after = state.player.current_hp;
        let monster_hp_after = state
            .enemy_uids
            .iter()
            .enumerate()
            .filter_map(|(slot, uid)| {
                state
                    .creatures
                    .iter()
                    .find(|c| c.id == uid.0)
                    .map(|c| (slot, c.current_hp))
            })
            .collect();

        decisions.push(DecisionRecord {
            decision_index,
            pre,
            root_outcomes: truncated,
            total_visits,
            recommended_idx,
            recommended_label,
            pv,
            player_hp_after,
            monster_hp_after,
            step_result: format!("{step:?}"),
        });
        decision_index += 1;
    }

    FightRecord {
        fixture_path,
        fixture_stem,
        encounter,
        floor,
        player_outcome_label,
        config: cfg.clone(),
        final_won: state.player_won,
        final_round: state.round,
        final_player_hp: state.player.current_hp,
        final_player_max_hp: state.player.max_hp,
        aborted_reason,
        decisions,
    }
}

fn propagate_terminal(state: &mut CombatState, step: StepResult) {
    match step {
        StepResult::CombatWon => {
            state.is_over = true;
            state.player_won = true;
        }
        StepResult::CombatLost => {
            state.is_over = true;
            state.player_won = false;
        }
        StepResult::Advanced
        | StepResult::PendingChoice
        | StepResult::InvalidTarget
        | StepResult::IllegalPlay => {}
    }
}

fn snapshot_state(state: &CombatState) -> StateSnapshot {
    let monsters = state
        .enemy_uids
        .iter()
        .enumerate()
        .filter_map(|(slot, uid)| {
            state
                .creatures
                .iter()
                .find(|c| c.id == uid.0)
                .map(|c| snapshot_monster(state, slot, c))
        })
        .collect();
    let hand = snapshot_hand(state);
    let player_powers = power_pairs(state, &state.player.powers);
    StateSnapshot {
        turn: state.round,
        phase: state.phase,
        player_hp: state.player.current_hp,
        player_max_hp: state.player.max_hp,
        player_block: state.player.block,
        player_energy: state.player.energy,
        player_max_energy: state.player.max_energy,
        draw_n: state.player.draw.len(),
        discard_n: state.player.discard.len(),
        exhaust_n: state.player.exhaust.len(),
        hand,
        monsters,
        player_powers,
        potions_held: u8::try_from(state.player.potions.len()).unwrap_or(u8::MAX),
    }
}

fn snapshot_hand(state: &CombatState) -> Vec<CardInHand> {
    state
        .player
        .hand
        .iter()
        .enumerate()
        .filter_map(|(i, uid)| {
            let card = state.cards.iter().find(|c| c.uid == *uid)?;
            Some(CardInHand {
                hand_index: u8::try_from(i).unwrap_or(u8::MAX),
                card_id: card.card_id,
                upgraded: card.upgraded,
                cost: i8::try_from(card.local_cost()).unwrap_or(card.base_cost()),
                card_type: card.card_type,
            })
        })
        .collect()
}

fn snapshot_monster(state: &CombatState, slot: usize, c: &Creature) -> MonsterSnapshot {
    let display_name = c
        .species
        .map_or_else(|| c.monster_id.0.to_string(), |k| format!("{k:?}"));
    MonsterSnapshot {
        slot,
        kind: c.species,
        display_name,
        hp: c.current_hp,
        max_hp: c.max_hp,
        block: c.block,
        alive: c.current_hp > 0,
        intent: c.intent,
        powers: power_pairs(state, &c.powers),
    }
}

fn power_pairs(
    state: &CombatState,
    uids: &SmallVec<[sts2_core::PowerUid; 8]>,
) -> Vec<(PowerKind, i32)> {
    uids.iter()
        .filter_map(|puid| {
            state
                .powers
                .iter()
                .find(|p| p.id == puid.0)
                .map(|p| (p.kind, p.amount))
        })
        .collect()
}

pub fn format_action(state: &CombatState, action: &Action) -> String {
    match action {
        Action::PlayCard { hand_index, target } => {
            let card_name = card_name_at(state, *hand_index);
            target.as_ref().map_or_else(
                || format!("Play {card_name} (h{hand_index})"),
                |t| format!("Play {card_name} (h{hand_index}) → enemy {t}"),
            )
        }
        Action::PlayCardAtAlly { hand_index, ally } => {
            let card_name = card_name_at(state, *hand_index);
            format!("Play {card_name} (h{hand_index}) → ally {ally}")
        }
        Action::DrinkPotionAtAlly { potion_slot, ally } => {
            let name = potion_name_at(state, *potion_slot);
            format!("Throw {name} (slot {potion_slot}) → ally {ally}")
        }
        Action::EndTurn => "End turn".to_string(),
        Action::UsePotion { slot, target } => {
            let name = potion_name_at(state, *slot);
            target.as_ref().map_or_else(
                || format!("Use {name} (slot {slot})"),
                |t| format!("Use {name} (slot {slot}) → enemy {t}"),
            )
        }
        Action::DrinkPotion {
            potion_slot,
            target,
        } => {
            let name = potion_name_at(state, *potion_slot);
            target.as_ref().map_or_else(
                || format!("Drink {name} (slot {potion_slot})"),
                |t| format!("Drink {name} (slot {potion_slot}) → enemy {t}"),
            )
        }
        Action::Channel { orb, target } => target.as_ref().map_or_else(
            || format!("Channel {orb:?}"),
            |t| format!("Channel {orb:?} → enemy {t}"),
        ),
        Action::Evoke { times } => format!("Evoke ×{times}"),
        Action::ResolveChoice { selection } => format!("Resolve {selection:?}"),
    }
}

fn card_name_at(state: &CombatState, hand_index: u8) -> String {
    let idx = usize::from(hand_index);
    let Some(uid) = state.player.hand.get(idx).copied() else {
        return format!("h{hand_index}?");
    };
    let Some(card) = state.cards.iter().find(|c| c.uid == uid) else {
        return format!("h{hand_index}?");
    };
    let plus = if card.upgraded { "+" } else { "" };
    format!("{:?}{plus}", card.card_id)
}

fn potion_name_at(state: &CombatState, slot: u8) -> String {
    state
        .player
        .potions
        .get(usize::from(slot))
        .map_or_else(|| format!("potion?{slot}"), |k| k.python_name().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;
    use sts2_core::cards::Card;
    use sts2_core::CardUid;

    /// Bead `sts2-rs-h9qge` regression: when the PV chain contains two
    /// `PlayCard { hand_index: 0, .. }` steps and the underlying cards
    /// differ (step 1 left the hand), the rendered labels must reflect
    /// the post-apply hand of each step. Formatting every step against
    /// the unadvanced outer state mislabels step 2 with step 1's name
    /// (observed as "Play Inflame → Play Inflame" in
    /// `target/vantom-lost-replays/*_vantom_boss.html`).
    ///
    /// We mutate the hand manually rather than calling engine `apply`
    /// to keep the test focused on the rendering invariant; the fix
    /// in `drive_combat` threads `apply` between calls so the same
    /// property holds end to end.
    #[test]
    fn pv_labels_track_hand_after_each_play() {
        let inflame_uid = CardUid::new(1).expect("non-zero");
        let strike_uid = CardUid::new(2).expect("non-zero");
        let mut state = CombatState::default();
        state.cards.push(Card {
            uid: inflame_uid,
            card_id: CardId::Inflame,
            card_type: CardType::Power,
            ..Card::pod_defaults()
        });
        state.cards.push(Card {
            uid: strike_uid,
            card_id: CardId::StrikeIronclad,
            ..Card::pod_defaults()
        });
        state.player.hand = smallvec![inflame_uid, strike_uid];

        let pv = vec![
            Action::PlayCard {
                hand_index: 0,
                target: None,
            },
            Action::PlayCard {
                hand_index: 0,
                target: None,
            },
        ];

        // Old (buggy) behaviour: formatting both steps against the
        // outer state produces duplicate labels.
        let stale: Vec<String> = pv.iter().map(|a| format_action(&state, a)).collect();
        assert_eq!(
            stale[0], stale[1],
            "sanity: outer-state formatting yields duplicate labels",
        );

        // New (fixed) behaviour: advance the hand between steps. The
        // Power card leaves the hand on play, so step 2 sees Strike
        // at h0.
        let mut advanced: Vec<String> = Vec::with_capacity(pv.len());
        let mut pv_state = state.clone();
        for a in &pv {
            advanced.push(format_action(&pv_state, a));
            if let Some((hand_index, _)) = a.card_play() {
                let idx = usize::from(hand_index);
                if idx < pv_state.player.hand.len() {
                    pv_state.player.hand.remove(idx);
                }
            }
        }
        assert_ne!(
            advanced[0], advanced[1],
            "advanced-state PV labels must differ when the cards differ — got {advanced:?}",
        );
        assert!(
            advanced[0].contains("Inflame"),
            "step 0 label: {}",
            advanced[0]
        );
        assert!(
            advanced[1].contains("Strike"),
            "step 1 label: {}",
            advanced[1]
        );
    }

    /// `format_action` labels every non-PlayCard arm, and the card/potion
    /// name helpers fall back gracefully on out-of-range indices.
    #[test]
    fn format_action_labels_all_arms_and_oob_fallbacks() {
        let state = CombatState::default();
        assert_eq!(format_action(&state, &Action::EndTurn), "End turn");
        assert!(format_action(
            &state,
            &Action::UsePotion {
                slot: 0,
                target: None
            }
        )
        .starts_with("Use"));
        assert!(format_action(
            &state,
            &Action::DrinkPotion {
                potion_slot: 1,
                target: Some(2)
            }
        )
        .contains("→ enemy 2"));
        assert!(format_action(
            &state,
            &Action::Channel {
                orb: sts2_core::orbs::Orb::Lightning,
                target: None
            }
        )
        .starts_with("Channel"));
        assert_eq!(
            format_action(&state, &Action::Evoke { times: 2 }),
            "Evoke ×2"
        );
        assert!(format_action(
            &state,
            &Action::ResolveChoice {
                selection: smallvec![1u8]
            }
        )
        .starts_with("Resolve"));
        // PlayCard with an out-of-range hand index hits the `h{n}?` path.
        let oob = format_action(
            &state,
            &Action::PlayCard {
                hand_index: 9,
                target: None,
            },
        );
        assert!(oob.contains("h9?"), "expected oob marker, got {oob}");
    }

    /// End-to-end recorder smoke: load a minimal Nibbit scenario from a
    /// temp toml and drive the combat with a tiny search budget. Asserts
    /// the record is well-formed (terminal, per-decision snapshots), and
    /// that rendering it produces a self-contained HTML page. Low sims
    /// keep this fast while still exercising `drive_combat`'s full loop.
    #[test]
    fn record_from_fixture_drives_and_renders() {
        use std::io::Write as _;

        let toml = "schema = 1\n\
                    [player]\nclass = \"Ironclad\"\nhp = 80\n\
                    [[monsters]]\nkind = \"Nibbit\"\n";
        let path = std::env::temp_dir().join(format!(
            "sts2rec_{}_{}.toml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(toml.as_bytes())
            .unwrap();

        let cfg = RecorderConfig {
            sims: 16,
            rollout_depth: 4,
            seed: 1,
            ..RecorderConfig::default()
        };
        let record = record_from_fixture(&path, &cfg).expect("record");
        let _ = std::fs::remove_file(&path);

        // The combat must terminate cleanly, not abort on a stall.
        assert!(
            record.aborted_reason.is_none(),
            "unexpected abort: {:?}",
            record.aborted_reason
        );
        assert!(
            !record.decisions.is_empty(),
            "recorder captured no decisions"
        );
        // Each decision carries a search table and a pre-state snapshot.
        for d in &record.decisions {
            assert!(!d.root_outcomes.is_empty(), "decision had no root outcomes");
            assert!(!d.pre.monsters.is_empty(), "snapshot had no monsters");
        }
        // The record renders to a complete HTML document.
        let html = crate::replay::render::render_fight_html(&record);
        assert!(html.starts_with("<!doctype html>") && html.ends_with("</body></html>"));
    }
}
