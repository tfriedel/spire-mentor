//! The coach for run-level choices (map, card reward, shop, rest, event, …).
//!
//! Only a policy distribution exists for these decisions, so the grade is
//! policy AGREEMENT — never a win-probability cost (the PPO probabilities
//! and critic are not win rates). `Workbench::live_action` snapshots a
//! `RunManager` clone of the position before a human choice
//! ([`Session::capture_choice`]); the grade and the on-demand rollout check
//! evaluate the policy on that snapshot, so neither replays the tree.

use serde_json::{json, Value};
use sts2_core::run_manager::{RunAction, RunManager};
use sts2_wire::emit;

use crate::advisors::{choice_check, native_action, Progress};
use crate::localization::resolve_state_names;
use crate::ppo;
use crate::render::describe_action;
use crate::runmode;
use crate::session::{Session, SessionError};

/// The top pick must carry at least this much probability to flag a choice.
pub const FLAG_TOP_MIN: f64 = 0.40;
/// ...and at least this multiple of the pick's own probability.
pub const FLAG_RATIO: f64 = 2.0;
/// A top pick at or above this is a "strong preference".
pub const STRONG_PREFERENCE: f64 = 0.70;

/// One policy option at the snapshotted decision.
#[derive(Debug, Clone)]
pub struct ChoiceOption {
    pub action: String,
    pub args: Value,
    pub label: String,
    pub prob: f64,
    pub native: RunAction,
}

/// The position a human choice was made from, captured before it applied.
/// The policy is evaluated lazily from `rm` (pure on the clone), so the
/// capture costs one clone when nobody asks for a grade.
#[derive(Debug, Clone)]
pub struct ChoiceSnapshot {
    pub parent_id: u64,
    pub decision: String,
    pub rm: RunManager,
    /// The e2e advisor's evaluation of the position (`Session::e2e_advice`)
    /// when the student advised the run: the grade is agreement with IT.
    pub e2e: Option<Value>,
}

/// [`grade_choice`]'s result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChoiceVerdict {
    /// `"agree"` | `"tossup"` | `"flag"`.
    pub verdict: &'static str,
    /// The pick's 1-based rank by probability.
    pub rank: usize,
    /// Index of the policy's top pick.
    pub top: usize,
    /// `"strong preference"` | `"preference"` — how sure the policy is of its top.
    pub strength: &'static str,
}

/// Grade pick `played` among `probs`: agreement when it is the top pick; a
/// flag only when the top pick is ≥ [`FLAG_TOP_MIN`] and ≥ [`FLAG_RATIO`]×
/// the pick's probability; otherwise a toss-up.
///
/// Twin: `sts2-pycli/python/choice_grade.py::grade_choice` (the live mod
/// coach). Keep the constants, tie-breaks and the unit cases below in step.
#[must_use]
pub fn grade_choice(probs: &[f64], played: usize) -> ChoiceVerdict {
    let top = probs
        .iter()
        .enumerate()
        .fold(0, |b, (i, p)| if *p > probs[b] { i } else { b });
    let (p_top, p_pick) = (probs[top], probs[played]);
    let rank = 1 + probs.iter().filter(|p| **p > p_pick).count();
    let strength = if p_top >= STRONG_PREFERENCE {
        "strong preference"
    } else {
        "preference"
    };
    let verdict = if played == top || p_pick >= p_top {
        "agree"
    } else if p_top >= FLAG_TOP_MIN && p_top >= FLAG_RATIO * p_pick {
        "flag"
    } else {
        "tossup"
    };
    ChoiceVerdict {
        verdict,
        rank,
        top,
        strength,
    }
}

/// The chip's name for a run-level decision; `None` = not graded here.
#[must_use]
pub fn choice_kind(decision: &str) -> Option<&'static str> {
    Some(match decision {
        "map_select" => "Map",
        "card_reward" => "Card reward",
        "bundle_select" => "Card bundle",
        "shop" => "Shop",
        "rest_site" => "Rest site",
        "event_choice" => "Event",
        "card_select" => "Card select",
        _ => return None,
    })
}

fn option_index(options: &[ChoiceOption], action: &str, args: &Value) -> Option<usize> {
    let want = native_action(Some(action), args);
    options
        .iter()
        .position(|o| o.action == action && native_action(Some(&o.action), &o.args) == want)
}

/// The e2e advisor's options (`Session::e2e_advice`) as coach options;
/// a class with no engine action (a multi-pick selection) is left out.
fn e2e_choice_options(adv: &Value) -> Vec<ChoiceOption> {
    adv["options"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|o| {
            Some(ChoiceOption {
                action: o["action"].as_str()?.to_string(),
                args: o["args"].clone(),
                label: o["label"].as_str().unwrap_or("").to_string(),
                prob: o["prob"].as_f64()?,
                native: crate::rmx::action_from_value(&o["run_action"]).ok()?,
            })
        })
        .collect()
}

/// An advisor label without its merged-class suffix (`"Strike (×2)"`).
fn class_label(label: &str) -> &str {
    match label.rfind(" (×") {
        Some(i) if label.ends_with(')') => &label[..i],
        _ => label,
    }
}

impl Session {
    /// Snapshot the run at the active node before a human choice is
    /// applied. Clears the previous snapshot first; leaves none at a combat
    /// node or an ungraded decision.
    pub fn capture_choice(&mut self) {
        self.last_choice = None;
        let Some(node_id) = self.active_id else {
            return;
        };
        let node = &self.nodes[&node_id];
        if node.combat.is_some() || choice_kind(&node.decision).is_none() {
            return;
        }
        let decision = node.decision.clone();
        let Some(rm) = self.game.as_ref().and_then(|g| g.native_run()).cloned() else {
            return;
        };
        // Under the e2e advisor the coach grades against the student (its
        // evaluation is the per-node cache the panel already filled).
        #[cfg(feature = "e2e")]
        let e2e = if self.advisor_is_e2e() {
            self.e2e_advice()
                .ok()
                .filter(|a| a["available"] == json!(true) && a["forced"] != json!(true))
        } else {
            None
        };
        #[cfg(not(feature = "e2e"))]
        let e2e = None;
        self.last_choice = Some(ChoiceSnapshot {
            parent_id: node_id,
            decision,
            rm,
            e2e,
        });
    }

    /// The policy's options at the snapshotted position, labelled as the
    /// PPO panel labels them; `None` without a loaded net or with fewer
    /// than two scored options.
    fn choice_options(&self, snap: &ChoiceSnapshot) -> Option<Vec<ChoiceOption>> {
        let named = resolve_state_names(Some(&self.nodes.get(&snap.parent_id)?.state), self.loc);
        let mut rm = snap.rm.clone();
        let ev = ppo::evaluate(self.ppo.as_ref(), &self.net_error(), &mut rm)?;
        let probs = ev.probs.as_ref()?;
        let deck = if ev.surface == "shop" {
            emit::run_manager::master_deck_detailed(&rm)
        } else {
            Vec::new()
        };
        let resolve =
            |native: &RunAction| runmode::resolve_workbench_verb(&rm, native, &ev.full_actions);
        // `panel_value` drops the options it can't resolve to a verb; keep
        // the natives and probabilities in step with its order.
        let kept: Vec<(RunAction, f64)> = ev
            .actions
            .iter()
            .zip(probs)
            .filter(|(n, _)| resolve(n).is_some())
            .map(|(n, p)| (*n, f64::from(*p)))
            .collect();
        let panel = ppo::panel_value(&ev, &self.ppo_impl, &deck, resolve, |verb, args| {
            describe_action(
                Some(&named),
                Some(&json!({"cmd": "action", "action": verb, "args": args})),
            )
        });
        let options: Vec<ChoiceOption> = panel["options"]
            .as_array()
            .into_iter()
            .flatten()
            .zip(kept)
            .map(|(o, (native, prob))| ChoiceOption {
                action: o["action"].as_str().unwrap_or("").to_string(),
                args: o["args"].clone(),
                label: o["label"].as_str().unwrap_or("").to_string(),
                prob,
                native,
            })
            .collect();
        (options.len() >= 2).then_some(options)
    }

    /// The snapshot, its policy options and the played option index for
    /// graded node `node_id`; `None` when the node isn't the snapshot's
    /// child, there is no policy, or its move isn't one of the options.
    fn graded(&self, node_id: u64) -> Option<(&ChoiceSnapshot, Vec<ChoiceOption>, usize)> {
        let snap = self.last_choice.as_ref()?;
        let node = self.nodes.get(&node_id)?;
        if node.parent_id != Some(snap.parent_id) {
            return None;
        }
        if let Some(adv) = snap.e2e.as_ref() {
            let options = e2e_choice_options(adv);
            if options.len() < 2 {
                return None;
            }
            // A move the student merged into one class with others (a second
            // copy of the same card) is that class: match it by its label.
            let played =
                option_index(&options, node.action.as_deref()?, &node.args).or_else(|| {
                    options
                        .iter()
                        .position(|o| class_label(&o.label) == node.label)
                })?;
            return Some((snap, options, played));
        }
        let options = self.choice_options(snap)?;
        let played = option_index(&options, node.action.as_deref()?, &node.args)?;
        Some((snap, options, played))
    }

    /// Grade the run-level choice that produced the active node.
    pub fn last_choice_grade(&self) -> Result<Option<Value>, SessionError> {
        let node = self.active_node()?;
        let Some((snap, options, played)) = self.graded(node.id) else {
            return Ok(None);
        };
        let probs: Vec<f64> = options.iter().map(|o| o.prob).collect();
        let g = grade_choice(&probs, played);
        let mut ranked: Vec<&ChoiceOption> = options.iter().collect();
        ranked.sort_by(|a, b| {
            b.prob
                .partial_cmp(&a.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let (you, top) = (&options[played], &options[g.top]);
        let mut out = json!({
            "node_id": node.id,
            "parent_id": snap.parent_id,
            "decision": snap.decision,
            "kind": choice_kind(&snap.decision),
            "verdict": g.verdict,
            "strength": g.strength,
            "played": {"label": you.label, "prob": you.prob, "rank": g.rank},
            "top": {"label": top.label, "prob": top.prob},
            "alternatives": ranked.iter().take(3)
                .map(|o| json!({"label": o.label, "prob": o.prob}))
                .collect::<Vec<_>>(),
            "n_options": options.len(),
            // The rollout check plays the PPO / search; the student's grade
            // is agreement only.
            "checkable": g.verdict != "agree" && snap.e2e.is_none(),
        });
        // Named only for the student (a PPO grade keeps its bytes).
        if snap.e2e.is_some() {
            out["policy"] = json!("e2e");
        }
        Ok(Some(out))
    }
}

/// The on-demand rollout check of graded node `config.node_id`: your pick
/// vs the policy's top pick, paired rollouts to the end of the act from the
/// snapshotted position. `None` when there is nothing to compare.
pub fn last_choice_check(
    sess: &Session,
    config: &Value,
    progress: Option<Progress<'_>>,
) -> Result<Option<Value>, SessionError> {
    let Some(node_id) = config.get("node_id").and_then(Value::as_u64) else {
        return Err(SessionError::new(400, "node_id must be an integer"));
    };
    let Some((snap, options, played)) = sess.graded(node_id) else {
        return Ok(None);
    };
    let probs: Vec<f64> = options.iter().map(|o| o.prob).collect();
    let g = grade_choice(&probs, played);
    if g.verdict == "agree" {
        return Ok(None);
    }
    let (you, ai) = (&options[played], &options[g.top]);
    Ok(Some(choice_check(
        &snap.rm,
        (you.native, &you.label),
        (ai.native, &ai.label),
        config,
        progress,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grade_choice_agrees_when_pick_is_top() {
        let g = grade_choice(&[0.6, 0.4], 0);
        assert_eq!((g.verdict, g.rank, g.top), ("agree", 1, 0));
    }

    #[test]
    fn grade_choice_flags_clear_disagreement() {
        let g = grade_choice(&[0.5, 0.3, 0.2], 2);
        assert_eq!((g.verdict, g.rank, g.top), ("flag", 3, 0));
        assert_eq!(g.strength, "preference");
    }

    #[test]
    fn grade_choice_tossup_below_ratio() {
        assert_eq!(grade_choice(&[0.45, 0.35, 0.2], 1).verdict, "tossup");
    }

    #[test]
    fn grade_choice_tossup_below_top_floor() {
        assert_eq!(grade_choice(&[0.38, 0.12, 0.25, 0.25], 1).verdict, "tossup");
    }

    #[test]
    fn grade_choice_strong_preference() {
        let g = grade_choice(&[0.8, 0.1, 0.1], 1);
        assert_eq!((g.verdict, g.strength), ("flag", "strong preference"));
    }

    #[test]
    fn grade_choice_flags_at_the_exact_thresholds() {
        assert_eq!(grade_choice(&[0.4, 0.2, 0.4], 1).verdict, "flag");
    }

    #[test]
    fn grade_choice_tie_with_the_top_is_agreement() {
        assert_eq!(grade_choice(&[0.5, 0.5], 1).verdict, "agree");
    }

    #[test]
    fn choice_kind_covers_the_run_level_decisions() {
        assert_eq!(choice_kind("card_reward"), Some("Card reward"));
        assert_eq!(choice_kind("event_choice"), Some("Event"));
        assert_eq!(choice_kind("combat_play"), None);
        assert_eq!(choice_kind("game_over"), None);
    }
}
