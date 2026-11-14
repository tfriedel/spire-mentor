//! Per-surface ablation wrapper for the E2 credit-assignment experiment
//! (full-run-AI plan §3.5; bead `sts2-rs-40ikw`).
//!
//! E2 ranks where the meta-policy's quality actually matters. For each
//! decision surface C, the plan runs the panel three ways with all other
//! surfaces held at the current policy: (a) C → random, (b) C → current,
//! (c) C → oracle. This module implements the **(a) leg** — the
//! *importance* half, `importance(C) = (b) − (a)` — by wrapping a base
//! [`RunPolicy`] and routing only surface-C decisions to a uniform-random
//! pick, delegating every other surface to the base unchanged.
//!
//! The oracle leg (c) (force tier-2/3 on every C decision) is deferred:
//! it needs a forced-escalation path and tier-3 is currently inert
//! (boss-relic surface unmodelled — see the experiments doc), so the
//! importance leg is the decisive, cheap measurement first.
//!
//! Surfaces map to engine phases/actions exactly as
//! [`HeuristicPolicy`](super::heuristic::HeuristicPolicy) dispatches them,
//! so the ablation is faithful to what the base policy treats as one
//! decision class. Potion economy is not yet a separable surface (buy/hold
//! is spread across [`AblationSurface::Shop`] and card-reward screens); it
//! folds into those for now and graduates when the plan's potion surface
//! is split out.

use sts2_core::full_run::{RandomPolicy, RunPolicy};
use sts2_core::run_manager::{Phase, RunAction, RunManager};

use crate::policy::VScoreSource;

/// One ablatable meta-decision class (plan §3.5 surface list, minus the
/// not-yet-separable potion-economy surface).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AblationSurface {
    /// Map pathing (`Phase::MapChoice`).
    Path,
    /// Every card-shaped pick: card reward + skip, boss relic, deck
    /// remove/upgrade/transform/enchant, and smith target.
    CardPick,
    /// Event-option resolution (`Phase::Event`).
    Event,
    /// Shop purchase/leave (`Phase::Shop`).
    Shop,
    /// Rest-site option (rest vs heal vs channel-injected), excluding the
    /// smith sub-state which is a card pick.
    Rest,
}

impl AblationSurface {
    /// Stable CLI / label token.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            AblationSurface::Path => "path",
            AblationSurface::CardPick => "card-pick",
            AblationSurface::Event => "event",
            AblationSurface::Shop => "shop",
            AblationSurface::Rest => "rest",
        }
    }

    /// Parse a CLI token (`path|card-pick|event|shop|rest`).
    #[must_use]
    pub fn from_token(s: &str) -> Option<Self> {
        match s {
            "path" => Some(AblationSurface::Path),
            "card-pick" | "cardpick" | "picks" => Some(AblationSurface::CardPick),
            "event" => Some(AblationSurface::Event),
            "shop" => Some(AblationSurface::Shop),
            "rest" => Some(AblationSurface::Rest),
            _ => None,
        }
    }
}

/// Classify the decision in front of the policy into its ablation surface,
/// mirroring `HeuristicPolicy::choose`'s dispatch (deck-mutation + smith
/// preempt, then phase). `None` = a surface the experiment does not ablate
/// (combat, treasure, run-over, or an empty action list).
#[must_use]
pub fn classify(rm: &RunManager, actions: &[RunAction]) -> Option<AblationSurface> {
    // Smith target (offered at a rest site) is a card pick, not a rest
    // option — matches the heuristic's `choose_smith` preempt.
    if matches!(actions.first(), Some(RunAction::SmithCard { .. })) {
        return Some(AblationSurface::CardPick);
    }
    // Deck-mutation choices are phase-agnostic homogeneous lists.
    if !actions.is_empty()
        && actions.iter().all(|a| {
            matches!(
                a,
                RunAction::RemoveDeckCard { .. }
                    | RunAction::UpgradeDeckCard { .. }
                    | RunAction::TransformDeckCard { .. }
                    | RunAction::ConfirmDeckSelection
                    | RunAction::EnchantDeckCard { .. }
            )
        })
    {
        return Some(AblationSurface::CardPick);
    }
    match rm.phase() {
        Phase::MapChoice => Some(AblationSurface::Path),
        Phase::CardReward | Phase::BossRelic => Some(AblationSurface::CardPick),
        Phase::Event => Some(AblationSurface::Event),
        Phase::Shop => Some(AblationSurface::Shop),
        Phase::RestSite => Some(AblationSurface::Rest),
        Phase::Combat | Phase::Treasure | Phase::RunOver => None,
    }
}

/// Wraps a base policy and routes the configured surfaces to a uniform
/// random pick, leaving every other surface to the base unchanged.
#[derive(Debug)]
pub struct AblationPolicy<M> {
    base: M,
    rng: RandomPolicy,
    targets: Vec<AblationSurface>,
    /// True when the decision just made was randomized — suppresses the
    /// base's stale V-scores in the audit log for that record.
    last_randomized: bool,
}

impl<M> AblationPolicy<M> {
    /// Wrap `base`, ablating `targets` (the policy RNG is seeded off the
    /// run seed the same way [`RandomPolicy::from_run_seed`] does, so a
    /// cell is byte-reproducible).
    #[must_use]
    pub fn new(base: M, targets: Vec<AblationSurface>, run_seed: i32) -> Self {
        Self {
            base,
            rng: RandomPolicy::from_run_seed(run_seed),
            targets,
            last_randomized: false,
        }
    }

    /// Borrow the wrapped base policy (for typed access to e.g. escalation
    /// counters when the base is a `TieredPolicy`).
    #[must_use]
    pub fn base(&self) -> &M {
        &self.base
    }

    fn is_target(&self, s: AblationSurface) -> bool {
        self.targets.contains(&s)
    }
}

impl<M: RunPolicy> RunPolicy for AblationPolicy<M> {
    fn prepare(&mut self, rm: &mut RunManager) {
        self.base.prepare(rm);
    }

    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        match classify(rm, actions) {
            Some(s) if self.is_target(s) => {
                // Pick uniformly at random on the ablated surface. We do
                // NOT drive the base here: `choose` is pure over `&rm`
                // (run RNG only advances via the harness's `take_action`),
                // so skipping it changes nothing downstream — and avoids
                // burning fight_delta probes if the base is a TieredPolicy.
                self.last_randomized = true;
                self.rng.choose(rm, actions)
            }
            _ => {
                self.last_randomized = false;
                self.base.choose(rm, actions)
            }
        }
    }
}

impl<M: VScoreSource> VScoreSource for AblationPolicy<M> {
    fn take_v_scores(&mut self) -> Option<(Vec<f64>, f64)> {
        // A randomized decision has no meaningful V-scores; still drain the
        // base's buffered scores (it was driven) so they don't bleed into
        // the next record.
        let base_scores = self.base.take_v_scores();
        if self.last_randomized {
            None
        } else {
            base_scores
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::HeuristicPolicy;
    use sts2_content::run_callbacks::new_run_manager_for_character;
    use sts2_core::potions::CharacterId;

    #[test]
    fn classify_maps_phases_to_surfaces() {
        // A fresh run manager opens on the Neow event (Phase::Event after
        // `.with_neow_start()`), or MapChoice otherwise — assert the
        // classifier returns *some* expected surface and never panics over
        // a live action list.
        let rm = new_run_manager_for_character(0, CharacterId::Ironclad);
        let actions = rm.get_available_actions();
        let s = classify(&rm, &actions);
        // Opening decision is a real ablatable surface (path or event),
        // never combat/none.
        assert!(
            matches!(s, Some(AblationSurface::Path | AblationSurface::Event)),
            "opening surface = {s:?}"
        );
    }

    #[test]
    fn token_roundtrip() {
        for s in [
            AblationSurface::Path,
            AblationSurface::CardPick,
            AblationSurface::Event,
            AblationSurface::Shop,
            AblationSurface::Rest,
        ] {
            assert_eq!(AblationSurface::from_token(s.token()), Some(s));
        }
    }

    #[test]
    fn non_target_surface_delegates_to_base() {
        // With an empty target set the wrapper must be a pass-through:
        // every pick equals the base's pick on the same manager.
        let seed = 12345;
        let mut base = HeuristicPolicy::new();
        let mut wrapped = AblationPolicy::new(HeuristicPolicy::new(), Vec::new(), seed);
        let rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let actions = rm.get_available_actions();
        if !actions.is_empty() {
            assert_eq!(
                wrapped.choose(&rm, &actions),
                base.choose(&rm, &actions),
                "empty-target ablation must pass through to the base"
            );
        }
    }
}
