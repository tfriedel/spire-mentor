//! `SampledDeckPolicy` — card-reward sampling for the deckbuilding oracle
//! (deck-oracle task 1.3).
//!
//! Card-reward decisions are sampled via a softmax over `[dv_add_card(offer_i)
//! + bias_strength * archetype_match(offer_i) ; skip_bias]` using the
//! per-`(run_seed, sample_seed)` [`SampleParams`] and [`policy_rng`].
//!
//! Every other run phase is delegated to an embedded [`HeuristicPolicy`] so
//! this task is independently testable before other surfaces are sampled.

use rand::rngs::StdRng;
use sts2_core::full_run::RunPolicy;
use sts2_core::run_manager::{Phase, RunAction, RunManager};

use crate::deck_archetype::card_matches_archetype;
use crate::eval::{HeuristicV, Weights};
use crate::policy::heuristic::HeuristicPolicy;
use crate::policy::sampling::{policy_rng, softmax_sample, SampleParams};
use crate::policy::surfaces::{picks, shop};
use crate::policy::{sampling, VScoreSource};
use crate::view::VisibleRunView;

/// Run policy that samples card-reward picks and delegates everything else
/// to an embedded [`HeuristicPolicy`].
///
/// Scores per offer: `dv_add_card(offer_i) + params.bias_strength * archetype_match(offer_i)`.
/// Skip option score: `params.skip_bias` (offset relative to 0).
/// Sampling: softmax at `params.temperature` via [`softmax_sample`].
pub struct SampledDeckPolicy {
    v: HeuristicV,
    pub params: SampleParams,
    rng: StdRng,
    inner: HeuristicPolicy,
    /// `(per-option scores, top-2 margin)` from the last card-reward decision —
    /// consumed by [`VScoreSource::take_v_scores`].
    last_scores: Option<(Vec<f64>, f64)>,
}

impl std::fmt::Debug for SampledDeckPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SampledDeckPolicy")
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

impl SampledDeckPolicy {
    /// Construct from `weights`, seeded deterministically by
    /// `(run_seed, sample_seed)`.
    ///
    /// - `params = sample_params_for(run_seed, sample_seed)`
    /// - `rng    = policy_rng(run_seed, sample_seed)`
    /// - `v` and `inner` share the same `weights`.
    #[must_use]
    pub fn new(weights: Weights, run_seed: i32, sample_seed: u64) -> Self {
        let params = sampling::sample_params_for(run_seed, sample_seed);
        let rng = policy_rng(run_seed, sample_seed);
        let v = HeuristicV::new(weights.clone());
        let inner = HeuristicPolicy::with_weights(weights);
        Self {
            v,
            params,
            rng,
            inner,
            last_scores: None,
        }
    }

    /// Record per-action scores (scores aligned to actions + skip-last)
    /// and compute the top-1 − top-2 margin for the audit log.
    ///
    /// Mirrors [`HeuristicPolicy::set_scores`] (heuristic.rs:70-78).
    fn set_scores(&mut self, scores: Vec<f64>) {
        let mut sorted: Vec<f64> = scores.iter().copied().filter(|s| s.is_finite()).collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = match sorted.as_slice() {
            [first, second, ..] => first - second,
            _ => 0.0,
        };
        self.last_scores = Some((scores, margin));
    }

    /// Index of `wanted` in `actions`, falling back to 0.
    ///
    /// Mirrors [`HeuristicPolicy::position_of`] (heuristic.rs:81-83).
    fn position_of(actions: &[RunAction], wanted: &RunAction) -> usize {
        actions.iter().position(|a| a == wanted).unwrap_or(0)
    }

    /// Sample a card-reward pick. Returns the index into `actions`.
    fn choose_card_reward(&mut self, view: &VisibleRunView, actions: &[RunAction]) -> usize {
        let offers = &view.offers.card_rewards;

        // Per-offer scores: dv_add_card + archetype bias.
        let mut scores: Vec<f64> = offers
            .iter()
            .map(|offer| {
                let dv = picks::dv_add_card(view, &self.v, *offer);
                let bias =
                    if card_matches_archetype(offer.card_id, offer.upgraded, self.params.archetype)
                    {
                        self.params.bias_strength
                    } else {
                        0.0
                    };
                dv + bias
            })
            .collect();

        // Skip option: score = skip_bias (0.0 baseline + skip_bias).
        scores.push(self.params.skip_bias);

        // Record for audit log.
        self.set_scores(scores.clone());

        // Softmax sample.
        let pos = softmax_sample(&scores, self.params.temperature, &mut self.rng);

        // Map sampled index to action: pos < offers.len() → pick offer,
        // else → skip (idx: None).
        let pick = if pos < offers.len() { Some(pos) } else { None };
        Self::position_of(actions, &RunAction::PickCardReward { idx: pick })
    }

    /// Sample a map move (`Phase::MapChoice`). Mirrors the heuristic's
    /// action-aligned `scores` vec (heuristic.rs:213-230) — each
    /// `Move { col }` carries its `scored_first_cols` score, everything else
    /// `NEG_INFINITY` — then softmax-samples that already-action-aligned vec
    /// and returns the index directly.
    fn sample_path(&mut self, view: &VisibleRunView, actions: &[RunAction]) -> usize {
        use crate::policy::surfaces::pathing;
        let Some((_, cols)) = pathing::scored_first_cols(view, &self.v) else {
            return 0;
        };
        let scores: Vec<f64> = actions
            .iter()
            .map(|a| match a {
                RunAction::Move { col } => cols
                    .iter()
                    .find(|(c, _)| c == col)
                    .map_or(f64::NEG_INFINITY, |(_, s)| *s),
                _ => f64::NEG_INFINITY,
            })
            .collect();
        self.set_scores(scores.clone());
        softmax_sample(&scores, self.params.temperature, &mut self.rng)
    }

    /// Sample the deck-mutation preempt for the surfaces this policy owns
    /// (`RemoveDeckCard` / `UpgradeDeckCard`); other preempts return `None`
    /// so the caller delegates to the embedded heuristic. Mirrors
    /// [`HeuristicPolicy::choose_deck_mutation`] (heuristic.rs:87-128) but
    /// softmax-samples the eligible-position rather than taking the argmax.
    fn sample_deck_mutation(
        &mut self,
        rm: &RunManager,
        view: &VisibleRunView,
        actions: &[RunAction],
    ) -> Option<usize> {
        match actions.first()? {
            RunAction::RemoveDeckCard { .. } => {
                // `RemoveDeckCard { idx }` indexes the engine's removal offer
                // order; `best_removal`'s returned eligible-position is exactly
                // that action index.
                let eligible = rm.run_state.removal_offer_order();
                let (_, _, scores) = picks::best_removal(view, &self.v, &eligible)?;
                self.set_scores(scores.clone());
                let pos = softmax_sample(&scores, self.params.temperature, &mut self.rng);
                Some(pos)
            }
            RunAction::UpgradeDeckCard { .. } => {
                // Index space = position within the engine's `is_upgradable()`
                // subset (Curse/Status/Quest cards are NOT offered) — see
                // `picks::upgradable_deck_indices` and the matching note in
                // `HeuristicPolicy::choose_deck_mutation`.
                let eligible = picks::upgradable_deck_indices(rm);
                let (_, _, scores) = picks::best_upgrade(view, &self.v, &eligible)?;
                self.set_scores(scores.clone());
                let pos = softmax_sample(&scores, self.params.temperature, &mut self.rng);
                Some(pos.min(actions.len().saturating_sub(1)))
            }
            // Transform / Enchant preempts are not sampled here.
            _ => None,
        }
    }

    /// Sample the smith target (`Phase::RestSite` with `SmithCard` actions).
    /// Mirrors [`HeuristicPolicy::choose_smith`] (heuristic.rs:132-154) but
    /// softmax-samples the upgrade scores rather than argmax.
    fn sample_smith(&mut self, view: &VisibleRunView, actions: &[RunAction]) -> usize {
        let options = &view.offers.smith_options;
        if options.is_empty() {
            return 0;
        }
        // Map each option to a distinct non-upgraded deck copy of its id.
        let mut used: Vec<usize> = Vec::new();
        let scores: Vec<f64> = options
            .iter()
            .map(|opt| {
                let deck_idx = view.deck.iter().enumerate().position(|(i, c)| {
                    c.card_id == opt.card_id && !c.upgraded && !used.contains(&i)
                });
                deck_idx.map_or(0.0, |i| {
                    used.push(i);
                    picks::dv_upgrade_card(view, &self.v, i)
                })
            })
            .collect();
        self.set_scores(scores.clone());
        let pos = softmax_sample(&scores, self.params.temperature, &mut self.rng);
        pos.min(actions.len().saturating_sub(1))
    }

    /// Sample one shop decision from the feasible candidate set (same set
    /// [`shop::best_shop_action`]'s knapsack considers) plus `Leave`,
    /// softmaxing over each candidate's `ΔV_net` (`Leave = 0.0`). Only the
    /// selection among the already-filtered feasible options is randomized.
    fn sample_shop_decision(&mut self, view: &VisibleRunView) -> shop::ShopDecision {
        let options = shop::feasible_shop_options(view, &self.v);
        // Candidates: each feasible buy/remove + the Leave option (ΔV 0.0).
        let mut scores: Vec<f64> = options.iter().map(|(_, dv, _)| *dv).collect();
        scores.push(0.0); // Leave baseline
        self.set_scores(scores.clone());
        let pos = softmax_sample(&scores, self.params.temperature, &mut self.rng);
        if pos < options.len() {
            shop::shop_decision_for(&options[pos].0)
        } else {
            shop::ShopDecision::Leave
        }
    }

    /// Sample a shop action, mapping the sampled [`shop::ShopDecision`] back
    /// to an index into `actions` (leaving if the engine didn't surface the
    /// wanted action). Mirrors [`HeuristicPolicy::choose_shop`]
    /// (heuristic.rs:170-184).
    fn choose_shop(&mut self, view: &VisibleRunView, actions: &[RunAction]) -> usize {
        let wanted = match self.sample_shop_decision(view) {
            shop::ShopDecision::BuyCard { idx } => RunAction::ShopBuyCard { idx },
            shop::ShopDecision::BuyRelic { idx } => RunAction::ShopBuyRelic { idx },
            shop::ShopDecision::BuyPotion { idx } => RunAction::ShopBuyPotion { idx },
            shop::ShopDecision::RemoveCard { deck_idx } => RunAction::ShopRemoveCard { deck_idx },
            shop::ShopDecision::Leave => RunAction::LeaveShop,
        };
        actions
            .iter()
            .position(|a| *a == wanted)
            .unwrap_or_else(|| Self::position_of(actions, &RunAction::LeaveShop))
    }
}

impl RunPolicy for SampledDeckPolicy {
    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        if actions.is_empty() {
            return None;
        }
        self.last_scores = None;
        let view = VisibleRunView::from_run_manager(rm);

        // Phase-agnostic deck-mutation preempts come first (they replace the
        // whole action list). Sample the surfaces we own (RemoveDeckCard /
        // UpgradeDeckCard); delegate Transform / Enchant / GrantDeckCard to
        // the heuristic.
        if matches!(
            actions.first(),
            Some(RunAction::RemoveDeckCard { .. } | RunAction::UpgradeDeckCard { .. })
        ) {
            if let Some(idx) = self.sample_deck_mutation(rm, &view, actions) {
                return Some(idx.min(actions.len() - 1));
            }
        }
        if matches!(
            actions.first(),
            Some(
                RunAction::TransformDeckCard { .. }
                    | RunAction::ConfirmDeckSelection
                    | RunAction::EnchantDeckCard { .. }
                    | RunAction::GrantDeckCard { .. }
            )
        ) {
            return self.inner.choose(rm, actions);
        }

        let idx = match rm.phase() {
            // Card-reward: this policy's Task 1.3 sampling (skip → heuristic
            // relic-reward sub-state below when there are no card offers).
            Phase::CardReward if !view.offers.card_rewards.is_empty() => {
                self.choose_card_reward(&view, actions)
            }
            Phase::MapChoice => self.sample_path(&view, actions),
            Phase::Shop => self.choose_shop(&view, actions),
            Phase::RestSite if matches!(actions.first(), Some(RunAction::SmithCard { .. })) => {
                self.sample_smith(&view, actions)
            }
            // Everything else (relic-reward sub-state, rest heal, boss relic,
            // events, combat, treasure, …) → embedded heuristic.
            _ => return self.inner.choose(rm, actions),
        };
        Some(idx.min(actions.len() - 1))
    }
}

impl VScoreSource for SampledDeckPolicy {
    fn take_v_scores(&mut self) -> Option<(Vec<f64>, f64)> {
        // Card-reward decisions set last_scores on self; all others set it
        // on inner. Return whichever is present (only one can be set per
        // decision due to the reset in choose()).
        self.last_scores
            .take()
            .or_else(|| self.inner.take_v_scores())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck_archetype::Archetype;
    use crate::policy::sampling::SampleParams;
    use crate::policy::surfaces::picks;
    use crate::view::{DeckCard, VisibleRunView};
    use sts2_content::run_callbacks::new_run_manager_for_character;
    use sts2_core::cards::CardId;
    use sts2_core::potions::CharacterId;
    use sts2_core::run_manager::RunAction;

    /// Build a view with a specific deck, starting from a fresh Ironclad RM.
    fn view_with_deck(deck: Vec<(CardId, bool)>) -> VisibleRunView {
        let rm = new_run_manager_for_character(42, CharacterId::Ironclad);
        let mut view = VisibleRunView::from_run_manager(&rm);
        view.deck = deck
            .into_iter()
            .map(|(card_id, upgraded)| DeckCard { card_id, upgraded })
            .collect();
        view
    }

    /// Build a fresh view from a run-manager (starter deck).
    fn fresh_view() -> VisibleRunView {
        let rm = new_run_manager_for_character(13, CharacterId::Ironclad);
        VisibleRunView::from_run_manager(&rm)
    }

    // ── Task 1.3 required tests (RED first) ───────────────────────────────

    /// Test (a): with `temperature=0.0` and zero archetype bias, `SampledDeckPolicy`
    /// reproduces `picks::best_card_reward`'s pick on a fixed view/offers.
    ///
    /// At temperature=0.0, softmax_sample reduces to argmax, so the sampled
    /// card-reward pick must equal the heuristic argmax pick.
    #[test]
    fn zero_temp_no_bias_matches_heuristic_argmax() {
        use crate::eval::{HeuristicV, Weights};

        let view = fresh_view();
        let v = HeuristicV::default();

        // Build offers: a good card (Inflame) and a bad card (Decay curse).
        let offers = vec![
            DeckCard {
                card_id: CardId::Decay,
                upgraded: false,
            },
            DeckCard {
                card_id: CardId::Inflame,
                upgraded: true,
            },
        ];

        // Ground truth from picks::best_card_reward.
        let (expected_pick, _) = picks::best_card_reward(&view, &v, &offers);

        // Build a SampledDeckPolicy with temperature=0.0, bias_strength=0.0.
        let weights = Weights::default();
        let mut policy = SampledDeckPolicy::new(weights, 1, 0);
        // Override params for the test: zero temperature, no bias.
        policy.params = SampleParams {
            temperature: 0.0,
            skip_bias: 0.0,
            archetype: Archetype::Generic,
            bias_strength: 0.0,
        };

        // Inject the offers into the view and build a fake action list.
        let mut test_view = view.clone();
        test_view.offers.card_rewards = offers.clone();

        // Build the action list: PickCardReward for each offer + None (skip).
        let mut actions: Vec<RunAction> = offers
            .iter()
            .enumerate()
            .map(|(i, _)| RunAction::PickCardReward { idx: Some(i) })
            .collect();
        actions.push(RunAction::PickCardReward { idx: None });

        // Call choose_card_reward directly.
        let sampled_idx = policy.choose_card_reward(&test_view, &actions);
        let sampled_action = &actions[sampled_idx];

        let expected_action = RunAction::PickCardReward { idx: expected_pick };
        assert_eq!(
            *sampled_action, expected_action,
            "at temperature=0 with no bias, sampled pick {sampled_action:?} must equal heuristic pick {expected_action:?}"
        );
    }

    /// Test (b): with Scaling archetype bias and a scaling card among
    /// non-scaling offers, the scaling card's selection probability rises
    /// across many samples compared to the no-bias baseline.
    #[test]
    fn scaling_bias_increases_scaling_card_probability() {
        use crate::eval::Weights;

        let view = view_with_deck(vec![
            (CardId::StrikeIronclad, false),
            (CardId::StrikeIronclad, false),
            (CardId::StrikeIronclad, false),
            (CardId::DefendIronclad, false),
            (CardId::DefendIronclad, false),
            (CardId::DefendIronclad, false),
            (CardId::Bash, false),
            (CardId::StrikeIronclad, false),
        ]);

        // Offers: a scaling card (Inflame) and two non-scaling cards.
        let offers = vec![
            DeckCard {
                card_id: CardId::Inflame,
                upgraded: false,
            }, // scaling
            DeckCard {
                card_id: CardId::StrikeIronclad,
                upgraded: false,
            }, // not scaling
            DeckCard {
                card_id: CardId::DefendIronclad,
                upgraded: false,
            }, // not scaling
        ];

        let actions: Vec<RunAction> = offers
            .iter()
            .enumerate()
            .map(|(i, _)| RunAction::PickCardReward { idx: Some(i) })
            .collect();

        const N: usize = 2000;

        // Count picks for scaling card (index 0) without bias.
        let no_bias_count = {
            let weights = Weights::default();
            let mut policy = SampledDeckPolicy::new(weights, 42, 0);
            policy.params = SampleParams {
                temperature: 1.0,
                skip_bias: 0.0,
                archetype: Archetype::Scaling,
                bias_strength: 0.0, // no bias
            };
            let mut test_view = view.clone();
            test_view.offers.card_rewards = offers.clone();
            (0..N)
                .filter(|_| {
                    let idx = policy.choose_card_reward(&test_view, &actions);
                    actions[idx] == (RunAction::PickCardReward { idx: Some(0) })
                })
                .count()
        };

        // Count picks for scaling card (index 0) with strong Scaling bias.
        let biased_count = {
            let weights = Weights::default();
            let mut policy = SampledDeckPolicy::new(weights, 42, 0);
            policy.params = SampleParams {
                temperature: 1.0,
                skip_bias: 0.0,
                archetype: Archetype::Scaling,
                bias_strength: 2.0, // strong positive bias on scaling cards
            };
            let mut test_view = view.clone();
            test_view.offers.card_rewards = offers.clone();
            (0..N)
                .filter(|_| {
                    let idx = policy.choose_card_reward(&test_view, &actions);
                    actions[idx] == (RunAction::PickCardReward { idx: Some(0) })
                })
                .count()
        };

        assert!(
            biased_count > no_bias_count,
            "biased_count={biased_count} should be > no_bias_count={no_bias_count} ({N} samples each)"
        );
    }

    /// Sanity check: `card_matches_archetype` returns true for a scaling card
    /// under the Scaling archetype.
    #[test]
    fn card_matches_archetype_scaling() {
        assert!(card_matches_archetype(
            CardId::Inflame,
            false,
            Archetype::Scaling
        ));
        assert!(card_matches_archetype(
            CardId::DemonFormCard,
            false,
            Archetype::Scaling
        ));
        assert!(!card_matches_archetype(
            CardId::StrikeIronclad,
            false,
            Archetype::Scaling
        ));
    }

    /// Sanity check: `card_matches_archetype` for Generic is always true.
    #[test]
    fn card_matches_archetype_generic_always_true() {
        for card_id in [
            CardId::StrikeIronclad,
            CardId::DefendIronclad,
            CardId::Inflame,
            CardId::Decay,
        ] {
            assert!(
                card_matches_archetype(card_id, false, Archetype::Generic),
                "{card_id:?} should match Generic"
            );
        }
    }

    /// Sanity check: `card_matches_archetype` for Thin is always false.
    #[test]
    fn card_matches_archetype_thin_always_false() {
        for card_id in [CardId::StrikeIronclad, CardId::Inflame, CardId::Bash] {
            assert!(
                !card_matches_archetype(card_id, false, Archetype::Thin),
                "{card_id:?} should not match Thin"
            );
        }
    }

    // ── Task 1.4: shop / removal / smith / path sampling ──────────────────

    use crate::policy::surfaces::shop;
    use sts2_core::full_run::RunPolicy;
    use sts2_core::run_manager::Phase;

    /// Override a policy's params to the given temperature with no bias.
    fn set_temp(policy: &mut SampledDeckPolicy, temperature: f64) {
        policy.params = SampleParams {
            temperature,
            skip_bias: 0.0,
            archetype: Archetype::Generic,
            bias_strength: 0.0,
        };
    }

    // ── Path (Phase::MapChoice) ───────────────────────────────────────────

    /// At temperature=0, the sampled map move equals the greedy heuristic
    /// move (both reduce to argmax over `scored_first_cols`).
    #[test]
    fn path_zero_temp_matches_heuristic() {
        for seed in [17, 23, 41, 55] {
            let rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
            assert_eq!(rm.phase(), Phase::MapChoice);
            let actions = rm.get_available_actions();

            let mut greedy = HeuristicPolicy::new();
            let greedy_idx = greedy.choose(&rm, &actions).expect("greedy pick");

            let mut sampled = SampledDeckPolicy::new(Weights::default(), seed, 0);
            set_temp(&mut sampled, 0.0);
            let sampled_idx = sampled.choose(&rm, &actions).expect("sampled pick");

            assert_eq!(
                actions[sampled_idx], actions[greedy_idx],
                "seed {seed}: temp=0 map move must equal greedy"
            );
        }
    }

    /// At high temperature the sampled map move covers >1 column across
    /// many draws (on a map that offers more than one first column).
    #[test]
    fn path_high_temp_covers_multiple_cols() {
        // Find a seed whose first MapChoice offers ≥2 moves.
        let seed = [17, 23, 41, 55, 7, 13, 29, 31]
            .into_iter()
            .find(|&s| {
                let rm = new_run_manager_for_character(s, CharacterId::Ironclad);
                rm.get_available_actions()
                    .iter()
                    .filter(|a| matches!(a, RunAction::Move { .. }))
                    .count()
                    >= 2
            })
            .expect("a seed with ≥2 first-move columns");

        let rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let actions = rm.get_available_actions();
        let mut sampled = SampledDeckPolicy::new(Weights::default(), seed, 0);
        set_temp(&mut sampled, 5.0);

        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..400 {
            let idx = sampled.choose(&rm, &actions).expect("pick");
            seen.insert(idx);
        }
        assert!(
            seen.len() >= 2,
            "high-temp path sampling should cover >1 column, saw {seen:?}"
        );
    }

    // ── Removal (RemoveDeckCard preempt) ──────────────────────────────────

    /// Build a run-manager with `n` forced deck removals pending, plus a
    /// curse appended so the offer order is non-trivial.
    fn rm_with_forced_removal(seed: i32) -> RunManager {
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let curse = sts2_content::cards::spec_index::card_pod_from_id(CardId::Decay, false)
            .expect("curse pod");
        rm.run_state.master_deck.push(curse);
        rm.run_state.pending_deck_removals = 1;
        rm
    }

    /// At temperature=0, sampled forced-removal equals the greedy heuristic
    /// pick (both argmax `best_removal` over the offer order).
    #[test]
    fn removal_zero_temp_matches_heuristic() {
        let rm = rm_with_forced_removal(41);
        let actions = rm.get_available_actions();
        assert!(matches!(
            actions.first(),
            Some(RunAction::RemoveDeckCard { .. })
        ));

        let mut greedy = HeuristicPolicy::new();
        let greedy_idx = greedy.choose(&rm, &actions).expect("greedy pick");

        let mut sampled = SampledDeckPolicy::new(Weights::default(), 41, 0);
        set_temp(&mut sampled, 0.0);
        let sampled_idx = sampled.choose(&rm, &actions).expect("sampled pick");

        assert_eq!(
            actions[sampled_idx], actions[greedy_idx],
            "temp=0 removal must equal greedy best_removal"
        );
    }

    /// High temperature covers >1 removal offer across many draws.
    #[test]
    fn removal_high_temp_covers_multiple_offers() {
        let rm = rm_with_forced_removal(41);
        let actions = rm.get_available_actions();
        assert!(actions.len() >= 2, "need ≥2 removal offers");

        let mut sampled = SampledDeckPolicy::new(Weights::default(), 41, 0);
        set_temp(&mut sampled, 10.0);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..600 {
            seen.insert(sampled.choose(&rm, &actions).expect("pick"));
        }
        assert!(
            seen.len() >= 2,
            "high-temp removal should cover >1 offer, saw {seen:?}"
        );
    }

    // ── Smith / Upgrade ───────────────────────────────────────────────────

    /// Build a run-manager with an `UpgradeDeckCard` preempt pending.
    ///
    /// The curse at the FRONT is load-bearing: it is in the `!upgraded` set
    /// but not in the engine's `is_upgradable()` offer, so the two index
    /// spaces shift apart and the tests below actually discriminate between
    /// them (GH #399 review — an untouched starter deck cannot).
    fn rm_with_upgrade_preempt(seed: i32) -> RunManager {
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let curse = sts2_content::cards::spec_index::card_pod_from_id(CardId::Decay, false)
            .expect("curse pod");
        rm.run_state.master_deck.insert(0, curse);
        rm.run_state.pending_deck_upgrades = 1;
        rm
    }

    /// The sampled policy must resolve `UpgradeDeckCard` in the engine's
    /// `is_upgradable()` index space: at temperature 0 the card the engine
    /// actually upgrades is the one `best_upgrade` scored highest.
    #[test]
    fn upgrade_zero_temp_upgrades_the_card_it_scored() {
        let mut rm = rm_with_upgrade_preempt(41);
        let eligible = picks::upgradable_deck_indices(&rm);
        assert!(
            eligible.len() < rm.run_state.master_deck.len(),
            "fixture must contain a non-upgradable card or it proves nothing"
        );
        let actions = rm.get_available_actions();
        let view = crate::view::VisibleRunView::from_run_manager(&rm);

        let mut sampled = SampledDeckPolicy::new(Weights::default(), 41, 0);
        set_temp(&mut sampled, 0.0);
        let (want_pos, _, _) = picks::best_upgrade(&view, &sampled.v, &eligible).expect("scored");
        let want_deck_idx = eligible[want_pos];

        let idx = sampled.choose(&rm, &actions).expect("sampled pick");
        let before: Vec<bool> = rm
            .run_state
            .master_deck
            .iter()
            .map(|c| c.upgraded)
            .collect();
        let _ = rm.take_action(actions[idx]);
        let flipped: Vec<usize> = rm
            .run_state
            .master_deck
            .iter()
            .enumerate()
            .filter(|(i, c)| c.upgraded && !before[*i])
            .map(|(i, _)| i)
            .collect();
        assert_eq!(flipped, vec![want_deck_idx], "upgraded the card it scored");
    }

    /// At temperature=0, sampled `UpgradeDeckCard` preempt equals the greedy
    /// heuristic pick (both argmax `best_upgrade`).
    #[test]
    fn upgrade_zero_temp_matches_heuristic() {
        let rm = rm_with_upgrade_preempt(41);
        let actions = rm.get_available_actions();
        assert!(matches!(
            actions.first(),
            Some(RunAction::UpgradeDeckCard { .. })
        ));

        let mut greedy = HeuristicPolicy::new();
        let greedy_idx = greedy.choose(&rm, &actions).expect("greedy pick");

        let mut sampled = SampledDeckPolicy::new(Weights::default(), 41, 0);
        set_temp(&mut sampled, 0.0);
        let sampled_idx = sampled.choose(&rm, &actions).expect("sampled pick");

        assert_eq!(
            actions[sampled_idx], actions[greedy_idx],
            "temp=0 upgrade must equal greedy best_upgrade"
        );
    }

    /// High temperature covers >1 upgrade target across many draws.
    #[test]
    fn upgrade_high_temp_covers_multiple_targets() {
        let rm = rm_with_upgrade_preempt(41);
        let actions = rm.get_available_actions();
        assert!(actions.len() >= 2, "need ≥2 upgrade targets");

        let mut sampled = SampledDeckPolicy::new(Weights::default(), 41, 0);
        set_temp(&mut sampled, 10.0);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..600 {
            seen.insert(sampled.choose(&rm, &actions).expect("pick"));
        }
        assert!(
            seen.len() >= 2,
            "high-temp upgrade should cover >1 target, saw {seen:?}"
        );
    }

    // ── Shop (Phase::Shop) ────────────────────────────────────────────────

    /// Build a run-manager view-substitute shop view with several cards.
    fn shop_view(gold: i32) -> VisibleRunView {
        use crate::view::{ShopCardOffer, ShopView};
        let rm = new_run_manager_for_character(29, CharacterId::Ironclad);
        let mut view = VisibleRunView::from_run_manager(&rm);
        view.gold = gold;
        view.offers.shop = Some(ShopView {
            cards: vec![
                Some(ShopCardOffer {
                    card: DeckCard {
                        card_id: CardId::Inflame,
                        upgraded: true,
                    },
                    price: 50,
                }),
                Some(ShopCardOffer {
                    card: DeckCard {
                        card_id: CardId::Inflame,
                        upgraded: false,
                    },
                    price: 50,
                }),
                Some(ShopCardOffer {
                    card: DeckCard {
                        card_id: CardId::Bash,
                        upgraded: false,
                    },
                    price: 60,
                }),
            ],
            relics: vec![],
            potions: vec![],
            removal_cost: 75,
            removal_used: true,
        });
        view
    }

    /// At temperature=0, the sampled shop decision equals `best_shop_action`.
    #[test]
    fn shop_zero_temp_matches_best_shop_action() {
        let v = HeuristicV::default();
        let reserve = v.weights.shop.gold_reserve;
        let view = shop_view(reserve + 500);

        let greedy = shop::best_shop_action(&view, &v);

        let mut sampled = SampledDeckPolicy::new(Weights::default(), 1, 0);
        set_temp(&mut sampled, 0.0);
        let decision = sampled.sample_shop_decision(&view);

        assert_eq!(decision, greedy, "temp=0 shop must equal best_shop_action");
    }

    /// With no affordable items above the reserve, the sampled shop leaves
    /// (the feasible set is just `Leave`).
    #[test]
    fn shop_leaves_when_nothing_feasible() {
        let v = HeuristicV::default();
        let reserve = v.weights.shop.gold_reserve;
        // gold below the cheapest price + reserve → only Leave is feasible.
        let view = shop_view(reserve + 10);

        let mut sampled = SampledDeckPolicy::new(Weights::default(), 1, 0);
        set_temp(&mut sampled, 5.0);
        for _ in 0..50 {
            assert_eq!(
                sampled.sample_shop_decision(&view),
                shop::ShopDecision::Leave
            );
        }
    }

    /// High temperature covers >1 shop decision across many draws (multiple
    /// feasible buys + Leave all reachable).
    #[test]
    fn shop_high_temp_covers_multiple_decisions() {
        let v = HeuristicV::default();
        let reserve = v.weights.shop.gold_reserve;
        let view = shop_view(reserve + 500);

        let mut sampled = SampledDeckPolicy::new(Weights::default(), 1, 0);
        set_temp(&mut sampled, 5.0);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..600 {
            seen.insert(format!("{:?}", sampled.sample_shop_decision(&view)));
        }
        assert!(
            seen.len() >= 2,
            "high-temp shop sampling should cover >1 decision, saw {seen:?}"
        );
    }

    /// The card-removal slot participates in shop selection: with the slot
    /// AVAILABLE (`removal_used: false`), an affordable removal cost, and a
    /// curse in the deck, the removal candidate is the highest-ΔV feasible
    /// option, so at temperature=0 `sample_shop_decision` must (a) equal
    /// `best_shop_action` and (b) actually be the `RemoveCard` slot — proving
    /// `feasible_shop_options`'s remove-slot branch is exercised by the pin
    /// (the other shop fixtures use `removal_used: true`).
    #[test]
    fn shop_zero_temp_selects_available_removal_slot() {
        use crate::view::{ShopCardOffer, ShopView};
        let v = HeuristicV::default();
        let reserve = v.weights.shop.gold_reserve;

        let rm = new_run_manager_for_character(29, CharacterId::Ironclad);
        // Curse in the deck → removing it is a strongly positive-ΔV option,
        // dominating the card buys.
        let mut view = VisibleRunView::from_run_manager(&rm).with_card_added(CardId::Decay, false);
        view.gold = reserve + 500;
        view.offers.shop = Some(ShopView {
            cards: vec![Some(ShopCardOffer {
                card: DeckCard {
                    card_id: CardId::Bash,
                    upgraded: false,
                },
                price: 60,
            })],
            relics: vec![],
            potions: vec![],
            removal_cost: 50,
            removal_used: false, // slot AVAILABLE
        });

        let greedy = shop::best_shop_action(&view, &v);
        // Sanity: the greedy knapsack itself picks the removal slot here, so
        // the pin below is non-vacuous (the slot must win selection).
        assert!(
            matches!(greedy, shop::ShopDecision::RemoveCard { .. }),
            "fixture must make the removal slot the greedy pick, got {greedy:?}"
        );

        let mut sampled = SampledDeckPolicy::new(Weights::default(), 1, 0);
        set_temp(&mut sampled, 0.0);
        let decision = sampled.sample_shop_decision(&view);

        assert_eq!(
            decision, greedy,
            "temp=0 shop with an available removal slot must equal best_shop_action (RemoveCard)"
        );
    }

    /// VScoreSource: card-reward decisions produce scores; take has consume
    /// semantics (second take returns None).
    #[test]
    fn v_scores_card_reward_take_semantics() {
        use crate::eval::Weights;

        let view = fresh_view();
        let offers = vec![
            DeckCard {
                card_id: CardId::Inflame,
                upgraded: false,
            },
            DeckCard {
                card_id: CardId::StrikeIronclad,
                upgraded: false,
            },
        ];

        let actions: Vec<RunAction> = offers
            .iter()
            .enumerate()
            .map(|(i, _)| RunAction::PickCardReward { idx: Some(i) })
            .collect();

        let weights = Weights::default();
        let mut policy = SampledDeckPolicy::new(weights, 7, 3);

        let mut test_view = view.clone();
        test_view.offers.card_rewards = offers.clone();

        let _idx = policy.choose_card_reward(&test_view, &actions);
        let first = policy.take_v_scores();
        let second = policy.take_v_scores();

        assert!(first.is_some(), "first take must return scores");
        let (scores, margin) = first.unwrap();
        assert_eq!(
            scores.len(),
            offers.len() + 1,
            "scores.len() = offers + 1 (skip)"
        );
        assert!(margin >= 0.0);
        assert!(
            second.is_none(),
            "second take must return None (consume semantics)"
        );
    }

    // ── GrantDeckCard dispatch regression (panic-fix, 2026-06-20) ─────────

    /// Regression test: seeds 5, 6, 11, 24, 34 hit a `GrantDeckCard`
    /// preempt while `phase()==MapChoice` (the RoomFullOfCheese "Gorge"
    /// card-grant surface). Before the fix, `SampledDeckPolicy::choose`
    /// mis-routed the `GrantDeckCard` action list to `sample_path`, which
    /// built an all-`NEG_INFINITY` score vec (no `Move` actions), causing
    /// `softmax_sample` to panic. The test drives the full run with combat
    /// budget `None` (same as the arena's `--combat none` mode) and asserts
    /// it completes without panicking for all five previously-failing seeds.
    ///
    /// Run RED first to confirm the panic (pre-fix), GREEN after the fix.
    #[test]
    fn grant_deck_card_preempt_does_not_panic() {
        use crate::rollout::ensure_content_installed;
        use sts2_content::run_callbacks::{
            initialize_run_setup, new_run_manager_for_character_with_ascension,
        };
        use sts2_core::full_run::{drive_full_run, FullRunConfig};

        // Seeds that panicked before the fix (all hit GrantDeckCard during MapChoice).
        for seed in [5_i32, 6, 11, 24, 34] {
            ensure_content_installed();
            let rm = new_run_manager_for_character_with_ascension(seed, CharacterId::Ironclad, 0);
            let mut rm = rm.with_neow_start();
            initialize_run_setup(&mut rm);
            let mut policy = SampledDeckPolicy::new(Weights::default(), seed, 0);
            let cfg = FullRunConfig::default();
            // Must not panic.
            let _report = drive_full_run(&mut rm, &mut policy, &cfg);
        }
    }
}
