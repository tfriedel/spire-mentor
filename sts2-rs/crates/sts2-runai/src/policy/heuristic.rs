//! `HeuristicPolicy` — the M1 tier-1 heuristic agent over all six meta
//! surfaces (full-run-AI plan §2, §5/M1; bead `sts2-rs-40ikw.5`).
//!
//! Dispatch per surface (plan §2):
//! - **Pathing** (`MapChoice`): exhaustive DAG enumeration scored by
//!   projected V at boss arrival ([`surfaces::pathing`]).
//! - **Card-shaped picks** (card reward incl. skip, boss relic, deck
//!   remove/upgrade/transform, smith target): `argmax ΔV` with the pure
//!   apply-candidate-to-view helpers ([`surfaces::picks`]).
//! - **Events**: per-event-option heuristic priors, default 0 ΔV
//!   ([`surfaces::events`]; outcome-distribution EV refines in M2).
//! - **Shop**: marginal-ΔV-per-gold knapsack with a gold reserve
//!   ([`surfaces::shop`]).
//! - **Rest**: heal-vs-smith V comparison ([`surfaces::rest`]).
//! - **Potion economy**: buy/hold ΔV with the boss-insurance term in
//!   `HeuristicV::potion_term` (in-combat *usage* is combat MCTS's job).
//! - **Combat / anything unhandled**: first option, never a panic — the
//!   arena composes this policy inside `MctsCombatPolicy` for real
//!   combat, and bins NotImplemented outcomes either way.
//!
//! §2.5: every evaluation goes through [`VisibleRunView`]; the live
//! `RunManager` is touched only for action-index bookkeeping
//! (`transform_eligible_deck_indices` — a deterministic index space the
//! player also sees, not hidden state).

use sts2_core::full_run::RunPolicy;
use sts2_core::run_manager::{Phase, RunAction, RunManager};

use crate::eval::{HeuristicV, Weights};
use crate::policy::surfaces::{combat_choice, events, pathing, picks, rest, shop};
use crate::policy::VScoreSource;
use crate::view::VisibleRunView;

/// Tier-1-only heuristic run policy (plan §5/M1). Ironclad-first: the
/// card table's hand-written supplement covers IC + colorless; other
/// characters degrade to spec-derived features.
#[derive(Clone, Debug, Default)]
pub struct HeuristicPolicy {
    v: HeuristicV,
    /// `(scores aligned to the surfaced actions, top-2 margin)` of the
    /// decision just made — consumed by [`VScoreSource::take_v_scores`].
    last_scores: Option<(Vec<f64>, f64)>,
}

impl HeuristicPolicy {
    /// Embedded default weights (`data/runai/heuristic_v.toml`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_weights(weights: Weights) -> Self {
        Self {
            v: HeuristicV::new(weights),
            last_scores: None,
        }
    }

    /// The tier-1 value function (so a wrapping policy — e.g.
    /// `TieredPolicy`'s meta-move escalation — can reuse the same `V`
    /// for candidate selection without rebuilding it).
    #[must_use]
    pub(crate) fn v(&self) -> &HeuristicV {
        &self.v
    }

    /// Record per-action scores for the audit log (plan §3.6); margin =
    /// top-1 − top-2 (0 when fewer than two finite options).
    fn set_scores(&mut self, scores: Vec<f64>) {
        let mut sorted: Vec<f64> = scores.iter().copied().filter(|s| s.is_finite()).collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = match sorted.as_slice() {
            [first, second, ..] => first - second,
            _ => 0.0,
        };
        self.last_scores = Some((scores, margin));
    }

    /// Index of `wanted` in `actions`, falling back to option 0.
    fn position_of(actions: &[RunAction], wanted: &RunAction) -> usize {
        actions.iter().position(|a| a == wanted).unwrap_or(0)
    }

    /// Deck-mutation preempt surfaces (phase-agnostic, homogeneous
    /// action lists — see `RunManager::get_available_actions`).
    fn choose_deck_mutation(
        &mut self,
        rm: &RunManager,
        view: &VisibleRunView,
        actions: &[RunAction],
    ) -> Option<usize> {
        match actions.first()? {
            RunAction::RemoveDeckCard { .. } => {
                // `RemoveDeckCard { idx }` indexes the engine's removal
                // offer order (removable cards, curses sorted to the front —
                // `RunState::removal_offer_order`), NOT raw master-deck
                // position. Score each offer by the ΔV of removing the
                // master-deck card it maps to; `best_removal` returns the
                // position within that order, which is exactly the action
                // index the engine expects. (A raw `0..deck.len()` space
                // removed the wrong card whenever a curse fronted the order.)
                let eligible = rm.run_state.removal_offer_order();
                let (pos, _, scores) = picks::best_removal(view, &self.v, &eligible)?;
                self.set_scores(scores);
                Some(pos)
            }
            RunAction::UpgradeDeckCard { .. } => {
                // Index space = position within the engine's `is_upgradable()`
                // subset, NOT merely the non-upgraded one: Curse/Status and the
                // three Quest cards are excluded from the offer, so a position
                // computed over `!upgraded` is off by one for every target
                // behind such a card and the engine upgrades a card this
                // policy never scored (GH #399 review; `pos.min(len-1)` hid it
                // by clamping). Same failure shape as the RemoveDeckCard
                // curse-front bug (RP94) above.
                let eligible = picks::upgradable_deck_indices(rm);
                let (pos, _, scores) = picks::best_upgrade(view, &self.v, &eligible)?;
                self.set_scores(scores);
                Some(pos.min(actions.len().saturating_sub(1)))
            }
            RunAction::TransformDeckCard { .. } => {
                // Transform = remove + unknown replacement: target the
                // card whose removal gains the most. Index space =
                // position within the engine's eligible subset.
                let eligible = rm.transform_eligible_deck_indices();
                let (pos, _, scores) = picks::best_removal(view, &self.v, &eligible)?;
                self.set_scores(scores);
                Some(pos.min(actions.len().saturating_sub(1)))
            }
            // Enchant priors are unknown at M1 — first option.
            RunAction::EnchantDeckCard { .. } => Some(0),
            _ => None,
        }
    }

    /// Full-belt potion-swap preempt (GH #252): swap out the
    /// lowest-value held potion iff the pending reward potion is
    /// strictly more valuable, else decline. "Value" is the rarity
    /// bucket — a coarse but monotone proxy (Rare > Uncommon >
    /// Common/Event > Token/None) that never trades a Rare away for a
    /// Common. Action list shape (see `get_available_actions`):
    /// `[decline, discard slot 0, discard slot 1, …]`.
    fn choose_potion_swap(rm: &RunManager, actions: &[RunAction]) -> usize {
        fn rarity_value(kind: sts2_core::potions::PotionKind) -> u8 {
            use sts2_core::potions::PotionRarity;
            match sts2_content::potions::all::spec_for(kind).rarity {
                PotionRarity::None | PotionRarity::Token => 0,
                PotionRarity::Common | PotionRarity::Event => 1,
                PotionRarity::Uncommon => 2,
                PotionRarity::Rare => 3,
            }
        }
        let Some(&incoming) = rm.run_state.pending_potion_swaps.first() else {
            return 0; // decline (shouldn't happen: surface implies pending)
        };
        let worst = rm
            .run_state
            .potions
            .iter()
            .enumerate()
            .min_by_key(|(_, &k)| rarity_value(k));
        match worst {
            Some((slot, &held)) if rarity_value(incoming) > rarity_value(held) => {
                Self::position_of(
                    actions,
                    &RunAction::PotionSwap {
                        discard_slot: Some(slot),
                    },
                )
            }
            _ => Self::position_of(actions, &RunAction::PotionSwap { discard_slot: None }),
        }
    }

    /// Smith sub-state: `SmithCard{idx}` indexes the pending-choice
    /// options; ΔV = upgrading the matching deck copy.
    fn choose_smith(&mut self, view: &VisibleRunView, actions: &[RunAction]) -> Option<usize> {
        let options = &view.offers.smith_options;
        if options.is_empty() {
            return Some(0);
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
        let best = picks::argmax(&scores)?;
        self.set_scores(scores);
        Some(best.min(actions.len().saturating_sub(1)))
    }

    /// Combat choice resolution (issue #164). The composing
    /// `MctsCombatPolicy` searches real combat; the only decisions this
    /// policy faces raw are open `pending_choice` selections (pile-search
    /// / multi-pick / Discover). Score them greedily via
    /// [`combat_choice`]; with no open choice (normal play) or an
    /// unscorable choice kind, keep the first-option default.
    fn choose_combat(&mut self, rm: &RunManager, actions: &[RunAction]) -> usize {
        if let Some(combat) = rm.combat.as_ref() {
            if let Some(pc) = combat.pending_choice.as_ref() {
                if let Some((idx, scores)) = combat_choice::choose_combat_choice(pc, actions) {
                    self.set_scores(scores);
                    return idx.min(actions.len().saturating_sub(1));
                }
            }
        }
        0
    }

    fn choose_card_reward(&mut self, view: &VisibleRunView, actions: &[RunAction]) -> usize {
        // An egg queued beside the open card reward is offered first, with
        // the card options still on screen (GH #1094): take it, since it only
        // upgrades the options.
        if actions.contains(&RunAction::PickRelicReward) {
            return Self::position_of(actions, &RunAction::PickRelicReward);
        }
        if !view.offers.card_rewards.is_empty() {
            let (pick, scores) = picks::best_card_reward(view, &self.v, &view.offers.card_rewards);
            self.set_scores(scores);
            return Self::position_of(actions, &RunAction::PickCardReward { idx: pick });
        }
        // Relic-reward sub-state: a free relic is ΔV-positive under the
        // flat relic values — take it.
        if view.offers.relic_reward.is_some() {
            return Self::position_of(actions, &RunAction::PickRelicReward);
        }
        0
    }

    fn choose_shop(&mut self, view: &VisibleRunView, actions: &[RunAction]) -> usize {
        let wanted = match shop::best_shop_action(view, &self.v) {
            shop::ShopDecision::BuyCard { idx } => RunAction::ShopBuyCard { idx },
            shop::ShopDecision::BuyRelic { idx } => RunAction::ShopBuyRelic { idx },
            shop::ShopDecision::BuyPotion { idx } => RunAction::ShopBuyPotion { idx },
            shop::ShopDecision::RemoveCard { deck_idx } => RunAction::ShopRemoveCard { deck_idx },
            shop::ShopDecision::Leave => RunAction::LeaveShop,
        };
        // If the engine didn't surface the wanted action (affordability
        // raced, index drift), leave rather than buy something random.
        actions
            .iter()
            .position(|a| *a == wanted)
            .unwrap_or_else(|| Self::position_of(actions, &RunAction::LeaveShop))
    }
}

impl RunPolicy for HeuristicPolicy {
    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        if actions.is_empty() {
            return None;
        }
        self.last_scores = None;
        let view = VisibleRunView::from_run_manager(rm);

        // Claws' optional "up to 6 into Maul" select (GH #1353): take the
        // first offered card until the cap, then confirm — the first six
        // candidates in deck order, the pick the engine made before the
        // select surfaced. The confirm is always last, so index 0 is the
        // next card while any is offered and the confirm once none is.
        if actions.last() == Some(&RunAction::ConfirmDeckSelection) {
            return Some(0);
        }
        // Phase-agnostic deck-mutation preempts come first (they replace
        // the whole action list — see get_available_actions).
        if matches!(
            actions.first(),
            Some(
                RunAction::RemoveDeckCard { .. }
                    | RunAction::UpgradeDeckCard { .. }
                    | RunAction::TransformDeckCard { .. }
                    | RunAction::EnchantDeckCard { .. }
            )
        ) {
            if let Some(idx) = self.choose_deck_mutation(rm, &view, actions) {
                return Some(idx.min(actions.len() - 1));
            }
            return Some(0);
        }

        // Full-belt potion-swap preempt (homogeneous list, phase-agnostic
        // like the deck mutations above). GH #252.
        if matches!(actions.first(), Some(RunAction::PotionSwap { .. })) {
            let idx = Self::choose_potion_swap(rm, actions);
            return Some(idx.min(actions.len() - 1));
        }

        let idx = match rm.phase() {
            Phase::MapChoice => match pathing::scored_first_cols(&view, &self.v) {
                Some((col, cols)) => {
                    // Audit / margin-gate scores aligned to the legal
                    // Move action order (M2.5 path_delta needs the margin).
                    let scores = actions
                        .iter()
                        .map(|a| match a {
                            RunAction::Move { col } => cols
                                .iter()
                                .find(|(c, _)| c == col)
                                .map_or(f64::NEG_INFINITY, |(_, s)| *s),
                            _ => f64::NEG_INFINITY,
                        })
                        .collect();
                    self.set_scores(scores);
                    Self::position_of(actions, &RunAction::Move { col })
                }
                None => 0,
            },
            Phase::CardReward => self.choose_card_reward(&view, actions),
            Phase::BossRelic => {
                if view.offers.boss_relics.is_empty() {
                    0
                } else {
                    let (best, scores) =
                        picks::best_boss_relic(&view, &self.v, &view.offers.boss_relics);
                    self.set_scores(scores);
                    Self::position_of(actions, &RunAction::PickBossRelic { idx: best })
                }
            }
            Phase::Shop => self.choose_shop(&view, actions),
            Phase::RestSite => {
                if matches!(actions.first(), Some(RunAction::SmithCard { .. })) {
                    return self.choose_smith(&view, actions);
                }
                let (opt, scores) = rest::best_rest_option(&view, &self.v);
                self.set_scores(scores);
                Self::position_of(actions, &RunAction::RestOption { idx: opt })
            }
            Phase::Event => {
                let (best, scores) = events::best_event_option(&view, actions.len());
                self.set_scores(scores);
                best
            }
            // Real combat play is the composing MctsCombatPolicy's job.
            // What this policy IS asked to resolve raw (combat budget
            // `none`, or the MCTS→meta fallback) are the pile-search /
            // multi-pick / Discover choices of issue #164 — score those
            // greedily; normal play (no open choice) stays first-option.
            Phase::Combat => self.choose_combat(rm, actions),
            // Treasure/RunOver/unknown: first option.
            Phase::Treasure | Phase::RunOver => 0,
        };
        Some(idx.min(actions.len() - 1))
    }
}

impl VScoreSource for HeuristicPolicy {
    fn take_v_scores(&mut self) -> Option<(Vec<f64>, f64)> {
        self.last_scores.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts2_content::run_callbacks::new_run_manager_for_character;
    use sts2_core::full_run::{drive_full_run, FullRunConfig, RunOutcome};
    use sts2_core::potions::CharacterId;

    /// Forced deck removal (e.g. Precise Scissors) surfaces one
    /// `RemoveDeckCard { idx }` per offer, where `idx` indexes the engine's
    /// curse-front removal offer order — NOT the raw master deck. The policy
    /// must pick the offer index that maps to the highest-ΔV card (the
    /// curse); a raw master-deck-index policy removed the wrong card.
    #[test]
    fn forced_removal_targets_curse_through_offer_order() {
        use sts2_core::cards::CardId;
        let mut rm = new_run_manager_for_character(41, CharacterId::Ironclad);
        let curse = sts2_content::cards::spec_index::card_pod_from_id(CardId::Decay, false)
            .expect("curse pod");
        rm.run_state.master_deck.push(curse);
        let curse_deck_idx = rm.run_state.master_deck.len() - 1;
        rm.run_state.pending_deck_removals = 1;

        let actions = rm.get_available_actions();
        assert!(
            matches!(actions.first(), Some(RunAction::RemoveDeckCard { .. })),
            "forced removal must preempt the action list"
        );

        let mut policy = HeuristicPolicy::new();
        let idx = policy.choose(&rm, &actions).expect("a pick");
        let RunAction::RemoveDeckCard { idx: offer_idx } = actions[idx] else {
            panic!("expected RemoveDeckCard, got {:?}", actions[idx]);
        };
        // The curse is sorted to the front of the offer order, so its offer
        // index differs from its master-deck index — the bug this guards.
        let order = rm.run_state.removal_offer_order();
        assert_eq!(
            order[offer_idx], curse_deck_idx,
            "policy must remove the curse, not whatever sits at that master-deck index"
        );
    }

    /// The `UpgradeDeckCard` index space is the engine's `is_upgradable()`
    /// subset (`run_manager.rs:954-968` offers it, `do_upgrade_deck_card_at`
    /// `:3044-3053` resolves it), which excludes Curse/Status **and** the
    /// three Quest cards — NOT merely `!upgraded`. With any such card in the
    /// deck the two subsets differ in length and every position after it is
    /// off by one, so the engine upgrades a card the policy never scored.
    /// G1 cannot catch this: both worlds run this same heuristic and diverge
    /// identically. Hence an identity assertion, on a deck containing a curse.
    #[test]
    fn upgrade_preempt_indexes_the_engines_upgradable_subset_not_merely_non_upgraded() {
        use sts2_core::cards::CardId;
        let mut rm = new_run_manager_for_character(41, CharacterId::Ironclad);
        // A curse at the FRONT: present in the `!upgraded` list, absent from
        // the `is_upgradable()` offer, so the two index spaces shift apart.
        let curse = sts2_content::cards::spec_index::card_pod_from_id(CardId::Decay, false)
            .expect("curse pod");
        rm.run_state.master_deck.insert(0, curse);
        rm.run_state.interactive_card_upgrade = true;
        rm.run_state.pending_deck_upgrades = 1;

        let upgradable: Vec<usize> = rm
            .run_state
            .master_deck
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.is_upgradable().then_some(i))
            .collect();
        assert!(
            upgradable.len() < rm.run_state.master_deck.len(),
            "fixture must contain a non-upgradable card or it proves nothing"
        );

        let actions = rm.get_available_actions();
        assert!(
            matches!(actions.first(), Some(RunAction::UpgradeDeckCard { .. })),
            "pending upgrade must preempt the action list"
        );
        assert_eq!(
            actions.len(),
            upgradable.len(),
            "engine offers one action per upgradable card"
        );

        let mut policy = HeuristicPolicy::new();
        // What the policy's own value function wants, resolved through the
        // engine's index space — computed independently of `choose`.
        let view = VisibleRunView::from_run_manager(&rm);
        let (want_pos, _, _) =
            picks::best_upgrade(&view, policy.v(), &upgradable).expect("a scored target");
        let want_deck_idx = upgradable[want_pos];

        let idx = policy.choose(&rm, &actions).expect("a pick");
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

        assert_eq!(flipped.len(), 1, "exactly one card upgrades");
        assert_eq!(
            flipped[0],
            want_deck_idx,
            "engine upgraded master-deck[{}] ({:?}) but the policy scored \
             master-deck[{want_deck_idx}] ({:?}) — the pick was made in the \
             `!upgraded` index space, not the engine's `is_upgradable()` one",
            flipped[0],
            rm.run_state.master_deck[flipped[0]].card_id,
            rm.run_state.master_deck[want_deck_idx].card_id,
        );
    }

    /// GH #399 flips `interactive_card_transform` on for every arena cell and
    /// every `PpoRunEnv` episode, so these two preempts are now reachable in
    /// ordinary runs — previously the engine auto-resolved them and this
    /// branch was only exercised by the workbench. Each must be answered by
    /// `choose_deck_mutation` over the engine's *eligible-subset* index space,
    /// never by a phase fallthrough.
    #[test]
    fn transform_and_enchant_preempts_resolve_in_eligible_index_space() {
        let mut policy = HeuristicPolicy::new();

        // ── Transform: index space = `transform_eligible_deck_indices`.
        let mut rm = new_run_manager_for_character(41, CharacterId::Ironclad);
        rm.run_state.interactive_card_transform = true;
        rm.run_state
            .pending_deck_transforms
            .push(sts2_core::PendingDeckTransform::default());
        let actions = rm.get_available_actions();
        assert!(
            matches!(actions.first(), Some(RunAction::TransformDeckCard { .. })),
            "pending transform must preempt the action list"
        );
        let idx = policy.choose(&rm, &actions).expect("a pick");
        let RunAction::TransformDeckCard { idx: offer_idx } = actions[idx] else {
            panic!("expected TransformDeckCard, got {:?}", actions[idx]);
        };
        assert!(offer_idx < rm.transform_eligible_deck_indices().len());

        // ── Enchant: no ΔV prior exists for enchantments, so the documented
        // rule is option 0 (`choose_deck_mutation`'s `EnchantDeckCard` arm).
        // Reached through the same `interactive_card_transform` flag — there
        // is no separate enchant flag.
        let mut rm = new_run_manager_for_character(41, CharacterId::Ironclad);
        rm.run_state.interactive_card_transform = true;
        rm.run_state
            .pending_deck_enchants
            .push(sts2_core::PendingDeckEnchant {
                enchantment: sts2_core::cards::EnchantmentId::Sown,
                amount: 1,
                type_restriction: None,
                secondary_type: None,
                optional: false,
            });
        let actions = rm.get_available_actions();
        assert!(
            matches!(actions.first(), Some(RunAction::EnchantDeckCard { .. })),
            "pending enchant must preempt the action list"
        );
        assert_eq!(
            policy.choose(&rm, &actions),
            Some(0),
            "enchant priors are unknown — the shared rule is option 0"
        );
    }

    /// Full-belt potion swap (GH #252): a rarer incoming potion swaps out
    /// the lowest-rarity held slot; an equal-or-worse one is declined.
    #[test]
    fn potion_swap_trades_up_by_rarity_and_declines_down() {
        use sts2_core::potions::PotionKind;
        let mut rm = new_run_manager_for_character(7, CharacterId::Ironclad);
        rm.run_state.interactive_potion_swap = true;
        rm.run_state.max_potion_slots = 2;
        rm.run_state.potions.clear();
        // Slot 0 Rare (BeetleJuice), slot 1 Common (BlockPotion).
        assert!(rm.run_state.add_potion_to_belt(PotionKind::BeetleJuice));
        assert!(rm.run_state.add_potion_to_belt(PotionKind::BlockPotion));
        let mut policy = HeuristicPolicy::new();

        // Uncommon incoming (Ashwater) beats the Common in slot 1.
        assert!(!rm.run_state.claim_potion_reward(PotionKind::Ashwater));
        let actions = rm.get_available_actions();
        let idx = policy.choose(&rm, &actions).expect("a pick");
        assert_eq!(
            actions[idx],
            RunAction::PotionSwap {
                discard_slot: Some(1)
            },
            "uncommon incoming swaps out the common, never the rare"
        );
        let _ = rm.take_action(actions[idx]);
        assert_eq!(
            rm.run_state.potions,
            vec![PotionKind::BeetleJuice, PotionKind::Ashwater]
        );

        // Common incoming (FirePotion) trades down against [Rare, Uncommon]
        // — decline.
        assert!(!rm.run_state.claim_potion_reward(PotionKind::FirePotion));
        let actions = rm.get_available_actions();
        let idx = policy.choose(&rm, &actions).expect("a pick");
        assert_eq!(
            actions[idx],
            RunAction::PotionSwap { discard_slot: None },
            "common incoming declines instead of trading down"
        );
    }

    /// End-to-end (issue #164): with an open combat `pending_choice`, the
    /// heuristic resolves it through the real `get_available_actions`
    /// surface — exhausting a curse rather than the first-in-list Strike
    /// (the old `Phase::Combat => 0` would have dumped the Strike).
    #[test]
    fn combat_choice_exhausts_curse_not_first_card() {
        use sts2_core::cards::CardId;
        use sts2_core::{ChoiceKind, ChoiceOption, PendingChoice};

        let mut rm = new_run_manager_for_character(3, CharacterId::Ironclad);
        let mut warmup = HeuristicPolicy::new();
        let mut guard = 0;
        while rm.phase() != Phase::Combat {
            let actions = rm.get_available_actions();
            let idx = warmup.choose(&rm, &actions).expect("warmup pick");
            let _ = rm.take_action(actions[idx]);
            guard += 1;
            assert!(guard < 300, "never reached combat");
        }

        // Inject a skippable exhaust choice: option 0 = Strike (keep),
        // option 1 = Decay curse (shed).
        let combat = rm.combat.as_mut().expect("in combat");
        combat.pending_choice = Some(PendingChoice {
            kind: ChoiceKind::ExhaustFromHand,
            options: vec![
                ChoiceOption {
                    card_id: CardId::StrikeIronclad,
                    upgraded: false,
                    card_uid: None,
                    source_pile: None,
                },
                ChoiceOption {
                    card_id: CardId::Decay,
                    upgraded: false,
                    card_uid: None,
                    source_pile: None,
                },
            ],
            min_select: 0,
            max_select: 1,
            can_skip: true,
            ..Default::default()
        });

        let actions = rm.get_available_actions();
        let mut policy = HeuristicPolicy::new();
        let idx = policy.choose(&rm, &actions).expect("combat-choice pick");
        assert_eq!(
            actions[idx],
            RunAction::CombatResolveChoice { option: Some(1) },
            "must exhaust the curse (option 1), not first-option the Strike; actions={actions:?}"
        );
    }

    /// First decision of a fresh run is a legal map move with replanned
    /// pathing.
    #[test]
    fn first_decision_is_legal_map_move() {
        let rm = new_run_manager_for_character(41, CharacterId::Ironclad);
        let actions = rm.get_available_actions();
        assert!(!actions.is_empty());
        let mut policy = HeuristicPolicy::new();
        let idx = policy.choose(&rm, &actions).expect("a pick");
        assert!(idx < actions.len());
        assert!(matches!(actions[idx], RunAction::Move { .. }));
    }

    /// M1 smoke (plan §5/M1 deliverable 6): drive 3 dev-200 seeds with
    /// HeuristicPolicy + combat=none in-process — must terminate, no
    /// panic. (Random-play combat depth is shallow; the gate is clean
    /// termination, not outcome.)
    #[test]
    fn smoke_three_seeds_terminate() {
        // First three rows of data/seed-panels/dev-200.json.
        for seed in [1_458_832_210, 521_362_408, 1_520_652_879] {
            let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
            let mut policy = HeuristicPolicy::new();
            let cfg = FullRunConfig::default();
            let report = drive_full_run(&mut rm, &mut policy, &cfg);
            assert!(
                !matches!(report.outcome, RunOutcome::MaxSteps),
                "seed {seed} did not terminate: {:?}",
                report.outcome
            );
        }
    }

    /// Every event room reaches `choose` BOUND, with its real options.
    ///
    /// While an event room is unbound the engine offers one placeholder
    /// `EventChoice`, which resolves option 0, so a policy that never bound
    /// never made an event choice in the arena while `PpoRunEnv` (which binds
    /// first, GH #302) let it choose: G1 diverged on seed 1, a LostWisp where
    /// the two picked differently (GH #841). `drive_full_run` now binds, so
    /// this spy implements ONLY `choose`, the way `TieredPolicy`, the
    /// labeler and every other wrapper do, and runs on the arena's own run
    /// construction (Neow start + `initialize_run_setup`).
    #[test]
    fn every_event_room_is_bound_before_choose() {
        struct ChooseOnly {
            inner: HeuristicPolicy,
            events: usize,
        }
        impl RunPolicy for ChooseOnly {
            fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
                if rm.phase() == Phase::Event {
                    assert!(
                        rm.run_state.current_event.is_some(),
                        "event room reached choose unbound at floor {}",
                        rm.run_state.total_floor
                    );
                    // Neow is pre-bound at run start; count the rooms that need it.
                    if rm.run_state.total_floor > 0 {
                        self.events += 1;
                    }
                }
                self.inner.choose(rm, actions)
            }
        }
        let mut seen = 0;
        for seed in 0..8 {
            let mut rm = crate::arena::new_arena_run_manager(
                seed,
                CharacterId::Ironclad,
                0,
                crate::arena::Act1Choice::Roll,
            );
            let mut spy = ChooseOnly {
                inner: HeuristicPolicy::new(),
                events: 0,
            };
            let _ = drive_full_run(&mut rm, &mut spy, &FullRunConfig::default());
            seen += spy.events;
        }
        assert!(
            seen >= 3,
            "only {seen} post-Neow event rooms in 8 runs: the test proves nothing"
        );
    }

    /// V scores reach the audit hook with take semantics.
    #[test]
    fn v_scores_flow_through_take() {
        let mut rm = new_run_manager_for_character(43, CharacterId::Ironclad);
        let mut policy = HeuristicPolicy::new();
        // Walk until a scored surface fires (card reward / event / rest).
        for _ in 0..400 {
            if rm.is_over() {
                break;
            }
            let actions = rm.get_available_actions();
            if actions.is_empty() {
                break;
            }
            let idx = policy.choose(&rm, &actions).expect("pick");
            if let Some((scores, margin)) = policy.take_v_scores() {
                assert!(!scores.is_empty());
                assert!(margin >= 0.0);
                assert!(policy.take_v_scores().is_none(), "take consumes");
                return;
            }
            let _ = rm.take_action(actions[idx]);
        }
        panic!("no scored decision fired in 400 steps");
    }
}
