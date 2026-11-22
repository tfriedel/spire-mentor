//! `VisibleRunView` — THE single feature chokepoint for every run-level
//! evaluator (full-run-AI plan §2.5, bead `sts2-rs-40ikw.5`).
//!
//! # Information discipline — HARD RULE (plan §2.5)
//!
//! The engine state is fully deterministic given the seed: event cursors,
//! reward streams, and encounter rolls are all *readable in advance* from
//! [`RunManager`]'s RNG state. An evaluator that peeks is clairvoyant —
//! it posts inflated, meaningless win rates and poisons every training
//! label downstream. Therefore:
//!
//! 1. **Every evaluator (heuristic V, ranker, projection, rollout
//!    scorer) takes `&VisibleRunView`, never `&RunManager`.** This view
//!    exposes ONLY player-visible information: deck, relics, gold,
//!    HP/max-HP, potions, map topology + room types, pinned boss
//!    identity, act/floor, ascension, seen events, and offers *currently
//!    on screen*.
//! 2. **No RNG stream state, no cursor positions, no unrevealed
//!    contents** ever cross this boundary. Chance nodes (event-room
//!    arrivals, unrevealed rewards) are modeled as distributions over
//!    visible pools, not by reading the cursor's next value.
//!
//! Structural enforcement (one chokepoint type) beats reviewer
//! discipline — if a feature isn't on this struct, an evaluator cannot
//! use it.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use sts2_core::cards::CardId;
use sts2_core::events::EventId;
use sts2_core::map::{MapCoord, MapPointType};
use sts2_core::potions::{CharacterId, PotionKind};
use sts2_core::run_manager::{Phase, RunManager};
use sts2_core::{MonsterKind, RelicId};

/// One master-deck (or offered) card: identity + upgrade flag. Order in
/// [`VisibleRunView::deck`] mirrors `RunState::master_deck` order, so
/// subset index spaces (upgradable subset, removable subset) computed
/// from the view line up with the engine's `RunAction` index spaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeckCard {
    pub card_id: CardId,
    pub upgraded: bool,
}

/// Pinned act-boss identity — player-visible from the map screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BossView {
    pub lead_kind: MonsterKind,
    pub display_name: &'static str,
}

/// One grid-live map node: room type + outgoing edges. A node with no
/// live children is on the final row and connects to the boss.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapNode {
    pub room: MapPointType,
    pub children: Vec<MapCoord>,
}

/// Player-visible act-map topology (room types are revealed on the STS
/// map screen; node *contents* — encounter identity, event identity —
/// are not, and are deliberately absent).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct MapView {
    /// Grid-live nodes only (pruned ghosts excluded), keyed by coord.
    pub nodes: BTreeMap<MapCoord, MapNode>,
    /// Player position; `None` before the first move (row 0).
    pub current: Option<MapCoord>,
    /// Legal first edges when `current` is `None` (start-node children).
    pub start_children: Vec<MapCoord>,
    /// Boss meta-node coord (final-row nodes implicitly connect here).
    pub boss_coord: Option<MapCoord>,
    /// Row index of the boss row (`ActMap::map_length`).
    pub boss_row: i32,
}

/// One shop slot the player can currently see (price + payload).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShopCardOffer {
    pub card: DeckCard,
    pub price: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShopRelicOffer {
    pub relic_id: RelicId,
    pub price: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShopPotionOffer {
    pub potion_kind: PotionKind,
    pub price: i32,
}

/// The shop inventory currently on screen. Card offers are the
/// concatenated `[character_cards… colorless_cards…]` slice in the same
/// index space as [`sts2_core::run_manager::RunAction::ShopBuyCard`];
/// sold-out slots appear as `None` entries so indices stay aligned.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ShopView {
    pub cards: Vec<Option<ShopCardOffer>>,
    pub relics: Vec<Option<ShopRelicOffer>>,
    pub potions: Vec<Option<ShopPotionOffer>>,
    pub removal_cost: i32,
    pub removal_used: bool,
}

/// Offers currently on screen (empty/`None` when the matching screen is
/// not up). These are *revealed* contents — showing them respects §2.5
/// because the player sees exactly the same thing.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Offers {
    /// CardReward options (with upgrade flags when rolled upgraded).
    pub card_rewards: Vec<DeckCard>,
    /// Head of the relic-reward queue (CardReward relic sub-state).
    pub relic_reward: Option<RelicId>,
    /// BossRelic phase 3-option pool.
    pub boss_relics: Vec<RelicId>,
    /// Shop inventory (Shop phase with a wired inventory).
    pub shop: Option<ShopView>,
    /// Smith pending-choice options (RestSite smith sub-state),
    /// parallel to `RunAction::SmithCard{idx}`.
    pub smith_options: Vec<DeckCard>,
}

/// Player-visible snapshot of a run. See the module docs for the §2.5
/// hard rule. Built via [`VisibleRunView::from_run_manager`]; mutated
/// only through the pure `with_*` candidate-application helpers (the M1
/// ΔV pick path — cloning `RunManager` for real rollouts is M2).
#[derive(Clone, Debug, PartialEq)]
pub struct VisibleRunView {
    pub character: CharacterId,
    pub ascension: u8,
    /// Act index, **0-based (0..=2)** — normalized from the engine's
    /// 1-based `RunState::act_index` at [`VisibleRunView::from_run_manager`]
    /// (GH #63). Per-act 0-based tables (`projection.rs`'s `*_hp_loss[act]`)
    /// and the `(act_index + …)/3` progress ramps (`act_progress`,
    /// `act_boss_progress`) read it directly — no `- 1` conversion. The
    /// engine field stays 1-based; only this view mirror is normalized.
    pub act_index: u8,
    /// 1-based floor within the act.
    pub act_floor: i32,
    pub hp: i32,
    pub max_hp: i32,
    pub gold: i32,
    /// Master deck in `master_deck` order.
    pub deck: Vec<DeckCard>,
    pub relics: Vec<RelicId>,
    pub potions: Vec<PotionKind>,
    pub max_potion_slots: i32,
    /// Pinned act-boss identity (map-screen visible).
    pub boss: Option<BossView>,
    pub map: Option<MapView>,
    /// Events already visited this run (their pool entries are spent).
    pub seen_events: BTreeSet<EventId>,
    /// The event being resolved right now (Event phase).
    pub current_event: Option<EventId>,
    pub phase: Phase,
    pub offers: Offers,
}

impl VisibleRunView {
    /// Extract the player-visible snapshot from a live [`RunManager`].
    /// This is the ONLY place run-AI code touches `RunManager` state for
    /// feature purposes — everything it copies is on-screen information.
    #[must_use]
    pub fn from_run_manager(rm: &RunManager) -> Self {
        let rs = &rm.run_state;

        let map = rs.current_act_map.as_ref().map(|act_map| {
            let boss_coord = act_map.boss.map(|b| act_map.point(b).coord);
            let mut nodes: BTreeMap<MapCoord, MapNode> = BTreeMap::new();
            for (coord, &idx) in &act_map.grid {
                if Some(idx) == act_map.boss {
                    continue;
                }
                let p = act_map.point(idx);
                let children: Vec<MapCoord> = p
                    .children
                    .iter()
                    .map(|&ci| act_map.point(ci).coord)
                    .filter(|c| act_map.grid.contains_key(c) || Some(*c) == boss_coord)
                    .collect();
                nodes.insert(
                    *coord,
                    MapNode {
                        room: p.point_type,
                        children,
                    },
                );
            }
            let start_children: Vec<MapCoord> = act_map
                .start
                .map(|s| {
                    act_map
                        .children_of(act_map.point(s).coord)
                        .into_iter()
                        .filter(|p| act_map.grid.contains_key(&p.coord))
                        .map(|p| p.coord)
                        .collect()
                })
                .unwrap_or_default();
            MapView {
                nodes,
                current: rs.current_room.and_then(|r| r.map_coord),
                start_children,
                boss_coord,
                boss_row: act_map.map_length,
            }
        });

        let upgraded_flags = rm.card_reward_upgraded_flags();
        let card_rewards: Vec<DeckCard> = rm
            .card_reward_options()
            .iter()
            .enumerate()
            .map(|(i, &card_id)| DeckCard {
                card_id,
                upgraded: upgraded_flags.get(i).copied().unwrap_or(false),
            })
            .collect();

        // Prices as the engine charges them: Membership Card / The Courier
        // discounts included (`merchant_price_with_modifiers`, the same
        // fold `RunManager`'s affordability gate and the PPO obs use).
        let shop_relics = rs.active_relics();
        let price_of =
            |base: i32| sts2_core::shop::merchant_price_with_modifiers(&shop_relics, base);
        let shop = rm.shop_inventory().map(|inv| ShopView {
            cards: inv
                .cards
                .iter()
                .chain(inv.colorless_cards.iter())
                .map(|e| {
                    e.card
                        .as_ref()
                        .filter(|_| sts2_core::shop::is_shop_entry_available(e.price))
                        .map(|c| ShopCardOffer {
                            card: DeckCard {
                                card_id: c.card_id,
                                upgraded: c.upgraded,
                            },
                            price: price_of(e.price),
                        })
                })
                .collect(),
            relics: inv
                .relics
                .iter()
                .map(|e| {
                    e.relic_id
                        .filter(|_| sts2_core::shop::is_shop_entry_available(e.price))
                        .map(|relic_id| ShopRelicOffer {
                            relic_id,
                            price: price_of(e.price),
                        })
                })
                .collect(),
            potions: inv
                .potions
                .iter()
                .map(|e| {
                    e.potion_kind
                        .filter(|_| sts2_core::shop::is_shop_entry_available(e.price))
                        .map(|potion_kind| ShopPotionOffer {
                            potion_kind,
                            price: price_of(e.price),
                        })
                })
                .collect(),
            removal_cost: price_of(inv.removal_cost),
            removal_used: inv.removal_used,
        });

        // Smith sub-state: RestSite + an open CardSelect pending choice.
        let smith_options: Vec<DeckCard> = match rs.pending_choice.as_ref() {
            Some(pc)
                if rm.phase() == Phase::RestSite
                    && pc.kind == sts2_core::ChoiceKind::CardSelect =>
            {
                pc.options
                    .iter()
                    .map(|o| DeckCard {
                        card_id: o.card_id,
                        upgraded: o.upgraded,
                    })
                    .collect()
            }
            _ => Vec::new(),
        };

        Self {
            character: rm.character_id,
            ascension: rm.ascension_level,
            // GH #63: normalize the engine's 1-based act_index (1..=3) to
            // 0-based (0..=2) here, the single view chokepoint, so the
            // progress ramps and per-act table lookups are convention-correct.
            act_index: rs.act_index.saturating_sub(1),
            act_floor: rs.act_floor,
            hp: rs.current_hp,
            max_hp: rs.max_hp,
            gold: rs.gold,
            deck: rs
                .master_deck
                .iter()
                .map(|c| DeckCard {
                    card_id: c.card_id,
                    upgraded: c.upgraded,
                })
                .collect(),
            relics: rs.relics.clone(),
            potions: rs.potions.clone(),
            max_potion_slots: rs.max_potion_slots,
            boss: rs.pinned_act_boss.as_ref().map(|b| BossView {
                lead_kind: b.lead_kind,
                display_name: b.display_name,
            }),
            map,
            seen_events: rs.visited_event_ids.clone(),
            current_event: rs.current_event,
            phase: rm.phase(),
            offers: Offers {
                card_rewards,
                relic_reward: rs.relic_rewards.first().map(|r| r.relic_id),
                boss_relics: rm.boss_relic_options().to_vec(),
                shop,
                smith_options,
            },
        }
    }

    /// Fraction of the run completed, in [0, 1] — acts 0..=2, each act's
    /// floors interpolating within its third. Used as the HP-importance
    /// ramp in `HeuristicV`.
    #[must_use]
    pub fn act_progress(&self) -> f64 {
        let boss_row = self
            .map
            .as_ref()
            .map_or(16.0, |m| f64::from(m.boss_row.max(1)));
        let within = (f64::from(self.act_floor) / boss_row).clamp(0.0, 1.0);
        ((f64::from(self.act_index) + within) / 3.0).clamp(0.0, 1.0)
    }

    /// Run-completion progress at *boss arrival for the current act* — i.e.
    /// the end of this act, `(act_index + 1) / 3` in [0, 1]. Used as the
    /// HP-importance ramp in pathing projection (`projection::finish_score`),
    /// which weighs HP at the act boss the path is walking toward. With the
    /// 0-based `act_index`, end-of-act-`i` is `(i + 1)/3` (act 0 → 1/3,
    /// act 2 → 1.0). Single source for the convention (GH #63).
    #[must_use]
    pub fn act_boss_progress(&self) -> f64 {
        ((f64::from(self.act_index) + 1.0) / 3.0).clamp(0.0, 1.0)
    }

    /// Proximity to the current act boss in [0, 1] (potion boss-insurance
    /// ramp).
    #[must_use]
    pub fn boss_proximity(&self) -> f64 {
        let boss_row = self
            .map
            .as_ref()
            .map_or(16.0, |m| f64::from(m.boss_row.max(1)));
        (f64::from(self.act_floor) / boss_row).clamp(0.0, 1.0)
    }

    // ── Pure candidate-application helpers (M1 ΔV picks, plan §2) ────
    //
    // These apply a *candidate outcome* to a cloned view so the picker
    // can score `V(view + candidate) − V(view)`. They deliberately do
    // NOT clone or simulate the RunManager — that is M2's rollout path.

    /// View with `card` appended to the deck (C# deck-adds append at the
    /// tail).
    #[must_use]
    pub fn with_card_added(&self, card_id: CardId, upgraded: bool) -> Self {
        let mut v = self.clone();
        v.deck.push(DeckCard { card_id, upgraded });
        v
    }

    /// View with the deck card at `idx` removed. No-op if out of range.
    #[must_use]
    pub fn with_deck_card_removed(&self, idx: usize) -> Self {
        let mut v = self.clone();
        if idx < v.deck.len() {
            v.deck.remove(idx);
        }
        v
    }

    /// View with the deck card at `idx` upgraded. No-op if out of range.
    #[must_use]
    pub fn with_deck_card_upgraded(&self, idx: usize) -> Self {
        let mut v = self.clone();
        if let Some(c) = v.deck.get_mut(idx) {
            c.upgraded = true;
        }
        v
    }

    /// View with `relic_id` appended to the relic list.
    #[must_use]
    pub fn with_relic_added(&self, relic_id: RelicId) -> Self {
        let mut v = self.clone();
        v.relics.push(relic_id);
        v
    }

    /// View with gold adjusted by `delta` (clamped at 0).
    #[must_use]
    pub fn with_gold_delta(&self, delta: i32) -> Self {
        let mut v = self.clone();
        v.gold = (v.gold + delta).max(0);
        v
    }

    /// View with `kind` added to the potion belt (no belt-capacity
    /// check — callers gate on capacity).
    #[must_use]
    pub fn with_potion_added(&self, kind: PotionKind) -> Self {
        let mut v = self.clone();
        v.potions.push(kind);
        v
    }

    /// View with current HP set to `hp` (clamped to `[0, max_hp]`).
    #[must_use]
    pub fn with_hp(&self, hp: i32) -> Self {
        let mut v = self.clone();
        v.hp = hp.clamp(0, v.max_hp);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts2_content::run_callbacks::new_run_manager_for_character;

    #[test]
    fn view_from_fresh_ironclad_run() {
        let rm = new_run_manager_for_character(3, CharacterId::Ironclad);
        let view = VisibleRunView::from_run_manager(&rm);
        assert_eq!(view.character, CharacterId::Ironclad);
        assert!(view.deck.len() >= 9, "starter deck present");
        assert!(view.hp > 0 && view.hp <= view.max_hp);
        let map = view.map.as_ref().expect("act map present");
        assert!(!map.nodes.is_empty(), "map nodes visible");
        assert!(!map.start_children.is_empty(), "start-row edges enumerable");
        assert!(view.boss.is_some(), "pinned boss identity visible");
        // Fresh run: nothing on screen yet.
        assert!(view.offers.card_rewards.is_empty());
        assert!(view.offers.shop.is_none());
    }

    /// The shop view carries the prices the engine charges: The Courier's
    /// and Membership Card's discounts included, as `RunManager`'s
    /// affordability gate and the PPO shop obs apply them. Base prices made
    /// the heuristic policies skip discounted items they could afford.
    #[test]
    fn shop_view_prices_include_merchant_discounts() {
        let mut rm = new_run_manager_for_character(7, CharacterId::Ironclad);
        let inv = sts2_content::cards::spec_index::generate_merchant_inventory(
            &mut rm.run_state,
            CharacterId::Ironclad,
        );
        rm.run_state.relics.push(RelicId::MembershipCard);
        rm.run_state.relics.push(RelicId::TheCourier);
        rm.debug_open_shop(inv.clone());
        let relics = rm.run_state.active_relics();
        let price = |base| sts2_core::shop::merchant_price_with_modifiers(&relics, base);
        let shop = VisibleRunView::from_run_manager(&rm)
            .offers
            .shop
            .expect("shop on screen");
        let card_prices: Vec<i32> = shop.cards.iter().flatten().map(|o| o.price).collect();
        let want: Vec<i32> = inv
            .cards
            .iter()
            .chain(inv.colorless_cards.iter())
            .filter(|e| e.card.is_some() && sts2_core::shop::is_shop_entry_available(e.price))
            .map(|e| price(e.price))
            .collect();
        assert!(!want.is_empty());
        assert_eq!(card_prices, want);
        let base: Vec<i32> = inv
            .cards
            .iter()
            .chain(inv.colorless_cards.iter())
            .filter(|e| e.card.is_some() && sts2_core::shop::is_shop_entry_available(e.price))
            .map(|e| e.price)
            .collect();
        assert_ne!(card_prices, base, "the discounts move the prices");
        let relic_prices: Vec<i32> = shop.relics.iter().flatten().map(|o| o.price).collect();
        let want_relics: Vec<i32> = inv
            .relics
            .iter()
            .filter(|e| e.relic_id.is_some() && sts2_core::shop::is_shop_entry_available(e.price))
            .map(|e| price(e.price))
            .collect();
        assert_eq!(relic_prices, want_relics);
        let potion_prices: Vec<i32> = shop.potions.iter().flatten().map(|o| o.price).collect();
        let want_potions: Vec<i32> = inv
            .potions
            .iter()
            .filter(|e| {
                e.potion_kind.is_some() && sts2_core::shop::is_shop_entry_available(e.price)
            })
            .map(|e| price(e.price))
            .collect();
        assert_eq!(potion_prices, want_potions);
        assert_eq!(shop.removal_cost, price(inv.removal_cost));
    }

    #[test]
    fn with_helpers_are_pure() {
        let rm = new_run_manager_for_character(5, CharacterId::Ironclad);
        let view = VisibleRunView::from_run_manager(&rm);
        let n = view.deck.len();
        let added = view.with_card_added(CardId::Inflame, true);
        assert_eq!(view.deck.len(), n, "original untouched");
        assert_eq!(added.deck.len(), n + 1);
        assert!(added.deck.last().unwrap().upgraded);

        let removed = view.with_deck_card_removed(0);
        assert_eq!(removed.deck.len(), n - 1);

        let upgraded = view.with_deck_card_upgraded(0);
        assert!(upgraded.deck[0].upgraded);
        assert!(!view.deck[0].upgraded);

        assert_eq!(view.with_gold_delta(-1_000_000).gold, 0);
        assert_eq!(view.with_hp(-5).hp, 0);
        assert_eq!(view.with_hp(10_000).hp, view.max_hp);
    }

    // ── GH #63: act_index convention (0-based on the view) ──────────────
    //
    // `RunState::act_index` is 1-based (1..=3); `VisibleRunView` normalizes
    // it to 0-based (0..=2) at `from_run_manager`, so `act_progress()` /
    // `act_boss_progress()` read a 0-based index in their `(idx + …)/3`
    // formulas. These tests pin the convention so it can't silently drift.

    /// Build a view at a given 0-based act / within-act floor, map-less so
    /// `boss_row` falls back to the 16.0 default (within = act_floor/16).
    fn view_at(act_index_0based: u8, act_floor: i32) -> VisibleRunView {
        let rm = new_run_manager_for_character(3, CharacterId::Ironclad);
        let mut v = VisibleRunView::from_run_manager(&rm);
        v.act_index = act_index_0based;
        v.act_floor = act_floor;
        v.map = None;
        v
    }

    #[test]
    fn from_run_manager_normalizes_act_index_to_zero_based() {
        let rm = new_run_manager_for_character(3, CharacterId::Ironclad);
        // A fresh run sits in engine act 1 (RunState::act_index == 1); the
        // view must mirror it as 0-based act 0.
        assert_eq!(rm.run_state.act_index, 1, "engine stays 1-based");
        let view = VisibleRunView::from_run_manager(&rm);
        assert_eq!(view.act_index, 0, "view normalizes act 1 -> 0");
    }

    #[test]
    fn act_progress_ramps_zero_based_across_acts() {
        // Start-of-act (floor 0, within = 0): each act's third opens at idx/3.
        assert!(
            (view_at(0, 0).act_progress() - 0.0).abs() < 1e-9,
            "act 1 opens at 0.0"
        );
        assert!(
            (view_at(1, 0).act_progress() - 1.0 / 3.0).abs() < 1e-9,
            "act 2 opens at 1/3"
        );
        assert!(
            (view_at(2, 0).act_progress() - 2.0 / 3.0).abs() < 1e-9,
            "act 3 opens at 2/3"
        );
        // End of act 1 (within = 1.0 at floor == boss_row default 16) meets
        // the start of act 2's third.
        assert!(
            (view_at(0, 16).act_progress() - 1.0 / 3.0).abs() < 1e-9,
            "act-1 boss == 1/3"
        );
    }

    #[test]
    fn act_boss_progress_is_end_of_act() {
        // boss-arrival importance is evaluated at the END of the current act:
        // end-of-act-i (0-based) == (i + 1)/3.
        assert!(
            (view_at(0, 0).act_boss_progress() - 1.0 / 3.0).abs() < 1e-9,
            "act 1 boss at 1/3"
        );
        assert!(
            (view_at(1, 0).act_boss_progress() - 2.0 / 3.0).abs() < 1e-9,
            "act 2 boss at 2/3"
        );
        assert!(
            (view_at(2, 0).act_boss_progress() - 1.0).abs() < 1e-9,
            "act 3 boss at 1.0"
        );
    }
}
