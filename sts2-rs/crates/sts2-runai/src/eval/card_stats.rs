//! Per-card static features for the heuristic V deck terms (plan §5/M1).
//!
//! Most fields are derived automatically from the `sts2-content`
//! [`CardSpec`] table (cost / base damage / base block / target type /
//! rarity / keywords), with the spec-level `upgrade_delta` applied for
//! upgraded copies. What specs do NOT encode — multi-hit counts, draw /
//! energy effects, scaling tags — comes from a small hand-written
//! override table covering Ironclad standouts (acceptable for M1 per the
//! plan; TODO `sts2-rs-40ikw.5` follow-up: generate the full table from
//! the C# decomp effect metadata instead of hand-curating).
//!
//! Honest-by-construction: unknown cards fall back to spec-only features
//! (or neutral zeros when even the spec is missing), never to invented
//! numbers.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use sts2_content::cards::spec_index::{
    card_pod_from_id, specs_for_character, COLORLESS_SPECS, STATUS_SPECS,
};
use sts2_content::cards::types::{CardRarity, CardSpec};
use sts2_core::cards::{CardId, CardType, TargetType};
use sts2_core::potions::CharacterId;

/// Effective cost assumed for X-cost (`cost == -1`) cards: scoring
/// assumes the player typically spends ~2 energy on them.
const X_COST_EFFECTIVE: f64 = 2.0;

/// Static per-card features consumed by `HeuristicV`'s deck terms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CardStats {
    /// Effective energy cost (X-cost ≈ 2, unplayables 0).
    pub cost: f64,
    /// Expected single-target damage per play (hits × per-hit damage).
    pub damage_per_play: f64,
    /// Block per play.
    pub block_per_play: f64,
    /// Hits all enemies (spec `TargetType::AllEnemies`).
    pub aoe: bool,
    /// Grows the deck's output over a long fight (hand-tagged).
    pub scaling: bool,
    /// Cards drawn on play (hand table).
    pub draw: f64,
    /// Energy gained on play (hand table).
    pub energy: f64,
    /// Status / Curse rarity (dead or harmful deck slot).
    pub curse_like: bool,
    pub is_attack: bool,
}

impl CardStats {
    const NEUTRAL: CardStats = CardStats {
        cost: 1.0,
        damage_per_play: 0.0,
        block_per_play: 0.0,
        aoe: false,
        scaling: false,
        draw: 0.0,
        energy: 0.0,
        curse_like: false,
        is_attack: false,
    };

    /// Frontloaded-damage contribution: damage discounted by cost
    /// (plan §5/M1 "frontloaded damage" curve metric).
    #[must_use]
    pub fn frontload(&self) -> f64 {
        self.damage_per_play / (1.0 + self.cost)
    }
}

/// Hand-written per-card supplement (Ironclad + a few colorless).
/// Values cross-checked against the sts2-content effect handlers (file
/// refs in comments). `hits`/`draw`/`energy` may depend on the upgrade.
struct Overrides {
    hits: f64,
    draw: f64,
    energy: f64,
    scaling: bool,
}

fn overrides(card_id: CardId, upgraded: bool) -> Overrides {
    let mut o = Overrides {
        hits: 1.0,
        draw: 0.0,
        energy: 0.0,
        scaling: false,
    };
    match card_id {
        // Multi-hit attacks (per-hit base_damage in the spec).
        CardId::TwinStrike => o.hits = 2.0, // twin_strike.rs (2 hits)
        CardId::SwordBoomerang => o.hits = if upgraded { 4.0 } else { 3.0 }, // sword_boomerang.rs:24
        // X-cost attacks: assume ~2 activations (X_COST_EFFECTIVE).
        CardId::Whirlwind | CardId::Cascade => o.hits = X_COST_EFFECTIVE,
        // Draw / energy riders (verified in the IC card handlers).
        CardId::PommelStrike => o.draw = if upgraded { 2.0 } else { 1.0 }, // pommel_strike.rs:13
        CardId::ShrugItOff => o.draw = 1.0,                                // shrug_it_off.rs:7
        CardId::BattleTrance => o.draw = if upgraded { 4.0 } else { 3.0 }, // battle_trance.rs:16
        CardId::Offering => {
            o.energy = 2.0; // offering.rs:19
            o.draw = if upgraded { 5.0 } else { 3.0 }; // offering.rs:22
        }
        CardId::Bloodletting => o.energy = if upgraded { 3.0 } else { 2.0 }, // bloodletting.rs:17
        // Scaling tags: cards that compound over a long fight.
        CardId::DemonFormCard
        | CardId::Inflame
        | CardId::RuptureCard
        | CardId::BarricadeCard
        | CardId::JuggernautCard
        | CardId::FeelNoPainCard
        | CardId::DarkEmbraceCard
        | CardId::CorruptionCard
        | CardId::Rampage
        | CardId::PerfectedStrike
        | CardId::Feed => o.scaling = true,
        _ => {}
    }
    o
}

/// Build the stats for one `(spec, upgraded)` flavor. Upgraded numeric
/// values come from [`card_pod_from_id`], which routes through the
/// spec-level `upgrade_delta` table (`apply_spec_upgrade`).
fn build_stats(spec: &CardSpec, upgraded: bool) -> CardStats {
    let (cost, damage, block) = match card_pod_from_id(spec.id, upgraded) {
        Some(pod) => (pod.base_cost(), pod.base_damage, pod.base_block),
        None => (spec.cost, spec.base_damage, spec.base_block),
    };

    let o = overrides(spec.id, upgraded);
    let eff_cost = if cost < 0 {
        X_COST_EFFECTIVE
    } else {
        f64::from(cost)
    };
    let curse_like =
        matches!(spec.rarity, CardRarity::Status | CardRarity::Curse) || spec.is_unplayable();

    CardStats {
        cost: eff_cost,
        damage_per_play: f64::from(damage.max(0)) * o.hits,
        block_per_play: f64::from(block.max(0)),
        aoe: spec.target_type == TargetType::AllEnemies,
        scaling: o.scaling,
        draw: o.draw,
        energy: o.energy,
        curse_like,
        is_attack: spec.card_type == CardType::Attack,
    }
}

/// Lazy `(base, upgraded)` stats index over every character pool +
/// colorless + status/curse.
fn stats_index() -> &'static BTreeMap<CardId, [CardStats; 2]> {
    static CACHE: OnceLock<BTreeMap<CardId, [CardStats; 2]>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut map = BTreeMap::new();
        let char_slices = [
            CharacterId::Ironclad,
            CharacterId::Silent,
            CharacterId::Defect,
            CharacterId::Regent,
            CharacterId::Necrobinder,
        ]
        .into_iter()
        .map(specs_for_character);
        for slice in char_slices.chain([COLORLESS_SPECS, STATUS_SPECS]) {
            for factory in slice {
                let spec = factory();
                map.insert(
                    spec.id,
                    [build_stats(&spec, false), build_stats(&spec, true)],
                );
            }
        }
        map
    })
}

/// Static features for `(card_id, upgraded)`. Unknown ids return
/// [`CardStats::NEUTRAL`] — never invented numbers.
#[must_use]
pub fn card_stats(card_id: CardId, upgraded: bool) -> CardStats {
    stats_index()
        .get(&card_id)
        .map_or(CardStats::NEUTRAL, |pair| pair[usize::from(upgraded)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strike_and_defend_derive_from_specs() {
        let strike = card_stats(CardId::StrikeIronclad, false);
        assert!(strike.is_attack);
        assert!(strike.damage_per_play > 0.0);
        assert!(!strike.aoe);
        let defend = card_stats(CardId::DefendIronclad, false);
        assert!(defend.block_per_play > 0.0);
        assert!(!defend.is_attack);
    }

    #[test]
    fn upgrade_increases_strike_damage() {
        let base = card_stats(CardId::StrikeIronclad, false);
        let up = card_stats(CardId::StrikeIronclad, true);
        assert!(up.damage_per_play > base.damage_per_play);
    }

    #[test]
    fn multi_hit_and_scaling_overrides() {
        let twin = card_stats(CardId::TwinStrike, false);
        let strike = card_stats(CardId::StrikeIronclad, false);
        assert!(twin.damage_per_play > strike.damage_per_play / 2.0);
        assert!(card_stats(CardId::DemonFormCard, false).scaling);
        assert!(card_stats(CardId::Inflame, false).scaling);
    }

    #[test]
    fn aoe_from_spec_target_type() {
        assert!(card_stats(CardId::Thunderclap, false).aoe);
        assert!(card_stats(CardId::Whirlwind, false).aoe);
        assert!(!card_stats(CardId::Bludgeon, false).aoe);
    }
}
