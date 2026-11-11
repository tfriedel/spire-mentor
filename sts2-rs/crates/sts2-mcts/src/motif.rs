//! Position motifs for the combat decision-quality suite (GH #446).
//!
//! A motif is a cheap, deterministic predicate on a *position* — never on
//! the answer — that names the kind of decision the position poses. The
//! position book stores them per entry so a regression on `gold_agr` says
//! **what** broke ("block-vs-telegraphed-damage dropped 6pp") rather than
//! only that a scalar moved. Two measured facts motivate the scheme: the
//! search is ~10pp weaker on the block axis than the damage axis of the
//! proven ordering pairs, and elite positions score 15pp below common in
//! skill-above-chance. A single number hid both.
//!
//! Every detector here is a lower-bound heuristic in the same spirit as
//! [`super::audit`] — it must never panic, never consult a search, and
//! must be a pure function of the [`CombatState`] so the tags are
//! reproducible from a frozen `state_hash`.
//!
//! Three tags are **not** position predicates and are supplied by the
//! caller, because they depend on facts the book knows and the state does
//! not: [`Motif::ProvenPair`] (an engine-proven ordering exists here — the
//! only class where the gold answer can be *audited* rather than merely
//! trusted, which is why it is tagged and scored separately instead of
//! excluded), and [`Motif::GoldEndTurn`] / [`Motif::GoldPotion`], which
//! name the shape of the frozen answer. They are in the same enum so the
//! whole vocabulary has one spelling.

use sts2_core::monsters::Intent;
use sts2_core::{legal_actions, Action, CombatState};

/// HP fraction (percent) at or below which a position is tagged
/// [`Motif::LowHp`]. Chosen to match the survival-pressure band the
/// lethal-fight autopsy found attrition deaths cluster in.
pub const LOW_HP_PCT: i32 = 30;

/// One motif tag. `as_str` is the on-disk spelling in the position book
/// and the column key in every per-motif table, so it is append-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Motif {
    /// The conservative lethal detector says the hand kills every living
    /// enemy this turn. Anything but taking the kill is a blunder.
    LethalAvailable,
    /// At least one living enemy telegraphs an Attack / MultiAttack. The
    /// block-versus-damage trade is live.
    IncomingAttack,
    /// Telegraphed damage meets or exceeds `hp + block`: the position must
    /// be answered or the run ends.
    LethalThreat,
    /// No living enemy telegraphs an attack. Block played here is (mostly)
    /// wasted — the `defend_no_attack_intent` audit class.
    NoAttackIntent,
    /// A 0-cost card is legal to play. Ending the turn with it in hand is
    /// the `end_turn_with_playable` shape at its cheapest.
    FreePlayInHand,
    /// A Power card is legal to play — the permanent-buff timing question.
    PowerInHand,
    /// A potion can be drunk. Potion timing is the class the potion-at-boss
    /// study found the search under-prices.
    PotionAvailable,
    /// Player HP at or below [`LOW_HP_PCT`] percent of max.
    LowHp,
    /// Two or more living enemies: a targeting decision exists.
    MultiEnemy,
    /// Caller-supplied: the position carries at least one engine-proven
    /// ordering pair (see `common::dominance` in the bench crate).
    ProvenPair,
    /// Caller-supplied: the frozen gold answer is `EndTurn`.
    GoldEndTurn,
    /// Caller-supplied: the frozen gold answer drinks a potion.
    GoldPotion,
}

impl Motif {
    /// Every motif, in report order.
    pub const ALL: [Self; 12] = [
        Self::LethalAvailable,
        Self::IncomingAttack,
        Self::LethalThreat,
        Self::NoAttackIntent,
        Self::FreePlayInHand,
        Self::PowerInHand,
        Self::PotionAvailable,
        Self::LowHp,
        Self::MultiEnemy,
        Self::ProvenPair,
        Self::GoldEndTurn,
        Self::GoldPotion,
    ];

    /// Stable on-disk / column spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LethalAvailable => "lethal_available",
            Self::IncomingAttack => "incoming_attack",
            Self::LethalThreat => "lethal_threat",
            Self::NoAttackIntent => "no_attack_intent",
            Self::FreePlayInHand => "free_play_in_hand",
            Self::PowerInHand => "power_in_hand",
            Self::PotionAvailable => "potion_available",
            Self::LowHp => "low_hp",
            Self::MultiEnemy => "multi_enemy",
            Self::ProvenPair => "proven_pair",
            Self::GoldEndTurn => "gold_end_turn",
            Self::GoldPotion => "gold_potion",
        }
    }

    /// Inverse of [`Self::as_str`]. `None` for an unknown spelling so a
    /// book written by a newer binary degrades to "untagged" rather than
    /// failing to load.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str() == s)
    }

    /// The answer-shape tags for a gold action rendered with `Debug`
    /// (`"EndTurn"`, `"DrinkPotion { .. }"`, `"PlayCard { .. }"`).
    #[must_use]
    pub fn from_gold_action(gold: &str) -> Vec<Self> {
        let mut out = Vec::new();
        if gold.starts_with("EndTurn") {
            out.push(Self::GoldEndTurn);
        }
        // `DrinkPotionAtAlly` starts with "DrinkPotion" too: a throw is a
        // potion move.
        if gold.starts_with("DrinkPotion") || gold.starts_with("UsePotion") {
            out.push(Self::GoldPotion);
        }
        out
    }
}

impl std::fmt::Display for Motif {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Player HP / block through the unified `Creature` mirror when the
/// combat has one, falling back to the legacy scalars. Same rule as the
/// dominance proof — reading only the legacy field goes stale in any
/// combat routed through the mirror.
fn player_hp_block(s: &CombatState) -> (i32, i32) {
    s.player
        .creature_uid
        .and_then(|u| s.creatures.iter().find(|c| c.id == u.0))
        .map_or((s.player.current_hp, s.player.block), |c| {
            (c.current_hp, c.block)
        })
}

fn living_enemies(s: &CombatState) -> impl Iterator<Item = &sts2_core::Creature> {
    s.creatures
        .iter()
        .filter(move |c| c.current_hp > 0 && s.enemy_uids.iter().any(|u| u.0 == c.id))
}

/// Telegraphed attack damage summed over living enemies. Lower bound: the
/// intent payload is the base per-hit number and does not fold in
/// Strength, Vulnerable or block — the same convention as the replay
/// audit's `lethal_intent_ignored`.
#[must_use]
pub fn incoming_attack_damage(s: &CombatState) -> i32 {
    living_enemies(s)
        .filter_map(|c| c.intent.as_ref())
        .map(|i| match *i {
            Intent::Attack { damage } => damage.max(0),
            Intent::MultiAttack { damage, hits } => damage.max(0) * i32::from(hits),
            _ => 0,
        })
        .sum()
}

fn any_attack_intent(s: &CombatState) -> bool {
    living_enemies(s)
        .filter_map(|c| c.intent.as_ref())
        .any(|i| matches!(i, Intent::Attack { .. } | Intent::MultiAttack { .. }))
}

/// Position-only motifs at `state`. Sorted and de-duplicated; empty when
/// the position is not a live play-phase decision.
#[must_use]
pub fn motifs_at(state: &CombatState) -> Vec<Motif> {
    let mut out = Vec::new();
    if state.is_over {
        return out;
    }
    let legal = legal_actions(state);

    if crate::detect_lethal(state) {
        out.push(Motif::LethalAvailable);
    }

    let (hp, block) = player_hp_block(state);
    let incoming = incoming_attack_damage(state);
    if any_attack_intent(state) {
        out.push(Motif::IncomingAttack);
        if incoming >= hp + block {
            out.push(Motif::LethalThreat);
        }
    } else {
        out.push(Motif::NoAttackIntent);
    }

    let mut free_play = false;
    let mut power_play = false;
    let mut potion = false;
    for a in &legal {
        match a {
            Action::PlayCard { hand_index, .. } | Action::PlayCardAtAlly { hand_index, .. } => {
                let Some(uid) = state.player.hand.get(usize::from(*hand_index)) else {
                    continue;
                };
                let Some(card) = state.cards.iter().find(|c| c.uid == *uid) else {
                    continue;
                };
                if card.base_cost() >= 0 && card.local_cost() <= 0 {
                    free_play = true;
                }
                if card.card_type == sts2_core::cards::CardType::Power {
                    power_play = true;
                }
            }
            Action::DrinkPotion { .. }
            | Action::UsePotion { .. }
            | Action::DrinkPotionAtAlly { .. } => {
                potion = true;
            }
            _ => {}
        }
    }
    if free_play {
        out.push(Motif::FreePlayInHand);
    }
    if power_play {
        out.push(Motif::PowerInHand);
    }
    if potion {
        out.push(Motif::PotionAvailable);
    }

    let max_hp = state.player.max_hp.max(1);
    if hp * 100 <= LOW_HP_PCT * max_hp {
        out.push(Motif::LowHp);
    }
    if living_enemies(state).count() >= 2 {
        out.push(Motif::MultiEnemy);
    }

    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use smallvec::SmallVec;
    use sts2_core::cards::{Card, CardId, CardTags, CardType, TargetType};
    use sts2_core::{
        CardUid, CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId,
        MonsterKind, PlayerCombatState, Side,
    };

    /// (card id, cost, base damage, target, type)
    type HandSpec = (CardId, i8, i32, TargetType, CardType);

    fn build(
        hand: &[HandSpec],
        monsters: &[(i32, Option<Intent>)],
        energy: i8,
        hp: i32,
    ) -> CombatState {
        sts2_content::cards::dispatch::ensure_installed();
        let mut cards = Vec::new();
        let mut hand_uids: SmallVec<[CardUid; 16]> = SmallVec::new();
        for (i, &(id, cost, dmg, target, card_type)) in hand.iter().enumerate() {
            let uid = CardUid::new((i + 1) as u32).unwrap();
            cards.push(Card {
                uid,
                card_id: id,
                cost: sts2_core::CardEnergyCost::new(cost),
                card_type,
                target_type: target,
                upgraded: false,
                base_damage: dmg,
                base_block: 0,
                combat_vars: sts2_core::cards::CombatVars::new(),
                enchantments: sts2_core::cards::CardEnchantments::new(),
                tags: CardTags::NONE,
                energy_spent: 0,
                base_replay_count: 0,
                keyword_override: None,
                affliction: None,
            });
            hand_uids.push(uid);
        }
        let mut creatures = Vec::new();
        let mut enemy_uids: SmallVec<[CreatureUid; 8]> = SmallVec::new();
        for (i, (mhp, intent)) in monsters.iter().enumerate() {
            let uid = CreatureUid::new((i + 1) as u32).unwrap();
            creatures.push(Creature {
                id: uid.0,
                current_hp: *mhp,
                max_hp: (*mhp).max(1),
                block: 0,
                powers: SmallVec::new(),
                species: Some(MonsterKind::Nibbit),
                monster_id: MonsterId("Nibbit"),
                intent: *intent,
                intent_history: IntentHistory::default(),
                ai: None,
            });
            enemy_uids.push(uid);
        }
        let player = PlayerCombatState {
            hand: hand_uids,
            energy: i32::from(energy),
            max_energy: energy.max(3),
            current_hp: hp,
            max_hp: 80,
            ..PlayerCombatState::default()
        };
        CombatState {
            phase: CombatPhase::PlayerPlayPhase,
            round: 1,
            side: Side::Player,
            rng: sts2_core::dotnet_rng::Rng::new(0_i32),
            creatures,
            cards: cards.into(),
            enemy_uids,
            player,
            ..CombatState::default()
        }
    }

    const STRIKE: HandSpec = (
        CardId::StrikeIronclad,
        1,
        6,
        TargetType::Enemy,
        CardType::Attack,
    );
    const DEFEND: HandSpec = (
        CardId::DefendIronclad,
        1,
        0,
        TargetType::Self_,
        CardType::Skill,
    );

    #[test]
    fn spelling_round_trips_for_every_motif() {
        for m in Motif::ALL {
            assert_eq!(Motif::parse(m.as_str()), Some(m), "{m:?}");
        }
        assert_eq!(Motif::parse("not_a_motif"), None);
        let mut names: Vec<&str> = Motif::ALL.iter().map(|m| m.as_str()).collect();
        names.dedup();
        assert_eq!(names.len(), Motif::ALL.len(), "as_str must be injective");
    }

    #[test]
    fn gold_action_tags() {
        assert_eq!(Motif::from_gold_action("EndTurn"), vec![Motif::GoldEndTurn]);
        assert_eq!(
            Motif::from_gold_action("DrinkPotion { potion_slot: 0, target: None }"),
            vec![Motif::GoldPotion]
        );
        assert!(Motif::from_gold_action("PlayCard { hand_index: 0, target: Some(0) }").is_empty());
    }

    #[test]
    fn quiet_enemy_tags_no_attack_intent_only() {
        let s = build(
            &[STRIKE, DEFEND],
            &[(30, Some(Intent::Defend { block: 5 }))],
            3,
            50,
        );
        let m = motifs_at(&s);
        assert!(m.contains(&Motif::NoAttackIntent), "{m:?}");
        assert!(!m.contains(&Motif::IncomingAttack), "{m:?}");
        assert!(!m.contains(&Motif::LethalThreat), "{m:?}");
        assert!(!m.contains(&Motif::FreePlayInHand), "{m:?}");
        assert!(!m.contains(&Motif::PowerInHand), "{m:?}");
        assert!(!m.contains(&Motif::PotionAvailable), "{m:?}");
        assert!(!m.contains(&Motif::LowHp), "{m:?}");
        assert!(!m.contains(&Motif::MultiEnemy), "{m:?}");
    }

    #[test]
    fn telegraphed_attack_is_incoming_and_lethal_when_it_covers_hp_plus_block() {
        let survivable = build(
            &[STRIKE],
            &[(30, Some(Intent::Attack { damage: 12 }))],
            3,
            50,
        );
        let m = motifs_at(&survivable);
        assert!(m.contains(&Motif::IncomingAttack), "{m:?}");
        assert!(!m.contains(&Motif::LethalThreat), "{m:?}");
        assert!(!m.contains(&Motif::NoAttackIntent), "{m:?}");

        let mut lethal = build(
            &[STRIKE],
            &[(30, Some(Intent::MultiAttack { damage: 5, hits: 3 }))],
            3,
            10,
        );
        lethal.player.block = 5;
        let m = motifs_at(&lethal);
        assert_eq!(incoming_attack_damage(&lethal), 15);
        assert!(m.contains(&Motif::LethalThreat), "{m:?}");
        assert!(m.contains(&Motif::LowHp), "{m:?}");
    }

    #[test]
    fn dead_enemies_do_not_telegraph() {
        let s = build(
            &[STRIKE],
            &[
                (0, Some(Intent::Attack { damage: 99 })),
                (
                    20,
                    Some(Intent::Buff {
                        kind: sts2_core::PowerKind::Strength,
                        amount: 2,
                    }),
                ),
            ],
            3,
            50,
        );
        let m = motifs_at(&s);
        assert!(m.contains(&Motif::NoAttackIntent), "{m:?}");
        assert!(
            !m.contains(&Motif::MultiEnemy),
            "only one enemy is alive: {m:?}"
        );
    }

    #[test]
    fn free_play_and_power_in_hand_read_the_legal_cards() {
        let free: HandSpec = (CardId::Anger, 0, 6, TargetType::Enemy, CardType::Attack);
        let power: HandSpec = (CardId::Inflame, 1, 0, TargetType::Self_, CardType::Power);
        let s = build(&[free, power], &[(30, None)], 3, 50);
        let m = motifs_at(&s);
        assert!(m.contains(&Motif::FreePlayInHand), "{m:?}");
        assert!(m.contains(&Motif::PowerInHand), "{m:?}");

        // With no energy the 1-cost power is not legal, so it is not a
        // decision the position poses; the 0-cost card still is.
        let starved = build(&[free, power], &[(30, None)], 0, 50);
        let m = motifs_at(&starved);
        assert!(m.contains(&Motif::FreePlayInHand), "{m:?}");
        assert!(!m.contains(&Motif::PowerInHand), "{m:?}");
    }

    #[test]
    fn multi_enemy_counts_living_enemies() {
        let s = build(&[STRIKE], &[(30, None), (30, None)], 3, 50);
        assert!(motifs_at(&s).contains(&Motif::MultiEnemy));
    }

    #[test]
    fn lethal_available_when_the_hand_kills_everything() {
        // 3 energy, three Strikes at 6 vs a 12-HP enemy: the conservative
        // detector proves the kill.
        let s = build(
            &[STRIKE, STRIKE, STRIKE],
            &[(12, Some(Intent::Attack { damage: 5 }))],
            3,
            50,
        );
        let m = motifs_at(&s);
        assert!(m.contains(&Motif::LethalAvailable), "{m:?}");
        let s = build(&[STRIKE], &[(30, None)], 3, 50);
        assert!(!motifs_at(&s).contains(&Motif::LethalAvailable));
    }

    #[test]
    fn finished_combat_has_no_motifs() {
        let mut s = build(&[STRIKE], &[(30, None)], 3, 50);
        s.is_over = true;
        assert!(motifs_at(&s).is_empty());
    }

    #[test]
    fn output_is_sorted_and_deduplicated() {
        let s = build(
            &[STRIKE],
            &[(30, Some(Intent::Attack { damage: 60 })), (30, None)],
            3,
            20,
        );
        let m = motifs_at(&s);
        let mut sorted = m.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(m, sorted);
    }
}
