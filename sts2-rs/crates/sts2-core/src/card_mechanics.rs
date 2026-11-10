//! What a card does to the orb belt and to Osty, the two companion systems a
//! card's id, type, cost, damage and block do not reveal.
//!
//! Any character can hold any character's cards (generated cards, Neow and
//! event rewards, relic-stamped pools), and both systems work for everyone:
//! C# `OrbCmd.Channel` (`OrbCmd.cs:68-90`) grows a one-slot belt for a
//! character whose `BaseOrbSlotCount` is 0, and `OstyCmd.Summon` raises an
//! Osty for any player. What does NOT carry across is the AI's knowledge of
//! them. A playout net or a move prior keys each card to its own learned row,
//! so a Zap on an Ironclad reads as an untrained 1-cost Skill with no damage
//! and no block, and a Poke as a 6-damage Attack even when there is no Osty
//! to deal it (it then does nothing: `Osty.CheckMissingWithAnim`). This table
//! is the shared, card-independent description those consumers can read
//! instead (`sts2_mcts::rollout_net`'s v5 features).
//!
//! Every row is transcribed from the v0.107.1 decompiled card models (plus
//! v0.111.0's `Hibernate`), and
//! `sts2-content/tests/card_mechanics_decomp.rs` checks the four card sets
//! against the decomp so a card that gains or loses one of these effects in a
//! new build fails loudly.

use crate::cards::{Card, CardId};

/// Orb kinds in feature order (the [`OrbChannels::by_kind`] index).
pub const ORB_KIND_ORDER: [crate::orbs::OrbKind; 5] = [
    crate::orbs::OrbKind::Lightning,
    crate::orbs::OrbKind::Frost,
    crate::orbs::OrbKind::Dark,
    crate::orbs::OrbKind::Plasma,
    crate::orbs::OrbKind::Glass,
];

const L: usize = 0;
const F: usize = 1;
const D: usize = 2;
const P: usize = 3;
const G: usize = 4;

/// How many orbs one play channels, when the count is not a constant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChannelRule {
    /// [`CardMechanics::channels`] is the count.
    #[default]
    Fixed,
    /// One per X (`Tempest`: `ResolveEnergyXValue()`, +1 when upgraded —
    /// the +1 is already in `channels`).
    PerX,
    /// One per hittable enemy (`Chill`: `foreach HittableEnemies`).
    PerEnemy,
    /// One per Lightning channeled earlier this combat (`Voltaic`'s
    /// `CalculatedChannels`: the `OrbChanneledEntry` history count).
    PerLightningSoFar,
}

/// The orb and Osty effects of one card at one upgrade level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CardMechanics {
    /// Orbs channeled by one play, per [`ORB_KIND_ORDER`] kind. Under a
    /// non-[`ChannelRule::Fixed`] rule this is the count per unit, plus a
    /// flat bonus where the card has one.
    pub channels: [u8; 5],
    /// Orbs of a random kind channeled by one play (`Chaos`:
    /// `OrbModel.GetRandomOrb`).
    pub channels_random: u8,
    /// How [`Self::channels`] scales.
    pub rule: ChannelRule,
    /// Reads or changes the belt beyond channeling: evokes (`Dualcast`,
    /// `MultiCast`, `Quadcast`, `Shatter`), scales with the orbs held
    /// (`Barrage`, `CompileDriver`, `Synchronize`, `TeslaCoil`, `Darkness`),
    /// changes Focus or slots (`Defragment`, `Hotfix`, `Capacitor`, `BulkUp`,
    /// …), or is a Power that keeps working the belt (`Loop`, `Storm`,
    /// `ConsumingShadow`, `Spinner`, …). Such a card is dead weight without
    /// orbs, which on a character other than Defect means without a channel
    /// card to feed it.
    pub orb_interaction: bool,
    /// Osty HP summoned by one play (C# `SummonVar`; per X for `Dirge`).
    pub summon: u8,
    /// Whether the summon is per X (`Dirge`).
    pub summon_per_x: bool,
    /// Damage dealt by Osty, not the player (`CardTag.OstyAttack`): the
    /// player's Strength, Weak and Vigor do not apply to it.
    pub osty_attack: bool,
    /// Does nothing without a living Osty (`Osty.CheckMissingWithAnim`): every
    /// Osty attack, and `Sacrifice`.
    pub needs_osty: bool,
}

impl CardMechanics {
    /// Whether the card touches neither system.
    #[must_use]
    pub const fn is_none(&self) -> bool {
        !self.channels_any() && !self.orb_interaction && self.summon == 0 && !self.needs_osty
    }

    /// Whether any play of the card can channel an orb.
    #[must_use]
    pub const fn channels_any(&self) -> bool {
        let c = self.channels;
        c[0] + c[1] + c[2] + c[3] + c[4] + self.channels_random > 0
            || !matches!(self.rule, ChannelRule::Fixed)
    }
}

const fn ch(kind: usize, n: u8) -> CardMechanics {
    let mut channels = [0u8; 5];
    channels[kind] = n;
    CardMechanics {
        channels,
        channels_random: 0,
        rule: ChannelRule::Fixed,
        orb_interaction: false,
        summon: 0,
        summon_per_x: false,
        osty_attack: false,
        needs_osty: false,
    }
}

const fn with_interaction(mut m: CardMechanics) -> CardMechanics {
    m.orb_interaction = true;
    m
}

const fn with_rule(mut m: CardMechanics, rule: ChannelRule) -> CardMechanics {
    m.rule = rule;
    m
}

const NONE: CardMechanics = CardMechanics {
    channels: [0; 5],
    channels_random: 0,
    rule: ChannelRule::Fixed,
    orb_interaction: false,
    summon: 0,
    summon_per_x: false,
    osty_attack: false,
    needs_osty: false,
};

const fn summon(n: u8) -> CardMechanics {
    let mut m = NONE;
    m.summon = n;
    m
}

/// The mechanics of `id` at its upgraded (`upgraded`) or base level.
#[must_use]
pub const fn card_mechanics(id: CardId, upgraded: bool) -> CardMechanics {
    let u = upgraded;
    let m = match id {
        // ── Channel ─────────────────────────────────────────────────────
        CardId::BallLightning | CardId::Zap => ch(L, 1),
        CardId::ColdSnap | CardId::Coolheaded => ch(F, 1),
        CardId::Glacier => ch(F, 2),
        CardId::IceLance => ch(F, 3),
        CardId::Fusion | CardId::Ignition => ch(P, 1),
        CardId::MeteorStrike => ch(P, 3),
        CardId::NullCard | CardId::ShadowShield => ch(D, 1),
        CardId::Glasswork => ch(G, 1),
        CardId::Refract => ch(G, 2),
        // v0.111.0 only: `Hibernate` channels 2 Frost (3 upgraded).
        CardId::HibernateCard => ch(F, if u { 3 } else { 2 }),
        CardId::Rainbow => CardMechanics {
            channels: [1, 1, 1, 0, 0],
            ..NONE
        },
        CardId::Chaos => CardMechanics {
            channels_random: if u { 2 } else { 1 },
            ..NONE
        },
        CardId::Chill => with_rule(ch(F, 1), ChannelRule::PerEnemy),
        CardId::Tempest => with_rule(ch(L, if u { 1 } else { 0 }), ChannelRule::PerX),
        CardId::Voltaic => with_rule(ch(L, 0), ChannelRule::PerLightningSoFar),
        // Darkness channels 1 Dark, then triggers every Dark orb's passive.
        CardId::DarknessCard => with_interaction(ch(D, 1)),
        // Powers that channel on play and keep working the belt.
        CardId::ConsumingShadow => with_interaction(ch(D, if u { 3 } else { 2 })),
        CardId::SpinnerCard => with_interaction(ch(G, if u { 1 } else { 0 })),

        // ── Belt payoffs / engines (no channel of their own) ────────────
        CardId::Barrage
        | CardId::BiasedCognitionCard
        | CardId::BulkUp
        | CardId::Capacitor
        | CardId::CompileDriver
        | CardId::Coolant
        | CardId::Defragment
        | CardId::Dualcast
        | CardId::FocusedStrikeCard
        | CardId::Hailstorm
        | CardId::Hotfix
        | CardId::Hyperbeam
        | CardId::LightningRod
        | CardId::LoopCard
        | CardId::Modded
        | CardId::MultiCast
        | CardId::Quadcast
        | CardId::Shatter
        | CardId::StormCard
        | CardId::Synchronize
        | CardId::TeslaCoil
        | CardId::TrashToTreasure => with_interaction(NONE),

        // ── Osty summons (`SummonVar` base / upgraded) ──────────────────
        CardId::Afterlife => summon(if u { 9 } else { 6 }),
        CardId::Bodyguard => summon(if u { 7 } else { 5 }),
        CardId::Cleanse | CardId::Spur => summon(if u { 5 } else { 3 }),
        CardId::Dirge => CardMechanics {
            summon: if u { 4 } else { 3 },
            summon_per_x: true,
            ..NONE
        },
        CardId::LegionOfBone => summon(if u { 8 } else { 6 }),
        CardId::NecroMasteryCard => summon(if u { 8 } else { 5 }),
        CardId::PullAggro => summon(if u { 5 } else { 4 }),
        CardId::Reanimate => summon(if u { 25 } else { 20 }),

        // ── Needs a living Osty ─────────────────────────────────────────
        CardId::Sacrifice => CardMechanics {
            needs_osty: true,
            ..NONE
        },
        _ => NONE,
    };
    if id.is_osty_attack() {
        CardMechanics {
            osty_attack: true,
            needs_osty: true,
            ..m
        }
    } else {
        m
    }
}

/// What one play of a card does to the belt and to Osty right now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OrbChannels {
    /// Orbs channeled, per [`ORB_KIND_ORDER`] kind.
    pub by_kind: [u32; 5],
    /// Orbs of a random kind channeled.
    pub random: u32,
}

impl OrbChannels {
    /// Every orb the play channels.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.by_kind.iter().sum::<u32>() + self.random
    }
}

/// The orbs `card` would channel if played now, its `rule` resolved against
/// `x_value` (the X an X-cost play would spend), `hittable_enemies` and the
/// combat's Lightning history.
#[must_use]
pub fn resolve_channels(
    state: &crate::CombatState,
    card: &Card,
    x_value: i32,
    hittable_enemies: u32,
) -> OrbChannels {
    let m = card_mechanics(card.card_id, card.upgraded);
    let mut by_kind = m.channels.map(u32::from);
    match m.rule {
        ChannelRule::Fixed => {}
        ChannelRule::PerEnemy => {
            for n in &mut by_kind {
                *n *= hittable_enemies;
            }
        }
        // Tempest: X Lightning on top of its flat upgrade bonus.
        ChannelRule::PerX => by_kind[L] += u32::try_from(x_value.max(0)).unwrap_or(0),
        ChannelRule::PerLightningSoFar => {
            by_kind[L] = u32::from(state.combat_log.lightning_orbs_channeled_this_combat);
        }
    }
    OrbChannels {
        by_kind,
        random: u32::from(m.channels_random),
    }
}

/// The Osty HP one play of `card` summons now (`x_value` for `Dirge`).
#[must_use]
pub fn resolve_summon(card: &Card, x_value: i32) -> u32 {
    let m = card_mechanics(card.card_id, card.upgraded);
    let n = u32::from(m.summon);
    if m.summon_per_x {
        n * u32::try_from(x_value.max(0)).unwrap_or(0)
    } else {
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(id: CardId, upgraded: bool) -> Card {
        Card {
            card_id: id,
            upgraded,
            ..Card::pod_defaults()
        }
    }

    #[test]
    fn strike_touches_neither_system() {
        assert!(card_mechanics(CardId::StrikeIronclad, false).is_none());
        assert!(card_mechanics(CardId::Bash, true).is_none());
    }

    #[test]
    fn osty_attacks_need_osty() {
        for id in CardId::all() {
            let m = card_mechanics(*id, false);
            assert_eq!(m.osty_attack, id.is_osty_attack(), "{id:?}");
            if m.osty_attack {
                assert!(m.needs_osty, "{id:?}");
            }
        }
        let s = card_mechanics(CardId::Sacrifice, false);
        assert!(s.needs_osty && !s.osty_attack);
    }

    #[test]
    fn fixed_channels_resolve_per_kind() {
        let state = crate::CombatState::default();
        let z = resolve_channels(&state, &card(CardId::Zap, false), 0, 1);
        assert_eq!(z.by_kind, [1, 0, 0, 0, 0]);
        let r = resolve_channels(&state, &card(CardId::Rainbow, false), 0, 1);
        assert_eq!(r.by_kind, [1, 1, 1, 0, 0]);
        let m = resolve_channels(&state, &card(CardId::MeteorStrike, true), 0, 1);
        assert_eq!(m.by_kind, [0, 0, 0, 3, 0]);
        let c = resolve_channels(&state, &card(CardId::Chaos, true), 0, 1);
        assert_eq!((c.by_kind, c.random, c.total()), ([0; 5], 2, 2));
        assert_eq!(
            resolve_channels(&state, &card(CardId::StrikeIronclad, false), 3, 2).total(),
            0
        );
    }

    #[test]
    fn variable_channels_resolve_against_the_board() {
        let mut state = crate::CombatState::default();
        // Tempest: X Lightning, +1 upgraded.
        let t = resolve_channels(&state, &card(CardId::Tempest, false), 3, 1);
        assert_eq!(t.by_kind, [3, 0, 0, 0, 0]);
        let t = resolve_channels(&state, &card(CardId::Tempest, true), 3, 1);
        assert_eq!(t.by_kind, [4, 0, 0, 0, 0]);
        // Chill: one Frost per hittable enemy.
        let c = resolve_channels(&state, &card(CardId::Chill, false), 0, 3);
        assert_eq!(c.by_kind, [0, 3, 0, 0, 0]);
        // Voltaic: the Lightning channeled so far this combat.
        assert_eq!(
            resolve_channels(&state, &card(CardId::Voltaic, false), 0, 1).total(),
            0
        );
        state.combat_log.lightning_orbs_channeled_this_combat = 4;
        let v = resolve_channels(&state, &card(CardId::Voltaic, false), 0, 1);
        assert_eq!(v.by_kind, [4, 0, 0, 0, 0]);
    }

    #[test]
    fn summons_resolve_with_upgrade_and_x() {
        assert_eq!(resolve_summon(&card(CardId::Bodyguard, false), 0), 5);
        assert_eq!(resolve_summon(&card(CardId::Bodyguard, true), 0), 7);
        assert_eq!(resolve_summon(&card(CardId::Dirge, false), 2), 6);
        assert_eq!(resolve_summon(&card(CardId::Dirge, true), 2), 8);
        assert_eq!(resolve_summon(&card(CardId::Poke, false), 2), 0);
    }
}
