//! The e2e content-ID table (`ppo.e2e.vocab`): one id per content name,
//! shared by every `X_ID_*` column and the pool entries.
//!
//! The table is `ppo/e2e/vocab.tsv` (game version v0.111.0), embedded
//! verbatim as `data/vocab_v0111.tsv`. Id 0 is NONE, ids 1-11 are
//! `UNK_<registry>`. Names are looked up through each registry's canonical
//! form (`CANON`).

use std::collections::HashMap;
use std::sync::OnceLock;

const TSV: &str = include_str!("../data/vocab_v0111.tsv");

/// A registry block of the table (`REGISTRIES`, in block order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Reg {
    Card,
    Relic,
    Potion,
    Power,
    Monster,
    Encounter,
    Event,
    Enchantment,
    Orb,
    Rest,
    Move,
}

impl Reg {
    const ALL: [Self; 11] = [
        Self::Card,
        Self::Relic,
        Self::Potion,
        Self::Power,
        Self::Monster,
        Self::Encounter,
        Self::Event,
        Self::Enchantment,
        Self::Orb,
        Self::Rest,
        Self::Move,
    ];

    fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "card" => Self::Card,
            "relic" => Self::Relic,
            "potion" => Self::Potion,
            "power" => Self::Power,
            "monster" => Self::Monster,
            "encounter" => Self::Encounter,
            "event" => Self::Event,
            "enchantment" => Self::Enchantment,
            "orb" => Self::Orb,
            "rest" => Self::Rest,
            "move" => Self::Move,
            _ => return None,
        })
    }

    /// `UNK_ID[registry]`.
    #[must_use]
    pub fn unk(self) -> u16 {
        let i = Self::ALL.iter().position(|r| *r == self).unwrap_or(0);
        u16::try_from(i + 1).unwrap_or(0)
    }

    /// `CANON[registry](name)`.
    #[must_use]
    pub fn canon(self, name: &str) -> String {
        match self {
            Self::Card | Self::Relic | Self::Potion => sts2_vocab::canonical(name),
            Self::Power | Self::Encounter | Self::Orb | Self::Rest => upper(name),
            Self::Monster | Self::Event => alnum_upper(name),
            Self::Enchantment => enchantment_norm(name),
            Self::Move => {
                let (species, mv) = name.split_once('.').unwrap_or((name, ""));
                format!("{}.{}", alnum_upper(species), upper(mv))
            }
        }
    }
}

/// The NONE id.
pub const NONE_ID: u16 = 0;

/// `str(name).strip().upper()`.
fn upper(name: &str) -> String {
    name.trim().to_uppercase()
}

/// `_NON_ALNUM.sub("", name.upper())` (the event canonicalizer's camel
/// underscores are dropped by the same substitution).
fn alnum_upper(name: &str) -> String {
    name.to_uppercase()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect()
}

/// `tokens.enchantment_norm` / `vocab._canon_enchantment`: `Tezcatara's
/// Ember` -> `TEZCATARAS_EMBER`.
#[must_use]
pub fn enchantment_norm(name: &str) -> String {
    let s = sts2_vocab::canonical(&name.replace(['\'', '\u{2019}'], ""));
    let mut out = String::with_capacity(s.len());
    let mut sep = false;
    for c in s.chars() {
        if c.is_ascii_uppercase() || c.is_ascii_digit() {
            if sep && !out.is_empty() {
                out.push('_');
            }
            sep = false;
            out.push(c);
        } else {
            sep = true;
        }
    }
    out
}

/// The loaded table.
#[derive(Debug)]
pub struct Vocab {
    idx: HashMap<(Reg, String), u16>,
}

impl Vocab {
    // Lookup tables built once per process (registry-sized).
    #[allow(clippy::disallowed_methods)]
    fn parse(text: &str) -> Self {
        let mut idx = HashMap::new();
        for line in text.lines() {
            let line = line.trim_end_matches('\r');
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split('\t');
            let (Some(id), Some(reg), Some(name)) = (it.next(), it.next(), it.next()) else {
                panic!("vocab.tsv: bad line {line:?}");
            };
            let id: u16 = id.parse().expect("vocab.tsv: id");
            if let Some(r) = Reg::from_name(reg) {
                idx.insert((r, r.canon(name)), id);
            }
        }
        Self { idx }
    }

    /// `Vocab.id(registry, name)`: NONE for `None`/empty, `UNK_<registry>`
    /// when the table lacks it.
    #[must_use]
    pub fn id(&self, reg: Reg, name: Option<&str>) -> u16 {
        match name {
            None | Some("") => NONE_ID,
            Some(n) => self
                .idx
                .get(&(reg, reg.canon(n)))
                .copied()
                .unwrap_or_else(|| reg.unk()),
        }
    }

    /// `Vocab.has(registry, name)`.
    #[must_use]
    pub fn has(&self, reg: Reg, name: &str) -> bool {
        self.idx.contains_key(&(reg, reg.canon(name)))
    }
}

/// The committed table (`vocab.load()`).
#[must_use]
pub fn load() -> &'static Vocab {
    static V: OnceLock<Vocab> = OnceLock::new();
    V.get_or_init(|| Vocab::parse(TSV))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_ids() {
        let v = load();
        assert_eq!(v.id(Reg::Card, None), 0);
        assert_eq!(v.id(Reg::Card, Some("")), 0);
        assert_eq!(v.id(Reg::Card, Some("NoSuchCardAtAll")), 1);
        assert_eq!(v.id(Reg::Move, Some("x")), 11);
        assert_eq!(
            v.id(Reg::Card, Some("StrikeIronclad")),
            v.id(Reg::Card, Some("CARD.STRIKE_IRONCLAD"))
        );
        assert!(v.id(Reg::Card, Some("StrikeIronclad")) >= 32);
        assert_eq!(enchantment_norm("Tezcatara's Ember"), "TEZCATARAS_EMBER");
    }
}
