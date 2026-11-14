//! The row layout of `ppo.e2e.contracts` (schema `ironclad-e2e-v111` rev 2):
//! sections, the categorical (`cat`, int16 `[T, 32]`) and continuous (`cont`,
//! float16 `[T, 210]`) columns, token flags, room and intent ids.
//!
//! Column indices are looked up by name at compile time ([`xc`], [`xf`],
//! [`sc`], [`sec`]), so a name that drifts from the tables below fails the
//! build instead of writing the wrong column.
//!
//! The "chars" fill event (GH #1373, `contracts` section 4b) names reserved
//! sections and spare columns by alias ([`CHARS_ALIASES`]); the lookups
//! resolve an alias to its slot.

/// S_ctx categorical slots (`SCTX_SLOTS`), columns 0-14 of `cat`.
pub const SCTX_SLOTS: [&str; 15] = [
    "card",
    "upgraded",
    "relic",
    "potion",
    "power",
    "monster",
    "enchantment",
    "affliction",
    "intent",
    "enemy_slot",
    "hand_slot",
    "choice_kind",
    "card_type",
    "choice_pile",
    "rl2key",
];
/// S_ctx categorical slots.
pub const SCTX_N_CAT: usize = 15;
/// S_ctx continuous columns.
pub const SCTX_N_CONT: usize = 122;

/// The trailing categorical columns (`X_CAT`), columns 15-31 of `cat`.
pub const X_CAT: [&str; 17] = [
    "X_DTYPE",
    "X_KIND",
    "X_ID_MAIN",
    "X_ID_MOD",
    "X_ID_REL",
    "X_ID_AUX",
    "X_ROOM",
    "X_POS",
    "X_INTENT_T2",
    "X_INTENT_T3",
    "X_FC1",
    "X_FC2",
    "X_FC3",
    "X_RES0",
    "X_RES1",
    "X_RES2",
    "X_RES3",
];
/// `cat` width.
pub const N_CAT: usize = SCTX_N_CAT + X_CAT.len();

/// The 122 S_ctx continuous columns (`SCTX_CONT`), in column order.
pub const SCTX_CONT: [&str; SCTX_N_CONT] = [
    "COST",
    "BASE_COST",
    "DMG",
    "BLK",
    "CALC",
    "HAS_CALC",
    "ETHEREAL",
    "UNPLAYABLE",
    "INNATE",
    "SLY",
    "SLY_TURN",
    "RETAIN_TURN",
    "ENCH_AMT",
    "AFF_STACKS",
    "COUNT",
    "PLAYABLE",
    "DMG_PREV",
    "BLK_PREV",
    "HP_FRAC",
    "HP",
    "MAXHP",
    "BLOCK",
    "ENERGY",
    "MAXE",
    "STARS",
    "ROUND",
    "PLAYED",
    "N_HAND",
    "N_DRAW",
    "N_DISC",
    "N_EXH",
    "POT_SLOTS",
    "IS_BOSS",
    "IS_ELITE",
    "IS_HALL",
    "CH_MIN",
    "CH_MAX",
    "CH_SKIP",
    "INCOMING",
    "UNBLOCKED",
    "E_HPF",
    "E_HP",
    "E_MAXHP",
    "E_BLOCK",
    "E_ALIVE",
    "E_REMOVED",
    "E_AMT",
    "E_EFF",
    "E_TIMES",
    "E_TOT",
    "H_LAG",
    "H_AMT",
    "H_EFF",
    "H_TIMES",
    "H_ALIVE",
    "P_AMT",
    "P_DISP",
    "P_CTR",
    "R_CTR",
    "R_HAS",
    "R_WAX",
    "R_MELT",
    "T_HPF",
    "T_EHP",
    "T_DMG",
    "N_SEL",
    "RL2_S00",
    "RL2_S01",
    "RL2_S02",
    "RL2_S03",
    "RL2_S04",
    "RL2_S05",
    "RL2_S06",
    "RL2_S07",
    "RL2_S08",
    "RL2_S09",
    "RL2_S10",
    "RL2_S11",
    "RL2_S12",
    "RL2_S13",
    "RL2_S14",
    "RL2_S15",
    "RL2_S16",
    "RL2_S17",
    "RL2_S18",
    "RL2_S19",
    "RL2_S20",
    "RL2_S21",
    "RL2_S22",
    "RL2_S23",
    "RL2_S24",
    "RL2_S25",
    "RL2_S26",
    "RL2_S27",
    "RL2_S28",
    "RL2_S29",
    "RL2_S30",
    "RL2_S31",
    "RL2_X00",
    "RL2_X01",
    "RL2_X02",
    "RL2_X03",
    "RL2_X04",
    "RL2_X05",
    "RL2_X06",
    "RL2_X07",
    "RL2_X08",
    "RL2_X09",
    "RL2_X10",
    "RL2_X11",
    "RL2_X12",
    "RL2_X13",
    "RL2_X14",
    "RL2_X15",
    "RL2_X16",
    "RL2_X17",
    "RL2_X18",
    "RL2_X19",
    "RL2_X20",
    "RL2_X21",
    "RL2_X22",
    "RL2_X23",
];
/// First rl2 state column (`SCTX_RL2_STATE_COLS`).
pub const C_RL2S: usize = 66;
/// First rl2 action column (`SCTX_RL2_ACTION_COLS`).
pub const C_RL2X: usize = 98;
/// rl2 state features.
pub const RL2_STATE_DIM: usize = 32;
/// rl2 action features.
pub const RL2_ACTION_DIM: usize = 24;

/// The trailing continuous columns (`X_CONT`), columns 122-209 of `cont`.
pub const X_CONT: [&str; 88] = [
    // RUNCTX
    "R_ACT",
    "R_FLOOR",
    "R_ACT_FLOOR",
    "R_ROWS_TO_BOSS",
    "R_BOSSES_LEFT",
    "R_HP",
    "R_MAXHP",
    "R_HP_FRAC",
    "R_GOLD",
    "R_DECK",
    "R_DECK_UPG",
    "R_RELICS",
    "R_POTIONS",
    "R_POT_SLOTS",
    "R_ASC",
    "R_IN_COMBAT",
    "R_SEL_MIN",
    "R_SEL_MAX",
    "R_SEL_PICKED",
    "R_SEL_CAN_SKIP",
    "R_N_CLASSES",
    "R_DOUBLE_BOSS",
    "R_HIST_MISSING",
    "R_RES0",
    // route
    "M_ELITE_MIN",
    "M_ELITE_MAX",
    "M_REST_MIN",
    "M_REST_MAX",
    "M_SHOP_MAX",
    "M_UNKNOWN_MAX",
    "M_FIGHTS_TO_REST",
    "M_ROWS_LEFT",
    // MAPNODE
    "N_ROW_OFF",
    "N_COL",
    "N_NEXT",
    "N_PATHS",
    // options
    "O_PRICE",
    "O_EV_GOLD",
    "O_EV_HP",
    "O_EV_MAXHP",
    "O_EV_RELICS_GAIN",
    "O_EV_POTIONS_GAIN",
    "O_EV_CARDS_GAIN",
    "O_EV_CARDS_LOSE",
    "O_EV_RELICS_LOSE",
    "O_EV_POTIONS_LOSE",
    // card extras
    "K_PICKED",
    "K_DECK_COPIES",
    // summaries
    "U_N",
    "U_UPG",
    "U_ENCH",
    "U_DISTINCT",
    // BOSS
    "B_ORDER",
    "B_ACT",
    // ENEMY extras
    "E_N_INTENTS",
    "E_T2_EFF",
    "E_T2_TIMES",
    "E_T3_EFF",
    "E_T3_TIMES",
    "E_F1_CERT",
    "E_F1_AMT",
    "E_F1_TIMES",
    "E_F1_TOT",
    "E_F2_CERT",
    "E_F2_AMT",
    "E_F2_TIMES",
    "E_F2_TOT",
    "E_F3_CERT",
    "E_F3_AMT",
    "E_F3_TIMES",
    "E_F3_TOT",
    "E_MAXDMG",
    "E_KILLABLE",
    // STATE extras
    "S_HP_LOSS_END",
    // PRIV_RUN
    "P_ROOMS_MONSTER",
    "P_ROOMS_ELITE",
    "P_ROOMS_EVENT",
    "P_ROOMS_SHOP",
    "P_ROOMS_REST",
    "P_ROOMS_TREASURE",
    // spare
    "X_C_RES0",
    "X_C_RES1",
    "X_C_RES2",
    "X_C_RES3",
    "X_C_RES4",
    "X_C_RES5",
    "X_C_RES6",
    "X_C_RES7",
];
/// `cont` width.
pub const N_CONT: usize = SCTX_N_CONT + X_CONT.len();

/// Sections (`SECTIONS`), the section embedding's 64 rows.
pub const SECTIONS: [&str; 64] = [
    "STATE",
    "HAND",
    "DRAW_UNK",
    "DRAW_TOP",
    "DRAW_BOT",
    "DISCARD",
    "EXHAUST",
    "PLAY",
    "RELIC",
    "POTION",
    "PPOWER",
    "ENEMY",
    "EPOWER",
    "EHIST",
    "ACT_PLAY",
    "ACT_END",
    "ACT_POTION",
    "ACT_CHOOSE",
    "ACT_SKIP",
    "ACT_OTHER",
    "ACT_RL2",
    "PRIV_DRAW",
    "RUNCTX",
    "BOSS",
    "SUM_DRAW",
    "SUM_DISCARD",
    "SUM_EXHAUST",
    "SUM_DECK",
    "SUM_RELICS",
    "DECK",
    "OFFER",
    "MAPNODE",
    "OPT_CARD",
    "OPT_RELIC",
    "OPT_POTION",
    "OPT_MOVE",
    "OPT_EVENT",
    "OPT_REST",
    "OPT_FIXED",
    "RES_SHOP_SHELF",
    "RES_OPT_REWARD",
    "RES_OPT_SPHERE",
    "RES_OPT_POTION_USE",
    "RES_OPT_MERCHANT",
    "RES_44",
    "RES_45",
    "RES_46",
    "RES_47",
    "PRIV_ENEMY",
    "PRIV_RUN",
    "RES_PRIV_50",
    "RES_PRIV_51",
    "RES_PRIV_52",
    "RES_PRIV_53",
    "RES_PRIV_54",
    "RES_PRIV_55",
    "RES_56",
    "RES_57",
    "RES_58",
    "RES_59",
    "RES_60",
    "RES_61",
    "RES_62",
    "RES_63",
];

/// Room-type ids (`ROOM_TYPES`, the `X_ROOM` column).
pub const ROOM_TYPES: [&str; 11] = [
    "none",
    "unassigned",
    "unknown",
    "monster",
    "elite",
    "boss",
    "shop",
    "rest_site",
    "treasure",
    "ancient",
    "event",
];

/// `SCTX_INTENT_KINDS` without `NONE`: `INTENT_IDS[k] = 1 + index`.
pub const INTENT_KINDS: [&str; 13] = [
    "Attack",
    "Defend",
    "Buff",
    "Debuff",
    "DebuffStrong",
    "Sleep",
    "Stun",
    "StatusCard",
    "Summon",
    "Heal",
    "DeathBlow",
    "Escape",
    "Unknown",
];

/// An option token (a member of some class).
pub const TF_OPTION: u8 = 1;
/// The class representative.
pub const TF_REP: u8 = 2;
/// Hidden from the student (a non-representative member).
pub const TF_HIDDEN: u8 = 4;
/// Inside the combat teacher's view (the S_ctx block).
pub const TF_TEACHER: u8 = 8;
/// A pooled summary token (owns pool entries).
pub const TF_POOLED: u8 = 16;

/// MAPNODE tokens per row at most (`MAX_MAPNODES`).
pub const MAX_MAPNODES: usize = 40;
/// BOSS tokens per row at most (`MAX_BOSS_TOKENS`).
pub const MAX_BOSS_TOKENS: usize = 2;

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The "chars" fill event's names (`contracts` section 4b, GH #1373): each
/// alias and the reserved section or spare column it takes. The tables above
/// keep the reserved names, as the hashed tables of `contracts` do.
pub const CHARS_ALIASES: [(&str, &str); 6] = [
    ("ALLY", "RES_44"),
    ("ORB", "RES_45"),
    ("X_CHAR", "X_RES0"),
    ("C_ORB_PASSIVE", "X_C_RES0"),
    ("C_ORB_EVOKE", "X_C_RES1"),
    ("E_DOOMED", "X_C_RES2"),
];
/// ORB tokens per row at most (`MAX_ORB_TOKENS`: a belt has at most 10 slots).
pub const MAX_ORB_TOKENS: usize = 10;
/// `contracts.CHARACTERS`: X_CHAR is 1 + the index; 0, the column's absent
/// value, is a row without "chars" data.
pub const CHARACTERS: [&str; 6] = [
    "Ironclad",
    "Silent",
    "Defect",
    "Regent",
    "Necrobinder",
    "Fool",
];

/// `contracts.char_id(name)`: X_CHAR of a run's character.
#[must_use]
pub fn char_id(name: &str) -> Option<i16> {
    CHARACTERS
        .iter()
        .position(|c| *c == name)
        .and_then(|i| i16::try_from(i + 1).ok())
}

const fn unalias(name: &str) -> &str {
    let mut i = 0;
    while i < CHARS_ALIASES.len() {
        if str_eq(CHARS_ALIASES[i].0, name) {
            return CHARS_ALIASES[i].1;
        }
        i += 1;
    }
    name
}

const fn find(names: &[&str], name: &str) -> usize {
    let name = unalias(name);
    let mut i = 0;
    while i < names.len() {
        if str_eq(names[i], name) {
            return i;
        }
        i += 1;
    }
    panic!("unknown column or section name");
}

/// The `cat` column of trailing categorical `name` (`XC[name]`).
#[must_use]
pub const fn xc(name: &str) -> usize {
    SCTX_N_CAT + find(&X_CAT, name)
}

/// The `cont` column of trailing continuous `name` (`XF[name]`).
#[must_use]
pub const fn xf(name: &str) -> usize {
    SCTX_N_CONT + find(&X_CONT, name)
}

/// The `cont` column of S_ctx continuous `name` (`SCTX_C[name]`).
#[must_use]
pub const fn sc(name: &str) -> usize {
    find(&SCTX_CONT, name)
}

/// The `cat` column of S_ctx slot `name` (`S[name]`).
#[must_use]
pub const fn slot(name: &str) -> usize {
    find(&SCTX_SLOTS, name)
}

/// Section id of `name` (`SEC[name]`).
#[must_use]
pub const fn sec(name: &str) -> u8 {
    #[allow(clippy::cast_possible_truncation)]
    {
        find(&SECTIONS, name) as u8
    }
}

/// `ROOM_ID[name]`.
#[must_use]
pub fn room_id(name: &str) -> Option<i16> {
    ROOM_TYPES
        .iter()
        .position(|r| *r == name)
        .and_then(|i| i16::try_from(i).ok())
}

/// `INTENT_IDS.get(kind, 0)`.
#[must_use]
pub fn intent_id(kind: &str) -> i16 {
    INTENT_KINDS
        .iter()
        .position(|k| *k == kind)
        .and_then(|i| i16::try_from(i + 1).ok())
        .unwrap_or(0)
}

/// The name of `cat` column `c`.
#[must_use]
pub fn cat_name(c: usize) -> &'static str {
    if c < SCTX_N_CAT {
        SCTX_SLOTS[c]
    } else {
        X_CAT.get(c - SCTX_N_CAT).copied().unwrap_or("?")
    }
}

/// The name of `cont` column `c`.
#[must_use]
pub fn cont_name(c: usize) -> &'static str {
    if c < SCTX_N_CONT {
        SCTX_CONT[c]
    } else {
        X_CONT.get(c - SCTX_N_CONT).copied().unwrap_or("?")
    }
}

/// The name of section `s`.
#[must_use]
pub fn section_name(s: u8) -> &'static str {
    SECTIONS.get(usize::from(s)).copied().unwrap_or("?")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_match_contracts() {
        assert_eq!(N_CAT, 32);
        assert_eq!(N_CONT, 210);
        assert_eq!(xf("R_ACT"), 122);
        assert_eq!(xf("X_C_RES7"), 209);
        assert_eq!(xc("X_DTYPE"), 15);
        assert_eq!(sc("N_SEL"), 65);
        assert_eq!(sc("RL2_S00"), C_RL2S);
        assert_eq!(sc("RL2_X00"), C_RL2X);
        assert_eq!(sec("RUNCTX"), 22);
        assert_eq!(sec("PRIV_RUN"), 49);
        assert_eq!(sec("OPT_FIXED"), 38);
        // The "chars" aliases resolve to their reserved slots.
        assert_eq!((sec("ALLY"), sec("ORB")), (44, 45));
        assert_eq!(xc("X_CHAR"), xc("X_RES0"));
        assert_eq!(xf("C_ORB_PASSIVE"), 202);
        assert_eq!(xf("C_ORB_EVOKE"), 203);
        assert_eq!(xf("E_DOOMED"), 204);
        assert_eq!(char_id("Ironclad"), Some(1));
        assert_eq!(char_id("Necrobinder"), Some(5));
        assert_eq!(char_id("Watcher"), None);
    }
}
