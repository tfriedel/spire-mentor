//! #1081's S_ctx token block (`ppo.combat_ctx.tokens.tokens(obs, actions,
//! rollout_features(bv, 1), "S_ctx")`): the 15 categorical slots and 122
//! continuous columns of every combat token, from the honest snapshot, the
//! actions in the snapshot's option order and rl2's v1 features.
//!
//! The content tables are the engine registries (`[names..., UNK, PAD]`),
//! built once ([`vocab`]); card types are the v0.111.0 static table
//! (`static_tables_v0111.CARD_TYPE`, `data/card_types_v0111.tsv`).
//! [`luts`] maps each content table's rows to e2e vocabulary ids
//! (`ppo.e2e.tokens.luts`).

use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::Value;

use crate::intent_log;
use crate::pyjson::{self, get, list_or_empty, truthy};
use crate::schema::{
    self, slot, C_RL2S, C_RL2X, RL2_ACTION_DIM, RL2_STATE_DIM, SCTX_N_CAT, SCTX_N_CONT,
};
use crate::vocab::{self, Reg};

const CARD_TYPES_TSV: &str = include_str!("../data/card_types_v0111.tsv");

/// `static_tables.CARD_TYPES`.
const CARD_TYPES: [&str; 6] = ["Attack", "Skill", "Power", "Status", "Curse", "Quest"];
const AFFLICTIONS: [&str; 9] = [
    "Bound",
    "Devoured",
    "Entangled",
    "Galvanized",
    "Hexed",
    "Ringing",
    "Smog",
    "Weighted",
    "Tainted",
];
const INTENT_KINDS: [&str; 14] = [
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
    intent_log::NONE_KIND,
];
const CHOICE_KINDS: [&str; 31] = [
    "card_select",
    "put_back",
    "exhaust_pile",
    "orb_slot",
    "enemy_target",
    "add_generated_to_hand",
    "add_generated_retained_to_hand",
    "add_generated_free_to_hand",
    "paels_tooth_stash",
    "upgrade_in_hand",
    "add_from_draw_by_type",
    "add_from_discard_free",
    "discard_from_hand",
    "exhaust_from_hand",
    "discard_from_hand_and_draw",
    "make_free_in_hand",
    "monster_curse_select",
    "add_from_draw_to_hand",
    "foregone_conclusion_from_draw",
    "add_from_discard_to_hand",
    "apply_retain_in_hand",
    "apply_sly_in_hand",
    "apply_ethereal_in_hand",
    "transform_in_hand_to_minion_strike",
    "transform_in_hand_to_minion_sacrifice",
    "transfigure_card_in_hand",
    "transform_draw_card_to_soul",
    "transform_draw_cards_to_minion_dive_bomb",
    "clone_chosen_card_next_turn",
    "retain_at_turn_end",
    "auto_play_skill_from_hand",
];
const CHOICE_PILES: [&str; 6] = ["hand", "draw", "discard", "exhaust", "play", "generated"];
const MAX_ENEMY_SLOTS: usize = 5;
const MAX_HAND_SLOTS: usize = 12;

// ── Python value helpers ─────────────────────────────────────────────────

/// Python's `str(x)` of a JSON value (strings bare, `None`, `True`, ints,
/// float repr).
#[must_use]
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                pyjson::py_float_repr(n.as_f64().unwrap_or(0.0))
            }
        }
        other => pyjson::dumps(other),
    }
}

/// Python's `sum(xs)` of floats: starts from `0` (Rust's float `Sum`
/// starts from `-0.0`, so an empty sum would be `-0.0`).
pub fn pysum(it: impl Iterator<Item = f64>) -> f64 {
    it.fold(0.0, |a, b| a + b)
}

/// Python's `int(x)` (floats truncate, numeric strings parse).
#[must_use]
pub fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => n.as_i64().or_else(|| {
            #[allow(clippy::cast_possible_truncation)]
            n.as_f64()
                .filter(|f| f.is_finite())
                .map(|f| f.trunc() as i64)
        }),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `combat_ctx.tokens._f`: `0.0 if x is None else float(x)`.
fn fz(v: Option<&Value>) -> f64 {
    match v {
        None | Some(Value::Null) => 0.0,
        Some(Value::Bool(b)) => f64::from(u8::from(*b)),
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(0.0),
        Some(_) => 0.0,
    }
}

// ── tables ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Norm {
    Id,
    Canonical,
    Enchantment,
}

impl Norm {
    fn apply(self, s: &str) -> String {
        match self {
            Self::Id => s.to_owned(),
            Self::Canonical => sts2_vocab::canonical(s),
            Self::Enchantment => vocab::enchantment_norm(s),
        }
    }
}

/// A content table `[names..., UNK, PAD]` keyed by a normalizer (first
/// writer wins on a normalized collision).
#[derive(Debug)]
pub struct Table {
    /// The names, in row order.
    pub names: Vec<String>,
    idx: HashMap<String, i16>,
    norm: Norm,
    /// The UNK row.
    pub unk: i16,
    /// The PAD row (the absent value).
    pub pad: i16,
}

impl Table {
    // Lookup tables built once per process (registry-sized).
    #[allow(clippy::disallowed_methods)]
    fn new(names: Vec<String>, norm: Norm) -> Self {
        let mut idx = HashMap::new();
        for (i, n) in names.iter().enumerate() {
            idx.entry(norm.apply(n))
                .or_insert(i16::try_from(i).expect("table fits int16"));
        }
        let unk = i16::try_from(names.len()).expect("table fits int16");
        Self {
            names,
            idx,
            norm,
            unk,
            pad: unk + 1,
        }
    }

    fn of(names: &[&str]) -> Self {
        Self::new(names.iter().map(|s| (*s).to_owned()).collect(), Norm::Id)
    }

    /// The row of `name` (UNK when unknown).
    #[must_use]
    pub fn index(&self, name: &str) -> i16 {
        match self.norm {
            Norm::Id => self.idx.get(name).copied(),
            n => self.idx.get(&n.apply(name)).copied(),
        }
        .unwrap_or(self.unk)
    }

    /// Rows including UNK and PAD.
    #[must_use]
    pub fn size(&self) -> usize {
        self.names.len() + 2
    }
}

/// The 15 S_ctx tables (`combat_ctx.tokens.vocab()`), in slot order.
#[derive(Debug)]
pub struct SctxVocab {
    /// One table per slot.
    pub tables: Vec<Table>,
    /// PAD of each slot (the absent row).
    pub pads: [i16; SCTX_N_CAT],
    card_type: HashMap<String, &'static str>,
    card_type_canon: HashMap<String, &'static str>,
}

fn reg_names(kind: &str) -> Vec<String> {
    sts2_wire::state_dict::registry_names_for(kind)
        .unwrap_or_else(|e| panic!("registry {kind}: {e}"))
}

fn rl2_key_universe(cards: &[String], potions: &[String]) -> Vec<String> {
    let mut out = vec![
        "end_turn".to_owned(),
        "choice".to_owned(),
        "other".to_owned(),
    ];
    for c in cards {
        out.push(format!("card:{c}"));
        out.push(format!("card:{c}+"));
    }
    out.extend(potions.iter().map(|p| format!("potion:{p}")));
    out
}

impl SctxVocab {
    // Lookup tables built once per process (registry-sized).
    #[allow(clippy::disallowed_methods)]
    fn build() -> Self {
        let cards = reg_names("card");
        let potions = reg_names("potion");
        let mut species: Vec<String> = sts2_core::MonsterKind::ALL_KNOWN
            .iter()
            .map(|k| k.python_name().to_owned())
            .collect();
        species.sort_unstable();
        species.dedup();
        let enemy_slots: Vec<String> = (0..MAX_ENEMY_SLOTS).map(|i| i.to_string()).collect();
        let hand_slots: Vec<String> = (0..MAX_HAND_SLOTS).map(|i| i.to_string()).collect();
        let tables = vec![
            Table::new(cards.clone(), Norm::Canonical),
            Table::of(&["0", "1"]),
            Table::new(reg_names("relic"), Norm::Id),
            Table::new(potions.clone(), Norm::Canonical),
            Table::new(reg_names("power"), Norm::Id),
            Table::new(species, Norm::Id),
            Table::new(reg_names("enchantment"), Norm::Enchantment),
            Table::of(&AFFLICTIONS),
            Table::of(&INTENT_KINDS),
            Table::new(enemy_slots, Norm::Id),
            Table::new(hand_slots, Norm::Id),
            Table::of(&CHOICE_KINDS),
            Table::of(&CARD_TYPES),
            Table::of(&CHOICE_PILES),
            Table::new(rl2_key_universe(&cards, &potions), Norm::Id),
        ];
        let mut pads = [0i16; SCTX_N_CAT];
        for (p, t) in pads.iter_mut().zip(&tables) {
            *p = t.pad;
        }
        let mut card_type = HashMap::new();
        let mut card_type_canon = HashMap::new();
        for line in CARD_TYPES_TSV.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (id, ty) = line.split_once('\t').expect("card_types tsv");
            let ty: &'static str = CARD_TYPES
                .iter()
                .find(|t| **t == ty)
                .copied()
                .unwrap_or_else(|| panic!("card type {ty:?}"));
            card_type.insert(id.to_owned(), ty);
            card_type_canon.insert(sts2_vocab::canonical(id), ty);
        }
        Self {
            tables,
            pads,
            card_type,
            card_type_canon,
        }
    }

    /// The table of slot `s` (`S[name]`).
    #[must_use]
    pub fn table(&self, s: usize) -> &Table {
        &self.tables[s]
    }

    /// `combat_ctx.tokens._card_type`: the v0.111.0 table by raw id, `"?"`
    /// when absent.
    #[must_use]
    pub fn card_type_raw(&self, id: &str) -> &'static str {
        self.card_type.get(id).copied().unwrap_or("?")
    }

    /// `e2e.tokens._card_type`: the raw id, else its `canonical` form.
    #[must_use]
    pub fn card_type_any(&self, id: &str) -> &'static str {
        match self.card_type.get(id) {
            Some(t) => t,
            None => self
                .card_type_canon
                .get(&sts2_vocab::canonical(id))
                .copied()
                .unwrap_or("?"),
        }
    }
}

/// The S_ctx tables, built once.
#[must_use]
pub fn vocab() -> &'static SctxVocab {
    static V: OnceLock<SctxVocab> = OnceLock::new();
    V.get_or_init(SctxVocab::build)
}

/// The content slots `luts` covers, with their e2e registries.
pub const ID_SLOTS: [(usize, Reg); 6] = [
    (slot("card"), Reg::Card),
    (slot("relic"), Reg::Relic),
    (slot("potion"), Reg::Potion),
    (slot("power"), Reg::Power),
    (slot("monster"), Reg::Monster),
    (slot("enchantment"), Reg::Enchantment),
];

/// `tokens.luts()`: per content slot, S_ctx table row -> e2e vocab id (PAD
/// -> NONE, UNK -> `UNK_<registry>`), indexed by S_ctx slot.
#[must_use]
pub fn luts() -> &'static [Vec<u16>; SCTX_N_CAT] {
    static L: OnceLock<[Vec<u16>; SCTX_N_CAT]> = OnceLock::new();
    L.get_or_init(|| {
        let tv = vocab();
        let v = vocab::load();
        let mut out: [Vec<u16>; SCTX_N_CAT] = Default::default();
        for (s, reg) in ID_SLOTS {
            let tab = tv.table(s);
            let mut lut = vec![0u16; tab.size()];
            for (i, name) in tab.names.iter().enumerate() {
                lut[i] = v.id(reg, Some(name));
            }
            lut[usize::try_from(tab.unk).unwrap_or(0)] = reg.unk();
            lut[usize::try_from(tab.pad).unwrap_or(0)] = vocab::NONE_ID;
            out[s] = lut;
        }
        out
    })
}

// ── building ─────────────────────────────────────────────────────────────

/// rl2's v1 inputs (`rollout_features(bv, 1)`): state features, per-action
/// key names and features.
#[derive(Debug)]
pub struct Rl2 {
    /// `state` (32).
    pub state: Vec<f32>,
    /// `keys[k]`: `ActionKey::name` of action `k`.
    pub keys: Vec<String>,
    /// `x[k]` (24 each).
    pub x: Vec<Vec<f32>>,
}

impl Rl2 {
    /// `sts2_rs.rollout_features(bv, version=1)`.
    #[must_use]
    pub fn of(bv: &sts2_core::CombatState) -> Self {
        let ex =
            sts2_mcts::rollout_net::export_features_spec(bv, &sts2_mcts::rollout_net::FEATURES_V1);
        let mut keys = Vec::with_capacity(ex.actions.len());
        let mut x = Vec::with_capacity(ex.actions.len());
        for (_a, k, xs) in ex.actions {
            keys.push(k.name());
            x.push(xs);
        }
        Self {
            state: ex.state,
            keys,
            x,
        }
    }
}

/// The S_ctx tokens: `cat` int16 `[T, 15]`, `cont` float32 `[T, 122]`,
/// `sec` `[T]`; observation tokens first (STATE at 0), then one token per
/// action (`n_act`).
#[derive(Clone, Debug, Default)]
pub struct SctxTokens {
    /// `[T * 15]`.
    pub cat: Vec<i16>,
    /// `[T * 122]`.
    pub cont: Vec<f32>,
    /// `[T]`.
    pub sec: Vec<u8>,
    /// Action tokens (the last `n_act`).
    pub n_act: usize,
}

impl SctxTokens {
    /// Tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sec.len()
    }

    /// No tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sec.is_empty()
    }

    /// The cat row of token `t`.
    #[must_use]
    pub fn cat_row(&self, t: usize) -> &[i16] {
        &self.cat[t * SCTX_N_CAT..(t + 1) * SCTX_N_CAT]
    }
}

struct B<'a> {
    v: &'a SctxVocab,
    out: SctxTokens,
}

// Continuous column indices (`combat_ctx.tokens.C_*`).
const C_COST: usize = schema::sc("COST");
const C_BASE_COST: usize = schema::sc("BASE_COST");
const C_DMG: usize = schema::sc("DMG");
const C_BLK: usize = schema::sc("BLK");
const C_CALC: usize = schema::sc("CALC");
const C_HAS_CALC: usize = schema::sc("HAS_CALC");
const C_ETHEREAL: usize = schema::sc("ETHEREAL");
const C_UNPLAYABLE: usize = schema::sc("UNPLAYABLE");
const C_INNATE: usize = schema::sc("INNATE");
const C_SLY: usize = schema::sc("SLY");
const C_SLY_TURN: usize = schema::sc("SLY_TURN");
const C_RETAIN_TURN: usize = schema::sc("RETAIN_TURN");
const C_ENCH_AMT: usize = schema::sc("ENCH_AMT");
const C_AFF_STACKS: usize = schema::sc("AFF_STACKS");
/// The pile-multiplicity / pin-rank column.
pub const C_COUNT: usize = schema::sc("COUNT");
const C_PLAYABLE: usize = schema::sc("PLAYABLE");
const C_DMG_PREV: usize = schema::sc("DMG_PREV");
const C_BLK_PREV: usize = schema::sc("BLK_PREV");

impl B<'_> {
    fn new_tok(&mut self, section: &str) -> usize {
        let t = self.out.sec.len();
        self.out.cat.extend_from_slice(&self.v.pads);
        self.out
            .cont
            .extend(std::iter::repeat_n(0.0f32, SCTX_N_CONT));
        self.out.sec.push(schema::sec(section));
        t
    }

    fn put(&mut self, t: usize, s: usize, name: &str) {
        let row = self.v.tables[s].index(name);
        self.out.cat[t * SCTX_N_CAT + s] = row;
    }

    fn set(&mut self, t: usize, c: usize, x: f64) {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.out.cont[t * SCTX_N_CONT + c] = x as f32;
        }
    }

    fn card_feats(&mut self, t: usize, c: &Value, count: f64) {
        let id = get(c, "id").map_or_else(|| "None".to_owned(), py_str);
        self.put(t, slot("card"), &id);
        let up = get(c, "upgraded").is_some_and(truthy);
        self.put(t, slot("upgraded"), if up { "1" } else { "0" });
        let ct = match get(c, "id") {
            Some(Value::String(s)) => self.v.card_type_raw(s),
            _ => "?",
        };
        self.put(t, slot("card_type"), ct);
        if let Some(e) = get(c, "enchantment").filter(|e| truthy(e)) {
            self.put(t, slot("enchantment"), &py_str(e));
        }
        if let Some(a) = get(c, "affliction").filter(|a| truthy(a)) {
            self.put(t, slot("affliction"), &py_str(a));
        }
        let flags: Vec<&str> = list_or_empty(c, "flags")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        let has = |f: &str| f64::from(u8::from(flags.contains(&f)));
        self.set(t, C_COST, fz(get(c, "cost")) / 3.0);
        self.set(t, C_BASE_COST, fz(get(c, "base_cost")) / 3.0);
        self.set(t, C_DMG, fz(get(c, "base_damage")) / 20.0);
        self.set(t, C_BLK, fz(get(c, "base_block")) / 20.0);
        if let Some(cv) = get(c, "calculated_value") {
            self.set(t, C_CALC, fz(Some(cv)) / 20.0);
            self.set(t, C_HAS_CALC, 1.0);
        }
        self.set(t, C_ETHEREAL, has("ethereal"));
        self.set(t, C_UNPLAYABLE, has("unplayable"));
        self.set(t, C_INNATE, has("innate"));
        self.set(t, C_SLY, has("sly"));
        self.set(t, C_SLY_TURN, has("sly_turn"));
        self.set(t, C_RETAIN_TURN, has("retain_turn"));
        self.set(t, C_ENCH_AMT, fz(get(c, "enchantment_amount")) / 5.0);
        self.set(t, C_AFF_STACKS, fz(get(c, "affliction_stacks")) / 5.0);
        self.set(t, C_COUNT, count / 5.0);
    }

    fn power_feats(&mut self, t: usize, p: &Value) {
        let name = get(p, "name").map_or_else(|| "None".to_owned(), py_str);
        self.put(t, slot("power"), &name);
        self.set(t, schema::sc("P_AMT"), fz(get(p, "amount")) / 10.0);
        self.set(t, schema::sc("P_DISP"), fz(get(p, "display_amount")) / 10.0);
        self.set(t, schema::sc("P_CTR"), fz(get(p, "counter")) / 10.0);
    }

    fn hand_card(&mut self, t: usize, hand: &[Value], i: usize, extras: &[(f64, f64, f64)]) {
        let Some(c) = hand.get(i) else { return };
        self.card_feats(t, c, 0.0);
        self.put(t, slot("hand_slot"), &i.to_string());
        let (playable, dmg, blk) = extras[i];
        self.set(t, C_PLAYABLE, playable);
        self.set(t, C_DMG_PREV, dmg);
        self.set(t, C_BLK_PREV, blk);
    }
}

/// `_intent_total` (combat_ctx): an Attack / DeathBlow intent's
/// `effective_amount (else amount) * times`.
fn intent_total(i: &Value) -> f64 {
    let kind = get(i, "kind").and_then(Value::as_str);
    if !matches!(kind, Some("Attack" | "DeathBlow")) {
        return 0.0;
    }
    let eff = match get(i, "effective_amount") {
        Some(v) => v.clone(),
        None => get(i, "amount").cloned().unwrap_or(Value::from(0)),
    };
    let eff = if truthy(&eff) { fz(Some(&eff)) } else { 0.0 };
    let times = get(i, "times").filter(|v| truthy(v));
    eff * times.map_or(1.0, |v| fz(Some(v)))
}

fn hand_extras(hand: &[Value], actions: &[Value]) -> Vec<(f64, f64, f64)> {
    let playable: Vec<i64> = actions
        .iter()
        .filter(|a| get(a, "action").and_then(Value::as_str) == Some("play"))
        .filter_map(|a| get(a, "card_index").and_then(py_int))
        .collect();
    hand.iter()
        .enumerate()
        .map(|(i, c)| {
            let mx = get(c, "dmg")
                .and_then(Value::as_object)
                .and_then(|m| m.values().filter_map(py_int).max())
                .unwrap_or(0);
            #[allow(clippy::cast_precision_loss)]
            let dmg = mx as f64 / 30.0;
            let pl = i64::try_from(i).is_ok_and(|ii| playable.contains(&ii));
            (f64::from(u8::from(pl)), dmg, fz(get(c, "blk")) / 20.0)
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn state_token(b: &mut B<'_>, obs: &Value, rl2s: &[f32]) {
    let t = b.new_tok("STATE");
    let p = get(obs, "player").cloned().unwrap_or(Value::Null);
    let hp = fz(get(&p, "hp"));
    let mhp = fz(get(&p, "max_hp")).max(1.0);
    let sc = schema::sc;
    b.set(t, sc("HP_FRAC"), hp / mhp);
    b.set(t, sc("HP"), hp / 100.0);
    b.set(t, sc("MAXHP"), mhp / 100.0);
    b.set(t, sc("BLOCK"), fz(get(&p, "block")) / 30.0);
    b.set(t, sc("ENERGY"), fz(get(&p, "energy")) / 5.0);
    b.set(t, sc("MAXE"), fz(get(&p, "max_energy")) / 5.0);
    b.set(t, sc("STARS"), fz(get(&p, "stars")) / 5.0);
    b.set(t, sc("ROUND"), fz(get(obs, "round")) / 20.0);
    let played = pysum(
        list_or_empty(&p, "cards_played_this_turn")
            .iter()
            .map(|x| fz(Some(x))),
    );
    b.set(t, sc("PLAYED"), played / 10.0);
    let pair_n = |k: &str| -> f64 {
        pysum(
            list_or_empty(obs, k)
                .iter()
                .map(|cn| fz(cn.as_array().and_then(|a| a.get(1)))),
        )
    };
    let n_draw = pair_n("draw_unknown")
        + list_or_empty(obs, "draw_top").len() as f64
        + list_or_empty(obs, "draw_bottom").len() as f64;
    b.set(
        t,
        sc("N_HAND"),
        list_or_empty(obs, "hand").len() as f64 / 30.0,
    );
    b.set(t, sc("N_DRAW"), n_draw / 30.0);
    b.set(t, sc("N_DISC"), pair_n("discard") / 30.0);
    b.set(t, sc("N_EXH"), pair_n("exhaust") / 30.0);
    b.set(t, sc("POT_SLOTS"), fz(get(&p, "potion_slots")) / 5.0);
    let room = get(obs, "room").and_then(Value::as_str).unwrap_or("");
    b.set(t, sc("IS_BOSS"), f64::from(u8::from(room == "boss")));
    b.set(t, sc("IS_ELITE"), f64::from(u8::from(room == "elite")));
    b.set(t, sc("IS_HALL"), f64::from(u8::from(room == "monster")));
    if let Some(ch) = get(obs, "choice").filter(|c| !c.is_null()) {
        let k = get(ch, "kind").map_or_else(|| "None".to_owned(), py_str);
        b.put(t, slot("choice_kind"), &k);
        b.set(t, sc("CH_MIN"), fz(get(ch, "min_select")) / 5.0);
        b.set(t, sc("CH_MAX"), fz(get(ch, "max_select")) / 5.0);
        b.set(
            t,
            sc("CH_SKIP"),
            f64::from(u8::from(get(ch, "can_skip").is_some_and(truthy))),
        );
    }
    let mut inc = 0.0;
    for m in list_or_empty(obs, "enemies") {
        if !get(m, "is_dead").is_some_and(truthy) {
            inc += pysum(list_or_empty(m, "intents").iter().map(intent_total));
        }
    }
    b.set(t, sc("INCOMING"), inc / 50.0);
    b.set(
        t,
        sc("UNBLOCKED"),
        (inc - fz(get(&p, "block"))).max(0.0) / 50.0,
    );
    let base = t * SCTX_N_CONT + C_RL2S;
    b.out.cont[base..base + RL2_STATE_DIM].copy_from_slice(&rl2s[..RL2_STATE_DIM]);
}

#[allow(clippy::cast_precision_loss)]
fn obs_tokens(b: &mut B<'_>, obs: &Value, actions: &[Value]) {
    let hand = list_or_empty(obs, "hand");
    let extras = hand_extras(hand, actions);
    for i in 0..hand.len() {
        let t = b.new_tok("HAND");
        b.hand_card(t, hand, i, &extras);
    }
    for (sec, pile) in [
        ("DRAW_UNK", "draw_unknown"),
        ("DISCARD", "discard"),
        ("EXHAUST", "exhaust"),
    ] {
        for cn in list_or_empty(obs, pile) {
            let a = cn.as_array().map_or(&[][..], Vec::as_slice);
            let t = b.new_tok(sec);
            let c = a.first().cloned().unwrap_or(Value::Null);
            b.card_feats(t, &c, fz(a.get(1)));
        }
    }
    for (sec, pile) in [("DRAW_TOP", "draw_top"), ("DRAW_BOT", "draw_bottom")] {
        for (r, c) in list_or_empty(obs, pile).iter().enumerate() {
            let t = b.new_tok(sec);
            b.card_feats(t, c, (r + 1) as f64);
        }
    }
    for c in list_or_empty(obs, "play") {
        let t = b.new_tok("PLAY");
        b.card_feats(t, c, 0.0);
    }
    for r in list_or_empty(obs, "relics") {
        let t = b.new_tok("RELIC");
        let id = get(r, "id").map_or_else(|| "None".to_owned(), py_str);
        b.put(t, slot("relic"), &id);
        if let Some(ctr) = get(r, "counter") {
            b.set(t, schema::sc("R_CTR"), fz(Some(ctr)) / 10.0);
            b.set(t, schema::sc("R_HAS"), 1.0);
        }
        b.set(
            t,
            schema::sc("R_WAX"),
            f64::from(u8::from(get(r, "is_wax").is_some_and(truthy))),
        );
        b.set(
            t,
            schema::sc("R_MELT"),
            f64::from(u8::from(get(r, "is_melted").is_some_and(truthy))),
        );
    }
    for pid in list_or_empty(obs, "potions") {
        let t = b.new_tok("POTION");
        b.put(t, slot("potion"), &py_str(pid));
    }
    for p in list_or_empty(obs, "powers") {
        let t = b.new_tok("PPOWER");
        b.power_feats(t, p);
    }
    let enemies = list_or_empty(obs, "enemies");
    let ehist = list_or_empty(obs, "ehist");
    for (j, m) in enemies.iter().enumerate() {
        let alive = f64::from(u8::from(!get(m, "is_dead").is_some_and(truthy)));
        let kind = get(m, "kind").map_or_else(|| "None".to_owned(), py_str);
        let js = j.to_string();
        let t = b.new_tok("ENEMY");
        b.put(t, slot("monster"), &kind);
        b.put(t, slot("enemy_slot"), &js);
        let hi = intent_log::head_intent(m);
        let hk = get(&hi, "kind").map_or_else(|| "None".to_owned(), py_str);
        b.put(t, slot("intent"), &hk);
        let mhp = fz(get(m, "max_hp")).max(1.0);
        let sc = schema::sc;
        b.set(t, sc("E_HPF"), fz(get(m, "hp")) / mhp);
        b.set(t, sc("E_HP"), fz(get(m, "hp")) / 200.0);
        b.set(t, sc("E_MAXHP"), mhp / 200.0);
        b.set(t, sc("E_BLOCK"), fz(get(m, "block")) / 30.0);
        b.set(t, sc("E_ALIVE"), alive);
        b.set(
            t,
            sc("E_REMOVED"),
            f64::from(u8::from(get(m, "removed_after_death").is_some_and(truthy))),
        );
        b.set(t, sc("E_AMT"), fz(get(&hi, "amount")) / 30.0);
        b.set(t, sc("E_EFF"), fz(get(&hi, "effective_amount")) / 30.0);
        b.set(t, sc("E_TIMES"), fz(get(&hi, "times")) / 5.0);
        b.set(t, sc("E_TOT"), intent_total(&hi) / 50.0);
        for p in list_or_empty(m, "powers") {
            let t = b.new_tok("EPOWER");
            b.power_feats(t, p);
            b.put(t, slot("monster"), &kind);
            b.put(t, slot("enemy_slot"), &js);
        }
        for s in list_or_empty(m, "stolen") {
            let t = b.new_tok("EPOWER");
            let c = serde_json::json!({
                "id": get(s, "id").cloned().unwrap_or(Value::Null),
                "upgraded": get(s, "upgraded").cloned().unwrap_or(Value::Null),
            });
            b.card_feats(t, &c, 0.0);
            b.put(t, slot("monster"), &kind);
            b.put(t, slot("enemy_slot"), &js);
        }
        if let Some(Value::Array(hs)) = ehist.get(j) {
            for h in hs {
                let t = b.new_tok("EHIST");
                let k = get(h, "kind").map_or_else(|| "None".to_owned(), py_str);
                b.put(t, slot("intent"), &k);
                b.put(t, slot("monster"), &kind);
                b.put(t, slot("enemy_slot"), &js);
                b.set(t, sc("H_LAG"), fz(get(h, "lag")) / 3.0);
                b.set(t, sc("H_AMT"), fz(get(h, "amount")) / 30.0);
                b.set(t, sc("H_EFF"), fz(get(h, "effective_amount")) / 30.0);
                b.set(t, sc("H_TIMES"), fz(get(h, "times")) / 5.0);
                b.set(t, sc("H_ALIVE"), alive);
            }
        }
    }
}

/// `_target`: the target enemy's slot, monster and HP columns; the enemy
/// when it exists.
fn target<'v>(b: &mut B<'_>, t: usize, obs: &'v Value, ti: Option<&Value>) -> Option<&'v Value> {
    let ti = ti.filter(|v| !v.is_null()).and_then(py_int)?;
    b.put(t, slot("enemy_slot"), &ti.to_string());
    let m = usize::try_from(ti)
        .ok()
        .and_then(|i| list_or_empty(obs, "enemies").get(i))?;
    let kind = get(m, "kind").map_or_else(|| "None".to_owned(), py_str);
    b.put(t, slot("monster"), &kind);
    let mhp = fz(get(m, "max_hp")).max(1.0);
    b.set(t, schema::sc("T_HPF"), fz(get(m, "hp")) / mhp);
    b.set(
        t,
        schema::sc("T_EHP"),
        (fz(get(m, "hp")) + fz(get(m, "block"))) / 100.0,
    );
    Some(m)
}

#[allow(clippy::cast_precision_loss)]
fn action_tokens(b: &mut B<'_>, obs: &Value, actions: &[Value], rl2: &Rl2) {
    let hand = list_or_empty(obs, "hand");
    let extras = hand_extras(hand, actions);
    let ch = get(obs, "choice").filter(|c| !c.is_null());
    for (k, a) in actions.iter().enumerate() {
        let kind = get(a, "action").and_then(Value::as_str).unwrap_or("");
        let sel = list_or_empty(a, "selection");
        let t;
        if kind == "play" {
            t = b.new_tok("ACT_PLAY");
            let i = get(a, "card_index").and_then(py_int).unwrap_or(0);
            let iu = usize::try_from(i).unwrap_or(usize::MAX);
            b.hand_card(t, hand, iu, &extras);
            if target(b, t, obs, get(a, "target_index")).is_some() {
                let key = get(a, "target_index")
                    .and_then(py_int)
                    .unwrap_or(0)
                    .to_string();
                let dmg = hand
                    .get(iu)
                    .and_then(|c| get(c, "dmg"))
                    .and_then(|d| get(d, &key));
                b.set(t, schema::sc("T_DMG"), fz(dmg) / 30.0);
            }
        } else if kind == "end_turn" {
            t = b.new_tok("ACT_END");
        } else if kind == "potion" || kind == "drink_potion" {
            t = b.new_tok("ACT_POTION");
            let s = get(a, "slot").and_then(py_int).unwrap_or(0);
            let pots = list_or_empty(obs, "potions");
            if let Some(p) = usize::try_from(s).ok().and_then(|s| pots.get(s)) {
                b.put(t, slot("potion"), &py_str(p));
            }
            if let Some(ti) = get(a, "target_index").filter(|v| !v.is_null()) {
                b.put(t, slot("enemy_slot"), &py_int(ti).unwrap_or(0).to_string());
            }
        } else if kind == "choose" && !sel.is_empty() {
            t = b.new_tok("ACT_CHOOSE");
            let opts = ch.map_or(&[][..], |c| list_or_empty(c, "options"));
            let s0 = sel.first().and_then(py_int).unwrap_or(0);
            if let Some(o) = usize::try_from(s0).ok().and_then(|s| opts.get(s)) {
                let id = get(o, "id").map_or_else(|| "None".to_owned(), py_str);
                b.put(t, slot("card"), &id);
                let up = get(o, "upgraded").is_some_and(truthy);
                b.put(t, slot("upgraded"), if up { "1" } else { "0" });
                let ct = match get(o, "id") {
                    Some(Value::String(s)) => b.v.card_type_raw(s),
                    _ => "?",
                };
                b.put(t, slot("card_type"), ct);
                let pile = get(o, "pile").map_or_else(|| "generated".to_owned(), py_str);
                b.put(t, slot("choice_pile"), &pile);
                if get(o, "pile").and_then(Value::as_str) == Some("hand") {
                    if let Some(hi) = get(o, "hand_index") {
                        b.put(t, slot("hand_slot"), &py_str(hi));
                    }
                }
            }
            if let Some(c) = ch {
                let k = get(c, "kind").map_or_else(|| "None".to_owned(), py_str);
                b.put(t, slot("choice_kind"), &k);
            }
            b.set(t, schema::sc("N_SEL"), sel.len() as f64 / 3.0);
        } else if kind == "choose" {
            t = b.new_tok("ACT_SKIP");
        } else {
            t = b.new_tok("ACT_OTHER");
        }
        b.put(t, slot("rl2key"), &rl2.keys[k]);
        let base = t * SCTX_N_CONT + C_RL2X;
        b.out.cont[base..base + RL2_ACTION_DIM].copy_from_slice(&rl2.x[k][..RL2_ACTION_DIM]);
    }
}

/// `tokens(obs, actions, rl2feat, "S_ctx")`.
///
/// # Errors
///
/// `rl2` is not aligned with `actions`.
pub fn tokens(obs: &Value, actions: &[Value], rl2: &Rl2) -> Result<SctxTokens, String> {
    if rl2.keys.len() != actions.len() || rl2.x.len() != actions.len() {
        return Err(format!(
            "rl2feat is not aligned with actions ({} keys, {} actions)",
            rl2.keys.len(),
            actions.len()
        ));
    }
    let mut b = B {
        v: vocab(),
        out: SctxTokens::default(),
    };
    state_token(&mut b, obs, &rl2.state);
    obs_tokens(&mut b, obs, actions);
    action_tokens(&mut b, obs, actions, rl2);
    b.out.n_act = actions.len();
    Ok(b.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_sizes_match_contracts() {
        // contracts.SCTX_TABLE_SIZES (pinned against the W1 wheel).
        let want = [601, 4, 299, 66, 272, 123, 25, 11, 16, 7, 14, 33, 8, 8, 1267];
        let got: Vec<usize> = vocab().tables.iter().map(Table::size).collect();
        assert_eq!(got, want);
    }
}
