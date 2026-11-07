//! The e2e tokenizer's policy part: `ppo.e2e.tokens.encode(d, rare=None,
//! priv=False)` for one env [`Decision`], bit-equal to Python's row
//! (`tests/tok_parity.rs`).
//!
//! A row is token 0 `RUNCTX`, the five pooled summaries (`SUM_*`), one or two
//! `BOSS` tokens, then
//!
//! * combat: #1081's S_ctx block ([`crate::sctx`]) with the trailing columns
//!   (content ids, `X_KIND`, `X_POS`, tactical, telegraph and forecast,
//!   `K_PICKED`); every legal action keeps its token, a merged class's
//!   members after the first carry `TF_HIDDEN`;
//! * out of combat: the run body (`DECK`, `RELIC`, `POTION`, `OFFER`,
//!   `MAPNODE` on map decisions) and one `OPT_*` token per merged class.
//!
//! The run is read as Python's `_Run` reads it: `honest_obs(run_obs)` first
//! (out of combat), else the run manager. The map and the bosses are read
//! once per `(act, total_floor)` and cached for the run ([`TokCache`]):
//! Python's `_STATIC` cache is semantic (a later decision on the same floor
//! sees the map as the floor's first decision read it), so a [`TokCache`]
//! must live exactly one run (`TokCache::new` or [`TokCache::begin_run`] at
//! run start, as `TokenizerRows.begin_run` clears Python's).
//!
//! Numerics follow numpy: values computed in f64, stored in a float32 buffer,
//! cast to float16 (round to nearest even); the 122 S_ctx columns of combat
//! tokens saturate at ±65504 first (`T.f16`), every other column casts
//! plainly. Pool weights cast f64 -> f16 directly.

use std::collections::BTreeMap;

use half::f16;
use serde_json::Value;
use sts2_core::run_manager::RunManager;
use sts2_core::CombatState;
use sts2_wire::emit::run_manager as emit_rm;

use crate::contracts::{ak, dt, kind_name, Selection};
use crate::env::Decision;
use crate::pyjson::{get, list_or_empty, truthy};
use crate::schema::{
    self, sc, slot, xc, xf, MAX_BOSS_TOKENS, MAX_MAPNODES, N_CAT, N_CONT, SCTX_N_CAT, SCTX_N_CONT,
    TF_HIDDEN, TF_OPTION, TF_POOLED, TF_REP, TF_TEACHER,
};
use crate::sctx::{self, py_int, py_str, pysum, Rl2};
use crate::tactical;
use crate::vocab::{self, Reg};

/// `T.F16_MAX`: the S_ctx columns saturate here.
pub const F16_MAX: f32 = 65504.0;

/// One decision's policy row: the arrays `contracts.ROW_FIELDS` names for
/// the policy (no PRIV), plus the row scalars.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokRow {
    /// `[T, 32]` int16, row-major.
    pub cat: Vec<i16>,
    /// `[T, 210]` float16 bit patterns, row-major.
    pub cont: Vec<u16>,
    /// `[T]` section ids.
    pub sec: Vec<u8>,
    /// `[T]` token flags.
    pub tok_flags: Vec<u8>,
    /// `[M]` token of each member.
    pub mem_tok: Vec<u16>,
    /// `[M]` merged class of each member.
    pub mem_class: Vec<u16>,
    /// `[C]` token of each class's representative.
    pub cls_tok: Vec<u16>,
    /// `[C]` action kind of each class.
    pub cls_kind: Vec<u8>,
    /// `[P]` pooled token of each pool entry.
    pub pool_tok: Vec<u16>,
    /// `[P]` vocabulary id of each pool entry.
    pub pool_vid: Vec<u16>,
    /// `[P]` pool weights, float16 bits.
    pub pool_w: Vec<u16>,
    /// First token of the S_ctx block (combat), else 0.
    pub sctx_lo: usize,
    /// Tokens of the S_ctx block (combat), else 0.
    pub n_sctx: usize,
    /// Decision type.
    pub dtype: u8,
    /// Combat row.
    pub is_combat: bool,
}

impl TokRow {
    /// Tokens.
    #[must_use]
    pub fn n_tok(&self) -> usize {
        self.sec.len()
    }
}

impl From<TokRow> for crate::net::PolicyRow {
    /// The arrays the policy forward reads.
    fn from(r: TokRow) -> Self {
        Self {
            cat: r.cat,
            cont: r.cont,
            sec: r.sec,
            tok_flags: r.tok_flags,
            cls_tok: r.cls_tok,
            cls_kind: r.cls_kind,
            pool_tok: r.pool_tok,
            pool_vid: r.pool_vid,
            pool_w: r.pool_w,
            is_combat: r.is_combat,
        }
    }
}

#[derive(Clone, Debug)]
struct Static {
    key: (u8, i32),
    fm: Value,
    bosses: Vec<String>,
}

/// The tokenizer's per-run state: the run-static reads (map, bosses) of the
/// current `(act, total_floor)`. One per run.
#[derive(Clone, Debug, Default)]
pub struct TokCache {
    /// Shared, so a host can snapshot a cache per position for the price of
    /// a reference count (the workbench keeps one per node).
    stat: Option<std::sync::Arc<Static>>,
}

impl TokCache {
    /// An empty cache (a run start).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the run-static reads (`tokens.clear_cache()` at a run start).
    pub fn begin_run(&mut self) {
        self.stat = None;
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

/// `tokens._f`: `float(x or 0.0)`, 0.0 for anything `float` refuses.
fn ff(v: Option<&Value>) -> f64 {
    match v {
        None => 0.0,
        Some(x) if !truthy(x) => 0.0,
        Some(Value::Bool(b)) => f64::from(u8::from(*b)),
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(0.0),
        Some(_) => 0.0,
    }
}

/// `_at(xs, i)`: `xs[int(i)]` when in range (a JSON null element is `None`).
fn at<'a>(xs: Option<&'a Value>, i: Option<&Value>) -> Option<&'a Value> {
    let xs = xs?.as_array()?;
    let i = usize::try_from(py_int(i?)?).ok()?;
    xs.get(i).filter(|v| !v.is_null())
}

fn pos(i: i64) -> i16 {
    i16::try_from((i + 1).min(63)).unwrap_or(63)
}

fn vid(reg: Reg, name: Option<&Value>) -> i16 {
    let id = match name {
        None | Some(Value::Null) => vocab::NONE_ID,
        Some(v) => vocab::load().id(reg, Some(&py_str(v))),
    };
    i16::try_from(id).unwrap_or(i16::MAX)
}

fn vid_s(reg: Reg, name: Option<&str>) -> i16 {
    i16::try_from(vocab::load().id(reg, name)).unwrap_or(i16::MAX)
}

/// `_ench_name(e)`.
fn ench_name(e: Option<&Value>) -> Option<String> {
    let e = e.filter(|x| truthy(x))?;
    match e {
        Value::Array(a) => a.first().map(py_str),
        Value::Object(_) => {
            let id = get(e, "id").filter(|x| truthy(x));
            let v = id.or_else(|| get(e, "name"))?;
            (!v.is_null()).then(|| py_str(v))
        }
        other => Some(py_str(other)),
    }
}

/// `_ench_amount(e)`.
fn ench_amount(e: Option<&Value>) -> f64 {
    match e {
        Some(Value::Array(a)) if a.len() > 1 => ff(a.get(1)),
        Some(o @ Value::Object(_)) => ff(get(o, "amount")),
        _ => 0.0,
    }
}

fn to_f16(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

// ── token buffer ─────────────────────────────────────────────────────────

fn absent_cat() -> [i16; N_CAT] {
    let mut row = [0i16; N_CAT];
    row[..SCTX_N_CAT].copy_from_slice(&sctx::vocab().pads);
    row
}

struct Buf {
    absent: [i16; N_CAT],
    cat: Vec<i16>,
    cont: Vec<f32>,
    sec: Vec<u8>,
    flags: Vec<u8>,
    pool_tok: Vec<u16>,
    pool_vid: Vec<u16>,
    pool_w: Vec<f64>,
}

impl Buf {
    fn new(cap: usize) -> Self {
        Self {
            absent: absent_cat(),
            cat: Vec::with_capacity(cap * N_CAT),
            cont: Vec::with_capacity(cap * N_CONT),
            sec: Vec::with_capacity(cap),
            flags: Vec::with_capacity(cap),
            pool_tok: Vec::new(),
            pool_vid: Vec::new(),
            pool_w: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.sec.len()
    }

    fn new_tok(&mut self, section: u8, flags: u8) -> usize {
        let t = self.sec.len();
        self.cat.extend_from_slice(&self.absent);
        self.cont.extend(std::iter::repeat_n(0.0f32, N_CONT));
        self.sec.push(section);
        self.flags.push(flags);
        t
    }

    fn c(&mut self, t: usize, col: usize, v: i16) {
        self.cat[t * N_CAT + col] = v;
    }

    fn f(&mut self, t: usize, col: usize, x: f64) {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.cont[t * N_CONT + col] = x as f32;
        }
    }
}

// ── run view ─────────────────────────────────────────────────────────────

/// `_Run`: `honest_obs(run_obs)` first, else the run manager.
struct Run<'a> {
    rm: &'a RunManager,
    obs: Option<&'a Value>,
    deck: Vec<(String, bool)>,
    ench: Vec<Value>,
    relics: Vec<String>,
    potions: Vec<String>,
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| a.iter().map(py_str).collect())
        .unwrap_or_default()
}

impl<'a> Run<'a> {
    fn new(rm: &'a RunManager, obs: Option<&'a Value>) -> Self {
        let obs = obs.filter(|o| truthy(o));
        let og = |k: &str| obs.and_then(|o| get(o, k));
        let deck = og("deck").map_or_else(
            || emit_rm::master_deck_detailed(rm),
            |d| {
                d.as_array()
                    .map(|a| {
                        a.iter()
                            .map(|e| {
                                let e = e.as_array().map_or(&[][..], Vec::as_slice);
                                (
                                    e.first().map(py_str).unwrap_or_default(),
                                    e.get(1).is_some_and(truthy),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            },
        );
        let ench = og("deck_enchantments").map_or_else(
            || {
                emit_rm::master_deck_enchantments(rm)
                    .into_iter()
                    .map(|e| serde_json::json!(e))
                    .collect()
            },
            |v| v.as_array().cloned().unwrap_or_default(),
        );
        let relics = og("relics").map_or_else(|| emit_rm::relics(rm), strings);
        let potions = og("potions").map_or_else(|| emit_rm::potions(rm), strings);
        Self {
            rm,
            obs,
            deck,
            ench,
            relics,
            potions,
        }
    }

    fn og(&self, k: &str) -> Option<&'a Value> {
        self.obs.and_then(|o| get(o, k))
    }

    fn act(&self) -> u8 {
        self.rm.run_state.act_index
    }

    fn floor(&self) -> i32 {
        self.rm.run_state.total_floor
    }

    fn full_map(&self) -> Value {
        self.og("map")
            .cloned()
            .unwrap_or_else(|| emit_rm::full_map(self.rm))
    }

    fn act_floor(&self) -> i64 {
        self.og("act_floor")
            .and_then(py_int)
            .unwrap_or_else(|| i64::from(self.rm.run_state.act_floor))
    }

    fn ascension(&self) -> i64 {
        self.og("ascension")
            .and_then(py_int)
            .unwrap_or_else(|| i64::from(self.rm.run_state.ascension_level))
    }
}

// ── head: RUNCTX, SUMMARY, BOSS ──────────────────────────────────────────

fn static_of<'c>(cache: &'c mut TokCache, run: &Run<'_>) -> &'c std::sync::Arc<Static> {
    let key = (run.act(), run.floor());
    if cache.stat.as_ref().is_none_or(|s| s.key != key) {
        let fm = run.full_map();
        let mut bosses = Vec::new();
        for k in ["boss", "second_boss"] {
            if let Some(b) = get(&fm, k).filter(|b| truthy(b)) {
                if let Some(id) = get(b, "id").filter(|i| truthy(i)) {
                    bosses.push(py_str(id));
                }
            }
        }
        cache.stat = Some(std::sync::Arc::new(Static { key, fm, bosses }));
    }
    cache.stat.as_ref().expect("static entry")
}

fn cur_row(fm: &Value) -> i64 {
    get(fm, "current_coord")
        .filter(|c| truthy(c))
        .and_then(|c| get(c, "row"))
        .and_then(py_int)
        .unwrap_or(-1)
}

fn boss_row(fm: &Value) -> i64 {
    let n_rows = i64::try_from(list_or_empty(fm, "rows").len()).unwrap_or(0);
    get(fm, "boss")
        .filter(|b| truthy(b))
        .and_then(|b| get(b, "row"))
        .and_then(py_int)
        .unwrap_or(n_rows)
}

/// `bosses_beaten(fm, in_combat)`.
fn bosses_beaten(fm: &Value, in_combat: bool) -> i64 {
    let cur = cur_row(fm);
    let mut n = 0;
    for k in ["boss", "second_boss"] {
        let Some(b) = get(fm, k).filter(|b| truthy(b)) else {
            continue;
        };
        if !get(b, "id").is_some_and(truthy) {
            continue;
        }
        let Some(r) = get(b, "row").filter(|r| !r.is_null()).and_then(py_int) else {
            continue;
        };
        if r < cur || (r == cur && !in_combat) {
            n += 1;
        }
    }
    n
}

/// `bosses_left(act, n_this_act, ascension)`.
fn bosses_left(act: i64, n_this_act: i64, asc: i64) -> i64 {
    let later: i64 = (act + 1..4)
        .map(|a| if a == 3 && asc >= 10 { 2 } else { 1 })
        .sum();
    n_this_act + later
}

fn event_name(d: &Decision, run: &Run<'_>) -> Option<String> {
    if let Some(e) = d.event_name.as_ref().filter(|e| !e.is_empty()) {
        return Some(e.clone());
    }
    if let Some(e) = run.og("event_name").filter(|e| truthy(e)) {
        return Some(py_str(e));
    }
    crate::env::current_event_name(run.rm)
}

#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn runctx(
    buf: &mut Buf,
    d: &Decision,
    run: &Run<'_>,
    fm: &Value,
    n_bosses: usize,
    sel: &Selection,
    room: Option<&str>,
    encounter: Option<&str>,
) {
    let t = buf.new_tok(schema::sec("RUNCTX"), 0);
    buf.c(t, xc("X_DTYPE"), i16::from(d.dtype));
    if d.dtype == dt::EVENT || d.dtype == dt::NEOW {
        let ev = event_name(d, run);
        buf.c(t, xc("X_ID_MAIN"), vid_s(Reg::Event, ev.as_deref()));
    }
    if d.is_combat() {
        let id = encounter
            .filter(|e| !e.is_empty())
            .map_or(0, |e| vid_s(Reg::Encounter, Some(e)));
        buf.c(t, xc("X_ID_AUX"), id);
    }
    let room_s: Option<String> = room.map_or_else(
        || {
            let cr = emit_rm::current_room(run.rm);
            get(&cr, "room_type").filter(|r| !r.is_null()).map(py_str)
        },
        |r| Some(r.to_owned()),
    );
    let rid = match room_s.as_deref() {
        Some(r) if !r.is_empty() => schema::room_id(&r.to_lowercase()),
        _ => schema::room_id("none"),
    };
    buf.c(
        t,
        xc("X_ROOM"),
        rid.unwrap_or_else(|| schema::room_id("unknown").unwrap_or(2)),
    );
    let act = i64::from(run.act());
    let rs = &run.rm.run_state;
    let hp = f64::from(rs.current_hp);
    let mhp = f64::from(rs.max_hp).max(1.0);
    let asc = run.ascension();
    buf.f(t, xf("R_ACT"), act as f64 / 3.0);
    buf.f(t, xf("R_FLOOR"), f64::from(run.floor()) / 50.0);
    buf.f(t, xf("R_ACT_FLOOR"), run.act_floor() as f64 / 17.0);
    buf.f(
        t,
        xf("R_ROWS_TO_BOSS"),
        (boss_row(fm) - cur_row(fm)).max(0) as f64 / 17.0,
    );
    let nb = i64::try_from(n_bosses).unwrap_or(0);
    let left = nb - bosses_beaten(fm, d.is_combat());
    buf.f(
        t,
        xf("R_BOSSES_LEFT"),
        bosses_left(act, left, asc) as f64 / 4.0,
    );
    buf.f(t, xf("R_HP"), hp / 100.0);
    buf.f(t, xf("R_MAXHP"), mhp / 100.0);
    buf.f(t, xf("R_HP_FRAC"), hp / mhp);
    buf.f(t, xf("R_GOLD"), f64::from(emit_rm::gold(run.rm)) / 300.0);
    buf.f(t, xf("R_DECK"), run.deck.len() as f64 / 40.0);
    buf.f(
        t,
        xf("R_DECK_UPG"),
        run.deck.iter().filter(|(_, u)| *u).count() as f64 / 20.0,
    );
    buf.f(t, xf("R_RELICS"), run.relics.len() as f64 / 20.0);
    buf.f(
        t,
        xf("R_POTIONS"),
        run.potions.iter().filter(|p| !p.is_empty()).count() as f64 / 5.0,
    );
    buf.f(t, xf("R_POT_SLOTS"), f64::from(rs.max_potion_slots) / 5.0);
    buf.f(t, xf("R_ASC"), asc as f64 / 10.0);
    buf.f(t, xf("R_IN_COMBAT"), f64::from(u8::from(d.is_combat())));
    buf.f(t, xf("R_SEL_MIN"), sel.min as f64 / 5.0);
    buf.f(t, xf("R_SEL_MAX"), sel.max as f64 / 5.0);
    buf.f(t, xf("R_SEL_PICKED"), sel.picked as f64 / 5.0);
    buf.f(t, xf("R_SEL_CAN_SKIP"), f64::from(u8::from(sel.can_skip)));
    buf.f(t, xf("R_N_CLASSES"), d.members.len() as f64 / 64.0);
    buf.f(t, xf("R_DOUBLE_BOSS"), f64::from(u8::from(n_bosses > 1)));
}

/// A summary's members: `(vid, count)` in first-seen order.
#[derive(Default)]
struct Members {
    v: Vec<(u16, f64)>,
}

impl Members {
    fn add(&mut self, id: u16, n: f64) {
        if let Some(e) = self.v.iter_mut().find(|e| e.0 == id) {
            e.1 += n;
        } else {
            self.v.push((id, n));
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn summary(buf: &mut Buf, section: &str, m: &Members, n_upg: i64, n_ench: i64) {
    let p = buf.len();
    let t = buf.new_tok(schema::sec(section), TF_POOLED);
    let n = pysum(m.v.iter().map(|e| e.1));
    buf.f(t, xf("U_N"), n / 40.0);
    buf.f(t, xf("U_UPG"), n_upg as f64 / 20.0);
    buf.f(t, xf("U_ENCH"), n_ench as f64 / 10.0);
    buf.f(t, xf("U_DISTINCT"), m.v.len() as f64 / 40.0);
    let p16 = u16::try_from(p).unwrap_or(u16::MAX);
    for &(id, w) in &m.v {
        if id != 0 {
            buf.pool_tok.push(p16);
            buf.pool_vid.push(id);
            buf.pool_w.push(w);
        }
    }
}

fn card_vid(id: Option<&Value>) -> u16 {
    match id {
        None | Some(Value::Null) => 0,
        Some(v) => vocab::load().id(Reg::Card, Some(&py_str(v))),
    }
}

/// `_pile_summary(counted, *singles)`.
fn pile_summary(counted: &[Value], singles: &[&[Value]]) -> (Members, i64, i64) {
    let mut m = Members::default();
    let (mut n_upg, mut n_ench) = (0i64, 0i64);
    for cn in counted {
        let a = cn.as_array().map_or(&[][..], Vec::as_slice);
        let card = a.first().unwrap_or(&Value::Null);
        let n = ff(a.get(1)); // float(n): counts are positive ints
        m.add(card_vid(get(card, "id")), n);
        #[allow(clippy::cast_possible_truncation)]
        let ni = n as i64;
        if get(card, "upgraded").is_some_and(truthy) {
            n_upg += ni;
        }
        if get(card, "enchantment").is_some_and(truthy) {
            n_ench += ni;
        }
    }
    for cards in singles {
        for card in *cards {
            m.add(card_vid(get(card, "id")), 1.0);
            if get(card, "upgraded").is_some_and(truthy) {
                n_upg += 1;
            }
            if get(card, "enchantment").is_some_and(truthy) {
                n_ench += 1;
            }
        }
    }
    (m, n_upg, n_ench)
}

fn head(
    d: &Decision,
    run: &Run<'_>,
    cache: &mut TokCache,
    snap: Option<&Value>,
    room: Option<&str>,
    encounter: Option<&str>,
) -> (Buf, Value) {
    let mut buf = Buf::new(if d.is_combat() { 8 } else { 64 });
    let st = static_of(cache, run).clone();
    runctx(
        &mut buf,
        d,
        run,
        &st.fm,
        st.bosses.len(),
        &d.sel,
        room,
        encounter,
    );
    if let Some(snap) = snap {
        let (m, u, e) = pile_summary(
            list_or_empty(snap, "draw_unknown"),
            &[
                list_or_empty(snap, "draw_top"),
                list_or_empty(snap, "draw_bottom"),
            ],
        );
        summary(&mut buf, "SUM_DRAW", &m, u, e);
        let (m, u, e) = pile_summary(list_or_empty(snap, "discard"), &[]);
        summary(&mut buf, "SUM_DISCARD", &m, u, e);
        let (m, u, e) = pile_summary(list_or_empty(snap, "exhaust"), &[]);
        summary(&mut buf, "SUM_EXHAUST", &m, u, e);
    } else {
        for s in ["SUM_DRAW", "SUM_DISCARD", "SUM_EXHAUST"] {
            summary(&mut buf, s, &Members::default(), 0, 0);
        }
    }
    // SUM_DECK
    let mut m = Members::default();
    let mut n_upg = 0;
    for (cid, up) in &run.deck {
        m.add(vocab::load().id(Reg::Card, Some(cid)), 1.0);
        if *up {
            n_upg += 1;
        }
    }
    let n_ench = run
        .ench
        .iter()
        .take(run.deck.len())
        .filter(|e| truthy(e))
        .count();
    summary(
        &mut buf,
        "SUM_DECK",
        &m,
        n_upg,
        i64::try_from(n_ench).unwrap_or(0),
    );
    let mut rel = Members::default();
    for r in &run.relics {
        rel.add(vocab::load().id(Reg::Relic, Some(r)), 1.0);
    }
    summary(&mut buf, "SUM_RELICS", &rel, 0, 0);
    let act = f64::from(run.act());
    for (k, b) in st.bosses.iter().take(MAX_BOSS_TOKENS).enumerate() {
        let t = buf.new_tok(schema::sec("BOSS"), 0);
        buf.c(t, xc("X_ID_MAIN"), vid_s(Reg::Encounter, Some(b)));
        #[allow(clippy::cast_precision_loss)]
        buf.f(t, xf("B_ORDER"), k as f64);
        buf.f(t, xf("B_ACT"), act / 3.0);
    }
    (buf, st.fm.clone())
}

// ── combat ───────────────────────────────────────────────────────────────

/// `combat_picks(snap, acts)`: option indices already picked in the open
/// choice.
fn combat_picks(snap: &Value, acts: &[Value]) -> Vec<usize> {
    let Some(ch) = get(snap, "choice").filter(|c| !c.is_null()) else {
        return Vec::new();
    };
    let offered: Vec<i64> = acts
        .iter()
        .filter(|a| get(a, "action").and_then(Value::as_str) == Some("choose"))
        .filter_map(|a| {
            let s = list_or_empty(a, "selection");
            s.first().and_then(py_int)
        })
        .collect();
    (0..list_or_empty(ch, "options").len())
        .filter(|i| !i64::try_from(*i).is_ok_and(|ii| offered.contains(&ii)))
        .collect()
}

fn intent_kind(i: &Value) -> String {
    let k = get(i, "kind").map_or_else(|| "Unknown".to_owned(), py_str);
    if k == "MultiAttack" {
        "Attack".to_owned()
    } else {
        k
    }
}

/// `_intent_cols` for enemy `j`: (cat column, value) and (cont column,
/// f32 value) writes.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn intent_cols(j: usize, tele: &Value, fc: &Value) -> (Vec<(usize, i16)>, Vec<(usize, f32)>) {
    let mut cats = Vec::new();
    let mut conts = Vec::new();
    if let Some(row) = tele.as_array().and_then(|a| a.get(j)) {
        let ints = row.as_array().map_or(&[][..], Vec::as_slice);
        conts.push((xf("E_N_INTENTS"), (ints.len() as f64 / 3.0) as f32));
        for (k, (c_kind, c_eff, c_times)) in [
            (
                1usize,
                (xc("X_INTENT_T2"), xf("E_T2_EFF"), xf("E_T2_TIMES")),
            ),
            (2, (xc("X_INTENT_T3"), xf("E_T3_EFF"), xf("E_T3_TIMES"))),
        ] {
            if let Some(i) = ints.get(k) {
                cats.push((c_kind, schema::intent_id(&intent_kind(i))));
                let eff = match get(i, "effective_amount") {
                    Some(v) => Some(v),
                    None => get(i, "amount"),
                };
                conts.push((c_eff, (ff(eff) / 30.0) as f32));
                conts.push((c_times, (ff(get(i, "times")) / 5.0) as f32));
            }
        }
    }
    if let Some(row) = fc.as_array().and_then(|a| a.get(j)) {
        let steps = row.as_array().map_or(&[][..], Vec::as_slice);
        let fcs = [
            (
                xc("X_FC1"),
                xf("E_F1_CERT"),
                xf("E_F1_AMT"),
                xf("E_F1_TIMES"),
                xf("E_F1_TOT"),
            ),
            (
                xc("X_FC2"),
                xf("E_F2_CERT"),
                xf("E_F2_AMT"),
                xf("E_F2_TIMES"),
                xf("E_F2_TOT"),
            ),
            (
                xc("X_FC3"),
                xf("E_F3_CERT"),
                xf("E_F3_AMT"),
                xf("E_F3_TIMES"),
                xf("E_F3_TOT"),
            ),
        ];
        for (st, (c_fc, c_cert, c_amt, c_times, c_tot)) in steps.iter().take(3).zip(fcs) {
            let cert = ff(get(st, "certainty"));
            let ints = list_or_empty(st, "intents");
            let head = ints.first().filter(|h| truthy(h));
            let id = head.map_or(0, |h| schema::intent_id(&intent_kind(h)));
            cats.push((c_fc, id));
            let atk = pysum(ints.iter().map(tactical::attack_total));
            conts.push((c_cert, cert as f32));
            conts.push((
                c_amt,
                (ff(head.and_then(|h| get(h, "amount"))) / 30.0) as f32,
            ));
            conts.push((
                c_times,
                (ff(head.and_then(|h| get(h, "times"))) / 5.0) as f32,
            ));
            conts.push((c_tot, (atk / 50.0) as f32));
        }
    }
    (cats, conts)
}

/// `_choice_ench(pc_cards, s, piles)`.
fn choice_ench(pc_cards: &[Value], s: usize, piles: &Value) -> i16 {
    let Some(opt) = pc_cards.get(s).filter(|o| truthy(o)) else {
        return 0;
    };
    let pile_key = match get(opt, "pile").map(py_str).as_deref() {
        Some("hand") => "hand",
        Some("draw") => "draw_pile",
        Some("discard") => "discard_pile",
        Some("exhaust") => "exhaust_pile",
        _ => "",
    };
    let Some(pile) = get(piles, pile_key).filter(|p| !p.is_null()) else {
        return 0;
    };
    let Some(card) = at(Some(pile), get(opt, "index")).filter(|c| truthy(c)) else {
        return 0;
    };
    let name = ench_name(get(card, "enchantment"));
    vid_s(Reg::Enchantment, name.as_deref())
}

/// The id gather (`_id_tables`): `(MAIN, MOD, REL, MAIN fallback)` content
/// ids of an S_ctx token of section `s`.
fn sctx_ids(s: u8, row: &[i16]) -> [u16; 4] {
    let l = sctx::luts();
    let g = |sl: usize| -> u16 {
        let i = usize::try_from(row[sl]).unwrap_or(0);
        l[sl].get(i).copied().unwrap_or(0)
    };
    let name = schema::section_name(s);
    match name {
        "HAND" | "DRAW_UNK" | "DRAW_TOP" | "DRAW_BOT" | "DISCARD" | "EXHAUST" | "PLAY" => {
            [g(slot("card")), g(slot("enchantment")), 0, 0]
        }
        "ACT_PLAY" => [
            g(slot("card")),
            g(slot("enchantment")),
            g(slot("monster")),
            0,
        ],
        "RELIC" => [g(slot("relic")), 0, 0, 0],
        "POTION" | "ACT_POTION" => [g(slot("potion")), 0, 0, 0],
        "PPOWER" => [g(slot("power")), 0, 0, 0],
        "EPOWER" => [g(slot("power")), 0, g(slot("monster")), g(slot("card"))],
        "ENEMY" => [g(slot("monster")), 0, 0, 0],
        "EHIST" => [0, 0, g(slot("monster")), 0],
        _ => [0, 0, 0, 0],
    }
}

fn i16_of(x: u16) -> i16 {
    i16::try_from(x).unwrap_or(i16::MAX)
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]
fn encode_combat(d: &Decision, rm: &RunManager, cache: &mut TokCache) -> Result<TokRow, String> {
    let cv = d
        .combat
        .as_ref()
        .ok_or("combat decision without a combat view")?;
    let bv: &CombatState = &cv.bv;
    let snap = &cv.snap;
    let acts = &d.obs_actions;
    let run = Run::new(rm, None);
    let picked = combat_picks(snap, acts);
    let (hb, _fm) = head(d, &run, cache, Some(snap), Some(cv.room), cv.encounter_id);
    let rl2 = Rl2::of(bv);
    let tk = sctx::tokens(snap, acts, &rl2)?;
    let tele = sts2_wire::emit::combat::enemy_telegraphs(bv).map_err(|e| e.to_string())?;
    let fc = sts2_wire::emit::combat::enemy_intent_forecast(bv, 3).map_err(|e| e.to_string())?;
    let n_mem = acts.len();
    if tk.n_act != n_mem {
        return Err(format!(
            "S_ctx built {} action tokens for {n_mem} members",
            tk.n_act
        ));
    }
    let t = tk.len();
    let lo = hb.len();
    let n = lo + t;
    let a0 = t - n_mem;

    // cat: head rows, then S_ctx slots + trailing columns.
    let mut cat = Vec::with_capacity(n * N_CAT);
    cat.extend_from_slice(&hb.cat);
    let mut xcat = vec![0i16; t * (N_CAT - SCTX_N_CAT)];
    let mut xcont = vec![0f32; t * (N_CONT - SCTX_N_CONT)];
    let xo = SCTX_N_CAT;
    let fo = SCTX_N_CONT;
    let xw = N_CAT - SCTX_N_CAT;
    let fw = N_CONT - SCTX_N_CONT;
    let setc = |xcat: &mut [i16], ti: usize, col: usize, v: i16| xcat[ti * xw + col - xo] = v;
    for ti in 0..t {
        let ids = sctx_ids(tk.sec[ti], tk.cat_row(ti));
        let main = if ids[0] > 0 { ids[0] } else { ids[3] };
        setc(&mut xcat, ti, xc("X_ID_MAIN"), i16_of(main));
        setc(&mut xcat, ti, xc("X_ID_MOD"), i16_of(ids[1]));
        setc(&mut xcat, ti, xc("X_ID_REL"), i16_of(ids[2]));
    }
    // Action tokens: kind, potion slot and target, choose content.
    for (k, &kind) in d.kinds.iter().enumerate() {
        setc(&mut xcat, a0 + k, xc("X_KIND"), i16::from(kind));
    }
    let ch = get(snap, "choice").filter(|c| !c.is_null());
    let piles = get(&cv.d_bv, "deck")
        .filter(|p| truthy(p))
        .cloned()
        .unwrap_or(Value::Null);
    let enemies = list_or_empty(snap, "enemies");
    let mut pc_cards: Option<Vec<Value>> = None;
    for (k, &kind) in d.kinds.iter().enumerate() {
        let a = &acts[k];
        if kind == ak::POTION || kind == ak::POTION_TARGET {
            let rep = &acts[d.members[d.classes[k]][0]];
            let s = get(rep, "slot")
                .and_then(py_int)
                .ok_or("potion action without slot")?;
            setc(&mut xcat, a0 + k, xc("X_POS"), pos(s));
            if let Some(ti) = get(a, "target_index").filter(|v| !v.is_null()) {
                let e = at(get(snap, "enemies"), Some(ti)).filter(|e| truthy(e));
                let id = e.map_or(0, |e| vid(Reg::Monster, get(e, "kind")));
                setc(&mut xcat, a0 + k, xc("X_ID_REL"), id);
            }
        } else if kind == ak::CHOOSE {
            let Some(ch) = ch else { continue };
            let s0 = list_or_empty(a, "selection")
                .first()
                .and_then(py_int)
                .ok_or("choose action without selection")?;
            let opts = get(ch, "options");
            let Some(opt) = at(opts, Some(&Value::from(s0))) else {
                continue;
            };
            let pcc = pc_cards.get_or_insert_with(|| {
                sts2_wire::emit::pending_choice(bv)
                    .and_then(|pc| get(&pc, "cards").and_then(Value::as_array).cloned())
                    .unwrap_or_default()
            });
            setc(
                &mut xcat,
                a0 + k,
                xc("X_ID_MAIN"),
                vid(Reg::Card, get(opt, "id")),
            );
            let su = usize::try_from(s0).unwrap_or(usize::MAX);
            setc(
                &mut xcat,
                a0 + k,
                xc("X_ID_MOD"),
                choice_ench(pcc, su, &piles),
            );
        }
    }
    // Potion tokens: the belt slot (the snapshot drops empty slots).
    let pm: Vec<usize> = (0..a0)
        .filter(|&i| tk.sec[i] == schema::sec("POTION"))
        .collect();
    if !pm.is_empty() {
        let belt = get(&cv.d_bv, "player")
            .and_then(|p| get(p, "potions"))
            .map_or(&[][..], |b| b.as_array().map_or(&[][..], Vec::as_slice));
        let mut slots: Vec<usize> = belt
            .iter()
            .enumerate()
            .filter(|(_, p)| truthy(p))
            .map(|(i, _)| i)
            .collect();
        if slots.is_empty() {
            slots = (0..pm.len()).collect();
        }
        for (j, &ti) in pm.iter().enumerate() {
            let s = slots.get(j).copied().unwrap_or(j);
            setc(
                &mut xcat,
                ti,
                xc("X_POS"),
                pos(i64::try_from(s).unwrap_or(62)),
            );
        }
    }
    // Picks: the hand cards an open multi-pick choice has selected.
    if let (false, Some(ch)) = (picked.is_empty(), ch) {
        let hand_tok: Vec<usize> = (0..a0)
            .filter(|&i| tk.sec[i] == schema::sec("HAND"))
            .collect();
        let opts = list_or_empty(ch, "options");
        for i in &picked {
            let hi = opts
                .get(*i)
                .filter(|o| truthy(o))
                .and_then(|o| get(o, "hand_index"))
                .filter(|h| !h.is_null())
                .and_then(py_int);
            if let Some(hi) = hi.and_then(|h| usize::try_from(h).ok()) {
                if hi < hand_tok.len() {
                    xcont[hand_tok[hi] * fw + xf("K_PICKED") - fo] = 1.0;
                }
            }
        }
    }
    // Tactical and intent columns per enemy.
    let (md, kill, loss) = tactical::tactical(snap, acts);
    let em: Vec<usize> = (0..a0)
        .filter(|&i| tk.sec[i] == schema::sec("ENEMY"))
        .collect();
    for (j, &ti) in em.iter().enumerate() {
        xcont[ti * fw + xf("E_MAXDMG") - fo] = md.get(j).map_or(0.0, |m| (m / 50.0) as f32);
        xcont[ti * fw + xf("E_KILLABLE") - fo] = kill.get(j).map_or(0.0, |k| *k as f32);
        let dead = enemies
            .get(j)
            .is_some_and(|e| get(e, "is_dead").is_some_and(truthy));
        if !dead {
            let (cats, conts) = intent_cols(j, &tele, &fc);
            for (col, v) in cats {
                xcat[ti * xw + col - xo] |= v;
            }
            for (col, v) in conts {
                xcont[ti * fw + col - fo] += v;
            }
        }
    }
    xcont[xf("S_HP_LOSS_END") - fo] = (loss / 50.0) as f32;

    for ti in 0..t {
        cat.extend_from_slice(tk.cat_row(ti));
        cat.extend_from_slice(&xcat[ti * xw..(ti + 1) * xw]);
    }
    let mut cont = Vec::with_capacity(n * N_CONT);
    cont.extend(hb.cont.iter().map(|&x| to_f16(x)));
    for ti in 0..t {
        let r = &tk.cont[ti * SCTX_N_CONT..(ti + 1) * SCTX_N_CONT];
        cont.extend(r.iter().map(|&x| to_f16(x.clamp(-F16_MAX, F16_MAX))));
        cont.extend(xcont[ti * fw..(ti + 1) * fw].iter().map(|&x| to_f16(x)));
    }
    let mut sec = hb.sec.clone();
    sec.extend_from_slice(&tk.sec);
    let mut flags = hb.flags.clone();
    flags.extend(std::iter::repeat_n(TF_TEACHER, t));
    let a0n = n - n_mem;
    for f in &mut flags[a0n..] {
        *f |= TF_OPTION | TF_HIDDEN;
    }
    let reps: Vec<usize> = d.members.iter().map(|m| m[0]).collect();
    for &r in &reps {
        flags[a0n + r] = TF_TEACHER | TF_OPTION | TF_REP;
    }
    let u16_of = |x: usize| u16::try_from(x).unwrap_or(u16::MAX);
    let mem_tok: Vec<u16> = (0..n_mem).map(|i| u16_of(a0n + i)).collect();
    let cls_tok: Vec<u16> = reps.iter().map(|&r| mem_tok[r]).collect();
    Ok(TokRow {
        cat,
        cont,
        sec,
        tok_flags: flags,
        mem_tok,
        mem_class: d.classes.iter().map(|&c| u16_of(c)).collect(),
        cls_tok,
        cls_kind: reps.iter().map(|&r| d.kinds[r]).collect(),
        pool_tok: hb.pool_tok,
        pool_vid: hb.pool_vid,
        pool_w: hb
            .pool_w
            .iter()
            .map(|&w| f16::from_f64(w).to_bits())
            .collect(),
        sctx_lo: lo,
        n_sctx: t,
        dtype: d.dtype,
        is_combat: true,
    })
}

// ── out of combat ────────────────────────────────────────────────────────

fn card_cats(buf: &mut Buf, t: usize, cid: &Value, up: bool, ench: Option<&Value>) {
    let tv = sctx::vocab();
    buf.c(
        t,
        slot("upgraded"),
        tv.table(slot("upgraded")).index(if up { "1" } else { "0" }),
    );
    let ct = if truthy(cid) {
        tv.card_type_any(&py_str(cid))
    } else {
        ""
    };
    buf.c(t, slot("card_type"), tv.table(slot("card_type")).index(ct));
    buf.c(t, xc("X_ID_MAIN"), vid(Reg::Card, Some(cid)));
    let en = ench_name(ench);
    buf.c(t, xc("X_ID_MOD"), vid_s(Reg::Enchantment, en.as_deref()));
}

fn deck_tokens(buf: &mut Buf, run: &Run<'_>) {
    // One DECK token per distinct (card, upgraded, enchantment), first-seen order.
    let mut groups: Vec<(String, bool, Value, usize)> = Vec::new();
    for (i, (cid, up)) in run.deck.iter().enumerate() {
        let e = run.ench.get(i).cloned().unwrap_or(Value::Null);
        if let Some(g) = groups
            .iter_mut()
            .find(|g| g.0 == *cid && g.1 == *up && g.2 == e)
        {
            g.3 += 1;
        } else {
            groups.push((cid.clone(), *up, e, 1));
        }
    }
    for (cid, up, e, n) in groups {
        let t = buf.new_tok(schema::sec("DECK"), 0);
        let e = (!e.is_null()).then_some(e);
        card_cats(buf, t, &Value::String(cid), up, e.as_ref());
        buf.f(t, sc("ENCH_AMT"), ench_amount(e.as_ref()) / 5.0);
        #[allow(clippy::cast_precision_loss)]
        buf.f(t, sc("COUNT"), n as f64 / 5.0);
    }
}

type Coord = (i64, i64);

struct MapNodes<'a> {
    order: Vec<Coord>,
    nodes: BTreeMap<Coord, &'a Value>,
}

fn map_nodes(fm: &Value) -> MapNodes<'_> {
    let mut order = Vec::new();
    let mut nodes = BTreeMap::new();
    for r in list_or_empty(fm, "rows") {
        for n in r.as_array().map_or(&[][..], Vec::as_slice) {
            let k = (
                get(n, "col").and_then(py_int).unwrap_or(0),
                get(n, "row").and_then(py_int).unwrap_or(0),
            );
            if nodes.insert(k, n).is_none() {
                order.push(k);
            }
        }
    }
    MapNodes { order, nodes }
}

fn kids(n: &Value) -> Vec<Coord> {
    list_or_empty(n, "children")
        .iter()
        .map(|c| {
            (
                get(c, "col").and_then(py_int).unwrap_or(0),
                get(c, "row").and_then(py_int).unwrap_or(0),
            )
        })
        .collect()
}

fn node_type(n: &Value) -> String {
    get(n, "type").map_or_else(|| "None".to_owned(), py_str)
}

/// `_route(nodes, boss_row)`: per node, the route summary to the boss.
#[allow(clippy::cast_precision_loss)]
fn route(mn: &MapNodes<'_>, boss_row: i64) -> BTreeMap<Coord, [f64; 8]> {
    let mut memo: BTreeMap<Coord, [f64; 8]> = BTreeMap::new();
    let mut keys = mn.order.clone();
    keys.sort_by_key(|k| -k.1); // stable, as Python's `sorted`
    for key in keys {
        let n = mn.nodes[&key];
        let typ = node_type(n);
        let b = |x: bool| f64::from(u8::from(x));
        let e = b(typ == "Elite");
        let r = b(typ == "RestSite");
        let sh = b(typ == "Shop");
        let un = b(typ == "Unknown" || typ == "Event");
        let fight = b(typ == "Monster" || typ == "Elite");
        let ks: Vec<[f64; 8]> = kids(n)
            .iter()
            .filter_map(|k| memo.get(k).copied())
            .collect();
        let st = if ks.is_empty() {
            [
                e,
                e,
                r,
                r,
                sh,
                un,
                if r == 0.0 { fight } else { 0.0 },
                (boss_row - key.1) as f64,
            ]
        } else {
            let mn_ = |i: usize| ks.iter().map(|k| k[i]).fold(f64::INFINITY, f64::min);
            let mx_ = |i: usize| ks.iter().map(|k| k[i]).fold(f64::NEG_INFINITY, f64::max);
            [
                e + mn_(0),
                e + mx_(1),
                r + mn_(2),
                r + mx_(3),
                sh + mx_(4),
                un + mx_(5),
                if r == 0.0 { fight + mn_(6) } else { 0.0 },
                (boss_row - key.1) as f64,
            ]
        };
        memo.insert(key, st);
    }
    memo
}

/// `_map_body`: MAPNODE tokens (the reachable cone, nearest rows first, at
/// most 40) and each move column's route summary.
#[allow(clippy::cast_precision_loss)]
fn map_body(buf: &mut Buf, fm: &Value, moves: &[Value]) -> BTreeMap<i64, [f64; 8]> {
    let mn = map_nodes(fm);
    let br = boss_row(fm);
    let rt = route(&mn, br);
    let cur = get(fm, "current_coord").filter(|c| truthy(c));
    let cr = cur
        .and_then(|c| get(c, "row"))
        .and_then(py_int)
        .unwrap_or(-1);
    let cur_key = cur.map(|c| (get(c, "col").and_then(py_int).unwrap_or(0), cr));
    let start: Vec<Coord> = match cur_key.and_then(|k| mn.nodes.get(&k)) {
        Some(n) => kids(n),
        None => mn.order.iter().copied().filter(|k| k.1 == cr + 1).collect(),
    };
    // _paths_through
    let mut cnt: BTreeMap<Coord, i64> = BTreeMap::new();
    for k in &start {
        if mn.nodes.contains_key(k) {
            *cnt.entry(*k).or_insert(0) += 1;
        }
    }
    let mut keys = mn.order.clone();
    keys.sort_by_key(|k| k.1);
    for key in keys {
        let c = cnt.get(&key).copied().unwrap_or(0);
        if c == 0 {
            continue;
        }
        for k in kids(mn.nodes[&key]) {
            if mn.nodes.contains_key(&k) {
                *cnt.entry(k).or_insert(0) += c;
            }
        }
    }
    let nxt: Vec<i64> = moves
        .iter()
        .filter(|a| get(a, "action").and_then(Value::as_str) == Some("move"))
        .filter_map(|a| get(a, "col").and_then(py_int))
        .collect();
    let mut cone: Vec<Coord> = cnt.keys().copied().collect();
    cone.sort_by_key(|k| (k.1, k.0));
    cone.truncate(MAX_MAPNODES);
    for k in cone {
        let n = mn.nodes[&k];
        let t = buf.new_tok(schema::sec("MAPNODE"), 0);
        buf.c(t, xc("X_ROOM"), room_of_node(&node_type(n)));
        buf.c(t, xc("X_POS"), pos(k.0));
        buf.f(t, xf("N_ROW_OFF"), (k.1 - cr) as f64 / 17.0);
        buf.f(t, xf("N_COL"), k.0 as f64 / 6.0);
        buf.f(
            t,
            xf("N_NEXT"),
            f64::from(u8::from(k.1 == cr + 1 && nxt.contains(&k.0))),
        );
        buf.f(t, xf("N_PATHS"), (cnt[&k] as f64).ln_1p() / 5.0);
    }
    let mut out = BTreeMap::new();
    for col in nxt {
        if let Some(st) = rt.get(&(col, cr + 1)) {
            out.insert(col, *st);
        }
    }
    out
}

fn room_of_node(typ: &str) -> i16 {
    let r = match typ {
        "Monster" => "monster",
        "Elite" => "elite",
        "Boss" => "boss",
        "Shop" => "shop",
        "RestSite" => "rest_site",
        "Treasure" => "treasure",
        "Ancient" => "ancient",
        "Event" => "event",
        _ => "unknown",
    };
    schema::room_id(r).unwrap_or(2)
}

/// `honest_event_deltas` -> `event_delta_columns`: the O_EV_* values.
fn event_delta_columns(info: &Value) -> [(usize, f64); 9] {
    const F: [&str; 9] = [
        "gold_delta",
        "hp_delta",
        "max_hp_delta",
        "relics_gained",
        "relics_lost",
        "potions_gained",
        "potions_lost",
        "cards_gained",
        "cards_lost",
    ];
    let mut d: BTreeMap<&str, i64> = BTreeMap::new();
    let has_preview = F[..3]
        .iter()
        .all(|k| get(info, k).is_some_and(|v| !v.is_null()));
    if has_preview {
        for k in F {
            d.insert(
                k,
                get(info, k)
                    .filter(|v| truthy(v))
                    .and_then(py_int)
                    .unwrap_or(0),
            );
        }
        if d["relics_gained"] > 0 {
            for k in [
                "gold_delta",
                "hp_delta",
                "max_hp_delta",
                "potions_gained",
                "cards_gained",
            ] {
                d.insert(k, 0);
            }
            d.insert("relics_gained", 1);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let g = |k: &str| d.get(k).copied().unwrap_or(0) as f64;
    [
        (xf("O_EV_GOLD"), g("gold_delta") / 100.0),
        (xf("O_EV_HP"), g("hp_delta") / 20.0),
        (xf("O_EV_MAXHP"), g("max_hp_delta") / 10.0),
        (xf("O_EV_RELICS_GAIN"), g("relics_gained") / 1.0),
        (xf("O_EV_POTIONS_GAIN"), g("potions_gained") / 3.0),
        (xf("O_EV_CARDS_GAIN"), g("cards_gained") / 3.0),
        (xf("O_EV_CARDS_LOSE"), g("cards_lost") / 3.0),
        (xf("O_EV_RELICS_LOSE"), g("relics_lost") / 3.0),
        (xf("O_EV_POTIONS_LOSE"), g("potions_lost") / 3.0),
    ]
}

/// `_neow_relic((label, option_id))`.
fn neow_relic(opt: &Value) -> i16 {
    let v = vocab::load();
    for k in ["label", "option_id"] {
        if let Some(s) = get(opt, k).filter(|s| truthy(s)) {
            let s = py_str(s);
            if v.has(Reg::Relic, &s) {
                return vid_s(Reg::Relic, Some(&s));
            }
        }
    }
    0
}

struct OptCtx<'a, 'r> {
    d: &'a Decision,
    run: &'a Run<'r>,
    copies: BTreeMap<String, i64>,
    route: BTreeMap<i64, [f64; 8]>,
    fm: &'a Value,
}

/// `_option`: the token of option `i` (its class's representative).
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn option(buf: &mut Buf, cx: &OptCtx<'_, '_>, i: usize, a: &Value, kind: u8) {
    let run = cx.run;
    let og = |k: &str| run.og(k);
    let name = kind_name(kind);
    let flags = TF_OPTION | TF_REP;
    let card_opt = |buf: &mut Buf, cid: &Value, up: bool, ench: Option<&Value>| -> usize {
        let t = buf.new_tok(schema::sec("OPT_CARD"), flags);
        card_cats(buf, t, cid, up, ench);
        buf.f(t, sc("ENCH_AMT"), ench_amount(ench) / 5.0);
        let copies = cx.copies.get(&py_str(cid)).copied().unwrap_or(0);
        buf.f(t, xf("K_DECK_COPIES"), copies as f64 / 5.0);
        t
    };
    let deck_opt = |buf: &mut Buf, deck_i: Option<&Value>| -> usize {
        let di = deck_i
            .and_then(py_int)
            .and_then(|x| usize::try_from(x).ok());
        let ent = di.and_then(|x| run.deck.get(x));
        let e = at(og("deck_enchantments"), deck_i);
        match ent {
            Some((cid, up)) => card_opt(buf, &Value::String(cid.clone()), *up, e),
            None => card_opt(buf, &Value::Null, false, e),
        }
    };
    let pair = |ent: Option<&Value>| -> (Value, bool) {
        match ent {
            Some(Value::Array(x)) => (
                x.first().cloned().unwrap_or(Value::Null),
                x.get(1).is_some_and(truthy),
            ),
            Some(v) => (v.clone(), false),
            None => (Value::Null, false),
        }
    };
    let idx = get(a, "idx");
    let t = match name {
        "PICK_CARD_REWARD" => {
            let cid = at(og("card_reward_option_ids"), idx)
                .cloned()
                .unwrap_or(Value::Null);
            let up = at(og("card_reward_option_upgraded"), idx).is_some_and(truthy);
            let e = at(og("card_reward_option_enchantments"), idx);
            card_opt(buf, &cid, up, e)
        }
        "PICK_CARD_BUNDLE" => {
            let pack = at(og("card_reward_bundles"), idx)
                .and_then(Value::as_array)
                .map_or(&[][..], Vec::as_slice);
            let first = pack.first().cloned().unwrap_or(Value::Null);
            let t = card_opt(buf, &first, false, None);
            buf.f(t, sc("COUNT"), pack.len() as f64 / 5.0);
            t
        }
        "SMITH_CARD" => {
            let (cid, up) = pair(at(og("smith_options"), idx));
            card_opt(buf, &cid, up, at(og("smith_option_enchantments"), idx))
        }
        "REMOVE_DECK_CARD" => {
            let (cid, up) = pair(at(og("card_removal_options"), idx));
            card_opt(
                buf,
                &cid,
                up,
                at(og("card_removal_option_enchantments"), idx),
            )
        }
        "GRANT_DECK_CARD" => {
            let (cid, up) = pair(at(og("card_grant_options"), idx));
            card_opt(buf, &cid, up, at(og("card_grant_option_enchantments"), idx))
        }
        "UPGRADE_DECK_CARD" | "TRANSFORM_DECK_CARD" | "ENCHANT_DECK_CARD" => {
            let key = match name {
                "UPGRADE_DECK_CARD" => "upgrade_select_deck_indices",
                "TRANSFORM_DECK_CARD" => "transform_select_deck_indices",
                _ => "enchant_select_deck_indices",
            };
            let di = at(og(key), idx).cloned();
            deck_opt(buf, di.as_ref())
        }
        "SHOP_REMOVE_CARD" => {
            let t = deck_opt(buf, get(a, "deck_idx"));
            let cost = og("shop").and_then(|s| get(s, "removal_cost"));
            buf.f(t, xf("O_PRICE"), ff(cost) / 300.0);
            t
        }
        "SHOP_BUY_CARD" | "SHOP_BUY_RELIC" | "SHOP_BUY_POTION" => {
            let shelf = match name {
                "SHOP_BUY_CARD" => "cards",
                "SHOP_BUY_RELIC" => "relics",
                _ => "potions",
            };
            let want = idx.and_then(py_int);
            let items = og("shop").map_or(&[][..], |s| list_or_empty(s, shelf));
            let empty = Value::Object(serde_json::Map::new());
            let e = items
                .iter()
                .rev()
                .find(|x| get(x, "idx").and_then(py_int) == want && want.is_some())
                .unwrap_or(&empty);
            let t = if name == "SHOP_BUY_CARD" {
                let cid = get(e, "id").cloned().unwrap_or(Value::Null);
                card_opt(
                    buf,
                    &cid,
                    get(e, "upgraded").is_some_and(truthy),
                    get(e, "enchantment"),
                )
            } else {
                let (sec, reg) = if name == "SHOP_BUY_RELIC" {
                    ("OPT_RELIC", Reg::Relic)
                } else {
                    ("OPT_POTION", Reg::Potion)
                };
                let t = buf.new_tok(schema::sec(sec), flags);
                buf.c(t, xc("X_ID_MAIN"), vid(reg, get(e, "id")));
                t
            };
            buf.f(t, xf("O_PRICE"), ff(get(e, "price")) / 300.0);
            t
        }
        "PICK_RELIC_REWARD" | "PICK_BOSS_RELIC" => {
            let lst = og(if name == "PICK_RELIC_REWARD" {
                "relic_reward_ids"
            } else {
                "boss_relic_options"
            });
            let t = buf.new_tok(schema::sec("OPT_RELIC"), flags);
            let zero = Value::from(0);
            let ii = get(a, "idx").unwrap_or(&zero);
            buf.c(t, xc("X_ID_MAIN"), vid(Reg::Relic, at(lst, Some(ii))));
            t
        }
        "POTION_SWAP_DISCARD" => {
            let t = buf.new_tok(schema::sec("OPT_POTION"), flags);
            let p = get(a, "discard_slot")
                .and_then(py_int)
                .and_then(|s| usize::try_from(s).ok())
                .and_then(|s| run.potions.get(s));
            buf.c(
                t,
                xc("X_ID_MAIN"),
                vid_s(Reg::Potion, p.map(String::as_str)),
            );
            t
        }
        "MOVE" => {
            let t = buf.new_tok(schema::sec("OPT_MOVE"), flags);
            let col = get(a, "col").and_then(py_int).unwrap_or(0);
            let mn = map_nodes(cx.fm);
            let cr = cur_row(cx.fm);
            let typ = mn
                .nodes
                .get(&(col, cr + 1))
                .map_or_else(|| "None".to_owned(), |n| node_type(n));
            buf.c(t, xc("X_ROOM"), room_of_node(&typ));
            if let Some(st) = cx.route.get(&col) {
                for (k, c) in [
                    "M_ELITE_MIN",
                    "M_ELITE_MAX",
                    "M_REST_MIN",
                    "M_REST_MAX",
                    "M_SHOP_MAX",
                    "M_UNKNOWN_MAX",
                    "M_FIGHTS_TO_REST",
                ]
                .iter()
                .enumerate()
                {
                    buf.f(t, xf(c), st[k] / 5.0);
                }
                buf.f(t, xf("M_ROWS_LEFT"), st[7] / 17.0);
            }
            buf.c(t, xc("X_KIND"), i16::from(kind));
            buf.c(t, xc("X_POS"), pos(col));
            return;
        }
        "EVENT_CHOICE" => {
            let infos_v;
            let infos = if let Some(v) = og("event_options").filter(|v| !v.is_null()) {
                v
            } else {
                infos_v = emit_rm::event_option_infos(run.rm);
                &infos_v
            };
            let empty = Value::Object(serde_json::Map::new());
            let opt = at(Some(infos), idx).filter(|o| truthy(o)).unwrap_or(&empty);
            let t = buf.new_tok(schema::sec("OPT_EVENT"), flags);
            let ev = event_name(cx.d, run);
            buf.c(t, xc("X_ID_MAIN"), vid_s(Reg::Event, ev.as_deref()));
            buf.c(t, xc("X_ID_MOD"), neow_relic(opt));
            for (col, v) in event_delta_columns(opt) {
                buf.f(t, col, v);
            }
            t
        }
        "REST_OPTION" => {
            let t = buf.new_tok(schema::sec("OPT_REST"), flags);
            let opts: Value = og("rest_site_options")
                .cloned()
                .unwrap_or_else(|| serde_json::json!(emit_rm::rest_site_options(run.rm)));
            let zero = Value::from(0);
            let ii = get(a, "idx").unwrap_or(&zero);
            let opt = at(Some(&opts), Some(ii));
            let rid = match opt {
                Some(Value::Array(x)) => x.first(),
                o => o,
            };
            let id = rid
                .filter(|r| truthy(r))
                .map_or(0, |r| vid(Reg::Rest, Some(r)));
            buf.c(t, xc("X_ID_MAIN"), id);
            t
        }
        _ => buf.new_tok(schema::sec("OPT_FIXED"), flags),
    };
    buf.c(t, xc("X_KIND"), i16::from(kind));
    buf.c(t, xc("X_POS"), pos(i64::try_from(i).unwrap_or(62)));
}

/// `_offer`: the potion a swap would add, the enchantment an enchant pick
/// stamps.
fn offer(buf: &mut Buf, d: &Decision, run: &Run<'_>) {
    if d.dtype == dt::POTION_SWAP {
        let pot = match run.og("pending_potion_swap").filter(|v| !v.is_null()) {
            Some(v) => Some(py_str(v)),
            None => run
                .rm
                .run_state
                .pending_potion_swaps
                .first()
                .map(|k| format!("{k:?}")),
        };
        if let Some(p) = pot.filter(|p| !p.is_empty()) {
            let t = buf.new_tok(schema::sec("OFFER"), 0);
            buf.c(t, xc("X_KIND"), i16::from(ak::POTION_SWAP_DISCARD));
            buf.c(t, xc("X_ID_MAIN"), vid_s(Reg::Potion, Some(&p)));
        }
    } else if d.dtype == dt::CARD_ENCHANT {
        let e: Option<Value> = match run.og("pending_enchantment").filter(|v| !v.is_null()) {
            Some(v) => Some(v.clone()),
            None => run
                .rm
                .run_state
                .pending_deck_enchants
                .first()
                .map(|p| Value::String(p.enchantment.python_name().to_owned())),
        };
        if let Some(e) = e.filter(truthy) {
            let t = buf.new_tok(schema::sec("OFFER"), 0);
            buf.c(t, xc("X_KIND"), i16::from(ak::ENCHANT_DECK_CARD));
            let en = ench_name(Some(&e));
            buf.c(t, xc("X_ID_MAIN"), vid_s(Reg::Enchantment, en.as_deref()));
        }
    }
}

fn encode_ooc(d: &Decision, rm: &RunManager, cache: &mut TokCache) -> Result<TokRow, String> {
    let ov = d
        .ooc
        .as_ref()
        .ok_or("ooc decision without a run observation")?;
    let run = Run::new(rm, Some(&ov.run_obs));
    let (mut buf, fm) = head(d, &run, cache, None, None, None);
    deck_tokens(&mut buf, &run);
    for r in &run.relics {
        let t = buf.new_tok(schema::sec("RELIC"), 0);
        buf.c(t, xc("X_ID_MAIN"), vid_s(Reg::Relic, Some(r)));
    }
    for (j, p) in run.potions.iter().enumerate() {
        if !p.is_empty() {
            let t = buf.new_tok(schema::sec("POTION"), 0);
            buf.c(t, xc("X_ID_MAIN"), vid_s(Reg::Potion, Some(p)));
            buf.c(t, xc("X_POS"), pos(i64::try_from(j).unwrap_or(62)));
        }
    }
    offer(&mut buf, d, &run);
    let route = if d.dtype == dt::MAP {
        map_body(&mut buf, &fm, &d.obs_actions)
    } else {
        BTreeMap::new()
    };
    let mut copies: BTreeMap<String, i64> = BTreeMap::new();
    for (c, _) in &run.deck {
        *copies.entry(c.clone()).or_insert(0) += 1;
    }
    let cx = OptCtx {
        d,
        run: &run,
        copies,
        route,
        fm: &fm,
    };
    let reps: Vec<usize> = d.members.iter().map(|m| m[0]).collect();
    let mut cls_tok = Vec::with_capacity(reps.len());
    for (c, &r) in reps.iter().enumerate() {
        cls_tok.push(u16::try_from(buf.len()).unwrap_or(u16::MAX));
        option(&mut buf, &cx, c, &d.obs_actions[r], d.kinds[r]);
    }
    let u16_of = |x: usize| u16::try_from(x).unwrap_or(u16::MAX);
    Ok(TokRow {
        cont: buf.cont.iter().map(|&x| to_f16(x)).collect(),
        cat: buf.cat,
        sec: buf.sec,
        tok_flags: buf.flags,
        mem_tok: d.classes.iter().map(|&c| cls_tok[c]).collect(),
        mem_class: d.classes.iter().map(|&c| u16_of(c)).collect(),
        cls_kind: reps.iter().map(|&r| d.kinds[r]).collect(),
        cls_tok,
        pool_tok: buf.pool_tok,
        pool_vid: buf.pool_vid,
        pool_w: buf
            .pool_w
            .iter()
            .map(|&w| f16::from_f64(w).to_bits())
            .collect(),
        sctx_lo: 0,
        n_sctx: 0,
        dtype: d.dtype,
        is_combat: false,
    })
}

/// Fill `cache` as [`encode`] would at `d` without building the row: a host
/// that replays a run without the net calls it at every decision so the
/// per-floor map read is the one the floor's first decision made.
pub fn prime(d: &Decision, rm: &RunManager, cache: &mut TokCache) {
    let obs = d.ooc.as_ref().map(|ov| &ov.run_obs);
    let run = Run::new(rm, obs);
    let _ = static_of(cache, &run);
}

/// `tokens.encode(d, rare=None, priv=False)`: decision `d`'s policy row.
/// `rm` is the live run (`E2eEnv::rm`) at `d`; `cache` is this run's
/// [`TokCache`].
///
/// # Errors
///
/// A decision without its view, an S_ctx block not aligned with the
/// members, or a telegraph the wire cannot emit.
pub fn encode(d: &Decision, rm: &RunManager, cache: &mut TokCache) -> Result<TokRow, String> {
    if d.is_combat() {
        encode_combat(d, rm, cache)
    } else {
        encode_ooc(d, rm, cache)
    }
}
