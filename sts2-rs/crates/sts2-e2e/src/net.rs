//! Policy forward of `ppo.e2e.net.E2ENet` (see the crate docs).
//!
//! Plain fp32 Rust, batch size one, no ML crate: [`E2eNet::from_bytes`] loads
//! an `.e2ew` file (`ppo.e2e.export_wb weights`) and [`E2eNet::logits`] runs
//! `E2ENet.policy(to_batch(row))` on one [`PolicyRow`].
//!
//! The forward, in `net.py`'s terms:
//!
//! - `to_batch`/`_index`: tokens flagged `TF_HIDDEN` are dropped; the class
//!   representatives (`cls_tok`) and pooled tokens (`pool_tok`) are remapped to
//!   positions among the visible tokens. At B=1 there is no padding, so every
//!   visible token is a key.
//! - `embed`/`_lookups`: section embedding + the four ID columns (shared ID
//!   table times a per-column gain) + the small-column `embedding_bag` (absent
//!   value -> row 0, i.e. nothing) + for a pooled token the weighted mean of its
//!   entries' ID rows (times `pool_gain`) and prior rows, the weight sum clamped
//!   to at least 1 (a sparse sum here, not the dense `[J, 4096]` matmul) + the
//!   numeric stem `Linear(SiLU(Linear(symlog([cont; prior]))))`, where `prior`
//!   is the content-prior row of `X_ID_MAIN`.
//! - `Block.forward` x `n_layers`: pre-norm RMSNorm, multi-head attention
//!   (scale `1/sqrt(head_dim)`), RMSNorm, SwiGLU, no biases. The last block
//!   runs its queries and MLP only at the tokens the heads read
//!   (`Block.forward_query`).
//! - final RMSNorm, then `PolicyHead` (`is_combat` picks combat or OOC):
//!   `out(SiLU(opt(h_opt) + kind(e_kind) + state(h_STATE)))`, `h_STATE` = token 0.
//!
//! Linear weights are packed into 8-column panels at load, and every matmul
//! keeps a 4-token x 8-output tile in registers while it streams a panel:
//! plain loops LLVM vectorises (SSE/AVX natively, simd128 on wasm).

// Linear-algebra code: `x`, `w`, `y`, `t`, `n`, `d` read better than long names.
#![allow(clippy::many_single_char_names)]

use std::fmt;

use serde::Deserialize;

/// File magic of an `.e2ew` weight file.
pub const MAGIC: &[u8; 4] = b"E2EW";
/// The only `.e2ew` version this loader reads.
pub const VERSION: u32 = 1;

// ── f16 ───────────────────────────────────────────────────────────────────

/// Exact IEEE binary16 -> binary32 widening of the bit pattern `h`.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal: man * 2^-24, exact in f32.
            let v = man as f32 * (1.0 / 16_777_216.0);
            return if sign == 0 { v } else { -v };
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// IEEE binary32 -> binary16 bit pattern, round to nearest even (numpy's
/// `astype(np.float16)`); for producers of [`PolicyRow`]'s f16 fields.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        // Inf / NaN (a NaN stays a quiet NaN).
        return sign
            | 0x7c00
            | if man == 0 {
                0
            } else {
                0x200 | (man >> 13) as u16
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal or zero in f16.
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1; // may carry into the exponent (up to inf): still correct
    }
    sign | r as u16
}

// ── header ────────────────────────────────────────────────────────────────

/// `net` of the header: `ppo.e2e.net.NetConfig` (the sizes the forward reads).
#[derive(Debug, Clone, Deserialize)]
pub struct NetDims {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn: usize,
    pub num_hidden: usize,
    pub head_hidden: usize,
}

/// `layout` of the header: the parts of `ppo.e2e.net.Layout` the forward reads.
#[derive(Debug, Clone, Deserialize)]
pub struct LayoutHeader {
    pub n_cat: usize,
    pub n_cont: usize,
    pub n_prior: usize,
    pub vocab_rows: usize,
    pub n_section: usize,
    pub n_kind: usize,
    pub cat_absent: Vec<i64>,
    pub id_cols: Vec<usize>,
    pub main_col: usize,
    pub small_cols: Vec<usize>,
    pub small_sizes: Vec<usize>,
    pub small_absent: Vec<i64>,
    pub small_offsets: Vec<usize>,
    pub tf_hidden: u8,
    pub rmsnorm_eps: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct TensorEntry {
    name: String,
    shape: Vec<usize>,
    offset: usize,
}

/// The `.e2ew` JSON header (stamps, net config, layout).
#[derive(Debug, Clone, Deserialize)]
pub struct E2ewHeader {
    pub format: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub source_sha256: Option<String>,
    pub net: NetDims,
    #[serde(default)]
    pub schema_sha256: Option<String>,
    #[serde(default)]
    pub vocab_sha256: Option<String>,
    #[serde(default)]
    pub prior_sha256: Option<String>,
    #[serde(default)]
    pub game_version: Option<String>,
    #[serde(default)]
    pub variant: Option<String>,
    #[serde(default)]
    pub filled: Option<Vec<String>>,
    pub layout: LayoutHeader,
    tensors: Vec<TensorEntry>,
    data_bytes: usize,
}

// ── the input row ─────────────────────────────────────────────────────────

/// The policy fields of one decision row (a `ppo.e2e.contracts` row batch
/// with one row), exactly as the tokenizer writes them. `T = sec.len()`
/// tokens; `cat` and `cont` are row-major `[T, n_cat]` / `[T, n_cont]`; `A =
/// cls_tok.len()` merged classes; `P = pool_tok.len()` pool entries.
///
/// The f16 fields (`cont`, `pool_w`) hold IEEE binary16 bit patterns, as the
/// row batch does (`np.float16`); [`f32_to_f16`] produces them from f32.
/// Token indices (`cls_tok`, `pool_tok`) count ALL tokens, hidden ones
/// included: the forward drops `TF_HIDDEN` tokens and remaps them itself.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicyRow {
    /// `[T, n_cat]` categorical columns (int16).
    pub cat: Vec<i16>,
    /// `[T, n_cont]` continuous columns, f16 bits.
    pub cont: Vec<u16>,
    /// `[T]` section of each token.
    pub sec: Vec<u8>,
    /// `[T]` token flags (`TF_HIDDEN` tokens are dropped).
    pub tok_flags: Vec<u8>,
    /// `[A]` token index of each merged class's representative.
    pub cls_tok: Vec<u16>,
    /// `[A]` action kind of each class.
    pub cls_kind: Vec<u8>,
    /// `[P]` token index of each pool entry's pooled token.
    pub pool_tok: Vec<u16>,
    /// `[P]` vocabulary id of each pool entry (`1..vocab_rows`).
    pub pool_vid: Vec<u16>,
    /// `[P]` weight of each pool entry, f16 bits.
    pub pool_w: Vec<u16>,
    /// Row scalar `is_combat`: picks the combat or the OOC policy head.
    pub is_combat: bool,
}

// ── weights ───────────────────────────────────────────────────────────────

/// A `Linear`, its weight packed as `TILE_C`-column panels: panel `p` holds
/// `W[p*C..(p+1)*C, :]` transposed, `[in, C]` contiguous, so a matmul streams
/// one panel (a few KB, L1-resident) per output tile.
struct Linear {
    wt: Vec<f32>,
    bias: Option<Vec<f32>>,
    n_in: usize,
    n_out: usize,
}

struct Block {
    n1: Vec<f32>,
    qkv: Linear,
    out: Linear,
    n2: Vec<f32>,
    /// `w1` and `v` as one `[in, 2*ffn]` linear: outputs `0..ffn` are `w1`,
    /// `ffn..2*ffn` are `v`.
    w1v: Linear,
    w2: Linear,
}

struct Head {
    opt: Linear,
    state: Linear,
    kind: Linear,
    out_w: Vec<f32>,
    out_b: f32,
}

/// The e2e student's policy half, ready to run.
pub struct E2eNet {
    header: E2ewHeader,
    d: usize,
    id_gain: Vec<f32>,
    pool_gain: Vec<f32>,
    prior: Vec<f32>,
    ids: Vec<f32>,
    small: Vec<f32>,
    sec: Vec<f32>,
    num_inp: Linear,
    num_out: Linear,
    blocks: Vec<Block>,
    norm: Vec<f32>,
    kind: Vec<f32>,
    head_combat: Head,
    head_ooc: Head,
}

impl fmt::Debug for E2eNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("E2eNet")
            .field("header", &self.header)
            .finish_non_exhaustive()
    }
}

struct Tensors<'a> {
    data: &'a [u8],
    entries: &'a [TensorEntry],
}

impl Tensors<'_> {
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let e = self
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| format!("e2ew: missing tensor {name}"))?;
        if e.shape != shape {
            return Err(format!(
                "e2ew: tensor {name} has shape {:?}, expected {shape:?}",
                e.shape
            ));
        }
        let n: usize = shape.iter().product();
        let end = e
            .offset
            .checked_add(n * 4)
            .ok_or("e2ew: tensor offset overflow")?;
        let bytes = self
            .data
            .get(e.offset..end)
            .ok_or_else(|| format!("e2ew: tensor {name} lies outside the data"))?;
        let v: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        if v.iter().any(|x| !x.is_finite()) {
            return Err(format!("e2ew: tensor {name} has a non-finite value"));
        }
        Ok(v)
    }

    /// `nn.Linear` weight `[out, in]` (+ bias) as a packed [`Linear`].
    fn linear(&self, name: &str, n_in: usize, n_out: usize, bias: bool) -> Result<Linear, String> {
        let w = self.get(&format!("{name}.weight"), &[n_out, n_in])?;
        let bias = if bias {
            Some(self.get(&format!("{name}.bias"), &[n_out])?)
        } else {
            None
        };
        Linear::new(&w, bias, n_in, n_out).map_err(|e| format!("{name}: {e}"))
    }

    fn head(&self, name: &str, d: usize, hh: usize) -> Result<Head, String> {
        Ok(Head {
            opt: self.linear(&format!("{name}.opt"), d, hh, true)?,
            state: self.linear(&format!("{name}.state"), d, hh, false)?,
            kind: self.linear(&format!("{name}.kind"), d, hh, false)?,
            out_w: self.get(&format!("{name}.out.weight"), &[1, hh])?,
            out_b: self.get(&format!("{name}.out.bias"), &[1])?[0],
        })
    }
}

impl E2eNet {
    /// Load an `.e2ew` file: `b"E2EW" | u32 version | u32 header_len | JSON |
    /// f32 LE data`. Checks the magic, the version, the data length, the
    /// layout's internal consistency and every tensor's shape against the
    /// header's net config and layout.
    pub fn from_bytes(bytes: &[u8]) -> Result<E2eNet, String> {
        if bytes.len() < 12 || &bytes[..4] != MAGIC {
            return Err("e2ew: bad magic".into());
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != VERSION {
            return Err(format!(
                "e2ew: version {version}, this loader reads {VERSION}"
            ));
        }
        let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let hb = bytes.get(12..12 + hl).ok_or("e2ew: truncated header")?;
        let header: E2ewHeader =
            serde_json::from_slice(hb).map_err(|e| format!("e2ew: header: {e}"))?;
        if header.format != "e2ew/1" {
            return Err(format!("e2ew: format {:?}", header.format));
        }
        let data = &bytes[12 + hl..];
        if data.len() != header.data_bytes {
            return Err(format!(
                "e2ew: {} data bytes, header says {}",
                data.len(),
                header.data_bytes
            ));
        }
        let (n, l) = (&header.net, &header.layout);
        let d = n.dim;
        if n.n_heads == 0 || d % n.n_heads != 0 {
            return Err(format!(
                "e2ew: dim {d} not divisible by n_heads {}",
                n.n_heads
            ));
        }
        let small_rows = 1 + l.small_sizes.iter().sum::<usize>();
        let ok_layout = l.id_cols.len() == 4
            && l.cat_absent.len() == l.n_cat
            && l.small_sizes.len() == l.small_cols.len()
            && l.small_absent.len() == l.small_cols.len()
            && l.small_offsets.len() == l.small_cols.len()
            && l.id_cols.iter().chain(&l.small_cols).all(|&c| c < l.n_cat)
            && l.main_col < l.n_cat
            && l.small_offsets
                .iter()
                .zip(&l.small_sizes)
                .all(|(&o, &s)| o >= 1 && o + s <= small_rows);
        if !ok_layout {
            return Err("e2ew: inconsistent layout".into());
        }
        let t = Tensors {
            data,
            entries: &header.tensors,
        };
        let stem_in = l.n_cont + l.n_prior;
        let mut blocks = Vec::with_capacity(n.n_layers);
        for i in 0..n.n_layers {
            let p = format!("blocks.{i}");
            let w1 = t.get(&format!("{p}.w1.weight"), &[n.ffn, d])?;
            let v = t.get(&format!("{p}.v.weight"), &[n.ffn, d])?;
            let w1v = [w1, v].concat();
            blocks.push(Block {
                n1: t.get(&format!("{p}.n1.weight"), &[d])?,
                qkv: t.linear(&format!("{p}.qkv"), d, 3 * d, false)?,
                out: t.linear(&format!("{p}.out"), d, d, false)?,
                n2: t.get(&format!("{p}.n2.weight"), &[d])?,
                w1v: Linear::new(&w1v, None, d, 2 * n.ffn)?,
                w2: t.linear(&format!("{p}.w2"), n.ffn, d, false)?,
            });
        }
        let net = E2eNet {
            d,
            id_gain: t.get("id_gain", &[4, d])?,
            pool_gain: t.get("pool_gain", &[d])?,
            prior: t.get("prior", &[l.vocab_rows, l.n_prior])?,
            ids: t.get("ids.weight", &[l.vocab_rows, d])?,
            small: t.get("small.weight", &[small_rows, d])?,
            sec: t.get("sec.weight", &[l.n_section, d])?,
            num_inp: t.linear("num.inp", stem_in, n.num_hidden, true)?,
            num_out: t.linear("num.out", n.num_hidden, d, true)?,
            blocks,
            norm: t.get("norm.weight", &[d])?,
            kind: t.get("kind.weight", &[l.n_kind, d])?,
            head_combat: t.head("head_combat", d, n.head_hidden)?,
            head_ooc: t.head("head_ooc", d, n.head_hidden)?,
            header,
        };
        Ok(net)
    }

    /// The parsed header (stamps: `schema_sha256`, `vocab_sha256`,
    /// `prior_sha256`, `game_version`, `variant`, `filled`; net config; layout).
    pub fn header(&self) -> &E2ewHeader {
        &self.header
    }

    /// One fp32 logit per merged class (`row.cls_tok.len()`), as
    /// `E2ENet.policy(to_batch(row), bf16=False)[0, :n_classes]`. Panics on a
    /// malformed row; see [`E2eNet::try_logits`].
    pub fn logits(&self, row: &PolicyRow) -> Vec<f32> {
        self.try_logits(row).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The greedy class: the first maximum of [`E2eNet::logits`] (torch's argmax).
    pub fn act_greedy(&self, row: &PolicyRow) -> usize {
        argmax(&self.logits(row))
    }

    /// [`E2eNet::logits`], with `to_batch`'s structural checks as errors.
    pub fn try_logits(&self, row: &PolicyRow) -> Result<Vec<f32>, String> {
        let l = &self.header.layout;
        let d = self.d;
        let nt = row.sec.len();
        if nt == 0 {
            return Err("e2e row: no tokens".into());
        }
        if row.cat.len() != nt * l.n_cat
            || row.cont.len() != nt * l.n_cont
            || row.tok_flags.len() != nt
        {
            return Err("e2e row: cat/cont/tok_flags lengths disagree with sec".into());
        }
        let na = row.cls_tok.len();
        if na == 0 || row.cls_kind.len() != na {
            return Err("e2e row: no classes, or cls_tok/cls_kind lengths differ".into());
        }
        let np = row.pool_tok.len();
        if row.pool_vid.len() != np || row.pool_w.len() != np {
            return Err("e2e row: pool_tok/pool_vid/pool_w lengths differ".into());
        }

        // Visible tokens and their positions (`_index`'s newpos).
        let mut newpos = vec![usize::MAX; nt];
        let mut vis = Vec::with_capacity(nt);
        for (g, &f) in row.tok_flags.iter().enumerate() {
            if f & l.tf_hidden == 0 {
                newpos[g] = vis.len();
                vis.push(g);
            }
        }
        if newpos[0] != 0 {
            return Err("e2e row: token 0 (RUNCTX) is hidden".into());
        }
        let n = vis.len();
        let mut cls_pos = Vec::with_capacity(na);
        for (&ct, &ck) in row.cls_tok.iter().zip(&row.cls_kind) {
            let p = *newpos
                .get(usize::from(ct))
                .ok_or("e2e row: class token outside the row")?;
            if p == usize::MAX {
                return Err("e2e row: a class representative is hidden".into());
            }
            if usize::from(ck) >= l.n_kind {
                return Err("e2e row: cls_kind outside its table".into());
            }
            cls_pos.push(p);
        }

        // ── embed ──
        let n_prior = l.n_prior;
        let stem_in = l.n_cont + n_prior;
        let mut x = vec![0.0f32; n * d];
        let mut stem = vec![0.0f32; n * stem_in];
        let mut bag = vec![0.0f32; d];
        for (p, &g) in vis.iter().enumerate() {
            let cat = &row.cat[g * l.n_cat..(g + 1) * l.n_cat];
            let xr = &mut x[p * d..(p + 1) * d];
            let s = usize::from(row.sec[g]);
            if s >= l.n_section {
                return Err("e2e row: section outside its table".into());
            }
            xr.copy_from_slice(&self.sec[s * d..(s + 1) * d]);
            for (k, &col) in l.id_cols.iter().enumerate() {
                let id = id_index(cat[col], l.vocab_rows)?;
                let e = &self.ids[id * d..(id + 1) * d];
                let gn = &self.id_gain[k * d..(k + 1) * d];
                for ((o, &a), &b) in xr.iter_mut().zip(e).zip(gn) {
                    *o += a * b;
                }
            }
            bag.fill(0.0);
            for (((&col, &size), &absent), &off) in l
                .small_cols
                .iter()
                .zip(&l.small_sizes)
                .zip(&l.small_absent)
                .zip(&l.small_offsets)
            {
                let v = i64::from(cat[col]);
                if v < 0 || v as usize >= size {
                    return Err(format!("e2e row: column {col} value {v} outside 0..{size}"));
                }
                if v == absent {
                    continue;
                }
                let r = v as usize + off;
                axpy(&mut bag, 1.0, &self.small[r * d..(r + 1) * d]);
            }
            axpy(xr, 1.0, &bag);
            let sr = &mut stem[p * stem_in..(p + 1) * stem_in];
            let cont = &row.cont[g * l.n_cont..(g + 1) * l.n_cont];
            for (o, &h) in sr[..l.n_cont].iter_mut().zip(cont) {
                *o = f16_to_f32(h);
            }
            let m = id_index(cat[l.main_col], l.vocab_rows)?;
            sr[l.n_cont..].copy_from_slice(&self.prior[m * n_prior..(m + 1) * n_prior]);
        }

        // Pools: per pooled token, the weighted mean of its entries.
        let mut done = vec![false; n];
        let mut pe = vec![0.0f32; d];
        let mut pp = vec![0.0f32; n_prior];
        for i in 0..np {
            let pt = row.pool_tok[i];
            let p = *newpos
                .get(usize::from(pt))
                .ok_or("e2e row: pool token outside the row")?;
            if p == usize::MAX {
                return Err("e2e row: a pool entry's token is hidden".into());
            }
            if done[p] {
                continue;
            }
            done[p] = true;
            let mut w_sum = 0.0f32;
            pe.fill(0.0);
            pp.fill(0.0);
            for j in i..np {
                if row.pool_tok[j] != pt {
                    continue;
                }
                let vid = usize::from(row.pool_vid[j]);
                if vid == 0 || vid >= l.vocab_rows {
                    return Err("e2e row: pool_vid outside 1..vocab_rows".into());
                }
                let w = f16_to_f32(row.pool_w[j]);
                w_sum += w;
                axpy(&mut pe, w, &self.ids[vid * d..(vid + 1) * d]);
                axpy(&mut pp, w, &self.prior[vid * n_prior..(vid + 1) * n_prior]);
            }
            let w = w_sum.max(1.0);
            let xr = &mut x[p * d..(p + 1) * d];
            for ((o, &a), &g) in xr.iter_mut().zip(&pe).zip(&self.pool_gain) {
                *o += a / w * g;
            }
            let sr = &mut stem[p * stem_in + l.n_cont..(p + 1) * stem_in];
            for (o, &a) in sr.iter_mut().zip(&pp) {
                *o += a / w;
            }
        }

        // Numeric stem: symlog -> Linear -> SiLU -> Linear.
        for v in &mut stem {
            *v = if *v == 0.0 {
                0.0
            } else {
                v.signum() * v.abs().ln_1p()
            };
        }
        let mut hid = vec![0.0f32; n * self.num_inp.n_out];
        self.num_inp.forward_sparse(&stem, n, &mut hid);
        for v in &mut hid {
            *v = silu(*v);
        }
        let mut add = vec![0.0f32; n * d];
        self.num_out.forward(&hid, n, &mut add);
        axpy(&mut x, 1.0, &add);

        // ── trunk ──
        let all: Vec<usize> = (0..n).collect();
        let mut want: Vec<usize> = std::iter::once(0).chain(cls_pos.iter().copied()).collect();
        want.sort_unstable();
        want.dedup();
        let mut ws = Work::new(n, d, self.header.net.ffn);
        let nl = self.blocks.len();
        for (i, blk) in self.blocks.iter().enumerate() {
            let q = if i + 1 == nl { &want } else { &all };
            self.block(blk, &mut x, n, q, &mut ws);
        }

        // ── heads ──
        let eps = l.rmsnorm_eps;
        let mut hn = vec![0.0f32; n * d];
        for &p in &want {
            rmsnorm(
                &x[p * d..(p + 1) * d],
                &self.norm,
                eps,
                &mut hn[p * d..(p + 1) * d],
            );
        }
        let head = if row.is_combat {
            &self.head_combat
        } else {
            &self.head_ooc
        };
        let hh = head.opt.n_out;
        let mut s = vec![0.0f32; hh];
        head.state.forward(&hn[..d], 1, &mut s);
        let mut z = vec![0.0f32; hh];
        let mut zk = vec![0.0f32; hh];
        let mut out = Vec::with_capacity(na);
        for (&p, &ck) in cls_pos.iter().zip(&row.cls_kind) {
            let k = usize::from(ck);
            head.opt.forward(&hn[p * d..(p + 1) * d], 1, &mut z);
            head.kind
                .forward(&self.kind[k * d..(k + 1) * d], 1, &mut zk);
            for ((a, &b), &c) in z.iter_mut().zip(&zk).zip(&s) {
                *a = silu(*a + b + c);
            }
            out.push(dot(&z, &head.out_w) + head.out_b);
        }
        Ok(out)
    }

    /// `Block.forward` (`q` = every token) or `Block.forward_query` (`q` a
    /// subset): updates rows `q` of `x` `[n, d]`; other rows go stale.
    fn block(&self, b: &Block, x: &mut [f32], n: usize, q: &[usize], ws: &mut Work) {
        let d = self.d;
        let eps = self.header.layout.rmsnorm_eps;
        let heads = self.header.net.n_heads;
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let m = q.len();
        for t in 0..n {
            rmsnorm(
                &x[t * d..(t + 1) * d],
                &b.n1,
                eps,
                &mut ws.h[t * d..(t + 1) * d],
            );
        }
        b.qkv.forward(&ws.h[..n * d], n, &mut ws.qkv[..n * 3 * d]);
        let qkv = &ws.qkv;
        let att = &mut ws.att[..m * d];
        att.fill(0.0);
        let sc = &mut ws.scores[..n];
        for (r, &qi) in q.iter().enumerate() {
            for hix in 0..heads {
                let qo = qi * 3 * d + hix * hd;
                let qv = &qkv[qo..qo + hd];
                let mut mx = f32::NEG_INFINITY;
                for (j, s) in sc.iter_mut().enumerate() {
                    let ko = j * 3 * d + d + hix * hd;
                    *s = dot(qv, &qkv[ko..ko + hd]) * scale;
                    mx = mx.max(*s);
                }
                let mut sum = 0.0f32;
                for s in sc.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                let o = &mut att[r * d + hix * hd..r * d + (hix + 1) * hd];
                for (j, &s) in sc.iter().enumerate() {
                    let vo = j * 3 * d + 2 * d + hix * hd;
                    axpy(o, s * inv, &qkv[vo..vo + hd]);
                }
            }
        }
        // x[q] += out(att); h = n2(x[q]); x[q] += w2(silu(w1 h) * v h).
        b.out.forward(att, m, &mut ws.proj[..m * d]);
        for (r, &qi) in q.iter().enumerate() {
            let xr = &mut x[qi * d..(qi + 1) * d];
            axpy(xr, 1.0, &ws.proj[r * d..(r + 1) * d]);
            rmsnorm(xr, &b.n2, eps, &mut ws.h[r * d..(r + 1) * d]);
        }
        let f = b.w2.n_in;
        b.w1v.forward(&ws.h[..m * d], m, &mut ws.ffn[..m * 2 * f]);
        for r in 0..m {
            let (a, v) = ws.ffn[r * 2 * f..(r + 1) * 2 * f].split_at(f);
            for ((g, &a), &v) in ws.gate[r * f..(r + 1) * f].iter_mut().zip(a).zip(v) {
                *g = silu(a) * v;
            }
        }
        b.w2.forward(&ws.gate[..m * f], m, &mut ws.proj[..m * d]);
        for (r, &qi) in q.iter().enumerate() {
            axpy(
                &mut x[qi * d..(qi + 1) * d],
                1.0,
                &ws.proj[r * d..(r + 1) * d],
            );
        }
    }
}

/// Per-forward scratch buffers, reused across layers.
struct Work {
    h: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    proj: Vec<f32>,
    ffn: Vec<f32>,
    gate: Vec<f32>,
    scores: Vec<f32>,
}

impl Work {
    fn new(n: usize, d: usize, ffn: usize) -> Work {
        Work {
            h: vec![0.0; n * d],
            qkv: vec![0.0; n * 3 * d],
            att: vec![0.0; n * d],
            proj: vec![0.0; n * d],
            ffn: vec![0.0; n * 2 * ffn],
            gate: vec![0.0; n * ffn],
            scores: vec![0.0; n],
        }
    }
}

fn id_index(v: i16, rows: usize) -> Result<usize, String> {
    match usize::try_from(v) {
        Ok(i) if i < rows => Ok(i),
        _ => Err(format!("e2e row: id {v} outside 0..{rows}")),
    }
}

/// The index of the first maximum (torch's `argmax`); 0 for an empty slice.
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (o, &v) in y.iter_mut().zip(x) {
        *o += a * v;
    }
}

/// Dot product with 16 independent accumulators (vectorisable).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const L: usize = 16;
    let mut acc = [0.0f32; L];
    let (ca, ra) = a.as_chunks::<L>();
    let (cb, rb) = b.as_chunks::<L>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..L {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `nn.RMSNorm`: `x * rsqrt(mean(x^2) + eps) * w`.
fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = dot(x, x) / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for ((o, &a), &g) in out.iter_mut().zip(x).zip(w) {
        *o = a * r * g;
    }
}

/// Output columns per register tile of [`Linear::forward`].
const TILE_C: usize = 8;
/// Tokens per register tile of [`Linear::forward`].
const TILE_R: usize = 4;

impl Linear {
    /// From an `nn.Linear` weight `w` `[n_out, n_in]` (row-major) and bias.
    fn new(w: &[f32], bias: Option<Vec<f32>>, n_in: usize, n_out: usize) -> Result<Linear, String> {
        if !n_out.is_multiple_of(TILE_C) || w.len() != n_in * n_out {
            return Err(format!(
                "linear {n_out}x{n_in}: out not a multiple of {TILE_C}"
            ));
        }
        let mut wt = vec![0.0; w.len()];
        for o in 0..n_out {
            let (p, k) = (o / TILE_C, o % TILE_C);
            for i in 0..n_in {
                wt[p * n_in * TILE_C + i * TILE_C + k] = w[o * n_in + i];
            }
        }
        Ok(Linear {
            wt,
            bias,
            n_in,
            n_out,
        })
    }

    fn init(&self, n: usize, y: &mut [f32]) {
        for yr in y[..n * self.n_out].chunks_exact_mut(self.n_out) {
            match &self.bias {
                Some(b) => yr.copy_from_slice(b),
                None => yr.fill(0.0),
            }
        }
    }

    /// `y[n, out] = x[n, in] @ W^T (+ b)`, register-tiled: per weight panel,
    /// a `TILE_R` x `TILE_C` block of outputs accumulates over all inputs in
    /// registers, each step one `TILE_C` panel row and `TILE_R` inputs.
    fn forward(&self, x: &[f32], n: usize, y: &mut [f32]) {
        let (ni, no) = (self.n_in, self.n_out);
        debug_assert!(x.len() >= n * ni && y.len() >= n * no);
        self.init(n, y);
        for (p, panel) in self.wt.chunks_exact(ni * TILE_C).enumerate() {
            let j0 = p * TILE_C;
            let mut t = 0;
            while t + TILE_R <= n {
                let xs: [&[f32]; TILE_R] =
                    std::array::from_fn(|r| &x[(t + r) * ni..(t + r + 1) * ni]);
                let mut acc = [[0.0f32; TILE_C]; TILE_R];
                for (i, w) in panel.as_chunks::<TILE_C>().0.iter().enumerate() {
                    for r in 0..TILE_R {
                        let a = xs[r][i];
                        for k in 0..TILE_C {
                            acc[r][k] += a * w[k];
                        }
                    }
                }
                for (r, ar) in acc.iter().enumerate() {
                    let yr = &mut y[(t + r) * no + j0..(t + r) * no + j0 + TILE_C];
                    for (o, &a) in yr.iter_mut().zip(ar) {
                        *o += a;
                    }
                }
                t += TILE_R;
            }
            while t < n {
                let xr = &x[t * ni..(t + 1) * ni];
                let mut acc = [0.0f32; TILE_C];
                for (&a, w) in xr.iter().zip(panel.as_chunks::<TILE_C>().0) {
                    for k in 0..TILE_C {
                        acc[k] += a * w[k];
                    }
                }
                let yr = &mut y[t * no + j0..t * no + j0 + TILE_C];
                for (o, &a) in yr.iter_mut().zip(&acc) {
                    *o += a;
                }
                t += 1;
            }
        }
    }

    /// [`Linear::forward`] for mostly-zero inputs (the numeric stem's): only
    /// the non-zero inputs are visited. Skipping a zero input is exact (the
    /// weights are finite).
    fn forward_sparse(&self, x: &[f32], n: usize, y: &mut [f32]) {
        let (ni, no) = (self.n_in, self.n_out);
        self.init(n, y);
        let mut nz = Vec::with_capacity(ni);
        for t in 0..n {
            let xr = &x[t * ni..(t + 1) * ni];
            nz.clear();
            nz.extend(
                xr.iter()
                    .enumerate()
                    .filter(|(_, &a)| a != 0.0)
                    .map(|(i, &a)| (i, a)),
            );
            let yr = &mut y[t * no..(t + 1) * no];
            for (panel, yp) in self
                .wt
                .chunks_exact(ni * TILE_C)
                .zip(yr.as_chunks_mut::<TILE_C>().0)
            {
                let mut acc = [0.0f32; TILE_C];
                for &(i, a) in &nz {
                    let w = &panel[i * TILE_C..(i + 1) * TILE_C];
                    for k in 0..TILE_C {
                        acc[k] += a * w[k];
                    }
                }
                for (o, &a) in yp.iter_mut().zip(&acc) {
                    *o += a;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_every_pattern() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "pattern {h:#06x} -> {f}");
        }
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(17664), 5.0);
        // Round to nearest even: 1 + 2^-11 ties between 1 and 1 + 2^-10.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(2f32.powi(-26)), 0);
        assert_eq!(f32_to_f16(3.0 * 2f32.powi(-26)), 1);
    }

    #[test]
    fn from_bytes_rejects_bad_files() {
        let hdr = br#"{"format":"e2ew/1"}"#;
        let file = |magic: &[u8], version: u32| {
            let mut b = magic.to_vec();
            b.extend(version.to_le_bytes());
            b.extend((hdr.len() as u32).to_le_bytes());
            b.extend(hdr);
            b
        };
        assert!(E2eNet::from_bytes(b"E2E").unwrap_err().contains("magic"));
        assert!(E2eNet::from_bytes(&file(b"E2EX", 1))
            .unwrap_err()
            .contains("magic"));
        assert!(E2eNet::from_bytes(&file(b"E2EW", 2))
            .unwrap_err()
            .contains("version"));
        assert!(E2eNet::from_bytes(&file(b"E2EW", 1))
            .unwrap_err()
            .contains("header"));
    }

    #[test]
    fn argmax_first_max() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[f32::NEG_INFINITY]), 0);
    }

    #[test]
    fn linear_matches_naive() {
        for (ni, no) in [(5, 8), (5, 32), (19, 48)] {
            let w: Vec<f32> = (0..ni * no).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..no).map(|i| i as f32 * 0.1).collect();
            let lin = Linear::new(&w, Some(b.clone()), ni, no).unwrap();
            for n in [1, 4, 6, 9] {
                let x: Vec<f32> = (0..n * ni)
                    .map(|i| {
                        if i % 3 == 0 {
                            0.0
                        } else {
                            (i as f32 * 0.71).cos()
                        }
                    })
                    .collect();
                for sparse in [false, true] {
                    let mut y = vec![f32::NAN; n * no];
                    if sparse {
                        lin.forward_sparse(&x, n, &mut y);
                    } else {
                        lin.forward(&x, n, &mut y);
                    }
                    for t in 0..n {
                        for o in 0..no {
                            let want: f32 =
                                b[o] + (0..ni).map(|i| w[o * ni + i] * x[t * ni + i]).sum::<f32>();
                            assert!((y[t * no + o] - want).abs() < 1e-5, "{ni}x{no} n={n}");
                        }
                    }
                }
            }
        }
    }
}
