//! Reader for `ppo.e2e.export_wb fixtures` files (`<seed>.jsonl.gz`), shared
//! by `tests/net_parity.rs` and `examples/net_bench.rs`.

#![allow(dead_code)]

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::Value;
use sts2_e2e::net::PolicyRow;

/// One decision of a fixture: the policy row and what torch computed.
#[derive(Debug)]
pub struct Decision {
    pub i: usize,
    pub row: PolicyRow,
    pub logits: Vec<f32>,
    pub chosen: usize,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}

/// `E2E_WEIGHTS`, default `~/runs/e2e-wb/export/ironclad_e2e_d2500_avg5_ema0999.e2ew`.
pub fn weights_path() -> PathBuf {
    std::env::var_os("E2E_WEIGHTS").map_or_else(
        || home().join("runs/e2e-wb/export/ironclad_e2e_d2500_avg5_ema0999.e2ew"),
        PathBuf::from,
    )
}

/// `E2E_FIXTURES`, default `~/runs/e2e-wb/fixtures`.
pub fn fixtures_dir() -> PathBuf {
    std::env::var_os("E2E_FIXTURES")
        .map_or_else(|| home().join("runs/e2e-wb/fixtures"), PathBuf::from)
}

/// Every `*.jsonl.gz` under `dir`, sorted.
pub fn fixture_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".jsonl.gz"))
        .collect();
    v.sort();
    v
}

fn data(rowb: &Value, key: &str) -> Vec<i64> {
    rowb[key]["data"]
        .as_array()
        .unwrap_or_else(|| panic!("rowb.{key}.data"))
        .iter()
        .map(|x| {
            x.as_i64()
                .unwrap_or_else(|| panic!("rowb.{key}: non-integer"))
        })
        .collect()
}

fn conv<T: TryFrom<i64>>(v: Vec<i64>, key: &str) -> Vec<T> {
    v.into_iter()
        .map(|x| T::try_from(x).unwrap_or_else(|_| panic!("rowb.{key}: {x} out of range")))
        .collect()
}

/// The decisions of one fixture file (header and terminal lines skipped).
/// A truncated gzip (a dump still being written) yields the decisions read
/// so far.
pub fn read_fixture(path: &Path) -> Vec<Decision> {
    let f = std::fs::File::open(path).expect("open fixture");
    let r = BufReader::new(flate2::read::GzDecoder::new(f));
    let mut out = Vec::new();
    for line in r.lines() {
        let Ok(line) = line else { break };
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            break;
        };
        let Some(rowb) = v.get("rowb") else { continue };
        let row = PolicyRow {
            cat: conv(data(rowb, "cat"), "cat"),
            cont: conv(data(rowb, "cont"), "cont"),
            sec: conv(data(rowb, "sec"), "sec"),
            tok_flags: conv(data(rowb, "tok_flags"), "tok_flags"),
            cls_tok: conv(data(rowb, "cls_tok"), "cls_tok"),
            cls_kind: conv(data(rowb, "cls_kind"), "cls_kind"),
            pool_tok: conv(data(rowb, "pool_tok"), "pool_tok"),
            pool_vid: conv(data(rowb, "pool_vid"), "pool_vid"),
            pool_w: conv(data(rowb, "pool_w"), "pool_w"),
            is_combat: v["row_scalars"]["is_combat"].as_i64().expect("is_combat") != 0,
        };
        let logits = v["logits"]
            .as_array()
            .expect("logits")
            .iter()
            .map(|x| x.as_f64().expect("logit") as f32)
            .collect();
        out.push(Decision {
            i: v["i"].as_u64().expect("i") as usize,
            row,
            logits,
            chosen: v["chosen"].as_u64().expect("chosen") as usize,
        });
    }
    out
}
