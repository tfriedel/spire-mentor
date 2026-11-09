//! Offline check: does `generate_act_map` with one room fewer reproduce the
//! game's co-op act map? (`StandardActMap`: `_mapLength =
//! actModel.GetNumberOfRooms(isMultiplayer) + 1`, and multiplayer is the only
//! input that changes; same `act_N_map` stream, counts and pruning.)
//!
//! stdin: one `SEED ACT NUM_ROOMS ASCENSION` per line (v0.111.0 seed string).
//! stdout: per line, the map's nodes as `col,row,TYPE,child;child|...`, the
//! boss last.
//!
//! `cargo run -p sts2-core --example coop_map_check < queries.txt`

use std::io::BufRead;

use sts2_core::map_generator::generate_act_map;
use sts2_core::run_rng_set::RunRngSet;
use sts2_core::{canonicalize_seed, GameVersion};

fn main() {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 4 {
            println!("ERR bad line");
            continue;
        }
        let seed = parts[0];
        let act: i32 = parts[1].parse().expect("act");
        let num_rooms: i32 = parts[2].parse().expect("num_rooms");
        let ascension: i32 = parts[3].parse().expect("ascension");
        let version = GameVersion::V0_111_0;
        let master = version.run_seed_hash(&canonicalize_seed(seed));
        let set = RunRngSet::new_for(version, master);
        let mut rng = set.named(&format!("act_{act}_map"));
        let map = generate_act_map(num_rooms, &mut rng, ascension, false, act, None);
        let mut out: Vec<String> = Vec::new();
        for (coord, &idx) in &map.grid {
            let p = map.point(idx);
            let kids: Vec<String> = p
                .children
                .iter()
                .map(|&c| {
                    let cp = map.point(c);
                    format!("{}:{}", cp.coord.col, cp.coord.row)
                })
                .collect();
            out.push(format!(
                "{},{},{:?},{}",
                coord.col,
                coord.row,
                p.point_type,
                kids.join(";")
            ));
        }
        if let Some(b) = map.boss {
            let bp = map.point(b);
            out.push(format!("{},{},Boss,", bp.coord.col, bp.coord.row));
        }
        println!("{}", out.join("|"));
    }
}
