# Spire Bench paired measure: real game vs full view, and net-input parity

**Date:** 2026-11-21 · **PR:** #1720 (the parity fixes) · **Data:** `2026-11-21-spire-bench-paired/`

## Question

Does an agent lose win rate when it plays the real game through the Spire Bench referee? On the referee it sees only the player view and rebuilds every fight from that view. The comparison arm plays the same dice with the full game state.

## Setup

- **Pairing.** Referee episode *n* and its full-view twin share one episode key, so they play the same dice (`run.sh`).
  - The referee arm runs through a local gateway (`bench.json`, keyed RNG, `max_workers` 16).
  - The local arm runs on a keyed host with the full view (`search_agent.py --local --paired-season-key`).
- **Agent.** Instinct: the e2e net, no search, Ironclad A10, game build v0.111.0.
- **Analysis.** `analyze.py` gives win rates with Wilson intervals and the paired difference. The CI is a normal approximation over pairs; the test is an exact McNemar test on the discordant pairs.

## Results

| Run | Net | Pairs | Referee | Full view | Referee − full view (paired 95% CI) | McNemar p |
|---|---|---|---|---|---|---|
| `instinct` (before the parity work) | d500 | 150 | 64.7% | 70.0% | −5.3 pp (±7.1) | 0.20 |
| `instinct_v2` | d500 | 150 | 64.7% | 68.7% | −4.0 pp (±6.9) | 0.35 |
| **`instinct_v3` (after #1720)** | **d2250** | **300** | **72.0%** | **73.3%** | **−1.3 pp (±3.8)** | **0.61** |

`instinct_v3` details:

- 34 discordant pairs: 15 won only on the referee, 19 only on the full view.
- 255 of 300 pairs end with the same result on the same floor.
- No errors.
- 56 fallbacks in each arm. A fallback is a decision Instinct handed to the advisor's rules.

Caveats:

- **The net changed between v2 and v3.** Main moved the e2e net from d500 to `allchar_e2e_d2250_ema0999` before v3 ran. Both v3 arms use the new net, so the paired difference holds, but v3's absolute rates are not comparable with v1/v2's. The v2 → v3 change in the gap therefore mixes the parity fixes with the net change.
- **The v3 referee arm ran in two passes.** The first launch used 20 processes against 16 gateway slots, and 60 referee episodes were refused at start (HTTP 429, 0 decisions). Their rows were dropped and episodes 240–299 were rerun with 4 processes. Refusal happened before any decision, so the rerun introduces no outcome bias.
- **Search arm.** The first search-agent pair (`search`, MCTS 1000/4000, 150 pairs) read referee 8.0% vs full view 14.0%, −6.0 pp (±5.3, p 0.049). `search_v2` holds identical rows and is not an independent run. Re-run after corpus batches 7–10 below (`search_v4`).

**Reading.** On 300 pairs the referee costs Instinct −1.3 pp, and the 95% CI is ±3.8 pp. The measure cannot tell this from zero. It rules out a loss larger than about 5 pp.

## Search agent after corpus batches 7–10 (`search_v4`, 2026-11-26)

Same pairing, search agent (MCTS 1000/4000 sims, `beliefdraw`), Ironclad A10, v0.111.0, 300 pairs (12 processes × 25; `all_v4.sh`, `all_v4_referee.sh`), main at `7ed0663b3` (after #1932). The first launch's referee arm met a stale gateway on port 8652 (HTTP 500 on all 300, 0 decisions); its rows were dropped and the referee arm was re-run alone on a fresh gateway. The local arm is the first launch's.

| Run | Pairs | Referee | Full view | Referee − full view (paired 95% CI) | McNemar p |
|---|---|---|---|---|---|
| `search` (before #1720) | 150 | 8.0% | 14.0% | −6.0 pp (±5.3) | 0.049 |
| `search_v4` | 300 | 7.7% (5.2–11.2) | 12.0% (8.8–16.2) | −4.3 pp (±4.4) | 0.072 |
| **`search_v5` (after #1935)** | **300** | **11.0%** (7.9–15.0) | **12.0%** (8.8–16.2) | **−1.0 pp (±4.7)** | **0.78** |

- 45 discordant pairs: 16 won only on the referee, 29 only on the full view. No errors, no fallbacks.
- The combat mirror desyncs on 23.9% of reconciles on the referee (26,760 / 111,931) and 5.1% on the full view (5,592 / 109,790).
- Most of that is the hidden draw order, by design: 21,657 of the referee's desyncs touch only the piles (the mirror sampled another draw and reseeds from the observation); 2,319 of the full view's do.
- Outside the piles the referee has 1.5× the full view's desyncs (5,103 vs 3,298). The largest excesses: enemy intents 2,841 vs 1,652, player powers 960 vs 280, player block 581 vs 103, enemy HP 2,693 vs 1,878. Split by time from `logs/diag/mirror_divergences.jsonl` (both arms share mirror labels).

**`search_v5`, after #1935** (main `ee2039134`, `all_v5.sh`, one launch). #1935 builds the move tracker's catalog for the fight's own build. Before it, on v0.111.0 the tracker named moves the fight's machine lacks: the rebuilt enemy was AI-less and telegraphed the engine's stand-in `Attack 5` (`MonsterKind::roll_intent`), and search planned against it. The full-view arm is deterministic and replays `search_v4` exactly (36/300, 5,592 desyncs).

- 51 discordant pairs: 24 won only on the referee, 27 only on the full view. No errors, no fallbacks.
- Stand-in `Attack 5` predictions in the mirror log: 384 → 180 (both arms). TorchHeadAmalgam and the Queen are gone from the list. Test Subject (94), Parafright (23), BattleFriend V1/V2 (31) and EyeWithTeeth (11) remain (#1933).
- The overall desync rate is unchanged at 23.9%, because the hidden-draw desyncs dominate it.

**Reading.** The search agent's referee cost went −6.0 → −4.3 → −1.0 pp. After #1935 it cannot be told from zero on 300 pairs (CI ±4.7 pp). Instinct's, on the same pairing, is −1.3 pp (±3.8). Most of the −4.3 pp came from the AI-less rebuilds #1935 removed; the stand-ins left (#1933) are the next candidate.

## Real game vs native (same seeds)

This is the measure that answers "does the real game cost win rate". Instinct plays the 300 game seed strings 170000000–170000299 twice, Ironclad A10, no search:

- **native:** in the Rust engine (`seed_match/run_*.sh`, `play_trace`);
- **real game:** through the real DLL, full view, the game's own RNG (`real_trace.py`).

| Net, code | Native | Real game | Native-only / real-only wins | McNemar p |
|---|---|---|---|---|
| d500, before #1720 (`native_d500.summary.jsonl`, `real_d500_fix4.summary.jsonl`) | 75.0% | 69.0% | 47 / 29 | 0.05 |
| **d2500, main after #1721 (`native_d2500.summary.jsonl`, `real_d2500.summary.jsonl`)** | **79.7%** | **78.7%** | **17 / 14** | **0.72** |

The real game now costs −1.0 pp against native on these seeds, down from −6.0 pp. That is within noise: 31 discordant seeds. The real run had no errors.

The two rows differ in both the code (#1720) and the net, so this table does not split the improvement between them.

About d2500:

- Exported from the all-character lane's 2.5B milestone (`pinned_d0002500107852_milestone.pt`, EMA 0.999) to `sts2-pycli/data/e2e/allchar_e2e_d2500_ema0999.e2ew`, sha256 `c410b8d8…`.
- The lane's own evaluation: 812/1000 = 81.2% on Ironclad A10 (DEV-1000).
- It is not committed. For the run, `realgame_instinct.WEIGHTS_RELPATH` was pointed at it locally.

## Net-input parity (seed_match)

Here "native" means Instinct playing the run in the Rust engine. The seed match runs it and the real game (headless host, referee view, seedless rebuilds) on 47 matched seeds and finds the first decision at which the net's input tokens differ (`seed_match/first_input_diff.py`). Native rows come from `rows2`; each `rowsN/diff.txt` is one re-dump.

| Dump | After | Mean first-diff floor (cap 50) | Seeds identical for the whole run |
|---|---|---|---|
| rows16 | #1705 | 24.2 | — |
| rows18 | #1712 | 27.7 | 4 |
| rows19 | #1715 | 28.6 | 7 |

No seed got worse from one dump to the next.

The fixes are in #1720, and each has its own issue. The view gained public values the UI does not print, under the rule in `spire-bench/PROTOCOL.md`:

- an attack's base damage (#1708);
- draw-pile cards with a known position (#1710);
- the combat-history counts (#1712);
- an enchant screen's amount (#1715).

Rebuild fixes: #1700, #1701, #1702, #1707, #1711, #1713. Driver fixes: #1698, #1704, #1705, #1715.

Open first-diff buckets in rows19:

- Gold rolls at event options, an accepted residual (4 seeds).
- Draw-pile order when a multiset is equal (5).
- Forecast intents (4).
- Removed enemy corpses (#1716).
- A Lord's Parasol shop screen (seed 168).
- A card select at the act-2 boss (#1706, seed 012).

The live mod does not send `base_damage` yet (#1709).

## Files

- `run.sh`, `all*.sh`, `analyze.py`, `bench.json`: the paired runs.
- `instinct*/`, `search*/`: per-episode rows (`referee.jsonl`, `local.jsonl`) and logs.
- `seed_match/`: the scripts (`real_trace.py`, `first_input_diff.py`, `row_diff.py`, `tokdiff.py`) and every dump's `diff.txt`. The row dumps themselves (14 GB) are not committed.

The full per-floor traces (about 300 MB) are not committed; the `*.summary.jsonl` files keep each seed's result, act and floor.
