# Plan: an AlphaDefect-style no-search agent for Ironclad A10 on v0.111.0 (a4000, Nov 2-10)

**Status:** approved 2026-11-02 ~01:00 CET with the D1-D11 defaults (D1, D3 and D7 confirmed explicitly); executing on `exp/e2e-ironclad-v111` (written 2026-11-02 00:30 CET). Six critics reviewed it: facts, compute, reinforcement learning (RL), statistics, red team and operations.

**Support files** sit next to this plan in `2026-11-02-ironclad-e2e-ppo-v0111/`. `kb/` holds the knowledge base the plan cites by key: [AD] AlphaDefect forensics, [ENG] engine API, [PPO] our proximal policy optimization (PPO) stack, [CPPO] #1081's combat-PPO branch, [LES] lessons, [LIT] literature, [CM] compute model, [V111] v0.111 readiness. `review/` holds the six critic reports, the pre-review plan and the scripts behind the budget, forecast and power numbers (`review/scripts/`). "(est.)" marks an estimate, "(calc)" arithmetic, and "(verified)" a check on origin/main 917900a5f or #1081's branch 87d17834f on 2026-11-01. Abbreviations are defined in Appendix A.

## Status, 2026-11-05 16:30 CET (live; the sections below are the plan as approved)

- **Launch.** G1 launched Wed Nov 4 12:08 CET, about 16 h after the plan's median (Tue ~20:00). The rate was ~3k decisions/s at launch, then 6.4-6.9k/s after the S1 serving swaps on Nov 4. S1a, S1b and S2 as planned never ran.
- **One epoch since Nov 5 16:06** (PREREG Amendment 5, user decision; the §9 E=1 rule's preconditions were not met). Rate ~9.1k decisions/s; the update dropped from ~15 s to ~9 s per 131K iteration. Pre-switch pin `pinned_d0000550652720_e1_switch.pt`; the §9 revert rule applies at +300M.
- **Dates moved +72 h** (Amendment 5): lr anneal and KL freeze Wed Nov 11 06:00, stop Thu Nov 12 06:00, the one TEST read Thu Nov 12 09:00. At ~9k/s that is ~5.5B decisions, about AlphaDefect's v130051 (70.9% Defect A10, ~4.3-5.2B steps). The §0.2 and §10.4 times and the §8 budget table are superseded.
- **Results so far** (DEV-1000, EMA0.996, paired against R3_W3 11.5% on the same seeds): 100M 16.9% (K2, +5.4pp), 250M 19.0% (+7.5pp), 500M 27.8% (+16.3pp, p 3e-23), 595M (K3) 38.6% (+27.1pp), 750M 43.6% (+32.1pp, A9-est 63.1%). The forecast's median path was 11% at 1B. The 2B/3B stall checks (§1.4) are unlikely to bind.
- **K3, K4** keep their bars and wall-clock backstops (Nov 5 18:00, Nov 7 08:00), so they read at ~0.6B and ~1.8B, not 1.4B and 3B.
- **Deviations so far:** Amendment 4 (K2 combat agreement waived), Amendment 5 (E=1 and dates). Next speed step under way: the collection side (more parallel envs; the serial cut), as a hot-swap.

## 0. Summary

**TL;DR:** Train one AlphaDefect-sized transformer on the a4000 to play every Ironclad A10 decision on v0.111.0 with no search, from Tue Nov 3 to Mon Nov 9, then read it once on 4,000 sealed seeds. The central forecast is ~27% wins (50% interval 14-44%). Reaching AlphaDefect's 56% is a 1-in-8 outcome, and the 2B and 3B milestones show early which way the run is heading.

### 0.1 The plan on a page

- **Net.** One entity transformer of AlphaDefect's size: 6 layers × 256 (4 × 256 if the T0 bench finds the update too slow), 4.82M trunk parameters, ~7.4M in total. One trunk feeds both the combat and the out-of-combat (OOC) heads, so run and combat decisions train together (§4, §5).
- **Inputs.** Only what the game shows: #1081's combat tokens plus run context, the current act's bosses, pooled pile summaries, the full enemy telegraph and a 3-step intent forecast. Hidden state (true draw order, monster moves, upcoming rooms) feeds only the value module (§3.4, §4).
- **Algorithm.** Asynchronous decoupled PPO on whole-run returns: win +1, +0.05 per boss, +0.002 per floor, γ = 1. Combat advantages chain into the run's value, so a fight is scored by what it does to the run (§6).
- **Warm start.** Our best no-search nets act as teachers for the first 80M decisions. A Kullback-Leibler (KL) term toward them decays to zero by 500M. A learning check shows that PPO alone learns before the teachers fade (§7).
- **Practice.** Snapshots of the student's own runs, taken before elites and bosses, are reseeded and replayed. The boss share reaches 75% by 2B decisions, weighted to the act-3 double boss (§3.3).
- **Throughput.** 6.5k decisions/s at launch, ~13k/s from Wed, ~25k/s from Thu with the lean server and one-epoch updates. That's 8.1-9.4B decisions by Mon Nov 9 at central rates; AlphaDefect showed 56% at ~3.4B (§8).
- **Experiments.** Every a4000 hour trains one lineage. This week's experiments cost no box time beyond a 35-min bench: laptop checks, the bench and pre-registered switch rules. Recipe experiments need ≥1B decisions per arm, so they run after the week (§9, §13).
- **Rails.** A version factory that can only build v0.111.0 runs, fidelity gates F1-F9, an honesty test on the policy inputs, one engine binary for every comparison, and a watchdog with pinned checkpoints (§3, §8).
- **Evaluation.** Greedy play, paired seeds, exact McNemar tests. One read of 4,000 sealed TEST seeds on Mon Nov 9, against the deployed search champion (R1) and the best no-search pair (C*) (§1.5, §11).

### 0.2 Schedule at a glance

Times assume you approve by Mon Nov 2, 09:00 (A). Each hour later shifts every pre-launch time by an hour. T0 never moves before Tue Nov 3, 08:00, when the a4000 frees up. §10.4 has the full schedule.

| When | What |
|---|---|
| A to A+1.5 h | worktrees, merge #1081, engine wheel W1, PREREG-0 pushed, GitHub issue, draft PR |
| A+2 h to A+10 h | DEV baselines R1-R3 on the laptop; fidelity gates F2-F6 |
| Mon 21:30 | G-schema: inputs, actions and vocabulary frozen |
| Mon 23:30-Tue 01:00 | Rust getters on main (or deferred); wheel W2 built; Rust freezes |
| Tue 01:30 | G-skeleton: one A10 run end to end on the laptop |
| Tue 07:30-08:00 | H3 read; from T0 = 08:00 the a4000 is handed over as soon as G0 has passed |
| Tue 09:30-Wed 01:30 | G0 (P ≈ 0.25 / 0.5 / 0.75 by Tue 09:30 / 17:30 / Wed 01:30, est.) → T0 bench → launch smoke → G1 launch (median Tue ~20:00) |
| Tue | sealed TEST-4000 baselines on the laptop (R1-R3) |
| Wed; Thu | throughput steps S1a; then S2 (lean server, one-epoch updates) |
| Thu-Sat | K3 at 1.4B, K4 at 3B, DEV-2000 at every 1B |
| Sun Nov 8, 06:00 | learning-rate anneal starts |
| Mon Nov 9, 06:00-09:00 | stop; pick Final on DEV-2000; one sealed TEST read |
| Tue Nov 10 | results PR, full gate block on the merge result |

### 0.3 What to expect

- **Target.** AlphaDefect's 56.1% and 70.9% on Defect translate to 36-61% and 52-75% for Ironclad, depending on how much harder Ironclad is (calc, §1.2). Pros win 40.8%; our search champion 25.4% (v0.107).
- **Forecast (est.).** Median 27%, 50% interval 14-44%. P(<15%) = 0.27, P(≥40.8%) = 0.29, P(≥56%) = 0.13 (§1.3). The outcome is nearly bimodal: a stall near teacher level, or a climb.
- **Why not higher.** The model assumes our new, untuned stack needs ~3× AlphaDefect's decisions for the same progress (10-90%: 1.4-6.5×), because our own combat PPO stalled early (dev_boss flat after 250M). That one unknown moves the median between 20% and 37%. ±1B decisions moves it ~4pp.
- **On track.** DEV-2000 ≥16% at 2B decisions, ≥21% at 3B and ≥31% at 5B. Below 11% at 2B or 14% at 3B is a stall: diagnose for ≤6 h while training continues (§1.4).
- **Bars** (one TEST read): Bar1 (must) beats the best no-search pair by ≥5pp at p<0.01. Bar2 (should) is non-inferior to the search champion. Bar3 (stretch) beats it by ≥3pp. BarR reports ≥40%. Pass chances 0.69 / 0.63 / 0.55 / 0.30 (est., §1.5).

### 0.4 Your decisions

| # | Decision | Recommendation (the default) |
|---|---|---|
| D1 | a4000 handover | #1081's runs keep the box until T0, and until our G0 passes if that's later. At approval, allow read-only copies of it06000 (R2's combat net) and H2's newest checkpoint, one `ldd --version` on the a4000, and, if its glibc is older than 2.44, a 15-30 min nice-19 wheel build there (Mon ~23:30). At handover, the #1081 session checkpoints and exits its runs within 30 min |
| D2 | Headline | Final's TEST win rate is the objective; Bar1 must, Bar2 should, Bar3 stretch, BarR reported |
| D3 | Teachers | our no-search nets as decaying teachers; no human behavioural cloning, no clairvoyant teachers, no MCTS at sampling time |
| D4 | Potion swap | student ON; R1 as served (OFF); no swap-ON R1 arm (it decides nothing) |
| D5 | Scope | research artifact; nothing enters `champions.toml` this week |
| D6 | After Nov 9 | continue iff paired DEV-2000 gained ≥ +2.0pp over the last 1B decisions and the box is free; TEST is read once regardless |
| D7 | Rust | work package WP-ENG-1 (four additive getters, §3.4) lands on main by Mon 23:30 with the full gate block; then Rust freezes until Nov 10, except one W3 at a restart (a late WP-ENG-1 or a sim-exploit fix, §8, §12) |
| D8 | Laptop | ~100-170 process-hours (proc-h) at 16-20 procs: A+2 h to A+10 h, Tue 07:30-16:00, G0's smoke and learning check (~75 min), ~2 h more if W1 ≠ W2, and the boss-state panel DEVBOSS at nice 10 Tue-Wed |
| D9 | Real-game gaps | after the week; reserve their action-kind ids now |
| D10 | #1081 merge path | stack the results pull request (PR) on `exp/combat-ppo-screen`, or land it after #1081 merges, with `--merge`; never push to or squash #1081 |
| D11 | #1069 (reward-screen potion pick) | coordinate, don't wait: in W2 if it's on main by Mon 23:30, else on a reserved action-kind id agreed at G-schema |

## 1. Goal, target, forecast and bars

### 1.1 Goal

By Mon Nov 9, one no-search net, trained on the a4000 in under a week, plays every decision of an Ironclad A10 v0.111.0 run and is measured once, paired, against the deployed search champion and the best no-search agents. It replicates AlphaDefect's recipe class: entity transformer, PPO, greedy play.

**Primary objective:** the final greedy A10 WR on TEST; bars are reported, not optimized. On AlphaDefect's curve, 25% → 30% needs only 1.14-1.23× more decisions (calc), so decisions are the default use of every box-hour; anything else needs an expected-WR case in PREREG-0.

### 1.2 Target

| Reference | A10 WR | Source |
|---|---|---|
| AlphaDefect v103049 (Defect) | 56.1%; picked on 384 of its 1,000 seeds (59.1% there, 54.2% on the rest) | [AD] |
| AlphaDefect v130051 (Defect) | 70.9%, not picked | [AD] |
| Their Ironclad-equivalents | 36-61% / 52-75% at shifts of −0.83 to +0.21 logits (calc) | release panels 2026-10-11 and 2026-10-26 (after Ironclad's A10 campaigns) |
| Pros (pros25 human set, 408/999 runs), Ironclad | 40.8% | GH #819 |
| Our Ironclad champion + MCTS, v0.107 | 25.4% | `champions.toml` |
| Our Ironclad no-search pairs, v0.107 | 8.4-12.2% | [CPPO] |
| Our Defect champion on AlphaDefect's seeds | 18.3% | 2026-10-31 |

- **Decisions:** one AlphaDefect version per fresh 32K → 65K batch puts v103049 at 3.38B and v130051 at 4.26-5.15B decisions (0.84-2.57B if multi-epoch; calc), at ~7-12k/s on a 3090 and a 4090 for a week. We have ~1.4× its GPU throughput per hour for 5.4-5.9 days against its ~6.5: ~1.2× its GPU time (est.).
- **Caveat:** its model.md lists "upcoming encounters" embeddings [AD]; if they reached its policy, its numbers include information ours lacks.

### 1.3 Forecast

The statistics review's Monte Carlo model, re-run at this plan's budget (`review/scripts/forecast_rev.py`), puts our run on AlphaDefect's logit-linear curve (slope U[1.2, 1.9] per e-fold of decisions, halving past 5.15B). Inputs: our efficiency M (decisions per AlphaDefect decision of equal progress), lognormal with median 3; a 0.2-1.0B warm-start credit; a character shift N(−0.4, 0.4) logits; a floor at teacher level − 2pp; a 0.2 stall branch at N(12, 2.5)%; a budget median of 5.3B (0.65 × the compute review's realistic 8.2B; §8's central scenarios span 6.1-9.4B).

**Result:** p10 10.7%, p25 14.2%, median 27.0%, p75 43.7%, p90 59.3%; P(<15%) 0.27, P(≥25%) 0.53, P(≥40.8%) 0.29, P(≥56%) 0.13.

The outcome is near bimodal: a stall at teacher level, or a climb. Two unmeasurable inputs dominate (calc, `review/scripts/forecast_sens.py`): M = 2 / 4.5 moves the median to 37% / 20%, and a shift of −0.83 / +0.2 to 20% / 40%. ±1B of budget moves it ~4pp.

### 1.4 Early signals

The model's median path for non-stall draws (p25-p75) is 11% (9-16%) at 1B decisions, 16% (11-24%) at 2B, 21.5% (14-31%) at 3B and 31.5% (21-44%) at 5B. The 1B read only checks for regression (teacher level by design). DEV-2000 below 11% at 2B or 14% at 3B is the stall branch (diagnose); below 21% at 5B is under the band's lower quartile. Earlier signals: the learning check, the T0 smoke, K2, per-boss conversion and a slope of ≥3pp per 1B; P7 refits logit(WR) on ln(decisions) at each 1B.

### 1.5 Bars

TEST: 4,000 seeds, greedy, paired, one binary, read once.

| Bar | Contrast | Rule | P(pass) (est.) |
|---|---|---|---|
| Bar1 (must) | Final − C* | Δ ≥ +5pp and exact McNemar p<0.01 | 0.69 |
| Bar2 (should) | Final − R1, as served | 95% Newcombe paired CI low end ≥ −3pp | 0.63 |
| Bar3 (stretch) | Final − R1 | Δ ≥ +3pp and p<0.01 → deployment candidate (§13) | 0.55 |
| BarR (reported) | WR(Final) | ≥40% (pro level); ≥56% also reported | 0.30; 0.13 |

- **Power (exact, calc):** paired SE 0.79pp at discordance d = 0.25 (Bar1) and 0.87pp at d ≈ 0.30 (vs R1). Bar1 passes with P = 0.11 / 0.51 / 0.90 / 0.99 at a true +4 / +5 / +6 / +7pp, so a true +5pp is a coin flip by design. Bar2 passes iff Δ̂ ≥ −1.3pp (P = 0.93 at parity; 0.69 at n = 2,000). Bar3 passes with P = 0.88 at +4pp.
- **Sequence:** Bar1 → Bar2 → Bar3, each at its own level, stopping at the first failure (familywise error ≤0.025 one-sided). Holm covers only the descriptive family (§11).
- **Meaning:** C* was trained on v0.107, so a Bar1 pass bundles the recipe, more RL and the version change; attribution needs a continuation control (§13). Every A10 number carries its A9-est.

### 1.6 Deviations from AlphaDefect

| Item | AlphaDefect [AD] | This plan (why) |
|---|---|---|
| Start | RL from scratch | teacher-acted warm start, KL to it 0 by 500M (random A10 play dies on floor ~2.7 [ENG]) |
| Update, lr | one step per fresh 32K → 65K batch; 1.5e-4 → 1.1e-4 | 131K-decision iterations, E=2 then E1a; 1e-4 × a KL-adaptive factor (#1081 was KL-limited) |
| Practice, reward | boss share 37% → 77%; HP lost counts | boss share 40% → 75%; no HP term (the hybrid's suspected cap [CPPO]) |
| Inputs | 4-step forecast; ~18 ID tables, among them upcoming and recent encounters (their docs don't say whether the upcoming ones feed the policy or only the training-only value heads; they state the policy never sees an upcoming random outcome) | full telegraph, 3-step forecast, one ID table plus the teacher's S_ctx tokens; upcoming rooms value-only (§3.4, enforced by F7) |

## 2. Baselines on v0.111

No current Ironclad agent has a v0.111 number (priors: our Defect champion 18.3%; Ironclad plr_it550 11.0% before #1029). XxHash64 seeding makes every panel new [V111].

| ID | Agent | Panels | Cost (calc) | When (laptop) | Expected (est.) |
|---|---|---|---|---|---|
| R1 | cont300_it1150 + shipped MCTS 1000/4000 (`net_greedy`, swap OFF) | DEV-1000, TEST | 12-20 proc-h per 1,000 (12.4-14.5 measured on v0.107 A10; the top allows ~1.4× for v0.111) | DEV at A+2 h (~1 h at 20 procs); TEST Tue (2.4-4 h) | 17-24% |
| R2 | champion run net + combat-PPO it06000, no search | DEV-2000, TEST | ~1.5-2 proc-h per 1,000 | DEV at A+2 h; TEST Tue | 8-12% |
| R3 | H2 pair (H3 if it reads GO); provisionally H2's newest checkpoint | DEV-2000, TEST | same | DEV at A+2 h, final Tue 08:00; TEST Tue | 10-13% |

- **Teacher:** the better of R2 and R3 on DEV-2000; within 2pp, R3 (its run net co-trained with no-search combat). **C\*:** the higher DEV-2000 WR of R2 and the final R3. Both rules are in PREREG-0. Loaders assert `ck["cfg"]["features"] == "card_mods,reward_upgrade"`, because `NetPolicy` silently falls back to default features (net_policy.py:35-47).
- it06000 (~600.6M decisions) and the H2/H3 nets live only on the a4000. Without the copies (D1), R2 and R3 DEV run Tue 08:00-09:00 and the G0 smoke uses the archived it01750.

**Runner** `sts2-pycli/scripts/e2e_arm.sh ARM PANEL N SEED_START -- <ppo.probe args>` (WP6, A+2 h) runs the venv's python, not `uv run` (wheel_guard.py:202-203), under `env -i` with only HOME, PATH, LANG, `OMP_NUM_THREADS=1`, `PYTHONPATH=<pin>:python`, `STS2_GAME_VERSION=v0.111.0` and the arm's own env, so none of the 52 `STS2_RUNAI_/ROLLOUT_/MCTS_` knobs leaks in. It logs `wheel_guard`, SHA and argv, runs 250-seed shards with skip-if-exists (`--out` truncates, workers.py:564), writes `<arm>.sha256`, and for TEST prints only counts (`evals/TEST/` at mode 700).

```text
cd ~/projects/sts2-sim-e2e-w/sts2-pycli && mkdir -p evals && free -g
systemd-run --user --scope -p MemoryMax=24G scripts/e2e_arm.sh R1 DEV 1000 160000000 -- \
  --policy net_greedy --ckpt data/ppo/ppo_fullrun_ironclad_cont300_it1150.pt \
  --sims 1000 --boss-sims 4000 --run-timeout 14400 --act1 roll --ascension 10 --procs 20
systemd-run --user --scope -p MemoryMax=24G scripts/e2e_arm.sh R2 DEV 2000 160000000 -- \
  --policy net_greedy --ckpt data/ppo/ppo_fullrun_ironclad_cont300_it1150.pt \
  --combat policy --combat-ckpt <it06000.pt> --run-timeout 3600 --act1 roll --ascension 10 --procs 20
# TEST: PANEL=TEST N=4000 SEED_START=161000000; commit only evals/TEST/<arm>.sha256
```

`--run-timeout` and `--procs` are explicit because `ppo.probe` defaults to 600 s and cpu_count − 2 workers (workers.py:31, :521).

## 3. Environment

### 3.1 Version factory, plumbing, seeds (WP1)

```python
# ppo/e2e/factory.py: the only construction site
assert 0 <= seed < 2**31
rm = sts2_rs.RunManager(seed, character="Ironclad", ascension_level=10, neow_start=True,
                        act1="roll", game_version="v0.111.0", mode="agent",
                        interactive_potion_swap=True)
assert rm.game_version == "v0.111.0" and rm.mode == "agent" and rm.interactive_potion_swap
```

- **Properties:** `rm.game_version`, `rm.mode` and `rm.interactive_potion_swap` are properties; the installed wheel 3532eacdb lacks `mode`, so W1 comes first. Agent mode is required (parity mode edits the deck front [ENG]); a new fight-start assert reads the `cs.game_version` getter (lib.rs:721-726).
- **Plumbing:** Rust never reads `STS2_GAME_VERSION`, so an omitted kwarg silently means v0.107.1 (game_version.rs:51-52). Pass `game_version` at the 11 `RunManager(` sites and the 8 unversioned `build_scenario_state(` sites under `ppo/` (combat_bench/build.py:640, :644, :900, :979; combat_bench/run.py:278; pilot_human_decks.py:876; rollout_learn.py:177; rollout_starts.py:118); `rollout_starts.build_fight` asserts matching versions. An AST guard test fails on any unversioned call in `ppo/e2e/`, `ppo/combat_ppo/`, `ppo/rollout_starts.py` or `ppo/combat_bench/`.
- **Static tables:** regenerate `combat_ctx/static_tables.py` from the v0.111.0 C# decompilation, with the DECOMP pin (tests/test_combat_ctx.py:54) and `snapshot.RELIC_COUNTER_DISPLAY`.
- **F8:** v0.107.1 objects injected into the env and a combat_bench fight fail loudly. **Stamps:** source digest, wheel, game version, vocab and schema sha256 in cfg, checkpoints, stats and eval rows.

**Other seeds** (<2^31; panels in §11; re-grep at PREREG-0, the repo's largest is 947M): probe-driven smoke and identity runs 162,000,000-162,000,999; practice reseeds 600,000,000-639,999,999 (per-worker PCG64). Trainer runs use #1081's slot layout, base + 7·(start iteration) + (worker·64 + env)·100,000 + run count (learner.py:250, actors.py:99), which spans ≤256M, so they sit above every panel: the student from 1,000,000,000 (<1.26B), and every throwaway trainer run (smokes, learning checks, B-T0) from 1,300,000,000 (<1.56B; they may share seeds, since none feeds a panel). A trainer based at 162M or 174M would reach DEVBOSS's 175,000,000 (calc).

### 3.2 Decisions (WP2)

- **Env:** `ppo/e2e/env.py` on `PpoRunEnv(combat="yield")` (merge tree only); classify by action kind, never `phase()`; call `materialize_event()` before reading options [ENG].
- **Auto-advance:** a step with one legal action or one merged class gets no forward pass and no row (15.7% of combat steps [CPPO]); `honest_snapshot` still runs (encode.py:59-66).
- **Dedupe:** plays merge by `(_canon(_card(card)), target_index)` (cost, enchantment, per-copy damage and block, flags), potions by (id, target), choices by `choose_outcome_key`, all from the belief view. Members have byte-identical tokens (asserted); teacher probabilities merge by log-sum-exp. `play_edge_key` would merge enchanted with plain Strikes and played with fresh Rampages (snapshot.py:487-503).
- **Caps:** 100 turns or 1,000 decisions per fight, 6,000 per run; capped = loss; alarm above 0.1%.
- **Action logs:** u8/u16 indices, zstd: every eval run, 2% of training runs and every act-3 boss-2 segment, capped at 10 GB (JSON: 200-600 GB, calc).
- **Pre-flight** (playbook §0.1-0.2) at the G0 smoke and at 100M: every surface and fine key (card reward, remove, upgrade, transform, enchant, grant and smith, boss relic, relic reward, potion swap, Neow) occurs, `unknown` = 0, and per-slice embedding drift beats never-looked-up padding rows (else the #317 failure: stop, audit).
- **Accepted gaps** (live only; reserved ids): #1069's reward pick, treasure, Crystal Sphere, OOC potions and discard, the Fake Merchant's single buy, empty intent history. Already handled: shop `upgraded` (d3278cfce), Sea Glass's optional 0..15 pick (8510b701c; its selection spec is checked at G-schema).

### 3.3 Practice on own runs (WP2; may enter at a restart)

- **Capture:** `rm.clone()` (4-10 µs [ENG]) plus the Move, at the run decision before an elite or boss node, in six cells: elites by act; the act-1, act-2 and act-3 double boss. Per-worker first-in, first-out queues of 32 (max age 50M decisions, ≤4 reuses).
- **Start:** `snap.clone()`, `reseed_run(s)` (run RNG only, run_manager.rs:1955-1958), then the Move. Boss 2 has no cell: a double-boss start that wins boss 1 trains boss 2.
- **End:** double-boss segments play to the true outcome; others end at the first OOC decision, a value-only bootstrap row.
- **Schedule** (linear over 0-2B): practice 40% → 25% of slots; boss share 40% → 75%; bosses 65% double, 20% act 2, 15% act 1; elites by act 20/40/40. A cell activates at ≥8 entries, else falls back; fallback rates are logged.

### 3.4 Honesty (WP3)

- **Policy whitelist** (snapshot.py FORBIDDEN plus a run whitelist): hand with `card_damage_preview`; draw pile as a sorted multiset from `cs.belief_view` (lib.rs:959), only pinned top cards placed; discard, exhaust, powers; relics with display counters; potions; `rm.master_deck_detailed()`; map, gold, offers; the current act's boss(es), including the visible A10 second boss.
- **Intents:** per enemy, the full telegraph from `current_move().intents` and a 3-step forecast from `peek_intents` (monster_ai.rs:1058-1092, verified: cloned cursor, no RNG, highest-weight branch, certainty = weight share), aligned by `performed_first_move`, because its step 1 is the telegraph only before a monster's first move. Python cannot read `performed_first_move` today, so the `peek_intents(3)` getter does the alignment in Rust.
- **WP-ENG-1** (D7): four additive pyo3 getters (telegraph, `peek_intents(3)`, encounter id, true top-k draw); the wire and `WIRE_VERSION` stay unchanged.
- **Privileged (value only):** true top-10 draw (replacing the second `state_to_dict(cs)` in `priv()`), true next moves, this act's rooms (`generated_rooms()[act_index-1]`; `[act_index]` is the next act [ENG]), future maps and bosses.
- **F7:** permuting draw order (`cs.sample_world`, lib.rs:970), combat RNG, rooms, future maps and bosses leaves policy logits bit-identical on both serving paths. PRIV is absent from the trunk input, not key-masked as in #1081 (nets.py:225-235); move state stays fixed (the forecast reads it). F7 runs at G0, at each serving hot-swap and at each 1B.

### 3.5 Fidelity gates (WP8; FIDELITY.md)

| Gate | Check | Pass | Blocks G1 |
|---|---|---|---|
| F1 | `PYTHONPATH=python <W2 python> -m ppo.wheel_guard`, both hosts | identical output and sha256 | yes |
| F2 | `experiments/alphadefect-compare/replay_parity.py --repo <AD worktree at 36bd1fd> --procs 16` on AlphaDefect's 1,000 own-engine runs (1ffcc38 moved the replays) | full_match ≥118/1000; no new combat_state class | no |
| F3 | `run_parity`, 200 A10 seeds, v0.111 oracle | 200/200 | yes |
| F4 | `python/act23_state_parity.py --character Ironclad --ascension 10 --game-version v0.111.0 --full-state --trace-split -n 50 --workers 16` | no new first-split class vs A0 (84/100) | no |
| F5 | #918 lockstep (`encounter_parity.py`), A10 | elites ≥99% (798/800 today); first boss baseline | no |
| F6 | random-target multi-hit cards vs the C# oracle | no failure | no |
| F7 | honesty invariance | bit-identical | yes |
| F8 | v0.107.1 injection | loud failure | yes |
| F9 | invariants on 1% of steps (HP ≤ max, energy ≥ 0, pile conservation) | 0 per 1e6, live all week | yes |

A non-blocking failure is fixed this week only if its fix merges by Mon 23:30 and so enters W2 (the DEV baselines then re-run on W2, §8). Otherwise it waits for after the week, or for the one W3 if §12's sim-exploit trigger fires. Human-fight sync (40.5% [V111]) is reported only.

## 4. Observation (WP3)

The combat stream keeps #1081's S_ctx tokens bit-for-bit (21 sections; 15 categorical, ~122 continuous columns [CM]), which the combat teacher reads through a section mask and column slice (X-C). New information enters only as new sections and trailing columns. The run stream is new; the run teacher's own `featurize_row` runs beside it while used (~0.09 ms per decision, est.).

| Section | Stream | Tokens (est.) | Content |
|---|---|---|---|
| S_ctx | combat | ~49 (p99 81) | hand, piles, powers, player, enemies, intents, relics, potions |
| RUNCTX | both | 1 | run state, bosses left, floors to boss, decision type, selection spec, encounter id |
| BOSS | both | 1-2 | the current act's boss(es) |
| SUMMARY | both | 5 | pooled piles, deck, relics: mean ID embedding plus counts [AD] |
| Run body | run | ~35-55 | deck, relics, potions, offers (cards, shop, events, rest, Neow, boss relics, upgrades) |
| MAP | map | ≤40 | reachable cone; 6 route summaries per Move [AD] |
| Trailing columns | combat | - | telegraph and forecast with certainty; Python tactical; 48-wide content prior; zero-init reserved slots |
| PRIV | value | ≤16 | §3.4 |

- **Size:** mean ≈57 tokens (calc; AlphaDefect ~60); buckets 32-256; ≤64 options (asserted). **Vocabulary:** one append-only engine-registry ID table, 4,096 rows (~1,550 used), sha256 in cfg.
- **Content prior:** a 48-column table, one row per vocabulary ID, from a new extractor over the v0.111.0 C# decompilation (type, rarity, cost, base numbers, keywords, target, upgrade deltas), with a zero row and a missing flag.
- **Tactical columns** (Python, 20-50 µs, est.): per enemy, the most damage dealable this turn (a knapsack over `card_damage_preview()`) and a killable flag; the HP loss if the turn ended now.
- **Lean numeric stem:** symlog → Linear → SiLU → Linear with no cross-column normalization first, so zero columns stay inert until filled. It replaces 16 gathers and the 1,464-wide sinusoid (34% of inference time [CM]).
- **Frozen at G-schema:** whether duplicate action tokens are removed (later removal changes all attention).

## 5. Network (WP4)

| Part | Spec | Params (est.) |
|---|---|---|
| ID table | 4,096 × 256, shared | 1.05M |
| Stems | section, type, slot embeddings; lean numeric MLP | 0.4M |
| Trunk | 6 pre-norm blocks, d = 256, 4 × 64 attention heads, RMSNorm, SwiGLU 704, no positions | 4.82M (AlphaDefect 4.74M) |
| Policy heads | combat and OOC MLPs over [h_action; h_STATE; e_kind] (the action's and the state token's trunk outputs, a decision-kind embedding) → 256 → 1; softmax over merged options | 0.3M |
| Value module | 1 block over [trunk outputs; PRIV]; V_combat, V_run; zero-init; fp32 projection | 0.85M |
| Total | | ~7.4M |

- Value gradients train the shared trunk, as in AlphaDefect [AD] (sharing doubles update speed [CM]).
- **Valve:** stop-grad from the value module into the trunk (PRIV lookups detached too), flipped at a checkpoint only if a decision type's entropy collapses while its value loss rises for 20 iterations. Low EV is not a trigger (stop-grad cost −0.24 EV in our screen [LES]). If EV falls >0.1 below its 50M level, add a zero-init value block at a restart.
- 6×256 unless B-T0 picks 4×256 (§9).

## 6. Algorithm (WP4)

| Item | Setting |
|---|---|
| Objective | decoupled PPO (`ranks.py`): ratio π_θ/π_prox clipped at 0.2 (π_prox from an update-start pre-pass); IS weight π_prox/π_behav clamped at 2; dual-clip c = 3 [LIT] |
| Batches | 131,072 decisions per iteration; E=2, E1a from S2; minibatch ~8,192 = 24 random strata of the length-sorted batch (384 strata of 341-342 rows; 8 per rank), redrawn each epoch (#1081's contiguous slices gave run-row shares of 9-46% per step, calc) |
| KL control | none in V-warm and U0. In U1-U2, KL on student-acted rows of a fixed 8,192-row probe after each epoch, with a hard stop above 0.06; lr factor m ×1.5 if probe KL <0.005, ÷1.5 if >0.02, within [0.3, 2], frozen Sun Nov 8 06:00. #1081's all-row per-minibatch stop fired in 60-86% of iterations |
| Returns | γ = 1. Combat rows chain by row adjacency: successor = the next stored row of the same run (combat or run) or the terminal; A_t = δ_t + λ_c·1[successor is combat]·A_{t+1}, λ_c = 0.95. So act-3 boss 1 chains into boss 2 (a per-fight split gives a won boss 1 E[δ] = −0.62). Run rows: λ_r = 0.97 over the run-row subsequence, held back until the run ends or for 4 iterations (bootstrap weight 0.78 → 0.08; staleness mean 3.7, max 5; ~39K rows). Combat rows cut every iteration. Bootstrap rows enter no loss or statistic and train once, next iteration. r_t sums all reward to the next stored row |
| Values | MSE × 0.5; no value clipping (delete ranks.py:308-309); update-start targets |
| Advantages | normalised per decision type by exponentially weighted moments of student-acted rows |
| Entropy | per type over merged classes / log(n_classes). β = 0 while λ_KS > 0.3 (to 374M), then geometric to 4B: combat 0.01 → 0.002, OOC 0.0125 → 0.006. A collapse (raw entropy <0.1 nats for 20 iterations) lifts the gate, β ×1.5, at most twice |
| Optimizer | one AdamW (weight decay 1e-4) with groups trunk side 1e-4 and value module 3e-4, × m × anneal(t); one all-reduce, one clip_grad_norm 1.0, one step per minibatch; `lr_warmup_iters` 10 apart from `value_only_iters` 3. Scaling rules put #1081's 1e-4 at 1.06-1.41e-4 here |
| Schedules | clocks in `state.pt`: anneal(t) = 1 − 0.7·clamp((t − Sun Nov 8 06:00)/24 h); entropy, λ_KS and β_act on decisions since T0; `budget_decisions` ≥ 1e10 (#1081's 1e9 anneals lr to 0 and stops at 1B) |
| Precision | bf16 autocast; fp32 weights, Adam, log-softmax and value projection (bf16 spacing at V = 1 is 0.0039 > the 0.002 floor reward) |
| EMA | 0.996 and 0.999 per iteration (~33M, ~131M decisions) |

**Reward (pre-registered):** win +1.0; +0.05 per boss beaten (act 1, act 2, act-3 boss 1, the last detected by the round reset, runslot.py:287-292); +0.002 per new floor; death or cap 0; no HP shaping. AlphaDefect's stronger checkpoint entered the act-1 boss with 9pp less HP and won more [AD].

**Unit tests before G0** (reference scripts in `review/scripts/`): T1 cut vs uncut GAE differ exactly by λ^(m−k)·B per chain; T2 boss 1 chains into boss 2, bonus once; T3 bootstrap rows are inert and train once; T4 Σr over stored rows equals the env total; T5 a deadlock ends at turn 100 as a loss; T6 enchanted vs plain Strike and played vs fresh Rampage stay apart, plain Strikes merge; T7 n_classes ≥ 2, entropy in [0, 1]; T8 teacher probabilities sum to 1 under a bijective option map; T9 KL on student rows, no U0 stop; T10 no value clip, one step per minibatch; T11 kills at iterations 7, 150 and random reproduce the next 3 iterations; T12 strata cover each row once, padding ≤1.05; T13 ≥95 distinct hands in 100 reseeds; T14 PRIV permutations leave S0 and S1b logits bit-identical.

## 7. Warm start (WP4)

| Phase | Decisions | Actor | Loss |
|---|---|---|---|
| V-warm | 3 iterations | teacher | value only |
| U0 | 0-80M | teacher with probability β_act (1 → 0), per fight (combat) and per act (run) | cross-entropy H(π_T, π_S) per type on all rows; PPO, advantage moments and statistics on student-acted rows; value on all |
| U1 | 80M-500M | student | PPO + λ_KS·KL(π_T‖π_S) per row type; λ_KS 1 → 0 linearly by 500M (0.3 at 374M); teachers off from 500M |
| U2 | 500M-end | student | PPO |

- Rows carry an actor id; a teacherless row type (potion swap may be rule-side, env.py:220-232) gets λ_KS = 0.
- No per-head KS cuts: a +3pp DEV-1000 read is ~2σ, so 16 reads make a spurious cut likely. Teacher log-probs run on GPU0 at sampling (+25-30% inference while used, est.).
- **Learning check** (laptop 45 min at G0; box 15 min at T0): random init, λ_KS = 0, no teacher, A10. Pass iff the mean floor of the last 10K finished runs beats the first 10K by ≥1.0 (random play reaches 2.7 [ENG]), combat EV > 0.1, no NaN. Fail → launch with λ_KS held at 1. The check re-runs on the box at K2 and at each restart, and the λ_KS decay starts only once it passes.
- **K2** (~100M, ~4-5 h after launch): student DEV-1000 ≥ teacher pair − 8pp, paired; agreement ≥85% combat, ≥75% run; finite value loss; EV within 0.1 of its 50M value. Fail → fix, relaunch from the last good pinned state.

## 8. Systems (WP5, WP5b, WP7)

**Layout L1:** GPU0 serves the student, teachers and a low-priority eval queue; GPU1-3 run the data-parallel update; 32 spawn workers × 32 envs (B-T0: 32 or 40) plus ≤6 eval workers at nice 10; evals ≤5% of wall clock.

**Hot-swap ladder.** Each step keeps the state dict, must pass the parity check, and retries at each later 25M checkpoint.

| Step | Change | Rate (calc) | Binds | When |
|---|---|---|---|---|
| S0 | shared trunk, lean stem, bf16, buckets, compile | 6.5k/s with teachers (3.8-9.0k); 8.3k without | serial Python server (~138 µs per decision) | T0 |
| S1a | two worker groups alternate on GPU0 | 12.9k with teachers; 14.4k without | 3-GPU update at E=2 (cap 16.0k) | Wed 10:00 |
| S1b | lean shared-memory server (46 → ≤15 µs), packed inference, double buffering, compact rows (15.7 → ~3.5 KB), each flagged | no gain at E=2 | - | Tue-Thu |
| S2 | E1a by the §9 rule | 25.0k (18.6-35.1k) | update; worker CPU ~1.0 ms | Thu 10:00 |

Rates use [CM]'s model × 0.88, the realised/modelled ratio of #1081's a4000 run; 3090 ratios are ±20%. E1b (no pre-pass) is dropped: plain PPO under staleness (~15% of rows outside the clip band, est.) for ~8% more speed.

**Budget by Mon Nov 9 06:00** (calc, `review/scripts/budget_rev.py`; launch Tue 14:00, S1a Wed 10:00, 4.5 h of restarts (S1a, S1b/E1a and 2 unplanned), 2% evals):

| Scenario | Low | Central | High |
|---|---|---|---|
| S1b + E1a Thu 10:00 | 6.6B | 9.4B | 13.0B |
| S1b + E1a Fri 10:00 | 5.6B | 8.5B | 11.6B |
| E=1 rejected | 4.1B | 6.1B | 8.9B |
| S1b never lands | 3.1B | 6.1B | 7.7B |
| Launch Wed 00:00, S1b + E1a Fri 10:00 | 5.4B | 8.1B | 11.1B |

The schedule's median launch (Tue ~20:00 at A = Mon 09:00) sits between the first and last rows. A 24-h S1b/S2 slip costs ~0.9B and a 10-h launch slip ~0.35B, so late T0 items get cut, never WP5 or WP5b time.

**Parity check:** old and new paths in fp32 on the same 10K states and a fixed 65K batch: |Δloss| < 1e-3, max |Δlogit| < 1e-2, equal greedy actions where the top-2 gap exceeds 0.05. The new path's bf16 is then checked against its fp32 by X-B's rule (two bf16 paths differ by rounding alone).

**One binary.** W1 (A+1 h, laptop: `cd sts2-pycli && WHEEL_VENV_NO_GC=1 just sim-engine-pgo`, 15-18 min; the variable acts only at build time) builds the merge commit (Rust = origin/main 917900a5f) for the first DEV baselines. W2 (Tue ~00:00) adds WP-ENG-1 and F-gate fixes and serves the student and every TEST arm. Rust then freezes, except one W3 at a restart: a late WP-ENG-1, under the identity rule (zero-init columns preserve the function), or §12's sim-exploit fix. If a W3 changes any identity-seed result, the R1-R3 TEST arms re-run on W3 (~3-5 h of laptop time) before the Mon Nov 9 read, so Final and every baseline share one binary.
- **glibc:** a laptop-built `sts2_rs.abi3.so` needs GLIBC_2.44. If the a4000's glibc is older, build W2 there at nice 19. Gate: the highest `GLIBC_` symbol in `objdump -T` ≤ the older host's, and equal sha256 on both hosts. Pin W2 outside `.venv-h`, which the wheel-venv garbage collector reaps: `pin_wheel("$HOME/runs/e2e-pin")`.
- **Identity:** W1 ≡ W2 on 200 R2 and 50 R1 smoke seeds (Tue ~01:00), else the DEV baselines and F3 re-run on W2 (~2 h). Cross-host at T0 on 20 R2 and 20 R1 seeds: each host binds its own libm, so MCTS can differ; any difference puts every TEST arm on the laptop.

**Ops (WP7).**
- **Process:** one systemd user service per lane (`Restart=on-failure`, `MemoryMax=200G`; linger and `systemd-run --user --scope` checked at T0, else the watchdog checks resident memory) runs `scripts/e2e_watchdog.sh` with `PYTHONPATH=python`, `STS2_GAME_VERSION=v0.111.0`, `STS2_RUNAI_POTION_SWAP=1`, `STS2_PPO_FEATURES=card_mods,reward_upgrade`, `OMP_NUM_THREADS=1` and the malloc settings [LES]. Launchers use `%h` and `$HOME`, never a hard-coded home directory.
- **Watchdog** (each failure path tested before T0): resumes after a crash; stops on `DONE`, >4 starts in 2 h or <30 GB free disk. A stall is a rate below half the process's own median for 10 min, counted from 30 min after any start. A NaN resumes from the second-newest full state with sampler seed +1e6; a second within 100M stops (HUMAN). The sampler RNG derives from (seed, iteration), fixing learner.py:172-180; scripts wait on their own process IDs, never `pgrep -f`.
- **Checkpoints:** `state.pt` every 5M decisions on the save job, not per update (which would write ~150 MB every 5-8 s); rolling full states every 25M (keep 3); never-rotated pinned states at every K-gate, restart, hot-swap, E=1 switch and 250M, with sha256 in `ckpt/MANIFEST.tsv`; slim states every 100M. Deliberate rollbacks (K2 relaunch, E=1 revert, exploit fix) start only from pinned states; only the NaN path above uses a rolling state. Disk: ≤40 GB of ~104 GB free by ~9B (est.); prune below 45 GB, stop below 30 GB.
- **Deploy:** each hot-swap is a new a4000 worktree at the new SHA, resumed from a pinned state; nobody edits or pulls a running worktree. The learner checks `worker_wheel_mismatch` and `worker_source_mismatch` every batch and records `python_source_digest()`.
- **Logs:** `stats.jsonl` per iteration (rates, staleness, per-type KL, clipfrac, IS tails, entropy, EV, agreement, λ_KS, β, lr vs closed form, caps, stamps); `throughput.jsonl` per minute.

**Runbook** (RUNBOOK.md next to PREREG-0; WP7, due Tue 09:30). The campaign's laptop session operates. `scripts/e2e_driver.py`, on a 5-min systemd timer, executes only pre-registered rules and writes `status.json` and `ALARM_<name>` files; you decide D1-D11 and HUMAN alarms. `~/runs/e2e/campaign_state.json` holds lane, SHA, wheel, last pinned state and next gate; actions append to CAMPAIGN_LOG.md. The laptop polls every 30 min and pushes a notification on any ALARM or a status older than 20 min (`tailscale ping a4000` before calling the box down). Stop and call HUMAN on lr off its closed form by >1e-6, F9, or a wheel or source mismatch. At RAM <20 GB stop evals, then the lane.

## 9. Experiments

**Rule:** an experiment runs this week only if its result can change a decision this week; recipe questions keep pre-registered priors and move to §13 (§0.1). One binary, greedy, paired seeds; no gate reads a treatment-selected subset such as P(win | act 3).

| ID | Decision it feeds | Pre-registered rule | Cost, when |
|---|---|---|---|
| X-C | schema accepted | combat-teacher logits bitwise equal through our schema on recorded states | laptop, Mon |
| X-T | tokenizer caching | new vs #1081 tokenizer on 20K states, one pinned performance core: mean ≤65 tokens and new/old ≤1.5×. At B-T0, worker CPU ≤1.0 ms per decision, else in order: PRIV top-k from the WP-ENG-1 getter (already the design, so a remedy only if WP-ENG-1 slips to W3; priv tokens are 17.5% of worker CPU [CM]), SUMMARY caching, map cone ≤30 | laptop, Mon |
| X-B | precision | 50K decisions, fp32 eager vs bf16 compiled: bf16 iff p99 \|Δlogp\| < 0.05 and KL < 1e-3, else fp32 heads | laptop GPU, 30 min |
| Learning check | KS decay may start | §7 | laptop; box |
| B-T0 | net size, workers | {6×256, 4×256} × {32, 40} workers, 7 min each. 6×256 iff its 3-GPU update-only rate (E=2, pre-pass, compiled) ≥12k decisions/s (≈11k realised at S1a); workers: best end-to-end rate at staleness ≤1. S0's end-to-end rate is server-bound and cannot separate sizes | a4000, ~35 min, right after T0 deploy |
| E=1 switch | one epoch or two | At the first restart after S1b passes parity, switch iff the learner waited on the update ≥25% of the last hour and a 20-min smoke keeps KL ≤0.03 and clipfrac ≤0.3. Revert from the pre-switch pinned state iff (a) KL > 0.03 or clipfrac > 0.3 for 20 iterations, or (b) at +300M, paired DEV-1000 (EMA0.996) is ≥3.0pp below pre-switch with exact McNemar p<0.05 (P(revert) 0.01-0.02 at a true 0) | smoke + routine DEV-1000 |
| Z-reads | P5, minibatch, audits | Z1: raw vs EMA0.996 vs EMA0.999 at each DEV-2000. Z2: gradient noise scale >4× the minibatch twice → minibatch 16,384 at the next restart [LIT]. Z3: V vs outcome per practice cell; act-3 expected calibration error > 0.1 at 1B → audit bootstrap rows. Z4: death funnel | 0 |
| DEVBOSS vs REF | diagnosis, follow-up design | descriptive; cluster bootstrap over states | laptop ~20 proc-h, Tue-Wed; student at each 1B |

**Moved to §13** (≥1B arms): the act-3 curriculum (MDE80 3.1-5.3pp at this week's arm size, vs 0.45-2pp of lineage gain), MCTS boss distillation (3.9-5.0pp on DEVBOSS, plus a labeller), E=1 vs E=2 (|expected Δ| ≤0.75pp, SE ≥0.71pp), per-head KS cuts, warm start vs scratch. Not planned: wider nets (7.2× width was null [LES]), critic splits, reward and hyperparameter sweeps, phasic policy gradient, the actor-critic decoupling, HL-Gauss (histogram-loss) value and reset changes that screened null [LES][LIT], human cloning, deployment search.

**No `ppo.screen` first:** it compares run policies under MCTS combat and has no end-to-end arm; the learning check, K2 and K3 are the screens.

## 10. Timeline

### 10.1 Work packages

| WP | Scope | Effort (est.) | Due |
|---|---|---|---|
| WP0 | worktree, merge, W1/W2, gate blocks, PREREG-0, GH issue, handover | 0.5 d + ongoing | A+1.5 h |
| WP1 | factory, AST guard, 19 sites, static tables, F8, stamps | 0.75 d | Mon 23:30 |
| WP2 | env, dedupe, caps, rewards, logs, pre-flight; practice | 1.25 d | core Tue 01:30; rest 07:30 |
| WP3 | SCHEMA.md and §4 inputs; X-C, F7, X-T | 1.75 d | schema Mon 21:30; rest Tue 05:30 |
| WP4 | net, learner (§6-§7), resume; T1-T3, T7-T12 | 2.0 d | skeleton Tue 01:30; rest 07:30 |
| WP5 | buckets, compile, X-B, B-T0; then S1a | 0.5 + 0.5 d | Tue 08:00; Wed 10:00 |
| WP5b | S1b components and E1a | ~3 d [CM] | A → Thu 10:00 |
| WP6 | `e2e_arm.sh`, `ppo.e2e.evaluate --sealed`, `score_test.py`, panels, DEVBOSS, REF | 1 d | runner A+2 h; rest Tue 05:30 |
| WP7 | §8 ops, RUNBOOK.md | 0.75 d | Tue 09:30 |
| WP8 | F2-F6, F9 checker | 0.5 d | A+10 h |
| WP-ENG-1 | §3.4 getters, PR to main; tests: telegraph = the wire's head intent on 1,000 states; `peek_intents` keeps the state hash | 0.5 d | Mon 23:30 |
| WP9 | PREREG-0/-1, FIDELITY, BASELINES, RESULTS | 0.5 d | A+1.5 h; Tue 05:30; Nov 10 |

- **Acceptance:** WP1 AST guard (red on an injected call) and F8; WP2 pre-flight, T4-T6, T13; WP3 X-C, F7, T14, X-T, stable vocab sha256; WP4 T1-T3, T7-T12; WP5/5b parity; WP6 repeatable rows, silent `--sealed`; WP7 failure-path tests; WP-ENG-1 the gate block.
- **Critical path:** ~10 agent-days fall before G0, against 12-17 engineer-days in the KB's estimates [ENG][CPPO][V111][CM]. Cut segments, bootstrap rows, chained GAE and teacher masks are new code: #1081 trains only finished per-fight episodes. Expect G0 Tue 09:30-Wed 01:30 (P ≈ 0.25 / 0.5 / 0.75 by Tue 09:30 / Tue 17:30 / Wed 01:30 at A = Mon 09:00, est.).
- **Cut order if late** (each returns at a restart unless noted): practice → run-row hold-back (launch without it) → U0 teacher acting → compile (back at S1a) → tactical → content prior → dual-clip → MLP heads (back only as a zero-init residual). Dedupe is never cut (frozen at G-schema); a section cut before G-schema, such as SUMMARY, stays out. WP5 and WP5b are never raided.

### 10.2 Agent protocol

- **Step 0:** worktree agents branch from origin/main, which lacks `ppo/combat_ppo/`. Each runs `git fetch origin && git reset --hard <campaign SHA>`, pastes `git log -3 --oneline` into its first report and stops if the head is wrong. Never switch branches in the shared main checkout.
- **Hand-back:** commits on pushed `wp/<n>-<name>` branches; WP0 merges with `git merge --no-ff`, runs `just check` and the touched tests on the result (§10.3) and pushes. No cherry-picks; no rebase after PREREG-0. Changes to `ranks.py`, the learner loop, rewards or the factory get a second-agent `/code-review`.
- **Baselines** run only from the frozen worktree `~/projects/sts2-sim-e2e-w`: the first DEV runs at the W1 SHA, everything after W2 (Tue ~01:00) at the W2 SHA, which adds WP1's static tables. The W1 ≡ W2 check (§8) compares the two SHAs, so any difference re-runs the DEV baselines on W2.

### 10.3 Gate block (G0, WP-ENG-1's merge to main, Nov 10)

Run on the merge result and on origin/main; block only on new failures; never during a 20-proc baseline. WP merges into the campaign branch run only `just check` and the touched tests (T1-T14 and the new `tests/e2e/`), because that branch reaches main only on Nov 10, behind a full block.

```text
cd sts2-rs
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo nextest run --workspace
cargo test --workspace --doc
cargo check --manifest-path fuzz/Cargo.toml   # if a fuzz-called signature changed
cd ../sts2-pycli
just build-cs-all
just wasm-check
just check                                    # format, lint, ty, deptry, check-champions, fast tests
uv run --frozen pytest tests/ --game-version v0.111.0
just test-slow
just regression-test
cd .. && just check-public && just test-public-shape
```

New exported files: no hard-coded home directory; tests needing AlphaDefect's runs, the decomp, teacher `.pt` files or CUDA get `_NEEDS` entries; text `open` names `encoding=`; code stays spawn-safe for the public Windows CI.

### 10.4 Schedule

| When | Event → gate or fallback |
|---|---|
| A to A+1.5 h (Mon 09:00-10:30) | WP0: worktrees `~/projects/sts2-sim-e2e` and the frozen `~/projects/sts2-sim-e2e-w`, branch `exp/e2e-ironclad-v111` off origin/main 917900a5f, `git merge --no-ff origin/exp/combat-ppo-screen`; `just build-cs-all`; W1; GH issue; copies (D1); **PREREG-0** pushed; draft PR (D10) |
| A+1.5 h to A+10 h (Mon 10:30-19:00) | `just check` on the merge; from A+2 h, DEV baselines on W1 at 20 procs (R2, provisional R3, R1) → provisional teacher; F2-F6 at 16 procs from A+4 h |
| Mon 21:30 | **G-schema:** sections, columns, action kinds and reserved ids, vocab, dedupe frozen |
| Mon 23:30-Tue 01:00 | WP-ENG-1 and F fixes merged, or they wait; W2 build and pin (glibc read at A); Rust freeze; W1 ≡ W2, and F3 re-runs on W2 if they differ |
| Tue 01:30 | **G-skeleton:** one A10 v0.111 run end to end on the laptop (env → tokens → net → cut GAE → one step, finite losses). Miss → apply the cut order now |
| Tue 05:30 | PREREG-1 (tighten-only, with `score_test.py`) |
| Tue 07:30-09:30 | H3 read (GO iff H3 − H2 ≥ +2pp, CI low end > 0); laptop: final R3 DEV-2000 → teacher and C* committed by 09:30 |
| Tue 09:30 (target) | **G0:** gate block; T1-T14; a 30-min smoke (losses, surfaces, pre-flight, F7, F8, resume); learning check. P ≈ 0.25 / 0.5 / 0.75 by Tue 09:30 / 17:30 / Wed 01:30 (est.); until then the #1081 session keeps the box |
| G0 and T0 both done, then ~2.5 h | a4000: deploy W2; cross-host identity; B-T0; learning check; 45-min launch smoke with kill and resume |
| median Tue ~20:00 (est.) | **G1:** launch iff the smoke is clean and F1, F3, F7, F8, F9 pass; a failed learning check still launches, with λ_KS held at 1 (§7). Else the box goes back to the #1081 session until a smoke passes |
| Tue 10:00-16:00 | laptop: TEST-4000 for R1, R2, R3 (sealed); then DEVBOSS and REF |
| ~100M (~4-5 h after launch) | **K2** (§7) |
| Wed ~10:00; Thu ~10:00 | S1a; **S2** (E1a by rule); S1b parts as each passes |
| 1.4B (Thu ~06:00-12:00; backstop Thu 18:00) | **K3:** NO-GO iff paired DEV-1000 student − C* ≤ −3pp → HUMAN, diagnose ≤6 h (training continues), relaunch only for a found bug |
| every 250M; every 1B | DEV-1000; DEV-2000 × {raw, EMA0.996, EMA0.999}, DEVBOSS, projection refit |
| 3B (Fri ~01:00-08:00; backstop Sat 08:00) | **K4:** NO-GO iff paired DEV-2000 student < C* → as K3 |
| Sun Nov 8 06:00 | lr anneal to 0.3× starts; KL factor freezes |
| Mon Nov 9 06:00-09:00 | stop; P5 selection (9 candidates, ~15 min); Final TEST-4000 (sealed); `score_test.py` once |
| Tue Nov 10 | merge main in (never rebase); gate block; results PR (D10) via `gh pr merge --merge`; `git merge-base --is-ancestor <PREREG-0 sha> origin/main` into RESULTS.md |

**T0 handover** (the #1081 session, else the operator with your OK): post on PR #1081 and message that session; after the H3 read, stop the H2/H3 watchdogs, SIGINT H2 and H3 (resumable) and wait on their process IDs; if the main run was resumed, checkpoint it and let it exit (a SIGSTOPped process keeps its GPU memory); copy both runs' final states, cfg, wheel record and stats to `~/projects/sts2/bench-archive/2026-11-03-combat-ppo-h2-h3/` with a sha256 manifest (teachers load only from it); confirm an empty `nvidia-smi`, ≥80 GB RAM and ≥75 GB disk free; log it; deploy.

Compute stages run back to back once you approve; gates are automatic reads, not pauses.

## 11. Evaluation (WP6)

| Panel | Seeds | Use |
|---|---|---|
| DEV-1000 | 160,000,000-160,000,999 | every 250M (EMA0.996); K2, K3; E=1 revert |
| DEV-2000 | 160,000,000-160,001,999 | teacher, C*, K4, every 1B, P5 |
| TEST (sealed) | 161,000,000-161,003,999 | R1, R2, R3, Final; read once |
| DEVBOSS | 175,000,000+; reseeds 175,500,000 + 4i + j | 600 boss-entry states (200 per act-1, act-2, act-3 double boss) from teacher-acted runs × 4 shared reseeds; REF = shipped MCTS on the same pairs |

- **Settings:** greedy argmax over merged options on W2 (the first DEV baselines on W1, §8) with the training env flags (swap OFF for R1), `--act1 roll`, A10; a no-search DEV-1000 costs ~0.5-0.8M decisions.
- **Never compare rates across panels** (one net read 43.1% vs 47.2% on two [LES]); training WR is liveness only; expect 1-2pp of DEV optimism [AD].
- **Report:** WR with Wilson CI, A9-est, deaths by act and boss, per-boss conversion, boss-entry HP, potions, decisions per run, caps.

**PREREG-0** (`docs/experiments/2026-11-03-ironclad-e2e-v111/PREREG.md`) is pushed before the first baseline run; amendments only tighten rules whose data does not exist yet, one commit each.
- **P1 Panels and seeds:** above and §3.1.
- **P2 Bars (§1.5):** Δ is over seeds present in both files. An errored seed re-runs once on the same binary; then a baseline error drops it from that contrast and a Final-only error is a Final loss. Over 20 such seeds (0.5%) make the contrast INVALID; ≠4,000 rows or an off-panel seed stops the read. Descriptive family (Holm, α = 0.05): Final − R2, Final − R3, A9-est Final − R1, DEVBOSS Final − REF. K-gate failures do not cancel the TEST read.
- **P3 Discordance:** d = 0.23-0.29 (Bar1), ≈0.30 vs R1 (0.71-0.90 × independence on 8 measured pairs); the read reports the observed d.
- **P4 Sealing:** TEST runs only through `e2e_arm.sh` or `ppo.e2e.evaluate --sealed`, which print counts and the sha256, committed per job. Nobody prints, greps or loads `evals/TEST/` before Mon Nov 9, 09:00, and only after the P5 pick, C* and every sha256 are committed; liveness uses `ls` and `wc -l`. `score_test.py` is the only reader, once. Determinism checks use smoke seeds.
- **P5 Selection** (DEV-2000): {M3 = the stop checkpoint; M2, M1 = the two 1B milestones before it} × {raw, EMA0.996, EMA0.999}; default M3-EMA0.996; switch only for ≥0.5pp paired (simulated regret 0.55pp at 0.5pp vs 0.96pp at 1.5pp).
- **P6 Rules:** §9 rules, K2 (§7), K3 and K4 (§10.4), all paired; D6 (P(continue) = 0.03 / 0.50 / 0.97 at a true 0 / 2 / 4pp per 1B).
- **P7 Forecast:** §1.3, scored after the read, with the 1B refits.

**Scorer** `ppo/e2e/score_test.py` (committed with PREREG-1; run as `PYTHONPATH=python <W2 python> -m ppo.e2e.score_test --prereg <PREREG.md>`): exact two-sided McNemar; Newcombe method-10 paired CI via `newcombe_paired` (docs/experiments/2026-10-26-ironclad-release-winrates/analysis/scripts/analyze.py:145), cross-checked by a paired bootstrap (20,000 resamples, `default_rng(20261109)`); Wilson; Holm; A9-est; validated on the cont300 gate files and synthetic panels. `ppo.ab_compare` stays a quick look: no CI, it silently drops unmatched seeds, and its p<0.05 verdict matches no bar.

## 12. Risks

- **Code misses T0** (P(G0 by Tue 08:00) ≈ 0.25 at A = Mon 09:00, est.): the cut order; the #1081 session keeps the box; ~0.35B lost per 10 h.
- **The teacher hides a broken PPO path** (learning check fails, or DEV flattens once λ_KS < 0.3): hold λ_KS at 1 until fixed.
- **Low throughput** (B-T0 update-only <12k/s, workers >1.0 ms per decision, ±20% model error [CM]): 4×256, the X-T remedies, half the eval cadence. S1b late: retry at each 25M; E=2 all week still gives 6.1B.
- **Sim exploit deep in A10** (DEV jumps >5pp, F9 hits, odd card or relic use): audit top runs against C#, patch, W3, resume from the last clean pinned state.
- **Stall** (P(<15%) ≈ 0.27; §1.4 thresholds, K3, K4): HUMAN; diagnose ≤6 h; relaunch only on a found bug; TEST is read regardless.
- **Host mismatch, laptop memory or disk:** §8's identity rules, `MemoryMax=24G` shards that skip if done, pruning.

## 13. After the week

- **Exporter:** a new `ppo/e2e/export_onnx.py`, because `export_onnx.py:64` builds `RunPolicyNet()` at defaults and `net_policy.py:22` hard-codes dim 192 (verified). Torch, the Open Neural Network Exchange (ONNX) export and tract (the Rust ONNX runtime) must agree on 10K decisions at |Δlogit| < 1e-3.
- **Deployment, only if Bar3 passes:** an `[ironclad_e2e]` lane in `champions.toml` (with `schema_sha` and `game_version`) via `just promote-champion`, per playbook §2 on a fresh panel; then a flag-gated no-search path in `realgame_advisor` and `realgame_ppo`.
- **Real-game gaps** on their reserved ids, by frequency in AlphaDefect's runs [ENG]: #1069 (275 resyncs per 1,000 runs), Fake Merchant multi-buy (53), treasure, OOC potions (38), Crystal Sphere.
- **Powered recipe experiments:** ≥1B decisions per arm from a shared checkpoint, each on its own 4,000-seed panel under a PREREG with magnitudes (at ≥1B, 10-30% efficiency gaps become 1-3pp, near the ~2pp MDE80). Order: (1) a continuation control, the v0.107 teacher pair continued on v0.111 at matched decisions (Bar1 attribution); (2) MCTS boss distillation (visit targets as an auxiliary KL on boss rows); (3) an act-3 curriculum (50% practice, 65% double boss, start weighting 4p(1−p) [LIT]); (4) warm start vs scratch; (5) E=1 vs E=2 at matched wall clock; (6) per-head KS cuts. Then a Rust tokenizer, no pre-pass (~43k/s), longer training (D6).
- **Stable v0.111.x:** after a Steam update, the next session decompiles, diffs, ports, rebuilds, runs F2-F5, re-measures R1 and the student on 163,000,000-163,003,999, plays real games through the mod, and relaxes the mod's strict game-build check (#807) to accept v0.111.x.
- **Tracking:** the campaign's GH issue (opened at A; project #2, linked to #802 "Train v0.111.0 champions" and #1081) gets a comment at G0, T0, K2-K4 and the read. Checkpoints, stats, sealed TEST rows and the W2 .so go to `~/projects/sts2/bench-archive/2026-11-03-e2e-ironclad-v111/` with a sha256 manifest; git gets the docs, summaries and hashes. RESULTS.md records the bars with the PREREG-0 SHA, every gate and switch verdict, deviations and negative results; then a memory note.

## Appendix A. Terms

Times are CET. A bare weekday means the first one after Sun Nov 1 (Mon = Nov 2, Tue = Nov 3, ..., Sun = Nov 8); Mon Nov 9 always carries its date. A: the time you approve this plan; the schedule assumes A = Mon 09:00. A9-est: the A9 rate estimated from A10 runs (1-4pp below true A9). Decision: a stored policy step with ≥2 merged options; forced steps auto-advance. GAE, KL, EV, EMA, IS: generalized advantage estimation, Kullback-Leibler divergence, explained variance, exponential moving average of weights, importance sampling. SE, MSE: standard error, mean squared error. lr: learning rate. bf16, fp32: 16-bit (bfloat) and 32-bit floats. MLP: multilayer perceptron. AST: abstract syntax tree. RNG: random number generator. HP: hit points. GH: GitHub. π_behav, π_prox, π_θ: the sampling, update-start (pre-pass) and current policies; π_T, π_S: teacher and student. E, E1a: epochs per batch; one epoch after a pre-pass. KS, λ_KS, β_act: the kickstarting term KL(teacher‖student), its weight, the teacher-acting probability. V-warm, U0-U2: warm-start phases (§7). S_ctx: #1081's combat-context tokens. RUNCTX, BOSS, SUMMARY, MAP, PRIV: token sections (§4); PRIV is privileged and value-only. OOC: out-of-combat. R1-R3, C*: baselines (§2). Final: the student checkpoint P5 picks (§11). H2, H3: #1081's running hybrid lanes, each a combat net trained with its own run net; H3 (hybrid_v) scores each fight by the change in the run critic's value; GO: H3's pre-registered win over H2 (§10.4). DEV-1000, DEV-2000, TEST, DEVBOSS: evaluation panels; REF: the MCTS reference on DEVBOSS (§11). S0-S2, L1: throughput steps and GPU layout (§8). W1-W3: engine wheels (§8). WP: work package (§10). B-T0, X-T, X-B, X-C: bench and checks (§9). F1-F9: fidelity gates (§3.5). G-schema, G-skeleton, G0, G1, K2-K4: launch and progress gates (§10). PREREG-0, P1-P7: the pre-registration and its sections; PREREG-1: its tighten-only amendment (§11). HUMAN: an alarm that stops and waits for you (§8). MDE80: the smallest effect detectable with 80% power.
