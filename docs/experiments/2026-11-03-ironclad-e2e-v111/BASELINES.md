# Baselines on v0.111.0 (Ironclad A10, #1097)

Agents are defined in PREREG-0 (`PREREG.md`, Agents table and amendment 1). Every row runs through `sts2-pycli/scripts/e2e_arm.sh` from the frozen worktree, greedy, `--act1 roll`, A10, potion swap OFF. Outputs live in `~/runs/e2e/evals/<PANEL>/<ARM>/`; each arm's sha256 manifest is in `sha256/`.

## DEV on W1

W1: PGO wheel, src_hash `acc813aa12f20ad2`, profdata sha256 `b9c308a93a9c…`, pinned at `~/runs/e2e-pin/cca32e15edf6bc5e` (`.so` sha256 `cca32e15…9d90`). Code at e0185edde.

| Arm | Agent | Panel | Wins / n | WR (Wilson 95%) | Errors | Decisions/run | Wall |
|---|---|---|---|---|---|---|---|
| R2 | cont300_it1150 + combat-PPO it06000, no search | DEV-2000 | 182 / 2000 | 9.10% (7.92-10.44%) | 0 | 56.1 | 01:31-01:37 |
| R3p | H2 v00392 + it02500 (mismatched pair; descriptive only, amendment 1) | DEV-2000 | 227 / 2000 | 11.35% (10.03-12.81%) | 0 | 57.6 | 01:37-01:44 |
| R3m | H2 v00375 + it02500 (matched; the provisional R3) | DEV-2000 | 222 / 2000 | 11.10% (9.80-12.55%) | 0 | 57.6 | 01:39-02:00 (8 procs, nice 5) |
| R1 | cont300_it1150 + shipped MCTS 1000/4000 | DEV-1000 | 233 / 1000 | 23.30% (20.79-26.02%) | 0 | 62.6 | 01:44-02:34 (57.9 s/run = 16.1 proc-h per 1,000) |

Paired contrasts on the same 2,000 DEV seeds (exact two-sided McNemar on the discordant seeds):

| Contrast | Seeds won only by the first | Seeds won only by the second | Difference | p |
|---|---|---|---|---|
| R3m − R2 | 174 | 134 | +2.00pp | 0.026 |
| R3p − R3m | 136 | 131 | +0.25pp | 0.81 |
| R1 − R2 (first 1,000 seeds) | 176 | 37 | +13.90pp | 7e-23 |
| R1 − R3m (first 1,000 seeds) | 174 | 52 | +12.20pp | 1e-16 |

**Provisional teacher: R3m** (H2 `run_ckpt_v00375.pt` + `it02500.pt`). The rule picks the better of R2 and R3 on DEV-2000, and R3 when they're within 2pp; R3m is ahead by 2.0pp, so the rule picks it either way. The pairing question doesn't move the number: the mismatched R3p is within 0.25pp of R3m. The final teacher is re-picked with the same rule from #1081's final matched pairs (`it04000`) after its 750M read, on W2.

Manifests (sha256 of each arm's `<PANEL>/<ARM>.sha256`): R2 `db41e596…a945`, R3p `547d4d45…96f4`, R3m `49676ad1…bcab`, R1 `be71a62b…0db2`.

R1 lands at the top of its forecast band (17-24%). On the 1,000 seeds all four arms share, R3m wins 111 and R2 94, so search is worth about 12pp over the best no-search pair here. R1's 16.1 proc-h per 1,000 runs puts TEST-4000 for R1 at ~64 proc-h, ~3.2 h at 20 procs. Every row records `so_sha256=9ab00690…` in `meta/config.txt`; that's the digest of the pin's sorted `.so` hashes, i.e. of the one `.so` `cca32e15…9d90`.

## DEV on W2

W2: PGO wheel, src_hash `5095b76e7b185ca0`, profdata sha256 `a28de557bd23…`, build_git 7c4c6acc0, pinned at `~/runs/e2e-pin/d83a29f6d5433be5` (`.so` sha256 `d83a29f6d543…06e1ec`). The W1 ≡ W2 identity check found differences (R2: 5 of 200 SMOKE seeds, R1: 4 of 50), so every DEV baseline re-ran on W2 from the frozen worktree at 7c4c6acc0.

| Arm | Agent | Panel | Wins / n | WR (Wilson 95%) | Errors | Decisions/run | Wall |
|---|---|---|---|---|---|---|---|
| R2_W2 | cont300_it1150 + combat-PPO it06000, no search | DEV-2000 | 179 / 2000 | 8.95% (7.78-10.28%) | 0 | 56.2 | 07:19-07:25 |
| R3m_W2 | H2 v00375 + it02500 (the provisional R3) | DEV-2000 | 215 / 2000 | 10.75% (9.47-12.18%) | 0 | 57.5 | 07:25-07:31 |
| R1_W2 | cont300_it1150 + shipped MCTS 1000/4000 | DEV-1000 | 237 / 1000 | 23.70% (21.17-26.43%) | 0 | 62.7 | 07:31-08:07 (42.3 s/run) |
| **R3_W2** | **final R3: H2 `run/ckpt/at_it04000.pt` + `ckpt/it04000.pt` (amendment 2)** | DEV-2000 | **237 / 2000** | **11.85% (10.51-13.34%)** | 0 | 58.0 | 10:31-11:09 (laptop shared, 19.5 s/run) |

Paired contrasts on the same DEV seeds (exact two-sided McNemar on the discordant seeds; Newcombe paired 95% CI):

| Contrast | Seeds won only by the first | Seeds won only by the second | Difference | 95% CI | p |
|---|---|---|---|---|---|
| R3_W2 − R2_W2 | 168 | 110 | +2.90pp | [+1.27, +4.54] | 0.0006 |
| R3_W2 − R3m_W2 | 169 | 147 | +1.10pp | [−0.64, +2.85] | 0.24 |
| R3m_W2 − R2_W2 | 170 | 134 | +1.80pp | [+0.09, +3.52] | 0.045 |
| R1_W2 − R3_W2 (first 1,000 seeds) | 178 | 58 | +12.00pp | [+9.08, +14.93] | 2e-15 |
| R1_W2 − R2_W2 (first 1,000 seeds) | 180 | 36 | +14.40pp | [+11.67, +17.16] | 3e-24 |

W1 against W2 on the same seeds: R2 plays 1,925 of 2,000 seeds identically (same final state and step count) and moves −0.15pp (p = 0.61); R3m plays 1,903 identically and moves −0.35pp (p = 0.065).

**Teacher: R3** (H2 `hybrid/run/ckpt/at_it04000.pt`, sha256 `7c05b482…2417`, + `hybrid/ckpt/it04000.pt`, `feaae909…6b2a`; archive `~/projects/sts2/bench-archive/2026-11-03-combat-ppo-h2-h3/`, `SHA256SUMS`). The rule picks the better of R2 and R3 on DEV-2000, and R3 when they're within 2pp; R3 leads R2 by 2.90pp, so it's R3 either way. The run net carries `cfg.features = "card_mods,reward_upgrade"`.

**C\* = R3** at 11.85% on DEV-2000, the higher of R2 (8.95%) and the final R3. Bar1, K3 and K4 compare against R3's rows.

Manifests (sha256 of each arm's `DEV/<ARM>.sha256`): R2_W2 `900408fc…8761`, R3m_W2 `0ee0d56a…57bb`, R1_W2 `6b7e25df…4a81`, R3_W2 `21c4344d…be93`.

**Game version.** Every panel here runs v0.111.0 (`e2e_arm.sh` sets `STS2_GAME_VERSION`). #1081 trains and evaluates every net in its line on its default build, v0.107.1 (no `STS2_GAME_VERSION` in its learner or eval environments; confirmed by #1081 at ~11:10). So R2 and R3 are v0.107.1-trained nets scored on v0.111.0, and #1081's own numbers aren't comparable with these: it reads the R3m pair at 15.4% on its v0.107.1 panel (seeds 154,000,000-999). The R1 vs R3 comparison is unaffected, because both arms run v0.111.0. A diagnostic on #1081's seeds measures the version effect:

| R3m pair on seeds 154,000,000-999 (n = 1000 each) | Wins | WR |
|---|---|---|
| W2, v0.107.1 (`~/runs/e2e/diag-1081/`) | 153 | 15.30% |
| #1081's own read600 H2 file, its wheel, v0.107.1 | 154 | 15.40% |
| W2, v0.111.0 | 106 | 10.60% |

- On v0.107.1, W2 matches #1081's file: −0.10pp [−1.79, +1.59], p = 1, with 604 of 1,000 seeds played identically. The wheel isn't a factor.
- The version effect on the same seeds is v0.111.0 − v0.107.1 = −4.70pp [−7.69, −1.72], exact McNemar p = 0.0025.
- On v0.111.0, #1081's seeds give 10.60% and our DEV seeds 10.75%. The seed range isn't a factor.

So these nets lose about a third of their wins on the version they're scored on here. The student trains on v0.111.0; its teacher and C* never did.

## TEST on W2 (sealed)

Only row counts, error counts and sha256 manifests appear here before the Mon Nov 9 read.

| Arm | Seeds | Rows | Errors after retry | Manifest (sha256 of `TEST/<ARM>.sha256`) | Wall |
|---|---|---|---|---|---|
| R2 | 161,000,000-161,003,999 | 4000 | 0 | `8d236b8c…fe13` | 08:10-08:21 |
| R1 | 161,000,000-161,003,999 | 4000 | 0 | `1a862377…e7fa` | 08:21-11:18 (nice 19 from 10:31) |
| R3 | 161,000,000-161,003,999 | 4000 | 0 | `2b532d9f…09ec` | 11:18-11:34 |

All three sealed baselines ran on W2 from the frozen worktree at 7c4c6acc0; R3 is the final pair (`at_it04000` + `it04000`), checked against the archive's `SHA256SUMS` before the run.

## DEV on W4

W4 (PREREG amendment 6): PGO wheel, src_hash `d0ef75046cf7cb85`, profdata sha256 `9507beadddac…`, build_git 0297d646a, pinned at `~/runs/e2e-pin/87de4d1a0bbffeeb` (`.so` sha256 `87de4d1a0bbf…3bc5e5`, the same on the a4000 and the laptop, F1). Every arm ran on the laptop from the frozen worktree at 2b9f06510, whose Rust is the build SHA's; `meta/config.txt` records `so_sha256=aadf9a0b…`. F3 (qx, fresh) and the F9 soak exit 0 (`~/runs/e2e/fidelity-w4`).

The W3 ≡ W4 SMOKE identity check finds differences: R2 154 of 200 seeds identical, R1 2 of 50. R1's search lines move almost everywhere because #1164's clone flag changes search hashes. So every baseline re-runs on W4.

| Arm | Agent | Panel | Wins / n | WR (Wilson 95%) | Errors | Decisions/run | Wall |
|---|---|---|---|---|---|---|---|
| R2_W4 | cont300_it1150 + combat-PPO it06000, no search | DEV-2000 | 157 / 2000 | 7.85% (6.75-9.11%) | 0 | 56.1 | 22:21-22:26 |
| R3_W4 | final R3: H2 `run/ckpt/at_it04000.pt` + `ckpt/it04000.pt` | DEV-2000 | 245 / 2000 | 12.25% (10.89-13.76%) | 0 | 58.1 | 22:26-22:32 |
| R1_W4 | cont300_it1150 + shipped MCTS 1000/4000 | DEV-1000 | 246 / 1000 | 24.60% (22.03-27.36%) | 0 | 61.9 | 22:32-23:11 (46.8 s/run) |

Paired contrasts on the same DEV seeds (exact two-sided McNemar on the discordant seeds; Newcombe paired 95% CI):

| Contrast | Seeds won only by the first | Seeds won only by the second | Difference | 95% CI | p |
|---|---|---|---|---|---|
| R3_W4 − R2_W4 | 194 | 106 | +4.40pp | [+2.72, +6.10] | 4e-07 |
| R1_W4 − R3_W4 (first 1,000 seeds) | 185 | 59 | +12.60pp | [+9.64, +15.57] | 3e-16 |
| R1_W4 − R2_W4 (first 1,000 seeds) | 201 | 32 | +16.90pp | [+14.11, +19.73] | 4e-31 |

W3 against W4 on the same seeds: R3 plays 1,348 of 2,000 seeds identically and moves +0.05pp (43 / 42, p = 1.0); R2 plays 1,491 identically and moves −0.70pp (26 / 40, p = 0.11); R1 plays 17 of 1,000 identically and moves +0.90pp (115 / 106, p = 0.59).

**C\*_W4 = R3** at 12.25% on DEV-2000. The rule picks the better of R2 and R3 on DEV-2000, and R3 when they're within 2pp. R3 leads R2 by 4.40pp, so it's R3 either way. The teacher is unchanged.

Manifests (sha256 of each arm's `DEV/<ARM>.sha256`, copied to `sha256/DEV/`): R2_W4 `1e0fbabc…1dee`, R3_W4 `0bec7e8c…02da`, R1_W4 `23717b0f…f892`.

## TEST on W4 (sealed)

Only row counts, error counts and sha256 manifests appear here before the read.

| Arm | Seeds | Rows | Errors after retry | Manifest (sha256 of `TEST/<ARM>.sha256`) | Wall |
|---|---|---|---|---|---|
| R2_W4 | 161,000,000-161,003,999 | 4000 | 0 | `b89ed830…0733` | 23:11-23:21 |
| R1_W4 | 161,000,000-161,003,999 | 4000 | 0 | `5ce1fa5b…8205` | 23:21-02:28 |
| R3_W4 | 161,000,000-161,003,999 | 4000 | 0 | `71913787…f7f5` | 02:28-02:46 |

All three ran on W4 from the frozen worktree at 2b9f06510, on the laptop, Nov 7-8. The manifests are in `sha256/TEST/`.
