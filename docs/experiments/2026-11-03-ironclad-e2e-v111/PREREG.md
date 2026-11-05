# PREREG-0: Ironclad end-to-end no-search PPO on v0.111.0 (#1097)

**Written** 2026-11-02 ~01:45 CET, before any baseline, training or TEST run. **Plan:** `docs/plans/2026-11-02-ironclad-e2e-ppo-v0111.md` (§ numbers below refer to it). **Amendments** only tighten a rule whose data doesn't exist yet, one commit each; PREREG-1 (due Tue Nov 3, 05:30) adds `score_test.py` and may only tighten. Nothing here is changed after its data exists.

## Objective

The primary objective is the greedy A10 win rate (WR) of **Final** on the sealed TEST panel. Final is the student checkpoint that P5 picks. Bars are reported, not optimized. Every A10 number is reported with its A9-est (the A9 rate estimated from the same A10 runs, as in the 2026-10-31 cont300 gate).

Box time goes to training decisions by default. The only other a4000 uses are the B-T0 bench (~35 min; it picks the net size and worker count, §9), the cross-host identity check and the learning check (~15 min), the launch smoke (45 min) and evals (≤5% of wall clock). Each one prevents a launch on a broken, slower or non-identical stack, and none can cost more than ~0.05B decisions (calc).

## P1 Panels and seeds

All seeds are below 2^31. A re-grep of the repo on 2026-11-02 found no earlier use of these ranges.

| Panel | Seeds | Use |
|---|---|---|
| DEV-1000 | 160,000,000-160,000,999 | every 250M (EMA0.996); K2, K3; the E=1 revert rule |
| DEV-2000 | 160,000,000-160,001,999 | teacher, C*, K4, every 1B, P5 |
| TEST (sealed) | 161,000,000-161,003,999 | R1, R2, R3, Final; read once |
| SMOKE | 162,000,000-162,000,999 | probe smoke, W1 ≡ W2 and cross-host identity, determinism |
| DEVBOSS | 175,000,000+; reseeds 175,500,000 + 4i + j | 600 boss-entry states (200 each at the act-1 boss, act-2 boss and act-3 double boss) from teacher-acted runs × 4 shared reseeds; REF = shipped MCTS on the same pairs |
| Practice reseeds | 600,000,000-639,999,999 | per-worker PCG64 |
| Student trainer runs | from 1,000,000,000 (< 1,260,000,000) | #1081's slot layout: base + 7·(start iteration) + (worker·64 + env)·100,000 + run count |
| Throwaway trainer runs | from 1,300,000,000 (< 1,560,000,000) | smokes, learning checks, B-T0; may share seeds, since none feeds a panel |

Every panel run uses `--act1 roll`, A10, Ironclad, v0.111.0, greedy play, and one engine binary per comparison (W1 for the first DEV baselines, W2 for everything else, §8).

## Agents

| ID | Agent | Settings | Panels |
|---|---|---|---|
| R1 | `ppo_fullrun_ironclad_cont300_it1150.pt` (sha256 `de15f4ba…e4af`) + shipped MCTS, 1000 sims / 4000 boss sims | `net_greedy`, potion swap OFF (as served), `--run-timeout 14400` | DEV-1000, TEST |
| R2 | the same run net + #1081 combat-PPO `run/it06000.pt` (sha256 `cc925b03…6724`), no search | `net_greedy`, `--combat policy`, swap OFF (how #1081 trained its combat nets and how the run net is served), `--run-timeout 3600` | DEV-2000, TEST |
| R3 | #1081's H2 pair: run net `hybrid/v*.pt` + combat net `hybrid/it*.pt`; H3's pair instead if H3 reads GO | as R2 | DEV-2000, TEST |
| Teacher | the better of R2 and R3 on DEV-2000; within 2pp, R3 | | |
| C* | the higher DEV-2000 WR of R2 and the final R3 | | |
| Final | the student checkpoint P5 picks | swap ON (its training setting) | DEV-2000, TEST |

- **Provisional R3** is H2's newest checkpoint on Mon Nov 2: `hybrid/v00392.pt` (sha256 `668049c3…0bf1`) + `hybrid/it02500.pt` (`4e1b49c9…eb09`). The **final R3** uses the H2 or H3 states copied at the T0 handover (`~/projects/sts2/bench-archive/2026-11-03-combat-ppo-h2-h3/`, sha256 manifest). Its DEV-2000 runs Tue Nov 3, 07:30-09:30, and the teacher and C* are committed by 09:30.
- **H3 GO** is #1081's pre-registered read: GO iff H3 − H2 ≥ +2pp with the CI's low end > 0.
- **Loaders** assert `ck["cfg"]["features"] == "card_mods,reward_upgrade"` for every run net. All three run nets above carry it (checked 2026-11-02).
- Full sha256 values live in `~/projects/sts2/bench-archive/2026-11-02-e2e-inputs/SHA256SUMS`.

**Commands** (frozen worktree `~/projects/sts2-sim-e2e-w/sts2-pycli`, pinned wheel `$E2E_PIN`, campaign venv python `$E2E_PY`):

```text
systemd-run --user --scope -p MemoryMax=24G scripts/e2e_arm.sh R1 DEV 1000 160000000 -- \
  --policy net_greedy --ckpt data/ppo/ppo_fullrun_ironclad_cont300_it1150.pt \
  --sims 1000 --boss-sims 4000 --run-timeout 14400 --act1 roll --ascension 10 --character Ironclad --procs 20
systemd-run --user --scope -p MemoryMax=24G scripts/e2e_arm.sh R2 DEV 2000 160000000 -- \
  --policy net_greedy --ckpt data/ppo/ppo_fullrun_ironclad_cont300_it1150.pt \
  --combat policy --combat-ckpt <inputs>/run/it06000.pt --run-timeout 3600 --act1 roll --ascension 10 --character Ironclad --procs 20
systemd-run --user --scope -p MemoryMax=24G scripts/e2e_arm.sh R3p DEV 2000 160000000 -- \
  --policy net_greedy --ckpt <inputs>/hybrid/v00392.pt \
  --combat policy --combat-ckpt <inputs>/hybrid/it02500.pt --run-timeout 3600 --act1 roll --ascension 10 --character Ironclad --procs 20
# TEST: PANEL=TEST N=4000 SEED_START=161000000, on W2; commit only the <arm>.sha256 manifest
```

## P2 Bars and statistics

| Bar | Contrast | Rule |
|---|---|---|
| Bar1 (must) | Final − C* | Δ ≥ +5pp and exact two-sided McNemar p < 0.01 |
| Bar2 (should) | Final − R1, as served | 95% Newcombe paired CI low end ≥ −3pp |
| Bar3 (stretch) | Final − R1 | Δ ≥ +3pp and exact two-sided McNemar p < 0.01 → deployment candidate (§13) |
| BarR (reported) | WR(Final) | ≥ 40% (pro level); ≥ 56% also reported |

- **Sequence:** Bar1 → Bar2 → Bar3, each at its own level, stopping at the first failure (familywise error ≤ 0.025 one-sided).
- **Descriptive family** (Holm, α = 0.05): Final − R2, Final − R3, A9-est Final − R1, DEVBOSS Final − REF.
- **Pairing:** Δ is over seeds present in both files. An errored seed re-runs once on the same binary (`e2e_arm.sh` does this). Then a baseline error drops that seed from the contrast, and a Final-only error counts as a Final loss. More than 20 such seeds (0.5%) make the contrast INVALID. A file with ≠ 4,000 rows or an off-panel seed stops the read.
- **Methods:** exact two-sided McNemar on discordant pairs; Newcombe method-10 paired CI (`newcombe_paired`, `docs/experiments/2026-10-26-ironclad-release-winrates/analysis/scripts/analyze.py:145`), cross-checked by a paired bootstrap (20,000 resamples, `default_rng(20261109)`); Wilson CIs for single rates; Holm for the descriptive family.
- K-gate failures do not cancel the TEST read.

## P3 Discordance

The power figures in §1.5 assume d = 0.23-0.29 for Bar1 and d ≈ 0.30 against R1 (0.71-0.90 × independence on 8 measured pairs). The read reports the observed d.

## P4 Sealing

- TEST runs only through `scripts/e2e_arm.sh` or `ppo.e2e.evaluate --sealed`. Both print only row counts, error counts and the sha256 manifest, and each job's `<arm>.sha256` is committed under `sha256/` next to this file.
- TEST rows live in `~/runs/e2e/evals/TEST/` (mode 700), outside every worktree. This is a deviation from the plan's in-worktree `evals/`, so the frozen worktree stays clean.
- Nobody prints, greps or loads `evals/TEST/` before Mon Nov 9, 09:00, and only after the P5 pick, C* and every sha256 are committed. Liveness uses `ls` and `wc -l`. The runner itself reads only each row's `seed` and whether `outcome == "error"`, and drops the probe's stdout (which holds its win rate) for TEST.
- `score_test.py` is the only reader, once. Determinism checks use SMOKE seeds.

## P5 Selection

On DEV-2000, the candidates are {M3 = the stop checkpoint; M2, M1 = the two 1B milestones before it} × {raw, EMA0.996, EMA0.999}. The default is M3-EMA0.996. Switch only for a ≥ 0.5pp paired gain.

## P6 Rules

All reads are paired, greedy, on one binary; no gate reads a treatment-selected subset such as P(win | act 3).

- **Learning check** (laptop 45 min at G0; box 15 min at T0; again at K2 and each restart): random init, λ_KS = 0, no teacher, A10. Pass iff the mean floor of the last 10K finished runs beats the first 10K by ≥ 1.0, combat EV > 0.1, and no NaN. A fail still launches, with λ_KS held at 1; the λ_KS decay starts only once a check passes.
- **K2** (~100M decisions): student DEV-1000 ≥ teacher pair − 8pp, paired; agreement ≥ 85% combat and ≥ 75% run; finite value loss; EV within 0.1 of its 50M value. Fail → fix, relaunch from the last good pinned state.
- **K3** (1.4B): NO-GO iff paired DEV-1000 student − C* ≤ −3pp → HUMAN alarm; diagnose ≤ 6 h while training continues; relaunch only for a found bug.
- **K4** (3B): NO-GO iff paired DEV-2000 student < C* → as K3.
- **X-B** (precision): 50K decisions, fp32 eager vs bf16 compiled; bf16 iff p99 |Δlogp| < 0.05 and KL < 1e-3, else fp32 heads.
- **X-T** (tokenizer): new vs #1081's tokenizer on 20K states on one pinned performance core: mean ≤ 65 tokens and new/old time ≤ 1.5×. At B-T0, worker CPU ≤ 1.0 ms per decision, else in order: PRIV top-k from the WP-ENG-1 getter, SUMMARY caching, map cone ≤ 30.
- **X-C** (schema): combat-teacher logits bitwise equal through the new schema on recorded states, else the schema is not accepted.
- **B-T0** (net size, workers): {6×256, 4×256} × {32, 40} workers, 7 min each. 6×256 iff its 3-GPU update-only rate (E=2, pre-pass, compiled) ≥ 12k decisions/s; workers: the best end-to-end rate at staleness ≤ 1.
- **E=1 switch:** at the first restart after S1b passes parity, switch iff the learner waited on the update ≥ 25% of the last hour and a 20-min smoke keeps KL ≤ 0.03 and clipfrac ≤ 0.3. Revert from the pre-switch pinned state iff (a) KL > 0.03 or clipfrac > 0.3 for 20 iterations, or (b) at +300M, paired DEV-1000 (EMA0.996) is ≥ 3.0pp below pre-switch with exact McNemar p < 0.05.
- **Z-reads** (descriptive, no box time): Z1 raw vs EMA0.996 vs EMA0.999 at each DEV-2000; Z2 gradient noise scale > 4× the minibatch twice → minibatch 16,384 at the next restart; Z3 V vs outcome per practice cell, and act-3 expected calibration error > 0.1 at 1B → audit bootstrap rows; Z4 death funnel.
- **Stall signals** (§1.4): DEV-2000 below 11% at 2B or below 14% at 3B → diagnose ≤ 6 h while training continues.
- **Hot-swaps** (§8) need the parity check: fp32 old vs new path on the same 10K states and a fixed 65K batch, |Δloss| < 1e-3, max |Δlogit| < 1e-2, equal greedy actions where the top-2 gap exceeds 0.05.
- **D6** (after Nov 9): continue iff paired DEV-2000 gained ≥ +2.0pp over the last 1B decisions and the box is free. TEST is read once regardless.

## Reward

Win +1.0; +0.05 per boss beaten (act 1, act 2, act-3 boss 1); +0.002 per new floor; death or a cap 0; no HP shaping; γ = 1. Caps: 100 turns or 1,000 decisions per fight, 6,000 per run; a capped run is a loss; alarm above 0.1%.

## P7 Forecast (scored after the read)

The statistics review's Monte Carlo model (`docs/plans/2026-11-02-ironclad-e2e-ppo-v0111/review/scripts/forecast_rev.py`): p10 10.7%, p25 14.2%, median 27.0%, p75 43.7%, p90 59.3%; P(< 15%) 0.27, P(≥ 25%) 0.53, P(≥ 40.8%) 0.29, P(≥ 56%) 0.13. Median path for non-stall draws: 11% at 1B, 16% at 2B, 21.5% at 3B, 31.5% at 5B decisions. Bar pass chances (est.): 0.69 / 0.63 / 0.55 / 0.30. Expected baselines (est.): R1 17-24%, R2 8-12%, R3 10-13%. The read refits logit(WR) on ln(decisions) at each 1B and reports the forecast's calibration.

## Amendment 1 (2026-11-02 01:38 CET): the provisional R3 pairs matched nets

#1081's session reported at ~01:35 that each lane's it02500 combat checkpoint was saved with run net v00375 (the acting copies at it02500 were already rotated out), and that `v00392.pt` is ~8-9 h newer than `hybrid/it02500.pt`. So the provisional R3 above pairs a run net with a combat net it never played with.

- The **provisional R3** is the matched pair `hybrid/run/ckpt/v00375.pt` + `hybrid/it02500.pt`, arm name `R3m`.
- The arm `R3p` (v00392 + it02500) was started at 01:37, before this correction, and is unchanged. It's reported descriptively only and is never R3, the teacher or C*.
- The **final R3** uses only matched pairs: the run net each lane saved with the chosen combat checkpoint.

This amendment is committed before R3p's result exists. It changes no bar, no TEST rule and no rule whose data exists.

## Amendment 2 (2026-11-02 03:06 CET): H3 stopped, so the final R3 is H2's it04000 pair

At ~03:05, #1081's session reported a plan change the user made. H3 stopped at 03:05 (it 2809, 631.6M decisions), and H2 now trains alone (restart reported at ~03:10). H3's final pre-registered read is therefore the ~600M one: H3 − H2 = +0.4pp, 95% CI [−2.5, +3.2] (1,000 paired full Ironclad A10 runs, greedy, no search; `HYBRID.md` at d9db2cf1d). GO needs ≥ +2pp with the CI's low end above 0, so H3 doesn't read GO and R3 stays on H2's pair. There's no 750M read.

- The **final R3** is H2's matched snapshot at it04000 (750.5M decisions): `hybrid/ckpt/it04000.pt` + `hybrid/run/ckpt/at_it04000.pt` (+ their `.src` files), which #1081 saves for this purpose. If the handover comes before H2 reaches it04000, the final R3 is the matched pair H2 saves at the handover.
- The pair is copied read-only to `~/projects/sts2/bench-archive/2026-11-03-combat-ppo-h2-h3/`, with a sha256 manifest, as soon as it exists, whether that's before the handover or at it. Its DEV-2000 runs on W2 right after the copy, and the teacher and C* are committed before the student launches (G1). This replaces the Tue 07:30-09:30 window above, which assumed a Tue 08:00 handover.

This amendment is committed before any data from it04000 or a later H2 state exists. It changes no bar, no TEST rule and no rule whose data exists.

## Amendment 3 (2026-11-04 ~17:30 CET): a pre-committed plateau remedy and a snapshot-disagreement read

Source: AlphaDefect's author reported (Discord, 2026-11-04) that their no-search PPO plateaus at 75-80% win rate, and that at that level dozens of snapshots rarely agree on a seed's outcome. They cited Beukman et al., "Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments" (arXiv:2603.06009, RLC 2026). That paper finds PPO plateaus where each update moves the policy too far for the noise in its loss estimate. The signature is flat return with update size and KL not falling. The fix is more samples per update at a fixed learning rate and minibatch size, which means more minibatches. Sweeping the learning rate does not fix it.

Written at G1 ≈ 0.06B decisions, in U0, before any DEV read of the student exists.

- **R-DDR (plateau remedy), U2 only (λ_KS = 0, from 500M).** Trigger: two consecutive DEV-1000 EMA0.996 milestone reads (the driver's 250M cadence) each gain < +1.0pp over the read before. Over the same 500M, the mean per-iteration `probe_kl` must not fall: its mean over the last 25M is ≥ its mean over the 25M before the first of the three reads. Remedy: at the next restart, or a planned one within 6 h, set `--iter-decisions 262144` (was 131,072). Keep `minibatch` 8192, `epochs`, the learning rates, `clip` and `dual_clip` unchanged, so each update runs twice the minibatches. Deploy it with `init-lane --hot-swap` (args only, no serving change). Run the learning re-check per P6, then a 20-min smoke that must keep KL ≤ 0.03 and clipfrac ≤ 0.3. Revert from the pre-switch pinned state iff, at +300M, paired DEV-1000 (EMA0.996) is ≥ 3.0pp below the pre-switch read with exact McNemar p < 0.05. It fires at most once. It never shares a restart with the E=1 switch: the two are ≥ 300M apart, E=1 first if both are due. It does not apply in U0/U1, where KS sets the step. The stall signals and K3/K4 still stand on their own.
- **Z5 (descriptive, no box time).** At each DEV-2000 milestone, the three variants (raw, EMA0.996, EMA0.999) already play the same 2,000 seeds. Report the share of seeds where they disagree on win or loss, and the share every variant wins. A high disagreement rate means a single-variant DEV-1000 read carries snapshot noise beyond its binomial CI. Z5 decides nothing. Gate reads already use EMA0.996 (P6, `EVAL_VARIANTS_DEV1000`), and P5 already picks among the variants.

This amendment is committed before any data it reads exists. It changes no bar, no TEST rule, no K gate and no rule whose data exists.

## Amendment 4 (2026-11-04 ~21:00 CET): K2 combat agreement waived, a recorded deviation

- **Read at K2 (pin `pinned_d0000100076112_k2.pt`, DEV-1000, EMA0.996).** Student 169/1000 (16.9%) against R3_W3 115/1000 (11.5%) on the same 1,000 seeds: +5.4pp, discordant 127/73, McNemar p = 0.00016, so the strength rule (teacher − 8pp) passes. Run agreement 0.879 (bar 0.75), value loss finite, combat EV 0.858 → 0.866 and run EV 0.346 → 0.276 from 50M (limit 0.1). Combat agreement 0.805 against the 0.85 bar: **miss**.
- **Decision (user, 2026-11-04):** waive the combat agreement bar and continue. P6 reads a miss as fix and relaunch. It is not applied here, because the bar stands in for "the student learned from the teacher", the student already outplays the teacher, and no fix has been identified. This is a deviation from P6, made after seeing the K2 read. K3 and K4 are unchanged.
- **Open caveat:** R3_W3 ran through `ppo.probe` and the student through `ppo.e2e.evaluate`. A control plays R3 through `ppo.e2e.evaluate` on the same seeds; its result is logged in CAMPAIGN_LOG. DEV seeds only.

## Amendment 5 (2026-11-05 ~16:00 CET): one epoch, and the end moved by 72 h (recorded deviations)

Decision (user, 2026-11-05), made at ~0.57B decisions, after the 500M DEV-1000 read (27.8%) and before any K3, K4 or TEST data of the student exists.

- **Why.** The schedule's wall-clock dates were set for a launch at Tue ~20:00 and 13-25k decisions/s; the lane launched Wed Nov 4 12:08 and runs ~6.4k/s, so the Sun 06:00 anneal would start at ~2.1B and the stop come at ~2.5B. AlphaDefect's published checkpoints reach 56.1% at ≤3.4B and 70.9% at ~4.3-5.2B game steps (their repo, `docs/model.md`; forensics in the plan's `kb/`).
- **E=1 now.** `--epochs 1` at this restart, from pinned state `e1_switch`. This departs from plan §9: the rule's precondition (the first restart after S1b) never occurs, and the learner blocks on the update ~20% of each iteration, below the 25% trigger. The rest of the §9 rule stands: a 20-min smoke window (KL ≤ 0.03 and clipfrac ≤ 0.3, the driver's `E1_REVERT`), and the revert iff at +300M paired DEV-1000 (EMA0.996) is ≥ 3.0pp below the pre-switch state with exact McNemar p < 0.05. Amendment 3's spacing applies: R-DDR cannot fire within 300M of this switch.
- **Dates, all +72 h.** lr anneal start and KL-factor freeze: Wed Nov 11 06:00 CET (`--sched` `anneal_start` = `kl_freeze_at` = 1794373200; the 24-h anneal and its 0.7 drop are unchanged). Stop: Thu Nov 12 06:00. TEST unseal and the single read: Thu Nov 12 09:00 CET (`panels.UNSEAL_AT`). The P5 pick on DEV-2000 moves with it. K3 and K4 keep their backstops (Nov 5 18:00, Nov 7 08:00) and bars. D6 now reads after Nov 12.
- Moving the unseal later only tightens P4. No bar, K gate or TEST rule changes otherwise.
