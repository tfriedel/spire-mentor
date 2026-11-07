# Plan: the e2e Ironclad net as a workbench brain

Status: draft, awaiting approval (2026-11-06). Worktree `~/projects/sts2-sim-wb-e2e`,
branch `feat/workbench-e2e-ironclad`, branched off `exp/e2e-ironclad-v111` (#1098).

## Goal

The browser workbench (Rust→wasm, Cloudflare Pages) gets a third brain, `e2e`. One
no-search transformer makes every decision, both combat and run-level, for Ironclad on
v0.111.0. It's an alternative to "PPO run-level + MCTS combat". The mod comes later
on the same Rust core.

## Facts this plan rests on

- Net: `E2ENet` 4×256 (4 heads, SwiGLU 704). The policy path is 5.1M parameters,
  5.9M with the value module. It's ~20 MB in fp32 and ~10 MB in fp16.
- Python speed (laptop, 1 thread, fp32): 14 ms per decision. The net takes 12.7 ms,
  the tokenizer 1.0 ms and the engine 0.6 ms. ~550 decisions per run, ~8 s per run.
- Nothing on the inference path exists in Rust. Every engine getter it reads does
  (`sts2-wire::emit`, `blindfold::belief_root`, `rollout_net::export_features_spec`).
- Size budgets: wasm 24.9 MB after `wasm-opt`, 25 MiB per-file limit → ~1.3 MB of
  headroom. tract can't grow; the net file must be fetched separately.

## Design decisions

1. **Hand-written forward in Rust, not ONNX/tract.** The forward is ~400 lines:
   embeddings, RMSNorm, attention with a key mask, SwiGLU, two heads. It avoids
   tract's dynamic-shape cost (the existing run net is 137 ms/decision under tract
   natively), adds almost no wasm, and B=1 needs no padding. Weights ship as one
   flat fp16 file with a JSON header (names, shapes, stamps: schema sha, vocab sha,
   prior sha, game version), widened to f32 at load.
2. **New crate `sts2-e2e`** (no PyO3): contracts tables, honest snapshot,
   IntentLog, S_ctx builder, e2e tokenizer, env driver (class merge, auto-advance,
   selections), net. The data tables (`vocab.tsv`, prior HITS, v0.111 card types,
   show-counter relics) are embedded with `include_str!`. They're ~2.6k lines of text,
   compressing well.
3. **Parity gates, decision level first:**
   - G-tok: Rust token arrays equal Python bitwise (policy fields only) on recorded
     decisions from fresh runs, every decision type, combat and OOC.
   - G-net: Rust logits vs torch fp32 within 1e-4, argmax equal on the same rows.
   - G-play: Rust drives full runs on the W3 engine and reproduces the Python
     chosen-class sequences seed for seed (the verify panel below records them).
   - G-wr: Rust win rate on 1,000 fresh seeds within noise of Python's.
4. **Workbench seam:** the brain string gets `e2e` (`set_run_mode`, RunLauncher,
   App, api.ts). `run_mode_step` routes to `e2e_step`, built like `agent_step`:
   native action, `NATIVE_ACTION` node so seek and replay work, class probabilities
   in `search` and ms per decision in `debug`. `recommended_action` and the
   autobattle champion pick also use it when it's active. The IntentLog is rebuilt
   from the node path on seek.
5. **Run setup:** e2e runs need v0.111.0, A10 default, `act1=roll` and
   `interactive_potion_swap=true` (the training config). It gets its own start
   option, since the faithful and interactive constructors force swap off and
   Overgrowth.
6. **Loading:** an `[ironclad_e2e]` entry in `champions.toml`. It's fetched on demand
   when the brain is chosen, not in the "download for offline" set by default.
7. **wasm speed:** turn on `+simd128` for the wasm build and measure ms per decision
   in the browser before and after.

## Steps

1. Python exporter: policy weights → fp16 file; a fixture dumper for G-tok and G-net
   (`decision_from_state` rows plus logits) and per-seed action sequences.
2. `sts2-e2e` net + G-net.
3. Tokenizer + env driver + G-tok, then G-play (native, on the W3 engine, then main).
4. Workbench brain, wasm loader, UI; local `vite preview` test.
5. Measure speed (native and wasm), G-wr, then `just deploy-static-cf-preview` for
   your test, then production deploy on your OK.
6. Gates per CLAUDE.md on the merge result.

## Open questions

- Which checkpoint ships first (proposal: 1.25B EMA0.999; refresh at the lane's end).
- The lane trains on W3; main has engine fixes since (#1143, #1145, #1149). The
  workbench runs main's engine, so a small drift from the W3 numbers is expected.
  G-play runs on W3 for exactness and on main for the drift size.
