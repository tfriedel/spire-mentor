# Ironclad A10, end-to-end no-search PPO on v0.111.0 (in progress)

An AlphaDefect-style experiment: one transformer plays every decision of an Ironclad
Ascension 10 run (map, rewards, shops, events, rest sites and every combat turn) with
no search, trained with PPO from a teacher-acted warm start. Training is running now;
these are the plans, shared for anyone interested. They are a snapshot of a live
campaign, not a result.

| File | What it is |
|---|---|
| [`PLAN.md`](PLAN.md) | The approved plan, with a status section at the top (launch, rates, reads so far, changed dates). |
| [`PREREG.md`](PREREG.md) | The pre-registration: panels, seeds, baselines, bars, gates and the one sealed TEST read, plus its amendments (each committed before the data it governs existed). |
| [`EVAL.md`](EVAL.md) | How the panels are run and how the sealed TEST read is scored. |
| [`SCHEMA.md`](SCHEMA.md) | The observation and action schema (tokens, inputs, honesty rules). |
| [`BASELINES.md`](BASELINES.md) | The baseline agents on v0.111.0 and their DEV results. |

Notes:

- The plan cites a knowledge base and critic reports by key ([AD], [ENG], [PPO], [CPPO],
  [LES], [LIT], [CM], [V111]) and the ops runbook and campaign log; those working files
  are not included here, so those links don't resolve.
- `#NNNN` numbers refer to issues and PRs in the private development repository.
- The code this plan describes (`sts2-pycli/python/ppo/e2e/`) is not in this snapshot yet.
- Times are CET. The TEST read was moved to Thu Nov 12, 09:00 CET by PREREG Amendment 5.
