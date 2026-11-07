"""PPO-trained combat policy that plays fights alone, with no search (combat-PPO screen).

Plan: `docs/experiments/2026-10-31-combat-ppo-screen/PLAN.md`. The policy is the
combat-context transformer `ppo.combat_ctx.model.CombatCtxNet` on the honest
`S_ctx` token schema, warm-started from the step-0 imitation checkpoint and
fine-tuned with PPO. Rollouts are played by spawn workers that encode tokens on
the CPU, while one process does the batched GPU forward and the update.

Modules:

* `pool`: the training start pool (human-bank fights, holdouts removed).
* `encode`: honest policy tokens and privileged critic tokens, plus packing.
* `nets`: the policy and critic nets, batching, checkpoint IO.
* `actors`: the worker process (live fights, single-option auto-steps).
* `learner`: the PPO loop (inference server, GAE, update, logs, evals).
"""
