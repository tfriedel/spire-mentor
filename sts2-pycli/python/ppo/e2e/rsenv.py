"""`E2EEnv` on the Rust training env (`sts2_rs.E2eTrainEnv`, GH #1097 speed).

`RsE2EEnv` is `env.E2EEnv` as the rollout worker drives it, with the env loop,
its bookkeeping (rewards, counters, fights, the action log, practice captures,
the F9 draw) and the tokenizer (`tokens.encode(d, rare=rare, priv=True)`) in
Rust: `sts2-e2e/src/train.rs` ports `E2EEnv`'s control flow line by line and
`sts2-e2e/src/tokens.rs` the tokenizer. The lockstep check
(`ppo.e2e.rsenv_lockstep`) holds the two to the same rows, rewards,
counters, captures, F9 states and tokens.

What stays in Python, here: the construction-time build checks, the F9 check
bodies (run on the post-action states the Rust env samples, in order, with
the env's own `InvariantChecker`), the practice capture callback, the run
fingerprint and the `E2ETerminal` the learner and the action log read.

Not supported (refused at construction): a shared `EnvConfig.f9` checker
(its draws would interleave across envs) and `EnvConfig.tokenizer`. The
decisions carry no `run_obs`, so the run teacher's features
(`actors.run_teacher_features`) are not available: the learner refuses
`--rust-env` with a run-teacher schedule at startup.

Two deliberate differences from `E2EEnv`: the process build selection
(`factory.require_selection`, half of `factory.check_fight`) is checked once
at construction rather than at every fight start (the Rust env checks the
other half, the combat state's build, at every fight start); and at most
`sts2_rs` `F9_QUEUE_MAX` sampled states are kept per `reset`/`step` call, the
rest counted `f9/dropped` (a hung fight driven to the step cap inside one
call would otherwise hold thousands of run copies).
"""

from __future__ import annotations

import collections
from typing import Any

import numpy as np
import sts2_client

from . import contracts as C
from . import factory
from .env import (
    _SEGMENT_SERIAL,
    RUN,
    Capture,
    E2ETerminal,
    EnvConfig,
    NoActions,
    Segment,
    check_campaign_run,
    ends_segment,
    make_run_manager,
    run_fingerprint,
)
from .invariants import InvariantChecker
from .tokens import N_CAT, N_CONT, Tok

_F16 = np.dtype(np.float16)


def _u16(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.uint16)


def _f16(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.uint16).view(_F16)


def tok_of(t: dict[str, Any]) -> Tok:
    """A `tokens.Tok` over the Rust tokenizer's bytes (`E2eTrainEnv` rows)."""
    n = int(t["n_tok"])
    return Tok(
        is_combat=bool(t["is_combat"]),
        dtype=int(t["dtype"]),
        cat=np.frombuffer(t["cat"], np.int16).reshape(n, N_CAT),
        cont=_f16(t["cont"]).reshape(n, N_CONT),
        sec=np.frombuffer(t["sec"], np.int8),
        tok_flags=np.frombuffer(t["tok_flags"], np.uint8),
        mem_tok=_u16(t["mem_tok"]),
        mem_class=_u16(t["mem_class"]),
        cls_tok=_u16(t["cls_tok"]),
        cls_kind=np.frombuffer(t["cls_kind"], np.uint8),
        sctx_lo=int(t["sctx_lo"]),
        n_sctx=int(t["n_sctx"]),
        pool_tok=_u16(t["pool_tok"]),
        pool_vid=_u16(t["pool_vid"]),
        pool_w=_f16(t["pool_w"]),
        pcat=np.frombuffer(t["pcat"], np.int16).reshape(-1, N_CAT),
        pcont=_f16(t["pcont"]).reshape(-1, N_CONT),
        psec=np.frombuffer(t["psec"], np.int8),
        ppool_tok=_u16(t["ppool_tok"]),
        ppool_vid=_u16(t["ppool_vid"]),
        ppool_w=_f16(t["ppool_w"]),
    )


class RsDecision:
    """A stored row of an `RsE2EEnv`: the fields the worker reads off an
    `E2EDecision`, and its tokenized row (`tok`)."""

    __slots__ = (
        "_tok",
        "_tok_error",
        "act",
        "bootstrap",
        "cell",
        "classes",
        "dtype",
        "floor",
        "is_combat",
        "members",
        "n_classes",
        "reward",
        "row",
        "unit",
    )

    def __init__(self, r: dict[str, Any], cell: int) -> None:
        self.row = int(r["row"])
        self.is_combat = bool(r["is_combat"])
        self.dtype = int(r["dtype"])
        self.n_classes = int(r["n_classes"])
        self.reward = float(r["reward"])
        self.bootstrap = bool(r["bootstrap"])
        kind, idx = r["unit"]
        self.unit = (str(kind), int(idx))
        self.cell = cell
        # Only with `RsE2EEnv(detail=True)` (the lockstep check); None otherwise.
        self.members = r.get("members")
        self.classes = r.get("classes")
        self.act = r.get("act")
        self.floor = r.get("floor")
        t = r.get("tok")
        self._tok = tok_of(t) if t is not None else None
        self._tok_error = r.get("tok_error")

    @property
    def tok(self) -> Tok:
        """The row (`tokens.encode(d, rare=rare, priv=True)`); raises what the
        Python tokenizer would have raised."""
        if self._tok is None:
            msg = str(self._tok_error or "the env did not tokenize this row")
            if msg.startswith("SchemaError"):
                raise C.SchemaError(msg)
            raise RuntimeError(msg)
        return self._tok

    @property
    def run_obs(self) -> Any:
        raise RuntimeError(
            "an RsE2EEnv row carries no run_obs: the run teacher's features need the "
            "Python env (actors.run_teacher_features)"
        )


class RsE2EEnv:
    """`E2EEnv(seed, rm=rm, cfg=cfg, segment=segment)` on the Rust env (module doc)."""

    rust = True

    def __init__(
        self,
        seed: int,
        *,
        rm: Any = None,
        cfg: EnvConfig | None = None,
        segment: Segment = RUN,
        rare: Any = None,
        detail: bool = False,
        f9_queue_max: int | None = None,
    ) -> None:
        """`detail` adds `members`, `classes`, `act` and `floor` to every row (a per-row
        copy the worker does not need); `f9_queue_max` overrides the F9 bound (module doc)."""
        import sts2_rs

        self.cfg = cfg = cfg or EnvConfig()
        if cfg.f9 is not None:
            raise ValueError("RsE2EEnv samples F9 per env: a shared EnvConfig.f9 is unsupported")
        if cfg.tokenizer is not None:
            raise ValueError("RsE2EEnv tokenizes in Rust: EnvConfig.tokenizer is unsupported")
        factory.require_selection()  # `check_fight`'s process half (module doc)
        if sts2_rs.potion_swap_is_rule_side():
            raise RuntimeError(
                "STS2_RUNAI_POTION_OC is set, so PpoRunEnv would resolve full-belt potion "
                "swaps rule-side; the campaign student decides them (plan D4). Unset it."
            )
        self.seed = int(seed)
        self.segment = segment
        if rm is None:
            rm = make_run_manager(self.seed)
        if not cfg.allow_any_run:
            check_campaign_run(rm)
        f9_seed = (self.seed << 32) ^ (int(segment.reseed or 0) & 0xFFFFFFFF)
        self._on_capture = cfg.on_capture if segment.kind == "run" else None
        self._env = sts2_rs.E2eTrainEnv(
            rm,
            self.seed,
            first_ooc=segment.end == "first_ooc",
            capture=self._on_capture is not None,
            f9_seed=f9_seed,
            f9_rate=float(cfg.f9_p),
            priv_=True,
            rare=rare,
            caps={
                "run_steps": int(cfg.max_run_steps),
                "fight_turns": int(cfg.max_fight_turns),
                "fight_decisions": int(cfg.max_fight_decisions),
                "run_decisions": int(cfg.max_run_decisions),
            },
            game_version=factory.GAME_VERSION,
            detail=bool(detail),
            **({} if f9_queue_max is None else {"f9_queue_max": int(f9_queue_max)}),
        )
        # The F9 check bodies run here, on the states the Rust env samples; the
        # checker's own sampler is unused (the Rust env draws the same stream).
        self.f9 = InvariantChecker(
            sample_rate=cfg.f9_p, seed=f9_seed, tolerate_known=cfg.f9_tolerate
        )
        self._f9_key = ("e2e", next(_SEGMENT_SERIAL))
        self.f9_violations: list[str] = []
        self._f9_counts: collections.Counter = collections.Counter()
        self.terminal: E2ETerminal | None = None
        self._done = False

    # ── the worker's surface ────────────────────────────────────────────────

    @property
    def rm(self) -> Any:
        """A copy of the live run (the Rust env owns the run)."""
        return self._env.rm()

    @property
    def rows(self) -> int:
        return int(self._env.rows)

    @property
    def log_idx(self) -> list[int]:
        return list(self._env.log_idx)

    @property
    def counts(self) -> collections.Counter:
        c = collections.Counter(self._env.counts())
        c.update(self._f9_counts)
        return c

    @property
    def done(self) -> bool:
        return self._done

    def reset(self) -> RsDecision | E2ETerminal:
        """As `E2EEnv.reset`: raises when the advance to the first output fails
        (the worker's `MAX_START_FAILS` counts that) or on a build mismatch."""
        return self._call(self._env.reset)

    def step(self, choice: int) -> RsDecision | E2ETerminal:
        return self._call(self._env.step, int(choice))

    def step_action(self, ai: int) -> RsDecision | E2ETerminal:
        return self._call(self._env.step_action, int(ai))

    # ── internals ───────────────────────────────────────────────────────────

    def _call(self, f: Any, *args: Any) -> RsDecision | E2ETerminal:
        import sts2_rs

        try:
            r = f(*args)
        except sts2_rs.E2eEnvRaise as ex:
            self._drain()  # the Python env checked these before it raised
            raise _python_error(*ex.args) from None
        return self._out(r)

    def _out(self, r: dict[str, Any]) -> RsDecision | E2ETerminal:
        self._drain()
        if r["kind"] == "decision":
            d = RsDecision(r, self.segment.cell_id)
            if d.bootstrap:
                self._finish(self._env.terminal())
            return d
        return self._finish(r)

    def _drain(self) -> None:
        """Captures to `cfg.on_capture`, sampled states through the F9 checks
        (`E2EEnv._f9_step`), both in the order the env made them."""
        cb = self._on_capture
        for c in self._env.drain_captures():
            if cb is not None:
                cb(
                    Capture(
                        cell=str(c["cell"]),
                        snap=c["snap"],
                        move=dict(c["move"]),
                        seed=self.seed,
                        steps=int(c["steps"]),
                        rows=int(c["rows"]),
                        idx=tuple(int(i) for i in c["idx"]),
                    )
                )
        chk = self.f9
        for st in self._env.drain_f9():
            tol = getattr(chk, "tolerated", None)
            before = dict(tol) if tol else None
            found = chk.check(st, None, run_key=self._f9_key)
            self._f9_counts["f9/checks"] += 1
            if tol:
                for reason, n in tol.items():
                    dn = n - (before or {}).get(reason, 0)
                    if dn:
                        self._f9_counts["f9/tolerated"] += dn
                        self._f9_counts[f"f9/tolerated/{reason}"] += dn
            for v in found:
                self._f9_counts["f9/violations"] += 1
                self._f9_counts[f"f9/{v.code}"] += 1
                if len(self.f9_violations) < 8:
                    self.f9_violations.append(f"{v.code}: {v.detail}")

    def _finish(self, t: dict[str, Any]) -> E2ETerminal:
        rm = self._env.rm()
        if t["outcome"] == "error":
            try:
                fp = run_fingerprint(rm)
            except BaseException as ex:  # a broken state still ends the segment
                if not ends_segment(ex):
                    raise
                fp = "unavailable"
        else:
            fp = run_fingerprint(rm)
        counts = collections.Counter(t["counts"])
        counts.update(self._f9_counts)
        err = t.get("error")
        etype = t.get("etype")
        if err is not None and etype and not str(err).startswith(f"{etype}:"):
            err = f"{etype}: {err}"
        self.terminal = E2ETerminal(
            outcome=str(t["outcome"]),
            won=bool(t["won"]),
            cap=t["cap"],
            reward=float(t["reward"]),
            total_reward=float(t["total_reward"]),
            rewards=[],
            unassigned_reward=float(t["unassigned_reward"]),
            rows=int(t["rows"]),
            steps=int(t["steps"]),
            floor=int(t["floor"]),
            act=int(t["act"]),
            seed=self.seed,
            segment=self.segment,
            bosses=int(t["bosses"]),
            reached_boss2=bool(t["reached_boss2"]),
            fingerprint=fp,
            log_idx=self.log_idx,
            counts=counts,
            error=err,
        )
        self._done = True
        return self.terminal


def _python_error(etype: str, msg: str) -> BaseException:
    """The exception `E2EEnv` raises where `E2eTrainEnv` raised `E2eEnvRaise(etype, msg)`."""
    text = msg if msg.startswith(f"{etype}:") else f"{etype}: {msg}"
    if etype == "GameVersionMismatch":
        return sts2_client.GameVersionMismatch(text)
    if etype == "SchemaError":
        return C.SchemaError(text)
    if etype == "NoActions":
        return NoActions(text)
    return RuntimeError(text)
