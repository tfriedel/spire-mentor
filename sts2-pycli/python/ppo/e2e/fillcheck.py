"""The fill check (GH #1097 A4): a launch's declared fill events match its rows.

The learner's `--filled` (the stamps' `filled`, `contracts.FILL_EVENTS`) names the
reserved inputs the rows already carry. A fresh net inits those inputs live
(`E2ENet(filled=...)`), and the identity rule keeps every other reserved input at zero
weight. A declaration that disagrees with the rows breaks one of the two:

* WP-ENG-1 declared on rows without the getters (the W1 wheel): live weights train
  on inputs that stay zero, and the stamps claim a fill the rows lack;
* the getters' rows without the declaration: the telegraph, forecast and encounter
  inputs start at zero weight, and a later restart that "adds" WP-ENG-1 adds an
  event the rows always had.

    rep = check(rb, filled)        # a row batch (`contracts.ROW_FIELDS`), PRIV included
    if not rep.ok:
        raise SystemExit("; ".join(rep.problems))

WP-ENG-1 declared, on combat rows:

* RUNCTX's X_ID_AUX (the fight's encounter) is nonzero on at least `AUX_MIN` (95%)
  of the rows;
* the telegraph columns (E_N_INTENTS, E_T2/T3_*, X_INTENT_T2/T3) are nonzero on at
  least `ENEMY_MIN` (50%) of the live-enemy tokens (ENEMY with E_ALIVE), and so are
  the forecast columns (E_F1-3_*, X_FC1-3);
* there are at least `ENEMY_MIN` PRIV_ENEMY tokens (`cs.peek_intents`) per live-enemy
  token.

WP-ENG-1 not declared: every WP-ENG-1 column is zero on every token and no row has a
PRIV_ENEMY token. Either way, dead enemies carry no WP-ENG-1 value, and every combat
row with a non-empty draw pile (STATE's N_DRAW) has PRIV_DRAW tokens: the tokenizer
reads `cs.true_draw_top` or the old `state_to_dict` fallback, so PRIV_DRAW does not
depend on the fill. Any other fill event not declared leaves no trace: its reserved
columns are zero and no row has its reserved decision types or action kinds. A
declared event other than WP-ENG-1 has no positive check here (`unchecked`).

Fewer than `MIN_ROWS` combat rows, or with WP-ENG-1 declared fewer than `MIN_ENEMIES`
live-enemy tokens, is inconclusive: `ok` is False, `inconclusive` is True and a problem
says so. The other problems hold on any sample.

The learner runs `check` on the first table of every start, fresh or resumed (until one
is conclusive), and stops with `FILL_MISMATCH` on a problem that is not inconclusive
(`Cfg.fill_check`). `wheel_filled` names the events this process's wheel fills.

`sample_rows` plays random A10 runs on this process's wheel and tokenizes them as the
workers do (`tokens.encode(d, priv=True)`), for the pre-flight
(`python -m ppo.e2e.preflight`), which runs the check against the declared events of a
learner run.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from . import contracts as C

WP: Final = "WP-ENG-1"
AUX_MIN: Final = 0.95
ENEMY_MIN: Final = 0.5
MIN_ROWS: Final = 100
MIN_ENEMIES: Final = 100

_TELE_CAT: Final = tuple(C.XC[n] for n in ("X_INTENT_T2", "X_INTENT_T3"))
_TELE_CONT: Final = tuple(
    C.XF[n] for n in ("E_N_INTENTS", "E_T2_EFF", "E_T2_TIMES", "E_T3_EFF", "E_T3_TIMES")
)
_FC_CAT: Final = tuple(C.XC[f"X_FC{s}"] for s in (1, 2, 3))
_FC_CONT: Final = tuple(
    C.XF[f"E_F{s}_{k}"] for s in (1, 2, 3) for k in ("CERT", "AMT", "TIMES", "TOT")
)


def reserved_cols(event: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The `cat` and `cont` columns `contracts` reserves for fill event `event`."""
    status = f"reserved:{event}"
    return (
        tuple(C.XC[c.name] for c in C.X_CAT if c.status == status),
        tuple(C.XF[c.name] for c in C.X_CONT if c.status == status),
    )


def _nonzero(cat: np.ndarray, cont: np.ndarray, cols: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Per token: any of the `cat` columns `cols[0]` or `cont` columns `cols[1]` nonzero."""
    ccat, ccont = cols
    out = np.zeros(len(cat), bool)
    if ccat:
        out |= (cat[:, list(ccat)] != 0).any(1)
    if ccont:
        out |= (cont[:, list(ccont)] != 0).any(1)
    return out


@dataclass(frozen=True)
class FillCheck:
    """The fill check of one row batch (`check`)."""

    ok: bool
    filled: tuple[str, ...]
    rows: int
    combat_rows: int
    aux_rows: int  # combat rows whose RUNCTX X_ID_AUX is nonzero
    enemies: int  # live-enemy tokens of combat rows
    tele: int  # of those, with a nonzero telegraph column
    forecast: int  # of those, with a nonzero forecast column
    priv_enemy: int  # PRIV_ENEMY tokens
    dead: int  # dead-enemy tokens
    dead_filled: int  # of those, with a nonzero WP-ENG-1 column
    draw_rows: int  # combat rows with a non-empty draw pile
    draw_missing: int  # of those, without a PRIV_DRAW token
    # Undeclared event -> tokens with a nonzero reserved column, plus rows of its
    # reserved decision types and option tokens of its reserved action kinds.
    stray: dict[str, int] = field(default_factory=dict)
    unchecked: tuple[str, ...] = ()  # declared events with no positive check
    problems: tuple[str, ...] = ()
    # Too small a sample for the positive checks (MIN_ROWS, MIN_ENEMIES): `problems`
    # says so; its other problems still hold.
    inconclusive: bool = False

    def definite(self) -> tuple[str, ...]:
        """The problems that hold on any sample (all but the inconclusive notes)."""
        return tuple(p for p in self.problems if not p.startswith("inconclusive:"))

    def summary(self) -> str:
        def pct(a: int, b: int) -> str:
            return f"{a}/{b} = {a / b:.1%}" if b else f"{a}/0"

        return (
            f"filled {list(self.filled)}: {self.combat_rows} combat rows of {self.rows}; "
            f"X_ID_AUX {pct(self.aux_rows, self.combat_rows)}; live enemies {self.enemies}: "
            f"telegraph {pct(self.tele, self.enemies)}, forecast "
            f"{pct(self.forecast, self.enemies)}, PRIV_ENEMY {self.priv_enemy}; dead enemies "
            f"{self.dead} ({self.dead_filled} filled); PRIV_DRAW missing on "
            f"{self.draw_missing} of {self.draw_rows} rows with a draw pile"
        )


def check(rb: Mapping[str, np.ndarray], filled: Iterable[str]) -> FillCheck:
    """Check row batch `rb` against the declared fill events `filled` (module docstring).

    Raises:
        contracts.SchemaError: `filled` names an event outside `contracts.FILL_EVENTS`,
            or a row does not start with its RUNCTX token.
    """
    key = tuple(sorted({str(e) for e in filled}))
    if bad := sorted(set(key) - set(C.FILL_EVENTS)):
        raise C.SchemaError(f"unknown fill events {bad}")
    off = C.row_offsets(rb)
    n = len(rb["n_tok"])
    cat, cont, sec = rb["cat"], rb["cont"], rb["sec"].astype(np.int64)
    pcat, pcont, psec = rb["pcat"], rb["pcont"], rb["psec"].astype(np.int64)
    comb = np.asarray(rb["is_combat"], bool)
    tok_row = np.repeat(np.arange(n), rb["n_tok"].astype(np.int64))
    priv_row = np.repeat(np.arange(n), rb["n_priv"].astype(np.int64))
    tok_comb = comb[tok_row]
    first = off["tok"][:-1]
    if n and (sec[first] != C.SEC["RUNCTX"]).any():
        raise C.SchemaError("a row does not start with its RUNCTX token")
    combat_rows = int(comb.sum())
    aux_rows = int(((cat[first, C.XC["X_ID_AUX"]] != 0) & comb).sum()) if n else 0
    wp = reserved_cols(WP)
    enemy = (sec == C.SEC["ENEMY"]) & tok_comb
    alive = cont[:, C.SCTX_C["E_ALIVE"]] > 0
    live, dead = enemy & alive, enemy & ~alive
    tele = int((live & _nonzero(cat, cont, (_TELE_CAT, _TELE_CONT))).sum())
    forecast = int((live & _nonzero(cat, cont, (_FC_CAT, _FC_CONT))).sum())
    wp_tok = _nonzero(cat, cont, wp)
    dead_filled = int((dead & wp_tok).sum())
    priv_enemy = int((psec == C.SEC["PRIV_ENEMY"]).sum())
    pile = np.zeros(n, bool)
    state = (sec == C.SEC["STATE"]) & tok_comb & (cont[:, C.SCTX_C["N_DRAW"]] > 0)
    pile[tok_row[state]] = True
    has_draw = np.zeros(n, bool)
    has_draw[priv_row[psec == C.SEC["PRIV_DRAW"]]] = True
    draw_rows = int((pile & comb).sum())
    draw_missing = int((pile & comb & ~has_draw).sum())
    n_live = int(live.sum())

    problems: list[str] = []
    stray: dict[str, int] = {}
    unchecked: list[str] = []
    inconclusive = combat_rows < MIN_ROWS or (WP in key and n_live < MIN_ENEMIES)
    kinds = cat[:, C.XC["X_KIND"]].astype(np.int64)
    opt = (rb["tok_flags"] & np.uint8(C.TF_OPTION)) != 0
    for ev in C.FILL_EVENTS:
        if ev in key:
            if ev != WP:
                unchecked.append(ev)
            continue
        cols = reserved_cols(ev)
        status = f"reserved:{ev}"
        dts = [d.id for d in C.DECISION_TYPES if d.status == status]
        aks = [k.id for k in C.ACTION_KINDS if k.status == status]
        ptok = _nonzero(pcat, pcont, cols)
        if ev == WP:  # PRIV_ENEMY is `cs.peek_intents`, a WP-ENG-1 getter
            ptok |= psec == C.SEC["PRIV_ENEMY"]
        hits = int(_nonzero(cat, cont, cols).sum()) + int(ptok.sum())
        hits += int(np.isin(rb["dtype"], dts).sum()) if dts else 0
        hits += int((opt & np.isin(kinds, aks)).sum()) if aks else 0
        if hits:
            stray[ev] = hits
            problems.append(f"{ev} is not declared but {hits} token(s) or row(s) carry it")
    if combat_rows < MIN_ROWS:
        problems.append(f"inconclusive: {combat_rows} combat rows < {MIN_ROWS}")
    elif WP in key:
        if aux_rows < AUX_MIN * combat_rows:
            problems.append(
                f"{WP} declared but X_ID_AUX is nonzero on {aux_rows}/{combat_rows} combat "
                f"rows (< {AUX_MIN:.0%})"
            )
        if n_live < MIN_ENEMIES:
            problems.append(f"inconclusive: {n_live} live-enemy tokens < {MIN_ENEMIES}")
        else:
            for what, got in (("telegraph", tele), ("forecast", forecast)):
                if got < ENEMY_MIN * n_live:
                    problems.append(
                        f"{WP} declared but the {what} columns are nonzero on {got}/{n_live} "
                        f"live-enemy tokens (< {ENEMY_MIN:.0%})"
                    )
            if priv_enemy < ENEMY_MIN * n_live:
                problems.append(
                    f"{WP} declared but there are {priv_enemy} PRIV_ENEMY tokens for "
                    f"{n_live} live enemies (< {ENEMY_MIN:.0%})"
                )
    if dead_filled:
        problems.append(f"{dead_filled} dead-enemy token(s) carry {WP} values")
    if draw_missing:
        problems.append(f"{draw_missing} combat row(s) with a draw pile have no PRIV_DRAW")
    return FillCheck(
        ok=not problems,
        filled=key,
        rows=n,
        combat_rows=combat_rows,
        aux_rows=aux_rows,
        enemies=n_live,
        tele=tele,
        forecast=forecast,
        priv_enemy=priv_enemy,
        dead=int(dead.sum()),
        dead_filled=dead_filled,
        draw_rows=draw_rows,
        draw_missing=draw_missing,
        stray=stray,
        unchecked=tuple(unchecked),
        problems=tuple(problems),
        inconclusive=inconclusive,
    )


#: The fill events a wheel probe covers (`wheel_filled`).
WHEEL_EVENTS: tuple[str, ...] = (WP,)


def wheel_filled() -> tuple[str, ...]:
    """The fill events this process's wheel fills: WP-ENG-1 when its `CombatState` has the
    WP-ENG-1 getters (`telegraph`, `peek_intents`, `true_draw_top`, `encounter_id`; the
    W2 wheel), none on W1."""
    import sts2_rs

    getters = ("telegraph", "peek_intents", "true_draw_top", "encounter_id")
    return (WP,) if all(hasattr(sts2_rs.CombatState, g) for g in getters) else ()


def random_actor(seed: int, play_bias: float = 0.85) -> Callable[[Any], int]:
    """A seeded random actor that prefers playing a card to ending the turn."""
    rng = random.Random(seed)

    def act(d: Any) -> int:
        if d.kind == "combat":
            ends = {
                c for c, a in zip(d.classes, d.actions, strict=True) if a["action"] == "end_turn"
            }
            others = [c for c in range(d.n_classes) if c not in ends]
            if others and rng.random() < play_bias:
                return rng.choice(others)
        return rng.randrange(d.n_classes)

    return act


def sample_rows(n_runs: int, seed0: int) -> tuple[dict[str, np.ndarray], list[Any]]:
    """The rows of `n_runs` random A10 runs (seeds `seed0`, `seed0 + 1`, ...) on this
    process's wheel, tokenized as the workers do (`tokens.encode(d, priv=True)`), and
    the runs' terminals (`env.E2ETerminal`)."""
    from . import env as E
    from . import tokens as TK

    toks: list[Any] = []
    uids: list[int] = []
    ts: list[int] = []
    terms: list[Any] = []
    for k in range(n_runs):
        seed = seed0 + k
        env = E.E2EEnv(seed, cfg=E.EnvConfig(f9_p=0.0))
        act = random_actor(seed)
        d = env.reset()
        t = 0
        while isinstance(d, E.E2EDecision):
            toks.append(TK.encode(d, priv=True))
            uids.append(k)
            ts.append(t)
            t += 1
            d = env.step(act(d))
        terms.append(env.terminal)
    if not toks:
        raise ValueError(f"{n_runs} runs from seed {seed0} stored no row")
    return TK.row_batch(toks, run_uid=uids, t=ts), terms
