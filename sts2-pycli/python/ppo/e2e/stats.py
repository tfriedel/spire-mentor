"""Statistics for the e2e campaign's reads (PREREG-0 P2), deterministic and stdlib-first.

- :func:`wilson`: Wilson score 95% interval for one rate.
- :func:`mcnemar_exact`: exact two-sided McNemar test on the discordant counts.
- :func:`newcombe_paired`: Newcombe (1998) method 10 paired 95% CI of p1 - p2,
  ported line for line from ``newcombe_paired`` in
  ``docs/experiments/2026-10-26-ironclad-release-winrates/analysis/scripts/analyze.py``
  (around line 145; a test checks the port against that file).
- :func:`paired_bootstrap`: percentile CI of p1 - p2, resampling paired seeds
  (P2: 20,000 resamples, ``numpy.random.default_rng(20261109)``).
- :func:`cluster_bootstrap`: the same over clusters (DEVBOSS: a state and its reseeds).
- :func:`holm`: Holm's step-down adjustment for the descriptive family.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

#: The two-sided 95% normal quantile, as analyze.py uses it.
Z95 = 1.959963984540054
#: P2: the paired bootstrap's resample count and RNG seed.
BOOT_RESAMPLES = 20_000
BOOT_SEED = 20261109
# Resamples drawn per numpy call. Generator.integers fills an array from one
# stream in order, so the draws do not depend on the chunk size.
_CHUNK = 500


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval (no continuity correction); (nan, nan) when n == 0."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p: 2 * P(X <= min(b, c)), X ~ Binom(b + c, 1/2), capped at 1."""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2**m
    return min(1.0, 2 * tail)


def newcombe_paired(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """95% CI of p1 - p2 for paired binary data, Newcombe (1998) method 10 (hybrid score, no CC).

    a = both win, b = arm1 only, c = arm2 only, d = both lose; p1 = (a+b)/n, p2 = (a+c)/n.
    phi is the plain phi coefficient of the 2x2 table, taken as 0 when a margin is empty.
    """
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / den if den > 0 else 0.0
    diff = p1 - p2
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (diff - dl, diff + du)


@dataclass(frozen=True)
class PairedTable:
    """A paired 2x2 table of arm 1 (e.g. Final) against arm 2 (a baseline)."""

    a: int  # both win
    b: int  # arm 1 only
    c: int  # arm 2 only
    d: int  # both lose

    @property
    def n(self) -> int:
        return self.a + self.b + self.c + self.d

    @property
    def p1(self) -> float:
        return (self.a + self.b) / self.n

    @property
    def p2(self) -> float:
        return (self.a + self.c) / self.n

    @property
    def diff(self) -> float:
        return (self.b - self.c) / self.n

    @property
    def discordance(self) -> float:
        """d = (b + c) / n, the share of discordant pairs (P3)."""
        return (self.b + self.c) / self.n

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[bool, bool]]) -> PairedTable:
        a = b = c = d = 0
        for w1, w2 in pairs:
            if w1 and w2:
                a += 1
            elif w1:
                b += 1
            elif w2:
                c += 1
            else:
                d += 1
        return cls(a, b, c, d)


def paired_bootstrap(
    pairs: Sequence[tuple[bool, bool]],
    *,
    resamples: int = BOOT_RESAMPLES,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """Percentile 95% CI and SE of p1 - p2, resampling the pairs with replacement.

    `pairs` must be in a fixed order (the scorer passes them in seed order), so
    the result is a function of the data alone. Each resample draws n pair
    indices from ``default_rng(seed)``.
    """
    n = len(pairs)
    if n == 0:
        return {"lo": math.nan, "hi": math.nan, "se": math.nan, "resamples": resamples}
    v = np.array([int(w1) - int(w2) for w1, w2 in pairs], dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    for lo in range(0, resamples, _CHUNK):
        k = min(_CHUNK, resamples - lo)
        idx = rng.integers(0, n, size=(k, n))
        out[lo : lo + k] = v[idx].sum(axis=1) / n
    return {
        "lo": float(np.percentile(out, 2.5)),
        "hi": float(np.percentile(out, 97.5)),
        "se": float(np.std(out, ddof=1)),
        "resamples": resamples,
        "seed": seed,
    }


def cluster_bootstrap(
    sums: Sequence[float],
    counts: Sequence[int],
    *,
    resamples: int = BOOT_RESAMPLES,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """Cluster bootstrap of a ratio mean: sum(sums) / sum(counts) over resampled clusters.

    Cluster i contributes `sums[i]` (e.g. the summed per-pair differences of one
    DEVBOSS state over its reseeds) and `counts[i]` pairs. Each resample draws
    as many clusters as there are, with replacement. Returns the point
    estimate, the percentile 95% CI, the SE and a two-sided p for "mean = 0"
    from the recentred resamples: (#{|m* - m| >= |m|} + 1) / (resamples + 1).
    """
    s = np.asarray(sums, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)
    k = len(s)
    if k == 0 or c.sum() == 0:
        nan = math.nan
        return {"mean": nan, "lo": nan, "hi": nan, "se": nan, "p": nan, "clusters": k}
    est = float(s.sum() / c.sum())
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    for lo in range(0, resamples, _CHUNK):
        m = min(_CHUNK, resamples - lo)
        idx = rng.integers(0, k, size=(m, k))
        num = s[idx].sum(axis=1)
        den = c[idx].sum(axis=1)
        out[lo : lo + m] = num / den
    extreme = int(np.count_nonzero(np.abs(out - est) >= abs(est) - 1e-12))
    return {
        "mean": est,
        "lo": float(np.percentile(out, 2.5)),
        "hi": float(np.percentile(out, 97.5)),
        "se": float(np.std(out, ddof=1)),
        "p": (extreme + 1) / (resamples + 1),
        "clusters": k,
        "resamples": resamples,
        "seed": seed,
    }


def holm(pvalues: Mapping[str, float | None], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down over a fixed family. A member with p None counts as p = 1.

    The family size is ``len(pvalues)``, so a member that could not be computed
    (not run, INVALID) keeps its place and never rejects. Returns, per member,
    the raw p, the Holm-adjusted p and whether it rejects at `alpha`. Ties keep
    the mapping's order.
    """
    m = len(pvalues)
    order = sorted(
        enumerate(pvalues.items()), key=lambda t: (1.0 if t[1][1] is None else t[1][1], t[0])
    )
    out: dict[str, dict] = {}
    running = 0.0
    for rank, (_, (name, p)) in enumerate(order):
        raw = 1.0 if p is None else float(p)
        running = max(running, min(1.0, (m - rank) * raw))
        out[name] = {"p": p, "p_holm": running, "reject": running <= alpha, "rank": rank + 1}
    return {name: out[name] for name in pvalues}
