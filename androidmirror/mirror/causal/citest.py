"""Conditional independence test for discrete data: the G-squared (likelihood ratio) test.

Each variable takes values in {-1, 0, 1} (low, typical, high), or None where the
situation leaves it unknown. Unknown values are handled by test-wise deletion:
each test uses only the rows where all of its variables are known. The test
sums G = 2 * sum O ln(O/E) over the strata of the conditioning set, with degrees
of freedom counted from the levels actually seen in each stratum. As in pcalg's
disCItest, a test with fewer than 10 rows per degree of freedom cannot reject
independence, and says so (`enough_data`), so "independent" and "untestable"
can be told apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Data = dict[str, list]  # values in {-1, 0, 1} or None (unknown)


@dataclass(frozen=True)
class CITest:
    x: str
    y: str
    s: tuple[str, ...]
    g2: float
    df: int
    p: float
    n: int
    enough_data: bool

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "given": list(self.s), "g2": round(self.g2, 3), "df": self.df,
                "p": round(self.p, 4), "n": self.n, "enough_data": self.enough_data}


def as_arrays(data: Data) -> dict[str, np.ndarray]:
    """Columns as small integer arrays (level + 1, unknown = -1), for fast counting."""
    return {k: (v if isinstance(v, np.ndarray) else np.asarray([-1 if x is None else x + 1 for x in v], dtype=np.int64))
            for k, v in data.items()}


def g2_test(data: Data, x: str, y: str, s: tuple[str, ...] = (), rows=None) -> CITest:
    cols = data if isinstance(data[x], np.ndarray) else as_arrays({v: data[v] for v in (x, y, *s)})
    if rows is not None:
        idx = np.asarray(rows)
        cols = {v: cols[v][idx] for v in (x, y, *s)}
    known = None
    for v in (x, y, *s):  # test-wise deletion of rows with an unknown value
        col = cols[v]
        if col.size and col.min() < 0:
            known = (col >= 0) if known is None else known & (col >= 0)
    if known is not None:
        cols = {v: cols[v][known] for v in (x, y, *s)}
    if not len(cols[x]):
        return CITest(x, y, tuple(s), 0.0, 0, 1.0, 0, False)
    strat = np.zeros(len(cols[x]), dtype=np.int64)
    for v in s:
        strat = strat * 3 + cols[v]
    cell = strat * 9 + cols[x] * 3 + cols[y]
    counts = np.bincount(cell, minlength=9 * 3 ** len(s)).reshape(-1, 3, 3).astype(float)
    tot = counts.sum(axis=(1, 2))
    counts, tot = counts[tot > 0], tot[tot > 0]
    n = int(tot.sum())
    xs, ys = counts.sum(axis=2), counts.sum(axis=1)
    df = int((((xs > 0).sum(axis=1) - 1) * ((ys > 0).sum(axis=1) - 1)).clip(min=0).sum())
    expected = xs[:, :, None] * ys[:, None, :] / tot[:, None, None]
    pos = counts > 0
    g2 = float(2.0 * (counts[pos] * np.log(counts[pos] / expected[pos])).sum())
    if df <= 0:
        return CITest(x, y, tuple(s), 0.0, 0, 1.0, n, False)
    enough = n >= 10 * df
    p = chi2_sf(g2, df) if enough else 1.0
    return CITest(x, y, tuple(s), g2, df, p, n, enough)


def chi2_sf(x: float, k: int) -> float:
    """P(chi2 with k degrees of freedom > x)."""
    if x <= 0:
        return 1.0
    return _gammaq(k / 2.0, x / 2.0)


def _gammaq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes gser/gcf)."""
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, total, delta = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-12:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - gln))
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)
