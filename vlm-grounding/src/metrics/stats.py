from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], stat: Callable = np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0, clusters: Sequence | None = None):
    """Percentile bootstrap. Pass ``clusters`` (e.g. image ids) to resample whole clusters."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = [rng.integers(0, len(x), len(x)) for _ in range(n_boot)]
        boots = [stat(x[i]) for i in idx]
    else:
        c = np.asarray(clusters)
        uniq = np.unique(c)
        groups = {u: x[c == u] for u in uniq}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            boots.append(stat(np.concatenate([groups[u] for u in pick])))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(x)), float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A right, B wrong), c (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
