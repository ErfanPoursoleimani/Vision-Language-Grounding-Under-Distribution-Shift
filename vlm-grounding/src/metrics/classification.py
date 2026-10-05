from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def binary_metrics(pred: Sequence[int], label: Sequence[int]) -> dict:
    p, y = np.asarray(pred, int), np.asarray(label, int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    n = len(y)
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * prec * rec, prec + rec) if not (math.isnan(prec) or math.isnan(rec)) else float("nan")
    return {"n": n, "accuracy": div(tp + tn, n), "precision": prec, "recall": rec, "f1": f1,
            "yes_ratio": div(tp + fp, n), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUROC with average ranks for ties (threshold-free)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
