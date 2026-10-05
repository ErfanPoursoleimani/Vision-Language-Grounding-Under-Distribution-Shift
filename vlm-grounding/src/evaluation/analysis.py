"""Diagnostics for probe runs (pure functions over per_example / per_item records)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..metrics.classification import auroc


def pooled_auroc_by_template(rows: List[dict]) -> Dict[int, float]:
    out = {}
    for t in sorted({r["template_idx"] for r in rows}):
        rt = [r for r in rows if r["template_idx"] == t]
        out[t] = auroc([r["logodds"] for r in rt], [r["label"] for r in rt])
    return out


def balanced_accuracy(pos: np.ndarray, neg: np.ndarray, th: float) -> float:
    return 0.5 * (float((pos > th).mean()) + float((neg <= th).mean()))


def best_threshold(pos, neg) -> tuple:
    """Threshold maximizing balanced accuracy (pos = label 1 scores, neg = label 0 scores); ties -> closest to 0."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = np.unique(np.concatenate([pos, neg]))
    cands = np.concatenate([[vals[0] - 1e-6], (vals[:-1] + vals[1:]) / 2, [vals[-1] + 1e-6]])
    ba = np.array([balanced_accuracy(pos, neg, c) for c in cands])
    best = np.flatnonzero(ba >= ba.max() - 1e-12)
    i = best[np.argmin(np.abs(cands[best]))]
    return float(cands[i]), float(ba[i])


def calibrate_thresholds(rows: List[dict]) -> Dict[int, float]:
    """Per-template decision thresholds from (dev) rows: pos vs ALL negatives, balanced accuracy.
    Fit on dev only, then FREEZE and apply to test."""
    th = {}
    for t in sorted({r["template_idx"] for r in rows}):
        rt = [r for r in rows if r["template_idx"] == t]
        th[t], _ = best_threshold([r["logodds"] for r in rt if r["label"] == 1],
                                  [r["logodds"] for r in rt if r["label"] == 0])
    return th


def negative_overlap(items: List[dict]) -> dict:
    neg = [i for i in items if i["label"] == 0]
    kinds: Dict[tuple, set] = defaultdict(set)
    for i in neg:
        kinds[(i["image_id"], i["category"])].add(i["kind"])
    return {"n_negative_items": len(neg), "n_unique_negative_pairs": len(kinds),
            "n_pairs_in_multiple_kinds": sum(len(v) > 1 for v in kinds.values())}


def fp_rate_by_kind(items: List[dict]) -> Dict[str, float]:
    out = {}
    for k in ("random", "popular", "adversarial"):
        sub = [i for i in items if i["kind"] == k]
        out[k] = float(np.mean([i["pred"] for i in sub])) if sub else float("nan")
    return out


def template_agreement(rows: List[dict]) -> List[List[float]]:
    g: Dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        g[(r["image_id"], r["category"], r["kind"])][r["template_idx"]] = r["logodds"]
    n_t = max(r["template_idx"] for r in rows) + 1
    m = np.array([[v.get(t, np.nan) for t in range(n_t)] for v in g.values()])
    return np.corrcoef(m.T).tolist()


def language_prior_share(items: List[dict]) -> dict:
    fps = [i for i in items if i["pred"] == 1 and i["label"] == 0 and i["blind_pred"] is not None]
    blind_all = [i["blind_pred"] for i in items if i["blind_pred"] is not None]
    return {"n_false_positives": len(fps), "n_fp_where_blind_also_yes": int(sum(i["blind_pred"] == 1 for i in fps)),
            "blind_yes_ratio_overall": float(np.mean(blind_all)) if blind_all else float("nan"),
            "caveat": "a gray-image baseline measures the no-evidence response; it does not capture scene-conditioned priors"}


def per_category(items: List[dict], min_neg: int = 15) -> List[dict]:
    g: Dict[str, list] = defaultdict(list)
    for i in items:
        g[i["category"]].append(i)
    out = []
    for c, xs in g.items():
        neg = [x for x in xs if x["label"] == 0]
        pos = [x for x in xs if x["label"] == 1]
        out.append({"category": c, "n_pos": len(pos), "n_neg": len(neg),
                    "fp_rate": float(np.mean([x["pred"] for x in neg])) if neg else None,
                    "recall": float(np.mean([x["pred"] for x in pos])) if pos else None,
                    "reliable": len(neg) >= min_neg and len(pos) >= min_neg})
    return sorted(out, key=lambda d: d["category"])
