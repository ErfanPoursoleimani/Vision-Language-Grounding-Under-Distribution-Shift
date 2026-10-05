from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..metrics.classification import auroc, binary_metrics
from ..metrics.stats import bootstrap_ci

NEG_KINDS = ("random", "popular", "adversarial")


def _nan_stats(vals) -> dict:
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def _subset(rows: List[dict], neg_kind: str) -> List[dict]:
    return [r for r in rows if r["kind"] in ("pos", neg_kind)]


def aggregate_probes(rows: List[dict], threshold: float = 0.0, n_boot: int = 1000, seed: int = 0) -> dict:
    """Per negative-sampling subset (pos + one negative kind, 50/50 balanced), for the model and the blind baseline:
    metrics per template, mean/min/max over templates, and a bootstrap CI (resampling images) on template-averaged accuracy."""
    systems = ["model"] + (["blind"] if rows and rows[0].get("blind_logodds") is not None else [])
    key = {"model": "logodds", "blind": "blind_logodds"}
    out: dict = {"threshold": threshold, "subsets": {}}
    for nk in NEG_KINDS:
        sub = _subset(rows, nk)
        if not sub:
            continue
        n_t = max(r["template_idx"] for r in sub) + 1
        res: dict = {}
        for sysname in systems:
            per_t = []
            for t in range(n_t):
                rt = [r for r in sub if r["template_idx"] == t]
                lo = [r[key[sysname]] for r in rt]
                lab = [r["label"] for r in rt]
                m = binary_metrics([int(x > threshold) for x in lo], lab)
                m["auroc"] = auroc(lo, lab)
                per_t.append(m)
            agg = {}
            for met in ("accuracy", "precision", "recall", "f1", "yes_ratio", "auroc"):
                agg[met] = _nan_stats([m[met] for m in per_t])
            # item-level correctness averaged over templates, clustered by image
            corr: Dict[tuple, list] = defaultdict(list)
            img_of: Dict[tuple, int] = {}
            for r in sub:
                k = (r["image_id"], r["category"], r["kind"])
                corr[k].append(float((r[key[sysname]] > threshold) == bool(r["label"])))
                img_of[k] = r["image_id"]
            ks = sorted(corr)
            est, lo_, hi_ = bootstrap_ci([float(np.mean(corr[k])) for k in ks], n_boot=n_boot, seed=seed,
                                         clusters=[img_of[k] for k in ks])
            agg["accuracy_ci95"] = [lo_, hi_]
            agg["accuracy_template_avg"] = est
            res[sysname] = {"per_template": per_t, "summary": agg}
        if "blind" in res:
            res["model_minus_blind_accuracy"] = (res["model"]["summary"]["accuracy"]["mean"]
                                                 - res["blind"]["summary"]["accuracy"]["mean"])
        out["subsets"][nk] = res
    return out


def item_level(rows: List[dict], threshold: float = 0.0) -> List[dict]:
    """One record per (image, category, kind): mean log-odds over templates, for model and blind."""
    g: Dict[tuple, list] = defaultdict(list)
    for r in rows:
        g[(r["image_id"], r["category"], r["kind"], r["label"])].append(r)
    items = []
    for (img, cat, kind, label), rs in sorted(g.items()):
        m = float(np.mean([r["logodds"] for r in rs]))
        b = [r["blind_logodds"] for r in rs if r["blind_logodds"] is not None]
        bm = float(np.mean(b)) if b else None
        items.append({"image_id": img, "category": cat, "kind": kind, "label": label, "mean_logodds": m,
                      "pred": int(m > threshold), "blind_mean_logodds": bm,
                      "blind_pred": (int(bm > threshold) if bm is not None else None)})
    return items


def failure_cases(items: List[dict], top_k: int = 50) -> List[dict]:
    """False positives (answered 'yes' for an annotated-absent object) and false negatives, most confident first.
    Labels are CANDIDATES: COCO annotation gaps can make a false positive a true positive, and a blind model
    answering 'yes' is evidence of language-prior dependence, not proof."""
    fps = [i for i in items if i["pred"] == 1 and i["label"] == 0]
    fns = [i for i in items if i["pred"] == 0 and i["label"] == 1]
    fps.sort(key=lambda i: -i["mean_logodds"])
    fns.sort(key=lambda i: i["mean_logodds"])
    out = []
    for i in fps[:top_k]:
        out.append({**i, "category_candidate": "object_hallucination",
                    "language_prior_candidate": bool(i["blind_pred"] == 1) if i["blind_pred"] is not None else None})
    for i in fns[:top_k]:
        out.append({**i, "category_candidate": "object_omission (visual blindness candidate)",
                    "language_prior_candidate": None})
    return out
