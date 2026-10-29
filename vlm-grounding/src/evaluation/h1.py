"""H1 (detection) at probe level: does visual information gain separate grounded 'yes' answers from hallucinated ones?

Population = items the model answers 'yes' to (its "mentions"). Target: label 1 = object really present (grounded),
0 = annotated absent (candidate hallucination). Scores (all oriented so that a HIGH score should mean 'grounded'):
  conf   log-odds(yes) with the image                         (confidence baseline; for a yes/no answer, entropy is monotone in it)
  pmi    log-odds(yes | image) - log-odds(yes | no image)     (S_global of the protocol, instantiated on odds)
  -prior log-odds(yes | no image)                             (what the language prior alone says; high prior -> more likely hallucination)
Protocol decision rules [proposed]: F1 survives if the 95% lower bound of AUROC(pmi) > 0.5; F2 if the within-category
AUROC(pmi) lower bound > 0.5 (controls object frequency); F3 if AUROC(pmi) - AUROC(conf) >= 0.02 with a positive lower
bound. One model, one dev split, COCO label noise: NOT evidence for the hypothesis on its own (needs >= 2 families, test split).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ..metrics.classification import auroc
from .counterfactual import cluster_bootstrap

AUDIT_VERDICTS = ("hallucination", "present_unannotated", "ambiguous")
# The audit sheets print the question the model was asked ("is there a cup?"). Answering it with yes/no is natural, so:
#   yes  = the object IS visible in the image  -> the model was right, the COCO label was wrong  -> present_unannotated
#   no   = the object is NOT visible           -> the model was wrong                           -> hallucination
_ALIASES = {"hallucination": "hallucination", "halluc": "hallucination", "h": "hallucination", "no": "hallucination", "n": "hallucination",
            "absent": "hallucination", "not_present": "hallucination",
            "present_unannotated": "present_unannotated", "present": "present_unannotated", "yes": "present_unannotated",
            "y": "present_unannotated", "visible": "present_unannotated",
            "ambiguous": "ambiguous", "unsure": "ambiguous", "?": "ambiguous", "maybe": "ambiguous", "unclear": "ambiguous", "": ""}


def normalize_verdict(v: Optional[str]) -> str:
    key = (v or "").strip().lower().replace(" ", "_")
    if key not in _ALIASES:
        raise ValueError(f"unknown audit verdict {v!r}; allowed: {AUDIT_VERDICTS} or yes/no (yes = object visible, no = not visible)")
    return _ALIASES[key]


def audit_summary(audit_rows: List[dict]) -> dict:
    from collections import Counter
    c = Counter(normalize_verdict(r.get("verdict")) for r in audit_rows)
    judged = sum(c[v] for v in AUDIT_VERDICTS)
    return {"n_rows": len(audit_rows), "n_judged": judged, "n_blank": c[""], **{v: c[v] for v in AUDIT_VERDICTS},
            "share_present_unannotated_among_judged": (c["present_unannotated"] / judged) if judged else None}


def merge_audits(files: List[List[dict]]) -> tuple:
    """Merge audit CSVs (later files win on conflicts). Returns (rows, conflicts) where conflicts lists items judged
    differently in two files: a direct measure of how consistent the human judgement is."""
    seen: dict = {}
    conflicts = []
    for rows in files:
        for r in rows:
            v = normalize_verdict(r.get("verdict"))
            if not v:
                continue
            k = (int(r["image_id"]), r["category"])
            if k in seen and seen[k][0] != v:
                conflicts.append({"image_id": k[0], "category": k[1], "earlier": seen[k][0], "later": v})
            seen[k] = (v, r)
    merged = [{**r, "verdict": v} for v, r in seen.values()]
    return merged, conflicts


def apply_audit(items: List[dict], audit_rows: List[dict]) -> List[dict]:
    """Relabel by human verdict: present_unannotated -> label 1 (it IS there), ambiguous -> dropped, hallucination/blank -> unchanged.
    Unknown verdict strings raise (they used to be ignored silently)."""
    v = {(int(r["image_id"]), r["category"]): normalize_verdict(r.get("verdict")) for r in audit_rows}
    out = []
    for it in items:
        verdict = v.get((int(it["image_id"]), it["category"]), "")
        if verdict == "ambiguous":
            continue
        out.append({**it, "label": 1} if verdict == "present_unannotated" else it)
    return out


def _units(items: List[dict]) -> List[dict]:
    seen, out = set(), []
    for i in items:  # an (image, category) pair can appear under several negative kinds: count it once
        k = (i["image_id"], i["category"])
        if k in seen or i.get("blind_mean_logodds") is None:
            continue
        seen.add(k)
        out.append({"scene_id": i["image_id"], "label": i["label"], "cat": i["category"], "conf": i["mean_logodds"],
                    "prior": i["blind_mean_logodds"], "pmi": i["mean_logodds"] - i["blind_mean_logodds"], "pred": i["pred"]})
    return out


def _auc(us: List[dict], key: str, sign: float = 1.0) -> float:
    labs = [u["label"] for u in us]
    return auroc([sign * u[key] for u in us], labs) if len(set(labs)) > 1 else float("nan")


def _within(us: List[dict], key: str, min_cat: int) -> float:
    by: Dict[str, list] = defaultdict(list)
    for u in us:
        by[u["cat"]].append(u)
    num = den = 0.0
    for xs in by.values():
        n1 = sum(u["label"] for u in xs)
        n0 = len(xs) - n1
        if len(xs) >= min_cat and n1 and n0:
            num += _auc(xs, key) * n1 * n0
            den += n1 * n0
    return num / den if den else float("nan")


def _population(us: List[dict], n_boot: int, seed: int, min_cat: int) -> dict:
    n1 = sum(u["label"] for u in us)
    res: dict = {"n": len(us), "n_grounded": n1, "n_hallucinated_candidates": len(us) - n1}
    if n1 < 5 or len(us) - n1 < 5:
        res["note"] = "too few items of one class for a meaningful AUROC"
        return res
    ci = lambda fn: dict(zip(("est", "lo", "hi"), cluster_bootstrap(us, fn, n_boot, seed)))  # noqa: E731
    res["AUROC_conf"] = ci(lambda x: _auc(x, "conf"))
    res["AUROC_pmi"] = ci(lambda x: _auc(x, "pmi"))
    res["AUROC_neg_prior"] = ci(lambda x: _auc(x, "prior", -1.0))
    res["AUROC_pmi_minus_conf"] = ci(lambda x: _auc(x, "pmi") - _auc(x, "conf"))
    res["AUROC_pmi_within_category"] = ci(lambda x: _within(x, "pmi", min_cat))
    res["AUROC_conf_within_category"] = ci(lambda x: _within(x, "conf", min_cat))
    d, w = res["AUROC_pmi"], res["AUROC_pmi_within_category"]
    diff = res["AUROC_pmi_minus_conf"]
    res["falsification_rules_survived"] = {
        "F1_pmi_AUROC_lower_bound_above_0.5": bool(d["lo"] > 0.5),
        "F2_within_category_lower_bound_above_0.5": bool(w["lo"] > 0.5),
        "F3_pmi_beats_confidence_by_0.02": bool(diff["est"] >= 0.02 and diff["lo"] > 0.0)}
    return res


def h1_detection(items: List[dict], n_boot: int = 500, seed: int = 0, min_cat: int = 4) -> dict:
    us = _units(items)
    return {"definition": "probe-level H1; see module docstring", "n_unique_items": len(us),
            "mentions_only (model answered yes)": _population([u for u in us if u["pred"] == 1], n_boot, seed, min_cat),
            "all_items": _population(us, n_boot, seed, min_cat),
            "caveat": "single model, dev split, COCO annotation gaps; hallucination labels are candidates until audited"}
