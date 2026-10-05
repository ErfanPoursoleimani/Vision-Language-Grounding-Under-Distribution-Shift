#!/usr/bin/env python
"""python scripts/analyze_probes.py --run-dir results/<run>  [--out analysis.json]
Reproducible diagnostics over per_example.jsonl / per_item.jsonl of a pope_style run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import analysis as A  # noqa: E402


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(a.run_dir)
    rows, items = read(d / "per_example.jsonl"), read(d / "per_item.jsonl")
    th = A.calibrate_thresholds(rows)
    res = {"pooled_auroc_by_template": A.pooled_auroc_by_template(rows),
           "dev_fit_thresholds_optimistic_if_same_split": th,
           "balanced_accuracy_raw_vs_fit": {
               t: {"raw_threshold_0": A.balanced_accuracy(
                       __import__("numpy").array([r["logodds"] for r in rows if r["template_idx"] == t and r["label"] == 1]),
                       __import__("numpy").array([r["logodds"] for r in rows if r["template_idx"] == t and r["label"] == 0]), 0.0),
                   "fit": A.best_threshold([r["logodds"] for r in rows if r["template_idx"] == t and r["label"] == 1],
                                           [r["logodds"] for r in rows if r["template_idx"] == t and r["label"] == 0])[1]}
               for t in th},
           "template_agreement_pearson": A.template_agreement(rows),
           "negative_overlap_between_kinds": A.negative_overlap(items),
           "fp_rate_by_kind_at_item_threshold": A.fp_rate_by_kind(items),
           "language_prior_share": A.language_prior_share(items),
           "per_category": A.per_category(items)}
    text = json.dumps(res, indent=2, default=float)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
