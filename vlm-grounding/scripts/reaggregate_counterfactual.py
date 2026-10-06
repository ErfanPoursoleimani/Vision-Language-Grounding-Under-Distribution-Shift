#!/usr/bin/env python
"""Recompute results from a finished counterfactual run WITHOUT re-running the model.

    python scripts/reaggregate_counterfactual.py --run-dir results/<run> [--out results_v2.json]
Uses per_example.jsonl and thresholds.json of that run with the CURRENT metric code (e.g. after the persistence
definition was fixed). Raw model outputs are never modified."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.counterfactual import aggregate_counterfactual  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--thresholds", help="default: <run-dir>/thresholds.json")
    ap.add_argument("--out", default="results_v2.json")
    a = ap.parse_args()
    d = Path(a.run_dir)
    rows = [json.loads(l) for l in (d / "per_example.jsonl").read_text().splitlines() if l.strip()]
    tf = Path(a.thresholds) if a.thresholds else d / "thresholds.json"
    th = {int(k): float(v) for k, v in json.loads(tf.read_text())["thresholds"].items()} if tf.exists() else None
    res, _ = aggregate_counterfactual(rows, th)
    res["threshold_mode"] = f"re-aggregated from {tf if th else 'raw threshold 0'}"
    (d / a.out).write_text(json.dumps(res, indent=2, default=float))
    print("wrote", d / a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
