#!/usr/bin/env python
"""H1 at probe level on a finished pope_style run.   python scripts/analyze_h1.py --run-dir results/<run> [--audit audit/audit_template.csv]
Needs the run's per_item.jsonl (written by every pope_style run with blind_baseline: true)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.h1 import apply_audit, audit_summary, h1_detection, merge_audits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--audit", nargs="+", help="one or more filled audit CSVs (relabels present/visible, drops ambiguous); later files win on conflicts")
    ap.add_argument("--out", default="h1.json")
    a = ap.parse_args()
    d = Path(a.run_dir)
    items = [json.loads(l) for l in (d / "per_item.jsonl").read_text().splitlines() if l.strip()]
    summary = None
    if a.audit:
        files = []
        for path in a.audit:
            with open(path, newline="", encoding="utf-8") as f:
                files.append(list(csv.DictReader(f)))
        rows, conflicts = merge_audits(files)  # raises on unknown verdicts
        summary = audit_summary(rows)
        summary["n_files"] = len(files)
        summary["n_items_judged_differently_in_two_files"] = len(conflicts)
        summary["conflicts"] = conflicts
        n_before = len(items)
        items = apply_audit(items, rows)
        summary["items_dropped_as_ambiguous"] = n_before - len(items)
    res = h1_detection(items)
    res["audit_applied"] = a.audit
    res["audit_summary"] = summary
    (d / a.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
