#!/usr/bin/env python
"""python scripts/run_evaluation.py --config configs/evaluation/<name>.yaml [--output-dir results] [--dry-run]

Writes results/<name>_<timestamp>/ with: manifest.json, per_example.jsonl, per_item.jsonl, results.json,
results.csv, failure_cases.jsonl, plots/*.png. Tasks: pope_style (existence probes + blind baseline).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.engine import TASKS  # noqa: E402
from src.models import build_model  # noqa: E402

REQUIRED = ("name", "task", "model", "dataset")


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"config {path} is missing keys: {missing}")
    if cfg["task"] not in TASKS:
        raise SystemExit(f"unknown task {cfg['task']!r}; available: {sorted(TASKS)}")
    if "from_file" in cfg["model"]:
        cfg["model"] = yaml.safe_load(Path(cfg["model"]["from_file"]).read_text())
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--dry-run", action="store_true", help="validate config, build (not load) the model, write nothing")
    a = ap.parse_args()
    cfg = load_config(a.config)
    model = build_model(cfg["model"])
    print(f"[ok] config '{cfg['name']}' task={cfg['task']} model={model.name} testing_only={bool(model.metadata().get('testing_only'))}")
    if a.dry_run:
        return 0
    out = TASKS[cfg["task"]](cfg, model, a.output_dir)
    print("results written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
