from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ("accuracy", "precision", "recall", "f1", "yes_ratio", "auroc")


def _jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_probe_outputs(out: Path, rows, items, summary, failures, manifest) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    _jsonl(out / "per_example.jsonl", rows)
    _jsonl(out / "per_item.jsonl", items)
    _jsonl(out / "failure_cases.jsonl", failures)
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "system", "template"] + list(METRICS) + ["n"])
        for sub, res in summary["subsets"].items():
            for sysname in ("model", "blind"):
                if sysname not in res:
                    continue
                for t, m in enumerate(res[sysname]["per_template"]):
                    w.writerow([sub, sysname, t] + [m[k] for k in METRICS] + [m["n"]])
                s = res[sysname]["summary"]
                w.writerow([sub, sysname, "mean"] + [s[k]["mean"] for k in METRICS] + [""])
    _plots(out / "plots", summary, items)


def _plots(d: Path, summary: dict, items) -> None:
    subs = list(summary["subsets"])
    if not subs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = range(len(subs))
    for off, sysname in ((-0.2, "model"), (0.2, "blind")):
        vals, err = [], [[], []]
        for s in subs:
            r = summary["subsets"][s].get(sysname)
            if r is None:
                vals.append(float("nan")); err[0].append(0); err[1].append(0); continue
            a = r["summary"]["accuracy"]["mean"]; lo, hi = r["summary"]["accuracy_ci95"]
            vals.append(a); err[0].append(max(0, a - lo)); err[1].append(max(0, hi - a))
        ax[0].bar([i + off for i in x], vals, 0.4, yerr=err, label=sysname, capsize=3)
        yr = [summary["subsets"][s].get(sysname, {}).get("summary", {}).get("yes_ratio", {}).get("mean", float("nan")) for s in subs]
        ax[1].bar([i + off for i in x], yr, 0.4, label=sysname)
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(subs); ax[0].set_ylim(0, 1); ax[0].set_title("accuracy (95% CI over images)")
    ax[0].axhline(0.5, color="gray", lw=0.8, ls="--"); ax[0].legend()
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(subs); ax[1].set_ylim(0, 1); ax[1].set_title("yes-ratio (0.5 = unbiased)")
    ax[1].axhline(0.5, color="gray", lw=0.8, ls="--")
    fig.tight_layout(); fig.savefig(d / "accuracy_yes_ratio.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for lab, name in ((1, "present"), (0, "absent")):
        ax.hist([i["mean_logodds"] for i in items if i["label"] == lab], bins=30, alpha=0.6, label=name)
    ax.axvline(summary["threshold"], color="k", lw=0.8); ax.set_xlabel("log-odds(yes) - log-odds(no)"); ax.legend()
    ax.set_title("model log-odds by ground truth"); fig.tight_layout(); fig.savefig(d / "logodds_hist.png", dpi=150); plt.close(fig)
