"""Robustness of existence probing under label-preserving degradations (protocol M7).

For every (op, severity) the SAME items are re-asked on degraded images and compared with the clean run:
threshold-free AUROC (and its drop with a CI that resamples images), accuracy at thresholds frozen on CLEAN dev data,
yes-ratio, consistency P(prediction unchanged vs clean) and mean |change in log-odds|.
"""
from __future__ import annotations

import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..metrics.classification import auroc
from ..perturbations.image_ops import OPS, apply
from .counterfactual import cluster_bootstrap
from .probes import make_question


def run_perturbed_probes(model, items, load_image, templates: Sequence[str], ops: Sequence[str], severities: Sequence[int],
                         seed: int = 0, progress: bool = True) -> List[dict]:
    by_img: Dict[int, list] = defaultdict(list)
    for it in items:
        by_img[it.image_id].append(it)
    rows = []
    conds = [("clean", 0)] + [(o, s) for o in ops for s in severities]
    for image_id in tqdm(sorted(by_img), disable=not progress, desc="images"):
        clean = load_image(image_id)
        for op, sev in conds:
            img = clean if op == "clean" else apply(clean, op, sev, seed=image_id + 1000003 * seed)
            for it in by_img[image_id]:
                for ti, tpl in enumerate(templates):
                    q = make_question(tpl, it.category)
                    rows.append({"image_id": image_id, "category": it.category, "label": it.label, "kind": it.kind,
                                 "template_idx": ti, "op": op, "severity": sev, "question": q,
                                 "logodds": float(model.yes_no_logodds(img, q))})
    return rows


def _collect(rows: List[dict], op: str, sev: int):
    d: Dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r["op"] == op and r["severity"] == sev:
            d[(r["image_id"], r["category"], r["kind"])][r["template_idx"]] = (r["image_id"], r["label"], r["logodds"])
    return d


def _units(rows: List[dict], op: str, sev: int, n_t: int) -> List[dict]:
    """One unit per item with per-template log-odds for the clean and the degraded image (same item, same templates)."""
    clean, pert = _collect(rows, "clean", 0), _collect(rows, op, sev)
    out = []
    for k in sorted(pert):
        if k in clean and all(t in clean[k] and t in pert[k] for t in range(n_t)):
            out.append({"scene_id": pert[k][0][0], "label": pert[k][0][1],
                        "clean": [clean[k][t][2] for t in range(n_t)], "pert": [pert[k][t][2] for t in range(n_t)]})
    return out


def _auc(us, key, n_t):
    labs = [u["label"] for u in us]
    if len(set(labs)) < 2:
        return float("nan")
    return float(np.mean([auroc([u[key][t] for u in us], labs) for t in range(n_t)]))


def aggregate_robustness(rows: List[dict], thresholds: Optional[Dict[int, float]], n_boot: int = 300, seed: int = 0) -> dict:
    n_t = max(r["template_idx"] for r in rows) + 1
    th = [(thresholds or {}).get(t, 0.0) for t in range(n_t)]
    ops = sorted({r["op"] for r in rows} - {"clean"}, key=lambda o: OPS.index(o) if o in OPS else 99)
    res: dict = {"thresholds": thresholds, "n_templates": n_t, "conditions": {}}
    clean = _collect(rows, "clean", 0)
    cu = [{"scene_id": v[0][0], "label": v[0][1], "clean": [v[t][2] for t in range(n_t)]}
          for v in (clean[k] for k in sorted(clean)) if all(t in v for t in range(n_t))]
    res["clean"] = {"n_items": len(cu), "AUROC": _auc(cu, "clean", n_t),
                    "accuracy": float(np.mean([(u["clean"][t] > th[t]) == bool(u["label"]) for u in cu for t in range(n_t)])),
                    "yes_ratio": float(np.mean([u["clean"][t] > th[t] for u in cu for t in range(n_t)]))}
    for op in ops:
        res["conditions"][op] = {}
        for sev in sorted({r["severity"] for r in rows if r["op"] == op}):
            us = _units(rows, op, sev, n_t)
            if not us:
                continue
            drop = lambda x: _auc(x, "clean", n_t) - _auc(x, "pert", n_t)  # noqa: E731
            est, lo, hi = cluster_bootstrap(us, drop, n_boot, seed)
            res["conditions"][op][str(sev)] = {
                "n_items": len(us), "AUROC": _auc(us, "pert", n_t),
                "AUROC_drop_vs_clean": {"est": est, "lo": lo, "hi": hi},
                "accuracy": float(np.mean([(u["pert"][t] > th[t]) == bool(u["label"]) for u in us for t in range(n_t)])),
                "yes_ratio": float(np.mean([u["pert"][t] > th[t] for u in us for t in range(n_t)])),
                "consistency_with_clean": float(np.mean([(u["pert"][t] > th[t]) == (u["clean"][t] > th[t]) for u in us for t in range(n_t)])),
                "mean_abs_logodds_change": float(np.mean([abs(u["pert"][t] - u["clean"][t]) for u in us for t in range(n_t)]))}
    return res


def run_pope_perturbed(cfg: dict, model, out_root, allow_testing_only: bool = False, progress: bool = True,
                       test_log="results/test_access_log.jsonl"):
    from ..datasets.coco import load_coco_instances
    from ..datasets.pope_style import build_pope_style
    from ..datasets.splits import log_test_access, split_ids
    from ..models.cache import CachedVLM
    from .analysis import calibrate_thresholds
    from .engine import _git
    from .probes import DEFAULT_TEMPLATES
    testing_only = bool(model.metadata().get("testing_only"))
    if testing_only and not allow_testing_only:
        raise SystemExit("refusing to write results for a testing-only model")
    d, pr = cfg["dataset"], cfg.get("probe", {})
    split = d.get("our_split", "dev")
    if split == "test" and not pr.get("thresholds_file"):
        raise SystemExit("test runs must use frozen thresholds: set probe.thresholds_file to the thresholds.json of a dev run")
    images, cats = load_coco_instances(d["root"], d.get("split", "val2017"))
    keep = set(split_ids([im.image_id for im in images], seed=d.get("split_seed", 0))[split])
    items = build_pope_style([im for im in images if im.image_id in keep], cats, n_images=d.get("n_images", 100), seed=d.get("seed", 0))
    paths = {im.image_id: im.path for im in images}
    if split == "test" and test_log and not testing_only:
        log_test_access(test_log, config=cfg["name"], model=model.metadata(), n_items=len(items), git=_git())
    rb = cfg.get("robustness", {})
    ops, sevs = rb.get("ops", list(OPS)), rb.get("severities", [1, 3, 5])
    templates = pr.get("templates", DEFAULT_TEMPLATES[:1])
    m = CachedVLM(model, cfg.get("cache", f"cache/{model.name}.sqlite"))
    t0 = time.perf_counter()
    rows = run_perturbed_probes(m, items, lambda i: Image.open(paths[i]).convert("RGB"), templates, ops, sevs, d.get("seed", 0), progress)
    if pr.get("thresholds_file"):
        th = {int(k): float(v) for k, v in json.loads(Path(pr["thresholds_file"]).read_text())["thresholds"].items()}
        mode = f"frozen from {pr['thresholds_file']}"
    else:
        th = calibrate_thresholds([r for r in rows if r["op"] == "clean"])
        mode = "fit on CLEAN images of this dev split (optimistic); applied unchanged to all degraded conditions"
    res = aggregate_robustness(rows, th)
    res["threshold_mode"] = mode
    out = Path(out_root) / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    (out / "plots").mkdir(parents=True)
    with open(out / "per_example.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    if not pr.get("thresholds_file"):
        (out / "thresholds.json").write_text(json.dumps({"fit_on": f"{cfg['name']} ({split}, clean)", "git": _git(), "thresholds": th}, indent=2))
    _csv_and_plot(out, res)
    (out / "manifest.json").write_text(json.dumps({
        "config": cfg, "model_metadata": model.metadata(), "git_commit": _git(), "python": platform.python_version(),
        "runtime": model.runtime_info() if hasattr(model, "runtime_info") else None, "n_items": len(items), "n_rows": len(rows),
        "seconds": time.perf_counter() - t0, "cache_hits": m.hits, "cache_misses": m.misses, "threshold_mode": mode,
        "note": "label-preserving degradations only; crop/occlusion are handled in the counterfactual track"}, indent=2, default=str))
    return out


def _csv_and_plot(out: Path, res: dict) -> None:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nz = lambda v: v if v == v else 0.0  # noqa: E731
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "severity", "n_items", "AUROC", "AUROC_drop", "drop_lo", "drop_hi", "accuracy", "yes_ratio", "consistency", "mean_abs_dlogodds"])
        w.writerow(["clean", 0, res["clean"]["n_items"], res["clean"]["AUROC"], 0, "", "", res["clean"]["accuracy"], res["clean"]["yes_ratio"], 1, 0])
        for op, d in res["conditions"].items():
            for sev, x in d.items():
                dr = x["AUROC_drop_vs_clean"]
                w.writerow([op, sev, x["n_items"], x["AUROC"], dr["est"], dr["lo"], dr["hi"], x["accuracy"], x["yes_ratio"],
                            x["consistency_with_clean"], x["mean_abs_logodds_change"]])
    if not res["conditions"]:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for op, d in res["conditions"].items():
        sv = sorted(d, key=int)
        xs = [int(s) for s in sv]
        err = [[max(0, nz(d[s]["AUROC_drop_vs_clean"]["est"]) - nz(d[s]["AUROC_drop_vs_clean"]["lo"])) for s in sv],
               [max(0, nz(d[s]["AUROC_drop_vs_clean"]["hi"]) - nz(d[s]["AUROC_drop_vs_clean"]["est"])) for s in sv]]
        ax[0].errorbar(xs, [d[s]["AUROC"] for s in sv], yerr=err, marker="o", capsize=2, label=op)
        ax[1].plot(xs, [d[s]["consistency_with_clean"] for s in sv], marker="o", label=op)
    ax[0].axhline(res["clean"]["AUROC"], color="k", ls="--", lw=0.8, label="clean"); ax[0].axhline(0.5, color="gray", ls=":", lw=0.8)
    ax[0].set_title("AUROC vs severity (bars: CI width of the drop)"); ax[0].set_xlabel("severity"); ax[0].legend(fontsize=7)
    ax[1].set_title("prediction consistency with clean"); ax[1].set_xlabel("severity"); ax[1].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(out / "plots" / "robustness.png", dpi=150); plt.close(fig)
