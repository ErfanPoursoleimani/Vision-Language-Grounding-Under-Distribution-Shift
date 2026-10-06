"""What can the model perceive at all? (original synthetic images only)

Run BEFORE interpreting counterfactual sensitivity: low AFR means nothing about grounding if the model cannot
answer the underlying question. Groups (gold yes vs gold no inside each group, threshold-free AUROC):
  color_atoms    'Is there a red object?'                      present vs absent colours
  shape_atoms    'Is there a circle?'                          present vs absent shapes
  binding        'Is there a red circle?' present vs FOIL      colour+shape both exist but the combination does not
  easy_existence 'Is there a red circle?' present vs absent colour
  relation       'Is the A left of the B?' vs reversed order   (symmetric pairs; gold differs)
Also reported: blind (gray image) AUROC, accuracy at thresholds, yes-ratio, and for relations the rate at which the
model answers YES to both directions of the same pair.
"""
from __future__ import annotations

import json
import platform
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..metrics.classification import auroc
from ..perturbations.synthetic import LeftOf, answer, competence_items, question_text, render
from .counterfactual import cluster_bootstrap, scenes_for_split

GROUPS = {"color_atoms": ["color_atom"], "shape_atoms": ["shape_atom"], "binding": ["exists_true", "foil"],
          "easy_existence": ["exists_true", "easy_neg"], "relation": ["relation"]}


def run_competence(model, scenes, templates=(0, 1, 2), blind=True, seed=0, progress=True) -> List[dict]:
    rows = []
    for sid, scene in tqdm(scenes, disable=not progress, desc="scenes"):
        img = render(scene)
        for ii, (kind, q) in enumerate(competence_items(scene, random.Random(f"{seed}:{sid}:comp"))):
            gold = int(answer(scene, q))
            pair_key = None
            if isinstance(q, LeftOf):
                pair_key = "|".join(sorted([f"{q.c1}-{q.s1}", f"{q.c2}-{q.s2}"]))
            for t in templates:
                text = question_text(q, t)
                rows.append({"scene_id": sid, "item_idx": ii, "kind": kind, "template_idx": t, "question": text,
                             "gold": gold, "pair_key": pair_key, "logodds": float(model.yes_no_logodds(img, text)),
                             "blind_logodds": float(model.yes_no_logodds(None, text)) if blind else None})
    return rows


def aggregate_competence(rows: List[dict], thresholds: Optional[Dict[int, float]] = None, n_boot: int = 300, seed: int = 0) -> dict:
    th = lambda r: (thresholds or {}).get(r["template_idx"], 0.0)  # noqa: E731
    out: dict = {"thresholds": thresholds, "groups": {}}
    for name, kinds in GROUPS.items():
        sub = [r for r in rows if r["kind"] in kinds]
        if not sub or len({r["gold"] for r in sub}) < 2:
            continue
        # unit = (scene, item): mean over templates, so templates do not inflate n
        g: Dict[tuple, list] = defaultdict(list)
        for r in sub:
            g[(r["scene_id"], r["item_idx"])].append(r)
        units = []
        for (sid, ii), rs in g.items():
            units.append({"scene_id": sid, "gold": rs[0]["gold"], "lo": float(np.mean([r["logodds"] for r in rs])),
                          "blo": (float(np.mean([r["blind_logodds"] for r in rs])) if rs[0]["blind_logodds"] is not None else None),
                          "acc": float(np.mean([(r["logodds"] > th(r)) == bool(r["gold"]) for r in rs])),
                          "yes": float(np.mean([r["logodds"] > th(r) for r in rs])),
                          "bacc": (float(np.mean([(r["blind_logodds"] > th(r)) == bool(r["gold"]) for r in rs])) if rs[0]["blind_logodds"] is not None else None)})
        a = lambda us: auroc([u["lo"] for u in us], [u["gold"] for u in us]) if len({u["gold"] for u in us}) > 1 else float("nan")  # noqa: E731
        e, lo, hi = cluster_bootstrap(units, a, n_boot, seed)
        d = {"n_units": len(units), "n_gold_yes": sum(u["gold"] for u in units), "AUROC": {"est": e, "lo": lo, "hi": hi},
             "accuracy": float(np.mean([u["acc"] for u in units])), "yes_ratio": float(np.mean([u["yes"] for u in units]))}
        if units[0]["blo"] is not None:
            b = lambda us: auroc([u["blo"] for u in us], [u["gold"] for u in us]) if len({u["gold"] for u in us}) > 1 else float("nan")  # noqa: E731
            d["blind_AUROC"] = b(units)
            d["blind_accuracy"] = float(np.mean([u["bacc"] for u in units]))
        out["groups"][name] = d
    rel = [r for r in rows if r["kind"] == "relation"]
    if rel:
        g2: Dict[tuple, list] = defaultdict(list)
        for r in rel:
            g2[(r["scene_id"], r["pair_key"], r["template_idx"])].append(r["logodds"] > th(r))
        both = [all(v) for v in g2.values() if len(v) == 2]
        out["relation_both_directions_yes_rate"] = float(np.mean(both)) if both else float("nan")
    return out


def run_synthetic_competence(cfg: dict, model, out_root, allow_testing_only: bool = False, progress: bool = True,
                             test_log="results/test_access_log.jsonl"):
    from ..datasets.splits import log_test_access
    from ..models.cache import CachedVLM
    from .analysis import calibrate_thresholds
    from .engine import _git
    testing_only = bool(model.metadata().get("testing_only"))
    if testing_only and not allow_testing_only:
        raise SystemExit("refusing to write results for a testing-only model")
    d, pr = cfg["dataset"], cfg.get("probe", {})
    split = d.get("our_split", "dev")
    if split == "test" and not pr.get("thresholds_file"):
        raise SystemExit("test runs must use frozen thresholds: set probe.thresholds_file to the thresholds.json of a dev run")
    scenes = scenes_for_split(split, d.get("n_scenes", 40), d.get("seed", 0), d.get("split_seed", 0))
    if split == "test" and test_log and not testing_only:
        log_test_access(test_log, config=cfg["name"], model=model.metadata(), n_scenes=len(scenes), git=_git())
    m = CachedVLM(model, cfg.get("cache", f"cache/{model.name}.sqlite"))
    t0 = time.perf_counter()
    rows = run_competence(m, scenes, tuple(pr.get("templates", (0, 1, 2))), pr.get("blind_baseline", True), d.get("seed", 0), progress)
    if pr.get("thresholds_file"):
        th = {int(k): float(v) for k, v in json.loads(Path(pr["thresholds_file"]).read_text())["thresholds"].items()}
        mode = f"frozen from {pr['thresholds_file']}"
    elif split == "dev":
        th = calibrate_thresholds([{"template_idx": r["template_idx"], "label": r["gold"], "logodds": r["logodds"]} for r in rows])
        mode = "fit on this dev split (optimistic; development only)"
    else:
        th, mode = None, "raw"
    res = aggregate_competence(rows, th)
    res["threshold_mode"] = mode
    out = Path(out_root) / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    (out / "plots").mkdir(parents=True)
    with open(out / "per_example.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    if th is not None and not pr.get("thresholds_file"):
        (out / "thresholds.json").write_text(json.dumps({"fit_on": f"{cfg['name']} ({split})", "git": _git(), "thresholds": th}, indent=2))
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gs = list(res["groups"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_units", "AUROC", "AUROC_lo", "AUROC_hi", "blind_AUROC", "accuracy", "blind_accuracy", "yes_ratio"])
        for g in gs:
            x = res["groups"][g]
            w.writerow([g, x["n_units"], x["AUROC"]["est"], x["AUROC"]["lo"], x["AUROC"]["hi"], x.get("blind_AUROC"), x["accuracy"], x.get("blind_accuracy"), x["yes_ratio"]])
    if gs:
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        xs = np.arange(len(gs))
        est = [res["groups"][g]["AUROC"]["est"] for g in gs]
        err = [[max(0, res["groups"][g]["AUROC"]["est"] - res["groups"][g]["AUROC"]["lo"]) for g in gs],
               [max(0, res["groups"][g]["AUROC"]["hi"] - res["groups"][g]["AUROC"]["est"]) for g in gs]]
        ax.bar(xs - 0.2, est, 0.4, yerr=err, capsize=3, label="model")
        ax.bar(xs + 0.2, [res["groups"][g].get("blind_AUROC", np.nan) for g in gs], 0.4, label="blind (gray image)")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8); ax.set_ylim(0, 1.05); ax.set_xticks(xs); ax.set_xticklabels(gs, fontsize=8)
        ax.set_title("competence on ORIGINAL images: AUROC (0.5 = chance)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "plots" / "competence.png", dpi=150); plt.close(fig)
    (out / "manifest.json").write_text(json.dumps({
        "config": cfg, "model_metadata": model.metadata(), "git_commit": _git(), "python": platform.python_version(),
        "runtime": model.runtime_info() if hasattr(model, "runtime_info") else None, "n_scenes": len(scenes),
        "n_rows": len(rows), "seconds": time.perf_counter() - t0, "cache_hits": m.hits, "cache_misses": m.misses,
        "threshold_mode": mode}, indent=2, default=str))
    return out
