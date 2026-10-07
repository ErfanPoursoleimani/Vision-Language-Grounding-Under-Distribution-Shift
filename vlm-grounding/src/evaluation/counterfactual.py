"""Counterfactual evaluation on synthetic scene pairs (protocol M2, M3, M4).

Unit of analysis = one probe (a question) on one (original, edited) image pair, averaged over question paraphrases.
* changed probe  : gold answer differs between the two images  -> a sensitive model must change its answer
* unchanged probe: gold answer identical (control edit, background edit, or binding foil) -> a stable model must not
AFR = P(answer flips | changed), SFR = P(answer flips | unchanged), CSS = AFR - SFR.
CPA = P(correct on both images | changed). Confidence intervals resample whole scenes.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from ..datasets.splits import assign_split
from ..metrics.classification import auroc
from ..perturbations.synthetic import (EXISTS_TEMPLATES, STYLES, Pair, build_pairs, question_text, random_scene, render)

N_TEMPLATES = len(EXISTS_TEMPLATES)


def scenes_for_split(split: str, n: int, seed: int, split_seed: int = 0, fractions=None, style: str = "small") -> list:
    fr = fractions or {"dev": 0.3, "test": 0.7}
    out, sid = [], 0
    while len(out) < n:
        if assign_split(sid, split_seed, fr) == split:
            out.append((sid, random_scene(random.Random(f"{seed}:{sid}"), style=STYLES[style])))
        sid += 1
    return out


def make_pairs(split: str, n_scenes: int, seed: int, split_seed: int = 0, style: str = "small") -> List[Pair]:
    pairs: List[Pair] = []
    for sid, scene in scenes_for_split(split, n_scenes, seed, split_seed, style=style):
        pairs += build_pairs(sid, scene, random.Random(f"{seed}:{sid}:pairs"), STYLES[style])
    return pairs


def run_counterfactual(model, pairs: Sequence[Pair], templates: Sequence[int] = (0, 1, 2), blind: bool = True,
                       progress: bool = True) -> List[dict]:
    rows: List[dict] = []
    for p in tqdm(pairs, disable=not progress, desc="pairs"):
        io, ic = render(p.orig), render(p.cf)
        for pi, pr in enumerate(p.probes):
            for t in templates:
                q = question_text(pr.question, t)
                rows.append({"pair_id": p.pair_id, "scene_id": p.scene_id, "edit_type": p.edit_type, "role": p.role,
                             "probe_idx": pi, "probe_kind": pr.kind, "template_idx": t, "question": q,
                             "gold_orig": int(pr.gold_orig), "gold_cf": int(pr.gold_cf),
                             "lo_orig": float(model.yes_no_logodds(io, q)), "lo_cf": float(model.yes_no_logodds(ic, q)),
                             "blind_lo": float(model.yes_no_logodds(None, q)) if blind else None,
                             "area_frac": p.area_frac, "area_ratio": p.area_ratio})
    return rows


def threshold_rows(rows: List[dict]) -> List[dict]:
    """Stack original+edited evaluations as labelled rows for threshold calibration (dev only)."""
    out = []
    for r in rows:
        out.append({"template_idx": r["template_idx"], "label": r["gold_orig"], "logodds": r["lo_orig"]})
        out.append({"template_idx": r["template_idx"], "label": r["gold_cf"], "logodds": r["lo_cf"]})
    return out


# ------------------------------------------------------------------ units
def make_units(rows: List[dict], thresholds: Optional[Dict[int, float]]) -> List[dict]:
    g: Dict[tuple, list] = defaultdict(list)
    for r in rows:
        g[(r["pair_id"], r["probe_idx"])].append(r)
    units = []
    for (pid, pi), rs in g.items():
        th = lambda r: (thresholds or {}).get(r["template_idx"], 0.0)  # noqa: E731
        po = [r["lo_orig"] > th(r) for r in rs]
        pc = [r["lo_cf"] > th(r) for r in rs]
        r0 = rs[0]
        go, gc = bool(r0["gold_orig"]), bool(r0["gold_cf"])
        d_lo = float(np.mean([r["lo_cf"] - r["lo_orig"] for r in rs]))
        sign = (int(gc) - int(go))  # +1: no->yes, -1: yes->no, 0: unchanged
        yn = go and not gc
        ny = (not go) and gc
        units.append({
            "n_yes_to_no": len(rs) if yn else 0, "n_pred_o_yes": sum(po) if yn else 0,
            "n_persist": sum(a and b for a, b in zip(po, pc)) if yn else 0, "n_false_yes_after": sum(pc) if yn else 0,
            "n_no_to_yes": len(rs) if ny else 0, "n_pred_o_no": sum(not a for a in po) if ny else 0,
            "n_nondet": sum((not a) and (not b) for a, b in zip(po, pc)) if ny else 0,
            "n_miss_after": sum(not b for b in pc) if ny else 0,
            "pair_id": pid, "scene_id": r0["scene_id"], "edit_type": r0["edit_type"], "role": r0["role"],
            "kind": r0["probe_kind"], "changed": go != gc, "gold_o": go, "gold_c": gc,
            "flip": float(np.mean([a != b for a, b in zip(po, pc)])),
            "correct_both": float(np.mean([(a == go) and (b == gc) for a, b in zip(po, pc)])),
            "correct_o": float(np.mean([a == go for a in po])), "correct_c": float(np.mean([b == gc for b in pc])),
            "pred_o": float(np.mean(po)), "pred_c": float(np.mean(pc)),
            "aligned_shift": sign * d_lo if sign else None, "abs_shift": abs(d_lo), "d_lo": d_lo,
            "blind_pred": (float(np.mean([r["blind_lo"] > th(r) for r in rs])) if r0["blind_lo"] is not None else None),
            "question": r0["question"]})
    return units


def _m(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def _unstable(u):  # unchanged probes that act as controls
    return (not u["changed"]) and (u["role"] in ("control", "background") or u["kind"] == "foil")


def stat_afr(us): return _m([u["flip"] for u in us if u["changed"]])
def stat_sfr(us): return _m([u["flip"] for u in us if _unstable(u)])
def stat_css(us): return stat_afr(us) - stat_sfr(us)
def stat_cpa(us): return _m([u["correct_both"] for u in us if u["changed"]])


def cluster_bootstrap(units: List[dict], fn: Callable, n_boot: int = 500, seed: int = 0):
    by: Dict[int, list] = defaultdict(list)
    for u in units:
        by[u["scene_id"]].append(u)
    ids = sorted(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        v = fn([u for i in pick for u in by[i]])
        if not np.isnan(v):
            vals.append(v)
    est = fn(units)
    if not vals:
        return est, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def shift_auroc(us: List[dict]) -> float:
    ch = [u["abs_shift"] for u in us if u["changed"]]
    un = [u["abs_shift"] for u in us if _unstable(u)]
    if not ch or not un:
        return float("nan")
    return auroc(ch + un, [1] * len(ch) + [0] * len(un))


def aggregate_counterfactual(rows: List[dict], thresholds: Optional[Dict[int, float]] = None, n_boot: int = 500,
                             seed: int = 0) -> dict:
    units = make_units(rows, thresholds)
    ci = lambda fn, us: dict(zip(("est", "lo", "hi"), cluster_bootstrap(us, fn, n_boot, seed)))  # noqa: E731
    res: dict = {"thresholds": thresholds, "n_units": len(units), "n_scenes": len({u["scene_id"] for u in units}),
                 "overall": {"AFR": ci(stat_afr, units), "SFR": ci(stat_sfr, units), "CSS": ci(stat_css, units),
                             "CPA": ci(stat_cpa, units), "shift_auroc": ci(shift_auroc, units)},
                 "by_edit_type": {}, "sfr_by_source": {}, "blind": {}}
    for et in sorted({u["edit_type"] for u in units if u["role"] == "relevant"}):
        sub = [u for u in units if u["edit_type"] == et and (u["role"] in ("relevant", "control"))]
        ch = [u for u in sub if u["role"] == "relevant" and u["changed"]]
        ctrl = [u for u in sub if u["role"] == "control"]
        afr = lambda us: _m([u["flip"] for u in us if u["role"] == "relevant" and u["changed"]])  # noqa: E731
        sfr = lambda us: _m([u["flip"] for u in us if u["role"] == "control"])  # noqa: E731
        css = lambda us: afr(us) - sfr(us)  # noqa: E731
        shift = lambda us: (auroc([u["abs_shift"] for u in us if u["role"] == "relevant" and u["changed"]] + [u["abs_shift"] for u in us if u["role"] == "control"],  # noqa: E731
                                  [1] * sum(1 for u in us if u["role"] == "relevant" and u["changed"]) + [0] * sum(1 for u in us if u["role"] == "control"))
                            if any(u["role"] == "control" for u in us) and any(u["role"] == "relevant" and u["changed"] for u in us) else float("nan"))
        d = {"n_changed_probes": len(ch), "n_control_probes": len(ctrl), "AFR": ci(afr, sub), "SFR_control": ci(sfr, sub),
             "CSS": ci(css, sub), "CPA": ci(lambda us: _m([u["correct_both"] for u in us if u["role"] == "relevant" and u["changed"]]), sub),
             "shift_auroc": ci(shift, sub),
             "mean_aligned_shift_changed": _m([u["aligned_shift"] for u in ch]),
             "mean_abs_shift_control": _m([u["abs_shift"] for u in ctrl])}
        et_units = [u for u in units if u["edit_type"] == et and u["role"] == "relevant"]
        ratio = lambda num, den: (lambda us: (sum(u[num] for u in us) / sum(u[den] for u in us)) if sum(u[den] for u in us) else float("nan"))  # noqa: E731
        if any(u["n_yes_to_no"] for u in et_units):
            # persistence: P(still 'yes' on the edited image | evidence for 'yes' removed AND model said 'yes' before)
            d["persistence_rate"] = ci(ratio("n_persist", "n_pred_o_yes"), et_units)
            # unconditional: P('yes' on the edited image | evidence removed)  (previously mislabeled 'persistence_rate')
            d["false_yes_after_edit"] = ci(ratio("n_false_yes_after", "n_yes_to_no"), et_units)
        if any(u["n_no_to_yes"] for u in et_units):
            d["non_detection_rate"] = ci(ratio("n_nondet", "n_pred_o_no"), et_units)   # said 'no' before, still 'no' after evidence added
            d["miss_after_edit"] = ci(ratio("n_miss_after", "n_no_to_yes"), et_units)
        res["by_edit_type"][et] = d
    for name, f in (("control_edits", lambda u: u["role"] == "control"), ("background_edits", lambda u: u["role"] == "background"),
                    ("binding_foils", lambda u: u["kind"] == "foil" and not u["changed"])):
        us = [u for u in units if f(u) and not u["changed"]]
        res["sfr_by_source"][name] = {"n": len(us), "SFR": _m([u["flip"] for u in us])}
    bl = [u for u in units if u["blind_pred"] is not None]
    if bl:
        res["blind"] = {"accuracy_on_original": _m([float(round(u["blind_pred"]) == int(u["gold_o"])) for u in bl]),
                        "yes_ratio": _m([u["blind_pred"] for u in bl]),
                        "CPA_changed": _m([0.0 if u["changed"] else None for u in bl]),
                        "note": "the blind answer cannot depend on the image, so it cannot be correct on both images of a changed probe"}
    ctrl_units = [u for u in units if u["role"] == "control" and not u["changed"]]
    ratios = {r["pair_id"]: r["area_ratio"] for r in rows if r["role"] == "control" and r["area_ratio"]}
    matched = [u for u in ctrl_units if u["pair_id"] in ratios and 0.8 <= ratios[u["pair_id"]] <= 1.25]
    res["control_area_matching"] = {"n_control_pairs_with_ratio": len(ratios),
                                    "fraction_within_20pct": (sum(0.8 <= v <= 1.25 for v in ratios.values()) / len(ratios)) if ratios else None,
                                    "SFR_all_controls": _m([u["flip"] for u in ctrl_units]),
                                    "SFR_area_matched_controls": _m([u["flip"] for u in matched])}
    res["accuracy"] = {"original": _m([u["correct_o"] for u in units]), "edited": _m([u["correct_c"] for u in units]),
                       "binding_foil_original": _m([u["correct_o"] for u in units if u["kind"] == "foil"])}
    return res, units


def failure_cases(units: List[dict], top_k: int = 30) -> List[dict]:
    ch = [u for u in units if u["changed"]]
    persist = sorted([u for u in ch if u["gold_o"] and not u["gold_c"] and u["pred_c"] >= 0.5 and u["pred_o"] >= 0.5],
                     key=lambda u: -u["d_lo"])
    nondet = sorted([u for u in ch if (not u["gold_o"]) and u["gold_c"] and u["pred_c"] < 0.5], key=lambda u: u["d_lo"])
    spurious = sorted([u for u in units if _unstable(u) and u["flip"] >= 0.5], key=lambda u: -u["abs_shift"])
    out = []
    for lab, lst in (("persistence_after_edit (answers yes although evidence removed/changed)", persist),
                     ("non_detection (misses newly present evidence)", nondet),
                     ("spurious_flip (answer changes though nothing relevant changed)", spurious)):
        out += [{**u, "category_candidate": lab} for u in lst[:top_k]]
    return out


# ------------------------------------------------------------------ task runner + reporting
def side_by_side(a, b):
    from PIL import Image
    w = Image.new("RGB", (a.width * 2 + 10, a.height), "black")
    w.paste(a, (0, 0)); w.paste(b, (a.width + 10, 0))
    return w


def run_counterfactual_synthetic(cfg: dict, model, out_root, allow_testing_only: bool = False, progress: bool = True,
                                 test_log="results/test_access_log.jsonl"):
    import json
    import platform
    import time
    from pathlib import Path

    from ..datasets.splits import log_test_access
    from .analysis import calibrate_thresholds
    from .engine import _git

    testing_only = bool(model.metadata().get("testing_only"))
    if testing_only and not allow_testing_only:
        raise SystemExit("refusing to write results for a testing-only model")
    d, pr = cfg["dataset"], cfg.get("probe", {})
    split = d.get("our_split", "dev")
    if split == "test" and not pr.get("thresholds_file"):
        raise SystemExit("test runs must use frozen thresholds: set probe.thresholds_file to the thresholds.json of a dev run")
    pairs = make_pairs(split, d.get("n_scenes", 40), d.get("seed", 0), d.get("split_seed", 0), d.get("style", "small"))
    if split == "test" and test_log and not testing_only:
        log_test_access(test_log, config=cfg["name"], model=model.metadata(), n_pairs=len(pairs), git=_git())
    from ..models.cache import CachedVLM
    m = CachedVLM(model, cfg.get("cache", f"cache/{model.name}.sqlite"))
    t0 = time.perf_counter()
    rows = run_counterfactual(m, pairs, tuple(pr.get("templates", (0, 1, 2))), pr.get("blind_baseline", True), progress)
    if pr.get("thresholds_file"):
        th = {int(k): float(v) for k, v in json.loads(Path(pr["thresholds_file"]).read_text())["thresholds"].items()}
        mode = f"frozen from {pr['thresholds_file']}"
    elif split == "dev":
        th, mode = calibrate_thresholds(threshold_rows(rows)), "fit on this dev split (optimistic; development only)"
    else:
        th, mode = None, "raw"
    res, units = aggregate_counterfactual(rows, th)
    res["threshold_mode"] = mode
    raw_res, _ = aggregate_counterfactual(rows, None)
    fails = failure_cases(units)
    out = Path(out_root) / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    (out / "plots").mkdir(parents=True)
    (out / "failure_images").mkdir()
    by_id = {p.pair_id: p for p in pairs}
    def jl(name, xs):
        with open(out / name, "w", encoding="utf-8") as f:
            for x in xs:
                f.write(json.dumps(x, default=float) + "\n")
    jl("per_example.jsonl", rows); jl("per_unit.jsonl", units); jl("failure_cases.jsonl", fails)
    jl("pairs.jsonl", [{"pair_id": p.pair_id, "scene_id": p.scene_id, "edit_type": p.edit_type, "role": p.role,
                        "area_frac": p.area_frac, "area_ratio": p.area_ratio, "matched_to": p.matched_to,
                        "orig": [vars(o) for o in p.orig.objects], "orig_bg": p.orig.background,
                        "cf": [vars(o) for o in p.cf.objects], "cf_bg": p.cf.background,
                        "probes": [{"question": vars(q.question) | {"type": type(q.question).__name__}, "kind": q.kind,
                                    "gold_orig": q.gold_orig, "gold_cf": q.gold_cf} for q in p.probes]} for p in pairs])
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "results_raw_threshold.json").write_text(json.dumps(raw_res, indent=2, default=float))
    if th is not None and not pr.get("thresholds_file"):
        (out / "thresholds.json").write_text(json.dumps({"fit_on": f"{cfg['name']} ({split})", "git": _git(), "thresholds": th}, indent=2))
    for i, f in enumerate(fails[:24]):
        p = by_id[f["pair_id"]]
        side_by_side(render(p.orig), render(p.cf)).save(out / "failure_images" / f"{i:02d}_{p.pair_id}.png")
    _cf_csv_and_plots(out, res, units)
    (out / "manifest.json").write_text(json.dumps({
        "config": cfg, "model_metadata": model.metadata(), "git_commit": _git(), "python": platform.python_version(),
        "runtime": model.runtime_info() if hasattr(model, "runtime_info") else None, "n_pairs": len(pairs),
        "n_scenes": len({p.scene_id for p in pairs}), "n_rows": len(rows), "seconds": time.perf_counter() - t0,
        "cache_hits": m.hits, "cache_misses": m.misses, "threshold_mode": mode,
        "dataset_note": "synthetic scenes with exact ground truth (not natural images): a distribution shift in itself"},
        indent=2, default=str))
    return out


def _cf_csv_and_plots(out, res, units):
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ets = list(res["by_edit_type"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edit_type", "n_changed", "n_control", "AFR", "AFR_lo", "AFR_hi", "SFR_control", "SFR_lo", "SFR_hi",
                    "CSS", "CSS_lo", "CSS_hi", "CPA", "shift_auroc", "persistence_rate", "non_detection_rate"])
        for et in ets:
            d = res["by_edit_type"][et]
            w.writerow([et, d["n_changed_probes"], d["n_control_probes"]] + [d[k][x] for k in ("AFR", "SFR_control", "CSS") for x in ("est", "lo", "hi")]
                       + [d["CPA"]["est"], d["shift_auroc"]["est"], d.get("persistence_rate"), d.get("non_detection_rate")])
    if not ets:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(ets))
    for off, key, lab in ((-0.2, "AFR", "AFR: flips when evidence changes"), (0.2, "SFR_control", "SFR: flips on matched control edit")):
        est = [res["by_edit_type"][e][key]["est"] for e in ets]
        lo = [max(0, res["by_edit_type"][e][key]["est"] - res["by_edit_type"][e][key]["lo"]) if res["by_edit_type"][e][key]["lo"] == res["by_edit_type"][e][key]["lo"] else 0 for e in ets]
        hi = [max(0, res["by_edit_type"][e][key]["hi"] - res["by_edit_type"][e][key]["est"]) if res["by_edit_type"][e][key]["hi"] == res["by_edit_type"][e][key]["hi"] else 0 for e in ets]
        ax[0].bar(x + off, np.nan_to_num(est), 0.4, yerr=[lo, hi], capsize=3, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(ets); ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=7); ax[0].set_title("answer flips (95% CI over scenes)")
    ch = [u["aligned_shift"] for u in units if u["changed"] and u["aligned_shift"] is not None]
    un = [u["abs_shift"] for u in units if (not u["changed"]) and u["role"] in ("control", "background")]
    ax[1].hist(ch, bins=30, alpha=0.6, label="changed probes: shift toward correct answer")
    ax[1].hist(un, bins=30, alpha=0.6, label="control probes: |shift|")
    ax[1].set_xlabel("change in log-odds(yes) - log-odds(no), edited vs original"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "plots" / "sensitivity.png", dpi=150); plt.close(fig)
