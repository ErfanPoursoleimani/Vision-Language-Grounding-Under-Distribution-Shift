import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from src.evaluation.h1 import apply_audit, h1_detection
from tests.test_probes_pipeline import coco  # noqa: F401


def _items(prior_informative: bool, n=60, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        for label in (1, 0):
            cat = f"c{(i + (0 if label else 1)) % 6}"   # an object is present OR absent in an image, never both
            if prior_informative:   # hallucinations: same confidence as grounded answers but a HIGH language prior
                conf = 10 + rng.normal(0, 1)
                prior = (8 if label == 0 else 0) + rng.normal(0, 1)
            else:                   # no blind information: pmi == conf shifted by a constant
                conf = (10 if label else 4) + rng.normal(0, 1)
                prior = 0.0
            items.append({"image_id": i, "category": cat, "kind": "random" if label == 0 else "pos", "label": label,
                          "mean_logodds": conf, "pred": 1, "blind_mean_logodds": prior})
    return items


def test_pmi_beats_confidence_when_the_prior_explains_hallucinations():
    r = h1_detection(_items(True), n_boot=60)["mentions_only (model answered yes)"]
    assert r["AUROC_pmi"]["est"] > 0.95 and 0.35 < r["AUROC_conf"]["est"] < 0.65
    rules = r["falsification_rules_survived"]
    assert rules["F1_pmi_AUROC_lower_bound_above_0.5"] and rules["F3_pmi_beats_confidence_by_0.02"]
    assert r["AUROC_neg_prior"]["est"] > 0.95


def test_pmi_equals_confidence_without_blind_information():
    r = h1_detection(_items(False), n_boot=60)["mentions_only (model answered yes)"]
    assert r["AUROC_pmi"]["est"] == pytest.approx(r["AUROC_conf"]["est"])
    assert not r["falsification_rules_survived"]["F3_pmi_beats_confidence_by_0.02"]  # no gain -> F3 falsified


def test_dedupe_and_small_class_note():
    items = _items(True, n=3)
    dup = items + [dict(items[0], kind="popular"), dict(items[1], kind="adversarial")]
    assert h1_detection(dup, n_boot=10)["n_unique_items"] == len(items)
    assert "note" in h1_detection(items, n_boot=10)["all_items"]  # fewer than 5 per class


def test_apply_audit_relabels_and_drops():
    items = [{"image_id": 1, "category": "cup", "label": 0}, {"image_id": 2, "category": "dog", "label": 0},
             {"image_id": 3, "category": "car", "label": 0}]
    audit = [{"image_id": "1", "category": "cup", "verdict": "present_unannotated"},
             {"image_id": "2", "category": "dog", "verdict": "ambiguous"},
             {"image_id": "3", "category": "car", "verdict": "hallucination"}]
    out = apply_audit(items, audit)
    assert [i["category"] for i in out] == ["cup", "car"] and out[0]["label"] == 1 and out[1]["label"] == 0


def test_verdict_aliases_and_strictness():
    from src.evaluation.h1 import audit_summary, normalize_verdict
    assert [normalize_verdict(x) for x in ("yes", "No", " ? ", "", "present unannotated", "Hallucination")] == [
        "present_unannotated", "hallucination", "ambiguous", "", "present_unannotated", "hallucination"]
    with pytest.raises(ValueError, match="unknown audit verdict"):
        normalize_verdict("probably")
    rows = [{"verdict": "yes"}] * 20 + [{"verdict": "no"}] * 11 + [{"verdict": ""}] * 9
    s = audit_summary(rows)
    assert s["n_judged"] == 31 and s["present_unannotated"] == 20 and s["hallucination"] == 11 and s["n_blank"] == 9
    # unknown verdicts must no longer be ignored silently by apply_audit
    with pytest.raises(ValueError):
        apply_audit([{"image_id": 1, "category": "cup", "label": 0}], [{"image_id": "1", "category": "cup", "verdict": "banana"}])
    out = apply_audit([{"image_id": 1, "category": "cup", "label": 0}], [{"image_id": "1", "category": "cup", "verdict": "yes"}])
    assert out[0]["label"] == 1


def _per_item(imgs, n=24):
    rows = []
    for k, im in enumerate(imgs[:n]):
        rows.append({"image_id": im.image_id, "category": "umbrella", "kind": "popular", "label": 0, "mean_logodds": 20.0 - k,
                     "pred": 1, "blind_mean_logodds": 0.0})
    return rows


@pytest.mark.parametrize("select", ["top", "random", "stratified"])
def test_audit_script_sheet_and_summarize(coco, tmp_path, select):
    from src.datasets.coco import load_coco_instances
    imgs, _ = load_coco_instances(coco, "val2017")
    run = tmp_path / "run"
    run.mkdir()
    (run / "per_item.jsonl").write_text("\n".join(json.dumps(r) for r in _per_item(imgs, 24)))
    out = tmp_path / "audit"
    r = subprocess.run([sys.executable, "scripts/audit_failures.py", "sheet", "--run-dir", str(run), "--coco-root", str(coco),
                        "--n", "9", "--select", select, "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(out / "audit_template.csv", newline="") as f:
        t = list(csv.DictReader(f))
    assert len(t) == 9 and t[0]["verdict"] == "" and (out / "sheet_01.png").exists() and (out / "sheet_02.png").exists()
    confs = sorted(float(x["model_mean_logodds"]) for x in t)
    if select == "top":
        assert confs[0] >= 20.0 - 8                           # only the 9 most confident (20 ... 12)
    if select == "stratified":
        assert confs[0] < 20.0 - 16 and confs[-1] > 20.0 - 8  # spans the confidence range
    for i, row in enumerate(t):
        row["verdict"] = ["yes", "no", "?", ""][i % 4]
    with open(out / "filled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t[0]))
        w.writeheader(); w.writerows(t)
    s = subprocess.run([sys.executable, "scripts/audit_failures.py", "summarize", "--csv", str(out / "filled.csv")], capture_output=True, text=True)
    res = json.loads(s.stdout)
    assert res["n_judged"] == 7 and res["present_unannotated"] == 3 and res["hallucination"] == 2 and res["ambiguous"] == 2 and res["n_blank"] == 2
    t[0]["verdict"] = "nonsense"
    with open(out / "bad.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t[0]))
        w.writeheader(); w.writerows(t)
    assert subprocess.run([sys.executable, "scripts/audit_failures.py", "summarize", "--csv", str(out / "bad.csv")], capture_output=True, text=True).returncode != 0


def test_merge_audits_reports_conflicts_and_later_wins():
    from src.evaluation.h1 import merge_audits
    a = [{"image_id": "1", "category": "cup", "verdict": "yes"}, {"image_id": "2", "category": "dog", "verdict": "no"},
         {"image_id": "3", "category": "car", "verdict": ""}]
    b = [{"image_id": "1", "category": "cup", "verdict": "no"}, {"image_id": "4", "category": "tv", "verdict": "?"}]
    rows, conflicts = merge_audits([a, b])
    d = {(int(r["image_id"]), r["category"]): r["verdict"] for r in rows}
    assert d == {(1, "cup"): "hallucination", (2, "dog"): "hallucination", (4, "tv"): "ambiguous"}   # blank ignored, later wins
    assert conflicts == [{"image_id": 1, "category": "cup", "earlier": "present_unannotated", "later": "hallucination"}]


def test_census_with_exclude_skips_judged_items(coco, tmp_path):
    from src.datasets.coco import load_coco_instances
    imgs, _ = load_coco_instances(coco, "val2017")
    run = tmp_path / "run"
    run.mkdir()
    (run / "per_item.jsonl").write_text("\n".join(json.dumps(r) for r in _per_item(imgs, 12)))
    first = tmp_path / "a1"
    base = [sys.executable, "scripts/audit_failures.py", "sheet", "--run-dir", str(run), "--coco-root", str(coco)]
    assert subprocess.run(base + ["--n", "5", "--select", "top", "--out", str(first)], capture_output=True, text=True).returncode == 0
    with open(first / "audit_template.csv", newline="") as f:
        t = list(csv.DictReader(f))
    for row in t:
        row["verdict"] = "no"
    with open(first / "filled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t[0]))
        w.writeheader(); w.writerows(t)
    second = tmp_path / "a2"
    r = subprocess.run(base + ["--select", "all", "--exclude", str(first / "filled.csv"), "--out", str(second)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(second / "audit_template.csv", newline="") as f:
        t2 = list(csv.DictReader(f))
    assert len(t2) == 12 - 5 and not ({x["image_id"] for x in t} & {x["image_id"] for x in t2})
