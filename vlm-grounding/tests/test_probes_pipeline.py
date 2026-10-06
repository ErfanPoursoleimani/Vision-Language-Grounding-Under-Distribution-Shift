import json
import random
from collections import Counter

import pytest
from PIL import Image

from src.datasets.coco import load_coco_instances
from src.datasets.pope_style import build_pope_style, with_article
from src.datasets.splits import assign_split, log_test_access, split_ids
from src.evaluation.aggregate import aggregate_probes, failure_cases, item_level
from src.evaluation.engine import run_pope_style
from src.evaluation.probes import make_question
from src.metrics.classification import auroc, binary_metrics
from src.models import build_model
from src.models.cache import CachedVLM

CATS = ["person", "car", "dog", "umbrella", "chair", "cup", "bus", "orange"]


@pytest.fixture()
def coco(tmp_path):
    rng = random.Random(1)
    root = tmp_path / "coco"
    (root / "val2017").mkdir(parents=True)
    (root / "annotations").mkdir()
    images, anns, aid = [], [], 1
    for i in range(1, 61):
        fn = f"{i:06d}.jpg"
        v = rng.randint(20, 235)
        Image.new("RGB", (64, 48), (v, v, v)).save(root / "val2017" / fn)
        images.append({"id": i, "file_name": fn, "width": 64, "height": 48})
        for c in rng.sample(range(len(CATS)), rng.randint(2, 6)):
            anns.append({"id": aid, "image_id": i, "category_id": c + 1, "area": 300.0 + c})
            aid += 1
    (root / "annotations" / "instances_val2017.json").write_text(json.dumps({
        "images": images, "annotations": anns, "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CATS)]}))
    return root


def test_loader(coco):
    imgs, cats = load_coco_instances(coco, "val2017")
    assert len(imgs) == 60 and cats == sorted(CATS)
    assert all(0 < sum(im.area_frac.values()) for im in imgs) and imgs[0].path.exists()
    with pytest.raises(FileNotFoundError):
        load_coco_instances(coco, "nope")


def test_splits_deterministic_disjoint():
    ids = list(range(2000))
    a, b = split_ids(ids, 0), split_ids(ids, 0)
    assert a == b and set(a["dev"]).isdisjoint(a["test"]) and len(a["dev"]) + len(a["test"]) == 2000
    assert abs(len(a["dev"]) / 2000 - 0.2) < 0.04
    assert split_ids(ids, 1) != a
    assert assign_split(5, 0) == assign_split(5, 0)
    with pytest.raises(ValueError):
        assign_split(1, 0, {"a": 0.5, "b": 0.4})


def test_pope_style_invariants(coco):
    imgs, cats = load_coco_instances(coco, "val2017")
    items = build_pope_style(imgs, cats, n_images=20, seed=3)
    assert items == build_pope_style(imgs, cats, n_images=20, seed=3)
    by = {im.image_id: im for im in imgs}
    per_img = Counter(it.image_id for it in items)
    assert len(per_img) == 20 and set(per_img.values()) == {12}  # requested n_images honoured
    for it in items:
        present = it.category in by[it.image_id].categories
        assert present == bool(it.label) and (it.kind == "pos") == bool(it.label)
    freq = Counter(c for im in imgs if len(im.categories) >= 3 and len(cats) - len(im.categories) >= 3 for c in im.categories)
    it0 = next(i for i in items if i.kind == "popular")
    absent = sorted(c for c in cats if c not in by[it0.image_id].categories)
    best = sorted(absent, key=lambda c: (-freq[c], c))[:3]
    assert {i.category for i in items if i.kind == "popular" and i.image_id == it0.image_id} == set(best)


def test_question_articles():
    assert with_article("umbrella") == "an umbrella" and with_article("dog") == "a dog"
    assert make_question("Is there {a_obj} in the image?", "orange") == "Is there an orange in the image?"


def test_binary_metrics_and_auroc():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["yes_ratio"] == 0.5 and m["tp"] == 1
    assert auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0 and auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0
    assert auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5


def _rows(perfect=True):
    rows = []
    for img in range(10):
        for kind, label in (("pos", 1), ("random", 0), ("popular", 0), ("adversarial", 0)):
            for t in range(2):
                lo = (2.0 if label else -2.0) if perfect else 1.0
                rows.append({"image_id": img, "category": f"c{kind}", "label": label, "kind": kind, "template_idx": t,
                             "question": "q", "logodds": lo, "blind_logodds": 1.0})
    return rows


def test_aggregate_known_values():
    s = aggregate_probes(_rows(True), n_boot=50)
    for nk in ("random", "popular", "adversarial"):
        r = s["subsets"][nk]
        assert r["model"]["summary"]["accuracy"]["mean"] == 1.0 and r["model"]["summary"]["auroc"]["mean"] == 1.0
        assert r["blind"]["summary"]["yes_ratio"]["mean"] == 1.0 and r["blind"]["summary"]["accuracy"]["mean"] == 0.5
        assert r["model_minus_blind_accuracy"] == pytest.approx(0.5)
        assert r["model"]["summary"]["accuracy_ci95"] == [1.0, 1.0]


def test_failure_cases_candidates():
    its = item_level(_rows(False))  # model always says yes -> every negative is a false positive
    f = failure_cases(its, top_k=5)
    fps = [x for x in f if x["category_candidate"] == "object_hallucination"]
    assert fps and all(x["label"] == 0 and x["language_prior_candidate"] is True for x in fps)


def test_cache_yes_no(tmp_path):
    m = CachedVLM(build_model({"type": "dummy_brightness"}), tmp_path / "c.sqlite")
    img = Image.new("RGB", (4, 4), (200, 200, 200))
    a, b = m.yes_no_logodds(img, "q"), m.yes_no_logodds(img, "q")
    assert a == b and (m.hits, m.misses) == (1, 1)


def test_end_to_end_with_dummy(coco, tmp_path):
    cfg = {"name": "e2e", "task": "pope_style", "cache": str(tmp_path / "c.sqlite"),
           "dataset": {"root": str(coco), "split": "val2017", "our_split": "dev", "split_seed": 0, "n_images": 5, "seed": 0}}
    # tiny fixture: with 60 images, dev has ~12; require enough eligible images
    out = run_pope_style(cfg, build_model({"type": "dummy_brightness"}), tmp_path / "res", allow_testing_only=True,
                         progress=False, test_log=None)
    for f in ("manifest.json", "per_example.jsonl", "per_item.jsonl", "results.json", "results.csv",
              "failure_cases.jsonl", "plots/accuracy_yes_ratio.png", "plots/logodds_hist.png"):
        assert (out / f).exists(), f
    res = json.loads((out / "results.json").read_text())
    assert set(res["subsets"]) == {"random", "popular", "adversarial"}
    man = json.loads((out / "manifest.json").read_text())
    assert man["n_items"] == 12 * man["n_images"] and "not the official POPE" in man["dataset_note"]


def test_testing_only_refused_by_default(coco, tmp_path):
    with pytest.raises(SystemExit):
        run_pope_style({"name": "x", "task": "pope_style", "dataset": {"root": str(coco)}},
                       build_model({"type": "dummy_brightness"}), tmp_path)


def test_test_access_log(tmp_path):
    log_test_access(tmp_path / "log.jsonl", config="c", n=1)
    log_test_access(tmp_path / "log.jsonl", config="c", n=2)
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 2


def test_calibration_and_analysis():
    from src.evaluation import analysis as A
    # template 0 unbiased, template 1 shifted by +1.5 (all 'yes' at threshold 0): calibrated thresholds must differ by ~1.5
    rows = []
    for i in range(50):
        for t, shift in ((0, 0.0), (1, 1.5)):
            rows.append({"image_id": i, "category": "c", "kind": "pos", "label": 1, "template_idx": t, "logodds": 1.0 + shift + 0.01 * (i % 5)})
            rows.append({"image_id": i, "category": "d", "kind": "random", "label": 0, "template_idx": t, "logodds": -1.0 + shift - 0.01 * (i % 5)})
    th = A.calibrate_thresholds(rows)
    assert abs((th[1] - th[0]) - 1.5) < 0.05 and abs(th[0]) < 0.05
    assert all(v == 1.0 for v in A.pooled_auroc_by_template(rows).values())
    ba = A.balanced_accuracy(*[__import__("numpy").array([r["logodds"] for r in rows if r["template_idx"] == 1 and r["label"] == l]) for l in (1, 0)], 0.0)
    assert ba == 0.5  # raw threshold 0 fails on the shifted template (everything 'yes') although AUROC is perfect
    items = [{"image_id": 1, "category": "x", "kind": "popular", "label": 0, "pred": 1, "blind_pred": 0},
             {"image_id": 1, "category": "x", "kind": "adversarial", "label": 0, "pred": 1, "blind_pred": 0},
             {"image_id": 1, "category": "y", "kind": "random", "label": 0, "pred": 0, "blind_pred": 1}]
    assert A.negative_overlap(items) == {"n_negative_items": 3, "n_unique_negative_pairs": 2, "n_pairs_in_multiple_kinds": 1}
    assert A.language_prior_share(items)["n_fp_where_blind_also_yes"] == 0


def _cfg(coco, tmp_path, **probe):
    return {"name": "e2e2", "task": "pope_style", "cache": str(tmp_path / "c2.sqlite"), "probe": probe,
            "dataset": {"root": str(coco), "split": "val2017", "our_split": "dev", "split_seed": 0, "n_images": 5, "seed": 0}}


def test_dev_run_writes_thresholds_and_test_requires_them(coco, tmp_path):
    m = build_model({"type": "dummy_brightness"})
    out = run_pope_style(_cfg(coco, tmp_path), m, tmp_path / "res", allow_testing_only=True, progress=False, test_log=None)
    thr = json.loads((out / "thresholds.json").read_text())
    assert set(thr["thresholds"]) == {"0", "1", "2"} and (out / "results_raw_threshold.json").exists()
    assert "optimistic" in json.loads((out / "manifest.json").read_text())["threshold_mode"]
    cfg = _cfg(coco, tmp_path)
    cfg["dataset"]["our_split"] = "test"
    with pytest.raises(SystemExit, match="frozen thresholds"):
        run_pope_style(cfg, m, tmp_path / "res2", allow_testing_only=True, progress=False, test_log=None)
    cfg = _cfg(coco, tmp_path, thresholds_file=str(out / "thresholds.json"))
    cfg["dataset"]["our_split"] = "test"
    out2 = run_pope_style(cfg, m, tmp_path / "res3", allow_testing_only=True, progress=False, test_log=None)
    assert "frozen" in json.loads((out2 / "manifest.json").read_text())["threshold_mode"]
