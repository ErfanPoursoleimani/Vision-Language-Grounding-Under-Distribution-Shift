import json
import random

import numpy as np
import pytest
from PIL import Image

from src.evaluation.engine import TASKS
from src.evaluation.robustness import aggregate_robustness
from src.models import build_model
from src.perturbations.image_ops import OPS, SEVERITIES, apply
from src.perturbations.synthetic import random_scene, render
from tests.test_probes_pipeline import coco  # noqa: F401  (fixture: tiny COCO-format dataset)


@pytest.fixture(scope="module")
def img():
    return render(random_scene(random.Random(5)))  # edges + flat regions, like a real structured image


def _mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


@pytest.mark.parametrize("op", OPS)
def test_ops_deterministic_preserve_shape_and_grow_with_severity(img, op):
    outs = [apply(img, op, s, seed=3) for s in SEVERITIES]
    assert all(o.size == img.size and o.mode == "RGB" for o in outs)
    assert all(apply(img, op, s, seed=3).tobytes() == o.tobytes() for s, o in zip(SEVERITIES, outs))
    d = [_mse(img, o) for o in outs]
    assert all(x <= y + 1e-9 for x, y in zip(d, d[1:])), d
    assert d[-1] > d[0] > 0 or op in ("blur", "pixelate")  # on flat scenes the weakest blur/pixelate may change almost nothing


def test_op_specific_properties(img):
    lap = lambda im: float(np.var(np.diff(np.asarray(im.convert("L"), float), axis=0)))  # noqa: E731
    assert lap(apply(img, "blur", 5)) < lap(img)                                         # blur removes high frequencies
    assert np.asarray(apply(img, "contrast", 5)).std() < np.asarray(img).std()           # contrast reduction shrinks spread
    a = np.asarray(apply(img, "pixelate", 5))                     # severity 5: factor 12 on a 384 px canvas -> constant 12x12 blocks
    assert img.size == (384, 384) and all((a[i:i + 12, j:j + 12] == a[i, j]).all() for i in (0, 120, 372) for j in (0, 48, 372))
    n1, n2 = apply(img, "noise", 3, seed=1), apply(img, "noise", 3, seed=2)
    assert n1.tobytes() != n2.tobytes()                                                  # seed changes the noise realisation
    resid = np.asarray(apply(img, "noise", 3, seed=1), float) - np.asarray(img, float)
    assert 20 < resid.std() < 36                                                         # severity 3 = sigma 28 (clipping reduces it a bit)
    with pytest.raises(ValueError):
        apply(img, "crop", 1)
    with pytest.raises(ValueError):
        apply(img, "blur", 9)


def _rows(pert_fn, n_img=12):
    rows = []
    for i in range(n_img):
        for label in (1, 0, 0):
            clean = 2.0 if label else -2.0
            rows.append({"image_id": i, "category": f"c{label}{len(rows)}", "label": label, "kind": "pos" if label else "random",
                         "template_idx": 0, "op": "clean", "severity": 0, "question": "q", "logodds": clean})
            rows.append({**rows[-1], "op": "blur", "severity": 3, "logodds": pert_fn(clean)})
    return rows


def test_aggregate_known_cases():
    same = aggregate_robustness(_rows(lambda x: x), {0: 0.0}, n_boot=30)
    c = same["conditions"]["blur"]["3"]
    assert same["clean"]["AUROC"] == 1.0 and c["AUROC"] == 1.0 and c["AUROC_drop_vs_clean"]["est"] == 0.0
    assert c["consistency_with_clean"] == 1.0 and c["mean_abs_logodds_change"] == 0.0
    dead = aggregate_robustness(_rows(lambda x: 0.5), {0: 0.0}, n_boot=30)   # degraded model answers the same for everything
    d = dead["conditions"]["blur"]["3"]
    assert d["AUROC"] == 0.5 and d["AUROC_drop_vs_clean"]["est"] == 0.5 and d["yes_ratio"] == 1.0
    assert d["consistency_with_clean"] == pytest.approx(1 / 3)               # rows are 1 positive : 2 negatives; only the positive keeps its answer


def _cfg(coco, tmp_path, **kw):
    c = {"name": "pert", "task": "pope_perturbed", "cache": str(tmp_path / "c.sqlite"), "probe": {"templates": ["Is there {a_obj} in the image?"]},
         "dataset": {"root": str(coco), "split": "val2017", "our_split": "dev", "split_seed": 0, "n_images": 4, "seed": 0},
         "robustness": {"ops": ["blur", "noise"], "severities": [1, 5]}}
    c["probe"].update(kw)
    return c


def test_end_to_end_and_test_split_discipline(coco, tmp_path):
    m = build_model({"type": "dummy_brightness"})
    out = TASKS["pope_perturbed"](_cfg(coco, tmp_path), m, tmp_path / "res", allow_testing_only=True, progress=False, test_log=None)
    for f in ("per_example.jsonl", "results.json", "results.csv", "thresholds.json", "manifest.json", "plots/robustness.png"):
        assert (out / f).exists(), f
    r = json.loads((out / "results.json").read_text())
    assert set(r["conditions"]) == {"blur", "noise"} and set(r["conditions"]["blur"]) == {"1", "5"}
    assert "CLEAN" in r["threshold_mode"]
    cfg = _cfg(coco, tmp_path)
    cfg["dataset"]["our_split"] = "test"
    with pytest.raises(SystemExit, match="frozen thresholds"):
        TASKS["pope_perturbed"](cfg, m, tmp_path / "r2", allow_testing_only=True, progress=False, test_log=None)
    with pytest.raises(SystemExit):
        TASKS["pope_perturbed"](_cfg(coco, tmp_path), m, tmp_path / "r3", progress=False, test_log=None)  # testing-only refused
