import json
import random

import pytest

from src.evaluation.counterfactual import aggregate_counterfactual
from src.evaluation.engine import TASKS
from src.models import build_model
from src.perturbations.synthetic import (HasColor, HasShape, answer, competence_items, question_text, random_scene)


def test_atomic_questions():
    assert question_text(HasColor("orange"), 0) == "Is there an orange object in the image?"
    assert question_text(HasColor("red"), 1) == "Does this image contain something red?"
    assert question_text(HasShape("square"), 2) == "Can you see a square in this picture?"
    sc = random_scene(random.Random(3))
    assert answer(sc, HasColor(sc.objects[0].color)) and answer(sc, HasShape(sc.objects[0].shape))


def test_competence_items_gold_is_computed_and_balanced():
    for sid in range(20):
        sc = random_scene(random.Random(sid))
        items = competence_items(sc, random.Random(sid))
        kinds = {k for k, _ in items}
        assert {"color_atom", "shape_atom", "exists_true", "foil", "easy_neg"} <= kinds
        for k, q in items:
            g = answer(sc, q)
            if k in ("exists_true",):
                assert g
            if k in ("foil", "easy_neg"):
                assert not g
        rel = [q for k, q in items if k == "relation"]
        assert len(rel) % 2 == 0
        for a, b in zip(rel[::2], rel[1::2]):  # reversed pair has the complementary gold
            assert answer(sc, a) != answer(sc, b)


def test_competence_end_to_end_color_oracle(tmp_path):
    cfg = {"name": "c", "task": "synthetic_competence", "cache": str(tmp_path / "c.sqlite"),
           "dataset": {"our_split": "dev", "n_scenes": 12, "seed": 0}, "probe": {}}
    out = TASKS["synthetic_competence"](cfg, build_model({"type": "dummy_color_oracle"}), tmp_path / "r",
                                        allow_testing_only=True, progress=False, test_log=None)
    r = json.loads((out / "results.json").read_text())
    g = r["groups"]
    assert g["color_atoms"]["AUROC"]["est"] == 1.0 and g["easy_existence"]["AUROC"]["est"] == 1.0
    assert g["relation"]["AUROC"]["est"] == 1.0 and r["relation_both_directions_yes_rate"] == 0.0
    assert g["shape_atoms"]["AUROC"]["est"] == 0.5 and g["binding"]["AUROC"]["est"] == 0.5  # shape-blind oracle
    assert all(x["blind_AUROC"] == 0.5 for x in g.values())
    assert (out / "plots" / "competence.png").exists() and (out / "thresholds.json").exists()


def test_test_split_requires_frozen_thresholds(tmp_path):
    cfg = {"name": "c", "task": "synthetic_competence", "dataset": {"our_split": "test", "n_scenes": 2}, "probe": {}}
    with pytest.raises(SystemExit, match="frozen thresholds"):
        TASKS["synthetic_competence"](cfg, build_model({"type": "dummy_color_oracle"}), tmp_path, allow_testing_only=True, progress=False, test_log=None)


def _rows(pred_o_yes, pred_c_yes, go, gc, n_scenes=6):
    rows = []
    for sid in range(n_scenes):
        for t in range(3):
            rows.append({"pair_id": f"{sid}-reshape-relevant", "scene_id": sid, "edit_type": "reshape", "role": "relevant",
                         "probe_idx": 0, "probe_kind": "exists", "template_idx": t, "question": "q", "gold_orig": go, "gold_cf": gc,
                         "lo_orig": 1.0 if pred_o_yes else -1.0, "lo_cf": 1.0 if pred_c_yes else -1.0, "blind_lo": 0.0,
                         "area_frac": 0.01, "area_ratio": None})
    return rows


def test_persistence_is_conditional_on_having_said_yes_before():
    # evidence removed (yes->no). Model said NO already on the original: no persistence to measure (nan), but a 'false yes' rate.
    r, _ = aggregate_counterfactual(_rows(False, True, 1, 0), None, n_boot=10)
    d = r["by_edit_type"]["reshape"]
    assert d["persistence_rate"]["est"] != d["persistence_rate"]["est"]  # nan: nobody said yes first
    assert d["false_yes_after_edit"]["est"] == 1.0
    # said yes before and still yes after -> persistence 1.0 ; said yes before, 'no' after -> 0.0
    assert aggregate_counterfactual(_rows(True, True, 1, 0), None, n_boot=10)[0]["by_edit_type"]["reshape"]["persistence_rate"]["est"] == 1.0
    assert aggregate_counterfactual(_rows(True, False, 1, 0), None, n_boot=10)[0]["by_edit_type"]["reshape"]["persistence_rate"]["est"] == 0.0
    # evidence added (no->yes): model said no before and still no after -> non-detection 1.0
    nd = aggregate_counterfactual(_rows(False, False, 0, 1), None, n_boot=10)[0]["by_edit_type"]["reshape"]
    assert nd["non_detection_rate"]["est"] == 1.0 and nd["miss_after_edit"]["est"] == 1.0


def test_default_style_is_bit_identical_to_original_behaviour():
    import hashlib

    from src.perturbations.synthetic import build_pairs, render
    h = [hashlib.md5(render(random_scene(random.Random(f"0:{s}"))).tobytes()).hexdigest()[:12] for s in range(6)]
    assert h == ["651016329eaa", "4269efdbcca0", "9c9bd4a7893d", "847805164f6c", "9ff35c091d2d", "44dd5e84dde9"]
    sc = random_scene(random.Random("0:3"))
    ps = build_pairs(3, sc, random.Random("0:3:pairs"))
    assert hashlib.md5("|".join(p.pair_id + str(len(p.probes)) for p in ps).encode()).hexdigest()[:12] == "74b3a6a1911e"


@pytest.mark.parametrize("style", ["medium", "large"])
def test_larger_styles_respect_constraints_and_stay_exact(style):
    import math

    from src.perturbations.synthetic import STYLES, build_pairs, detect_colors, render
    st = STYLES[style]
    for sid in range(12):
        sc = random_scene(random.Random(sid), style=st)
        assert len(sc.objects) in st.n_objects
        assert all(st.size[0] <= o.size <= st.size[1] for o in sc.objects)
        assert all(math.hypot(a.cx - b.cx, a.cy - b.cy) >= st.min_dist for i, a in enumerate(sc.objects) for b in sc.objects[i + 1:])
        for p in build_pairs(sid, sc, random.Random(f"p{sid}"), st):
            assert set(detect_colors(render(p.cf))) == {o.color for o in p.cf.objects}
            changed = [x for x in p.probes if x.gold_orig != x.gold_cf]
            assert bool(changed) == (p.role == "relevant")


def test_reaggregate_script_writes_results_v2(tmp_path):
    import subprocess
    import sys
    cfg = {"name": "rr", "task": "counterfactual_synthetic", "cache": str(tmp_path / "c.sqlite"),
           "dataset": {"our_split": "dev", "n_scenes": 4, "seed": 0}, "probe": {}}
    out = TASKS["counterfactual_synthetic"](cfg, build_model({"type": "dummy_color_oracle"}), tmp_path / "res",
                                            allow_testing_only=True, progress=False, test_log=None)
    r = subprocess.run([sys.executable, "scripts/reaggregate_counterfactual.py", "--run-dir", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    v2 = json.loads((out / "results_v2.json").read_text())
    assert "persistence_rate" in v2["by_edit_type"]["reshape"] and "re-aggregated" in v2["threshold_mode"]
