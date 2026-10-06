import json
import math
import random

import pytest

from src.evaluation.counterfactual import (aggregate_counterfactual, cluster_bootstrap, make_pairs, run_counterfactual,
                                           scenes_for_split, stat_afr)
from src.evaluation.engine import TASKS
from src.models import build_model
from src.perturbations.synthetic import (COLORS, MIN_DIST, Exists, LeftOf, answer, build_pairs, detect_colors,
                                         question_text, random_scene, render)


def test_render_deterministic_and_verifier_matches_spec():
    for s in range(40):
        sc = random_scene(random.Random(s))
        assert render(sc).tobytes() == render(sc).tobytes()
        assert set(detect_colors(render(sc))) == {o.color for o in sc.objects}
        assert all(math.hypot(a.cx - b.cx, a.cy - b.cy) >= MIN_DIST for i, a in enumerate(sc.objects) for b in sc.objects[i + 1:])


def test_every_edit_is_what_it_claims_by_independent_pixel_check():
    n_checked = 0
    for sid in range(40):
        sc = random_scene(random.Random(sid))
        for p in build_pairs(sid, sc, random.Random(f"p{sid}")):
            dets = [detect_colors(render(p.orig)), detect_colors(render(p.cf))]
            assert set(dets[0]) == {o.color for o in p.orig.objects} and set(dets[1]) == {o.color for o in p.cf.objects}
            for pr in p.probes:
                q = pr.question
                for det, scene, gold in ((dets[0], p.orig, pr.gold_orig), (dets[1], p.cf, pr.gold_cf)):
                    if isinstance(q, LeftOf):  # pixel centroids must agree with the computed gold
                        assert (det[q.c1][1] < det[q.c2][1]) == gold
                        n_checked += 1
                    else:  # colour absent in pixels => gold must be False
                        if q.color not in det:
                            assert gold is False
                        assert gold == answer(scene, q)
    assert n_checked > 10


def test_pair_invariants():
    pairs = make_pairs("dev", 25, 0)
    again = make_pairs("dev", 25, 0)
    assert [p.pair_id for p in pairs] == [p.pair_id for p in again]
    assert [render(p.cf).tobytes() for p in pairs[:5]] == [render(p.cf).tobytes() for p in again[:5]]
    ids = {p.pair_id for p in pairs}
    assert {p.role for p in pairs} == {"relevant", "control", "background"}
    for p in pairs:
        changed = [x for x in p.probes if x.gold_orig != x.gold_cf]
        if p.role == "relevant":
            assert changed
        else:
            assert not changed and p.probes
        if p.role == "control":
            assert p.matched_to in ids
        for x in p.probes:
            if x.kind == "foil":
                assert x.gold_orig is False and x.gold_cf is False
    dev = {sid for sid, _ in scenes_for_split("dev", 30, 0)}
    test = {sid for sid, _ in scenes_for_split("test", 30, 0)}
    assert dev.isdisjoint(test)


def test_question_text():
    assert question_text(Exists("orange", "square"), 0) == "Is there an orange square in the image?"
    assert question_text(Exists("red", "circle"), 1) == "Does this image contain a red circle?"
    assert question_text(LeftOf("red", "circle", "blue", "square"), 0) == "Is the red circle to the left of the blue square?"


def _rows(flip_changed, flip_unchanged, n_scenes=10):
    rows = []
    for sid in range(n_scenes):
        for role, go, gc, flip in (("relevant", 1, 0, flip_changed), ("control", 1, 1, flip_unchanged)):
            for t in range(3):
                lo_o = 2.0 if go else -2.0
                lo_c = (-lo_o if flip else lo_o)
                rows.append({"pair_id": f"{sid}-remove-{role}", "scene_id": sid, "edit_type": "remove", "role": role,
                             "probe_idx": 0, "probe_kind": "exists", "template_idx": t, "question": "q",
                             "gold_orig": go, "gold_cf": gc, "lo_orig": lo_o, "lo_cf": lo_c, "blind_lo": 0.0,
                             "area_frac": 0.01, "area_ratio": 1.0 if role == "control" else None})
    return rows


@pytest.mark.parametrize("fc,fu,afr,sfr,cpa,persist,shift_auroc", [
    (True, False, 1.0, 0.0, 1.0, 0.0, 1.0),   # ideal: flips only when evidence changes
    (False, False, 0.0, 0.0, 0.0, 1.0, 0.5),  # image-insensitive: no shift anywhere (all ties -> AUROC 0.5), keeps saying yes after removal
    (True, True, 1.0, 1.0, 1.0, 0.0, 0.5),    # unstable: flips on everything, CSS exposes it
])
def test_aggregate_known_models(fc, fu, afr, sfr, cpa, persist, shift_auroc):
    res, _ = aggregate_counterfactual(_rows(fc, fu), None, n_boot=30)
    o, d = res["overall"], res["by_edit_type"]["remove"]
    assert o["AFR"]["est"] == afr and o["SFR"]["est"] == sfr and o["CSS"]["est"] == afr - sfr
    assert d["CPA"]["est"] == cpa and d["persistence_rate"]["est"] == persist
    assert d["shift_auroc"]["est"] == pytest.approx(shift_auroc)
    assert res["blind"]["CPA_changed"] == 0.0


def test_cluster_bootstrap_contains_estimate():
    res, units = aggregate_counterfactual(_rows(True, False), None, n_boot=20)
    est, lo, hi = cluster_bootstrap(units, stat_afr, n_boot=50)
    assert lo <= est <= hi


def _cfg(tmp_path, **kw):
    c = {"name": "cf", "task": "counterfactual_synthetic", "cache": str(tmp_path / "c.sqlite"),
         "dataset": {"our_split": "dev", "n_scenes": 8, "seed": 0}, "probe": {}}
    c["probe"].update(kw)
    return c


def test_end_to_end_color_oracle(tmp_path):
    out = TASKS["counterfactual_synthetic"](_cfg(tmp_path), build_model({"type": "dummy_color_oracle"}), tmp_path / "res",
                                            allow_testing_only=True, progress=False, test_log=None)
    for f in ("manifest.json", "per_example.jsonl", "per_unit.jsonl", "pairs.jsonl", "results.json", "results.csv",
              "failure_cases.jsonl", "thresholds.json", "plots/sensitivity.png"):
        assert (out / f).exists(), f
    r = json.loads((out / "results.json").read_text())
    bt = r["by_edit_type"]
    # colour-only oracle: exact on colour-visible edits, blind to shape changes and fooled by binding foils
    for et in ("remove", "insert", "recolor", "swap"):
        assert bt[et]["AFR"]["est"] == 1.0 and bt[et]["CPA"]["est"] == 1.0
    assert bt["reshape"]["AFR"]["est"] == 0.0
    assert r["accuracy"]["binding_foil_original"] == 0.0
    assert r["sfr_by_source"]["control_edits"]["SFR"] == 0.0
    assert any((out / "failure_images").iterdir())


def test_test_split_discipline_and_testing_only(tmp_path):
    m = build_model({"type": "dummy_color_oracle"})
    with pytest.raises(SystemExit):
        TASKS["counterfactual_synthetic"](_cfg(tmp_path), m, tmp_path / "r0", progress=False, test_log=None)
    out = TASKS["counterfactual_synthetic"](_cfg(tmp_path), m, tmp_path / "r1", allow_testing_only=True, progress=False, test_log=None)
    cfg = _cfg(tmp_path)
    cfg["dataset"]["our_split"] = "test"
    with pytest.raises(SystemExit, match="frozen thresholds"):
        TASKS["counterfactual_synthetic"](cfg, m, tmp_path / "r2", allow_testing_only=True, progress=False, test_log=None)
    cfg = _cfg(tmp_path, thresholds_file=str(out / "thresholds.json"))
    cfg["dataset"]["our_split"] = "test"
    out2 = TASKS["counterfactual_synthetic"](cfg, m, tmp_path / "r3", allow_testing_only=True, progress=False, test_log=None)
    assert "frozen" in json.loads((out2 / "manifest.json").read_text())["threshold_mode"]
