import math

import pytest

from src.metrics.concepts import concept_mentioned, extract_mentions
from src.metrics.hallucination import chair
from src.metrics.sensitivity import (counterfactual_pair_accuracy, flip_rate, local_sensitivity,
                                     pmi, residual_mention_rate, sensitivity_scores)
from src.metrics.stats import bootstrap_ci, mcnemar_exact


def test_concept_negation():
    assert concept_mentioned("A red car next to a bicycle.", ["bicycle", "bike"])
    assert not concept_mentioned("There is no bicycle in the scene.", ["bicycle"])
    assert not concept_mentioned("The car isn't near a bike", ["bike"])
    assert concept_mentioned("There is no car but a bike is visible", ["bike"])
    assert concept_mentioned("hot dog stand", ["hot dog"])


def test_extract_mentions():
    lex = {"car": ["car", "automobile"], "bicycle": ["bicycle", "bike"]}
    assert extract_mentions("A car and a bike", lex) == {"car", "bicycle"}


def test_chair():
    r = chair([{"car", "bicycle"}, {"dog"}, set()], [{"car"}, {"dog"}, {"cat"}])
    assert r["chair_s"] == pytest.approx(1 / 3)
    assert r["chair_i"] == pytest.approx(1 / 3)  # 1 hallucinated of 3 mentions
    with pytest.raises(ValueError):
        chair([set()], [])


def test_sensitivity():
    assert flip_rate([1, 1, 0], [0, 1, 1]) == pytest.approx(2 / 3)
    s = sensitivity_scores([True, True, False, True], [False, False, True, False])
    assert s["css"] == pytest.approx(0.75 - 0.25)
    assert counterfactual_pair_accuracy(["y", "y"], ["n", "y"], ["y", "y"], ["n", "n"]) == 0.5
    assert residual_mention_rate([True, True, False], [True, False, True]) == 0.5
    assert pmi(-0.2, -1.5) == pytest.approx(1.3)
    assert local_sensitivity(-0.1, -2.0) == pytest.approx(1.9)
    assert math.isnan(residual_mention_rate([False], [True]))


def test_stats():
    est, lo, hi = bootstrap_ci([0, 1] * 50, seed=1)
    assert lo <= est <= hi and 0.35 < lo and hi < 0.65
    est2, lo2, hi2 = bootstrap_ci([1.0, 0.0, 1.0, 0.0], clusters=["a", "a", "b", "b"], seed=1)
    assert lo2 <= est2 <= hi2
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 * (1 / 2 ** 10))
    assert mcnemar_exact(5, 5) == 1.0
