"""Counterfactual sensitivity metrics. Definitions and rationale: docs/evaluation_protocol.md (M2-M5)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _mean(x) -> float:
    x = list(x)
    return float(np.mean(x)) if x else float("nan")


def flip_rate(pred_orig: Sequence, pred_cf: Sequence) -> float:
    if len(pred_orig) != len(pred_cf):
        raise ValueError("length mismatch")
    return _mean(a != b for a, b in zip(pred_orig, pred_cf))


def counterfactual_pair_accuracy(pred_o, pred_c, gold_o, gold_c) -> float:
    """M2: fraction of pairs answered correctly on BOTH the original and the counterfactual image."""
    n = len(pred_o)
    if not (n == len(pred_c) == len(gold_o) == len(gold_c)):
        raise ValueError("length mismatch")
    return _mean(po == go and pc == gc for po, pc, go, gc in zip(pred_o, pred_c, gold_o, gold_c))


def sensitivity_scores(relevant_flips: Sequence[bool], control_flips: Sequence[bool]) -> dict:
    """M3. AFR = P(answer changes | relevant edit); SFR = P(answer changes | irrelevant control edit).
    CSS = AFR - SFR. Control edits use the same editing pipeline on concepts the question is NOT about."""
    afr, sfr = _mean(relevant_flips), _mean(control_flips)
    return {"afr": afr, "sfr": sfr, "css": afr - sfr}


def residual_mention_rate(mentioned_orig: Sequence[bool], mentioned_cf: Sequence[bool]) -> float:
    """M5: among pairs where the concept was mentioned for the original image and then removed,
    the fraction still mentioned for the edited image (persistence = ungrounded mention)."""
    kept = [c for o, c in zip(mentioned_orig, mentioned_cf) if o]
    return _mean(kept)


def pmi(logp_with_image: float, logp_blank: float) -> float:
    """Global visual information gain: log p(a|I,q) - log p(a|no image,q)."""
    return logp_with_image - logp_blank


def local_sensitivity(logp_orig: float, logp_cf: float) -> float:
    """Drop in log-likelihood of the original-image answer after the evidence is edited out."""
    return logp_orig - logp_cf
