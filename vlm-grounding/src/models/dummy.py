"""Deterministic stand-in used ONLY to unit-test the pipeline plumbing.

It answers from mean image brightness. It is not a language model, its outputs are never
research results, and anything produced with it is tagged ``testing_only``.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Capability, VLM, VLMOutput, logsigmoid
from .registry import register_model


def _brightness(image) -> float:
    if image is None:
        return 127.5
    return float(np.asarray(image.convert("L"), dtype=np.float64).mean())


@register_model("dummy_brightness")
class DummyBrightnessVLM(VLM):
    capabilities = frozenset({Capability.GENERATE, Capability.SCORE})

    def _logit_yes(self, image) -> float:
        return (_brightness(image) - 127.5) / 32.0

    def generate(self, image, prompt, *, max_new_tokens=128, temperature=0.0, seed=0):
        yes = self._logit_yes(image) > 0
        return VLMOutput(text="Yes" if yes else "No", meta={"testing_only": True})

    def score(self, image, prompt, continuation):
        z = self._logit_yes(image)
        c = continuation.strip().lower()
        if c == "yes":
            return logsigmoid(z)
        if c == "no":
            return logsigmoid(-z)
        return -float(len(continuation)) * math.log(2.0)

    def metadata(self):
        return {**super().metadata(), "testing_only": True}


_DET_CACHE: dict = {}


@register_model("dummy_color_oracle")
class DummyColorOracleVLM(VLM):
    """TESTING ONLY. Answers synthetic-scene questions from pixel colours (ignores shapes), so the counterfactual
    pipeline can be tested end-to-end: it should be perfect on colour-only facts and fooled by binding foils."""
    capabilities = frozenset({Capability.GENERATE, Capability.SCORE})

    def generate(self, image, prompt, *, max_new_tokens=128, temperature=0.0, seed=0):
        return VLMOutput(text="Yes" if self.yes_no_logodds(image, prompt) > 0 else "No", meta={"testing_only": True})

    def yes_no_logodds(self, image, question):
        import re

        from ..perturbations.synthetic import COLORS, SHAPES, detect_colors
        if image is None:
            return 0.0
        found = re.findall(r"\b(" + "|".join(COLORS) + r")\s+(?:" + "|".join(SHAPES) + r")\b", question.lower())
        import hashlib
        key = hashlib.md5(image.convert("RGB").tobytes()).hexdigest()
        cache = _DET_CACHE
        if key not in cache:
            cache[key] = detect_colors(image)
        det = cache[key]
        if not found:  # atomic questions: colour-only is answerable from pixels, shape-only is not (oracle is shape-blind)
            cols = re.findall(r"\b(" + "|".join(COLORS) + r")\b", question.lower())
            return (3.0 if cols[0] in det else -3.0) if len(cols) == 1 else 0.0
        if len(found) == 1:
            return 3.0 if found[0] in det else -3.0
        if len(found) == 2:
            a, b = found
            return 3.0 if (a in det and b in det and det[a][1] < det[b][1]) else -3.0
        return 0.0

    def score(self, image, prompt, continuation):
        z = self.yes_no_logodds(image, prompt)
        return logsigmoid(z) if continuation.strip().lower() == "yes" else logsigmoid(-z)

    def metadata(self):
        return {**super().metadata(), "testing_only": True}
