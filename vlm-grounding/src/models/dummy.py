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
