"""Model-agnostic VLM interface.

Every backend (CLIP-style, BLIP-2, LLaVA-family, ...) implements this interface so that
datasets, perturbations, metrics and the evaluation engine never import a concrete model.
Backends declare what they support via ``capabilities``; callers must check or call
``require``. ``image=None`` means "no image" and is used for language-prior (blind) baselines.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Capability(str, Enum):
    GENERATE = "generate"
    SCORE = "score"  # log p(continuation | image, prompt)
    IMAGE_TEXT_SIMILARITY = "image_text_similarity"
    HIDDEN_STATES = "hidden_states"
    ATTENTIONS = "attentions"
    ABLATE_IMAGE_TOKENS = "ablate_image_tokens"


class UnsupportedCapability(NotImplementedError):
    pass


@dataclass
class ImageContext:
    """Handle for "one image, many questions". Backends may stash a prefilled KV cache in ``state``
    so the (expensive) image-token prefill is paid once per image instead of once per probe."""
    image: Any
    prefix: str = ""
    state: Any = None


@dataclass
class VLMOutput:
    text: str
    token_logprobs: Optional[list] = None
    meta: dict = field(default_factory=dict)


class VLM(ABC):
    capabilities: frozenset = frozenset()

    def __init__(self, name: str, **config: Any) -> None:
        self.name = name
        self.config = dict(config)

    # ---- capability handling -------------------------------------------------
    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def require(self, cap: Capability) -> None:
        if not self.supports(cap):
            raise UnsupportedCapability(f"{self.name} does not support '{cap.value}'")

    # ---- required API --------------------------------------------------------
    @abstractmethod
    def generate(self, image: Any, prompt: str, *, max_new_tokens: int = 128,
                 temperature: float = 0.0, seed: int = 0) -> VLMOutput:
        """Generate text. temperature=0 must mean greedy decoding."""

    # ---- optional API --------------------------------------------------------
    def score(self, image: Any, prompt: str, continuation: str) -> float:
        """Sum of token log-probabilities of ``continuation`` given image and prompt."""
        raise UnsupportedCapability(f"{self.name} does not implement score()")

    def generate_batch(self, images, prompts, **kw):
        """Default: sequential. Backends with real batching should override (throughput)."""
        return [self.generate(i, p, **kw) for i, p in zip(images, prompts)]

    def open_context(self, image: Any, prefix: str = "") -> ImageContext:
        """Fallback has no speed-up; backends override to prefill image tokens once and reuse them."""
        return ImageContext(image=image, prefix=prefix)

    def score_in_context(self, ctx: ImageContext, suffix: str, continuation: str) -> float:
        return self.score(ctx.image, ctx.prefix + suffix, continuation)

    def yes_no_logodds(self, image: Any, question: str) -> float:
        """log p('Yes') - log p('No') for a closed question (POPE-style probe)."""
        return self.score(image, question, "Yes") - self.score(image, question, "No")

    def similarity(self, image: Any, text: str) -> float:
        raise UnsupportedCapability(f"{self.name} does not implement similarity()")

    def hidden_states(self, image: Any, prompt: str, **kw: Any):
        raise UnsupportedCapability(f"{self.name} does not implement hidden_states()")

    def attentions(self, image: Any, prompt: str, **kw: Any):
        raise UnsupportedCapability(f"{self.name} does not implement attentions()")

    def metadata(self) -> dict:
        """Everything needed to reproduce / cache-key this model (id, revision, dtype, quantization...)."""
        return {"name": self.name, **self.config}


def logsigmoid(x: float) -> float:
    return -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))
