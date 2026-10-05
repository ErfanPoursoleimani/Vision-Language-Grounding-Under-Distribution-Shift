"""CHAIR-style object hallucination rates (after Rohrbach et al., EMNLP 2018).

Deviation from the original: mentions are *unique objects per caption* (sets), not mention
instances. Report this when comparing against published CHAIR numbers.
"""
from __future__ import annotations

from typing import Collection, Sequence


def chair(mentioned: Sequence[Collection[str]], gold: Sequence[Collection[str]]) -> dict:
    if len(mentioned) != len(gold):
        raise ValueError("mentioned and gold must have equal length")
    n_caps = n_caps_h = n_mentions = n_h = 0
    for m, g in zip(mentioned, gold):
        m, g = set(m), set(g)
        h = m - g
        n_caps += 1
        n_caps_h += bool(h)
        n_mentions += len(m)
        n_h += len(h)
    return {
        "chair_s": n_caps_h / n_caps if n_caps else float("nan"),
        "chair_i": n_h / n_mentions if n_mentions else float("nan"),
        "n_captions": n_caps,
        "n_mentions": n_mentions,
    }
