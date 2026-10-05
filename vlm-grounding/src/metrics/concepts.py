"""Lexicon-based concept mention extraction (with crude negation handling).

Known limitation: synonym lexicons and a 3-token negation window miss paraphrase, scope
and implicature. Extraction precision/recall must be audited by hand before any reported
mention-based number (see docs/evaluation_protocol.md, M5).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Set

NEGATION_CUES = {"no", "not", "without", "neither", "nor", "lacks", "lacking", "absent", "never"}


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def concept_mentioned(text: str, synonyms: Iterable[str], handle_negation: bool = True) -> bool:
    toks = _tokens(text)
    for syn in synonyms:
        s = _tokens(syn)
        n = len(s)
        if n == 0:
            continue
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] != s:
                continue
            if handle_negation:
                window = toks[max(0, i - 3):i]
                if any(w in NEGATION_CUES or w.endswith("n't") for w in window):
                    continue
            return True
    return False


def extract_mentions(text: str, lexicon: Dict[str, Iterable[str]], handle_negation: bool = True) -> Set[str]:
    return {c for c, syns in lexicon.items() if concept_mentioned(text, syns, handle_negation)}
