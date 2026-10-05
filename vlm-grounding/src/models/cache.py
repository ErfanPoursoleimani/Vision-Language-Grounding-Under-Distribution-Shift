"""On-disk cache for model outputs (sqlite). Key = hash(model metadata, image pixels, prompt, params)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .base import VLM, VLMOutput


def image_fingerprint(image) -> str:
    if image is None:
        return "no-image"
    rgb = image.convert("RGB")
    h = hashlib.sha256(rgb.tobytes())
    h.update(str(rgb.size).encode())
    return h.hexdigest()


class CachedVLM:
    """Wraps a VLM; delegates everything except generate/score, which are cached."""

    def __init__(self, model: VLM, path: str | Path):
        self.model = model
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.hits = 0
        self.misses = 0

    def __getattr__(self, item):
        return getattr(self.model, item)

    def _key(self, op, image, prompt, extra) -> str:
        blob = json.dumps([op, self.model.metadata(), image_fingerprint(image), prompt, extra],
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _get(self, k):
        row = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return None if row is None else json.loads(row[0])

    def _put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
        self.db.commit()

    def generate(self, image, prompt, **kw):
        k = self._key("generate", image, prompt, kw)
        v = self._get(k)
        if v is not None:
            self.hits += 1
            return VLMOutput(**v)
        self.misses += 1
        out = self.model.generate(image, prompt, **kw)
        self._put(k, {"text": out.text, "token_logprobs": out.token_logprobs, "meta": out.meta})
        return out

    def score(self, image, prompt, continuation):
        k = self._key("score", image, prompt, continuation)
        v = self._get(k)
        if v is not None:
            self.hits += 1
            return v
        self.misses += 1
        s = self.model.score(image, prompt, continuation)
        self._put(k, s)
        return s

    def yes_no_logodds(self, image, question):
        k = self._key("yn", image, question, None)
        v = self._get(k)
        if v is not None:
            self.hits += 1
            return v
        self.misses += 1
        v = self.model.yes_no_logodds(image, question)
        self._put(k, v)
        return v
