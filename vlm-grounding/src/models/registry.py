from __future__ import annotations

from typing import Callable, Dict

from .base import VLM

_REGISTRY: Dict[str, Callable[..., VLM]] = {}


def register_model(key: str):
    def deco(cls):
        if key in _REGISTRY:
            raise ValueError(f"model type '{key}' already registered")
        _REGISTRY[key] = cls
        return cls
    return deco


def available_models() -> list:
    return sorted(_REGISTRY)


def build_model(cfg: dict) -> VLM:
    """cfg = {"type": <registry key>, "name": <label>, "params": {...}}"""
    key = cfg["type"]
    if key not in _REGISTRY:
        raise KeyError(
            f"model type '{key}' is not implemented yet. Available: {available_models()}. "
            "Real backends (CLIP / BLIP-2 / LLaVA-family) are added in phase 2b."
        )
    return _REGISTRY[key](name=cfg.get("name", key), **cfg.get("params", {}))
