"""Back-of-envelope VRAM budgeting BEFORE downloading or loading a model.

These are ESTIMATES (they ignore allocator fragmentation, kernel workspaces and framework
overhead). Their purpose is to rule configurations in or out cheaply; `scripts/profile_gpu.py`
measures the truth and its numbers override anything computed here.
"""
from __future__ import annotations

import argparse
import json

GIB = 1024 ** 3
# Approximate effective bits per quantized weight (block-wise scales included). Assumptions, not specs.
EFFECTIVE_BITS = {"fp16": 16.0, "int8": 8.25, "nf4": 4.5, "nf4_dq": 4.13}
OPT_BYTES_PER_PARAM = {"adamw32": 8.0, "adamw8bit": 2.0}


def weights_gib(quantized_params_b: float, quant: str, unquantized_params_b: float = 0.0) -> float:
    """quantized_params_b: billions of params in quantized Linear layers.
    unquantized_params_b: embeddings / lm_head / vision tower etc. kept in fp16 (2 bytes)."""
    return (quantized_params_b * 1e9 * EFFECTIVE_BITS[quant] / 8 + unquantized_params_b * 1e9 * 2) / GIB


def kv_cache_gib(layers: int, kv_heads: int, head_dim: int, seq_len: int, batch: int = 1, bytes_per: int = 2) -> float:
    return 2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per / GIB


def lora_params(r: int, hidden: int, intermediate: int, n_heads: int, kv_heads: int, head_dim: int,
                layers: int, targets: str = "all") -> int:
    q = (hidden, n_heads * head_dim)
    k = v = (hidden, kv_heads * head_dim)
    o = (n_heads * head_dim, hidden)
    mlp = [(hidden, intermediate), (hidden, intermediate), (intermediate, hidden)]
    mats = [q, k, v, o] + (mlp if targets == "all" else [])
    return layers * sum(r * (i + o_) for i, o_ in mats)


def train_state_gib(trainable: int, optimizer: str = "adamw8bit") -> float:
    """fp32 LoRA weights (4B) + fp32 grads (4B) + optimizer state."""
    return trainable * (4 + 4 + OPT_BYTES_PER_PARAM[optimizer]) / GIB


def estimate(*, quantized_params_b, unquantized_params_b, quant, layers, kv_heads, head_dim, seq_len,
             batch=1, train=False, lora=None, optimizer="adamw8bit", activation_reserve_gib=0.75,
             cuda_context_gib=0.4, display_gib=0.0, vram_gib=6.0) -> dict:
    parts = {
        "weights": weights_gib(quantized_params_b, quant, unquantized_params_b),
        "kv_cache": kv_cache_gib(layers, kv_heads, head_dim, seq_len, batch),
        "activations_reserve": activation_reserve_gib,
        "cuda_context": cuda_context_gib,
        "display_and_other_processes": display_gib,
    }
    if train:
        n = lora_params(**lora)
        parts["lora_train_state"] = train_state_gib(n, optimizer)
        parts["lora_trainable_params_M"] = n / 1e6  # informational, not GiB
    total = sum(v for k, v in parts.items() if not k.endswith("_M"))
    return {"parts_gib": {k: round(v, 3) for k, v in parts.items()}, "total_gib": round(total, 3),
            "budget_gib": vram_gib, "headroom_gib": round(vram_gib - total, 3), "fits": total <= vram_gib}


def vram_from_env(path: str) -> float:
    """Free VRAM (GiB) of GPU 0 as measured by scripts/env_report.py (memory.free, MiB)."""
    info = json.load(open(path))
    return float(info["gpu"][0]["memory.free"]) / 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantized-params-b", type=float, required=True)
    ap.add_argument("--unquantized-params-b", type=float, default=0.0)
    ap.add_argument("--quant", choices=list(EFFECTIVE_BITS), default="nf4_dq")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--vram-gib", type=float, default=6.0)
    ap.add_argument("--env", help="env_report.json: use its measured free VRAM instead of --vram-gib")
    ap.add_argument("--display-gib", type=float, default=0.0)
    a = ap.parse_args()
    if a.env:
        a.vram_gib = vram_from_env(a.env)
    print(json.dumps(estimate(quantized_params_b=a.quantized_params_b, unquantized_params_b=a.unquantized_params_b,
                              quant=a.quant, layers=a.layers, kv_heads=a.kv_heads, head_dim=a.head_dim,
                              seq_len=a.seq_len, batch=a.batch, vram_gib=a.vram_gib, display_gib=a.display_gib), indent=2))


if __name__ == "__main__":
    main()
