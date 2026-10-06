#!/usr/bin/env python
"""Measure real peak VRAM and throughput for a configured model on THIS machine.

    python scripts/profile_gpu.py --config configs/models/<model>.yaml --image path.jpg \
        --batch-sizes 1,2,4,8 --max-new-tokens 64 --out results/hardware/<name>.json

Climbs a batch-size ladder and stops at the first CUDA OOM, recording the largest size that
worked. Output is machine measurements only (written to results/hardware/). Requires torch+CUDA
and an implemented backend; it exits with a clear message otherwise (never fabricates numbers).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models import build_model  # noqa: E402
import env_report  # noqa: E402


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def run_ladder(measure, sizes):
    """measure(size) -> dict of measurements. Stops at the first OOM; returns (records, max_ok)."""
    records, max_ok = [], None
    for s in sizes:
        try:
            rec = measure(s)
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                records.append({"batch_size": s, "oom": True})
                break
            raise
        records.append({"batch_size": s, "oom": False, **rec})
        max_ok = s
    return records, max_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe the image in detail.")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch is not installed; install the model extras first (pip install -e .[models]).")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; nothing measured.")

    from PIL import Image
    used_before_mib = env_report.smi_memory_used_mib()  # read BEFORE touching CUDA (excludes our own context)
    cfg = yaml.safe_load(Path(a.config).read_text())
    model = build_model(cfg if "type" in cfg else cfg["model"])
    if model.metadata().get("testing_only"):
        sys.exit("refusing to profile a testing-only model.")
    img = Image.open(a.image).convert("RGB")
    sizes = [int(x) for x in a.batch_sizes.split(",")]

    def measure(bs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = model.generate_batch([img] * bs, [a.prompt] * bs, max_new_tokens=a.max_new_tokens, temperature=0.0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        smi = env_report.smi_snapshot()  # right after the run, while the GPU is still loaded/hot
        n_tok = sum(len(o.token_logprobs or []) for o in outs) or None
        return {"seconds": dt, "images_per_s": bs / dt, "new_tokens_per_s": (n_tok / dt) if n_tok else None,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30, "smi_after": smi}

    measure(1)  # warm-up (kernel selection, lazy init) - discarded
    records, max_ok = run_ladder(measure, sizes)
    free, total = torch.cuda.mem_get_info()
    result = {"model": model.metadata(), "gpu": torch.cuda.get_device_name(0), "total_gib": total / 2 ** 30,
              "nvidia_smi_used_mib_before_cuda_init": used_before_mib,
              "free_gib_after": free / 2 ** 30, "max_batch_ok": max_ok, "records": records,
              "max_new_tokens": a.max_new_tokens, "torch": torch.__version__}
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
