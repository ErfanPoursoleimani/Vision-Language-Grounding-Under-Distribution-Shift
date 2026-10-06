#!/usr/bin/env python
"""Hardware/plumbing smoke test for ONE model backend on THIS machine.

    python scripts/smoke_model.py --config configs/models/blip2_opt_2p7b.yaml [--image photo.jpg] \
        --out results/hardware/smoke_blip2.json

Loads the model, reports VRAM/time, quantization census, a few real outputs, and runs consistency
checks that catch implementation bugs (not research claims). The JSON it writes contains only
what the model actually produced / what was measured. Exit code is 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models import build_model  # noqa: E402
import env_report  # noqa: E402


def synthetic_images():
    a = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(a).rectangle([100, 120, 280, 260], fill=(200, 30, 30))
    b = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(b).ellipse([100, 100, 280, 280], fill=(30, 60, 200))
    return [a, b]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", help="optional real photo (used as image A)")
    ap.add_argument("--question", default="Is there a person in the image?")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    cfg = cfg if "type" in cfg else cfg["model"]
    model = build_model(cfg)
    imgs = synthetic_images()
    if a.image:
        imgs[0] = Image.open(a.image).convert("RGB")

    report: dict = {"model": model.metadata(), "checks": {}, "outputs": {},
                    "nvidia_smi_used_mib_before_cuda_init": env_report.smi_memory_used_mib()}
    torch = None
    try:
        import torch  # noqa: F811
    except ImportError:
        pass
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        report["checks"][name] = {"pass": bool(cond), "detail": detail}
        ok &= bool(cond)
        print(("[PASS] " if cond else "[FAIL] ") + name + (f"  ({detail})" if detail else ""))

    try:
        t0 = time.perf_counter()
        if hasattr(model, "runtime_info"):
            report["runtime"] = model.runtime_info()  # triggers load
        report["load_seconds"] = time.perf_counter() - t0
        if torch is not None and torch.cuda.is_available():
            report["vram_after_load_gib"] = torch.cuda.memory_allocated() / 2 ** 30
            report["vram_peak_during_load_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30

        q = a.question
        cap = model.generate(imgs[0], "", max_new_tokens=30)
        vqa = model.generate(imgs[0], q, max_new_tokens=10)
        report["outputs"] = {"caption_A": cap.text, "vqa_A": vqa.text, "question": q}
        print("caption A :", cap.text, "\nvqa A     :", vqa.text)

        s_yes, s_no = model.score(imgs[0], q, "Yes"), model.score(imgs[0], q, "No")
        lo1 = model.yes_no_logodds(imgs[0], q)
        lo_blind = model.yes_no_logodds(None, q)
        report["outputs"].update(score_yes=s_yes, score_no=s_no, logodds_single_forward=lo1, logodds_blind=lo_blind)
        check("log-probs finite and <= 0", all(math.isfinite(x) and x <= 1e-3 for x in (s_yes, s_no)))
        check("p(Yes)+p(No) <= 1", math.exp(s_yes) + math.exp(s_no) <= 1.001, f"{math.exp(s_yes)+math.exp(s_no):.4f}")
        check("single-forward log-odds == score(Yes)-score(No)", abs(lo1 - (s_yes - s_no)) < 0.15,
              f"{lo1:.4f} vs {s_yes - s_no:.4f}")
        check("blind baseline finite", math.isfinite(lo_blind), f"{lo_blind:.4f}")

        g1 = model.generate(imgs[0], q, max_new_tokens=10).text
        check("greedy decoding deterministic", g1 == vqa.text, f"{g1!r} vs {vqa.text!r}")

        batch = model.generate_batch([imgs[0], imgs[1]], [q, q], max_new_tokens=10)
        single = [model.generate(i, q, max_new_tokens=10).text for i in imgs]
        report["outputs"].update(batch=[b.text for b in batch], single=single)
        check("batch == single decoding", [b.text for b in batch] == single, f"{[b.text for b in batch]} vs {single}")

        # throughput probe (timing only)
        t0 = time.perf_counter()
        for _ in range(5):
            model.yes_no_logodds(imgs[0], q)
        if torch is not None:
            torch.cuda.synchronize()
        report["yes_no_probes_per_s"] = 5 / (time.perf_counter() - t0)
        if torch is not None and torch.cuda.is_available():
            report["vram_peak_total_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            report["vram_reserved_gib"] = torch.cuda.memory_reserved() / 2 ** 30
    except Exception:  # noqa: BLE001
        ok = False
        report["error"] = traceback.format_exc()
        print(report["error"])

    report["all_checks_passed"] = ok
    text = json.dumps(report, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print("wrote", a.out)
    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
