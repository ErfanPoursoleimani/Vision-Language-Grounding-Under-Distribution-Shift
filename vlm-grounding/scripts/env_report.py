#!/usr/bin/env python
"""Collect facts about THIS machine (OS, CPU threads, RAM, GPU, who is using VRAM) as JSON.

    python scripts/env_report.py --out results/hardware/env.json

Run it while the machine is idle (browser closed, on AC power, performance mode) and again under
your normal working conditions: the difference in `gpu.memory_used_mib` is the VRAM that other
processes take from experiments. Nothing is estimated here; fields the driver does not expose
(common on laptops, e.g. power limits) are reported as null.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys

GPU_FIELDS = ["name", "driver_version", "memory.total", "memory.used", "memory.free", "power.limit",
              "power.max_limit", "clocks.max.sm", "clocks.max.mem", "pcie.link.gen.max", "temperature.gpu"]


def parse_csv_query(text: str, fields: list) -> list:
    """Parse `nvidia-smi --format=csv,noheader,nounits` output into dicts; '[N/A]' / '[Not Supported]' -> None."""
    rows = []
    for line in text.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(fields):
            continue
        rows.append({f: (None if v.startswith("[") else v) for f, v in zip(fields, vals)})
    return rows


def _smi(args: list):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        return subprocess.check_output([exe] + args, text=True, stderr=subprocess.STDOUT, timeout=20)
    except Exception:  # noqa: BLE001
        return None


def ram_gib():
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 2 ** 20
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullTotalPhys / 2 ** 30
    except Exception:  # noqa: BLE001
        pass
    return None


def smi_snapshot() -> dict | None:
    """Best-effort live clocks/power/temperature/throttle reasons (field support varies by driver/laptop)."""
    for fields in (["clocks.sm", "power.draw", "temperature.gpu", "clocks_throttle_reasons.active"],
                   ["clocks.sm", "power.draw", "temperature.gpu"], ["clocks.sm", "temperature.gpu"]):
        q = _smi([f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
        if q:
            rows = parse_csv_query(q, fields)
            if rows:
                return rows[0]
    return None


def smi_memory_used_mib():
    """GPU 0 memory.used from nvidia-smi (MiB) or None. Call BEFORE this process initializes CUDA, otherwise
    the process's own CUDA context (a few hundred MiB) is counted as 'used by others'."""
    q = _smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    rows = parse_csv_query(q, ["memory.used"]) if q else []
    try:
        return float(rows[0]["memory.used"]) if rows and rows[0]["memory.used"] is not None else None
    except ValueError:
        return None


def collect() -> dict:
    out = {"platform": platform.platform(), "python": platform.python_version(), "cpu_logical_threads": os.cpu_count(),
           "cpu_name": platform.processor() or None, "ram_total_gib": ram_gib(), "gpu": None, "gpu_processes": None,
           "torch": None}
    q = _smi([f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"])
    if q:
        out["gpu"] = parse_csv_query(q, GPU_FIELDS)
    p = _smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if p is not None:
        out["gpu_processes"] = parse_csv_query(p, ["pid", "process_name", "used_memory_mib"])
    try:
        import torch
        out["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}
    except ImportError:
        pass
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    text = json.dumps(collect(), indent=2)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(text)
