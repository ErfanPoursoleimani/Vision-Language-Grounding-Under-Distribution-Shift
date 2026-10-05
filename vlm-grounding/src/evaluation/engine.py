"""Task runners. Each returns the output directory. Real model outputs only; testing-only models are refused
unless allow_testing_only=True (used by unit tests with a tmp output dir)."""
from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

from PIL import Image

from ..datasets.coco import load_coco_instances
from ..datasets.pope_style import build_pope_style
from ..datasets.splits import log_test_access, split_ids
from ..models.cache import CachedVLM
from .aggregate import aggregate_probes, failure_cases, item_level
from .probes import DEFAULT_TEMPLATES, run_probes
from .reporting import write_probe_outputs


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_pope_style(cfg: dict, model, out_root: str | Path, allow_testing_only: bool = False,
                   progress: bool = True, test_log: str | Path | None = "results/test_access_log.jsonl") -> Path:
    testing_only = bool(model.metadata().get("testing_only"))
    if testing_only and not allow_testing_only:
        raise SystemExit("refusing to write results for a testing-only model")
    d, pr = cfg["dataset"], cfg.get("probe", {})
    images, cats = load_coco_instances(d["root"], d.get("split", "val2017"))
    our_split = d.get("our_split", "test")
    ids = split_ids([im.image_id for im in images], seed=d.get("split_seed", 0))
    keep = set(ids[our_split])
    pool = [im for im in images if im.image_id in keep]
    items = build_pope_style(pool, cats, n_images=d.get("n_images", 200), seed=d.get("seed", 0))
    paths = {im.image_id: im.path for im in images}
    if our_split == "test" and test_log and not testing_only:
        log_test_access(test_log, config=cfg["name"], model=model.metadata(), n_items=len(items), git=_git())
    cache_path = cfg.get("cache", f"cache/{model.name}.sqlite")
    m = CachedVLM(model, cache_path)
    t0 = time.perf_counter()
    rows = run_probes(m, items, lambda i: Image.open(paths[i]).convert("RGB"),
                      pr.get("templates", DEFAULT_TEMPLATES), blind=pr.get("blind_baseline", True), progress=progress)
    thr = pr.get("threshold", 0.0)
    summary = aggregate_probes(rows, thr)
    its = item_level(rows, thr)
    fails = failure_cases(its, cfg.get("output", {}).get("top_k_failures", 50))
    out = Path(out_root) / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    manifest = {"config": cfg, "model_metadata": model.metadata(), "git_commit": _git(), "python": platform.python_version(),
                "runtime": model.runtime_info() if hasattr(model, "runtime_info") else None,
                "n_images": len({i.image_id for i in items}), "n_items": len(items), "n_rows": len(rows),
                "seconds": time.perf_counter() - t0, "cache_hits": m.hits, "cache_misses": m.misses,
                "dataset_note": "POPE-style probes built from COCO instance annotations; not the official POPE files"}
    write_probe_outputs(out, rows, its, summary, fails, manifest)
    return out


TASKS = {"pope_style": run_pope_style}
