#!/usr/bin/env python
"""Human audit of COCO existence 'errors' (is a false positive really a hallucination, or an object COCO did not annotate?).

  python scripts/audit_failures.py sheet --run-dir results/<run> --coco-root data/coco --n 40 --out audit
      -> audit/sheet_01.png ... (8 images per sheet, numbered) and audit/audit_template.csv (fill the 'verdict' column)
  python scripts/audit_failures.py summarize --csv audit/audit_template.csv
      -> shares of: hallucination | present_unannotated | ambiguous
Verdicts answer the question printed above each image ("is there a cup?"): yes = the object IS visible (the COCO label was wrong),
no = not visible (the model hallucinated), ? = unclear/tiny/partly visible. (Long names also work: present_unannotated,
hallucination, ambiguous.) Cases are shown in random order without the model's confidence. Use --select stratified or random for an
unbiased estimate: --select top shows only the most confident false positives, which are the most likely to be annotation gaps."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.datasets.coco import load_coco_instances  # noqa: E402
from src.evaluation.h1 import AUDIT_VERDICTS, audit_summary  # noqa: E402


def sheet(a) -> int:
    import random
    run = Path(a.run_dir)
    items = [json.loads(l) for l in (run / "per_item.jsonl").read_text().splitlines() if l.strip()]
    if a.kind == "fp":
        pool = [i for i in items if i["pred"] == 1 and i["label"] == 0]
        pool.sort(key=lambda i: -i["mean_logodds"])                       # most confident first
    else:
        pool = [i for i in items if i["pred"] == 0 and i["label"] == 1]
        pool.sort(key=lambda i: i["mean_logodds"])                        # most confidently missed first
    seen, uniq = set(), []
    for i in pool:                                                        # an (image, category) pair can occur under several negative kinds
        k = (i["image_id"], i["category"])
        if k not in seen:
            seen.add(k)
            uniq.append(i)
    if a.exclude:  # skip items that were already judged in earlier audit files
        done = set()
        for path in a.exclude:
            with open(path, newline="", encoding="utf-8") as f:
                done |= {(int(r["image_id"]), r["category"]) for r in csv.DictReader(f) if (r.get("verdict") or "").strip()}
        uniq = [i for i in uniq if (i["image_id"], i["category"]) not in done]
    rng = random.Random(a.seed)
    if a.select == "all":
        picked = list(uniq)
    elif a.select == "top":
        picked = uniq[:a.n]
    elif a.select == "random":
        picked = rng.sample(uniq, min(a.n, len(uniq)))
    else:  # stratified: equal numbers from the confident / middle / least-confident thirds (uniq is ordered by confidence)
        third = max(1, len(uniq) // 3)
        parts = [uniq[:third], uniq[third:2 * third], uniq[2 * third:]]
        per = max(1, a.n // 3)
        picked = [x for p_ in parts for x in rng.sample(p_, min(per, len(p_)))]
    rng.shuffle(picked)                                                   # judge in random order, blind to confidence
    images, _ = load_coco_instances(a.coco_root, a.split)
    paths = {im.image_id: im.path for im in images}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_sheet = 8
    for s_ in range(0, len(picked), per_sheet):
        chunk = picked[s_:s_ + per_sheet]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (j, f) in zip(axes.flat, enumerate(chunk, start=s_ + 1)):
            ax.imshow(Image.open(paths[f["image_id"]]).convert("RGB"))
            ax.set_title(f"#{j}  is there {'an' if f['category'][0] in 'aeiou' else 'a'} {f['category']}?", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / f"sheet_{s_ // per_sheet + 1:02d}.png", dpi=90)
        plt.close(fig)
    with open(out / "audit_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "image_id", "category", "model_mean_logodds", "verdict"])
        for j, x in enumerate(picked, start=1):
            w.writerow([j, x["image_id"], x["category"], round(x["mean_logodds"], 3), ""])
    print(f"{len(picked)} cases ({a.select}) from {len(uniq)} unique candidates -> {out}/sheet_*.png and {out}/audit_template.csv")
    print("Answer the question printed above each image in the 'verdict' column: yes (object visible) / no (not visible) / ?  (unclear).")
    return 0


def summarize(a) -> int:
    with open(a.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    try:
        res = audit_summary(rows)
    except ValueError as e:
        raise SystemExit(str(e))
    print(json.dumps(res, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--run-dir", required=True); s.add_argument("--coco-root", default="data/coco")
    s.add_argument("--split", default="val2017"); s.add_argument("--n", type=int, default=40)
    s.add_argument("--kind", choices=["fp", "fn"], default="fp"); s.add_argument("--out", default="audit")
    s.add_argument("--select", choices=["top", "random", "stratified", "all"], default="stratified",
                   help="top = most confident only (BIASED toward annotation gaps); random; stratified = equal thirds by confidence; all = census of every candidate")
    s.add_argument("--exclude", nargs="*", default=[], help="earlier audit CSVs whose already-judged items are skipped (use with --select all to finish a census)")
    s.add_argument("--seed", type=int, default=0)
    u = sub.add_parser("summarize")
    u.add_argument("--csv", required=True)
    a = ap.parse_args()
    return sheet(a) if a.cmd == "sheet" else summarize(a)


if __name__ == "__main__":
    raise SystemExit(main())
