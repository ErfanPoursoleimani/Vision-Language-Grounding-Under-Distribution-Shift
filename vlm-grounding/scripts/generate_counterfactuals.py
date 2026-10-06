#!/usr/bin/env python
"""Generate (and audit visually) the synthetic counterfactual pairs.

    python scripts/generate_counterfactuals.py --split dev --n-scenes 40 --out data/synthetic/dev
Writes images/<pair_id>_{orig,cf}.png, pairs.jsonl and preview.png (a grid of one example per edit type, with
the questions whose gold answer changes). LOOK at preview.png: this is the human audit step of the protocol.
Images are deterministic functions of the scene specs, so the evaluation can also regenerate them in memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.counterfactual import make_pairs  # noqa: E402
from src.perturbations.synthetic import question_text, render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synthetic/dev")
    ap.add_argument("--no-images", action="store_true", help="only write preview + pairs.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    pairs = make_pairs(a.split, a.n_scenes, a.seed)
    with open(out / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps({"pair_id": p.pair_id, "edit_type": p.edit_type, "role": p.role, "area_ratio": p.area_ratio,
                                "probes": [{"q": question_text(x.question, 0), "kind": x.kind, "gold_orig": x.gold_orig,
                                            "gold_cf": x.gold_cf} for x in p.probes]}) + "\n")
            if not a.no_images:
                render(p.orig).save(out / "images" / f"{p.pair_id}_orig.png")
                render(p.cf).save(out / "images" / f"{p.pair_id}_cf.png")
    ets = ["remove", "recolor", "reshape", "swap", "insert", "background"]
    first = {e: next((p for p in pairs if p.edit_type == e and p.role in ("relevant", "background")), None) for e in ets}
    fig, axes = plt.subplots(len(ets), 4, figsize=(11, 2.6 * len(ets)))
    for r, e in enumerate(ets):
        p = first[e]
        if p is None:
            continue
        ctrl = next((c for c in pairs if c.matched_to == p.pair_id), None)
        shown = [(p, "original"), (p, "edited"), (ctrl, "control: original") if ctrl else (None, ""), (ctrl, "control: edited") if ctrl else (None, "")]
        for c, (pp, lab) in enumerate(shown):
            ax = axes[r][c]; ax.axis("off")
            if pp is None:
                continue
            ax.imshow(render(pp.orig if "original" in lab else pp.cf))
            ax.set_title(f"{e}: {lab}", fontsize=8)
        ch = [x for x in p.probes if x.gold_orig != x.gold_cf]
        axes[r][0].text(0.0, -0.04, "gold answer changes: " + " | ".join(f"{question_text(x.question, 0)} {'Y' if x.gold_orig else 'N'}->{'Y' if x.gold_cf else 'N'}" for x in ch[:2]),
                        fontsize=6.5, transform=axes[r][0].transAxes, ha="left", va="top", clip_on=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02, hspace=0.35, wspace=0.05)
    fig.savefig(out / "preview.png", dpi=110)
    print(f"{len(pairs)} pairs from {a.n_scenes} {a.split} scenes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
