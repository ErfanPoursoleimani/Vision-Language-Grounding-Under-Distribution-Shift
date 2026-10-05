from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence

from tqdm import tqdm

from ..datasets.pope_style import ProbeItem, with_article

DEFAULT_TEMPLATES = ["Is there {a_obj} in the image?", "Does this image contain {a_obj}?",
                     "Can you see {a_obj} in this picture?"]


def make_question(template: str, category: str) -> str:
    return template.format(obj=category, a_obj=with_article(category))


def run_probes(model, items: Sequence[ProbeItem], load_image: Callable[[int], object], templates: Sequence[str],
               blind: bool = True, progress: bool = True) -> List[dict]:
    """Image-major loop (so backends can later share per-image work). Real model calls only."""
    by_img: Dict[int, list] = defaultdict(list)
    for it in items:
        by_img[it.image_id].append(it)
    rows: List[dict] = []
    for image_id in tqdm(sorted(by_img), disable=not progress, desc="images"):
        img = load_image(image_id)
        for it in by_img[image_id]:
            for ti, tpl in enumerate(templates):
                q = make_question(tpl, it.category)
                row = {"image_id": image_id, "category": it.category, "label": it.label, "kind": it.kind,
                       "template_idx": ti, "question": q, "logodds": float(model.yes_no_logodds(img, q))}
                row["blind_logodds"] = float(model.yes_no_logodds(None, q)) if blind else None
                rows.append(row)
    return rows
