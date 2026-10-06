"""Synthetic scenes with EXACT ground truth, and the edits applied to them.

Scenes: 3-4 coloured shapes (unique colour per object, shapes may repeat) on a plain background.
Because the scene is a data structure, every question's answer is computed, never annotated, and every
edit (remove / recolor / reshape / swap positions / insert / background change) is exact and leaves
no inpainting artifacts. Images are deterministic functions of the scene spec.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

CANVAS = 384
COLORS = {"red": (220, 40, 40), "green": (40, 160, 60), "blue": (40, 80, 220),
          "yellow": (240, 210, 40), "purple": (140, 60, 180), "orange": (240, 140, 30)}
SHAPES = ("circle", "square", "triangle")
BACKGROUNDS = {"white": (255, 255, 255), "light gray": (225, 225, 225), "beige": (240, 230, 210)}
MIN_SIZE, MAX_SIZE, MIN_DIST, MIN_DX = 56, 84, 100, 60


@dataclass(frozen=True)
class Obj:
    color: str
    shape: str
    cx: int
    cy: int
    size: int

    @property
    def area(self) -> float:
        f = {"circle": math.pi / 4, "square": 1.0, "triangle": 0.5}[self.shape]
        return f * self.size ** 2


@dataclass(frozen=True)
class Scene:
    objects: Tuple[Obj, ...]
    background: str = "white"


# ---------------------------------------------------------------- rendering
def render(scene: Scene) -> Image.Image:
    img = Image.new("RGB", (CANVAS, CANVAS), BACKGROUNDS[scene.background])
    d = ImageDraw.Draw(img)
    for o in scene.objects:
        h, col = o.size / 2, COLORS[o.color]
        if o.shape == "circle":
            d.ellipse([o.cx - h, o.cy - h, o.cx + h, o.cy + h], fill=col)
        elif o.shape == "square":
            d.rectangle([o.cx - h, o.cy - h, o.cx + h, o.cy + h], fill=col)
        else:
            d.polygon([(o.cx, o.cy - h), (o.cx - h, o.cy + h), (o.cx + h, o.cy + h)], fill=col)
    return img


def detect_colors(img: Image.Image, min_pixels: int = 300) -> dict:
    """Independent (pixel-based) check: palette colour -> (pixel count, centroid x, centroid y)."""
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.int32)
    ys, xs = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    out = {}
    for c, ref in COLORS.items():
        m = ((arr - np.array(ref)) ** 2).sum(axis=2) < 40 ** 2
        n = int(m.sum())
        if n >= min_pixels:
            out[c] = (n, float(xs[m].mean()), float(ys[m].mean()))
    return out


# ---------------------------------------------------------------- questions
@dataclass(frozen=True)
class Exists:
    color: str
    shape: str


@dataclass(frozen=True)
class LeftOf:
    c1: str
    s1: str
    c2: str
    s2: str


@dataclass(frozen=True)
class HasColor:
    color: str


@dataclass(frozen=True)
class HasShape:
    shape: str


def answer(scene: Scene, q) -> bool:
    if isinstance(q, HasColor):
        return any(o.color == q.color for o in scene.objects)
    if isinstance(q, HasShape):
        return any(o.shape == q.shape for o in scene.objects)
    if isinstance(q, Exists):
        return any(o.color == q.color and o.shape == q.shape for o in scene.objects)
    a = next((o for o in scene.objects if o.color == q.c1 and o.shape == q.s1), None)
    b = next((o for o in scene.objects if o.color == q.c2 and o.shape == q.s2), None)
    return bool(a and b and a.cx < b.cx)  # False if either is missing (we only ask when both exist)


def _article(phrase: str) -> str:
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


EXISTS_TEMPLATES = ["Is there {a_obj} in the image?", "Does this image contain {a_obj}?",
                    "Can you see {a_obj} in this picture?"]
LEFT_TEMPLATES = ["Is the {o1} to the left of the {o2}?", "Is the {o1} on the left side of the {o2}?",
                  "Is the {o1} located left of the {o2}?"]


COLOR_TEMPLATES = ["Is there {a_col} object in the image?", "Does this image contain something {col}?",
                   "Can you see {a_col} shape in this picture?"]
SHAPE_TEMPLATES = ["Is there {a_shape} in the image?", "Does this image contain {a_shape}?",
                   "Can you see {a_shape} in this picture?"]


def question_text(q, template_idx: int) -> str:
    if isinstance(q, HasColor):
        return COLOR_TEMPLATES[template_idx].format(a_col=_article(q.color), col=q.color)
    if isinstance(q, HasShape):
        return SHAPE_TEMPLATES[template_idx].format(a_shape=_article(q.shape))
    if isinstance(q, Exists):
        p = f"{q.color} {q.shape}"
        return EXISTS_TEMPLATES[template_idx].format(a_obj=_article(p), obj=p)
    return LEFT_TEMPLATES[template_idx].format(o1=f"{q.c1} {q.s1}", o2=f"{q.c2} {q.s2}")


# ---------------------------------------------------------------- scene generation
def _place(rng: random.Random, existing: List[Obj], color: str, shape: Optional[str] = None,
           size: Optional[int] = None) -> Optional[Obj]:
    for _ in range(300):
        s = size or rng.randint(MIN_SIZE, MAX_SIZE)
        m = s // 2 + 10
        cx, cy = rng.randint(m, CANVAS - m), rng.randint(m, CANVAS - m)
        if all(math.hypot(cx - o.cx, cy - o.cy) >= MIN_DIST for o in existing):
            return Obj(color, shape or rng.choice(SHAPES), cx, cy, s)
    return None


def random_scene(rng: random.Random, n_objects: Optional[int] = None) -> Scene:
    n = n_objects or rng.choice([3, 4])
    for _ in range(100):
        colors = rng.sample(list(COLORS), n)
        objs: List[Obj] = []
        for c in colors:
            o = _place(rng, objs, c)
            if o is None:
                break
            objs.append(o)
        if len(objs) == n and len({o.shape for o in objs}) > 1:
            return Scene(tuple(objs), rng.choice(list(BACKGROUNDS)))
    raise RuntimeError("could not place scene")


# ---------------------------------------------------------------- edits (all exact)
def remove(s: Scene, i: int) -> Scene:
    return replace(s, objects=tuple(o for k, o in enumerate(s.objects) if k != i))


def recolor(s: Scene, i: int, color: str) -> Scene:
    return replace(s, objects=tuple(replace(o, color=color) if k == i else o for k, o in enumerate(s.objects)))


def reshape(s: Scene, i: int, shape: str) -> Scene:
    return replace(s, objects=tuple(replace(o, shape=shape) if k == i else o for k, o in enumerate(s.objects)))


def swap(s: Scene, i: int, j: int) -> Scene:
    a, b = s.objects[i], s.objects[j]
    objs = list(s.objects)
    objs[i], objs[j] = replace(a, cx=b.cx, cy=b.cy), replace(b, cx=a.cx, cy=a.cy)
    return replace(s, objects=tuple(objs))


def insert(s: Scene, o: Obj) -> Scene:
    return replace(s, objects=s.objects + (o,))


def set_background(s: Scene, name: str) -> Scene:
    return replace(s, background=name)


# ---------------------------------------------------------------- pairs
@dataclass
class Probe:
    question: object
    kind: str            # exists | relation | foil | other
    gold_orig: bool
    gold_cf: bool


@dataclass
class Pair:
    pair_id: str
    scene_id: int
    edit_type: str       # remove | recolor | reshape | swap | insert | background
    role: str            # relevant | control | background
    orig: Scene
    cf: Scene
    probes: List[Probe]
    area_frac: float
    matched_to: Optional[str] = None
    area_ratio: Optional[float] = None


def _colors_in(q) -> set:
    if isinstance(q, (Exists, HasColor)):
        return {q.color}
    if isinstance(q, HasShape):
        return set()
    return {q.c1, q.c2}


def _mk(pair_id, scene_id, edit, role, orig, cf, questions_kinds, area, matched_to=None, ratio=None,
        keep_only_unchanged=False, touched=frozenset()) -> Optional[Pair]:
    """touched = colours of objects the edit acts on. Foil/other probes mentioning them are dropped so that
    controls are pure (a question about 'red square' is not independent of removing the red circle)."""
    probes = []
    for q, kind in questions_kinds:
        if kind in ("foil", "other") and (_colors_in(q) & set(touched)):
            continue
        go, gc = answer(orig, q), answer(cf, q)
        if keep_only_unchanged and go != gc:
            continue
        probes.append(Probe(q, kind, go, gc))
    if not probes:
        return None
    if role == "relevant" and not any(p.gold_orig != p.gold_cf for p in probes):
        return None
    if role in ("control", "background") and any(p.gold_orig != p.gold_cf for p in probes):
        return None
    return Pair(pair_id, scene_id, edit, role, orig, cf, probes, area / CANVAS ** 2, matched_to, ratio)


def _foil(scene: Scene, rng: random.Random, avoid: Tuple = ()) -> Optional[Exists]:
    """A colour/shape combination that is ABSENT although both colour and shape occur in the scene (binding foil)."""
    cands = [Exists(a.color, b.shape) for a in scene.objects for b in scene.objects
             if a.shape != b.shape and not answer(scene, Exists(a.color, b.shape)) and Exists(a.color, b.shape) not in avoid]
    return rng.choice(cands) if cands else None


def _closest_area(scene: Scene, target: int, rng: random.Random) -> Tuple[Optional[int], Optional[float]]:
    others = [k for k in range(len(scene.objects)) if k != target]
    if not others:
        return None, None
    ta = scene.objects[target].area
    k = min(others, key=lambda k: (abs(scene.objects[k].area - ta), k))
    return k, scene.objects[k].area / ta


def build_pairs(scene_id: int, scene: Scene, rng: random.Random) -> List[Pair]:
    pairs: List[Pair] = []
    O = scene.objects
    n = len(O)
    unused = [c for c in COLORS if c not in {o.color for o in O}]

    def add(p):
        if p is not None:
            pairs.append(p)

    def base_questions(i, extra):
        qs = [(Exists(O[i].color, O[i].shape), "exists")] + extra
        f = _foil(scene, rng)
        if f is not None:
            qs.append((f, "foil"))
        return qs

    # ---- remove
    i = rng.randrange(n)
    other = [k for k in range(n) if k != i]
    qs = base_questions(i, [(Exists(O[other[0]].color, O[other[0]].shape), "other")])
    pid = f"{scene_id}-remove"
    add(_mk(pid + "-relevant", scene_id, "remove", "relevant", scene, remove(scene, i), qs, O[i].area, touched={O[i].color}))
    m, ratio = _closest_area(scene, i, rng)
    if m is not None:
        qs_c = [(q, k) for q, k in qs if not (isinstance(q, Exists) and q.color == O[m].color and q.shape == O[m].shape)]
        add(_mk(pid + "-control", scene_id, "remove", "control", scene, remove(scene, m), qs_c, O[m].area,
                pid + "-relevant", ratio, keep_only_unchanged=True, touched={O[m].color}))

    # ---- recolor
    if len(unused) >= 2:
        i = rng.randrange(n)
        new = unused[0]
        qs = [(Exists(O[i].color, O[i].shape), "exists"), (Exists(new, O[i].shape), "exists")]
        f = _foil(scene, rng)
        if f is not None:
            qs.append((f, "foil"))
        pid = f"{scene_id}-recolor"
        add(_mk(pid + "-relevant", scene_id, "recolor", "relevant", scene, recolor(scene, i, new), qs, O[i].area,
                touched={O[i].color, new}))
        m, ratio = _closest_area(scene, i, rng)
        if m is not None:
            add(_mk(pid + "-control", scene_id, "recolor", "control", scene, recolor(scene, m, unused[1]),
                    qs, O[m].area, pid + "-relevant", ratio, keep_only_unchanged=True,
                    touched={O[m].color, unused[1]}))

    # ---- reshape
    i = rng.randrange(n)
    new_shape = rng.choice([s for s in SHAPES if s != O[i].shape])
    qs = [(Exists(O[i].color, O[i].shape), "exists"), (Exists(O[i].color, new_shape), "exists")]
    f = _foil(scene, rng)
    if f is not None:
        qs.append((f, "foil"))
    pid = f"{scene_id}-reshape"
    add(_mk(pid + "-relevant", scene_id, "reshape", "relevant", scene, reshape(scene, i, new_shape), qs, O[i].area,
            touched={O[i].color}))
    m, ratio = _closest_area(scene, i, rng)
    if m is not None:
        add(_mk(pid + "-control", scene_id, "reshape", "control", scene,
                reshape(scene, m, rng.choice([s for s in SHAPES if s != O[m].shape])), qs, O[m].area,
                pid + "-relevant", ratio, keep_only_unchanged=True, touched={O[m].color}))

    # ---- swap positions (relation flips)
    sep = [(a, b) for a in range(n) for b in range(a + 1, n) if abs(O[a].cx - O[b].cx) >= MIN_DX]
    if sep:
        a, b = rng.choice(sep)
        q = LeftOf(O[a].color, O[a].shape, O[b].color, O[b].shape)
        pid = f"{scene_id}-swap"
        add(_mk(pid + "-relevant", scene_id, "swap", "relevant", scene, swap(scene, a, b), [(q, "relation")],
                (O[a].area + O[b].area) / 2))
        rest = [k for k in range(n) if k not in (a, b)]
        if len(rest) >= 2:
            k, l = rest[0], rest[1]
            add(_mk(pid + "-control", scene_id, "swap", "control", scene, swap(scene, k, l), [(q, "relation")],
                    (O[k].area + O[l].area) / 2, pid + "-relevant", None, keep_only_unchanged=True))

    # ---- insert
    if len(unused) >= 2:
        size = rng.randint(MIN_SIZE, MAX_SIZE)
        new = _place(rng, list(O), unused[0], size=size)
        ctrl = _place(rng, list(O) + ([new] if new else []), unused[1], size=size)
        if new is not None:
            qs = [(Exists(new.color, new.shape), "exists"), (Exists(O[0].color, O[0].shape), "other")]
            pid = f"{scene_id}-insert"
            add(_mk(pid + "-relevant", scene_id, "insert", "relevant", scene, insert(scene, new), qs, new.area,
                touched={new.color}))
            if ctrl is not None:
                add(_mk(pid + "-control", scene_id, "insert", "control", scene, insert(scene, ctrl), qs, ctrl.area,
                        pid + "-relevant", ctrl.area / new.area, keep_only_unchanged=True, touched={ctrl.color}))

    # ---- background change (irrelevant to every object question)
    bg = rng.choice([b for b in BACKGROUNDS if b != scene.background])
    qs = [(Exists(o.color, o.shape), "exists") for o in O]
    f = _foil(scene, rng)
    if f is not None:
        qs.append((f, "foil"))
    add(_mk(f"{scene_id}-background", scene_id, "background", "background", scene, set_background(scene, bg), qs, 0.0))
    return pairs


# ---------------------------------------------------------------- competence items (original images only)
def competence_items(scene: Scene, rng: random.Random) -> list:
    """(kind, question) on ONE scene. kinds: color_atom, shape_atom, exists_true, foil, easy_neg, relation.
    Used to find out what the model can perceive at all, before interpreting any sensitivity number."""
    O = scene.objects
    items = []
    present_c = {o.color for o in O}
    absent_c = [c for c in COLORS if c not in present_c]
    for c in sorted(present_c):
        items.append(("color_atom", HasColor(c)))
    for c in rng.sample(absent_c, min(3, len(absent_c))):
        items.append(("color_atom", HasColor(c)))
    for sh in SHAPES:
        items.append(("shape_atom", HasShape(sh)))
    for o in O:
        items.append(("exists_true", Exists(o.color, o.shape)))
    foils = [Exists(a.color, b.shape) for a in O for b in O if a.shape != b.shape and not answer(scene, Exists(a.color, b.shape))]
    for q in rng.sample(foils, min(3, len(foils))):
        items.append(("foil", q))
    for c in rng.sample(absent_c, min(2, len(absent_c))):
        items.append(("easy_neg", Exists(c, rng.choice(SHAPES))))
    sep = [(a, b) for a in range(len(O)) for b in range(a + 1, len(O)) if abs(O[a].cx - O[b].cx) >= MIN_DX]
    for a, b in rng.sample(sep, min(2, len(sep))):
        items.append(("relation", LeftOf(O[a].color, O[a].shape, O[b].color, O[b].shape)))
        items.append(("relation", LeftOf(O[b].color, O[b].shape, O[a].color, O[a].shape)))
    return items
