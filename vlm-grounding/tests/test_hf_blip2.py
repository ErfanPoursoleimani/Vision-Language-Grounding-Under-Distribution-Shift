import importlib.util

import pytest

from src.models import available_models, build_model


def test_registered_and_validated():
    assert "hf_blip2" in available_models()
    with pytest.raises(ValueError):
        build_model({"type": "hf_blip2", "params": {"quantize": "3bit"}})
    with pytest.raises(ValueError):
        build_model({"type": "hf_blip2", "params": {"dtype": "float32"}})


def test_prompt_formatting_and_blind_image():
    m = build_model({"type": "hf_blip2", "name": "b"})
    assert m.format_prompt("") == "" and m.format_prompt("   ") == ""
    assert m.format_prompt("Is there a car?") == "Question: Is there a car? Answer:"
    assert m.format_prompt("Question: x Answer:") == "Question: x Answer:"
    img = m._as_rgb(None)
    assert img.size == (224, 224) and img.getpixel((5, 5)) == (128, 128, 128)
    md = m.metadata()
    assert md["quantize"] == "4bit" and md["revision"] is None  # cache key inputs, config only


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="only meaningful without torch")
def test_helpful_error_without_torch():
    m = build_model({"type": "hf_blip2"})
    with pytest.raises(ImportError, match="needs torch"):
        m.generate(None, "q")


def test_smoke_script_parses():
    import py_compile
    py_compile.compile("scripts/smoke_model.py", doraise=True)


def test_cached_commit_hash(tmp_path):
    from src.models.hf_blip2 import cached_commit_hash
    repo = tmp_path / "models--org--name"
    (repo / "refs").mkdir(parents=True)
    (repo / "snapshots" / "abc123").mkdir(parents=True)
    assert cached_commit_hash("org/name", hub_cache=str(tmp_path)) == "abc123"  # single snapshot fallback
    (repo / "refs" / "main").write_text("def456\n")
    assert cached_commit_hash("org/name", hub_cache=str(tmp_path)) == "def456"
    assert cached_commit_hash("org/missing", hub_cache=str(tmp_path)) is None


def test_chat_backend_pure_parts():
    from src.models.hf_chat_vlm import build_messages, yes_no_token_ids
    m = build_model({"type": "hf_chat_vlm", "name": "q"})
    assert m.closed_question("Is there a car?") == "Is there a car? Answer with Yes or No."
    assert m.metadata()["blind_mode"] == "text_only" and m._as_image(None) is None
    g = build_model({"type": "hf_chat_vlm", "params": {"blind_mode": "gray"}})
    assert g._as_image(None).size == (224, 224)
    with pytest.raises(ValueError):
        build_model({"type": "hf_chat_vlm", "params": {"blind_mode": "none"}})
    with pytest.raises(ValueError):
        build_model({"type": "hf_chat_vlm", "params": {"quantize": "2bit"}})
    msgs = build_messages("Q?", True)
    assert msgs == [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q?"}]}]
    assert build_messages("Q?", False, "sys")[0]["role"] == "system" and build_messages("Q?", False)[0]["content"] == [{"type": "text", "text": "Q?"}]
    table = {"Yes": [10], "yes": [11], " Yes": [12], " yes": [11], "No": [20], "no": [21], " No": [22], " no": [21]}
    yes, no = yes_no_token_ids(lambda s: table[s])
    assert yes == [10, 11, 12] and no == [20, 21, 22]
    assert yes_no_token_ids(lambda s: [7])[0] == [] and yes_no_token_ids(lambda s: [])[1] == []  # overlap/empty handled


def test_default_closed_question_is_identity():
    assert build_model({"type": "dummy_brightness"}).closed_question("q?") == "q?"
    assert build_model({"type": "hf_blip2"}).closed_question("q?") == "q?"


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="only meaningful without torch")
def test_chat_backend_helpful_error_without_torch():
    with pytest.raises(ImportError, match="needs torch"):
        build_model({"type": "hf_chat_vlm"}).generate(None, "q")


def test_extend_token_aligned_fields():
    import numpy as np

    from src.models.hf_chat_vlm import extend_token_aligned
    enc = {"input_ids": np.ones((1, 168), dtype=np.int64), "attention_mask": np.ones((1, 168), dtype=np.int64),
           "mm_token_type_ids": np.full((1, 168), 1, dtype=np.int64),           # per-token int field: must grow
           "pixel_values": np.zeros((576, 1536), dtype=np.float32),             # float, other shape: untouched
           "image_grid_thw": np.array([[1, 12, 12]], dtype=np.int64),           # int but not token-aligned: untouched
           "float_per_token": np.zeros((1, 168), dtype=np.float32)}             # float per-token: untouched
    out = extend_token_aligned(enc, 168, 2, lambda xs, d: np.concatenate(xs, axis=d),
                               lambda shape, like: np.zeros(shape, dtype=like.dtype))
    assert out["mm_token_type_ids"].shape == (1, 170) and (out["mm_token_type_ids"][0, -2:] == 0).all()
    assert out["mm_token_type_ids"][0, :168].min() == 1
    assert out["input_ids"].shape == (1, 168) and out["attention_mask"].shape == (1, 168)  # caller extends these itself
    assert out["pixel_values"].shape == (576, 1536) and out["image_grid_thw"].shape == (1, 3)
    assert out["float_per_token"].shape == (1, 168)


def test_skip_patterns_match_like_transformers_5_18():
    import re

    from src.models.hf_chat_vlm import skip_patterns

    def should_convert_module(full_name, patterns):  # same logic as transformers 5.18.0 quantizers_utils (read from the wheel)
        return not any(re.match(f"{k}\\.", full_name) or re.match(f"{k}", full_name) or full_name.endswith(k) for k in patterns)

    pats = skip_patterns(False)
    keep_fp = ["model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1", "visual.blocks.3.mlp.fc1",
               "model.vision_tower.encoder.layers.0.q_proj", "vision_model.encoder.layers.0.self_attn.qkv",
               "model.multi_modal_projector.linear_1", "lm_head"]
    quantize = ["model.language_model.layers.0.self_attn.q_proj", "model.language_model.layers.35.mlp.down_proj",
                "language_model.model.decoder.layers.0.fc1"]
    assert not any(should_convert_module(n, pats) for n in keep_fp)
    assert all(should_convert_module(n, pats) for n in quantize)
    assert should_convert_module("model.visual.blocks.0.attn.qkv", ["visual", "lm_head"])  # the OLD pattern failed to skip it (the bug)
    assert should_convert_module("model.visual.blocks.0.attn.qkv", skip_patterns(True))        # quantize_vision=True quantizes it
