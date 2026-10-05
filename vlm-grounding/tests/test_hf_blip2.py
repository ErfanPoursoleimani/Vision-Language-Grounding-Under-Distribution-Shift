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
