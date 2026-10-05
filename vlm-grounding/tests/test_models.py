import pytest
from PIL import Image

from src.models import Capability, UnsupportedCapability, available_models, build_model
from src.models.cache import CachedVLM


def _img(v):
    return Image.new("RGB", (8, 8), (v, v, v))


def test_registry_and_error():
    assert "dummy_brightness" in available_models()
    with pytest.raises(KeyError, match="not implemented yet"):
        build_model({"type": "hf_llava"})


def test_dummy_counterfactual_flip_and_blind_prior():
    m = build_model({"type": "dummy_brightness", "name": "dummy"})
    assert m.metadata()["testing_only"] is True
    assert m.generate(_img(220), "Is there a car?").text == "Yes"
    assert m.generate(_img(30), "Is there a car?").text == "No"
    # image-ablated (blind) prior is exactly uninformative for this dummy
    assert m.yes_no_logodds(None, "q") == pytest.approx(0.0)
    assert m.yes_no_logodds(_img(220), "q") > 0 > m.yes_no_logodds(_img(30), "q")


def test_capability_guard():
    m = build_model({"type": "dummy_brightness"})
    with pytest.raises(UnsupportedCapability):
        m.require(Capability.ATTENTIONS)
    with pytest.raises(UnsupportedCapability):
        m.hidden_states(None, "x")


def test_cache(tmp_path):
    m = CachedVLM(build_model({"type": "dummy_brightness"}), tmp_path / "c.sqlite")
    a = m.generate(_img(200), "q", max_new_tokens=5)
    b = m.generate(_img(200), "q", max_new_tokens=5)
    assert a.text == b.text and (m.hits, m.misses) == (1, 1)
    m.generate(_img(10), "q", max_new_tokens=5)
    assert m.misses == 2
    s1, s2 = m.score(_img(200), "q", "Yes"), m.score(_img(200), "q", "Yes")
    assert s1 == s2 and m.hits == 2
