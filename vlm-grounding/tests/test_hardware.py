import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from src.models import build_model
from src.utils.memory_budget import estimate, kv_cache_gib, lora_params, weights_gib

spec = importlib.util.spec_from_file_location("profile_gpu", Path("scripts/profile_gpu.py"))
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)


def test_kv_exact():
    # 2 * 32 layers * 32 heads * 128 dim * 800 tokens * 2 bytes = 419,430,400 B
    assert kv_cache_gib(32, 32, 128, 800) * 2 ** 30 == 419_430_400


def test_weights_ordering_and_fp16():
    assert weights_gib(7, "fp16") == pytest.approx(7e9 * 2 / 2 ** 30)
    assert weights_gib(7, "nf4_dq") < weights_gib(7, "nf4") < weights_gib(7, "int8") < weights_gib(7, "fp16")


def test_lora_params_formula():
    # single square matrix pair check via tiny model: hidden=8, inter=16, heads=2, kv=2, hd=4, 1 layer, r=2
    n = lora_params(2, 8, 16, 2, 2, 4, 1, targets="attn")
    assert n == 4 * 2 * (8 + 8)  # q,k,v,o each r*(in+out)=2*16


def test_estimate_flags_overflow():
    big = estimate(quantized_params_b=13, unquantized_params_b=0.3, quant="nf4_dq", layers=40, kv_heads=40,
                   head_dim=128, seq_len=800)
    small = estimate(quantized_params_b=3, unquantized_params_b=0.5, quant="nf4_dq", layers=36, kv_heads=2,
                     head_dim=128, seq_len=800)
    assert not big["fits"] and small["fits"]


def test_ladder_stops_at_first_oom():
    calls = []

    def measure(s):
        calls.append(s)
        if s >= 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return {"seconds": 1.0}

    recs, ok = pg.run_ladder(measure, [1, 2, 4, 8])
    assert ok == 2 and calls == [1, 2, 4] and recs[-1] == {"batch_size": 4, "oom": True}
    with pytest.raises(ValueError):
        pg.run_ladder(lambda s: (_ for _ in ()).throw(ValueError("x")), [1])


def test_context_fallback_matches_direct_score():
    m = build_model({"type": "dummy_brightness"})
    img = Image.new("RGB", (4, 4), (200, 200, 200))
    ctx = m.open_context(img, prefix="")
    assert m.score_in_context(ctx, "q", "Yes") == m.score(img, "q", "Yes")
    assert [o.text for o in m.generate_batch([img, img], ["q", "q"])] == ["Yes", "Yes"]


def test_env_report_parsing_and_graceful_absence():
    spec2 = importlib.util.spec_from_file_location("env_report", Path("scripts/env_report.py"))
    er = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(er)
    rows = er.parse_csv_query("NVIDIA GeForce RTX 3050, 555.1, 6144, 300, 5844, [N/A], [N/A], 1500, 7000, 4, 45",
                              er.GPU_FIELDS)
    assert rows[0]["memory.total"] == "6144" and rows[0]["power.limit"] is None
    assert er.parse_csv_query("bad,line", er.GPU_FIELDS) == []
    info = er.collect()  # must not raise on machines without nvidia-smi
    assert "cpu_logical_threads" in info


def test_vram_from_env_and_snapshot(tmp_path):
    import json
    from src.utils.memory_budget import vram_from_env
    f = tmp_path / "env.json"
    f.write_text(json.dumps({"gpu": [{"memory.free": "6002"}]}))
    assert vram_from_env(str(f)) == pytest.approx(6002 / 1024)
    spec3 = importlib.util.spec_from_file_location("env_report2", Path("scripts/env_report.py"))
    er = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(er)
    assert er.smi_snapshot() is None or isinstance(er.smi_snapshot(), dict)  # None without nvidia-smi
