import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("verify_references", Path("scripts/verify_references.py"))
vr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vr)


def test_bib_parses_and_status_counts():
    entries = vr.parse_bib(Path("docs/references.bib").read_text(encoding="utf-8"))
    keys = {e["key"] for e in entries}
    assert {"li2023pope", "rohrbach2018chair", "radford2021clip"} <= keys
    assert all("STATUS:" in e.get("note", "") for e in entries), "every entry needs an explicit STATUS"
    verified = [e for e in entries if "STATUS: verified" in e["note"]]
    assert len(verified) == 8
    assert all("eprint" in e for e in entries)


def test_norm():
    assert vr.norm("{OPERA}: Alleviating  Hallucination") == "opera alleviating hallucination"
