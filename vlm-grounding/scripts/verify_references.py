#!/usr/bin/env python
"""Check docs/references.bib against arXiv / Crossref. Run LOCALLY (needs internet).

    python scripts/verify_references.py docs/references.bib            # report only
    python scripts/verify_references.py docs/references.bib --strict   # exit 1 if any entry fails

For each entry with an `eprint` (arXiv id) it fetches the arXiv title and compares it to the bib
title (normalized); for entries with a `doi` it also compares the Crossref title. It does NOT
check authors, venue or year automatically: a pass means "the title resolves to this identifier",
so review venue/year by hand and then change STATUS to `verified <date>`.
(The offline parser is covered by tests/test_verify_parse.py; the network calls were not exercised
in the sandbox where this script was written.)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


def parse_bib(text: str) -> list:
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S):
        body = m.group(3)
        fields = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,|$)", body, flags=re.S | re.M)}
        entries.append({"key": m.group(2).strip(), "type": m.group(1), **fields})
    return entries


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"[{}\\]", "", s).lower()).strip()


def arxiv_title(eprint: str) -> str:
    url = "http://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(eprint)
    root = ET.fromstring(urllib.request.urlopen(url, timeout=20).read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    t = root.find("a:entry/a:title", ns)
    return " ".join(t.text.split()) if t is not None and t.text else ""


def crossref_title(doi: str) -> str:
    data = json.load(urllib.request.urlopen("https://api.crossref.org/works/" + urllib.parse.quote(doi), timeout=20))
    return (data["message"].get("title") or [""])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bib")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    bad = 0
    for e in parse_bib(open(a.bib, encoding="utf-8").read()):
        checks = []
        try:
            if "eprint" in e:
                checks.append(("arXiv", arxiv_title(e["eprint"])))
            if "doi" in e:
                checks.append(("Crossref", crossref_title(e["doi"])))
        except Exception as ex:  # network / parse problems are reported, not hidden
            print(f"[ERROR] {e['key']}: {ex}")
            bad += 1
            continue
        if not checks:
            print(f"[SKIP ] {e['key']}: no eprint/doi to check")
            continue
        for src, title in checks:
            ok = norm(title) == norm(e.get("title", ""))
            print(f"[{'OK   ' if ok else 'DIFF '}] {e['key']} via {src}: {title!r}")
            bad += not ok
    return 1 if (a.strict and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
