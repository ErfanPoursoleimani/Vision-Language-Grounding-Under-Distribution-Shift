# Vision-Language Grounding Under Distribution Shift

Research project: how robustly do open vision-language models ground their text in visual evidence? Controlled counterfactual image edits, hallucination and compositional evaluation, representation analysis, and lightweight (LoRA) adaptation — designed for a 6 GB GPU with an option to use an external one.

## Status (honest)
| Phase | State |
|---|---|
| 1 Literature review | Draft; 8 references verified, the rest flagged unverified; 2025–2026 sweep not done (`docs/literature_review.md`) |
| 1b Evaluation protocol | Draft v0.1, thresholds not frozen (`docs/evaluation_protocol.md`) |
| 2 Model abstraction | Interface, registry, disk cache, shared-image-context API, test-only dummy: done. BLIP-2 backend written (`src/models/hf_blip2.py`) but **not yet run on a GPU**; CLIP and LLaVA-family backends: not yet |
| Hardware | VRAM budget estimator + GPU profiler written (`docs/hardware_plan.md`); **never run on a GPU yet** |
| 3–4 Dataset loader + baseline evaluation | COCO loader, hashed dev/test split with test-access log, POPE-style probes, blind baseline, aggregation, CSV/JSON/plots/failure cases: written, tested on a synthetic fixture with the dummy model. **Not yet run on real COCO + BLIP-2** |
| 5–12 | Not started (perturbations, counterfactual benchmark, CHAIR-style caption task, interventions, mitigation, robustness, ablations, paper) |

No results exist yet. Nothing in this repo reports model outputs.

## Quickstart
```bash
pip install -r requirements.txt
python -m pytest -q                                   # 34 tests
python scripts/run_evaluation.py --config configs/evaluation/example_dryrun.yaml --dry-run
python scripts/env_report.py --out results/hardware/env.json   # what is using your VRAM / RAM / power limit
python scripts/run_evaluation.py --config configs/evaluation/pope_style_blip2_dev.yaml   # needs data/coco, see docs/datasets.md
python scripts/smoke_model.py --config configs/models/blip2_opt_2p7b.yaml --out results/hardware/smoke_blip2.json   # GPU, downloads weights
python scripts/verify_references.py docs/references.bib   # needs internet
python -m src.utils.memory_budget --quantized-params-b 2.8 --unquantized-params-b 0.6 --layers 36 --kv-heads 2 --head-dim 128 --seq-len 800
```
`run_evaluation.py` currently validates the config, builds the model and writes a manifest; it stops with an explicit error rather than producing placeholder numbers.

## Integrity rules
Model outputs are only ever machine-generated and stored; the `dummy_brightness` model is for plumbing tests and cannot write to `results/`. Attention maps are never used alone as evidence of grounding. Unverified references must not be cited.

## Roadmap
Literature → model abstraction → dataset loaders → baseline evaluation → perturbation generator → counterfactual benchmark → grounding metrics → representation/attention analysis → mitigation → robustness → ablations → paper.
