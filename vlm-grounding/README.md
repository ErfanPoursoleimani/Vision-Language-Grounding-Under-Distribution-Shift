# Vision-Language Grounding Under Distribution Shift

Research project: how robustly do open vision-language models ground their text in visual evidence? Controlled counterfactual image edits, hallucination and compositional evaluation, representation analysis, and lightweight (LoRA) adaptation — designed for a 6 GB GPU with an option to use an external one.

## Status (honest)
| Phase | State |
|---|---|
| 1 Literature review | Draft; 8 references verified, the rest flagged unverified; 2025–2026 sweep not done (`docs/literature_review.md`) |
| 1b Evaluation protocol | Draft v0.1, thresholds not frozen (`docs/evaluation_protocol.md`) |
| 2 Model abstraction | Interface, registry, disk cache, shared-image-context API, test-only dummy: done. BLIP-2 backend run on GPU (smoke test passed); generic chat-VLM backend for Qwen3-VL (`hf_chat_vlm`) written, **not yet run**; CLIP and LLaVA backends: not yet |
| Hardware | VRAM budget estimator + GPU profiler written (`docs/hardware_plan.md`); **never run on a GPU yet** |
| 3–4 Dataset loader + baseline evaluation | COCO loader, hashed dev/test split with test-access log, POPE-style probes, blind baseline, aggregation, CSV/JSON/plots/failure cases: written, tested on a synthetic fixture with the dummy model. **Not yet run on real COCO + BLIP-2** |
| 5–6 Perturbation generator + counterfactual benchmark | **Synthetic track** written and tested (exact edits, matched controls, binding foils, independent pixel verification, AFR/SFR/CSS/CPA metrics, thresholds frozen dev→test). Verified with test models only; **not yet run on BLIP-2**. Real-photo track (COCO masks + inpainting): not started |
| 10 Robustness | **Label-preserving degradations** (blur, noise, JPEG, contrast, pixelate; 5 severities) with thresholds frozen on clean data, AUROC drop with CIs, consistency: written and tested with the dummy model; **not yet run on a real model**. Crop/occlusion deliberately live in the counterfactual track (they can change the correct answer) |
| 7–9, 11–12 | Not started (CHAIR-style caption task, layer-wise interventions, mitigation, ablations, paper) |

No results exist yet. Nothing in this repo reports model outputs.

## Quickstart
```bash
pip install -r requirements.txt
python -m pytest -q                                   # 81 tests
python scripts/run_evaluation.py --config configs/evaluation/example_dryrun.yaml --dry-run
python scripts/env_report.py --out results/hardware/env.json   # what is using your VRAM / RAM / power limit
python scripts/run_evaluation.py --config configs/evaluation/pope_style_blip2_dev.yaml   # needs data/coco, see docs/datasets.md
python scripts/smoke_model.py --config configs/models/blip2_opt_2p7b.yaml --out results/hardware/smoke_blip2.json   # GPU, downloads weights
python scripts/verify_references.py docs/references.bib   # needs internet
python -m src.utils.memory_budget --quantized-params-b 2.8 --unquantized-params-b 0.6 --layers 36 --kv-heads 2 --head-dim 128 --seq-len 800
```
`run_evaluation.py` currently validates the config, builds the model and writes a manifest; it stops with an explicit error rather than producing placeholder numbers.

## Counterfactual (synthetic) workflow
1. Inspect the edits: `python scripts/generate_counterfactuals.py --split dev --n-scenes 10 --out data/synthetic/dev --no-images` and open `data/synthetic/dev/preview.png`.
2. Dev run (~12 min at the measured speed, fits thresholds): `python scripts/run_evaluation.py --config configs/evaluation/cf_synthetic_blip2_dev.yaml`
3. Read `results.json`, `plots/sensitivity.png`, `failure_images/`. Check the precondition in `experiments/counterfactual/synthetic_v1_plan.md` first (accuracy on original images vs blind).
4. Competence check on original images (~8 min): `python scripts/run_evaluation.py --config configs/evaluation/competence_synthetic_blip2_dev.yaml`
4b. Scene-difficulty check (is the weak competence due to small objects?): `competence_synthetic_blip2_dev_medium.yaml` and `..._large.yaml` (same task, bigger/fewer objects; `dataset.style`). Image style is also selectable for `generate_counterfactuals.py --style`.
5. Recompute a finished run (writes `results_v2.json` INSIDE that run's own folder; the competence runs have no such file) with the current metric code (no model needed): `python scripts/reaggregate_counterfactual.py --run-dir results/<run>`
6. Second model family: `python scripts/smoke_model.py --config configs/models/qwen3vl_4b_instruct_nf4.yaml --out results/hardware/smoke_qwen3vl.json`, then `competence_synthetic_qwen3vl_dev_small.yaml` and `..._large.yaml`.
7. Robustness (label-preserving degradations; ~1 h for BLIP-2, run it for Qwen too): `python scripts/run_evaluation.py --config configs/evaluation/pope_perturbed_blip2_dev.yaml` (or `..._qwen3vl_dev.yaml`)
8. Second model, in this order: `cf_synthetic_qwen3vl_dev.yaml` (counterfactual), `competence_degraded_qwen3vl_dev_small.yaml` (which capability breaks first under blur/noise/JPEG/contrast/pixelate on synthetic scenes), `pope_style_qwen3vl_dev.yaml` (COCO, same protocol as BLIP-2), `pope_perturbed_qwen3vl_dev.yaml`. Predictions are written down in `experiments/counterfactual/2026-10-25_competence_qwen3vl_dev.md` BEFORE these runs.
9. Audit COCO false positives, then test H1: `python scripts/audit_failures.py sheet --run-dir results/<your actual pope_style run folder> --coco-root data/coco --n 45 --select stratified --out audit` (answer the question printed above each image in the `verdict` column: yes = object visible, no = not visible, ? = unclear; `--select top` is biased toward annotation gaps), `python scripts/audit_failures.py summarize --csv audit/audit_template.csv`, then `python scripts/analyze_h1.py --run-dir results/<run> --audit audit/audit_template.csv` (several audit files can be given; to finish a census of all candidates use `--select all --exclude <earlier audit CSVs>`).
10. Test run only with frozen thresholds (`cf_synthetic_blip2_test.yaml`).

## Pinning the model revision (do before any test run)
`resolved_commit_hash` was null in the first run. The snapshot folder name is the commit hash:
`dir %USERPROFILE%\.cache\huggingface\hub\models--Salesforce--blip2-opt-2.7b\snapshots` (or under `HF_HOME`). Put it in `configs/models/blip2_opt_2p7b.yaml` as `revision:`. This changes the cache key, so probes are recomputed once (~18 min for the dev set). The runtime info now also tries to resolve it from the cache automatically.

## Dev → test workflow
1. Dev run (writes `thresholds.json`): `python scripts/run_evaluation.py --config configs/evaluation/pope_style_blip2_dev.yaml`
2. Analysis: `python scripts/analyze_probes.py --run-dir results/<dev run>`
3. Test run only with frozen thresholds: set `dataset.our_split: test` and `probe.thresholds_file: results/<dev run>/thresholds.json` (the runner refuses otherwise and logs the access).

## Integrity rules
Model outputs are only ever machine-generated and stored; the `dummy_brightness` model is for plumbing tests and cannot write to `results/`. Attention maps are never used alone as evidence of grounding. Unverified references must not be cited.

## Roadmap
Literature → model abstraction → dataset loaders → baseline evaluation → perturbation generator → counterfactual benchmark → grounding metrics → representation/attention analysis → mitigation → robustness → ablations → paper.
