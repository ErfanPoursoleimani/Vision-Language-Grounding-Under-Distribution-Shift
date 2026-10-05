# Experiment log — POPE-style probes, BLIP-2 OPT-2.7B (NF4 LM), COCO val2017 **dev** split

**Status: exploratory, dev split only. No test split has been touched.** Numbers below are copied from the run's own output files (`results.json`, `per_item.jsonl`, `per_example.jsonl`, `manifest.json`) and from `scripts/analyze_probes.py`, which reproduces them.

## Setup (from manifest)
Config `pope_style_blip2_dev`, git `b417e2709c80c4493b015c34f446e1e16f1aa171`, torch 2.11.0+cu128, transformers 5.18.0, model revision **not pinned** (`resolved_commit_hash` was null), 200 dev images, 2,400 items (3 present + 3 random + 3 popular + 3 adversarial absent per image), 3 question paraphrases, 7,200 model probes + blind (gray-image) probes, 1,080 s, 6,036 cache misses / 8,364 hits. Raw threshold: log-odds(Yes) − log-odds(No) > 0.

## Results (threshold 0, mean over 3 templates; 95% CI resamples images)
| negatives | accuracy (CI) | pooled AUROC | blind accuracy | model − blind |
|---|---|---|---|---|
| random | 0.722 (0.704–0.740) | 0.817 | 0.506 | +0.216 |
| popular | 0.667 (0.652–0.682) | 0.779 | 0.536 | +0.131 |
| adversarial | 0.649 (0.634–0.663) | 0.756 | 0.524 | +0.125 |

## Observations (what the data show)
1. **Accuracy at a fixed threshold is dominated by template calibration, not by discrimination.** Template 3 ("Can you see … in this picture?") has yes-ratio 0.75–0.87, versus 0.39–0.47 for templates 1–2; its accuracy on adversarial negatives is 0.555 versus 0.696. Pooled AUROC (positives vs all negatives) is nearly identical: 0.790 / 0.782 / 0.780. Balanced accuracy at threshold 0 is 0.720 / 0.718 / 0.600; with a per-template threshold fit on this same split (so optimistic) it is 0.723 / 0.719 / 0.712. Cross-template Pearson correlation of log-odds on the same (image, object): 0.74–0.86.
2. **Discrimination is real but modest.** AUROC ≈ 0.78–0.84 by negative type; log-odds range only about ±1, and the present/absent histograms overlap heavily.
3. **Harder negatives are harder, in the direction POPE reports.** False-positive rate at the item threshold: random 0.248, popular 0.330, adversarial 0.382. **Caveat:** 457 of 1,332 unique negative (image, object) pairs appear in more than one negative kind (mostly popular ∩ adversarial), so these subsets are not independent, and COCO annotation gaps (small or occluded objects such as cups, bottles, chairs, tables) can turn "false positives" into correct answers. A manual audit of the top false positives is required before calling any of them hallucinations.
4. **The gray-image baseline does not explain the false positives, but it is a weak control.** Of 576 false positives, only 9 (1.6%) have a blind "yes"; the blind model says "no" to ~92% of questions. That measures the response to *no evidence*, not necessarily a language prior; scene-conditioned priors (e.g. kitchen → cup) are not captured. Model and blind log-odds correlate 0.36 on absent items (0.09 on present). This control cannot separate "visual misperception" from "context prior"; the counterfactual edits planned in phase 5 can.
5. **Per-category numbers are not reliable at this size** (most categories have < 15 negatives in 200 images). Spearman(category frequency, false-positive rate) = 0.32 over 34 categories: weak, confounded with annotation completeness; no claim made.
6. Top-50 false positives are led by dining table, bottle, chair, cup, car, bowl; top-50 omissions by person (18), car, handbag, backpack. Labels in `failure_cases.jsonl` are *candidates*.

## Measured hardware (smoke test, same machine)
Model load 1,219 s (first load, includes download/quantization), VRAM after load 3.70 GiB, peak 3.74 GiB, reserved 3.80 GiB (my estimate was ≈ 4.3 GiB); 8.4 yes/no probes/s on a repeated image; ≈ 5.6 model probes/s effective in the full run. All smoke consistency checks passed. Quantization census: language model Linear4bit ×192; vision tower, Q-Former and projection stayed fp16 `Linear`, as intended. The "VRAM used by others before load = 0.97 GiB" in that file was measured *after* CUDA initialization, so it includes this process's own CUDA context; the measurement now uses `nvidia-smi` before CUDA init.

## Decisions / next
- Headline metric for probes becomes **AUROC** (threshold-free) plus accuracy at **per-template thresholds fit on dev and frozen**; raw threshold-0 accuracy is reported but flagged. Test runs now refuse to start without `probe.thresholds_file`.
- Pin the model revision before any test run (see README).
- Scale the test run to ~1,000 images (≈ 1.8 h at the measured rate) once thresholds are frozen.
- Audit ~100 top false positives by eye before using the word "hallucination" for them.
