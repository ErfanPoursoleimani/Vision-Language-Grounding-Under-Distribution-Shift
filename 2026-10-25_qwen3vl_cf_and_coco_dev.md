# Experiment log — Qwen3-VL-4B-Instruct (4-bit LM): synthetic counterfactuals and COCO existence probing, dev

**Status: exploratory, dev only; test split untouched.** Both files were identified from their numbers and the config names (`cf_synthetic_qwen3vl_dev`, `pope_style_qwen3vl_dev`); no manifest was uploaded, so the model identity and `runtime.vision_quantized` are **unconfirmed**. Thresholds were fit on the same dev data (optimistic; irrelevant for the synthetic run where the log-odds separate by ~45 units).

## 1. Synthetic counterfactuals (40 dev scenes, small objects, 3 paraphrases, 1,033 probes)
| edit | AFR | SFR (matched control) | CSS | CPA | shift-AUROC |
|---|---|---|---|---|---|
| remove | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| insert | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| recolor | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| reshape | 1.000 | 0.003 | 0.997 (0.990–1.000) | 1.000 | 1.000 |
| swap (left/right) | 0.958 (0.917–0.992) | 0.053 (0.000–0.158, only 19 control probes) | 0.906 (0.792–0.983) | 0.958 | 1.000 |
| **overall** | **0.994** (0.988–0.999) | **0.002** (0.000–0.005) | **0.992** (0.985–0.997) | 0.994 | 0.9998 |

Mean log-odds shift toward the correct answer on changed probes ≈ 45 (swap 26); mean |shift| on control probes 0.7–1.4. Flips on irrelevant probes: control edits 0.003 (n = 409), background-colour edits 0.000 (n = 179), binding foils 0.000 (n = 207). Accuracy on original images 1.000, on edited images 0.997, on binding foils 1.000; text-only blind accuracy 0.468.

**Prediction 1 (written before the run): confirmed.** AFR ≥ 0.9, SFR ≤ 0.1, CSS ≥ 0.8 for every edit type (swap's CI lower bound, 0.792, is just under 0.8; its point estimate passes). Compare BLIP-2 on the same scenes: overall AFR 0.29, SFR 0.20, CSS 0.09.

**What it means.** On clean synthetic scenes this model's answers follow the evidence essentially perfectly, and are not disturbed by irrelevant edits. The domain is **saturated**: it cannot rank conditions or reveal failure modes for this model. The informative experiments are those that remove the easy signal (degradations, natural photographs, real-photo counterfactuals). Swap (relations) is the only non-perfect edit and rests on a small control set.

## 2. COCO existence probing (200 dev images, 2,400 items, 3 paraphrases, per-template thresholds fit on dev)
| negatives | accuracy (95% CI) | AUROC (mean of 3 templates) | BLIP-2 AUROC | text-only accuracy | text-only AUROC | text-only yes-ratio |
|---|---|---|---|---|---|---|
| random | 0.952 (0.940–0.963) | 0.994 | 0.817 | 0.624 | 0.685 | 0.587 |
| popular | 0.906 (0.891–0.922) | 0.971 | 0.779 | 0.400 | 0.233 | 0.811 |
| adversarial | 0.894 (0.879–0.910) | 0.961 | 0.756 | 0.413 | 0.247 | 0.798 |

Recall on present objects 0.91–0.92 (shared positives). False positives per template (of 600 negatives): random 6–8, popular 53–69, adversarial 66–83. AUROC spread across the three paraphrases ≤ 0.004; accuracy spread at the tuned thresholds ≤ 0.009. Dev-fit thresholds on the log-odds scale: −1.6, −3.2, −1.9 (log-odds range roughly ±25).

**Prediction 3: confirmed** for AUROC (0.96–0.99 vs 0.76–0.82) and for stability across wording at tuned thresholds. The comparison "accuracy at raw threshold 0" with BLIP-2's wording sensitivity is **not yet possible**: it needs `results_raw_threshold.json` of this run.

**Observations (descriptive).**
1. The text-only baseline is now a genuine language-prior measurement (unlike BLIP-2's gray image). It answers "yes" to 81% and 80% of the popular and adversarial questions and ranks those absent objects *above* the present ones (AUROC 0.23–0.25): the frequency/co-occurrence prior points the wrong way, which is what makes these negatives hard. With the image the model overcomes it (AUROC 0.96–0.97); model minus text-only accuracy is +0.33, +0.51, +0.48.
2. Hallucination-candidate counts rise monotonically random < popular < adversarial, as for BLIP-2.
3. **Label noise now dominates the interpretation.** With AUROC ≈ 0.97 a large share of the remaining "false positives" may be objects COCO did not annotate. No error here should be called a hallucination before the audit below.

## Decisions / next
- New tools: `scripts/audit_failures.py` (contact sheets of the most confident false positives + a verdict CSV; verdicts: hallucination / present_unannotated / ambiguous) and `scripts/analyze_h1.py` (probe-level test of H1: among 'yes' answers, does image information gain, i.e. log-odds with the image minus text-only log-odds, separate grounded from hallucinated better than plain confidence? F1–F3 rules from the protocol; within-category AUROC controls object frequency; the audit relabels unannotated-but-present items). **Order: audit first, then H1 (with `--audit`).** The H1 numbers on this dev run are exploratory; the formal test needs ≥ 2 model families and the test split.
- Still pending: degradation ladder (prediction 2 untouched), `pope_perturbed` for Qwen, manifest check (`vision_quantized`), pinned revisions.
