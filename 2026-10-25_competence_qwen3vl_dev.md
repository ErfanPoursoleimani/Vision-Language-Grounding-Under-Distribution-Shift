# Experiment log — competence of the second model family: Qwen3-VL-4B-Instruct (4-bit language model), synthetic scenes, dev

**Status: exploratory, dev only.** Same 40 dev scenes and item sets as the BLIP-2 competence runs (so unit counts are identical: small 240/120/259/219/160, large 200/120/160/160/64), 3 paraphrases, thresholds fit on dev (irrelevant when classes separate perfectly; any threshold inside the gap gives the same accuracy). Blind = text-only (no image in the message). Files identified by the unit counts and the `fit_on` field of the thresholds files.

| group | BLIP-2 small | BLIP-2 large | **Qwen3-VL small** | **Qwen3-VL large** |
|---|---|---|---|---|
| colour atoms | 0.672 | 0.840 | **1.000** (1.000–1.000) | **1.000** (1.000–1.000) |
| shape atoms | 0.746 | 0.756 | **1.000** | **1.000** |
| easy existence | 0.659 | 0.838 | **1.000** | **1.000** |
| binding | 0.506 | 0.512 | **1.000** | **1.000** |
| relation | 0.489 | 0.489 | **0.9955** (0.9856–0.9998) | **1.000** |
| yes to both directions of a relation | 99.6% | 94.8% | 2.5% | 1.0% |
Qwen accuracy at the fitted thresholds: 1.000 everywhere except relation (0.971 small, 0.995 large). Text-only blind AUROC: 0.36–0.53 (chance-level scatter caused by which colours/shapes happen to be present in 40 scenes; the blind answer depends only on the question).

## What this shows
1. Under the rule fixed earlier (a group is interpretable only if its competence AUROC has a 95% lower bound ≥ 0.65 [proposed]), **every group is interpretable for Qwen3-VL** at both sizes (lowest lower bound 0.986).
2. The binding and relation failures of BLIP-2 were **properties of that model/readout, not of my scene design**: the same images and questions give 0.51/0.49 → 1.00/1.00 with an instruction-tuned model.
3. The synthetic domain is not the obstacle either: a 4B model with a fp16/bf16 vision tower reads it perfectly.

## Limits of this result
- **Ceiling.** AUROC 1.0 with a zero-width interval cannot distinguish "very good" from "perfect", and says nothing about where the model fails. The informative experiments are now the ones that make the task harder.
- Dev only, 40 scenes, one seed, one quantization (4-bit language model, bf16 vision). I have not yet seen this run's `manifest.json`: please confirm `runtime.vision_quantized` is `false` (the first smoke run had the vision tower quantized because of a skip-pattern bug, fixed afterwards).
- Clean, simple, saturated shapes on a plain background; nothing here speaks to natural photographs.

## Predictions fixed BEFORE the next runs (so they can fail)
1. **Qwen counterfactual dev run (`cf_synthetic_qwen3vl_dev`, small scenes):** AFR ≥ 0.9, SFR ≤ 0.1 and CSS ≥ 0.8 for remove, insert, recolor, reshape and swap. If true, sensitivity to clean edits is not the bottleneck on this domain, and the study shifts to where grounding breaks.
2. **Degradation ladder (`competence_degraded_qwen3vl_dev_small`):** relation and binding AUROC fall at lower severity than colour and shape atoms under noise, blur and pixelation (compositional information is more fragile than presence). Falsified if colour/shape fall first or all groups fall together.
3. **COCO existence probing (`pope_style_qwen3vl_dev`):** pooled AUROC clearly above BLIP-2's 0.78 (≈0.78–0.79 per template) with a much smaller dependence of accuracy on question wording.

## Next runs (in this order)
`cf_synthetic_qwen3vl_dev` → `competence_degraded_qwen3vl_dev_small` → `pope_style_qwen3vl_dev` → `pope_perturbed_qwen3vl_dev`; optionally `competence_degraded_blip2_dev_large` for the weak-model comparison at the easiest scene size.
