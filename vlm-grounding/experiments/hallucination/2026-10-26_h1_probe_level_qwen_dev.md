# Experiment log — probe-level test of H1, Qwen3-VL-4B-Instruct on COCO dev (run `pope_style_qwen3vl_dev_20261026-122729`)

**Status: exploratory, dev only, one model. The numbers below were computed WITHOUT the human audit taking effect** (see "Tool bug"). They must be recomputed with the fixed tools before they are used for anything.

## Tool bug (disclosed)
The audit CSV contained `yes` (20) and `no` (11) verdicts, which answer the question printed on the sheets ("is there a cup?"). `audit_failures.py summarize` rejected them, and `analyze_h1.py` silently **ignored** unknown verdicts, so the H1 output reported `audit_applied` although no item was relabeled. Fixed: yes/no are now accepted (yes = object visible → label corrected to present; no = not visible → stays a hallucination candidate; `?` = unclear → dropped), unknown verdicts raise an error, and the H1 output includes an audit summary. My instruction also used a literal placeholder (`<same folder>`) that was typed verbatim.

## Population and results (items the model answered "yes" to; 1,932 unique (image, object) items after removing duplicates across negative kinds)
| quantity | value (95% CI over images) |
|---|---|
| mentions | 635 = 550 labeled present + 85 labeled absent (candidate hallucinations, 13%) |
| AUROC, confidence (log-odds with the image) | 0.906 (0.877–0.934) |
| AUROC, **PMI** = log-odds(image) − log-odds(text-only) | **0.934 (0.910–0.953)** |
| AUROC, −prior (text-only log-odds alone) | 0.807 (0.748–0.859) |
| PMI − confidence | **+0.0275 (0.0143–0.0384)** |
| within-category AUROC, PMI and confidence | 0.868 (0.795–0.919) — identical |
| all 1,932 items: confidence / PMI / difference | 0.979 / 0.982 / +0.003 (0.0016–0.0044) |
Rules as written [proposed]: F1 survives, F2 survives, F3 survives for mentions (est ≥ 0.02, lower bound > 0) and is falsified for the all-items population.

## Reading (what the data support, and what they do not)
1. **The PMI advantage is a category-prior correction, not instance-level visual dependence.** For a fixed object category and paraphrase the text-only log-odds is a constant, so within a category PMI and confidence rank items identically (hence the identical within-category numbers). PMI − confidence is therefore a function of the category alone: PMI is "confidence corrected for how strongly the language prior alone says yes to this category". The +0.03 gain comes entirely from between-category differences.
2. **That between-category signal is partly built into the benchmark.** Popular and adversarial negatives are chosen by frequency / co-occurrence, and the text-only prior says "yes" to those categories (80–81%, previous log). So −prior alone separates hallucination candidates from grounded answers with AUROC 0.81 among mentions, and it drops to 0.56 over all items. Controlling for category (F2) lowers PMI's AUROC from 0.934 to 0.868.
3. **Consequence for the hypothesis.** In this probe setting "hallucination = low visual information gain" cannot be separated from "hallucination = the category has a strong language prior". Testing *instance-level* visual dependence needs an intervention on the specific object (mask or remove it and see whether the answer survives): the planned local-evidence test (H1b) on real photographs. H1 is **not falsified** on dev, but calling it supported would be an overclaim.
4. **Sample size.** 85 candidates, of which an unknown (probably large) share are annotation gaps; the formal test needs more images (the test split, ≥ 1,000 images), a second model family, and audited labels.

## Audit status
31 of the rows were judged (20 yes, 11 no; the rest blank or not reported), taken from the **most confident** false positives only. If `yes` means "visible", ~65% of these cases are objects COCO did not annotate, **but this sample is biased**: the most confident false positives are the ones most likely to be annotation gaps, so 65% must not be extrapolated to all false positives. A stratified (equal thirds by confidence) or random sample is needed for an unbiased estimate (`--select stratified|random`, default stratified).

## Next
1. Update `src/evaluation/h1.py`, `scripts/audit_failures.py`, `scripts/analyze_h1.py` (zip) and rerun `summarize` and `analyze_h1` on the existing audit file (please confirm `yes` meant "the object is visible").
2. New unbiased audit: `python scripts/audit_failures.py sheet --run-dir results/<run> --coco-root data/coco --n 45 --select stratified --out audit_strat`.
3. Build the real-photo counterfactual track (object masking with COCO instance masks and area-matched control masks) to test instance-level dependence (H1b).

---
# Update — audited results (same run; tools fixed; both audits applied separately)

## The audits
| audit | rows judged | object visible (COCO label wrong) | not visible (hallucination) | share visible, Wilson 95% |
|---|---|---|---|---|
| top-confidence false positives (biased) | 31 | 20 | 11 | 0.645 (0.469–0.789) |
| **stratified by confidence (equal thirds)** | 45 | 15 | 30 | **0.333 (0.214–0.479)** |
There are only **85 unique false-positive candidates** among the 635 "yes" answers, so the stratified audit already covers 53% of them. The two shares differ by a factor of two: the model's most confident false positives are mostly annotation gaps, the rest mostly genuine errors. Extrapolating the stratified share: about 57 true hallucinations among the 85 candidates (44–67); precision of "yes" answers ≈ 0.91 instead of the naive 0.87, i.e. measured accuracy on this benchmark understates the model by a few points. (Interval ignores the 53% sampling fraction, so it is conservative.)

## H1 recomputed (mentions only: items answered "yes")
| labels used | grounded / candidates | AUROC confidence | AUROC PMI | PMI − confidence (95% CI) | F3 (≥ 0.02 and lower bound > 0) |
|---|---|---|---|---|---|
| unaudited | 550 / 85 | 0.906 | 0.934 | +0.0275 (0.0143–0.0384) | met |
| top-confidence audit applied | 570 / 65 | 0.931 (0.907–0.951) | 0.950 (0.930–0.966) | +0.0192 (0.0062–0.0324) | **not met** |
| stratified audit applied | 565 / 70 | 0.916 (0.888–0.940) | 0.935 (0.912–0.956) | +0.0197 (0.0063–0.0319) | **not met** |
Within-category AUROC (PMI = confidence by construction): 0.913 and 0.879 after the two audits; text-only prior alone (−prior): 0.788 and 0.777; all-items PMI − confidence: +0.002 in both. F1 and F2 survive in every version.

## Reading
1. After auditing, PMI beats plain confidence by about **+0.02 AUROC**, with a lower bound above 0: a small but detectable gain. The rule fixed in advance (point estimate ≥ 0.02) is **marginally not met** (0.0192, 0.0197); I am not moving the threshold. The three estimates (0.0275, 0.0192, 0.0197) lie within each other's intervals, so the pass/fail flip is a threshold crossing, not a large change.
2. As established earlier, any PMI advantage is a **category-level** language-prior correction. Together: on audited dev data and one model, "visual information gain" is at most a small refinement of confidence. **H1 as instantiated here is not supported as a better-than-confidence detector.** F1/F2 only show that confidence-like scores separate grounded from candidate answers.
3. Consequence for the project's direction: the contribution should not rest on a PMI detector. The stronger pieces are the controlled counterfactual evaluation with matched controls, the audited-label methodology, and the **instance-level intervention** on real photographs (H1b), which this probe setting cannot test.
4. Caveats unchanged: one model, dev split, 70 candidate errors with partially audited labels (rows of the two audits overlap by an unknown amount; the merged analysis reports items judged differently as a consistency check), COCO annotation gaps also affect items the model answered "no" (not corrected here).

## Next
1. Finish the census (every one of the 85 candidates judged once): `python scripts/audit_failures.py sheet --run-dir results/<run> --coco-root data/coco --select all --exclude audit/audit_template.csv audit_strat/audit_template.csv --out audit_rest`; fill `verdict`; then `python scripts/analyze_h1.py --run-dir results/<run> --audit audit/audit_template.csv audit_strat/audit_template.csv audit_rest/audit_template.csv` (reports how many items were judged differently across files).
2. Build the real-photo masking track for instance-level dependence (H1b).
3. Pending runs: degradation ladder (prediction 2), `pope_perturbed_qwen3vl_dev`, manifest/`vision_quantized`, pinned revisions.
