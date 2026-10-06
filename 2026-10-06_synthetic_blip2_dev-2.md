# Experiment log — synthetic counterfactuals, BLIP-2 OPT-2.7B (NF4 LM), **dev** split

**Status: exploratory, dev only.** 40 dev scenes, 419 pairs, 1,033 probes (unit = one question on one image pair, averaged over 3 paraphrases), thresholds fit on this same dev split (optimistic), 95% CIs resample scenes. Numbers are copied from the run's `results.json`. No test split touched.

## Results
| edit | AFR (flip when evidence changed) | SFR (flip on matched control) | **CSS** = AFR − SFR | CPA | shift-AUROC | mean shift toward correct vs mean \|shift\| on control |
|---|---|---|---|---|---|---|
| remove | 0.475 (0.337–0.600) | 0.215 (0.140–0.298) | **0.260 (0.098–0.408)** | 0.450 | 0.761 (0.654–0.854) | 0.194 vs 0.085 |
| insert | 0.367 (0.283–0.458) | 0.229 (0.164–0.300) | 0.137 (0.027–0.258) | 0.333 | 0.523 (0.409–0.636) | 0.074 vs 0.086 |
| recolor | 0.292 (0.213–0.371) | 0.174 (0.106–0.236) | 0.117 (0.026–0.213) | 0.271 | 0.617 (0.529–0.705) | 0.078 vs 0.065 |
| reshape | 0.292 (0.208–0.388) | 0.213 (0.154–0.288) | 0.079 (**−0.015**–0.184) | 0.179 | 0.544 (0.440–0.648) | 0.018 vs 0.080 |
| swap (left/right) | 0.000 | 0.000 | 0.000 | 0.000 | 0.514 (0.342–0.686) | −0.009 vs 0.037 |
| **overall** | 0.287 (0.248–0.327) | 0.200 (0.172–0.228) | 0.087 (0.051–0.126) | 0.240 | 0.560 (0.525–0.598) | |

Other: flips on *irrelevant* probes by source — control edits 0.196, **background-colour-only edits 0.196**, binding foils 0.232. Accuracy on original images 0.599 (blind gray-image baseline 0.481, which answers "no" ~90% of the time); accuracy on edited images 0.575; **accuracy on binding foils (original images) 0.291**. Only 55.6% of controls are within ±20% area of the edited object; SFR is 0.196 over all controls and 0.217 over the area-matched ones, so size mismatch does not explain the noise floor. Four failure images were inspected: recolor (orange→red square), remove (red circle), swap (two circles exchanged), reshape (blue circle→square); each edit is exactly what it claims, so those failures are model failures, not editing artifacts.

## What the data support (dev, descriptive)
1. **Small, edit-dependent sensitivity.** CSS is above 0 for remove, insert and recolor, strongest for remove; for reshape the interval includes 0 (by the decision rule fixed in the plan: no evidence of tracking); relation questions get the same answer on every image (AFR = SFR = 0), so left/right sensitivity is **not measurable** here (a floor effect, not evidence about grounding).
2. **A large noise floor.** About 20% of answers flip when only the background colour changes. Roughly 70% (0.200/0.287) of the flips seen on probes whose answer *should* change are no more frequent than flips on irrelevant edits. Log-odds are compressed (shifts mostly within ±0.3), so answers near the threshold flip easily.
3. **Flip-based and shift-based views disagree for insert.** CSS says detectable (0.137), but the continuous shift barely separates changed from control probes (AUROC 0.52, CI includes 0.5; the signed shift toward the correct answer is not larger than the control's absolute shift). Part of the insert "sensitivity" is threshold crossing. Only **removal** is supported by both views.
4. **Binding is poor.** Below-chance accuracy on foils means the model usually answers "yes" to a colour+shape combination whose colour and shape both appear but not together. This is consistent with attribute recombination, but it cannot be separated from simple inability to see these shapes without the competence check below.

## Caveats and deviations (stated, not hidden)
- **Metric definition fix.** `persistence_rate` / `non_detection_rate` in this results.json are *unconditional* (P(yes on edited image | evidence removed), etc.), not the conditional quantities defined in the plan, and carry no CIs. The code now computes the planned conditional versions with CIs (`persistence_rate`, `non_detection_rate`) and keeps the unconditional ones under new names (`false_yes_after_edit`, `miss_after_edit`). Recompute without re-running the model: `scripts/reaggregate_counterfactual.py --run-dir <run>`. Until then, don't quote the old persistence numbers (reshape 0.65, recolor 0.48, remove 0.32, swap 1.00).
- **Precondition (a) of the plan is only weakly met** (0.60 vs 0.48). Therefore these sensitivity numbers are descriptive and must not be read as "the model ignores the image" until perception is checked. The `synthetic_competence` task was added **after seeing these results** (forking path: disclose in any write-up); it splits the question into colour-only, shape-only, binding, easy existence and relation items on original images.
- Thresholds were fit on the same dev data; n = 40 scenes; 5 edit types × several metrics with no multiplicity correction (only the remove CSS has a comfortable margin); one model, one quantization; synthetic cartoon shapes are themselves a distribution shift.

## Next
Run the competence dev task, re-aggregate, then decide: if colour/shape atoms are answered well but binding is not, the finding is about binding/hallucination; if atoms are poor too, the domain/model is too weak and a clearer scene (larger objects, higher resolution) and a second, stronger model family are needed before any claim.
