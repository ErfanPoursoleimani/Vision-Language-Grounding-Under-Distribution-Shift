# Experiment log — competence on ORIGINAL synthetic images, BLIP-2 OPT-2.7B (NF4 LM), dev, "small" objects

**Status: exploratory, dev only** (40 scenes, 3 paraphrases, thresholds fit on this dev set and pooled across groups, so **use AUROC; per-group accuracy at the pooled threshold is not meaningful**, e.g. shape atoms: 74% gold-yes but yes-ratio 0.29). 95% CIs resample scenes; no CI was computed for the blind AUROC. Numbers copied from `results.json`.

| group (gold yes vs no) | n units (yes) | model AUROC | blind AUROC |
|---|---|---|---|
| colour atoms ("Is there a red object?") | 240 (139) | 0.672 (0.606–0.738) | 0.501 |
| shape atoms ("Is there a circle?") | 120 (89) | 0.746 (0.671–0.819) | 0.428 |
| easy existence (present colour+shape vs absent colour) | 219 (139) | 0.659 (0.597–0.721) | 0.501 |
| **binding** (present combination vs foil) | 259 (139) | **0.506 (0.478–0.538)** | 0.514 |
| **relation** (A left of B vs reversed) | 160 (80) | **0.489 (0.454–0.524)** | 0.387 |

Relations: the model answered "yes" to **both** directions of the same pair in 99.6% of cases (overall yes-ratio 0.998).

## What this means
1. **The yes/no probes are only weakly competent even on the easiest facts.** Colour and shape presence are separable (AUROC 0.67 / 0.75) but far from reliable on saturated shapes on a plain background.
2. **No usable binding or relation signal at all** (AUROC ≈ 0.5, CI at chance). The pre-registered precondition (a) is therefore **not met** for binding and relations, only weakly for atoms.
3. **Consequences for the first counterfactual dev run.** Results for reshape and swap, and everything computed from binding foils, cannot be read as grounding (or its failure): the model gives no discriminating answer to these questions even on unedited images. The modest remove/insert/recolor sensitivity is consistent with weak colour detection, not with compositional use of the image.
4. **Correction to my earlier reading.** I wrote that the foil accuracy of 0.29 was "consistent with attribute recombination". The competence data say something more modest: the model's yes/no log-odds carry **no detectable information about colour–shape binding** (AUROC 0.51), and the below-chance accuracy comes from a yes-bias at the fitted thresholds. Recombination is not distinguishable from "cannot see the combination" here.

## Open question this log cannot answer
Is the weakness from (a) my scene design (objects only ≈ 33–49 px after downsizing to 224 px, 3–4 objects), (b) BLIP-2-OPT being non-instruction-tuned with a yes/no logit readout, or (c) BLIP-2's genuine limits? Test (a) first because it is cheap: the same competence task with `medium` (2–3 objects, 100–130 px) and `large` (2 objects, 150–190 px) scenes.

**Decision rule, fixed before running [proposed thresholds].** If colour-atom AUROC ≥ 0.85 at medium or large, adopt that style for the main counterfactual track and re-run the counterfactual dev task there. If it stays below 0.80 at all sizes, treat BLIP-2 + yes/no probing as not competent on this domain, keep the existing results as a documented weak-model case, and bring in a second (instruction-tuned) model family before making any further claims about evidence tracking.
