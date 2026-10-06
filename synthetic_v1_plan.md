# Counterfactual experiment 1 — synthetic scenes (written BEFORE any model was run on it)

**Question.** When the visual evidence for a concept is edited, does the model's answer about that concept change appropriately, and does it stay put when something irrelevant is edited?

**Design.** 3–4 coloured shapes (unique colours) on a plain background; answers computed from the scene spec (exact). Edits: remove, recolor, reshape, swap positions (relation), insert, background change. Every relevant edit gets a control edit of the same type on a different object (area-matched where possible; the fraction within ±20% is reported), and binding foils (e.g. "red square" when only a red circle and a blue square exist) probe attribute binding. Closed yes/no probes, 3 paraphrases, thresholds fit on **dev** and frozen for **test**.

**Metrics** (probe level, scene-clustered bootstrap CIs): AFR = P(answer flips | gold changes); SFR = P(flips | gold unchanged: control edits, background edits, foils); CSS = AFR − SFR; CPA = P(correct on both images); persistence rate = P(still "yes" after the evidence for a "yes" was removed/changed); non-detection rate; shift-AUROC = threshold-free separability of |Δ log-odds| between changed and unchanged probes.

**Preconditions (checked first, otherwise the sensitivity numbers are not interpreted).** (a) accuracy on *original* synthetic images clearly above the blind baseline — otherwise the model is incompetent in this domain and low AFR says nothing about grounding; (b) controls are pure: SFR of the colour-oracle test model is 0 (verified by unit tests); (c) the preview images were inspected by a human (`scripts/generate_counterfactuals.py` → `preview.png`).

**Decision rules (fixed now).**
- Per edit type, "no evidence that the model tracks this evidence" if the 95% CI of CSS includes 0.
- "Ungrounded persistence" is reported when persistence rate on removal/recolor/reshape exceeds 0.5 with the CI above 0.5.
- Differences between edit types or between paraphrases are described, not ranked, unless CIs are disjoint.
- Nothing here is evidence about natural images: this is a controlled but artificial domain (itself a distribution shift). A real-photo track (COCO masks + inpainting, with edit-artifact checks) follows.

**Known limits.** One object-color/shape vocabulary; 3–4 objects; yes/no probes only (no free generation yet); BLIP-2 is not instruction-tuned and sees 224-px resized images; fp16 vision tower, 4-bit language model.
