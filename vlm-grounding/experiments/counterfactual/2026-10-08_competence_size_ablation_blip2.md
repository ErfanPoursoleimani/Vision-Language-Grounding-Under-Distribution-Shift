# Experiment log — does object size explain BLIP-2's weak competence? (small vs medium vs large scenes)

**Status: exploratory, dev only** (40 scenes each, 3 paraphrases, thresholds fit on dev; use AUROC, not accuracy). Files identified by unit counts (large: 2 objects → 200 colour-atom units; medium: 2–3 objects → 219). 95% CIs resample scenes. Blind AUROC for single groups varies with how often each colour/shape happens to be present in the 40 scenes (the blind answer depends only on the question), so differences of ±0.05–0.07 from 0.5 are sampling noise.

| group | small (56–84 px, 3–4 obj) | medium (100–130 px, 2–3 obj) | large (150–190 px, 2 obj) |
|---|---|---|---|
| colour atoms | 0.672 (0.606–0.738) | 0.792 (0.737–0.844) | 0.840 (0.790–0.894) |
| shape atoms | 0.746 (0.671–0.819) | 0.742 (0.701–0.795) | 0.756 (0.713–0.817) |
| easy existence | 0.659 (0.597–0.721) | 0.778 (0.718–0.839) | 0.838 (0.785–0.886) |
| **binding** | 0.506 (0.478–0.538) | 0.502 (0.480–0.522) | 0.512 (0.500–0.526) |
| **relation** | 0.489 (0.454–0.524) | 0.457 (0.395–0.520) | 0.489 (0.422–0.561) |
| yes to both directions of a relation | 99.6% | 100% | 94.8% |

## Findings
1. **Object size matters for colour and for existence-level questions** (colour 0.67 → 0.79 → 0.84; easy existence 0.66 → 0.78 → 0.84), so my original small-object scenes did make the task harder than necessary.
2. **It does not matter for shape, binding or relations.** Shape atoms stay ≈ 0.75; binding stays at chance with tight intervals at every size (large: 0.500–0.526); relation answers are "yes" to both directions 95–100% of the time. These failures are therefore **not an artifact of my scene design**: BLIP-2-OPT's yes/no readout carries no detectable colour–shape binding or left/right information in this setting.
3. **The pre-stated decision rule did not trigger either branch** (colour AUROC ≥ 0.85 at medium/large → adopt; < 0.80 at all sizes → not competent): medium is 0.792 and large 0.840, both with intervals that straddle the cut-offs. The rule had a gap between 0.80 and 0.85; I am not moving the thresholds after the fact. What follows is a **new, post-hoc decision**, labelled as such.

## Decision (post-hoc, made after seeing these data)
- Binding/relation conclusions above do not depend on the colour cut-off, so they stand: with BLIP-2 the compositional counterfactual questions (reshape, swap, binding foils) remain uninterpretable at every size.
- The first counterfactual dev run (small scenes) stays documented as a weak-model result. Existence/colour edits at `large` size could be re-run for BLIP-2 later as a limited study.
- The main compositional/counterfactual track needs a **second, instruction-tuned model family**. First candidate: Qwen3-VL-4B-Instruct via the generic `hf_chat_vlm` backend (written, not yet run).
- **Interpretation rule for every model going forward [proposed threshold]:** a counterfactual question group may only be interpreted for a model if its competence AUROC on original images has a 95% lower bound ≥ 0.65 for the matching group (colour/shape atoms for existence edits, binding for reshape/foils, relation for swap). Otherwise the group is reported as "not measurable for this model".
