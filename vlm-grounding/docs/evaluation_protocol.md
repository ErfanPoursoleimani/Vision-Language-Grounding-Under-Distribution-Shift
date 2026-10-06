# Evaluation Protocol (v0.1 — DRAFT, not yet frozen)

Thresholds marked **[proposed]** must be fixed and committed *before* any test-split run. After freezing, changes go in the change log (§12) with a reason; they never silently overwrite.

## 1. Scope
How strongly do open VLM outputs depend on the visual evidence they claim to describe? We measure (a) correctness, (b) *response to evidence change* (sensitivity) and *non-response to irrelevant change* (invariance), (c) behaviour under perturbation and shift. Models are public checkpoints, evaluated at inference; adaptation is LoRA/PEFT only.

## 2. Notation
Model `f`, image `I`, prompt `q`. `I^{-c}` = `I` with concept `c` edited out (or changed); `I^{~c}` = `I` edited by the *same pipeline* on a concept *unrelated* to `q` (control). `I∅` = blank image (language-prior baseline); for models that cannot run without pixels this is a neutral gray image, which is an approximation of "no image" (the LM still receives visual-token embeddings of a blank image) and is reported as such. Answer `a_c` = the answer supporting `c` (e.g. "Yes" to "Is there a c?").

Three output modes, always reported separately: **probe** (closed yes/no, count, multiple choice), **likelihood** (`log p(a|I,q)`), **generative** (free caption; mention extraction).

## 3. Hypotheses and falsification criteria
- **H1 (detection).** Hallucinated object mentions have lower visual information gain than grounded ones: `S_global(c) = log p(a_c|I,q) − log p(a_c|I∅,q)`.
  *Falsified if* (F1) AUROC of `S_global` for separating hallucinated vs grounded mentions has a 95% CI lower bound ≤ 0.5 on ≥ 2 model families; or (F2) after controlling for object frequency/co-occurrence and caption length (partial AUROC / logistic regression), the effect vanishes; or (F3) it does not beat token log-prob and predictive-entropy baselines by ≥ 0.02 AUROC **[proposed]**.
- **H1b (local evidence).** For grounded objects, removing the evidence lowers support: `S_local(c) = log p(a_c|I,q) − log p(a_c|I^{-c},q)` exceeds the same quantity under control edits. *Falsified if* the paired difference to `I^{~c}` has CI including 0 (if so, the apparent effect is inpainting artifact or generic instability).
- **H2 (mitigation).** LoRA with `L_total = L_language + λ·L_grounding` lowers CHAIR_s/CHAIR_i and raises POPE-adversarial F1 relative to ordinary LoRA instruction tuning on the same images and steps, without capability loss. *Falsified if*: gains are within seed variability (≥ 3 seeds; paired bootstrap CI includes 0), or ordinary tuning matches it, or capability (VQAv2-subset accuracy / caption quality) drops by more than a margin **[proposed: 1.0 absolute point]**.
- **H3 (mechanism, exploratory — no direction preregistered).** At which layers is image-dependent information present in/needed by the residual stream? Reported as exploratory; never used to support H1/H2.

## 4. Counterfactual pair construction
**Edit types:** removal, insertion, replacement, color change, spatial rearrangement, crop, blur, masking, background replacement.
**Sources:** (A) *Synthetic renderer* (`src/perturbations/synthetic.py`, implemented) — exact ground truth, no editing artifacts; primary source for color/shape/position/relation. Unique colour per object so an independent pixel-based verifier can check every edit; controls are made pure by dropping foil/other probes that mention an edited object's colour; the fraction of controls with area within ±20% of the edited object is reported (about 40–50% in the first check), with a matched-only SFR as a robustness view. Probe-level definitions: AFR/SFR/CSS/CPA/persistence/shift-AUROC (see `experiments/counterfactual/synthetic_v1_plan.md`). Count edits are not implemented yet. (B) *Natural images* — instance masks/boxes from COCO/VG; removal via inpainting, recolor via masked hue shift, insertion/rearrangement via copy-paste, background via mask compositing. Specific inpainting/segmentation tools are chosen and verified in phase 5.
**Pair record:** `{pair_id, source_id, edit_type, concept, mask_area_frac, gold_orig, gold_cf, control_id}`.
**Validity checks (all reported):**
1. Automatic: an open-vocabulary detector *not among the evaluated models* no longer finds the removed concept (and still finds others).
2. Human audit: random sample (target N=200 per edit type) rated for "edit succeeded / no other change / plausible"; report rates; drop failures.
3. **Controls:** every relevant pair gets a control edit by the same pipeline on an unrelated object with matched mask area (±20% **[proposed]**).
4. **Artifact detectability:** train a small classifier to separate edited from unedited images; if accuracy is high, a model could exploit edit traces — report it and weight conclusions accordingly.
5. Prompt robustness: ≥ 3 paraphrases per probe; report mean ± spread.

## 5. Metrics
| ID | Name | Definition |
|---|---|---|
| M1 | Probe AUROC, calibrated accuracy, yes-ratio | POPE-style existence probes on originals and edits. **Headline = AUROC** (threshold-free) and accuracy at per-template thresholds fit on dev and frozen before any test run; raw threshold-0 accuracy is reported but flagged, because paraphrases shift the yes-bias (observed on dev: same AUROC, accuracy 0.56–0.70). Yes-ratio exposes answer bias. |
| M2 | Counterfactual pair accuracy (CPA) | P(correct on `I` **and** on `I^{-c}`). Control-pair idea follows HallusionBench. |
| M3 | Sensitivity–invariance | AFR = P(answer changes ∣ relevant edit); SFR = P(answer changes ∣ control edit); **CSS = AFR − SFR**. Report with M2 because AFR counts any flip. |
| M4 | Likelihood sensitivity | `S_global`, `S_local` (§3) with bootstrap CIs. Cheap, deterministic, no generation needed. |
| M5 | Generative grounding | CHAIR_s, CHAIR_i (unique-object variant, see `src/metrics/hallucination.py`); **residual mention rate** RMR = P(c still mentioned for `I^{-c}` ∣ mentioned for `I`). Mention extraction uses a synonym lexicon with crude negation handling → manual audit (N=100) reports its precision/recall. |
| M6 | Compositional accuracy | Forced-choice per category (color, count, position, relation, identity) **plus the blind (`I∅`/no-image) baseline**; only the margin over blind is interpreted. |
| M7 | Robustness | For each corruption × severity: accuracy, relative degradation, prediction *consistency* (P(unchanged) when gold unchanged), and yes-ratio drift. |
| M8 | Layer-wise (exploratory) | Interventions: ablate/mean-replace image-token hidden states from layer ℓ onward → output change; linear probes for object presence per layer **with control tasks** (shuffled labels) to report selectivity. Attention mass on image tokens is descriptive only. |

*Justification of the metric family:* object-level existence from CHAIR/POPE; control pairs from HallusionBench; contrast between visual conditions from VCD; language-prior framing from CHAIR's and HallusionBench's findings. The exact functional forms of CSS and `S_global/S_local` are **proposals to be validated** (stability across prompts, agreement with human judgments, behaviour on the synthetic set where truth is exact), not established metrics.

## 6. Splits and leakage rules
| Split | Content |
|---|---|
| IID test | COCO held-out images (never used for adaptation or tuning). IID is relative to *adaptation* data; base models may have seen COCO-derived data → check each model card and report. |
| Compositional | Held-out attribute–object / relation combinations (synthetic + GQA/SugarCrepe-style). |
| Counterfactual | Pairs from §4 built from held-out images only. |
| Distribution shift | Flickr30k, TextCaps, and synthetic domain shifts (style, background). |
| Adversarial/perturbation | Corruption ladders on IID images (blur, noise, crop, occlusion, background). |

Rules: splits are by **image id** (and by source image for all derived edits); adaptation uses COCO train only; all tuning (λ, margin, rank, steps) on a validation split disjoint from every test split; each test split is run once per frozen configuration and every test run is logged (`results/**/manifest.json`); prompt wording is fixed on validation.

**Blind-baseline caveat (added after the first dev run).** A gray image measures the response to *absence of evidence*, not necessarily the language prior (BLIP-2 answered "no" to ~92% of gray-image questions). Scene-conditioned priors need a different control: same scene with the target object edited out (phase 5), and/or a mismatched-image control. Failure-case labels such as `language_prior_candidate` are therefore weak evidence.

**Overlapping negatives.** Popular and adversarial negatives overlap (457 of 1,332 unique negative pairs appeared in more than one kind on dev); compare kinds with paired statistics on the shared items, not as independent samples.

## 7. Baselines
No adaptation; blind/no-image; ordinary LoRA instruction tuning (matched images, steps, rank); VCD (training-free); OPERA if hardware/code compatibility allows; at least one training-based mitigation (e.g. preference-style) if feasible. Candidate `L_grounding` forms to *compare, not assume*: (a) margin on likelihood drop, (b) unlikelihood on the original answer under `I^{-c}`, (c) preference pairs (chosen = image-consistent, rejected = counterfactual-consistent), each with an **invariance term** on control edits to prevent learning edit-artifact shortcuts.

## 8. Statistics
Greedy decoding for headline numbers (sampling analyses use ≥ 5 seeds). Percentile bootstrap CIs resampling by source image; paired comparisons by McNemar (exact) for binary outcomes and paired bootstrap for continuous; Holm correction over the preregistered comparison list; ≥ 3 training seeds; report effect sizes, not only p-values. Never select a configuration by test performance.

## 9. Logging and reproducibility
Every run writes `manifest.json` (config, model id + pinned revision, quantization, git commit, versions) and per-example JSONL of the **actual model outputs** (text, logprobs, prompt, image hash). Outputs are cached by `hash(model metadata, image pixels, prompt, params)`. Models flagged `testing_only` cannot write into `results/`. No output is ever typed in by hand.

## 10. Hardware plan (RTX 3050 6 GB / 24 GB RAM — the only GPU available)
Details and VRAM budgets: `docs/hardware_plan.md`. Inference-first; 4/8-bit quantization; prefer ≤ ~4B-parameter VLMs locally; **peak memory for each model/quantization is measured and recorded, not assumed** (7B at 4-bit is borderline). LoRA locally only for the smallest models; larger runs via a configurable external GPU. Attention/hidden-state extraction needs eager attention and is restricted to short outputs and sampled subsets.

## 11. Threats to validity
Editing artifacts as shortcut; synonym-lexicon extraction error; detector bias in validity checks; pretraining contamination of "IID"; prompt sensitivity; quantization changing behaviour (evaluate a quantized-vs-fp16 check on a subset); small-N pair counts for rare edit types; metric proposals that correlate with simple baselines (F3).

## 12. Change log
- v0.1 (2026-10-03): initial draft; nothing frozen.
