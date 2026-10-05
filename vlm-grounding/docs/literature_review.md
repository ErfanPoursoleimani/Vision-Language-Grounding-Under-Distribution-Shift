# Literature Review — Vision-Language Grounding Under Distribution Shift

**Status:** Phase 0 draft, 2026-10-03. Coverage is deliberately partial; read §0 first.

## 0. Verification policy and coverage (read this first)

- ✅ **Verified** — title, authors, venue, year (and pages/DOI where shown) were checked against a proceedings, publisher or arXiv page located by web search in this session. Statements about these papers are limited to what their abstracts/pages say; I did not read the full texts.
- ⚠️ **Unverified** — cited from background knowledge only. Do not cite in the paper until `scripts/verify_references.py` (run locally) and a manual venue/year check have passed. Status is tracked per entry in `docs/references.bib`.
- **Coverage gap:** the verified set is 2018–2024 and mostly CVPR/EMNLP/NeurIPS/NAACL. I have **not** yet swept ICCV/ECCV/ICLR/ACL/TMLR/TPAMI or 2025–2026 work, and "modern open VLMs" in October 2026 are very likely newer than the families I know. §9 lists the searches still to run. Anything described below as "the gap" is a hypothesis about the literature until that sweep is done.

## 1. Foundations (all ⚠️ unverified — to be checked)

| Family | Role in this project |
|---|---|
| CLIP, ALIGN (contrastive dual encoders) | Baselines for image–text matching/compositional tests; CLIP-style encoders are the vision tower of many LVLMs (the MMVP paper below argues this matters). |
| BLIP, BLIP-2 | Bridged architectures (frozen encoders + lightweight connector + LM). BLIP-2 with a small LM is a realistic 6 GB candidate. |
| Flamingo | Cross-attention-based interleaved VLM; reference design for "where does visual information enter the LM" questions. Weights are not generally open — treat as background. |
| LLaVA, LLaVA-1.5, InstructBLIP | Visual instruction tuning; the standard open recipe whose instruction data is a suspected source of object-frequency priors (see POPE below). |
| Newer open VLMs | **Not yet surveyed.** Model choice must be made from a fresh search (size ≤ ~4B at 4/8-bit for local LoRA; larger only on external GPU). Check license per model card. |

## 2. Hallucination: definitions, measurement, causes, mitigation

**Measurement**
- ✅ **CHAIR** (Rohrbach et al., EMNLP 2018) scores captions against ground-truth object labels to quantify object hallucination. Reported findings: models that score best on standard sentence-similarity metrics do not necessarily hallucinate less, and models that hallucinate more tend to make errors driven by language priors.
- ✅ **POPE** (Li et al., EMNLP 2023) turns the question into polling yes/no existence queries. Its authors report that LVLMs hallucinate heavily and that objects frequent in the visual instruction data, or co-occurring with objects in the image, are especially prone to being hallucinated.
- ✅ **HallusionBench** (Guan et al., CVPR 2024): 346 images / 1,129 questions with a *control-pair* structure (question variants and edited images with different expected answers), used to separate *language hallucination* (conclusions drawn without using the image) from *visual illusion* (misperceived visual input). At publication, the best model reached 31.42% question-pair accuracy and the other 14 were below 16%. **Design relevance:** the control-pair idea is the closest published precedent to the counterfactual pairs here.
- ⚠️ Others to verify and consider: AMBER, MMHal-Bench, M-HalDetect, FaithScore, and the Bai et al. hallucination survey.

**Mitigation**
- ✅ **VCD** (Leng et al., CVPR 2024): training-free; contrasts the next-token distribution under the original image with that under a *distorted* image, aiming to reduce reliance on statistical bias and unimodal priors. **Closest prior to H1/H2:** it already uses "what changes when vision changes" as a signal. It differs from this project in using generic distortion at decoding time rather than semantically targeted edits and a training objective.
- ✅ **OPERA** (Huang et al., CVPR 2024): decoding-time over-trust penalty on beam-search logits plus a rollback step, motivated by an observed attention pattern over "summary" tokens. Training-free; needs a specific modified `transformers` fork per its repository, so hardware/compatibility must be checked before it can serve as a baseline.
- ⚠️ Training-based: LURE, LLaVA-RLHF (factually augmented RLHF), RLHF-V, HA-DPO and related preference/instruction-data approaches. These are the right baselines for a *training* contribution and must be verified and read.

## 3. Visual blindness and compositionality

- ✅ **MMVP / "Eyes Wide Shut"** (Tong et al., CVPR 2024): identifies "CLIP-blind pairs" (images CLIP embeds as similar despite clear visual differences), builds MMVP from them, and reports that patterns hard for CLIP correlate with patterns hard for multimodal LLMs; proposes mixing in vision-only self-supervised features. Supports the hypothesis that some "hallucination" is perception failure in the encoder, not only language prior.
- ✅ **SugarCrepe** (Hsieh et al., NeurIPS 2023 D&B): earlier compositionality benchmarks had exploitable text-side biases — blind models with no image access beat state-of-the-art VLMs on them — and apparent gains from compositionality-inducing training were overestimated. **Methodological consequence adopted here:** every compositional test reports an image-ablated (blind) baseline, and a result only counts if it exceeds it.
- ⚠️ Winoground, ARO (and related CREPE/VALSE-type benchmarks) — verify and read.

## 4. Counterfactual and robustness evaluation

- ✅ HallusionBench (above) uses edited images/control groups for diagnosis.
- ⚠️ VQA v2 "complementary images" (Goyal et al.), VQA-CP (Agrawal et al.), counterfactual VQA (Niu et al.): precedent for minimally different inputs with different answers, and for language-prior analysis.
- ⚠️ ImageNet-C-style corruptions (Hendrycks & Dietterich) as the template for blur/noise/occlusion severity ladders; whether VLM-specific corruption benchmarks exist and are better is an open item for the sweep.

## 5. Interpretability: caution on attention

- ✅ **Jain & Wallace (NAACL 2019)** report, for NLP models, that attention weights are often uncorrelated with gradient-based importance and that very different attention distributions can give equivalent predictions; they conclude attention should not be treated as an explanation by default. This is an NLP result and transfers to VLMs only by analogy — which is why the protocol uses *interventions* (ablating image-token states, patch masking) and probes with control tasks, and reports attention maps only descriptively.
- ⚠️ Wiegreffe & Pinter, "Attention is not not Explanation" (a nuanced reply) — read before writing the interpretability section.
- ⚠️ Layer-wise visual information flow in LVLMs (e.g., work on image-token pruning/redundancy, cross-modal information flow, "middle layer" findings, representation editing for hallucination): the 2024–2026 literature is large and I have not verified any of it. It is the main reading task for §11 of the brief.

## 6. Lightweight adaptation

- ⚠️ LoRA, QLoRA: the adaptation toolkit for 6 GB hardware.
- ⚠️ Preference-style hallucination training (RLHF-V, LLaVA-RLHF, HA-DPO): relevant because a "chosen = image-consistent / rejected = counterfactual-consistent" pair construction is a natural fit for counterfactual data.

## 7. Synthesis — where this project could contribute (hypotheses, pending the sweep)

1. **Targeted vs generic visual contrast.** VCD-style methods contrast against generic distortions; a *semantically targeted* counterfactual (remove/replace the specific concept) with *matched irrelevant-edit controls* allows separating "answer follows evidence" from "answer is merely unstable".
2. **Two-sided criterion.** Sensitivity (change when evidence changes) must be reported together with invariance (no change when it does not), otherwise a noisy model looks "sensitive".
3. **Blind and artifact baselines.** SugarCrepe's lesson (text-only shortcuts) extends to edited images: inpainting artifacts can be a shortcut. Protocol adds artifact-detectability checks and same-pipeline controls.
4. **Meaning of "IID".** Many open VLM instruction mixtures include COCO-derived images and QA data (check each model card), so "IID" can only be defined relative to the *adaptation* data, not the base model's pretraining.

## 8. Novelty risk (stated up front)

A grounding-consistency training loss may already exist under another name (contrastive/counterfactual/unlikelihood objectives for multimodal models). Before any claim of novelty: search the backlog below, and if a close method exists, reposition the contribution as **evaluation** (the paired, controlled protocol) and a controlled **comparison** of objective variants. H1 and H2 are formulated as falsifiable either way.

## 9. Search backlog (not yet done)

- Venues/years: ICCV 2025, ECCV 2024, ICLR 2024–2026, NeurIPS 2024–2025, ACL/EMNLP/NAACL 2024–2026, TMLR and TPAMI 2024–2026, CVPR 2025–2026.
- Queries: "counterfactual image editing hallucination LVLM"; "visual contrastive training hallucination"; "unlikelihood grounding multimodal"; "object removal inpainting benchmark VLM"; "language prior vs visual evidence LVLM"; "layer-wise visual information LVLM"; "image token ablation hallucination"; "hallucination detection LVLM uncertainty/PMI"; "VLM robustness corruption benchmark"; "small open VLMs 1B–4B 2025/2026"; "compositional reasoning benchmark VLM 2025".
- For each hit: record venue/year, whether code and licenses exist, and whether it competes with H1/H2.
