# Datasets and licensing

**All license entries below are from background knowledge and are UNVERIFIED.** Before downloading or redistributing anything (especially derived counterfactual images), read the license on the official dataset page and record the date checked in the last column.

| Dataset | Planned use | Notes / believed terms (verify) | Checked |
|---|---|---|---|
| MS-COCO (captions, instances, val/test splits) | Hallucination (CHAIR), POPE-style probes, source images + masks for counterfactual edits, adaptation training data (train split only) | Annotations believed CC BY 4.0; images carry Flickr per-image terms → **do not redistribute edited images publicly without checking** | ☐ |
| Visual Genome | Relations/attributes, region/box annotations for local edits | Believed CC BY 4.0; overlaps COCO images (take care with split leakage) | ☐ |
| VQAv2 | Capability-retention check after adaptation (non-inferiority) | Images from COCO; annotation terms to check | ☐ |
| GQA | Compositional questions; held-out compositions | Check site terms; built on Visual Genome imagery | ☐ |
| RefCOCO/+/g | Referring-expression grounding; mask/box for target objects | Built on COCO images | ☐ |
| Flickr30k | Natural distribution-shift test (different image source/caption style) | Research-use terms; Flickr image owners' licenses | ☐ |
| TextCaps | Shift toward text-in-image; stress language prior vs reading | Check annotation + image source terms | ☐ |
| POPE | Reuse the published polling protocol | Derived from COCO; check repository license; may rebuild from COCO annotations instead | ☐ |
| HallusionBench | External validity check; control-pair design reference | A third-party aggregator lists BSD-3-Clause for the code repo — confirm on the repo itself; image terms separate | ☐ |
| MMVP, SugarCrepe | Perception / compositional external checks | Check repo + data licenses | ☐ |
| Synthetic scenes (own renderer) | Exactly-controlled color/count/position/relation edits with no inpainting artifacts | Own data; license freely | n/a |

**Use mapping:** IID test → COCO held-out; compositional → held-out attribute–object combinations (synthetic + GQA/SugarCrepe); counterfactual → generated pairs (COCO+VG masks, synthetic); distribution shift → Flickr30k / TextCaps / style or domain shift; adversarial/perturbation → corruption ladders on the IID images. See `docs/evaluation_protocol.md` §6 for split rules.

## Getting COCO val2017 for the first baseline (manual download; not done by code)
Source: https://cocodataset.org/#download (I could not reach it from my sandbox; confirm file names and sizes there and read the terms of use first). You need the **2017 Val images** and the **2017 Train/Val annotations**. Unzip so the layout is:
```
data/coco/val2017/*.jpg
data/coco/annotations/instances_val2017.json
```
`data/` is git-ignored. Our own split of val2017 is by hashed image id (`split_seed`): 20% **dev** (prompt wording, threshold, debugging) and 80% **test** (final numbers only). COCO train2017 is disjoint from val2017 and is reserved for later LoRA adaptation.

**Probe set.** `src/datasets/pope_style.py` builds POPE-style existence probes (3 present + 3 random + 3 popular + 3 adversarial absent categories per image) from the instance annotations. It is *not* the official POPE file set; for numbers comparable with published POPE results, obtain the official files and add a loader. COCO annotations are incomplete, so some "absent" labels are wrong; failure-case labels are candidates, not verdicts.
