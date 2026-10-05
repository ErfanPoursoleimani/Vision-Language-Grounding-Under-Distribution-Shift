# Hardware plan — squeezing one RTX 3050 6 GB (+ 24 GB RAM) as hard as it will go

Only this GPU is available (no external GPU). Everything below is organised by *payoff*. Numbers marked **est.** come from `src/utils/memory_budget.py` (arithmetic, with stated assumptions) and are superseded by anything `scripts/profile_gpu.py` measures on the real machine. I could not run a GPU in the sandbox where this was written, so nothing here has been measured yet.

## 0. This machine — measured on 2026-10-03 (`scripts/env_report.py`)
| Fact | Value | Meaning |
|---|---|---|
| GPU | RTX 3050 6 GB Laptop, driver 581.86, PCIe 4.0 | |
| VRAM at idle | 6002 of 6144 MiB free (0 MiB reported used) | The iGPU is driving the display, so ~5.86 GiB is really available. The CUDA context (~0.3–0.5 GiB) comes out of that. |
| Max clocks | SM 2100 MHz, memory 7001 MHz | If the bus is 96-bit (my understanding of this card — confirm in a spec sheet), that is roughly 168 GB/s. |
| Power | limit not exposed; max limit 95 W | The enforced limit is unknown; the profiler now records clocks/power/temperature/throttle reasons after each run. |
| CPU / RAM | 20 logical threads (consistent with a 13650HX: 6 performance + 8 efficiency cores, my understanding); 23.7 GiB RAM | |
| OS / Python | Windows 11; Python 3.14.7; **torch not installed** | See §0.1. |
| GPU process list | one desktop app (`ChatGPT.exe`) is registered on the GPU; Windows does not report its VRAM | Idle usage is ~0, but close GPU-using apps before long runs and re-check with `env_report.py`. |

**Power and heat (laptop).** Always on AC, performance mode, airflow under the machine; never benchmark on battery. Keep long jobs chunked and resumable (the output cache makes reruns cheap) and compare the throttle fields in the profiler output across runs.

**CPU use.** Keep the GPU fed: a worker pool (start at 6–8 workers, tune by measuring images/s) decodes/resizes images and builds the perturbation ladder for the next image while the GPU scores the current one; pin host memory; set `OMP_NUM_THREADS` to match. Counterfactual edits that are not GPU-bound (copy-paste insertion, recolor, background swap, classical or small-network inpainting) can be generated on CPU concurrently; diffusion inpainting remains its own GPU phase.

**RAM.** Windows and background apps take several GB, so plan on ~16–18 GB for experiments. Keep cached vision features as memory-mapped arrays on the SSD.

### 0.1 Software setup (do this before anything else)
Your system Python is 3.14.7. PyTorch's install page lists Windows support for Python 3.10–3.14, but I also found a user report of CUDA wheels not resolving on 3.14 (it used old CUDA indexes, so it may be stale), and compiled ML packages usually lag the newest Python. Lowest-risk choice: **a separate Python 3.12 virtual environment** for this project.
```powershell
winget install Python.Python.3.12          # or: uv python install 3.12
py -3.12 -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
# PyTorch: copy the Windows / Pip / CUDA command from https://pytorch.org/get-started/locally/
pip install transformers accelerate peft bitsandbytes
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m bitsandbytes                      # diagnostic: should report a working CUDA backend
```
`pip install bitsandbytes` is documented to work on Windows with prebuilt CUDA binaries (third-party guide and the library's install docs; I have not run it on your machine). If the diagnostic fails, use WSL2 (set its memory limit in `.wslconfig`).

## 1. Where the budget goes (est., budget = your measured 5.86 GiB free)
Reproduce with `python -m src.utils.memory_budget --env results/hardware/env.json ...`. Estimates include 0.4 GiB CUDA context and a 0.75 GiB activation reserve (1.5 GiB for training) — assumptions, to be replaced by profiler numbers.

| Configuration (800 tokens incl. 576 image tokens unless noted) | Total est. | Headroom | Verdict |
|---|---|---|---|
| 7B Llama-2-shaped LLM, NF4+double-quant, vision tower fp16 | 5.71 GiB | +0.15 | fits barely — batch 1, nothing else on the GPU |
| …same, 600 tokens | 5.62 GiB | +0.25 | fits |
| …same, 400 tokens | 5.52 GiB | +0.34 | fits |
| ~3B GQA-class LLM, NF4+DQ | 3.64 GiB | +2.22 | fits (batch 8 KV: +2.03) |
| ~3B GQA-class, int8 | 4.98 GiB | +0.88 | fits |
| ~3B GQA-class, fp16 | 7.51 GiB | −1.65 | does not fit |
| QLoRA r=16, ~3B GQA-class, vision features cached | 4.11 GiB | +1.75 | fits |
| QLoRA r=8, 7B, vision features cached | 6.09 GiB | −0.23 | does not fit |

"~3B GQA-class" is an illustrative shape (36 layers, 2 KV heads), not a specific model. **Consequence:** 7B-class models are inference-only at batch 1 with ≲ 600–800 tokens; all LoRA training uses ≤ ~3B-class models.

## 2. Highest payoff: stop doing redundant work
1. **Image-prefix sharing.** Each image is probed many times (existence, attribute, count, paraphrases, original + control + counterfactual). For models where image tokens precede the question, prefill them once per image and reuse the KV cache for every question. `VLM.open_context()/score_in_context()` is the API (fallback has no speed-up; backends override). Loop *image-major*, probe-minor. Typically the largest single speed-up for this project.
2. **Read logits, don't generate.** Yes/No probes, multiple choice and likelihood metrics need one forward pass and the logits of a few tokens. Generate only for caption-based metrics (CHAIR, RMR).
3. **Cache per-image features and per-call outputs.** Output cache exists (`CachedVLM`). Add a vision-feature cache keyed by image hash (edited images are encoded once, reused across prompts/seeds/models sharing an encoder).
4. **Batch by similar length**, and use `generate_batch` overrides in backends (the default is sequential).
5. **Slice before the LM head:** never materialize full-vocabulary logits over image positions; compute logits only at the positions you score (large saving for 150k-token vocabularies).

## 3. Memory squeeze (inference)
- Free the card: run headless / attach the display to the iGPU, close GPU-accelerated browsers, check `nvidia-smi`. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
- NF4 + double quantization for the LM; keep the vision tower fp16 (or int8 if needed); embeddings and `lm_head` usually stay fp16 (the estimator counts them).
- Visual-token count is a **fixed experimental setting** (resolution/pooling), never a silent per-run trick: it changes model behaviour, so it is recorded in the manifest and held constant across conditions.
- `torch.inference_mode()`, SDPA/flash attention for scoring; **eager attention only** for attention extraction, with per-layer hooks on small sampled subsets (full attention tensors for 32 layers × 32 heads × 800² positions are ≈ 1.3 GiB per forward in fp16).
- Hidden states: hook only the layers being probed, move to CPU immediately, never keep `output_hidden_states` for all layers on GPU.

## 4. Training (LoRA) within 6 GiB
- Base ≤ ~3B-class, NF4, LoRA on LM linear layers only; vision tower and projector **frozen**, with **vision features pre-computed and stored on disk** (so the tower is not even resident). Disk cost scales with images × tokens × dim — use a 5–10k-image subset and measure.
- Gradient checkpointing; micro-batch 1 with gradient accumulation; 8-bit paged AdamW; loss on answer tokens only; sequence cap.
- The grounding loss needs the model's log-probs under both `I` and `I^{-c}`: about 2× compute per example. A *stop-gradient on the counterfactual branch* roughly halves activation memory but changes the method — treat it as an ablation, not a default.
- Everything fits only if the profiler agrees; the estimator's activation reserve (1.5 GiB) is a guess.

## 5. Pipeline staging (never co-resident on the GPU)
Phase A: generate counterfactuals (segmentation + inpainting; a diffusion inpainting model competes for the same 6 GiB) → Phase B: feature caching → Phase C: evaluation → Phase D: training. Free the GPU between phases; use the 24 GB RAM for dataset staging, not for pretending the GPU is bigger.

## 6. Where the limits are (stated so they're not discovered late)
- No fp16 reference for ≥ 4B models locally. Quantization-drift check (NF4 vs fp16, on a subset of probes) is done on a ≤ ~2B sibling model that fits in fp16, and the result is reported as evidence about *that* model only.
- CPU offload of a large model works but is far too slow for sweeps; use it only for small spot checks, and note that offload paths may not expose hidden states/attentions.
- Sustained clocks: check `nvidia-smi -q -d CLOCK,PERFORMANCE,POWER` during long runs. Laptop 3050 variants have lower power limits; avoid overclocking for a reproducible study.
- Decode speed is memory-bandwidth-bound (tokens/s ≲ bandwidth ÷ bytes read per token). Check your card's bandwidth with `nvidia-smi`/specs; 4-bit dequantization overhead usually puts real speed well under that ceiling.

## 7. Candidate models (leads only — from secondary web sources, NOT verified)
Criteria: (1) native `transformers` support with hidden-state/attention access; (2) open license compatible with your use; (3) fits §1; (4) **≥ 2 distinct architecture families** (needed by H1's falsification test).
- Different-family workhorse: **BLIP-2 OPT-2.7B** (only 32 visual tokens → tiny KV cache). Its ViT-g vision tower is large (~1B parameters, my understanding, ≈ 2 GB in fp16), so the *LM* is what gets quantized: est. ≈ 4.3 GiB with NF4 LM + fp16 vision tower; int8 LM ≈ 5.8 GiB (too tight). Default config is NF4. The BLIP-2 OPT model is not instruction-tuned and the official checkpoint download is large (order of 15 GB, unverified — check free disk and set `HF_HOME` to a drive with space).
- Modern small VLMs worth checking on their model cards: Qwen3-VL-4B, Gemma 3 4B, MiniCPM-V 4.x (1B-class), InternVL 2.5 (1B–2B), DeepSeek-VL2-Tiny, Qwen2.5-VL-3B. Verify existence, license, size, `transformers` support and memory before committing.
- Classic reference with published hallucination numbers: **LLaVA-1.5-7B** NF4 (headless only, tight).
- Contrastive baseline: CLIP ViT-B/32 (trivial memory).

## 8. First measurements to take (when backends exist)
`python scripts/profile_gpu.py --config configs/models/<m>.yaml --image <img> --batch-sizes 1,2,4,8,16 --out results/hardware/<m>.json` for every model × quantization; record peak allocated/reserved, max batch, images/s. Update §1 with measured values and replace estimates.
