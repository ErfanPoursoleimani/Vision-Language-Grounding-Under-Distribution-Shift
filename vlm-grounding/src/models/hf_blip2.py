"""BLIP-2 (OPT) backend on Hugging Face transformers + bitsandbytes.

STATUS: written against the documented transformers API (processor(images, text) -> model.generate)
but NOT yet executed against transformers 5.x on a GPU. Run `scripts/smoke_model.py` first; its
self-checks (single-forward vs two-forward yes/no log-odds, batch vs single decoding, determinism)
exist to catch slicing/padding mistakes in this file.

Notes that matter for the research:
* BLIP-2-OPT is NOT instruction-tuned. Use the empty prompt for captions and the
  "Question: ... Answer:" format for closed probes; do not expect chat behaviour.
* "No image" (language-prior baseline) is approximated by a neutral gray image, because the
  model needs pixel input: the LM still receives query embeddings of a blank image, not nothing.
* The vision tower (ViT-g, ~1B params) stays fp16 by default so quantization does not confound
  perception; only the language model is quantized (see quantization_census()).
* torch / transformers are imported lazily so the rest of the repo works without them.
"""
from __future__ import annotations

from typing import Any, Optional

from PIL import Image

from .base import Capability, VLM, VLMOutput
from .registry import register_model

_VISION_SKIP = ["vision_model", "qformer", "language_projection"]


@register_model("hf_blip2")
class HFBlip2(VLM):
    capabilities = frozenset({Capability.GENERATE, Capability.SCORE})
    QUANT = ("none", "8bit", "4bit")

    def __init__(self, name: str, hf_id: str = "Salesforce/blip2-opt-2.7b", revision: Optional[str] = None,
                 quantize: str = "4bit", quantize_vision: bool = False, dtype: str = "float16", device: int = 0,
                 blind_gray: int = 128, question_template: str = "Question: {q} Answer:") -> None:
        if quantize not in self.QUANT:
            raise ValueError(f"quantize must be one of {self.QUANT}, got {quantize!r}")
        if dtype not in ("float16", "bfloat16"):
            raise ValueError("dtype must be 'float16' or 'bfloat16'")
        super().__init__(name, hf_id=hf_id, revision=revision, quantize=quantize, quantize_vision=quantize_vision,
                         dtype=dtype, device=device, blind_gray=blind_gray, question_template=question_template)
        self._model = None
        self._proc = None
        self._torch = None
        self._dtype = None

    # ---- prompt / image handling (pure, unit-tested) -----------------------------
    def format_prompt(self, prompt: str) -> str:
        p = (prompt or "").strip()
        if not p:
            return ""
        if p.startswith("Question:"):
            return p
        return self.config["question_template"].format(q=p)

    def _as_rgb(self, image: Any) -> Image.Image:
        if image is None:
            g = int(self.config["blind_gray"])
            return Image.new("RGB", (224, 224), (g, g, g))
        return image.convert("RGB")

    # ---- loading ----------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import transformers
            from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig
        except ImportError as e:  # noqa: BLE001
            raise ImportError("hf_blip2 needs torch and transformers (see docs/hardware_plan.md section 0.1)") from e
        if not torch.cuda.is_available():
            raise RuntimeError("hf_blip2 requires a CUDA GPU")
        c = self.config
        self._torch = torch
        self._dtype = getattr(torch, c["dtype"])
        kw: dict = {"device_map": {"": c["device"]}}
        if c["quantize"] != "none":
            skip = ([] if c["quantize_vision"] else list(_VISION_SKIP)) + ["lm_head"]
            if c["quantize"] == "8bit":
                kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
            else:
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=self._dtype, llm_int8_skip_modules=skip)
        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        kw[dtype_key] = self._dtype
        self._proc = Blip2Processor.from_pretrained(c["hf_id"], revision=c["revision"])
        self._proc.tokenizer.padding_side = "left"  # required for batched generation
        self._model = Blip2ForConditionalGeneration.from_pretrained(c["hf_id"], revision=c["revision"], **kw).eval()

    def runtime_info(self) -> dict:
        """Facts about the loaded model for run manifests (kept out of metadata() so cache keys stay stable)."""
        self._load()
        import transformers
        return {"torch": self._torch.__version__, "transformers": transformers.__version__,
                "resolved_commit_hash": getattr(self._model.config, "_commit_hash", None),
                "quantization_census": self.quantization_census()}

    def quantization_census(self) -> dict:
        self._load()
        groups = ("vision_model", "qformer", "language_projection", "language_model", "lm_head")
        out: dict = {}
        for n, m in self._model.named_modules():
            t = type(m).__name__
            if t not in ("Linear4bit", "Linear8bitLt", "Linear"):
                continue
            g = next((x for x in groups if x in n.split(".")), "other")
            out[f"{g}:{t}"] = out.get(f"{g}:{t}", 0) + 1
        return out

    # ---- tensors ----------------------------------------------------------------
    def _inputs(self, images: list, texts: list) -> dict:
        self._load()
        kw: dict = {"images": images, "return_tensors": "pt"}
        if any(texts):
            kw.update(text=texts, padding=True)
        enc = self._proc(**kw)
        dev = f"cuda:{self.config['device']}"
        return {k: (v.to(dev, dtype=self._dtype) if v.is_floating_point() else v.to(dev)) for k, v in enc.items()}

    # ---- API --------------------------------------------------------------------
    def generate(self, image, prompt, *, max_new_tokens=128, temperature=0.0, seed=0) -> VLMOutput:
        return self.generate_batch([image], [prompt], max_new_tokens=max_new_tokens,
                                   temperature=temperature, seed=seed)[0]

    def generate_batch(self, images, prompts, *, max_new_tokens=128, temperature=0.0, seed=0):
        self._load()
        torch = self._torch
        enc = self._inputs([self._as_rgb(i) for i in images], [self.format_prompt(p) for p in prompts])
        kw: dict = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                    "return_dict_in_generate": True, "output_scores": True}
        if temperature > 0:
            kw["temperature"] = temperature
            torch.manual_seed(seed)
        with torch.inference_mode():
            out = self._model.generate(**enc, **kw)
        n_new = len(out.scores)
        new = out.sequences[:, -n_new:] if n_new else out.sequences[:, :0]
        try:
            lps = self._model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        except Exception:  # noqa: BLE001  (logprobs are optional; text is what matters)
            lps = None
        tok = self._proc.tokenizer
        stop = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
        results = []
        for b in range(new.shape[0]):
            ids = new[b].tolist()
            cut = next((j for j, t in enumerate(ids) if t in stop), len(ids))
            text = tok.decode(ids[:cut], skip_special_tokens=True).strip()
            tl = lps[b, :cut].tolist() if lps is not None else None
            results.append(VLMOutput(text=text, token_logprobs=tl, meta={"n_new_tokens": cut}))
        return results

    def _cont_ids(self, continuation: str) -> list:
        cont = continuation if continuation.startswith(" ") else " " + continuation
        return self._proc.tokenizer(cont, add_special_tokens=False).input_ids

    def score(self, image, prompt, continuation) -> float:
        self._load()
        torch = self._torch
        fmt = self.format_prompt(prompt)
        if not fmt:
            raise ValueError("score() needs a non-empty prompt (use e.g. 'Is there a car?')")
        enc = self._inputs([self._as_rgb(image)], [fmt])
        ids = self._cont_ids(continuation)
        dev = enc["input_ids"].device
        full = torch.cat([enc["input_ids"], torch.tensor([ids], device=dev)], dim=1)
        mask = torch.cat([enc["attention_mask"], torch.ones((1, len(ids)), dtype=enc["attention_mask"].dtype,
                                                            device=dev)], dim=1)
        with torch.inference_mode():
            logits = self._model(**{**enc, "input_ids": full, "attention_mask": mask}).logits
        n = len(ids)
        lp = torch.log_softmax(logits[0, -(n + 1):-1, :].float(), dim=-1)  # tail slicing works whether or not
        return float(lp[torch.arange(n, device=lp.device), torch.tensor(ids, device=lp.device)].sum())  # query rows are prepended

    def yes_no_logodds(self, image, question) -> float:
        """One forward pass: log p(' Yes') - log p(' No') for the first answer token."""
        self._load()
        torch = self._torch
        fmt = self.format_prompt(question)
        if not fmt:
            raise ValueError("yes_no_logodds() needs a non-empty question")
        enc = self._inputs([self._as_rgb(image)], [fmt])
        yes, no = self._cont_ids("Yes")[0], self._cont_ids("No")[0]
        with torch.inference_mode():
            logits = self._model(**enc).logits
        lp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
        return float(lp[yes] - lp[no])
