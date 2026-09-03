"""Thin Qwen3-VL-8B-Instruct wrapper for PHASE C (generation + solving).

Import ONLY after src.runtime.setup_runtime() (setproctitle + CUDA_VISIBLE_DEVICES).
Deterministic greedy decoding everywhere. One process = one visible GPU (cuda:0).
"""
from __future__ import annotations

import torch

try:
    from transformers import Qwen3VLForConditionalGeneration as _CausalVL
except Exception:  # pragma: no cover - older transformers
    _CausalVL = None
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as _AutoITT
except Exception:  # pragma: no cover
    _AutoITT = None

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class Qwen3VL:
    def __init__(self, cfg: dict):
        m = cfg["model"]
        self.model_name = m["name"]
        self.gen_cfg = cfg["generation"]
        dtype = _DTYPES[m.get("torch_dtype", "bfloat16")]
        pk = {}
        if m.get("min_pixels"):
            pk["min_pixels"] = int(m["min_pixels"])
        if m.get("max_pixels"):
            pk["max_pixels"] = int(m["max_pixels"])
        self.processor = AutoProcessor.from_pretrained(self.model_name, **pk)

        Loader = _CausalVL or _AutoITT
        if Loader is None:
            raise RuntimeError("transformers has neither Qwen3VLForConditionalGeneration nor "
                               "AutoModelForImageTextToText — upgrade transformers >= 4.57")
        try:
            self.model = Loader.from_pretrained(self.model_name, dtype=dtype,
                                                attn_implementation="sdpa").to("cuda:0")
        except TypeError:  # older signature
            self.model = Loader.from_pretrained(self.model_name, torch_dtype=dtype,
                                                attn_implementation="sdpa").to("cuda:0")
        self.model.eval()
        gc = self.model.generation_config
        gc.do_sample = False
        for k in ("temperature", "top_p", "top_k"):
            setattr(gc, k, None)

    # ---- low level -----------------------------------------------------------------
    @torch.inference_mode()
    def _gen(self, messages: list, max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda:0")
        out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                                  do_sample=False, num_beams=1)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        txt = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        return txt.strip()

    @staticmethod
    def _img_msg(image_path: str, text: str) -> list:
        # transformers 5.x chat-template image loading wants a bare local path (not file://).
        return [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": text.strip()}]}]

    @staticmethod
    def _txt_msg(text: str) -> list:
        return [{"role": "user", "content": [{"type": "text", "text": text.strip()}]}]

    # ---- modes --------------------------------------------------------------------
    def caption(self, image_path: str, instruction: str, max_new_tokens: int) -> str:
        return self._gen(self._img_msg(image_path, instruction), max_new_tokens)

    def vqa_image(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        return self._gen(self._img_msg(image_path, prompt), max_new_tokens)

    def vqa_text(self, prompt: str, max_new_tokens: int) -> str:
        return self._gen(self._txt_msg(prompt), max_new_tokens)
