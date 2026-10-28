"""v0: zero-shot judge using image-text similarity (SigLIP). No training.

The model-specific part is the backend (encode text / encode image). ZeroShotJudge only does
similarity -> softmax over choices -> Decision, so it can be tested with a fake backend.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from jevlite.api import Choice, Decision, decision_from_probs, start_judging
from jevlite.config import load_config


class EmbeddingBackend(Protocol):
    logit_scale: float

    def text_embeddings(self, prompts: list[list[str]]) -> np.ndarray:
        """(n_choices, d), L2-normalized. Multiple prompts per choice are averaged."""
        ...

    def image_embedding(self, image: np.ndarray) -> np.ndarray:
        """(d,), L2-normalized. `image` is HxWx3 uint8 RGB."""
        ...


def softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def probs_from_embeddings(
    image_emb: np.ndarray, text_embs: np.ndarray, scale: float, names: list[str]
) -> dict[str, float]:
    logits = scale * (text_embs.astype(np.float64) @ image_emb.astype(np.float64))
    return {n: float(p) for n, p in zip(names, softmax(logits), strict=True)}


class ZeroShotJudge:
    def __init__(
        self,
        backend: EmbeddingBackend,
        choices_cfg: dict[str, Any],
        softmax_scale: float | None = None,
        warmup: int = 0,
        source: str = "v0",
    ):
        entries = choices_cfg["choices"]
        self.names = [Choice.parse(e["name"]).name for e in entries]
        if sorted(self.names) != sorted(c.name for c in Choice):
            raise ValueError(f"choices.yaml must list each of {[c.name for c in Choice]} once")
        self.backend = backend
        self.text_embs = backend.text_embeddings([list(e["prompts"]) for e in entries])
        self.scale = float(softmax_scale) if softmax_scale else float(backend.logit_scale)
        self.source = source
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(warmup):
            self.judge(dummy)

    def judge(self, image: np.ndarray, t_capture: float | None = None) -> Decision:
        """Decision.t_mono = t_capture (camera frame) or, if None, the time judging started."""
        start = start_judging(t_capture)
        emb = self.backend.image_embedding(image)
        probs = probs_from_embeddings(emb, self.text_embs, self.scale, self.names)
        return decision_from_probs(probs, self.source, start)


class SiglipBackend:
    """HF transformers SigLIP. Heavy imports are deferred so tests need no torch."""

    def __init__(self, model_id: str, revision: str, device: str = "auto", dtype: str = "float16"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = (
            AutoModel.from_pretrained(model_id, revision=revision, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.logit_scale = float(self.model.logit_scale.exp().item())

    @staticmethod
    def _pooled(out):
        # transformers>=5 returns a model output; older versions return the tensor itself.
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def _normalize(self, x):
        return x / x.norm(dim=-1, keepdim=True)

    def text_embeddings(self, prompts: list[list[str]]) -> np.ndarray:
        torch = self._torch
        rows = []
        with torch.inference_mode():
            for group in prompts:
                # SigLIP was trained with max_length padding; other padding hurts accuracy.
                inp = self.processor(text=group, padding="max_length", return_tensors="pt")
                emb = self._normalize(
                    self._pooled(self.model.get_text_features(**inp.to(self.device))).float()
                )
                rows.append(self._normalize(emb.mean(dim=0)))
        return torch.stack(rows).cpu().numpy()

    def image_embedding(self, image: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            inp = self.processor(images=image, return_tensors="pt")
            pixel_values = inp["pixel_values"].to(self.device, self.dtype)
            emb = self._pooled(self.model.get_image_features(pixel_values=pixel_values))
            return self._normalize(emb.float())[0].cpu().numpy()


def build_zeroshot_judge(
    zeroshot_cfg: dict[str, Any] | None = None, choices_cfg: dict[str, Any] | None = None
) -> ZeroShotJudge:
    zcfg = zeroshot_cfg or load_config("zeroshot.yaml")
    ccfg = choices_cfg or load_config("choices.yaml")
    backend = SiglipBackend(
        zcfg["model_id"], zcfg["revision"], zcfg.get("device", "auto"), zcfg.get("dtype", "float16")
    )
    return ZeroShotJudge(
        backend, ccfg, softmax_scale=zcfg.get("softmax_scale"), warmup=int(zcfg.get("warmup", 0))
    )
