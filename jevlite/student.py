"""v1 / v2: students that judge none / edge / center from one camera image (Jev-like).

Every judge here computes zone logits (order ZONES), adds the per-zone logit bias of the config
for its kind (safe-side tuning chosen on val, 0 by default), turns them into probabilities, maps
zones to choices (m4a.yaml `zones`: none -> CONTINUE, edge -> SLOW, center -> STOP) and returns a
Decision through the common judge(image, t_capture) entrance (jevlite.api).

  v1  SigLIP image features (frozen) -> multinomial logistic regression (LinearHead, numpy)
  v2  MobileNetV3-Small fine-tuned (torchvision)

Preprocessing is the same for training and judging (GpuPre): crop the fixed pixel box around the
band (configs/judge_sim.yaml `crop`), resize on the GPU (bilinear, antialiased), normalize.
Heavy imports (torch, transformers, torchvision) are deferred, so the numpy parts can be tested
without them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jevlite.api import Choice, Decision, decision_from_probs, start_judging

ZONES = ("none", "edge", "center")
SIGLIP_MEAN = SIGLIP_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.atleast_2d(np.asarray(z, np.float64))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def bias_vector(logit_bias: dict[str, float] | None) -> np.ndarray:
    b = logit_bias or {}
    unknown = set(b) - set(ZONES)
    if unknown:
        raise ValueError(f"logit_bias has unknown zones {sorted(unknown)}")
    v = np.array([float(b.get(z, 0.0)) for z in ZONES])
    if not np.all(np.isfinite(v)):
        raise ValueError(f"logit_bias must be finite, got {b}")
    return v


def zone_choices(zones: dict[str, str]) -> list[Choice]:
    """Choice of each zone in ZONES order (m4a.yaml `zones`)."""
    if set(zones) != set(ZONES):
        raise ValueError(f"zones must map each of {ZONES}, got {sorted(zones)}")
    return [Choice.parse(zones[z]) for z in ZONES]


def choice_probs(zone_p: np.ndarray, choices: Sequence[Choice]) -> dict[str, float]:
    """Zone probabilities -> probability per choice (summed if zones share a choice)."""
    out = {c.name: 0.0 for c in Choice}
    for p, c in zip(zone_p, choices, strict=True):
        out[c.name] += float(p)
    return out


def check_box(box: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    u0, v0, u1, v1 = (int(v) for v in box)
    if not (0 <= u0 < u1 <= width and 0 <= v0 < v1 <= height):
        raise ValueError(f"crop {box} is not inside a {width}x{height} image")
    return u0, v0, u1, v1


class ZoneJudge:
    """judge(image, t_capture) -> Decision from a zone-logits function."""

    def __init__(self, logits_fn: Callable[[np.ndarray], np.ndarray], zones: dict[str, str],
                 source: str, logit_bias: dict[str, float] | None = None):  # fmt: skip
        self.logits_fn, self.source = logits_fn, source
        self.choices = zone_choices(zones)
        self.bias = bias_vector(logit_bias)

    def zone_probs(self, image: np.ndarray) -> np.ndarray:
        return softmax_rows(np.asarray(self.logits_fn(image), np.float64) + self.bias)[0]

    def judge(self, image: np.ndarray, t_capture: float | None = None) -> Decision:
        start = start_judging(t_capture)
        p = self.zone_probs(image)
        return decision_from_probs(choice_probs(p, self.choices), self.source, start)


class LinearHead:
    """Standardize, then logits = W x + b (multinomial logistic regression, classes = ZONES)."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean = np.asarray(mean, np.float64)
        self.scale = np.asarray(scale, np.float64)
        self.coef = np.asarray(coef, np.float64)
        self.intercept = np.asarray(intercept, np.float64)
        d = self.mean.shape[0]
        if self.scale.shape != (d,) or self.coef.shape != (len(ZONES), d) \
                or self.intercept.shape != (len(ZONES),):  # fmt: skip
            raise ValueError("LinearHead shapes do not match")

    def logits(self, feats: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(np.asarray(feats, np.float64)) - self.mean) / self.scale
        return x @ self.coef.T + self.intercept

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept)

    @classmethod
    def load(cls, path: Path) -> LinearHead:
        z = np.load(path)
        return cls(z["mean"], z["scale"], z["coef"], z["intercept"])


def resolve_device(device: str):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class GpuPre:
    """uint8 HxWx3 RGB images -> (B, 3, h, w) tensor: crop, resize, scale to [0, 1], normalize.
    normalize=False returns the resized uint8 tensor (for caching training inputs)."""

    def __init__(self, box: Sequence[int], size_wh: Sequence[int], mean, std, device, dtype):
        import torch

        self.torch = torch
        self.box = tuple(int(v) for v in box)
        self.size_hw = (int(size_wh[1]), int(size_wh[0]))
        self.device, self.dtype = device, dtype
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def resized(self, images: Sequence[np.ndarray]):
        torch = self.torch
        u0, v0, u1, v1 = self.box
        h, w = images[0].shape[:2]
        check_box(self.box, w, h)
        a = np.stack([np.ascontiguousarray(im[v0:v1, u0:u1, :3]) for im in images])
        x = torch.from_numpy(a).to(self.device, non_blocking=True).permute(0, 3, 1, 2).float()
        return torch.nn.functional.interpolate(x, size=self.size_hw, mode="bilinear",
                                               antialias=True, align_corners=False)  # fmt: skip

    def normalize(self, x01):
        """x01: (B, 3, h, w) float in [0, 1]."""
        return ((x01 - self.mean) / self.std).to(self.dtype)

    def __call__(self, images: Sequence[np.ndarray]):
        return self.normalize(self.resized(images) / 255.0)


class SiglipFeatures:
    """Frozen SigLIP vision tower: images -> features (B, D) of the kinds asked for.
    pooled: the attention-pool (MAP) output (768). grid2: patch tokens averaged on a 2 x 2 grid."""

    KINDS = ("pooled", "grid2")

    def __init__(self, cfg: dict[str, Any], box: Sequence[int], kinds: Sequence[str]):
        import torch
        from transformers import AutoModel

        bad = set(kinds) - set(self.KINDS)
        if bad or not kinds:
            raise ValueError(f"feature kinds must be from {self.KINDS}, got {list(kinds)}")
        self.torch, self.kinds = torch, list(kinds)
        self.device = resolve_device(cfg.get("device", "auto"))
        self.dtype = getattr(torch, cfg.get("dtype", "float16")) \
            if self.device.type == "cuda" else torch.float32  # fmt: skip
        model = AutoModel.from_pretrained(cfg["model_id"], revision=cfg["revision"],
                                          dtype=self.dtype)  # fmt: skip
        self.vision = model.vision_model.to(self.device).eval()
        size = int(cfg["size"])
        self.pre = GpuPre(box, (size, size), SIGLIP_MEAN, SIGLIP_STD, self.device, self.dtype)

    def from_pixels(self, pv) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            out = self.vision(pixel_values=pv)
            parts = []
            for k in self.kinds:
                if k == "pooled":
                    parts.append(out.pooler_output.float())
                else:
                    t = out.last_hidden_state.float()
                    n = int(round(t.shape[1] ** 0.5))
                    g = t.view(t.shape[0], n, n, -1).permute(0, 3, 1, 2)
                    parts.append(torch.nn.functional.adaptive_avg_pool2d(g, 2).flatten(1))
            return torch.cat(parts, dim=1).cpu().numpy()

    def __call__(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return self.from_pixels(self.pre(images))


def build_mobilenet(arch: str, weights: str | None, n_classes: int = len(ZONES)):
    import torch
    import torchvision

    if arch != "mobilenet_v3_small":
        raise ValueError(f"unsupported v2 arch {arch!r}")
    net = torchvision.models.mobilenet_v3_small(weights=weights)
    last = net.classifier[-1]
    net.classifier[-1] = torch.nn.Linear(last.in_features, n_classes)
    return net


class CnnLogits:
    """v2 network on the GPU: image -> zone logits (fp16 on CUDA, channels_last)."""

    def __init__(self, net, box, size_wh, device):
        import torch

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.net = net.to(device).to(self.dtype).to(memory_format=torch.channels_last).eval()
        self.pre = GpuPre(box, size_wh, IMAGENET_MEAN, IMAGENET_STD, device, self.dtype)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            x = self.pre([image]).contiguous(memory_format=self.torch.channels_last)
            return self.net(x).float().cpu().numpy()


def load_student(model_dir: Path, judge_cfg: dict[str, Any], zones: dict[str, str],
                 device: str = "auto") -> ZoneJudge:  # fmt: skip
    """A trained v1 / v2 judge from its folder (meta.json + head.npz / model.pt).
    The logit bias comes from judge_cfg (configs/judge_sim.yaml); crop and sizes from meta.json
    (what the model was trained with). logit_bias is per kind: judge_cfg["logit_bias"][kind]."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    bias = (judge_cfg.get("logit_bias") or {}).get(meta["kind"])
    if meta["kind"] == "v1":
        feats = SiglipFeatures({**meta["siglip"], "device": device}, meta["crop"],
                               meta["features"])  # fmt: skip
        head = LinearHead.load(model_dir / "head.npz")
        return ZoneJudge(lambda im: head.logits(feats([im])), zones, meta["source"], bias)
    if meta["kind"] == "v2":
        import torch

        dev = resolve_device(device)
        net = build_mobilenet(meta["arch"], None)
        net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        cnn = CnnLogits(net, meta["crop"], meta["input"], dev)
        return ZoneJudge(cnn, zones, meta["source"], bias)
    raise ValueError(f"unknown student kind {meta['kind']!r}")


def zero_shot_zone_judge(backend, prompts: dict[str, list[str]], zones: dict[str, str],
                         source: str = "v0-zones") -> ZoneJudge:  # fmt: skip
    """v0 with one prompt group per zone (image-text similarity, no training). `backend` is a
    jevlite.zeroshot EmbeddingBackend (full image, its own preprocessing, as v0)."""
    text = backend.text_embeddings([list(prompts[z]) for z in ZONES])
    scale = float(backend.logit_scale)
    return ZoneJudge(lambda im: scale * (text @ backend.image_embedding(im)), zones, source)
