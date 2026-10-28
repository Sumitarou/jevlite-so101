"""Generate small synthetic scenes into samples/synthetic/ (no people, safe to publish).

These only exercise the pipeline (loading, judging, latency). Their "correct" labels are
not meaningful; use real images in samples/real/ to judge accuracy.

    uv run python scripts/make_synthetic_samples.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 320, 240
SKIN = (224, 172, 140)


def _table(rng: np.random.Generator) -> Image.Image:
    base = np.array([150, 115, 80], dtype=np.float64)  # wooden table
    grad = np.linspace(0.85, 1.1, H)[:, None, None]
    img = np.clip(base * grad + rng.normal(0, 4, (H, W, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _arm(draw: ImageDraw.ImageDraw, base_x: int, angle: float) -> tuple[int, int]:
    """Draw a simple 3-link grey arm from the bottom; return the gripper position."""
    x, y = float(base_x), float(H - 10)
    draw.rectangle([base_x - 22, H - 22, base_x + 22, H], fill=(40, 40, 45))
    for length, da in ((80, 0.0), (65, 0.6), (35, 0.5)):
        a = angle + da
        nx, ny = x + length * np.sin(a), y - length * np.cos(a)
        draw.line([x, y, nx, ny], fill=(70, 70, 78), width=14)
        draw.ellipse([nx - 9, ny - 9, nx + 9, ny + 9], fill=(30, 30, 30))
        x, y = nx, ny
    return int(x), int(y)


def _hand(draw: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    draw.ellipse([cx - 28, cy - 20, cx + 28, cy + 24], fill=SKIN)  # palm
    for i in range(4):
        fx = cx - 21 + i * 14
        draw.rounded_rectangle([fx - 5, cy - 52, fx + 5, cy - 12], radius=5, fill=SKIN)
    draw.rounded_rectangle([cx + 22, cy - 18, cx + 50, cy - 6], radius=6, fill=SKIN)  # thumb
    draw.rectangle([cx - 24, cy + 18, cx + 24, H + 20], fill=(60, 90, 140))  # sleeve


SCENES = {
    # name: (arm angle, hand position relative to gripper or None, with cube)
    "empty_table": (None, None, False),
    "arm_only_a": (-0.3, None, True),
    "arm_only_b": (0.2, None, False),
    "hand_far_a": (-0.3, (230, 60), True),
    "hand_far_b": (0.2, (-230, 40), False),
    "hand_mid": (-0.1, (110, 30), True),
    "hand_near_a": (-0.3, (35, 25), True),
    "hand_near_b": (0.2, (-30, 20), False),
}


def make(name: str, seed: int) -> Image.Image:
    angle, hand_off, cube = SCENES[name]
    rng = np.random.default_rng(seed)
    img = _table(rng)
    draw = ImageDraw.Draw(img)
    if cube:
        draw.rectangle([200, 120, 228, 148], fill=(200, 40, 40))
    if angle is not None:
        gx, gy = _arm(draw, W // 2 - 40, angle)
        if hand_off is not None:
            hx = int(np.clip(gx + hand_off[0], 30, W - 30))
            _hand(draw, hx, gy + hand_off[1])
    return img.filter(ImageFilter.GaussianBlur(0.6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("samples/synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(SCENES):
        path = args.out / f"{name}.png"
        make(name, args.seed + i).save(path, optimize=True)
        print(path)


if __name__ == "__main__":
    main()
