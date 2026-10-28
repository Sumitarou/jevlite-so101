"""Run the v0 zero-shot judge on every image under a folder and report latency.

    uv run python scripts/run_zeroshot_on_images.py --dir samples/ [--csv outputs/v0.csv]

Latency is measured inside judge() (preprocess + model + softmax). Image file loading and the
warmup runs done at init are excluded.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from jevlite.api import Choice, Decision, Judge

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def latency_stats(latencies_ms: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(latencies_ms), dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def format_row(rel: str, d: Decision) -> str:
    probs = " ".join(f"{c.name}={d.probs[c.name]:.3f}" for c in Choice)
    return f"{rel:<40} {d.choice.name:<8} {probs}  {d.latency_ms:7.2f} ms"


def run(judge: Judge, root: Path, csv_path: Path | None = None, out=sys.stdout) -> dict:
    paths = find_images(root)
    rows = []
    for p in paths:
        d = judge.judge(load_rgb(p))
        rel = str(p.relative_to(root))
        print(format_row(rel, d), file=out)
        rows.append({"path": rel, "choice": d.choice.name, **d.probs, "latency_ms": d.latency_ms})
    stats = latency_stats(r["latency_ms"] for r in rows)
    if stats["n"]:
        print(
            f"\n{stats['n']} images  mean={stats['mean']:.2f} ms  p50={stats['p50']:.2f} ms  "
            f"p95={stats['p95']:.2f} ms  max={stats['max']:.2f} ms",
            file=out,
        )
    else:
        print(f"no images found under {root}", file=out)
    if csv_path and rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("samples"))
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args(argv)

    from jevlite.zeroshot import build_zeroshot_judge

    stats = run(build_zeroshot_judge(), args.dir, args.csv)
    return 0 if stats["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
