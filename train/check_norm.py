"""Range of the normalized policy inputs/targets, with the stats a checkpoint actually uses.

    uv run --project train python train/check_norm.py models/act/<run> [--checkpoint last]

lerobot's MEAN_STD normalizer computes (x - mean) / (std + 1e-8) with the stats saved in
pretrained_model/policy_preprocessor_step_*_normalizer_processor.safetensors. This applies them
to every frame of the dataset (observation.state and action) and reports, per dimension, the
largest |z|, and what a 0.002 rad offset becomes. For the camera it reports the per-channel
mean/std (ImageNet stats when dataset.use_imagenet_stats) and the z range of pixels in [0, 1].
Writes outputs/sim/<stamp>_m3b_norm/norm.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sim.outputs import new_run_dir  # noqa: E402

EPS = 1e-8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--checkpoint", default="last")
    args = ap.parse_args()

    import pandas as pd
    from safetensors.numpy import load_file

    pm = args.run / "lerobot" / "checkpoints" / args.checkpoint / "pretrained_model"
    stats = load_file(next(iter(sorted(pm.glob("policy_preprocessor_step_*_normalizer_*")))))
    tcfg = json.loads((pm / "train_config.json").read_text())
    root = Path(tcfg["dataset"]["root"])
    names = json.loads((root / "meta" / "info.json").read_text())["features"]["action"]["names"]
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/*/*.parquet"))))

    out: dict = {"checkpoint": str(pm.relative_to(args.run)), "frames": len(df), "eps": EPS}
    for key in ("observation.state", "action"):
        x = np.stack(df[key].values).astype(np.float64)
        mean, std = stats[f"{key}.mean"].astype(np.float64), stats[f"{key}.std"].astype(np.float64)
        z = (x - mean) / (std + EPS)
        out[key] = [
            {
                "dim": n,
                "mean": float(mean[j]),
                "std": float(std[j]),
                "z_absmax": float(np.abs(z[:, j]).max()),
                "z_p99": float(np.percentile(np.abs(z[:, j]), 99)),
                "z_of_2mrad": 0.002 / (std[j] + EPS),
            }
            for j, n in enumerate(names)
        ]
    k = "observation.images.overhead"
    m, s = stats[f"{k}.mean"].reshape(-1), stats[f"{k}.std"].reshape(-1)
    out[k] = {
        "mean": m.tolist(),
        "std": s.tolist(),
        "z_range_of_pixels_0_1": [
            float(((0 - m) / (s + EPS)).min()),
            float(((1 - m) / (s + EPS)).max()),
        ],
    }
    run_out = new_run_dir("m3b_norm")
    (run_out / "norm.json").write_text(json.dumps(out, indent=2))
    for key in ("observation.state", "action"):
        print(f"{key}:")
        for r in out[key]:
            print(
                f"  {r['dim']:<14} std {r['std']:.4f}  max|z| {r['z_absmax']:.2f}  "
                f"p99|z| {r['z_p99']:.2f}  z(2 mrad) {r['z_of_2mrad']:.3f}"
            )
    print(f"{k}: mean {np.round(m, 3).tolist()} std {np.round(s, 3).tolist()} "
          f"z range {np.round(out[k]['z_range_of_pixels_0_1'], 2).tolist()}")  # fmt: skip
    print(f"-> {run_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
