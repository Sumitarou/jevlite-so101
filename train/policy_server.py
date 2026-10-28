"""Serve a trained policy to scripts/sim/eval_policy.py over a local socket (one client).

    JEVLITE_POLICY_KEY=<hex> uv run --project train python train/policy_server.py \
        --checkpoint models/act/<run>/lerobot/checkpoints/030000 --address <socket>
    ... --fake hold        # no model: command the current joint state (for the evaluator's tests)

eval_policy.py normally starts this itself. Protocol: scripts/sim/policy_link.py. Settings
(joints, fixed joints, image key, task) come from configs/train_act.yaml `dataset`.
--checkpoint takes a checkpoint folder (…/checkpoints/<step> or …/last) or its pretrained_model/.
Before serving, one warm-up step runs on a blank observation (CUDA/cuDNN start-up), followed by
a reset, so the first real tick is not slowed by it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from policy_runtime import PolicyRuntime, hold_runtime  # noqa: E402
from scripts.sim.policy_link import KEY_ENV, serve  # noqa: E402
from units import ARM6  # noqa: E402


def pretrained_dir(path: Path) -> Path:
    path = path.resolve()
    return path if (path / "config.json").exists() else path / "pretrained_model"


def make_handler(rt: PolicyRuntime, info: dict):
    def handle(msg: dict) -> dict:
        cmd = msg.get("cmd")
        if cmd == "reset":
            return {"queue_len": rt.begin_episode(), "episode": rt.episodes}
        if cmd == "act":
            r = rt.act(np.asarray(msg["image"]), np.asarray(msg["state"]))
            return {**r, "names": list(info["command_names"])}
        if cmd == "info":
            return info
        raise ValueError(f"unknown cmd {cmd!r}")

    return handle


def warm_up(rt: PolicyRuntime, image_shape: tuple[int, int, int]) -> float:
    t = time.perf_counter()
    rt.begin_episode()
    rt.act(np.zeros(image_shape, np.uint8), np.zeros(len(rt.joints), np.float32))
    rt.begin_episode()
    rt.episodes = 0  # count only the evaluator's episodes
    return (time.perf_counter() - t) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "train_act.yaml"))
    ap.add_argument("--address", required=True, help="AF_UNIX socket path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path)
    g.add_argument("--fake", choices=["hold"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-action-steps", type=int)
    ap.add_argument("--temporal-ensemble-coeff", type=float)
    ap.add_argument("--image-shape", type=int, nargs=3, default=[480, 640, 3])
    args = ap.parse_args()
    key = os.environ.get(KEY_ENV)
    if not key:
        print(f"{KEY_ENV} is not set", file=sys.stderr)
        return 2

    ds = yaml.safe_load(Path(args.config).read_text())["dataset"]
    t = time.perf_counter()
    if args.fake:
        rt, src = hold_runtime(ds), f"fake:{args.fake}"
    else:
        pm = pretrained_dir(args.checkpoint)
        rt = PolicyRuntime.from_checkpoint(
            pm, ds, args.device, args.n_action_steps, args.temporal_ensemble_coeff
        )
        src = str(pm.relative_to(ROOT) if pm.is_relative_to(ROOT) else pm)
    load_ms = (time.perf_counter() - t) * 1e3
    warm_ms = warm_up(rt, tuple(args.image_shape))
    pcfg = getattr(rt.policy, "config", None)
    info = {
        "policy": src,
        "device": rt.device,
        "joints": rt.joints,
        "fixed_joints": rt.fixed,
        "command_names": list(ARM6),
        "chunk_size": getattr(pcfg, "chunk_size", None),
        "n_action_steps": getattr(pcfg, "n_action_steps", None),
        "temporal_ensemble_coeff": getattr(pcfg, "temporal_ensemble_coeff", None),
        "load_ms": round(load_ms, 1),
        "warmup_ms": round(warm_ms, 1),
    }
    serve(
        args.address,
        key,
        make_handler(rt, info),
        on_ready=lambda: print("READY " + json.dumps(info), flush=True),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
