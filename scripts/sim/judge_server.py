"""M4b: the image judge as a server for the Gazebo closed loop (uv env: torch, SigLIP).

    JEVLITE_POLICY_KEY=<hex> uv run python scripts/sim/judge_server.py --address <sock> \
        --kind v1 --model models/judge/<stamp>_v1

Started and stopped by scripts/sim/gate_demo_sim.py --judge <name> (system Python, ROS), like the
policy server: the two Python environments talk over scripts/sim/policy_link (AF_UNIX socket with
an authkey from the environment; numpy arrays as raw bytes). Requests and replies:

  {"cmd": "judge", "image": HxWx3 uint8} -> {"ok": True, "probs": {CONTINUE, SLOW, STOP},
                                             "latency_ms": float, "source": str}
  {"cmd": "info"}                        -> {"ok": True, ...}
  {"cmd": "close"}                       -> the server stops
latency_ms is the judge's own Decision.latency_ms (preprocess + inference, wall time); the
client turns it into the sim-time delay (scripts/sim/sim_judge.RemoteJudge + JudgeRunner).
Kinds: v1 / v2 (jevlite.student.load_student), v0-zones (zero-shot with the rod prompts).
Settings: configs/judge_sim.yaml (siglip, v0 prompts, logit_bias, zones_config).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.sim.policy_link import KEY_ENV, serve  # noqa: E402

KINDS = ("v1", "v2", "v0-zones")


def build(kind: str, model: Path | None, cfg: dict, device: str):
    from jevlite.student import load_student, zero_shot_zone_judge

    zones = yaml.safe_load((ROOT / cfg["zones_config"]).read_text())["zones"]
    if kind in ("v1", "v2"):
        if model is None:
            raise ValueError(f"--model is needed for {kind}")
        return load_student(model, cfg, zones, device=device)
    from jevlite.zeroshot import SiglipBackend

    s = cfg["siglip"]
    backend = SiglipBackend(s["model_id"], s["revision"], device, s["dtype"])
    return zero_shot_zone_judge(backend, cfg["v0"]["prompts"], zones)


def make_handler(judge, info: dict):
    def handle(msg: dict) -> dict:
        cmd = msg.get("cmd")
        if cmd == "judge":
            d = judge.judge(np.asarray(msg["image"]))
            return {"probs": d.probs, "latency_ms": d.latency_ms, "source": d.source}
        if cmd == "info":
            return dict(info)
        raise ValueError(f"unknown command {cmd!r}")

    return handle


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M4b: image judge server")
    ap.add_argument("--address", required=True, help="AF_UNIX socket path")
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--model", type=Path, help="models/judge/<stamp>_<v1|v2>")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "judge_sim.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--image-shape", type=int, nargs=3, default=[480, 640, 3])
    args = ap.parse_args(argv)
    key = os.environ.get(KEY_ENV)
    if not key:
        print(f"{KEY_ENV} is not set", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(args.config.read_text())
    t = time.perf_counter()
    judge = build(args.kind, args.model, cfg, args.device)
    load_ms = (time.perf_counter() - t) * 1e3
    dummy = np.zeros(tuple(args.image_shape), np.uint8)
    lat = [judge.judge(dummy).latency_ms for _ in range(max(args.warmup, 1))]
    info = {"kind": args.kind, "model": str(args.model) if args.model else None,
            "source": judge.source, "load_ms": round(load_ms, 1),
            "warmup_last_ms": round(lat[-1], 3)}  # fmt: skip
    serve(args.address, key, make_handler(judge, info),
          on_ready=lambda: print("READY " + json.dumps(info), flush=True))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
