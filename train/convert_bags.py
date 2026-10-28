"""Convert the Gazebo push demos (rosbag2 mcap) into a local LeRobotDataset (no upload).

    uv run --project train python train/convert_bags.py      # settings: configs/train_act.yaml
    uv run --project train python train/convert_bags.py --limit 2 --root /tmp/x   # quick try

One frame = one /demo/action message (sim-time 30 Hz ticks):
  action                      the joint commands of dataset.joints (rad), picked by name
  observation.state           /joint_states of the same joints, interpolated at the action stamp
  observation.images.overhead newest camera image at or before the action stamp (RGB, video)
Frames before the first /joint_states (the recorder may subscribe late) are dropped
(scripts/sim/bag_align.py). The per-episode alignment stats, seeds and demo results go to
<root>/jevlite_episodes.jsonl and <root>/jevlite_convert.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from bagio import episode_dirs, read_episode  # noqa: E402
from scripts.sim.bag_align import align  # noqa: E402


def features(cfg: dict, h: int, w: int) -> dict:
    joints = list(cfg["joints"])
    vec = {"dtype": "float32", "shape": (len(joints),), "names": joints}
    return {
        "action": dict(vec),
        "observation.state": dict(vec),
        cfg["image_key"]: {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
    }


def check_ticks(t_action: np.ndarray, fps: int, max_jitter_s: float, name: str) -> float:
    dt = np.diff(t_action) / 1e9
    jitter = float(np.max(np.abs(dt - 1.0 / fps))) if dt.size else 0.0
    if jitter > max_jitter_s:
        raise ValueError(f"{name}: action interval off 1/fps by {jitter * 1e3:.1f} ms")
    return jitter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "train_act.yaml"))
    ap.add_argument("--src", help="override dataset.src")
    ap.add_argument("--root", help="override dataset.root (must not exist)")
    ap.add_argument("--limit", type=int, help="convert only the first N episodes")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["dataset"]
    src = Path(args.src or ROOT / cfg["src"])
    root = Path(args.root or ROOT / cfg["root"])
    if root.exists():
        print(f"{root} exists; choose another --root (nothing is overwritten)", file=sys.stderr)
        return 2

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    results = {}
    jl = src / "episodes.jsonl"
    if jl.exists():
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            results[r["bag"]] = r
    dirs = episode_dirs(src)
    if cfg.get("only_success", True):
        dirs = [d for d in dirs if results.get(d.name, {}).get("result", {}).get("success")]
    if args.limit:
        dirs = dirs[: args.limit]
    if not dirs:
        print(f"no episodes in {src}", file=sys.stderr)
        return 1

    ds = None
    rows, t0 = [], time.monotonic()
    for i, d in enumerate(dirs):
        e = read_episode(d, cfg["topics"], cfg["joints"])
        jitter = check_ticks(e.t_action, cfg["fps"], cfg["max_tick_jitter_s"], d.name)
        al = align(e.t_action, e.t_state, e.state, e.t_image, hold_state=cfg["hold_state"])
        st = al.stats()
        if st["image_age_ms_max"] is None or st["image_age_ms_max"] > cfg["max_image_age_s"] * 1e3:
            raise ValueError(f"{d.name}: image age {st['image_age_ms_max']} ms over the limit")
        keep = np.flatnonzero(al.keep)
        if np.any(np.diff(keep) != 1):
            raise ValueError(f"{d.name}: kept frames are not one contiguous run")
        if ds is None:
            h, w = e.images[0].shape[:2]
            v = cfg["video"]
            ds = LeRobotDataset.create(
                repo_id=cfg["repo_id"],
                fps=cfg["fps"],
                features=features(cfg, h, w),
                root=root,
                robot_type=cfg["robot_type"],
                use_videos=True,
                rgb_encoder=RGBEncoderConfig(
                    vcodec=v["vcodec"], crf=v["crf"], pix_fmt=v["pix_fmt"], g=v["g"]
                ),
            )
        for k in keep:
            ds.add_frame(
                {
                    "action": e.action[k].astype(np.float32),
                    "observation.state": al.state[k].astype(np.float32),
                    cfg["image_key"]: e.images[al.image_index[k]],
                    "task": cfg["task"],
                }
            )
        ds.save_episode()
        rows.append(
            {
                "episode_index": i,
                "bag": d.name,
                "seed": e.info["seed"],
                "frames": len(keep),
                "first_action_tick": int(keep[0]),
                "t_first_ns": int(e.t_action[keep[0]]),
                "tick_jitter_ms": jitter * 1e3,
                "align": st,
                "demo_result": results.get(d.name, {}).get("result"),
            }
        )
        print(
            f"[{i + 1}/{len(dirs)}] {d.name}: {len(keep)} frames "
            f"(dropped {st['dropped_no_image']} no image, {st['dropped_no_state']} no state), "
            f"image age max {st['image_age_ms_max']:.1f} ms, {time.monotonic() - t0:.0f} s",
            flush=True,
        )
    ds.finalize()

    (root / "jevlite_episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ages = [r["align"]["image_age_ms_max"] for r in rows]
    summary = {
        "src": str(src.relative_to(ROOT) if src.is_relative_to(ROOT) else src),
        "config": cfg,
        "episodes": len(rows),
        "frames": int(sum(r["frames"] for r in rows)),
        "frames_per_episode": [min(r["frames"] for r in rows), max(r["frames"] for r in rows)],
        "dropped_no_state_total": int(sum(r["align"]["dropped_no_state"] for r in rows)),
        "dropped_no_image_total": int(sum(r["align"]["dropped_no_image"] for r in rows)),
        "image_age_ms_max": max(ages),
        "image_age_ms_mean": float(
            sum(r["align"]["image_age_ms_mean"] * r["frames"] for r in rows)
            / sum(r["frames"] for r in rows)
        ),
        "state_gap_ms_max": max(r["align"]["state_gap_ms_max"] for r in rows),
        "tick_jitter_ms_max": max(r["tick_jitter_ms"] for r in rows),
        "convert_s": round(time.monotonic() - t0, 1),
    }
    (root / "jevlite_convert.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
