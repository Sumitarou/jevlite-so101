"""Save one camera frame of a recorded demo episode as PNG.

    source /opt/ros/jazzy/setup.bash && \
      python3 scripts/sim/demo_frame.py data/sim_demos/<run>/ep_0000_seed0 [--phase push] \
        [--at 0.5] [--out FILE.png]

Runs with the system Python (rosbag2_py). The time is taken from the bag's episode.json:
the phase's start + `at` x the phase's length (sim time from the start of the recording).
Without --out the PNG goes to a new folder outputs/sim/<YYYYmmdd_HHMMSS>_demo_frame/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from PIL import Image as PILImage
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.sim.outputs import new_run_dir  # noqa: E402


def frame_time(info: dict, phase: str, at: float) -> float:
    """Sim time [s] (absolute) of the frame to extract."""
    starts = info["phase_start_s"]
    names = list(starts)
    t0 = starts[phase]
    i = names.index(phase)
    t1 = starts[names[i + 1]] if i + 1 < len(names) else info["sim_duration_s"]
    return info["sim_start"] + t0 + at * (t1 - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bag", type=Path)
    ap.add_argument("--phase", default="push")
    ap.add_argument("--at", type=float, default=0.5, help="fraction of the phase (0..1)")
    ap.add_argument("--topic", default="/overhead_camera/image")
    ap.add_argument("--out", type=Path, help="PNG path (default: a new folder per run)")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = new_run_dir("demo_frame") / f"{args.bag.name}_{args.phase}_{args.at}.png"
    info = json.loads((args.bag / "episode.json").read_text())
    want = frame_time(info, args.phase, args.at)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))  # fmt: skip
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))
    best = None
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, Image)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if best is None or abs(t - want) < abs(best[0] - want):
            best = (t, msg)
    if best is None:
        print(f"no {args.topic} in {args.bag}", file=sys.stderr)
        return 1
    t, msg = best
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    arr = arr[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        arr = arr[..., ::-1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(arr).save(args.out)
    print(json.dumps({"bag": str(args.bag), "phase": args.phase, "sim_time": round(t, 3),
                      "wanted": round(want, 3), "size": [msg.width, msg.height],
                      "encoding": msg.encoding, "out": str(args.out)}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
