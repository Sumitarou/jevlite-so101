"""Export camera frames + intruder labels (none / edge / center) from an M4a run, for M4b.

    uv run --project train python train/export_intruder_frames.py \
        outputs/sim/<stamp>_m4a_gate [--every 3] [--out data/sim_intruder/<name>]

Reads each episode bag of a scripts/sim/gate_demo_sim.py run (rosbags, no ROS): the camera images
and /intruder/pose. Each kept frame gets two labels:
  zone       from the scripted rod position at the image stamp (scenario keys + the loop's t0,
             the same truth the oracle judge used)
  zone_meas  from the measured rod pose (latest /intruder/pose at or before the stamp)
Frames before the loop starts are skipped. Writes <out>/images/<scenario>_<n>.png and
<out>/labels.csv (git-ignored data/: the frames are synthetic, but data/ never goes in git).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sim.intruder import Scenario, zone_of  # noqa: E402

FIELDS = ["file", "scenario", "t_rel", "stamp", "zone", "zone_meas", "rod_x", "rod_y",
          "rod_x_meas", "rod_y_meas", "gate_state"]  # fmt: skip


def scenario_of(e: dict, m4a: dict, sim_cfg: dict) -> Scenario:
    """The scenario as it was run (keys from the episode, places from the run's config copy)."""
    p, itr = m4a["path"], sim_cfg["intruder"]
    z = float(p["hover"]) + float(itr["length"]) / 2
    points = {n: (float(p["x"]), float(y), z) for n, y in p["places"].items()}
    keys = [(float(t), str(n)) for t, n in e.get("keys", [])]
    return Scenario(e["scenario"], e.get("desc", ""), keys, e.get("judge", {}), points,
                    tuple(float(v) for v in itr["park"]))  # fmt: skip


def label_rows(sc: Scenario, sim_cfg: dict, loop_t0: float, end_t: float, stamps: np.ndarray,
               poses: tuple[np.ndarray, np.ndarray], ticks: list[dict],
               every: int) -> list[dict]:  # fmt: skip
    """Rows for the frames inside the loop (every `every`-th one). poses = (t, xy) measured."""
    pt, pxy = poses
    tick_t = np.array([loop_t0 + k["t"] for k in ticks]) if ticks else np.array([])
    rows, n = [], 0
    for idx, s in enumerate(stamps):
        if s < loop_t0 - 1e-6 or s > end_t + 1e-6:
            continue
        if n % every:
            n += 1
            continue
        n += 1
        x, y, _ = sc.position(s - loop_t0)
        row = {"index": idx, "scenario": sc.id, "t_rel": round(s - loop_t0, 4),
               "stamp": round(float(s), 4), "zone": zone_of(x, y, sim_cfg),
               "rod_x": round(x, 5), "rod_y": round(y, 5), "zone_meas": None,
               "rod_x_meas": None, "rod_y_meas": None, "gate_state": None}  # fmt: skip
        k = int(np.searchsorted(pt, s, side="right")) - 1
        if k >= 0:
            mx, my = pxy[k]
            row.update(zone_meas=zone_of(mx, my, sim_cfg), rod_x_meas=round(float(mx), 5),
                       rod_y_meas=round(float(my), 5))  # fmt: skip
        j = int(np.searchsorted(tick_t, s + 1e-6, side="right")) - 1
        if j >= 0:
            row["gate_state"] = ticks[j]["gate"]["state"]
        rows.append(row)
    return rows


def read_bag(bag: Path, image_topic: str, pose_topic: str):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    stamps, images, pt, pxy = [], [], [], []
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as r:
        conns = [c for c in r.connections if c.topic in (image_topic, pose_topic)]
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if c.topic == image_topic:
                stamps.append(t)
                images.append(np.frombuffer(bytes(m.data), np.uint8)
                              .reshape(m.height, m.step)[:, : m.width * 3]
                              .reshape(m.height, m.width, 3))  # fmt: skip
            else:
                pt.append(t)
                pxy.append((m.pose.position.x, m.pose.position.y))
    order = np.argsort(stamps, kind="stable")
    po = np.argsort(pt, kind="stable")
    return (np.asarray(stamps)[order], [images[k] for k in order],
            (np.asarray(pt)[po], np.asarray(pxy).reshape(-1, 2)[po]))  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--every", type=int, default=3, help="keep every n-th frame (30 fps -> 10)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    from PIL import Image

    summary = json.loads((args.run / "summary.json").read_text())
    m4a = summary["config"]
    import yaml

    sim_cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    out = args.out or ROOT / "data" / "sim_intruder" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.run.name)  # fmt: skip
    (out / "images").mkdir(parents=True, exist_ok=False)
    all_rows = []
    for line in (args.run / "results.jsonl").read_text().splitlines():
        e = json.loads(line)
        if "loop_t0" not in e:
            print(f"{e['scenario']}: no loop_t0 (older run), skipped", file=sys.stderr)
            continue
        bag = args.run / "bags" / e["bag"]
        ticks = [json.loads(x) for x in (bag / "ticks.jsonl").read_text().splitlines()]
        stamps, images, poses = read_bag(bag, sim_cfg["camera"]["topic"],
                                         sim_cfg["intruder"]["pose_topic"])  # fmt: skip
        end_t = e["loop_t0"] + ticks[-1]["t"]
        rows = label_rows(scenario_of(e, m4a, sim_cfg), sim_cfg, e["loop_t0"], end_t, stamps,
                          poses, ticks, args.every)  # fmt: skip
        for n, r in enumerate(rows):
            r["file"] = f"images/{e['scenario']}_{n:05d}.png"
            Image.fromarray(images[r.pop("index")]).save(out / r["file"])
        all_rows += rows
        zones = {z: sum(r["zone"] == z for r in rows) for z in ("none", "edge", "center")}
        print(f"{e['scenario']}: {len(rows)} frames {zones}", flush=True)
    with (out / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    agree = [r["zone"] == r["zone_meas"] for r in all_rows if r["zone_meas"] is not None]
    print(f"{len(all_rows)} frames -> {out}; scripted vs measured zone agree "
          f"{sum(agree)}/{len(agree)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
