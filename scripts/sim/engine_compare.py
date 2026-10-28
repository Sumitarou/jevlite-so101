"""Compare physics engines on the same push episodes (bags written by gen_demos.py).

    source /opt/ros/jazzy/setup.bash && \
      python3 scripts/sim/engine_compare.py data/sim_demos/<dart run> data/sim_demos/<bullet run>

Runs with the system Python (rosbag2_py). For every episode of every run (the run's engine is
read from its summary.json) it measures, from the recorded topics:
  - the block's motion: displacement from its start to the end of the settle phase, the angle
    between the push direction and that displacement (the DART friction problem: y only)
  - stability: block height error (sinking / jumping), tilt, speed outside the push phases
  - the arm: tracking error |command - joint state|, joint speed while holding, the lowest
    point of the arm meshes (forward kinematics) against the white area
and, for runs with the same seed, the largest joint difference between the engines.
Everything goes to a new folder outputs/sim/<YYYYmmdd_HHMMSS>_<name>/ (--name, default
engine_compare): the mid-push camera frame of each episode (engine_<engine>_seed<seed>.png),
one side-by-side image per seed (compare_seed<seed>.png) and engine_compare.json; a Markdown
table is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rosbag2_py  # noqa: E402
import yaml  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from PIL import ImageDraw  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from sensor_msgs.msg import Image, JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from scripts.sim.demo_plan import ARM, surface_z  # noqa: E402
from scripts.sim.kinematics import Robot, tool_pose, transform  # noqa: E402
from scripts.sim.outputs import new_run_dir  # noqa: E402

TYPES = {"/block/pose": PoseStamped, "/joint_states": JointState, "/demo/action": JointState,
         "/overhead_camera/image": Image}  # fmt: skip


def stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def read_bag(bag: Path, want_image_t: float) -> dict:
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))  # fmt: skip
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(TYPES)))
    out = {"block": [], "joints": [], "action": [], "image": None}
    names = None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, TYPES[topic])
        t = stamp(msg)
        if topic == "/block/pose":
            p, o = msg.pose.position, msg.pose.orientation
            out["block"].append([t, p.x, p.y, p.z, o.x, o.y, o.z, o.w])
        elif topic in ("/joint_states", "/demo/action"):
            if names is None:
                names = list(msg.name)
            d = dict(zip(msg.name, msg.position, strict=False))
            v = dict(zip(msg.name, msg.velocity, strict=False)) if len(msg.velocity) else {}
            row = [t] + [d[j] for j in names] + [v.get(j, np.nan) for j in names]
            out["joints" if topic == "/joint_states" else "action"].append(row)
        elif out["image"] is None or abs(t - want_image_t) < abs(out["image"][0] - want_image_t):
            out["image"] = (t, msg)
    out["names"] = names
    for k in ("block", "joints", "action"):
        out[k] = np.array(out[k])
    return out


def image_array(msg: Image) -> np.ndarray:
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    arr = arr[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return arr[..., ::-1] if msg.encoding == "bgr8" else arr


def tilt_deg(q) -> np.ndarray:
    qx, qy = q[:, 0], q[:, 1]
    return np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1)))


def yaw_deg(q) -> np.ndarray:
    qx, qy, qz, qw = q.T
    return np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def load_info(bag: Path) -> tuple[dict, bool]:
    """episode.json, or (episode stopped early) the start message on /demo/episode."""
    if (bag / "episode.json").exists():
        return json.loads((bag / "episode.json").read_text()), True
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))  # fmt: skip
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/demo/episode"]))
    _, data, _ = reader.read_next()
    return json.loads(deserialize_message(data, String).data), False


def windows(info: dict, rate: float) -> dict[str, tuple[float, float]]:
    """Phase windows in sim time from the planned ticks (the same times for every engine)."""
    return {n: (info["sim_start"] + a / rate, info["sim_start"] + b / rate)
            for n, a, b in info["phases"]}  # fmt: skip


def rnd(x, nd):
    return None if x is None else round(float(x), nd)


def masked_max(x, m):
    return float(np.nanmax(x[m])) if m.any() else None


def analyse(cfg: dict, robot: Robot, bag: Path, engine: str, out: Path) -> dict:
    info, complete = load_info(bag)
    rate = float(cfg["demo"]["rate_hz"])
    w = windows(info, rate)
    d = read_bag(bag, w["push"][0] + 0.5 * (w["push"][1] - w["push"][0]))
    seed = info["seed"]
    h = cfg["table"]["height"]
    z_expected = surface_z(cfg) + cfg["block"]["size"][2] / 2
    blk = d["block"]
    t, xy, z, q = blk[:, 0], blk[:, 1:3], blk[:, 3] - h, blk[:, 4:8]
    in_push = (t >= w["push"][0]) & (t <= w["after_push"][1])

    start = np.array(info["block_start"][:2])
    final = np.array(info["final_block_pose"][:2]) if complete else xy[-1]
    disp = final - start
    pdir = np.array(info["push_dir"])
    ang = np.degrees(np.arctan2(pdir[0] * disp[1] - pdir[1] * disp[0], pdir @ disp))
    planned = np.linalg.norm(np.array(info["target"]) - start)

    v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    quiet = ~((t[1:] >= w["push"][0]) & (t[1:] <= w["lift"][1]))  # outside push..lift

    names = d["names"]
    n = len(names)
    js, act = d["joints"], d["action"]
    # tracking: joint state at each command time (linear interpolation)
    err = np.stack([np.interp(act[:, 0], js[:, 0], js[:, 1 + i]) - act[:, 1 + i]
                    for i in range(n)], axis=1)  # fmt: skip
    ap = (act[:, 0] >= w["push"][0]) & (act[:, 0] <= w["push"][1])
    bp = (act[:, 0] >= w["before_push"][0]) & (act[:, 0] <= w["before_push"][1])
    jvel = np.abs(js[:, 1 + n : 1 + 2 * n]).max(axis=1)
    hold = (js[:, 0] >= w["settle"][0]) & (js[:, 0] <= w["settle"][1])

    # TCP height: commanded vs actual (forward kinematics) while waiting before the push
    rc, base = cfg["reach_check"], transform(cfg["robot"]["xyz"], cfg["robot"]["rpy"])

    def tcp_z(rows):
        jv = {j: rows[:, 1 + names.index(j)] for j in ARM + ["gripper"]}
        return tool_pose(robot, jv, base, rc["tip_link"], rc["tcp"])[0][:, 2] - surface_z(cfg)

    qa = np.concatenate([act[:, :1], act[:, 1 : 1 + n] + err], axis=1)
    tcp_gap = (tcp_z(qa[bp]) - tcp_z(act[bp])).mean() if bp.any() else None

    # lowest point of the arm meshes (all links but the base), every joint state (100 Hz)
    jv = {j: js[:, 1 + names.index(j)] for j in ARM + ["gripper"]}
    links = set(robot.vertices) - {robot.root_link}
    low = robot.points(jv, base, links=links, kind="vertices")[..., 2].min(axis=1)

    frame = None
    if d["image"] is not None:
        frame = out / f"engine_{engine}_seed{seed}.png"
        PILImage.fromarray(image_array(d["image"][1])).save(frame)
    res = info.get("result", {})
    return {
        "engine": engine, "seed": seed, "bag": str(bag.resolve().relative_to(ROOT)),
        "complete": complete, "success": res.get("success"), "error_m": res.get("error_m"),
        "block_start": info["block_start"], "push_dir": info["push_dir"],
        "push_dir_deg_from_y": round(float(np.degrees(np.arctan2(pdir[0], pdir[1]))), 1),
        "disp": [round(float(v_), 4) for v_ in disp],
        "disp_m": round(float(np.linalg.norm(disp)), 4), "planned_m": round(float(planned), 4),
        "angle_err_deg": round(float(ang), 1),
        "yaw_change_deg": round(float(yaw_deg(q[-1:])[0] - yaw_deg(q[:1])[0]), 1),
        "block_z_err_mm": [round(float((z - z_expected).min() * 1000), 2),
                           round(float((z - z_expected).max() * 1000), 2)],
        "block_tilt_max_deg": round(float(tilt_deg(q).max()), 2),
        "block_tilt_max_push_deg": rnd(masked_max(tilt_deg(q), in_push), 2),
        "block_speed_max": round(float(v.max()), 3),
        "block_speed_max_quiet": rnd(masked_max(v, quiet), 4),
        "tcp_above_cmd_before_push_mm": rnd(None if tcp_gap is None else tcp_gap * 1000, 2),
        "track_err_before_push_max_rad": rnd(masked_max(np.abs(err).max(1), bp), 4),
        "track_err_push_max_rad": rnd(masked_max(np.abs(err).max(1), ap), 4),
        "joint_speed_max_hold": rnd(masked_max(jvel, hold), 4),
        "arm_low_min_mm": round(float((low.min() - surface_z(cfg)) * 1000), 2),
        "sim_reached_s": round(float(js[-1, 0] - info["sim_start"]), 2),
        "rtf_stream": info.get("rtf_stream"), "wall_s": info.get("wall_duration_s"),
        "frame": str(frame.relative_to(ROOT)) if frame else None,
        "frame_sim_t": round(d["image"][0] - info["sim_start"], 3) if d["image"] else None,
        "_joints": js[:, : 1 + n], "_t0": info["sim_start"],
    }  # fmt: skip


def side_by_side(rows: list[dict], path: Path):
    imgs = [PILImage.open(ROOT / r["frame"]) for r in rows]
    w, hgt = imgs[0].size
    out = PILImage.new("RGB", (w * len(imgs), hgt + 24), "white")
    draw = ImageDraw.Draw(out)
    for i, (r, im) in enumerate(zip(rows, imgs, strict=True)):
        out.paste(im, (i * w, 24))
        draw.text((i * w + 6, 6), f"{r['engine']}  seed {r['seed']}  t={r['frame_sim_t']} s "
                  f"(mid-push)  angle err {r['angle_err_deg']} deg", fill="black")  # fmt: skip
    out.save(path)


def joint_diff(a: dict, b: dict) -> float:
    """Largest |q_a - q_b| over the episode (time from each recording's start)."""
    ja, jb = a["_joints"], b["_joints"]
    ta, tb = ja[:, 0] - a["_t0"], jb[:, 0] - b["_t0"]
    m = (ta >= max(ta[0], tb[0])) & (ta <= min(ta[-1], tb[-1]))
    return float(max(np.abs(np.interp(ta[m], tb, jb[:, 1 + i]) - ja[m, 1 + i]).max()
                     for i in range(ja.shape[1] - 1)))  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--name", default="engine_compare", help="suffix of the output folder")
    args = ap.parse_args(argv)
    out = new_run_dir(args.name)
    cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    robot = Robot()
    rows = []
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        engine = summary.get("physics_engine", "dartsim")
        for bag in sorted(p for p in run.iterdir() if list(p.glob("*.mcap"))):
            rows.append(analyse(cfg, robot, bag, engine, out))
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    pairs = {}
    for seed, rs in by_seed.items():
        if len(rs) >= 2 and all(r["frame"] for r in rs):
            img = out / f"compare_seed{seed}.png"
            side_by_side(rs, img)
            pairs[seed] = {"joint_max_diff_rad": round(joint_diff(rs[0], rs[1]), 4),
                           "image": str(img.relative_to(ROOT))}  # fmt: skip
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    runs = [str(r.resolve().relative_to(ROOT)) for r in args.runs]
    (out / "engine_compare.json").write_text(
        json.dumps({"runs": runs, "episodes": clean, "pairs": pairs}, indent=2) + "\n"
    )
    print(f"outputs: {out.relative_to(ROOT)}")
    cols = ["engine", "seed", "complete", "success", "error_m", "block_start",
            "push_dir_deg_from_y", "disp", "angle_err_deg", "yaw_change_deg", "block_z_err_mm",
            "block_tilt_max_deg", "block_speed_max", "block_speed_max_quiet",
            "tcp_above_cmd_before_push_mm", "track_err_before_push_max_rad",
            "track_err_push_max_rad", "joint_speed_max_hold", "arm_low_min_mm", "sim_reached_s",
            "rtf_stream", "wall_s"]  # fmt: skip
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in clean:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print(json.dumps(pairs, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
