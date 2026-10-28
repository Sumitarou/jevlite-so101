"""M4b: labelled camera images of the rod (none / edge / center) for training the image judge.

    source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
      timeout --signal=INT --kill-after=30 7200 python3 scripts/sim/gen_intruder_data.py

    ... --episodes 3 --name pilot      # a short trial
    ... --dry-run                      # the estimate only

Runs with the system Python (rclpy), like gate_demo_sim.py. Settings: configs/judge_sim.yaml
`data` (episodes, placements, split), configs/sim.yaml (table, rod, camera, demo), configs/m4a.yaml
(the rod's hover height and the M4a path). Gazebo starts once; for each episode (seed =
seed_base + i):
  1. plan a push demo (scripts/sim/demo_plan.Planner: block start and arm motion from the seed),
     move the arm home, put the block at its start, park the rod
  2. stream the demo commands; every hold_ticks ticks the rod is moved (set_pose) to a new
     placement from scripts/sim/intruder_sampling.IntruderSampler (not into the block or robot)
  3. one image per placement is saved: the first image taken settle_frames camera frames after
     the measured rod pose first shows the new position. The label is the zone of the measured
     pose (intruder.zone_of rule); images where it differs from the request by more than
     pose_tol_m, or that never came in time, are skipped and counted
No bag is recorded. Output: data/sim_intruder/<stamp>_<name>/ with images/*.png, labels.csv
(one row per image: split, placement, zone, distances to the boundaries, arm clearance, joints,
block pose) and run.json (config copies, git state, counts, timings). data/ is git-ignored.
All processes are stopped at the end and checked to be gone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import yaml  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402

from scripts.sim.demo_plan import Planner, surface_z  # noqa: E402
from scripts.sim.eval_policy import image_to_array, leftover_processes  # noqa: E402
from scripts.sim.gate_demo_sim import GateNode, _stamp, make_node, start_launch  # noqa: E402
from scripts.sim.gen_demos import stop_group  # noqa: E402
from scripts.sim.intruder import load_m4a  # noqa: E402
from scripts.sim.intruder_sampling import (  # noqa: E402
    ArmGeometry,
    IntruderSampler,
    boundary_distance,
    footprint_distances,
    load_judge_sim,
    make_forbidden,
    split_of,
    zone_from_distances,
)
from scripts.sim.outputs import git_state, new_run_dir  # noqa: E402
from scripts.sim.robustness import block_xyyaw  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FIELDS = ["file", "episode", "seed", "split", "placement", "tick", "t_image", "kind", "target",
          "near", "x_req", "y_req", "x", "y", "zone", "d_white", "d_outer", "d_boundary",
          "arm_clear", "arm_near", "block_x", "block_y", "block_yaw",
          *[f"q_{j}" for j in JOINTS]]  # fmt: skip
WALL_PER_EPISODE_S = 38.0  # measured in the 3-episode trial (31-39 s, RTF 0.38)
IMAGES_PER_EPISODE = 80  # measured in the trial (70-92)


class DataNode(GateNode):
    """GateNode that keeps a short history of the measured rod poses."""

    def __init__(self, cfg, arm_joints, grip_joints):
        super().__init__(cfg, arm_joints, grip_joints)
        self.intr_hist: deque[tuple[float, tuple[float, float, float]]] = deque(maxlen=60)

    def _on_intr(self, msg: PoseStamped):
        super()._on_intr(msg)
        p = msg.pose.position
        z = p.z - self.cfg["table"]["height"] - surface_z(self.cfg)
        self.intr_hist.append((_stamp(msg), (p.x, p.y, z)))


def first_match(hist, t_req: float, xyz, tol: float) -> float | None:
    """Stamp of the first measured pose at or after t_req that is within tol of xyz."""
    for t, p in hist:
        if t >= t_req - 1e-9 and math.dist(p, xyz) <= tol:
            return t
    return None


def pose_at(hist, t: float):
    """Latest measured pose at or before t (None if there is none)."""
    best = None
    for s, p in hist:
        if s <= t + 1e-9:
            best = p
    return best


class Episode:
    """The placements of one episode, driven from the demo stream's on_tick."""

    def __init__(self, node, cfg, data, m4a, sampler, arm, out, index, seed, writer):
        self.node, self.cfg, self.d, self.m4a = node, cfg, data, m4a
        self.sampler, self.arm, self.out, self.writer = sampler, arm, out, writer
        self.index, self.seed = index, seed
        self.split = split_of(seed, data["split"])
        self.rng = np.random.default_rng([seed, 1])
        self.cur: dict | None = None
        self.n_place = 0
        self.saved = 0
        self.skipped = {"not_in_time": 0, "pose_mismatch": 0}
        self.set_pose_failed = 0
        self.frame_s = 1.0 / float(cfg["camera"]["update_rate"])
        self.t0 = node.sim_now()

    def _finish(self):
        if self.cur is not None and not self.cur["saved"]:
            self.skipped["not_in_time"] += 1

    def _place(self, i: int):
        self._finish()
        node = self.node
        bx, by, _ = block_xyyaw(node)
        forbidden, _ = make_forbidden(self.arm, dict(node.joints), (bx, by), self.d["sampling"])
        p = self.sampler.sample(self.rng, forbidden)
        self.set_pose_failed += node.move_intruder(p["x"], p["y"], p["z"])
        self.cur = {**p, "t_req": node.sim_now(), "tick": i, "placement": self.n_place,
                    "saved": False}  # fmt: skip
        self.n_place += 1

    def _try_save(self):
        c, node = self.cur, self.node
        if c is None or c["saved"] or node.image is None:
            return
        req = (c["x"], c["y"], c["z"])
        tol = float(self.d["pose_tol_m"])
        t_match = first_match(node.intr_hist, c["t_req"], req, tol)
        t_img = _stamp(node.image)
        if t_match is None or t_img < t_match + self.d["settle_frames"] * self.frame_s - 1e-6:
            return
        meas = pose_at(node.intr_hist, t_img)
        c["saved"] = True
        if meas is None or math.dist(meas, req) > tol:
            self.skipped["pose_mismatch"] += 1
            return
        self._write(c, meas, image_to_array(node.image), t_img)

    def _write(self, c: dict, meas, img: np.ndarray, t_img: float):
        from PIL import Image as PILImage

        node, sim = self.node, self.cfg
        name = f"ep{self.index:04d}_p{c['placement']:03d}.png"
        PILImage.fromarray(img).save(self.out / "images" / name,
                                     compress_level=int(self.d["png_compress_level"]))
        x, y = float(meas[0]), float(meas[1])
        dw, do = footprint_distances(x, y, sim)
        joints = dict(node.joints)
        arm_pts, _ = self.arm.points(joints)
        clear = self.arm.clearance(arm_pts, x, y)
        bx, by, byaw = block_xyyaw(node)
        row = {"file": name, "episode": self.index, "seed": self.seed, "split": self.split,
               "placement": c["placement"], "tick": c["tick"],
               "t_image": round(t_img - self.t0, 4), "kind": c["kind"], "target": c["target"],
               "near": int(c["near"]), "x_req": round(c["x"], 5), "y_req": round(c["y"], 5),
               "x": round(x, 5), "y": round(y, 5), "zone": zone_from_distances(dw, do),
               "d_white": round(dw, 5), "d_outer": round(do, 5),
               "d_boundary": round(boundary_distance(x, y, sim), 5),
               "arm_clear": round(clear, 5) if math.isfinite(clear) else "",
               "arm_near": int(clear <= self.d["arm_near_m"]),
               "block_x": round(bx, 5), "block_y": round(by, 5), "block_yaw": round(byaw, 5),
               **{f"q_{j}": round(float(joints.get(j, 0.0)), 5) for j in JOINTS}}  # fmt: skip
        self.writer.writerow(row)
        self.saved += 1

    def on_tick(self, i: int):
        self._try_save()
        if i % int(self.d["hold_ticks"]) == 0:
            self._place(i)

    def close(self):
        self._try_save()
        self._finish()


def run_episode(node, planner, cfg, data, m4a, sampler, arm, out, index, writer) -> dict:
    wall0 = time.monotonic()
    seed = int(data["seed_base"]) + index
    plan = planner.plan(seed)
    names, rate, demo = plan.joint_names, plan.rate_hz, cfg["demo"]
    if not node.place_intruder(*sampler.park):
        raise RuntimeError("set_pose of the intruder failed")
    cur = np.array([node.joints.get(j, 0.0) for j in names])
    n = max(1, int(round(demo["home_move_s"] * rate)))
    s = (np.arange(1, n + 1) / n)[:, None]
    node.stream(names, cur + s * (plan.q[0] - cur), rate)
    x, y, yaw = plan.block_start
    if not node.place_block(x, y, yaw):
        raise RuntimeError("set_pose of the block failed")
    node.spin_for(demo["reset_settle_s"])
    ep = Episode(node, cfg, data, m4a, sampler, arm, out, index, seed, writer)
    w0, s0 = time.monotonic(), node.sim_now()
    limit = demo["episode_timeout_s"] - (time.monotonic() - wall0)
    if not node.stream(names, plan.q, rate, ep.on_tick, wall_limit=limit):
        raise RuntimeError("episode timed out")
    node.spin_for(0.2)
    ep.close()
    return {"episode": index, "seed": seed, "split": ep.split, "placements": ep.n_place,
            "saved": ep.saved, "skipped": ep.skipped, "set_pose_failed": ep.set_pose_failed,
            "block_start": [round(v, 5) for v in plan.block_start],
            "rtf": round((node.sim_now() - s0) / max(time.monotonic() - w0, 1e-9), 3),
            "wall_s": round(time.monotonic() - wall0, 2)}  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M4b: labelled rod images from headless Gazebo")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "judge_sim.yaml")
    ap.add_argument("--episodes", type=int, help="override data.episodes")
    ap.add_argument("--start", type=int, default=0, help="first episode index")
    ap.add_argument("--name", default="gen", help="output folder suffix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    data = load_judge_sim(args.config)["data"]
    cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    m4a = load_m4a()
    n = args.episodes if args.episodes is not None else int(data["episodes"])
    idx = list(range(args.start, args.start + n))
    est = len(idx) * WALL_PER_EPISODE_S
    s0 = int(data["seed_base"])
    print(f"{len(idx)} episodes (seeds {s0 + idx[0]}..{s0 + idx[-1]}), about "
          f"{len(idx) * IMAGES_PER_EPISODE} images, estimated wall time {est / 60:.0f} min",
          flush=True)  # fmt: skip
    if args.dry_run:
        return 0
    out = new_run_dir(args.name, base=ROOT / data["out_dir"])
    (out / "images").mkdir()
    t_wall = time.monotonic()
    summary: dict = {"out": str(out.relative_to(ROOT)), "config": data, "git": git_state(),
                     "args": {k: str(v) for k, v in vars(args).items()},
                     "physics_engine": cfg["physics"].get("engine", "dartsim"),
                     "sim_config": {k: cfg[k] for k in ("table", "intruder", "camera", "robot")},
                     "m4a_path": m4a["path"], "episodes": []}  # fmt: skip
    launch = log = None
    code = 1
    node = None
    try:
        launch, log = start_launch(cfg, out)
        node = make_node(cfg, DataNode)
        planner = Planner(cfg)
        sampler = IntruderSampler(data, cfg, m4a)
        arm = ArmGeometry(cfg, m4a, robot=planner.robot)
        with (out / "labels.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for k, i in enumerate(idx):
                info = run_episode(node, planner, cfg, data, m4a, sampler, arm, out, i, writer)
                f.flush()
                summary["episodes"].append(info)
                print(f"[{k + 1}/{len(idx)}] ep {i} seed {info['seed']} ({info['split']}): "
                      f"saved {info['saved']}/{info['placements']}, skipped {info['skipped']}, "
                      f"rtf {info['rtf']}, wall {info['wall_s']} s", flush=True)  # fmt: skip
        code = 0
    except Exception as e:  # recorded; everything is still stopped below
        summary["error"] = f"{type(e).__name__}: {e}"
        print(summary["error"], file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if launch is not None:
            summary["shutdown"] = stop_group(launch, cfg["check"]["shutdown_timeout_s"])
        if log is not None:
            log.close()
        eps = summary["episodes"]
        summary["images"] = sum(e["saved"] for e in eps)
        summary["placements"] = sum(e["placements"] for e in eps)
        summary["skipped"] = {k: sum(e["skipped"][k] for e in eps)
                              for k in ("not_in_time", "pose_mismatch")}  # fmt: skip
        summary["leftover_processes"] = leftover_processes()
        summary["total_wall_s"] = round(time.monotonic() - t_wall, 1)
        (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary.get(k) for k in ("images", "placements", "skipped",
                                                       "leftover_processes", "error",
                                                       "total_wall_s")}, indent=2))  # fmt: skip
        print(f"-> {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
