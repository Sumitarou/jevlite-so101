"""Plan one push demonstration from a random seed (numpy only, no ROS).

    uv run python scripts/sim/demo_plan.py --seed 0     # print a summary of the plan

Settings: configs/sim.yaml `demo` (and `reach_check`, `target_frame`, `block`, `robot`, `table`).
All heights are TCP heights above the white area's top.

1. The seed draws the block's start (centre x, y in `start_x` / `start_y`, yaw within
   +-`start_yaw_deg`) with numpy's default_rng(seed).
2. The push goes straight from the block's centre towards the target frame's centre. The push
   starts `pre_push_gap` before the block's face (the block's half extent along the push,
   which depends on its yaw, + the jaw's half thickness) and ends when the block's centre would
   be on the frame's centre (the face then lies flat against the jaws).
3. TCP path: home -> above the start (travel height) -> approach height -> push height ->
   (pause) push -> (pause) lift -> (settle) -> home. Straight lines at the configured speeds,
   sampled at `rate_hz`.
4. Joint angles by IK (kinematics.dls) with the tool pointing down (leaning away from the base
   only where needed, up to reach_check.max_tilt_deg) and the jaw line perpendicular to the
   push. Each segment is solved in one batch, warm-started from the previous tick.
5. Checks: IK errors, joint limits, joint step per tick, the arm above the white area, and the
   arm away from the block before the push (start pose) and after it (target pose). A start
   that fails is redrawn from the same random stream (up to `max_attempts`).
The plan is open loop: the commands do not depend on what the simulator does.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

import numpy as np
import yaml

from scripts.sim.kinematics import (
    SIM_YAML,
    Robot,
    dls,
    tool_errors,
    tool_pose,
    tool_residual,
    transform,
)

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class PlanError(RuntimeError):
    pass


@dataclass
class Plan:
    seed: int
    attempts: int
    block_start: tuple[float, float, float]  # x, y, yaw
    target: tuple[float, float]
    push_dir: tuple[float, float]
    joint_names: list[str]
    q: np.ndarray  # (T, 6) joint commands at rate_hz, in joint_names order
    tcp: np.ndarray  # (T, 3) planned TCP (table frame, z above the table top)
    phases: list[tuple[str, int, int]] = field(default_factory=list)  # name, first, end tick
    rate_hz: float = 30.0

    @property
    def duration_s(self) -> float:
        return len(self.q) / self.rate_hz

    def summary(self) -> dict:
        return {
            "seed": self.seed,
            "attempts": self.attempts,
            "block_start": [round(v, 5) for v in self.block_start],
            "target": list(self.target),
            "push_dir": [round(v, 5) for v in self.push_dir],
            "ticks": len(self.q),
            "duration_s": round(self.duration_s, 3),
            "phases": [[n, a, b] for n, a, b in self.phases],
        }


def surface_z(cfg) -> float:
    return 2 * cfg["table"]["overlay_thickness"]


def half_extent(size_xy, yaw_rel: float) -> float:
    """Half extent of a box (x, y sizes) along a direction at yaw_rel from its x axis."""
    return 0.5 * (size_xy[0] * abs(np.cos(yaw_rel)) + size_xy[1] * abs(np.sin(yaw_rel)))


def sample_start(rng: np.random.Generator, demo: dict) -> tuple[float, float, float]:
    x = rng.uniform(*demo["start_x"])
    y = rng.uniform(*demo["start_y"])
    yaw = np.radians(rng.uniform(-demo["start_yaw_deg"], demo["start_yaw_deg"]))
    return float(x), float(y), float(yaw)


def push_geometry(cfg, start) -> dict:
    """Push direction (demo.push_aim), the TCP's start / end points (x, y) and the block's end
    point for a block start (x, y, yaw)."""
    demo, blk = cfg["demo"], cfg["block"]
    c = np.array(start[:2])
    t = np.asarray(cfg["target_frame"]["center"], dtype=float)
    aim = demo.get("push_aim", "center")
    if aim == "straight_y":  # along +y only: the block keeps its x
        t = np.array([c[0], t[1]])
    elif aim != "center":
        raise ValueError(f"unknown demo.push_aim: {aim!r} (straight_y / center)")
    d = t - c
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        raise PlanError("the block starts on the target")
    d /= dist
    yaw_rel = start[2] - np.arctan2(d[1], d[0])
    back = half_extent(blk["size"], yaw_rel) + demo["jaw_half_thickness"]
    pre = c - d * (back + demo["pre_push_gap"])
    end = t - d * (blk["size"][1] / 2 + demo["jaw_half_thickness"])  # face flat on the jaws
    return {"dir": d, "pre": pre, "end": end, "distance": dist, "target": t}


def _line(p0, p1, speed, rate) -> np.ndarray:
    """Points from p0 (excluded) to p1 (included) at `speed`, one per tick."""
    n = max(1, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / speed * rate)))
    s = np.arange(1, n + 1)[:, None] / n
    return np.asarray(p0) + s * (np.asarray(p1) - np.asarray(p0))


def _hold(p, seconds, rate) -> np.ndarray:
    return np.repeat(np.asarray(p)[None], max(1, int(round(seconds * rate))), axis=0)


def tcp_path(cfg, geo, home_tcp) -> tuple[np.ndarray, np.ndarray, list]:
    """TCP points (T, 3, table frame), jaw directions (T, 2) and phases."""
    demo = cfg["demo"]
    rate, z0 = demo["rate_hz"], cfg["table"]["height"] * 0 + surface_z(cfg)
    pre, end, d = geo["pre"], geo["end"], geo["dir"]

    def at(xy, h):
        return np.array([xy[0], xy[1], z0 + h])

    home = np.asarray(home_tcp, dtype=float) + [0, 0, z0]
    push_h = cfg["reach_check"]["tip_height"]
    segs = [
        ("to_start", _line(home, at(pre, demo["travel_height"]), demo["speed_travel"], rate)),
        (
            "descend",
            _line(
                at(pre, demo["travel_height"]),
                at(pre, demo["approach_height"]),
                demo["speed_descend"],
                rate,
            ),
        ),  # fmt: skip
        (
            "descend",
            _line(
                at(pre, demo["approach_height"]), at(pre, push_h), demo["speed_descend"] / 2, rate
            ),
        ),  # fmt: skip
        ("before_push", _hold(at(pre, push_h), demo["pause_before_push"], rate)),
        ("push", _line(at(pre, push_h), at(end, push_h), demo["speed_push"], rate)),
        ("after_push", _hold(at(end, push_h), demo["pause_after_push"], rate)),
        ("lift", _line(at(end, push_h), at(end, demo["lift_height"]), demo["speed_descend"], rate)),
        ("settle", _hold(at(end, demo["lift_height"]), demo["settle_after_lift"], rate)),
        ("to_home", _line(at(end, demo["lift_height"]), home, demo["speed_travel"], rate)),
    ]
    pts, phases, i = [], [], 0
    for name, p in segs:
        pts.append(p)
        if phases and phases[-1][0] == name:
            phases[-1] = (name, phases[-1][1], i + len(p))
        else:
            phases.append((name, i, i + len(p)))
        i += len(p)
    pts = np.concatenate(pts)
    # jaw direction: +y at home, the push direction from above the start until the lift,
    # turning smoothly on the way there and back
    n0 = phases[0][2]
    nh = phases[-1][2] - phases[-1][1]
    home_dir = np.array([0.0, 1.0])
    dirs = np.repeat(d[None], len(pts), axis=0)
    a0, a1 = np.arctan2(home_dir[1], home_dir[0]), np.arctan2(d[1], d[0])
    s = np.arange(1, n0 + 1) / n0
    dirs[:n0] = np.stack([np.cos(a0 + s * (a1 - a0)), np.sin(a0 + s * (a1 - a0))], axis=1)
    s = np.arange(1, nh + 1) / nh
    dirs[-nh:] = np.stack([np.cos(a1 + s * (a0 - a1)), np.sin(a1 + s * (a0 - a1))], axis=1)
    return pts, dirs, phases


class Planner:
    """Holds the robot model and the solved home pose; plans episodes from seeds."""

    def __init__(self, cfg: dict, robot: Robot | None = None):
        self.cfg = cfg
        self.robot = robot or Robot()
        self.rc = cfg["reach_check"]
        self.demo = cfg["demo"]
        self.base = transform(cfg["robot"]["xyz"], cfg["robot"]["rpy"])
        self.lo = np.array([self.robot.limits[j][0] for j in ARM])
        self.hi = np.array([self.robot.limits[j][1] for j in ARM])
        self.gripper = float(self.rc["fixed"]["gripper"])
        self.q_home = self._solve_home()

    # ---------- IK ----------
    def _pose(self, q):
        jv = {"gripper": np.full(len(q), self.gripper), **{j: q[:, i] for i, j in enumerate(ARM)}}
        return tool_pose(self.robot, jv, self.base, self.rc["tip_link"], self.rc["tcp"])

    def _lean(self, pts):
        away = pts[:, :2] - self.base[:2, 3]
        return away / np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-12)

    def _solve(self, pts, dirs, q0, iters):
        lean = self._lean(pts)

        def residual(q):
            return tool_residual(*self._pose(q), pts, dirs, self.rc["w_rot"], 0.0, lean,
                                 self.rc["w_vertical"])  # fmt: skip

        return dls(residual, q0, self.lo, self.hi, iters, self.rc["damping"])

    def errors(self, q, pts, dirs) -> dict:
        return tool_errors(*self._pose(q), pts, dirs, 0.0, self._lean(pts))

    def _solve_home(self) -> np.ndarray:
        p = np.asarray(self.demo["home_tcp"], dtype=float)[None] + [0, 0, surface_z(self.cfg)]
        d = np.array([[0.0, 1.0]])
        rng = np.random.default_rng(self.rc["seed"])
        q0 = rng.uniform(self.lo, self.hi, size=(64, len(ARM)))
        q = self._solve(np.repeat(p, 64, 0), np.repeat(d, 64, 0), q0, 150)
        e = self.errors(q, np.repeat(p, 64, 0), np.repeat(d, 64, 0))
        ok = (e["pos"] < self.rc["pos_tol"]) & (e["jaw"] < np.radians(self.rc["rot_tol_deg"]))
        if not ok.any():
            raise PlanError("no IK solution for the home pose")
        i = int(np.argmin(np.where(ok, e["tilt"], np.inf)))
        return q[i]

    def track(self, pts, dirs, q_start) -> np.ndarray:
        """IK along a path: all ticks from q_start, then refined from each previous tick."""
        q = np.repeat(q_start[None], len(pts), 0)
        q = self._solve(pts, dirs, q, self.demo["ik_iters"])
        for _ in range(3):
            seed = np.concatenate([q_start[None], q[:-1]])
            q = self._solve(pts, dirs, seed, self.demo["ik_iters"] // 2)
        return q

    def first_command(self, start) -> np.ndarray:
        """The first command (6 joints, plan joint order) of the plan for a given block start
        (x, y, yaw): the pose eval_policy.py moves the arm to before an episode. No checks, so
        it also works for starts outside the demo range (M3e). Each tick's IK depends only on
        its own point and q_home, so this equals plan(seed).q[0] for the same start."""
        geo = push_geometry(self.cfg, start)
        pts, dirs, _ = tcp_path(self.cfg, geo, self.demo["home_tcp"])
        q = self.track(pts[:1], dirs[:1], self.q_home)
        return np.append(q[0], self.gripper)

    # ---------- checks ----------
    def _arm_points(self, q):
        jv = {"gripper": np.full(len(q), self.gripper), **{j: q[:, i] for i, j in enumerate(ARM)}}
        links = set(self.robot.vertices) - {self.robot.root_link}
        return self.robot.points(jv, self.base, links=links, kind="vertices")

    def _in_block(self, pts, pose, margin) -> np.ndarray:
        """(T,) True where any arm point lies inside the block (expanded by margin)."""
        x, y, yaw = pose
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = pts[..., 0] - x, pts[..., 1] - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        sx, sy, sz = (v / 2 + margin for v in self.cfg["block"]["size"])
        z0 = surface_z(self.cfg)
        inside = (np.abs(lx) < sx) & (np.abs(ly) < sy) & (pts[..., 2] < z0 + 2 * sz - margin)
        return inside.any(axis=1)

    def check(self, q, pts, dirs, phases, start, target_pose) -> list[str]:
        problems = []
        e = self.errors(q, pts, dirs)
        if (e["pos"] > self.rc["pos_tol"]).any():
            problems.append(f"TCP error up to {e['pos'].max() * 1000:.1f} mm")
        if (e["tilt"] > np.radians(self.rc["max_tilt_deg"])).any():
            problems.append(f"tool tilt up to {np.degrees(e['tilt'].max()):.1f} deg")
        if (e["jaw"] > np.radians(self.rc["rot_tol_deg"])).any():
            problems.append(f"jaw line off by up to {np.degrees(e['jaw'].max()):.1f} deg")
        steps = np.abs(np.diff(np.concatenate([self.q_home[None], q]), axis=0)).max()
        if steps > self.demo["max_joint_step"]:
            problems.append(f"joint step {steps:.3f} rad per tick")
        arm = self._arm_points(q)
        low = arm[..., 2].min(1)
        if (low < surface_z(self.cfg) + self.rc["clearance"] - 1e-9).any():
            problems.append(
                f"arm {(surface_z(self.cfg) - low.min()) * 1000:.1f} mm into the white area"
            )
        ph = {n: (a, b) for n, a, b in phases}
        before = slice(0, ph["before_push"][1])
        after = slice(ph["settle"][0], len(q))  # during the lift the jaws still touch the face
        m = self.demo["block_clearance"]
        if self._in_block(arm[before], start, m).any():
            problems.append("arm touches the block before the push")
        if self._in_block(arm[after], target_pose, m).any():
            problems.append("arm touches the block after the push")
        return problems

    # ---------- episode ----------
    def plan(self, seed: int) -> Plan:
        rng = np.random.default_rng(seed)
        last = []
        for attempt in range(1, self.demo["max_attempts"] + 1):
            start = sample_start(rng, self.demo)
            geo = push_geometry(self.cfg, start)
            pts, dirs, phases = tcp_path(self.cfg, geo, self.demo["home_tcp"])
            q = self.track(pts, dirs, self.q_home)
            t = geo["target"]  # where the block's centre should end up
            end_yaw = float(np.arctan2(geo["dir"][1], geo["dir"][0]))  # face flat on the jaws
            last = self.check(q, pts, dirs, phases, start, (t[0], t[1], end_yaw))
            if not last:
                q6 = np.concatenate([q, np.full((len(q), 1), self.gripper)], axis=1)
                return Plan(seed, attempt, start, (float(t[0]), float(t[1])),
                            (float(geo["dir"][0]), float(geo["dir"][1])), ARM + ["gripper"],
                            q6, pts, phases, float(self.demo["rate_hz"]))  # fmt: skip
        raise PlanError(f"seed {seed}: no plan in {self.demo['max_attempts']} attempts ({last})")


def judge(cfg, pose) -> dict:
    """Success from the block's exact pose (x, y, z, qx, qy, qz, qw), table frame
    (z above the table top)."""
    x, y, z, qx, qy, qz, qw = pose
    tf, demo = cfg["target_frame"], cfg["demo"]["success"]
    half = tf["inner"] / 2
    dx, dy = x - tf["center"][0], y - tf["center"][1]
    z_expected = surface_z(cfg) + cfg["block"]["size"][2] / 2
    # tilt: angle between the block's z axis and the vertical
    zz = 1 - 2 * (qx * qx + qy * qy)
    tilt = float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))
    inside = abs(dx) <= half and abs(dy) <= half
    on_table = abs(z - z_expected) <= demo["z_tol"]
    upright = tilt < demo["max_tilt_deg"]
    return {
        "success": bool(inside and on_table and upright),
        "inside": bool(inside), "on_table": bool(on_table), "upright": bool(upright),
        "error_xy": [round(float(dx), 4), round(float(dy), 4)],
        "error_m": round(float(np.hypot(dx, dy)), 4),
        "yaw_deg": round(yaw, 2), "tilt_deg": round(tilt, 2),
    }  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(SIM_YAML.read_text())
    try:
        plan = Planner(cfg).plan(args.seed)
    except (PlanError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps(plan.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
