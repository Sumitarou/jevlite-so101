"""M3e robustness evaluation: conditions, block perturbations and metrics (numpy only, no ROS).

Used by scripts/sim/robustness.py (runs the episodes in Gazebo, system Python) and
scripts/sim/robustness_report.py (tables and figures, uv env). Settings: configs/robustness.yaml.

Conditions are written as offsets from `center`; nothing is drawn at random, so the same config
gives the same evaluation. Per tick, robustness.py records the joint state (5 arm joints) and the
block pose (x, y, yaw, table frame); the TCP comes from forward kinematics here, with the same
robot model and TCP as the demo planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import yaml

from scripts.sim.demo_plan import ARM, half_extent, surface_z
from scripts.sim.kinematics import ROOT, SIM_YAML, Robot, tool_pose, transform

ROBUSTNESS_YAML = ROOT / "configs" / "robustness.yaml"
SUITES = ("start", "perturb")


def load_configs(path=ROBUSTNESS_YAML, sim_path=SIM_YAML) -> tuple[dict, dict]:
    return yaml.safe_load(open(path).read()), yaml.safe_load(open(sim_path).read())


@dataclass
class Condition:
    id: str  # unique within a suite (also names the bag)
    suite: str
    name: str
    start: tuple[float, float, float]  # block start x, y [m], yaw [rad]
    n_action_steps: int
    perturbation: dict | None = None  # name, t_s, dx, dyaw_deg, phase (perturb suite)
    repeat: int = 1
    outside: dict = field(default_factory=dict)  # x / y / yaw: outside the demo range

    def meta(self) -> dict:
        return {"condition": self.id, "suite": self.suite, "name": self.name,
                "n_action_steps": self.n_action_steps, "perturbation": self.perturbation,
                "repeat": self.repeat, "outside_demo_range": self.outside}  # fmt: skip


def outside_demo_range(start, demo: dict, eps: float = 1e-9) -> dict:
    x, y, yaw = start
    return {
        "x": not (demo["start_x"][0] - eps <= x <= demo["start_x"][1] + eps),
        "y": not (demo["start_y"][0] - eps <= y <= demo["start_y"][1] + eps),
        "yaw": abs(math.degrees(yaw)) > demo["start_yaw_deg"] + eps,
    }


def _start(center: dict, c: dict) -> tuple[float, float, float]:
    return (
        float(center["x"] + c.get("dx", 0.0)),
        float(center["y"] + c.get("dy", 0.0)),
        float(math.radians(center["yaw_deg"] + c.get("dyaw_deg", 0.0))),
    )


def build_conditions(rcfg: dict, sim_cfg: dict, suite: str) -> list[Condition]:
    """The suite's episodes in run order (grouped by n_action_steps, so the policy server is
    restarted as few times as possible)."""
    center, demo, reps = rcfg["center"], sim_cfg["demo"], int(rcfg.get("repeats", 1))
    c0 = _start(center, {})
    out: list[Condition] = []

    def add(cid, name, start, n, pert=None):
        for r in range(1, reps + 1):
            out.append(Condition(cid if r == 1 else f"{cid}_r{r}", suite, name, start, int(n),
                                 pert, r, outside_demo_range(start, demo)))  # fmt: skip

    if suite == "start":
        s = rcfg["start_suite"]
        for c in s["conditions"]:
            add(c["name"], c["name"], _start(center, c), s["n_action_steps"])
    elif suite == "perturb":
        s = rcfg["perturb_suite"]
        for n in s["n_action_steps"]:
            if n in s.get("baseline_n_action_steps", []):
                add(f"baseline_n{n}", "baseline", c0, n)
            for p in s["perturbations"]:
                pert = {
                    "name": p["name"],
                    "t_s": float(p["t_s"]),
                    "dx": float(p.get("dx", 0.0)),
                    "dyaw_deg": float(p.get("dyaw_deg", 0.0)),
                    "phase": p.get("phase"),
                }
                add(f"{p['name']}_n{n}", p["name"], c0, n, pert)
    else:
        raise ValueError(f"unknown suite {suite!r} ({', '.join(SUITES)})")
    ids = [c.id for c in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate condition ids in suite {suite}")
    return out


def estimate_wall_s(n_episodes: int, duration_s: float, rtf: float = 0.36,
                    per_episode_s: float = 10.0, per_server_s: float = 15.0,
                    n_servers: int = 1) -> float:  # fmt: skip
    """Rough wall time: M3d ran 15 s episodes in about 52 s (RTF 0.36 plus the reset)."""
    return n_episodes * (duration_s / rtf + per_episode_s) + n_servers * per_server_s + 30.0


# ---------- block perturbation ----------


def yaw_of(qx, qy, qz, qw) -> float:
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def perturbed_pose(cur, dx: float, dyaw: float, size_xy, y_clearance: float):
    """New block pose (x, y, yaw) and the +y shift used. A shift in x keeps y and yaw. A turn
    would push a corner into the jaws behind the block (on its -y side), so the block is moved
    forward by the growth of its half extent along y, plus y_clearance."""
    x, y, yaw = cur
    new_yaw = yaw + dyaw
    dy = 0.0
    if dyaw != 0.0:
        grow = half_extent(size_xy, new_yaw - math.pi / 2) - half_extent(size_xy, yaw - math.pi / 2)
        dy = max(0.0, grow) + y_clearance
    return (x + dx, y + dy, new_yaw), dy


def block_corners(pose, size_xy) -> np.ndarray:
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = size_xy[0] / 2, size_xy[1] / 2
    local = np.array([[hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy]])
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


# ---------- metrics ----------


class TcpFK:
    """TCP position (table frame, z above the white area) from the 5 arm joints."""

    def __init__(self, sim_cfg: dict, robot: Robot | None = None):
        self.robot = robot or Robot()
        self.rc = sim_cfg["reach_check"]
        self.base = transform(sim_cfg["robot"]["xyz"], sim_cfg["robot"]["rpy"])
        self.z0 = surface_z(sim_cfg)
        self.gripper = float(self.rc["fixed"]["gripper"])

    def __call__(self, q5: np.ndarray) -> np.ndarray:
        q5 = np.atleast_2d(np.asarray(q5, float))
        jv = {"gripper": np.full(len(q5), self.gripper), **{j: q5[:, i] for i, j in enumerate(ARM)}}
        pos, _ = tool_pose(self.robot, jv, self.base, self.rc["tip_link"], self.rc["tcp"])
        return pos - [0.0, 0.0, self.z0]


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def episode_metrics(ticks: list[dict], fk: TcpFK, sim_cfg: dict, mcfg: dict) -> dict:
    """Tracking metrics of one episode from its ticks (each with `t`, `state`, `block`,
    `inferred`, maybe `event`)."""
    t = np.array([k["t"] for k in ticks], float)
    tcp = fk(np.array([k["state"] for k in ticks], float))
    blk = np.array([k["block"] for k in ticks], float)
    gap = tcp[:, 0] - blk[:, 0]  # TCP x - block x
    behind = blk[:, 1] - tcp[:, 1]
    win = (tcp[:, 2] < mcfg["push_z"]) & (behind >= 0) & (behind <= mcfg["push_behind"])
    size = sim_cfg["block"]["size"]
    jaw = sim_cfg["demo"]["jaw_half_thickness"]
    c, s = np.cos(blk[:, 2]), np.sin(blk[:, 2])
    dx, dy = tcp[:, 0] - blk[:, 0], tcp[:, 1] - blk[:, 1]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    in_foot = ((np.abs(lx) < size[0] / 2 + jaw) & (np.abs(ly) < size[1] / 2 + jaw)
               & (tcp[:, 2] < size[2] + jaw))  # fmt: skip
    ev = [k for k in ticks if "event" in k]
    # moved by the arm: with a set_pose move, the jump itself (from the tick before the move to
    # perturb_check_ticks after it) is left out
    spans = [(0, len(blk) - 1)]
    if ev:
        i0 = ev[0]["i"]
        ic = min(i0 + int(mcfg["perturb_check_ticks"]), len(blk) - 1)
        spans = [(0, max(i0 - 1, 0)), (ic, len(blk) - 1)]
    moved = sum(float(np.hypot(*(blk[b, :2] - blk[a, :2]))) for a, b in spans)
    dyaw = sum(math.remainder(blk[b, 2] - blk[a, 2], 2 * math.pi) for a, b in spans)
    out = {
        "push_ticks": int(win.sum()),
        "push_t": [_r(t[win][0], 3), _r(t[win][-1], 3)] if win.any() else None,
        "tcp_x_low": _r(tcp[win, 0].mean()) if win.any() else None,
        "gap_mean": _r(gap[win].mean()) if win.any() else None,
        "gap_abs_mean": _r(np.abs(gap[win]).mean()) if win.any() else None,
        "gap_abs_max": _r(np.abs(gap[win]).max()) if win.any() else None,
        "tcp_min_height": _r(tcp[:, 2].min()),
        "block_moved_m": _r(moved),
        "block_yaw_change_deg": _r(math.degrees(dyaw), 2),
        "touched_footprint": bool(in_foot.any()),
        "touched": bool(moved > mcfg["touch_move"] or in_foot.any()),
    }
    if ev:
        out["perturb"] = perturb_metrics(ticks, t, gap, win, ev[0], mcfg)
    return out


def perturb_metrics(ticks, t, gap, win, ev_tick, mcfg) -> dict:
    e, i0 = ev_tick["event"], ev_tick["i"]
    te = t[i0]
    inf = [k["t"] for k in ticks[i0:] if k["inferred"]]
    ic = min(i0 + int(mcfg["perturb_check_ticks"]), len(ticks) - 1)
    b = ticks[ic]["block"]
    err_x = abs(b[0] - e["to"][0])
    err_yaw = abs(math.degrees(math.remainder(b[2] - e["to"][2], 2 * math.pi)))
    ok = bool(e.get("set_pose_ok", True) and err_x <= mcfg["perturb_tol"]
              and err_yaw <= mcfg["perturb_yaw_tol_deg"])  # fmt: skip
    # the block pose in the ticks right after the move may still be the old one: look from ic on
    back = np.nonzero(np.abs(gap[ic:]) <= mcfg["reaim_tol"])[0]
    after = np.nonzero(win[ic:])[0]
    return {
        "name": e.get("name"),
        "t_event": _r(te, 3),
        "first_inference_after_s": _r(inf[0] - te, 3) if inf else None,
        "perturb_ok": ok,
        "perturb_err_x": _r(err_x),
        "perturb_err_yaw_deg": _r(err_yaw, 2),
        "gap_after": _r(abs(gap[ic])),
        "reaim_s": _r(t[ic + back[0]] - te, 3) if len(back) else None,
        "gap_end": _r(abs(gap[ic + after[-1]])) if len(after) else None,
        "push_ticks_after": int(len(after)),
    }


def _x_points(rows: list[dict]) -> list[tuple[float, float]]:
    """(block start x, TCP x while pushing) of the start conditions that vary only x."""
    return [(r["block_start"][0], r["metrics"]["tcp_x_low"]) for r in rows
            if r.get("name") == "center" or str(r.get("name", "")).startswith("x")
            if r.get("metrics", {}).get("tcp_x_low") is not None]  # fmt: skip


def _fit(pts) -> dict | None:
    if len(pts) < 2:
        return None
    x, y = np.array(pts).T
    k, c = np.polyfit(x, y, 1)
    return {"slope": round(float(k), 3), "intercept_m": round(float(c), 4), "points": len(pts),
            "x_range_m": [round(float(x.min()), 4), round(float(x.max()), 4)]}  # fmt: skip


def x_slope(rows: list[dict]) -> dict | None:
    """Fit TCP x while pushing against the block's start x over the start conditions that vary
    only x: slope 1 = follows the block, 0 = replays one path regardless of the block."""
    return _fit(_x_points(rows))


def x_slopes(rows: list[dict], demo: dict, eps: float = 1e-9) -> dict:
    """The same fit over all points, inside the demo's start_x range (edges included), and on
    each side outside it. A policy that follows the block only within its demos gives a slope
    near 1 inside and near 0 outside, which one line over all points hides."""
    lo, hi = demo["start_x"]
    pts = _x_points(rows)
    return {
        "all": _fit(pts),
        "inside": _fit([p for p in pts if lo - eps <= p[0] <= hi + eps]),
        "outside_minus": _fit([p for p in pts if p[0] < lo - eps]),
        "outside_plus": _fit([p for p in pts if p[0] > hi + eps]),
    }
