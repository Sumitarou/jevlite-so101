"""M4b: where to put the rod for the judge's training images, and the label geometry (numpy only).

Used by scripts/sim/gen_intruder_data.py (system Python, ROS) and the evaluation (uv env).
Settings: configs/judge_sim.yaml `data`; table, rod and camera: configs/sim.yaml; the M4a path
(x of the rod's line, hover height): configs/m4a.yaml `path`.

A placement is one of
  park   the rod's parking pose (out of the image) -> none
  path   near the M4a path (x = path.x +- path_dx, +y half of the table)
  table  anywhere on the table around the band (up to none_margin outside it)
For path / table, a target zone is drawn by the `zones` weights, and with probability near_frac
the footprint must be within near_m of a zone boundary (the white area's edge or the band's
outer edge, whichever is closer). Candidates are drawn uniformly and rejected until they match,
show the rod's foot in the image and are not `forbidden` (block, arm, robot base).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from scripts.sim.demo_plan import surface_z
from scripts.sim.fit_camera import project
from scripts.sim.intruder import TOUCH_TOL, ZONES
from scripts.sim.kinematics import Robot, transform

ROOT = Path(__file__).resolve().parents[2]
JUDGE_SIM_YAML = ROOT / "configs" / "judge_sim.yaml"
KINDS = ("park", "path", "table")


def load_judge_sim(path: Path = JUDGE_SIM_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text())


def rect_signed(half_x: float, half_y: float, x: float, y: float) -> float:
    """Signed distance from a point to the rectangle |x| <= half_x, |y| <= half_y (< 0 inside)."""
    dx, dy = abs(x) - half_x, abs(y) - half_y
    return math.hypot(max(dx, 0.0), max(dy, 0.0)) + min(max(dx, dy), 0.0)


def footprint_distances(x: float, y: float, sim_cfg: dict) -> tuple[float, float]:
    """Signed distances of the rod's footprint (circle) to the white area and to the band's
    outer rectangle: <= 0 means overlapping (touching counts, as in intruder.zone_of)."""
    t = sim_cfg["table"]
    r = float(sim_cfg["intruder"]["radius"])
    wx, wy, b = t["white"][0] / 2, t["white"][1] / 2, t["band_width"]
    return rect_signed(wx, wy, x, y) - r, rect_signed(wx + b, wy + b, x, y) - r


def zone_from_distances(d_white: float, d_outer: float) -> str:
    if d_white <= TOUCH_TOL:
        return "center"
    if d_outer <= TOUCH_TOL:
        return "edge"
    return "none"


def boundary_distance(x: float, y: float, sim_cfg: dict) -> float:
    """How far the footprint is from changing zone [m] (the nearer of the two boundaries)."""
    dw, do = footprint_distances(x, y, sim_cfg)
    return min(abs(dw), abs(do))


def rod_center_z(sim_cfg: dict, m4a: dict) -> float:
    """z of the rod's centre above the white area (the M4a hover height)."""
    return float(m4a["path"]["hover"]) + float(sim_cfg["intruder"]["length"]) / 2


def foot_uv(x: float, y: float, sim_cfg: dict, m4a: dict) -> tuple[float, float, float]:
    """Image position (u, v) and depth of the rod's foot (bottom centre)."""
    c = sim_cfg["camera"]
    z = surface_z(sim_cfg) + float(m4a["path"]["hover"])  # camera z is above the table top
    u, v, d = project(np.array([[x, y, z]]), c["xyz"][0], c["xyz"][2], c["rpy"][1],
                      c["width"], c["height"], c["hfov"])  # fmt: skip
    return float(u[0, 0]), float(v[0, 0]), float(d[0, 0])


def foot_in_view(x: float, y: float, sim_cfg: dict, m4a: dict, margin_px: float = 4.0) -> bool:
    c = sim_cfg["camera"]
    u, v, d = foot_uv(x, y, sim_cfg, m4a)
    return (d > 0 and margin_px <= u <= c["width"] - margin_px
            and margin_px <= v <= c["height"] - margin_px)  # fmt: skip


def split_of(seed: int, split_cfg: dict) -> str:
    """train / val / test for an episode, from its seed only (stable when episodes are added)."""
    u = np.random.default_rng([int(split_cfg["seed"]), int(seed)]).random()
    if u < split_cfg["train"]:
        return "train"
    return "val" if u < split_cfg["train"] + split_cfg["val"] else "test"


class IntruderSampler:
    def __init__(self, data_cfg: dict, sim_cfg: dict, m4a: dict):
        self.s, self.sim, self.m4a = data_cfg["sampling"], sim_cfg, m4a
        t = sim_cfg["table"]
        self.ox = t["white"][0] / 2 + t["band_width"] + self.s["none_margin"]
        self.oy = t["white"][1] / 2 + t["band_width"] + self.s["none_margin"]
        self.z = rod_center_z(sim_cfg, m4a)
        self.park = tuple(float(v) for v in sim_cfg["intruder"]["park"])
        w = self.s["zones"]
        if set(w) != set(ZONES) or min(w.values()) < 0 or sum(w.values()) <= 0:
            raise ValueError(f"sampling.zones needs non-negative weights for {ZONES}, got {w}")
        self.zone_p = np.array([w[z] for z in ZONES], float) / sum(w.values())
        for k in ("park_frac", "path_frac", "near_frac"):
            if not 0.0 <= self.s[k] <= 1.0:
                raise ValueError(f"sampling.{k} must be in [0, 1], got {self.s[k]}")
        if self.s["park_frac"] + self.s["path_frac"] > 1.0:
            raise ValueError("sampling.park_frac + path_frac must be <= 1")

    def _candidate(self, rng: np.random.Generator, kind: str) -> tuple[float, float]:
        if kind == "path":
            px = float(self.m4a["path"]["x"])
            return (rng.uniform(px - self.s["path_dx"], px + self.s["path_dx"]),
                    rng.uniform(0.0, self.oy))  # fmt: skip
        return rng.uniform(-self.ox, self.ox), rng.uniform(-self.oy, self.oy)

    def sample(self, rng: np.random.Generator,
               forbidden: Callable[[float, float], bool] | None = None) -> dict:  # fmt: skip
        """One placement: {kind, target, near, x, y, z} (rod centre, table frame)."""
        u = rng.random()
        if u < self.s["park_frac"]:
            x, y, z = self.park
            return {"kind": "park", "target": "none", "near": False, "x": x, "y": y, "z": z}
        kind = "path" if u < self.s["park_frac"] + self.s["path_frac"] else "table"
        target = ZONES[int(rng.choice(len(ZONES), p=self.zone_p))]
        near = bool(rng.random() < self.s["near_frac"])
        for _ in range(int(self.s["max_tries"])):
            x, y = self._candidate(rng, kind)
            dw, do = footprint_distances(x, y, self.sim)
            if zone_from_distances(dw, do) != target:
                continue
            if near and min(abs(dw), abs(do)) > self.s["near_m"]:
                continue
            if not foot_in_view(x, y, self.sim, self.m4a):
                continue
            if forbidden is not None and forbidden(x, y):
                continue
            return {"kind": kind, "target": target, "near": near, "x": float(x), "y": float(y),
                    "z": self.z}  # fmt: skip
        raise RuntimeError(f"no {kind} placement for {target} (near={near}) "
                           f"in {self.s['max_tries']} tries")  # fmt: skip


class ArmGeometry:
    """Arm and robot-base points (mesh support points) in the table frame, for keeping the rod
    out of the robot and for the 'arm is near' subset of the evaluation."""

    def __init__(self, sim_cfg: dict, m4a: dict, robot: Robot | None = None):
        self.robot = robot or Robot()
        self.base = transform(sim_cfg["robot"]["xyz"], sim_cfg["robot"]["rpy"])
        self.r = float(sim_cfg["intruder"]["radius"])
        z0 = surface_z(sim_cfg) + float(m4a["path"]["hover"])
        self.z_range = (z0, z0 + float(sim_cfg["intruder"]["length"]))  # rod, above table top
        self.base_link = self.robot.root_link

    def points(self, joints: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """(moving-link points (M, 3), base points (K, 3)) for one joint state."""
        q = {name: np.array([float(joints.get(name, 0.0))]) for name in self.robot.limits}
        moving = [ln for ln in self.robot.vertices if ln != self.base_link]
        arm = self.robot.points(q, self.base, links=moving, kind="vertices")[0]
        base = self.robot.points(q, self.base, links=[self.base_link], kind="vertices")[0]
        return arm, base

    def clearance(self, pts: np.ndarray, x: float, y: float) -> float:
        """Horizontal distance from the rod's surface to the nearest point at the rod's heights
        (inf if no point is at those heights)."""
        lo, hi = self.z_range
        p = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= hi)]
        if len(p) == 0:
            return math.inf
        return float(np.min(np.hypot(p[:, 0] - x, p[:, 1] - y))) - self.r


def make_forbidden(arm: ArmGeometry, joints: dict[str, float], block_xy, s: dict):
    """forbidden(x, y) for one moment: too close to the block's centre, or inside the robot."""
    arm_pts, base_pts = arm.points(joints)
    pts = np.concatenate([arm_pts, base_pts])
    bx, by = block_xy

    def forbidden(x: float, y: float) -> bool:
        if math.hypot(x - bx, y - by) < s["block_clear_m"]:
            return True
        return arm.clearance(pts, x, y) < s["arm_clear_m"]

    return forbidden, arm_pts
