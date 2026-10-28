"""M4a intruder (a vertical rod): its scripted path and the zone it is in (numpy only, no ROS).

Used by scripts/sim/gate_demo_sim.py (moves the rod with set_pose each tick, system Python), the
oracle judge (scripts/sim/sim_judge.py) and the report. Settings: configs/m4a.yaml (path,
scenarios, zone -> choice) and configs/sim.yaml (intruder model, table areas).

Zone of the rod from its footprint, a circle of radius intruder.radius on the table:
  center  the circle overlaps the white area (protected)
  edge    otherwise, it overlaps the green band's outer rectangle (warning)
  none    otherwise
Positions are in the table frame (origin at the centre of the white area, z above the white
area's top surface).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from jevlite.api import Choice

ROOT = Path(__file__).resolve().parents[2]
M4A_YAML = ROOT / "configs" / "m4a.yaml"
ZONES = ("none", "edge", "center")
TOUCH_TOL = 1e-9  # [m] a footprint this close to an area counts as touching it


def load_m4a(path: Path = M4A_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text())


def rect_circle_overlap(half_x: float, half_y: float, x: float, y: float, r: float) -> bool:
    """Does a circle (x, y, r) overlap the rectangle |x| <= half_x, |y| <= half_y?
    Touching counts as overlap (safe side), with TOUCH_TOL for rounding."""
    dx = max(abs(x) - half_x, 0.0)
    dy = max(abs(y) - half_y, 0.0)
    return math.hypot(dx, dy) <= r + TOUCH_TOL


def zone_of(x: float, y: float, sim_cfg: dict) -> str:
    t = sim_cfg["table"]
    r = float(sim_cfg["intruder"]["radius"])
    wx, wy = t["white"][0] / 2, t["white"][1] / 2
    if rect_circle_overlap(wx, wy, x, y, r):
        return "center"
    b = t["band_width"]
    if rect_circle_overlap(wx + b, wy + b, x, y, r):
        return "edge"
    return "none"


def edge_distance(x: float, y: float, sim_cfg: dict) -> float:
    """Signed distance from the rod's footprint to the white area [m]: < 0 overlapping it,
    0 .. band_width over the band (roughly), larger outside. Used for plotting."""
    t = sim_cfg["table"]
    r = float(sim_cfg["intruder"]["radius"])
    wx, wy = t["white"][0] / 2, t["white"][1] / 2
    dx, dy = abs(x) - wx, abs(y) - wy
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside - r


def choice_of(zone: str, m4a: dict) -> Choice:
    return Choice.parse(m4a["zones"][zone])


@dataclass
class Scenario:
    id: str
    desc: str
    keys: list[tuple[float, str]]  # (t, place name)
    judge: dict  # the judge settings (defaults merged with the scenario's overrides)
    points: dict[str, tuple[float, float, float]] = field(repr=False)  # place -> (x, y, z)
    park: tuple[float, float, float] = (0.0, 0.0, 0.0)
    duration_s: float | None = None  # this scenario's length (None: the config's duration_s)

    @property
    def has_intruder(self) -> bool:
        return bool(self.keys)

    def position(self, t: float) -> tuple[float, float, float]:
        """Rod centre (x, y, z) at time t: linear between keys, held before / after them;
        the park pose when there are no keys."""
        if not self.keys:
            return self.park
        ts = [k[0] for k in self.keys]
        pts = np.array([self.points[k[1]] for k in self.keys])
        if t <= ts[0]:
            return tuple(float(v) for v in pts[0])
        if t >= ts[-1]:
            return tuple(float(v) for v in pts[-1])
        return tuple(float(np.interp(t, ts, pts[:, j])) for j in range(3))

    def zone(self, t: float, sim_cfg: dict) -> str:
        if not self.keys:
            return "none"
        x, y, _ = self.position(t)
        return zone_of(x, y, sim_cfg)

    def meta(self) -> dict:
        return {"scenario": self.id, "desc": self.desc, "keys": [list(k) for k in self.keys],
                "judge": self.judge, "duration_s": self.duration_s}  # fmt: skip


def build_scenarios(m4a: dict, sim_cfg: dict) -> list[Scenario]:
    p, itr = m4a["path"], sim_cfg["intruder"]
    z = float(p["hover"]) + float(itr["length"]) / 2  # centre of the rod
    points = {name: (float(p["x"]), float(y), z) for name, y in p["places"].items()}
    park = tuple(float(v) for v in itr["park"])
    out, seen = [], set()
    for s in m4a["scenarios"]:
        if s["id"] in seen:
            raise ValueError(f"duplicate scenario id {s['id']}")
        seen.add(s["id"])
        keys = [(float(t), str(name)) for t, name in s.get("keys", [])]
        if any(b[0] <= a[0] for a, b in zip(keys, keys[1:], strict=False)):
            raise ValueError(f"{s['id']}: key times must increase")
        missing = [k[1] for k in keys if k[1] not in points]
        if missing:
            raise ValueError(f"{s['id']}: unknown places {missing}")
        judge = {**m4a["judge"], **(s.get("judge") or {})}
        dur = s.get("duration_s")
        out.append(Scenario(s["id"], s.get("desc", ""), keys, judge, points, park,
                            None if dur is None else float(dur)))  # fmt: skip
    return out
