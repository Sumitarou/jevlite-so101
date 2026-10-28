"""Forward kinematics of the SO-101 URDF and STL helpers (numpy only, no ROS).

Used by scripts/sim/fit_camera.py and scripts/sim/check_reach.py. Joint values are given as
name -> (N,) arrays, so N configurations are evaluated at once.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM_YAML = ROOT / "configs" / "sim.yaml"
DESC = ROOT / "sim_ws" / "src" / "so101_description"
URDF = DESC / "urdf" / "so101_new_calib.urdf"
MESH_PREFIX = "package://so101_description/meshes/"
FETCH_HINT = "meshes are missing: run `python3 scripts/sim/fetch_so101_assets.py` first"


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF / SDF fixed-axis roll-pitch-yaw: R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])  # fmt: skip


def transform(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def axis_angle(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(N,) angles about a unit axis -> (N, 4, 4) rotations (Rodrigues)."""
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    out = np.tile(np.eye(4), (len(q), 1, 1))
    out[:, :3, :3] = np.eye(3) + s * kx + (1 - c) * (kx @ kx)
    return out


def stl_vertices(path: Path) -> np.ndarray:
    """(K, 3) unique vertices of a binary or ASCII STL."""
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    if len(data) == 84 + 50 * n:
        rec = np.frombuffer(data, dtype=np.uint8, offset=84).reshape(n, 50)
        v = rec[:, 12:48].copy().view("<f4").reshape(-1, 3).astype(float)
    else:
        nums = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
        v = np.array(nums, dtype=float)
    return np.unique(v, axis=0)


def stl_bbox(path: Path) -> np.ndarray:
    """(2, 3) min/max of the vertices of an STL."""
    v = stl_vertices(path)
    return np.stack([v.min(0), v.max(0)])


def support_points(v: np.ndarray, n_dirs: int) -> np.ndarray:
    """Vertices that are extreme along `n_dirs` directions spread over the sphere (Fibonacci).

    They lie on the convex hull, so the lowest point of a link (min of a linear function) is
    found among them, up to the gaps between the directions."""
    i = np.arange(n_dirs) + 0.5
    z = 1 - 2 * i / n_dirs
    r = np.sqrt(1 - z**2)
    phi = np.pi * (1 + 5**0.5) * i
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    return v[np.unique(np.argmax(v @ dirs.T, axis=0))]


def box_corners(bb: np.ndarray) -> np.ndarray:
    """(2, 3) min/max -> (8, 3) corners."""
    return np.array(
        [[bb[i, 0], bb[j, 1], bb[k, 2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    )


def _floats(el, attr) -> list[float]:
    return [float(x) for x in el.get(attr, "0 0 0").split()] if el is not None else [0.0] * 3


class Robot:
    """Kinematic tree of a URDF. Per link: visual bounding-box corners and the meshes' support
    points (convex-hull vertices along `n_dirs` directions), both in the link frame."""

    def __init__(self, urdf: Path = URDF, mesh_dir: Path = DESC / "meshes", n_dirs: int = 600):
        root = ET.parse(urdf).getroot()
        self.joints = []
        for j in root.findall("joint"):
            o, lim, ax = j.find("origin"), j.find("limit"), j.find("axis")
            self.joints.append({
                "name": j.get("name"),
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "T": transform(_floats(o, "xyz"), _floats(o, "rpy")),
                "axis": np.array(_floats(ax, "xyz")) if ax is not None else np.zeros(3),
                "limits": (float(lim.get("lower")), float(lim.get("upper")))
                if lim is not None else None,
            })  # fmt: skip
        children = {j["child"] for j in self.joints}
        self.root_link = next(
            ln.get("name") for ln in root.findall("link") if ln.get("name") not in children
        )
        missing = []
        self.visual_corners: dict[str, np.ndarray] = {}
        self.vertices: dict[str, np.ndarray] = {}
        for ln in root.findall("link"):
            corners, verts = [], []
            for vis in ln.findall("visual"):
                mesh = vis.find("geometry/mesh")
                if mesh is None:
                    continue
                f = mesh_dir / mesh.get("filename").removeprefix(MESH_PREFIX)
                if not f.is_file():
                    missing.append(f.name)
                    continue
                o = vis.find("origin")
                t = transform(_floats(o, "xyz"), _floats(o, "rpy"))
                v = stl_vertices(f)
                bb = np.stack([v.min(0), v.max(0)])
                corners.append(box_corners(bb) @ t[:3, :3].T + t[:3, 3])
                verts.append(support_points(v, n_dirs) @ t[:3, :3].T + t[:3, 3])
            if corners:
                self.visual_corners[ln.get("name")] = np.concatenate(corners)
                self.vertices[ln.get("name")] = np.concatenate(verts)
        if missing:
            raise FileNotFoundError(f"{FETCH_HINT} (missing: {', '.join(sorted(set(missing)))})")
        self.movable = [j for j in self.joints if j["type"] in ("revolute", "continuous")]
        self.limits = {j["name"]: j["limits"] for j in self.movable}

    def link_transforms(self, q: dict[str, np.ndarray], base: np.ndarray) -> dict[str, np.ndarray]:
        """Joint values (name -> (N,)) -> link name -> (N, 4, 4) world transforms."""
        n = len(next(iter(q.values())))
        tf = {self.root_link: np.tile(base, (n, 1, 1))}
        pending = list(self.joints)
        while pending:
            for j in list(pending):
                if j["parent"] not in tf:
                    continue
                t = tf[j["parent"]] @ j["T"]
                if j["type"] != "fixed":
                    t = t @ axis_angle(j["axis"], q[j["name"]])
                tf[j["child"]] = t
                pending.remove(j)
        return tf

    def points(self, q, base, links=None, kind="corners") -> np.ndarray:
        """World coordinates of the links' points, (N, M, 3). kind: "corners" or "vertices"."""
        tf = self.link_transforms(q, base)
        src = self.visual_corners if kind == "corners" else self.vertices
        out = []
        for link, c in src.items():
            if links is not None and link not in links:
                continue
            t = tf[link]
            out.append(np.einsum("nij,mj->nmi", t[:, :3, :3], c) + t[:, None, :3, 3])
        return np.concatenate(out, axis=1)

    def sample(self, n: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {name: rng.uniform(*lim, size=n) for name, lim in self.limits.items()}


DOWN = np.array([0.0, 0.0, -1.0])


def dls(residual, q: np.ndarray, lo, hi, iters: int, damping: float, max_step=0.3, h=1e-6):
    """Damped least squares on a batch: residual(q (M, J)) -> (M, K) is driven towards 0.

    Numerical Jacobian, joint limits by clipping, at most `max_step` rad per iteration."""
    lam2 = damping**2
    for _ in range(iters):
        r = residual(q)
        jac = np.empty((len(q), r.shape[1], q.shape[1]))
        for i in range(q.shape[1]):
            dq = q.copy()
            dq[:, i] += h
            jac[:, :, i] = (residual(dq) - r) / h
        jjt = jac @ jac.transpose(0, 2, 1) + lam2 * np.eye(r.shape[1])
        step = -(jac.transpose(0, 2, 1) @ np.linalg.solve(jjt, r[..., None]))[..., 0]
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, max_step / np.maximum(norm, 1e-12))
        q = np.clip(q + step, lo, hi)
    return q


def tool_pose(robot: Robot, jv: dict, base: np.ndarray, tip_link: str, tcp) -> tuple:
    """TCP position (N, 3) and the tip link's rotation (N, 3, 3); `tcp` is in the tip frame."""
    t = robot.link_transforms(jv, base)[tip_link]
    rot = t[:, :3, :3]
    return rot @ np.asarray(tcp, dtype=float) + t[:, :3, 3], rot


def tool_axis(lean: np.ndarray, tilt: float) -> np.ndarray:
    """Desired tool axis (N, 3): straight down, leaning by `tilt` rad towards the horizontal
    unit vectors `lean` (N, 2). This arm's tool axis stays in the vertical plane through the
    shoulder_pan axis, so `lean` is the direction away from the base (the tip ahead of the
    wrist)."""
    d = np.concatenate([lean, np.zeros((len(lean), 1))], axis=1)
    return np.cos(tilt) * DOWN + np.sin(tilt) * d


def tool_errors(pos, rot, targets, dirs=None, tilt=0.0, lean=None) -> dict:
    """Position error [m]; with push directions (N, 2): angle between the tool axis (tip frame
    +z) and tool_axis(lean, tilt), and misalignment of the jaw line (tip frame x) from
    perpendicular to the push direction [rad]."""
    out = {"pos": np.linalg.norm(pos - targets, axis=1)}
    if dirs is not None:
        axis = tool_axis(dirs if lean is None else lean, tilt)
        out["tilt"] = np.arccos(np.clip((rot[:, :, 2] * axis).sum(1), -1.0, 1.0))
        out["jaw"] = np.arcsin(np.clip(np.abs((rot[:, :2, 0] * dirs).sum(1)), 0.0, 1.0))
    return out


def tool_residual(
    pos, rot, targets, dirs=None, w_rot=0.05, tilt=0.0, lean=None, w_axis=None
) -> np.ndarray:
    """Residual for dls(): position, and with `dirs` the tool axis along tool_axis(lean, tilt)
    and the jaw line perpendicular to the push direction (orientation rows scaled by w_rot
    [m/rad]). With a small `w_axis` the tool axis is only pulled towards that direction (a
    soft preference): the arm leans the tool where it must to reach."""
    r = [pos - targets]
    if dirs is not None:
        w = w_rot if w_axis is None else w_axis
        r.append(w * (rot[:, :, 2] - tool_axis(dirs if lean is None else lean, tilt)))
        r.append(w_rot * (rot[:, :2, 0] * dirs).sum(1, keepdims=True))
    return np.concatenate(r, axis=1)
