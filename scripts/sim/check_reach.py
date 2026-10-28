"""Can the gripper tip reach every point of the white area at block-pushing height?

    uv run python scripts/sim/check_reach.py

No ROS needed. Settings: configs/sim.yaml `reach_check` (and `table`, `robot`).
For each point of a grid over the white area, inverse kinematics (damped least squares over
`ik_joints`, several random initial configurations, joint limits enforced) puts the TCP (`tcp`
in the `tip_link` frame) at `tip_height` above the white area. With `orientation: tool_down`
the tool also points straight down with the jaw line perpendicular to the push direction
(towards the target frame's centre). A point counts as reachable if the position error is below
`pos_tol`, the orientation errors below `rot_tol_deg`, and no mesh vertex of the arm (base
excluded) goes below the white area (+ `clearance`). Self-collision is not checked.

Outputs (git-ignored): outputs/sim/reach_map.png (top view, oriented like the camera image:
far side up, +y left) and outputs/sim/reach_result.json, with the white area's 4 corners
reported separately. If some points cannot be reached (the white area's size is kept):
1) the same IK over a wider grid around the base (`map_forward`, `map_lateral`, base frame)
   gives the coverage for small shifts of the base (up to `base_shift_max` in x and y), with
   and without the base footprint overlapping the green band;
2) the pushing area is restricted to the reachable part: the widest full-depth strip of the
   white area that starts at the edge next to the base.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import yaml

from scripts.sim.kinematics import (
    ROOT,
    SIM_YAML,
    Robot,
    dls,
    tool_errors,
    tool_pose,
    tool_residual,
    transform,
)

OUT = ROOT / "outputs" / "sim"


def _ik(robot, tgt, q, rc, lo, hi, base, dirs=None):
    """Damped least squares from initial configurations q (M, J) towards targets tgt (M, 3).

    With push directions `dirs` (M, 2) the tool also points straight down with the jaw line
    perpendicular to the push direction. Returns q and the errors (tool_errors)."""
    names = rc["ik_joints"]
    fixed = {k: np.full(len(q), float(v)) for k, v in rc["fixed"].items()}
    tcp = rc.get("tcp", [0.0, 0.0, 0.0])

    def pose(qq):
        jv = {**fixed, **{j: qq[:, i] for i, j in enumerate(names)}}
        return tool_pose(robot, jv, base, rc["tip_link"], tcp)

    tilt = np.radians(rc.get("tool_tilt_deg", 0.0))
    lean = None
    if dirs is not None:  # lean away from the base (the tool axis stays in the arm's plane)
        away = tgt[:, :2] - base[:2, 3]
        lean = away / np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-12)

    def residual(qq):
        return tool_residual(
            *pose(qq), tgt, dirs, rc.get("w_rot", 0.05), tilt, lean, rc.get("w_vertical")
        )

    q = dls(residual, q, lo, hi, rc["iters"], rc["damping"])
    return q, tool_errors(*pose(q), tgt, dirs, tilt, lean)


def _lowest(robot, q, rc, base) -> np.ndarray:
    """Lowest point of the arm's meshes (base excluded) for configurations q (M, J)."""
    names = rc["ik_joints"]
    arm_links = set(robot.vertices) - {robot.root_link}
    out = np.empty(len(q))
    for chunk in np.array_split(np.arange(len(q)), max(1, len(q) // 500)):
        if len(chunk) == 0:
            continue
        jv = {**{k: np.full(len(chunk), float(v)) for k, v in rc["fixed"].items()},
              **{j: q[chunk, i] for i, j in enumerate(names)}}  # fmt: skip
        pts = robot.points(jv, base, links=arm_links, kind="vertices")
        out[chunk] = pts[:, :, 2].min(1)
    return out


def solve_ik(robot: Robot, targets: np.ndarray, base: np.ndarray, rc: dict, dirs=None) -> dict:
    """IK for (N, 3) world targets: `seeds` random starts, then `refine_passes` passes that
    restart the failed targets from the solutions of their nearest reachable neighbours.

    dirs (N, 2): push directions for the tool-down mode (None: position only)."""
    names, s, n = rc["ik_joints"], rc["seeds"], len(targets)
    rng = np.random.default_rng(rc["seed"])
    lo = np.array([robot.limits[j][0] for j in names])
    hi = np.array([robot.limits[j][1] for j in names])
    rot_tol = np.radians(rc.get("rot_tol_deg", 3.0))
    # with a soft vertical preference (w_vertical) the tilt may grow up to max_tilt_deg
    max_tilt = np.radians(rc["max_tilt_deg"]) if rc.get("w_vertical") else None
    floor = rc.get("surface_z", 0.0) + rc["clearance"]
    reachable = np.zeros(n, dtype=bool)
    converged = np.zeros(n, dtype=bool)
    min_err = np.full(n, np.inf)
    q_best = np.full((n, len(names)), np.nan)

    def attempt(idx, q0):  # idx: target index per start, q0: start configurations
        d = None if dirs is None else dirs[idx]
        q, e = _ik(robot, targets[idx], q0, rc, lo, hi, base, d)
        conv = e["pos"] < rc["pos_tol"]
        if d is not None:
            tilt_ok = e["tilt"] < (rot_tol if max_tilt is None else max_tilt)
            conv &= tilt_ok & (e["jaw"] < rot_tol)
        low = np.full(len(q), -np.inf)
        low[conv] = _lowest(robot, q[conv], rc, base)
        ok = conv & (low >= floor - 1e-9)
        np.minimum.at(min_err, idx, e["pos"])
        converged[idx[conv]] = True
        reachable[idx[ok]] = True
        q_best[idx[ok]] = q[ok]

    idx = np.repeat(np.arange(n), s)
    attempt(idx, rng.uniform(lo, hi, size=(len(idx), len(names))))
    for _ in range(rc["refine_passes"]):
        fail, good = np.flatnonzero(~reachable), np.flatnonzero(reachable)
        if len(fail) == 0 or len(good) == 0:
            break
        d = np.linalg.norm(targets[fail, None, :2] - targets[None, good, :2], axis=2)
        near = good[np.argsort(d, axis=1)[:, : rc["neighbours"]]]  # (F, k)
        idx = np.repeat(fail, near.shape[1])
        q0 = q_best[near.ravel()] + rng.normal(0, 0.05, size=(len(idx), len(names)))
        attempt(idx, np.clip(q0, lo, hi))
    return {"reachable": reachable, "converged": converged, "min_err": min_err, "q": q_best}


def surface_z(cfg) -> float:
    """Top of the white area above the table top (two overlay plates)."""
    return 2 * cfg["table"]["overlay_thickness"]


def push_dirs(cfg, xy: np.ndarray):
    """Push directions for the tool-down mode: towards the target frame's centre (+y when the
    point is within 1 cm of it). None in the position-only mode."""
    if cfg["reach_check"].get("orientation") != "tool_down":
        return None
    d = np.asarray(cfg["target_frame"]["center"], dtype=float) - xy[:, :2]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(norm > 0.01, d / np.maximum(norm, 1e-12), np.array([0.0, 1.0]))


def with_surface(cfg) -> dict:
    """reach_check settings with the white area's height (heights are measured from it)."""
    return {**cfg["reach_check"], "surface_z": surface_z(cfg)}


def white_grid(cfg) -> np.ndarray:
    t, rc = cfg["table"], cfg["reach_check"]
    step = rc["grid_step"]
    wx, wy = t["white"][0] / 2, t["white"][1] / 2
    xs = np.arange(-wx, wx + 1e-9, step)
    ys = np.arange(-wy, wy + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    z = surface_z(cfg) + rc["tip_height"]
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)


def base_pose(rb, shift=(0.0, 0.0)) -> tuple[np.ndarray, float]:
    """The base origin on the table (x, y) and its yaw, optionally shifted by (dx, dy)."""
    return np.asarray(rb["xyz"][:2], dtype=float) + shift, float(rb["rpy"][2])


def to_base(xy: np.ndarray, rb, shift=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Table (x, y) -> (forward, lateral) in the base frame (forward = where the arm points)."""
    o, yaw = base_pose(rb, shift)
    d = xy - o
    c, s = np.cos(yaw), np.sin(yaw)
    return c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]


def to_table(f: np.ndarray, lat: np.ndarray, rb, shift=(0.0, 0.0)) -> np.ndarray:
    o, yaw = base_pose(rb, shift)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([o[0] + c * f - s * lat, o[1] + s * f + c * lat], axis=-1)


def footprint_local(robot: Robot) -> np.ndarray:
    """(4, 2) corners of the base footprint (bounding box of base_link's meshes), base frame."""
    zero = {name: np.zeros(1) for name in robot.limits}
    bp = robot.points(zero, np.eye(4), links={robot.root_link})[0]
    (x0, y0), (x1, y1) = bp[:, :2].min(0), bp[:, :2].max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def overlaps_band(fp_table: np.ndarray, t) -> bool:
    """Does the footprint's bounding box overlap the green band's outer rectangle?"""
    gx = t["white"][0] / 2 + t["band_width"]
    gy = t["white"][1] / 2 + t["band_width"]
    (x0, y0), (x1, y1) = fp_table.min(0), fp_table.max(0)
    return bool(x0 < gx and x1 > -gx and y0 < gy and y1 > -gy)


def white_corners(cfg) -> dict[str, np.ndarray]:
    """Corners of the white area. near = -x (camera side), left = +y (image left)."""
    wx, wy = (w / 2 for w in cfg["table"]["white"])
    return {"near-left": np.array([-wx, wy]), "far-left": np.array([wx, wy]),
            "near-right": np.array([-wx, -wy]), "far-right": np.array([wx, -wy])}  # fmt: skip


def base_map(robot, cfg, base):
    """Reachability around the base on a (forward, lateral) grid in the base frame."""
    rc, rb = cfg["reach_check"], cfg["robot"]
    step = rc["grid_step"]
    fs = np.round(np.arange(rc["map_forward"][0], rc["map_forward"][1] + 1e-9, step), 4)
    ls = np.round(np.arange(rc["map_lateral"][0], rc["map_lateral"][1] + 1e-9, step), 4)
    gf, gl = np.meshgrid(fs, ls, indexing="ij")
    xy = to_table(gf.ravel(), gl.ravel(), rb)
    pts = np.concatenate([xy, np.full((len(xy), 1), surface_z(cfg) + rc["tip_height"])], axis=1)
    res = solve_ik(robot, pts, base, with_surface(cfg), push_dirs(cfg, xy))
    return fs, ls, res["reachable"].reshape(gf.shape)


def lookup(fs, ls, reach, f, lat) -> np.ndarray:
    """Reachability of points at (forward, lateral) in the base frame, conservatively: a point
    counts only if every map cell around it (floor / ceil in both axes) is reachable. Outside
    the map: False."""
    step = fs[1] - fs[0]
    fi = (np.asarray(f) - fs[0]) / step
    li = (np.asarray(lat) - ls[0]) / step
    out = np.ones(np.shape(fi), dtype=bool)
    for i in (np.floor(fi + 1e-6), np.ceil(fi - 1e-6)):
        for j in (np.floor(li + 1e-6), np.ceil(li - 1e-6)):
            i_, j_ = i.astype(int), j.astype(int)
            inside = (i_ >= 0) & (i_ < len(fs)) & (j_ >= 0) & (j_ < len(ls))
            hit = np.zeros(np.shape(fi), dtype=bool)
            hit[inside] = reach[i_[inside], j_[inside]]
            out &= hit
    return out


def reachable_at(cfg, fs, ls, reach, xy, shift=(0.0, 0.0)) -> np.ndarray:
    return lookup(fs, ls, reach, *to_base(xy, cfg["robot"], shift))


def work_strip(cfg, ok: np.ndarray) -> dict:
    """The part of the white area to push in: a full-depth strip from the base's side.

    The strip spans the whole depth (x) and runs from the white area's edge next to the base
    towards the other side, as far as every grid point in it is reachable."""
    grid = white_grid(cfg)
    side = np.sign(cfg["robot"]["xyz"][1]) or 1.0  # the base is at -y: strip starts at -y
    across = -side * grid[:, 1]  # distance-like coordinate from the base's side
    cols = np.unique(np.round(across, 4))
    edge = cols[0]
    last = None
    for c in cols:
        if ok[np.isclose(across, c)].all():
            last = c
        else:
            break
    wy = cfg["table"]["white"][1]
    if last is None:
        return {"strip_width": 0.0, "fraction_of_white": 0.0}
    width = float(last - edge)
    ys = sorted([round(float(-side * edge), 3), round(float(-side * last), 3)])
    return {"strip_width": round(width, 3), "fraction_of_white": round(width / wy, 3),
            "strip_y": ys}  # fmt: skip


def proposals(cfg, fs, ls, reach, fp_local) -> dict:
    """1) shift the base a little (white area unchanged); 2) restrict the pushing area."""
    t, rb, rc = cfg["table"], cfg["robot"], cfg["reach_check"]
    step = rc["grid_step"]
    grid = white_grid(cfg)[:, :2]
    corners = white_corners(cfg)
    m = rc["base_shift_max"]
    shifts = np.round(np.arange(-m, m + 1e-9, step), 4)
    rows = []
    for dx in shifts:
        for dy in shifts:
            sh = np.array([dx, dy])
            ok = reachable_at(cfg, fs, ls, reach, grid, sh)
            cok = reachable_at(cfg, fs, ls, reach, np.array(list(corners.values())), sh)
            fp = to_table(fp_local[:, 0], fp_local[:, 1], rb, sh)
            rows.append({"shift": [float(dx), float(dy)], "coverage": round(float(ok.mean()), 4),
                         "corners": dict(zip(corners, map(bool, cok), strict=True)),
                         "overlaps_band": overlaps_band(fp, t), "_ok": ok})  # fmt: skip

    def pick(cands):
        if not cands:
            return None
        r = max(cands, key=lambda r: (r["coverage"], -np.hypot(*r["shift"])))
        out = {k: v for k, v in r.items() if k != "_ok"}
        out["work_strip"] = work_strip(cfg, r["_ok"])
        return out

    full = [r for r in rows if r["coverage"] >= 1.0]
    return {
        "1_best_shift_outside_band": pick([r for r in rows if not r["overlaps_band"]]),
        "1_smallest_shift_reaching_all": pick(
            [min(full, key=lambda r: np.hypot(*r["shift"]))] if full else []
        ),
    }


def draw(cfg, grid, res, path, fp_table, fs=None, ls=None, reach=None, strip=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    t, rb = cfg["table"], cfg["robot"]
    wx, wy = t["white"][0] / 2, t["white"][1] / 2
    gx, gy = wx + t["band_width"], wy + t["band_width"]
    fig, ax = plt.subplots(figsize=(8, 7))

    def rect(x0, x1, y0, y1, **kw):  # plot axes: horizontal = -y (image left = +y), up = +x
        ax.add_patch(Rectangle((-y1, x0), y1 - y0, x1 - x0, **kw))

    rect(-t["size"][0] / 2, t["size"][0] / 2, -t["size"][1] / 2, t["size"][1] / 2,
         fc="0.85", ec="0.5")  # fmt: skip
    rect(-gx, gx, -gy, gy, fc="#52c072", ec="none")
    rect(-wx, wx, -wy, wy, fc="white", ec="none")
    ax.add_patch(Polygon(np.c_[-fp_table[:, 1], fp_table[:, 0]], fc="#e6d84a", ec="k"))
    if reach is not None:  # outline of the reachable region around the base (wider map)
        gf, gl = np.meshgrid(fs, ls, indexing="ij")
        xy = to_table(gf, gl, rb)
        ax.contour(-xy[..., 1], xy[..., 0], reach.astype(float), levels=[0.5],
                   colors="tab:blue", linewidths=1)  # fmt: skip
    if strip and strip.get("strip_y"):
        y0, y1 = strip["strip_y"]
        ax.add_patch(Rectangle((-y1, -wx), y1 - y0, 2 * wx, fill=False, ec="tab:purple",
                               lw=2, ls="--", label="pushing area (full depth)"))  # fmt: skip
    ok, conv = res["reachable"], res["converged"]
    ax.plot(-grid[ok, 1], grid[ok, 0], ".", ms=3, color="tab:green", label="reachable")
    m = ~ok & conv
    ax.plot(-grid[m, 1], grid[m, 0], "x", ms=4, color="tab:orange",
            label="position reached but the arm goes below the table top")  # fmt: skip
    m = ~conv
    ax.plot(-grid[m, 1], grid[m, 0], "x", ms=4, color="tab:red",
            label="position (or tool orientation) not reached")  # fmt: skip
    tf = cfg.get("target_frame")
    if tf:  # inner square of the target frame
        h = tf["inner"] / 2
        cx, cy = tf["center"]
        rect(cx - h, cx + h, cy - h, cy + h, fill=False, ec="tab:blue", lw=1.5)
    for name, c in white_corners(cfg).items():
        ax.annotate(name, (-c[1], c[0]), fontsize=7, ha="center",
                    xytext=(0, 8 if c[0] > 0 else -12), textcoords="offset points")  # fmt: skip
    ax.set_xlim(-t["size"][1] / 2 - 0.02, t["size"][1] / 2 + 0.02)
    ax.set_ylim(-t["size"][0] / 2 - 0.02, t["size"][0] / 2 + 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("-y [m] (image left is +y)")
    ax.set_ylabel("x [m] (far side up, camera at the bottom)")
    rc = cfg["reach_check"]
    mode = " pointing down" if rc.get("orientation") == "tool_down" else ""
    ax.set_title(f"TCP{mode} at {rc['tip_height'] * 1000:.0f} mm above the white area: "
                 f"{ok.sum()}/{len(ok)} points reachable")  # fmt: skip
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--reuse-map",
        action="store_true",
        help="reuse outputs/sim/reach_base_map.npz if the settings are unchanged",
    )
    ap.add_argument(
        "--no-proposals",
        action="store_true",
        help="skip the map around the base and the change proposals",
    )
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(SIM_YAML.read_text())
    rc, rb = cfg["reach_check"], cfg["robot"]
    try:
        robot = Robot()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    base = transform(rb["xyz"], rb["rpy"])
    grid = white_grid(cfg)
    res = solve_ik(robot, grid, base, with_surface(cfg), push_dirs(cfg, grid))
    ok = res["reachable"]
    fp_local = footprint_local(robot)
    corners = white_corners(cfg)
    cidx = {n: int(np.argmin(np.linalg.norm(grid[:, :2] - c, axis=1))) for n, c in corners.items()}
    out = {
        "points": len(grid),
        "reachable": int(ok.sum()),
        "position_not_reached": int((~res["converged"]).sum()),
        "below_table_only": int((res["converged"] & ~ok).sum()),
        "corners_reachable": {n: bool(ok[i]) for n, i in cidx.items()},
        "unreachable_xy": grid[~ok, :2].round(3).tolist(),
        "settings": rc,
        "robot": rb,
        "white": cfg["table"]["white"],
    }
    fs = ls = reach = None
    if not ok.all() and not args.no_proposals:
        cache = OUT / "reach_base_map.npz"  # git-ignored; reused only with the same settings
        key = json.dumps([rc, rb, cfg["target_frame"], surface_z(cfg)], sort_keys=True)
        if args.reuse_map and cache.is_file() and str(np.load(cache)["key"]) == key:
            z = np.load(cache)
            fs, ls, reach = z["fs"], z["ls"], z["reach"]
        else:
            fs, ls, reach = base_map(robot, cfg, base)
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez(cache, fs=fs, ls=ls, reach=reach, key=key)
        out["proposals"] = proposals(cfg, fs, ls, reach, fp_local)
    OUT.mkdir(parents=True, exist_ok=True)
    fp_table = to_table(fp_local[:, 0], fp_local[:, 1], rb)
    strip = work_strip(cfg, ok) if not ok.all() else None
    out["work_strip_from_white_grid"] = strip
    draw(cfg, grid, res, OUT / "reach_map.png", fp_table, fs, ls, reach, strip)
    (OUT / "reach_result.json").write_text(json.dumps(out, indent=2) + "\n")
    summary = {k: v for k, v in out.items() if k not in ("unreachable_xy", "settings")}
    print(json.dumps(summary, indent=2))
    if not ok.all():
        xy = grid[~ok, :2]
        print(f"unreachable region: x {xy[:, 0].min():.2f}..{xy[:, 0].max():.2f}, "
              f"y {xy[:, 1].min():.2f}..{xy[:, 1].max():.2f}")  # fmt: skip
    return 0 if ok.all() else 1


if __name__ == "__main__":
    sys.exit(main())
