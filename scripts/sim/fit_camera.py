"""Compute the overhead camera pose from the conditions in configs/sim.yaml `camera_fit`.

    uv run python scripts/sim/fit_camera.py            # print the fit
    uv run python scripts/sim/fit_camera.py --write    # also write camera.xyz / rpy to sim.yaml

No ROS needed. Camera model: pinhole, Gazebo's convention (looks along +x of its frame, z up;
pitch > 0 looks down), square pixels, principal point at the image centre. The camera stays on
the symmetry plane y = 0 on the near side of the table (roll = yaw = 0); unknowns: x, z, pitch.

Conditions (`camera_fit`):
- must be in the image with `margin` left on every side: the 4 outer corners of the green band
  and the robot base's footprint (bounding box of base_link's meshes on the table top). The arm
  itself (raised high, or behind the base) may leave the image.
- height above the table top <= `max_height`, pitch inside `pitch_range` (oblique view)
- the green band's outer outline spans >= `min_green_width` of the image width
Among the poses that satisfy all of them, the highest camera is chosen (closest to a top-down
view, least occluded by the arm). The widest band a camera can show shrinks with its height, so
that height is found by bisection; for each height, x and pitch come from a grid search.
If no pose satisfies the conditions, the script reports which one limits it and exits with 1.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from scripts.sim.kinematics import ROOT, SIM_YAML, Robot, box_corners, transform

GREEN_NAMES = ["near-left", "near-right", "far-right", "far-left"]  # near = -x (camera side)


def project(points: np.ndarray, x, z, pitch, width: int, height: int, hfov: float):
    """Camera(s) at (x, 0, z) with rpy (0, pitch, 0). x, z, pitch: scalars or (n,) arrays.

    Returns u, v, depth with shape (n, k) for k points (n = 1 for scalars).
    """
    x, z, pitch = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (x, z, pitch))
    x, z, pitch = np.broadcast_arrays(x, z, pitch)
    c, s = np.cos(pitch)[:, None], np.sin(pitch)[:, None]
    dx = points[None, :, 0] - x[:, None]
    dy = points[None, :, 1] + 0.0 * x[:, None]
    dz = points[None, :, 2] - z[:, None]
    depth = c * dx - s * dz
    zu = s * dx + c * dz
    f = (width / 2) / np.tan(hfov / 2)
    return width / 2 - f * dy / depth, height / 2 - f * zu / depth, depth


def margins(points, x, z, pitch, cam) -> np.ndarray:
    """Free space at the top, bottom, left and right, as fractions of the image; (n, 4)."""
    w, h = cam["width"], cam["height"]
    u, v, depth = project(points, x, z, pitch, w, h, cam["hfov"])
    m = np.stack([v.min(1) / h, 1 - v.max(1) / h, u.min(1) / w, 1 - u.max(1) / w], axis=1)
    m[np.any(depth <= 0.05, axis=1)] = -1.0  # a point behind (or at) the camera
    return m


def green_width(green, x, z, pitch, cam) -> np.ndarray:
    u, _, _ = project(green, x, z, pitch, cam["width"], cam["height"], cam["hfov"])
    return (u.max(1) - u.min(1)) / cam["width"]


def best_at_height(must, green, z, cam, fit) -> tuple[float, float, float]:
    """Widest green band (fraction of the image width) at height z: (fraction, x, pitch).

    fraction is -1 if no (x, pitch) keeps the margins at this height."""
    lo, hi = fit["pitch_range"]
    step = fit["grid_step"]
    xs = np.arange(fit["x_range"][0], fit["x_range"][1] + 1e-9, step)
    ps = np.arange(lo, hi + 1e-9, step)
    best = (-1.0, np.nan, np.nan)
    for _ in range(3):  # coarse grid, then two finer grids around the best point
        gx, gp = (a.ravel() for a in np.meshgrid(xs, ps, indexing="ij"))
        ok = margins(must, gx, z, gp, cam).min(1) >= fit["margin"]
        frac = np.where(ok, green_width(green, gx, z, gp, cam), -1.0)
        i = int(np.argmax(frac))
        if frac[i] > best[0]:
            best = (float(frac[i]), float(gx[i]), float(gp[i]))
        if best[0] < 0:
            return best
        xs = np.linspace(best[1] - step, best[1] + step, 21)
        ps = np.clip(np.linspace(best[2] - step, best[2] + step, 21), lo, hi)
        step /= 10
    return best


def fit_camera(must, green, cam, fit) -> dict:
    """The highest camera satisfying every condition, or the reason why there is none."""
    z_lo, z_hi = fit["min_height"], fit["max_height"]
    target = fit["min_green_width"]
    top = best_at_height(must, green, z_hi, cam, fit)
    if top[0] >= target:
        return {"ok": True, "z": z_hi, "fraction": top[0], "x": top[1], "pitch": top[2],
                "limited_by": "max_height", "fraction_at_max_height": top[0]}  # fmt: skip
    low = best_at_height(must, green, z_lo, cam, fit)
    if low[0] < target:
        why = ("no pose keeps the margin" if low[0] < 0 else
               f"the widest band is {low[0]:.3f} of the image width")  # fmt: skip
        reason = (
            f"min_green_width {target} is not reached even at min_height {z_lo}: {why} "
            f"(margin {fit['margin']}, pitch_range {fit['pitch_range']})"
        )
        return {"ok": False, "fraction_at_min_height": low[0], "fraction_at_max_height": top[0],
                "reason": reason}  # fmt: skip
    best = (*low, z_lo)
    for _ in range(int(np.ceil(np.log2((z_hi - z_lo) / fit["height_tol"])))):
        mid = (z_lo + z_hi) / 2
        r = best_at_height(must, green, mid, cam, fit)
        if r[0] >= target:
            z_lo, best = mid, (*r, mid)
        else:
            z_hi = mid
    frac, x, pitch, z = best
    return {"ok": True, "z": z, "fraction": frac, "x": x, "pitch": pitch,
            "limited_by": "min_green_width", "fraction_at_max_height": top[0]}  # fmt: skip


def scene_points(cfg: dict, robot: Robot) -> dict[str, np.ndarray]:
    t, rb = cfg["table"], cfg["robot"]
    gx = t["white"][0] / 2 + t["band_width"]
    gy = t["white"][1] / 2 + t["band_width"]
    green = np.array([[-gx, gy, 0.0], [-gx, -gy, 0.0], [gx, -gy, 0.0], [gx, gy, 0.0]])
    zero = {name: np.zeros(1) for name in robot.limits}
    base_pts = robot.points(zero, transform(rb["xyz"], rb["rpy"]), links={robot.root_link})[0]
    lo, hi = base_pts.min(0), base_pts.max(0)
    footprint = box_corners(np.array([[lo[0], lo[1], 0.0], [hi[0], hi[1], 0.0]]))[::2]
    return {"green_corners": green, "base_footprint": footprint}


def write_camera(path: Path, xyz, rpy) -> None:
    """Replace only the camera's xyz/rpy lines (keeps comments and the rest of the file)."""
    text = path.read_text()
    sec = re.search(r"^camera:.*?(?=^\S)", text, flags=re.S | re.M)
    body = sec.group(0)
    fmt = lambda v: "[" + ", ".join(f"{x:.3f}" for x in v) + "]"  # noqa: E731
    body = re.sub(r"(^\s+xyz:\s*)\[[^\]]*\]", lambda m: m.group(1) + fmt(xyz), body, flags=re.M)
    body = re.sub(r"(^\s+rpy:\s*)\[[^\]]*\]", lambda m: m.group(1) + fmt(rpy), body, flags=re.M)
    path.write_text(text[: sec.start()] + body + text[sec.end() :])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write camera.xyz / rpy to sim.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(SIM_YAML.read_text())
    cam, fit = cfg["camera"], cfg["camera_fit"]
    try:
        robot = Robot()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    pts = scene_points(cfg, robot)
    must = np.concatenate([pts["green_corners"], pts["base_footprint"]])
    res = fit_camera(must, pts["green_corners"], cam, fit)
    if not res["ok"]:
        print("no camera pose satisfies camera_fit:", res["reason"])
        return 1
    # the values as written to sim.yaml (3 decimals); everything below is reported for them
    x, z, pitch = round(res["x"], 3), round(res["z"], 3), round(res["pitch"], 3)
    print(f"camera: xyz [{x:.3f}, 0.000, {z:.3f}], pitch {pitch:.3f} rad "
          f"({np.degrees(pitch):.1f} deg down); limited by {res['limited_by']}")  # fmt: skip
    print(f"green band width: {green_width(pts['green_corners'], x, z, pitch, cam)[0]:.3f} "
          f"of the image (widest possible at max_height {fit['max_height']}: "
          f"{res['fraction_at_max_height']:.3f})")  # fmt: skip
    m = margins(must, x, z, pitch, cam)[0]
    print("margins top/bottom/left/right:", " ".join(f"{a:.3f}" for a in m))
    for name, p in pts.items():
        u, v, _ = project(p, x, z, pitch, cam["width"], cam["height"], cam["hfov"])
        uv = " ".join(f"({a:.0f},{b:.0f})" for a, b in zip(u[0], v[0], strict=True))
        print(f"{name} (u, v): {uv}")
    if args.write:
        write_camera(SIM_YAML, (x, 0.0, z), (0.0, pitch, 0.0))
        print(f"written to {SIM_YAML.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
