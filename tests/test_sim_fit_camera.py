"""scripts/sim/fit_camera.py and the camera / check poses in configs/sim.yaml (no ROS needed)."""

import json
import shutil

import numpy as np
import pytest
import yaml

from scripts.sim import fit_camera as fc
from scripts.sim import kinematics as kin

CAM = {"width": 640, "height": 480, "hfov": 1.229}
CFG = yaml.safe_load(kin.SIM_YAML.read_text())
HAVE_MESHES = all(
    (kin.DESC / "meshes" / n).is_file()
    for n in json.loads((kin.DESC / "assets.lock").read_text())["meshes"]
)
needs_meshes = pytest.mark.skipif(not HAVE_MESHES, reason="SO-101 meshes not fetched")


@pytest.fixture(scope="module")
def robot():
    return kin.Robot()


# ---------- camera model ----------
def test_point_on_the_optical_axis_projects_to_the_image_centre():
    pitch, x, z = 0.9, -0.3, 1.0
    p = np.array([x, 0, z]) + 0.8 * np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    u, v, depth = fc.project(p[None], x, z, pitch, **CAM)
    assert (u[0, 0], v[0, 0], depth[0, 0]) == pytest.approx((320, 240, 0.8))


def test_image_axes_follow_gazebo_camera_convention():
    ahead = np.array([[1.0, 0.0, 1.0]])
    u, _, _ = fc.project(ahead + [0, 0.1, 0], 0.0, 1.0, 0.0, **CAM)
    _, v, _ = fc.project(ahead + [0, 0, 0.1], 0.0, 1.0, 0.0, **CAM)
    assert u[0, 0] < 320  # world +y (camera left) is on the left of the image
    assert v[0, 0] < 240  # higher is nearer the top
    assert u[0, 0] == pytest.approx(320 - 320 / np.tan(1.229 / 2) * 0.1)


def test_project_accepts_arrays_of_camera_poses():
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])
    u, v, _ = fc.project(pts, np.array([-0.5, -0.4]), 0.6, np.array([1.0, 1.1]), **CAM)
    assert u.shape == v.shape == (2, 2)
    u1, _, _ = fc.project(pts, -0.4, 0.6, 1.1, **CAM)
    assert u[1] == pytest.approx(u1[0])


def test_margins_flag_points_behind_the_camera():
    behind = np.array([[-1.0, 0.0, 0.0]])
    assert (fc.margins(behind, 0.0, 0.5, 0.9, CAM) == -1).all()


# ---------- camera fit ----------
def _fit(**over):
    fit = {"margin": 0.05, "max_height": 0.8, "min_height": 0.15, "min_green_width": 0.6,
           "pitch_range": [0.5, 1.3], "x_range": [-1.2, 0.3], "grid_step": 0.01,
           "height_tol": 0.001}  # fmt: skip
    return {**fit, **over}


GREEN = np.array([[-0.25, 0.2, 0.0], [-0.25, -0.2, 0.0], [0.25, -0.2, 0.0], [0.25, 0.2, 0.0]])
FOOT = np.array([[0.27, 0.05, 0.0], [0.27, -0.05, 0.0], [0.36, -0.05, 0.0], [0.36, 0.05, 0.0]])


def test_fit_meets_every_condition_and_takes_the_highest_camera():
    must = np.concatenate([GREEN, FOOT])
    res = fc.fit_camera(must, GREEN, CAM, _fit())
    assert res["ok"] and res["limited_by"] == "min_green_width"
    assert res["z"] <= 0.8
    assert fc.green_width(GREEN, res["x"], res["z"], res["pitch"], CAM)[0] >= 0.6
    assert fc.margins(must, res["x"], res["z"], res["pitch"], CAM).min() >= 0.05 - 1e-9
    higher = fc.best_at_height(must, GREEN, res["z"] + 0.01, CAM, _fit())
    assert higher[0] < 0.6  # a higher camera cannot show the band that wide


def test_fit_stops_at_max_height_when_the_band_is_wide_enough_there():
    res = fc.fit_camera(np.concatenate([GREEN, FOOT]), GREEN, CAM, _fit(min_green_width=0.3))
    assert res["ok"] and res["limited_by"] == "max_height" and res["z"] == 0.8


def test_fit_reports_when_the_conditions_cannot_be_met():
    res = fc.fit_camera(np.concatenate([GREEN, FOOT]), GREEN, CAM, _fit(min_green_width=0.95))
    assert not res["ok"]
    assert "min_green_width" in res["reason"]


def test_write_camera_changes_only_the_camera_pose(tmp_path):
    y = tmp_path / "sim.yaml"
    shutil.copy(kin.SIM_YAML, y)
    before = y.read_text().splitlines()
    fc.write_camera(y, (-0.1, 0.0, 1.5), (0.0, 1.0, 0.0))
    after = y.read_text().splitlines()
    changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
    assert len(changed) <= 2  # camera xyz and rpy only; comments and other sections kept
    cfg = yaml.safe_load(y.read_text())
    assert cfg["camera"]["xyz"] == [-0.1, 0.0, 1.5]
    assert cfg["camera"]["rpy"] == [0.0, 1.0, 0.0]
    assert cfg["robot"] == CFG["robot"]


@needs_meshes
def test_configured_camera_meets_camera_fit(robot):
    pts = fc.scene_points(CFG, robot)
    must = np.concatenate([pts["green_corners"], pts["base_footprint"]])
    cam, fit = CFG["camera"], CFG["camera_fit"]
    x, z, pitch = cam["xyz"][0], cam["xyz"][2], cam["rpy"][1]
    assert cam["xyz"][1] == 0 and cam["rpy"][0] == 0 and cam["rpy"][2] == 0
    assert z <= fit["max_height"]
    assert fit["pitch_range"][0] <= pitch <= fit["pitch_range"][1]
    # written with 3 decimals, so allow a little rounding
    assert fc.margins(must, x, z, pitch, cam).min() >= fit["margin"] - 0.002
    assert (
        fc.green_width(pts["green_corners"], x, z, pitch, cam)[0] >= fit["min_green_width"] - 0.002
    )


# ---------- check poses ----------
@needs_meshes
def test_reach_white_pose_tip_is_over_the_white_area(robot):
    q = {k: np.array([v]) for k, v in CFG["check"]["poses"]["reach_white"].items()}
    base = kin.transform(CFG["robot"]["xyz"], CFG["robot"]["rpy"])
    tip = robot.link_transforms(q, base)["gripper_frame_link"][0, :3, 3]
    wx, wy = (w / 2 for w in CFG["table"]["white"])
    assert abs(tip[0]) < wx and abs(tip[1]) < wy and 0 < tip[2] < 0.1
