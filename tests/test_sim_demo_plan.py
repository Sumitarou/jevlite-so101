"""scripts/sim/demo_plan.py: push geometry, TCP path, planned joint trajectories, success judge."""

import json

import numpy as np
import pytest
import yaml

from scripts.sim import demo_plan as dp
from scripts.sim import kinematics as kin

CFG = yaml.safe_load(kin.SIM_YAML.read_text())
DEMO = CFG["demo"]
HAVE_MESHES = all(
    (kin.DESC / "meshes" / n).is_file()
    for n in json.loads((kin.DESC / "assets.lock").read_text())["meshes"]
)
needs_meshes = pytest.mark.skipif(not HAVE_MESHES, reason="SO-101 meshes not fetched")
Z0 = dp.surface_z(CFG)


def test_half_extent_of_the_block():
    size = CFG["block"]["size"]
    assert dp.half_extent(size, 0.0) == pytest.approx(0.025)
    assert dp.half_extent(size, np.pi / 4) == pytest.approx(0.025 * 2**0.5)


def test_sample_start_is_reproducible_and_in_range():
    a = [dp.sample_start(np.random.default_rng(7), DEMO) for _ in range(2)]
    assert a[0] == a[1]
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y, yaw = dp.sample_start(rng, DEMO)
        assert DEMO["start_x"][0] <= x <= DEMO["start_x"][1]
        assert DEMO["start_y"][0] <= y <= DEMO["start_y"][1]
        assert abs(np.degrees(yaw)) <= DEMO["start_yaw_deg"]


def _with_aim(aim):
    return {**CFG, "demo": {**DEMO, "push_aim": aim}}


def test_push_geometry_ends_with_the_block_centre_on_the_target():
    start = (0.03, -0.05, np.radians(10))
    g = dp.push_geometry(_with_aim("center"), start)
    t = np.array(CFG["target_frame"]["center"])
    d = g["dir"]
    assert np.linalg.norm(d) == pytest.approx(1.0)
    to_t = t - np.array(start[:2])
    assert d[0] * to_t[1] - d[1] * to_t[0] == pytest.approx(0.0, abs=1e-12)  # towards t
    # the block's face (half size) lies on the jaws at the end
    back = CFG["block"]["size"][1] / 2 + DEMO["jaw_half_thickness"]
    assert g["end"] + d * back == pytest.approx(t)
    # the push starts behind the (rotated) block with the configured gap
    gap = np.dot(np.array(start[:2]) - g["pre"], d) - dp.half_extent(
        CFG["block"]["size"], start[2] - np.arctan2(d[1], d[0])
    ) - DEMO["jaw_half_thickness"]  # fmt: skip
    assert gap == pytest.approx(DEMO["pre_push_gap"])


def test_push_geometry_straight_y_pushes_along_y_to_the_frame_centre_line():
    # DART's friction lets the block slide only along the world axes, so the push is +y only
    start = (0.023, -0.07, np.radians(-12))
    g = dp.push_geometry(_with_aim("straight_y"), start)
    t = np.array(CFG["target_frame"]["center"])
    assert g["dir"] == pytest.approx([0.0, 1.0])
    assert g["target"] == pytest.approx([start[0], t[1]])  # x stays, y of the frame centre
    back = CFG["block"]["size"][1] / 2 + DEMO["jaw_half_thickness"]
    assert g["end"] + g["dir"] * back == pytest.approx(g["target"])
    assert g["distance"] == pytest.approx(t[1] - start[1])
    gap = (start[1] - g["pre"][1]) - dp.half_extent(
        CFG["block"]["size"], start[2] - np.pi / 2
    ) - DEMO["jaw_half_thickness"]  # fmt: skip
    assert gap == pytest.approx(DEMO["pre_push_gap"])
    assert g["pre"][0] == pytest.approx(start[0])


def test_push_geometry_center_aims_at_the_frame_centre():
    g = dp.push_geometry(_with_aim("center"), (0.02, -0.05, 0.0))
    assert g["target"] == pytest.approx(CFG["target_frame"]["center"])


def test_push_geometry_rejects_an_unknown_aim():
    with pytest.raises(ValueError, match="push_aim"):
        dp.push_geometry(_with_aim("diagonal"), (0.0, -0.05, 0.0))


def test_config_pushes_straight_and_starts_inside_the_frame_width():
    assert DEMO["push_aim"] == "straight_y"
    half = CFG["target_frame"]["inner"] / 2
    # 1 cm margin inside the frame for the x drift and turning of the block while it slides
    assert max(abs(v) for v in DEMO["start_x"]) <= half - 0.01 + 1e-12


def test_tcp_path_phases_heights_and_speeds():
    g = dp.push_geometry(CFG, (0.0, -0.05, 0.0))
    pts, dirs, phases = dp.tcp_path(CFG, g, DEMO["home_tcp"])
    names = [n for n, _, _ in phases]
    assert names == ["to_start", "descend", "before_push", "push", "after_push", "lift",
                     "settle", "to_home"]  # fmt: skip
    ph = {n: (a, b) for n, a, b in phases}
    push = pts[ph["push"][0] : ph["push"][1]]
    assert push[:, 2] == pytest.approx(Z0 + CFG["reach_check"]["tip_height"])
    step = np.linalg.norm(np.diff(push, axis=0), axis=1)
    assert step.max() <= DEMO["speed_push"] / DEMO["rate_hz"] + 1e-9
    hold = ph["settle"][1] - ph["settle"][0]
    assert hold == round(DEMO["settle_after_lift"] * DEMO["rate_hz"])
    assert pts[-1] == pytest.approx(np.array(DEMO["home_tcp"]) + [0, 0, Z0])
    assert np.linalg.norm(dirs, axis=1) == pytest.approx(1.0)
    assert dirs[ph["push"][0]] == pytest.approx(g["dir"])


@pytest.fixture(scope="module")
def planner():
    return dp.Planner(CFG)


@needs_meshes
def test_plan_is_reproducible_and_passes_its_checks(planner):
    a, b = planner.plan(3), planner.plan(3)
    assert np.array_equal(a.q, b.q) and a.block_start == b.block_start
    assert a.joint_names == dp.ARM + ["gripper"]
    q = a.q[:, :5]
    e = planner.errors(q, a.tcp, None)
    assert e["pos"].max() < CFG["reach_check"]["pos_tol"]
    lo, hi = planner.lo, planner.hi
    assert (q >= lo - 1e-12).all() and (q <= hi + 1e-12).all()
    assert np.abs(np.diff(q, axis=0)).max() <= DEMO["max_joint_step"]
    assert (a.q[:, 5] == CFG["reach_check"]["fixed"]["gripper"]).all()  # gripper stays closed
    assert 5 < a.duration_s < 20


@needs_meshes
def test_plan_pushes_straight_along_y_with_the_configured_aim(planner):
    for seed in (0, 1):
        p = planner.plan(seed)
        assert p.push_dir == pytest.approx((0.0, 1.0))
        assert p.target[0] == pytest.approx(p.block_start[0])
        assert p.target[1] == pytest.approx(CFG["target_frame"]["center"][1])


@needs_meshes
def test_push_tcp_from_forward_kinematics_follows_the_straight_line(planner):
    p = planner.plan(0)
    ph = {n: (s, e) for n, s, e in p.phases}
    s, e = ph["push"]
    pos, rot = planner._pose(p.q[s:e, :5])
    d = np.array(p.push_dir)
    rel = pos[:, :2] - pos[0, :2]
    across = rel[:, 0] * -d[1] + rel[:, 1] * d[0]
    assert np.abs(across).max() < 0.002  # stays on the push line
    assert pos[:, 2] == pytest.approx(Z0 + CFG["reach_check"]["tip_height"], abs=0.002)
    tilt = np.degrees(np.arccos(-rot[:, 2, 2]))
    assert tilt.max() <= CFG["reach_check"]["max_tilt_deg"] + 0.5


def _quat(roll=0.0, yaw=0.0):
    """(qx, qy, qz, qw) of a roll about x followed by a yaw about z (q = q_yaw * q_roll)."""
    cr, sr, cy, sy = np.cos(roll / 2), np.sin(roll / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    return (cy * sr, sy * sr, sy * cr, cy * cr)


def test_judge_inside_outside_flipped_and_off_the_table():
    t = CFG["target_frame"]["center"]
    zc = Z0 + CFG["block"]["size"][2] / 2
    half = CFG["target_frame"]["inner"] / 2
    ok = dp.judge(CFG, (t[0] + 0.01, t[1] - 0.02, zc, *_quat(yaw=0.3)))
    assert ok["success"] and ok["error_m"] == pytest.approx(np.hypot(0.01, 0.02), abs=1e-4)
    assert ok["yaw_deg"] == pytest.approx(np.degrees(0.3), abs=0.01)
    assert dp.judge(CFG, (t[0], t[1] + half - 1e-4, zc, *_quat()))["success"]
    assert not dp.judge(CFG, (t[0], t[1] + half + 1e-3, zc, *_quat()))["inside"]
    assert not dp.judge(CFG, (t[0], t[1], zc, *_quat(roll=np.radians(20))))["upright"]
    assert not dp.judge(CFG, (t[0], t[1], zc - 0.01, *_quat()))["on_table"]
