"""scripts/sim/kinematics.py and check_reach.py (no ROS needed)."""

import json
import struct

import numpy as np
import pytest
import yaml

from scripts.sim import check_reach as cr
from scripts.sim import kinematics as kin

CFG = yaml.safe_load(kin.SIM_YAML.read_text())
HAVE_MESHES = all(
    (kin.DESC / "meshes" / n).is_file()
    for n in json.loads((kin.DESC / "assets.lock").read_text())["meshes"]
)
needs_meshes = pytest.mark.skipif(not HAVE_MESHES, reason="SO-101 meshes not fetched")


@pytest.fixture(scope="module")
def robot():
    return kin.Robot()


def test_rpy_matrix_pitch_turns_x_downward():
    r = kin.rpy_matrix(0.0, 0.5, 0.0)
    assert r @ [1, 0, 0] == pytest.approx([np.cos(0.5), 0, -np.sin(0.5)])
    rot = kin.rpy_matrix(0.1, 0.2, 0.3)
    assert rot @ rot.T == pytest.approx(np.eye(3))


# ---------- STL ----------
def _tri(v):
    return [(0.0, 0.0, 1.0), *v]


def test_stl_bbox_binary_and_ascii(tmp_path):
    tris = [_tri([(0, 0, 0), (1, 2, 0), (0, 0, 3)]), _tri([(-1, 0.5, 0), (0, 0, 0), (0, 0, -2)])]
    binary = tmp_path / "b.stl"
    data = b"\0" * 80 + struct.pack("<I", len(tris))
    for t in tris:
        data += struct.pack("<12fH", *[c for vec in t for c in vec], 0)
    binary.write_bytes(data)
    ascii_ = tmp_path / "a.stl"
    lines = ["solid x"]
    for t in tris:
        lines += ["facet normal 0 0 1", "outer loop"]
        lines += [f"vertex {x} {y} {z}" for x, y, z in t[1:]]
        lines += ["endloop", "endfacet"]
    ascii_.write_text("\n".join(lines + ["endsolid x"]))
    for f in (binary, ascii_):
        assert kin.stl_bbox(f) == pytest.approx(np.array([[-1, 0, -2], [1, 2, 3]]))
        assert len(kin.stl_vertices(f)) == 5  # (0, 0, 0) is shared


def test_support_points_keep_the_extremes():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(5000, 3))
    sp = kin.support_points(v, 300)
    assert len(sp) < len(v)
    for d in (np.array([0, 0, -1.0]), np.array([1.0, 1, 0]) / 2**0.5):
        assert (sp @ d).max() == pytest.approx((v @ d).max())


# ---------- configured layout ----------
def test_configured_table_is_the_real_one():
    t = CFG["table"]
    assert t["white"] == [0.20, 0.32]  # depth (x) x width (y), measured
    assert t["band_width"] == 0.05


@needs_meshes
def test_configured_base_sits_outside_the_band_with_the_recorded_gap(robot):
    fp = cr.to_table(*cr.footprint_local(robot).T, CFG["robot"])
    band_outer = CFG["table"]["white"][1] / 2 + CFG["table"]["band_width"]
    assert not cr.overlaps_band(fp, CFG["table"])
    assert -fp[:, 1].max() - band_outer == pytest.approx(0.01, abs=1e-3)  # docs/sim_layout.md
    assert fp[:, 0].mean() == pytest.approx(0.0, abs=1e-3)  # centred on the white area's depth


def _z(h):
    return cr.surface_z(CFG) + h  # heights are measured from the white area's top


@needs_meshes
def test_configured_base_reaches_the_pushing_region_with_the_tool_down(robot):
    rc = cr.with_surface(CFG)
    base = kin.transform(CFG["robot"]["xyz"], CFG["robot"]["rpy"])
    c = cr.white_corners(CFG)
    tf = CFG["target_frame"]["center"]
    end = [tf[0], tf[1] - 0.031]  # TCP at the end of a push along +y into the frame
    pts = np.array([[*c["near-right"], _z(rc["tip_height"])],
                    [*c["far-right"], _z(rc["tip_height"])],
                    [0.0, 0.0, _z(rc["tip_height"])],
                    [*end, _z(rc["tip_height"])],
                    [end[0], end[1] + 0.02, _z(rc["tip_height"])],  # 2 cm margin beyond
                    [*c["near-left"], _z(rc["tip_height"])]])  # fmt: skip
    res = cr.solve_ik(robot, pts, base, rc, cr.push_dirs(CFG, pts))
    assert res["reachable"].tolist() == [True, True, True, True, True, False]


# ---------- IK and proposals ----------
def _rc(**over):
    return {**cr.with_surface(CFG), "seeds": 8, "refine_passes": 1, "iters": 150, **over}


def test_tool_axis_and_residual():
    lean = np.array([[0.0, 1.0]])
    assert kin.tool_axis(lean, 0.0)[0] == pytest.approx([0, 0, -1])
    assert kin.tool_axis(lean, np.pi / 2)[0] == pytest.approx([0, 1, 0], abs=1e-12)
    # a tool pointing straight down with the jaw line (x) along +x, pushing along +y: no error
    rot = np.array([[[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]])
    pos = np.array([[0.1, 0.2, 0.01]])
    r = kin.tool_residual(pos, rot, pos, lean)
    assert np.abs(r).max() == pytest.approx(0.0)
    e = kin.tool_errors(pos, rot, pos, np.array([[1.0, 0.0]]))  # push along the jaw line
    assert e["tilt"][0] == pytest.approx(0.0) and e["jaw"][0] == pytest.approx(np.pi / 2)


@needs_meshes
def test_tcp_is_the_lowest_point_of_the_closed_jaws_when_pointing_down(robot):
    rc = CFG["reach_check"]
    q = {n: np.zeros(1) for n in robot.limits}
    q.update({k: np.array([float(v)]) for k, v in rc["fixed"].items()})
    tf = robot.link_transforms(q, np.eye(4))
    inv = np.linalg.inv(tf[rc["tip_link"]][0])
    pts = []
    for link in ("gripper_link", "moving_jaw_so101_v1_link"):
        t = inv @ tf[link][0]
        pts.append(robot.vertices[link] @ t[:3, :3].T + t[:3, 3])
    pts = np.concatenate(pts)  # in the tip frame; +z points out of the jaws
    tcp = np.array(rc["tcp"])
    assert tcp[2] == pytest.approx(pts[:, 2].max(), abs=1e-3)
    tipzone = pts[pts[:, 2] > tcp[2] - 0.015]
    assert tcp[0] == pytest.approx((tipzone[:, 0].min() + tipzone[:, 0].max()) / 2, abs=2e-3)


@needs_meshes
def test_ik_reaches_with_the_tool_down_and_not_a_far_point(robot):
    base = kin.transform(CFG["robot"]["xyz"], CFG["robot"]["rpy"])
    rc = _rc()
    pts = np.array([[0.05, 0.05, _z(0.005)], [0.05, -0.1, _z(0.005)], [-0.9, 0.0, _z(0.005)]])
    dirs = cr.push_dirs(CFG, pts)
    res = cr.solve_ik(robot, pts, base, rc, dirs)
    assert res["reachable"].tolist() == [True, True, False]
    assert not res["converged"][2]
    jv = {**{k: np.array([float(v)]) for k, v in rc["fixed"].items()},
          **{j: res["q"][0:1, i] for i, j in enumerate(rc["ik_joints"])}}  # fmt: skip
    pos, rot = kin.tool_pose(robot, jv, base, rc["tip_link"], rc["tcp"])
    e = kin.tool_errors(pos, rot, pts[:1], dirs[:1])
    assert e["pos"][0] < rc["pos_tol"]
    assert e["tilt"][0] <= np.radians(rc["max_tilt_deg"])
    assert e["jaw"][0] < np.radians(rc["rot_tol_deg"])


@needs_meshes
def test_ik_rejects_a_pose_through_the_white_area(robot):
    base = kin.transform(CFG["robot"]["xyz"], CFG["robot"]["rpy"])
    pts = np.array([[0.05, 0.0, 0.0]])  # TCP 2 mm below the white area's top
    res = cr.solve_ik(robot, pts, base, _rc(), cr.push_dirs(CFG, pts))
    assert res["converged"][0] and not res["reachable"][0]


def _synthetic_map(radius):
    step = 0.01
    fs = np.round(np.arange(-0.05, 0.6 + 1e-9, step), 4)
    ls = np.round(np.arange(-0.4, 0.4 + 1e-9, step), 4)
    ff, ll = np.meshgrid(fs, ls, indexing="ij")
    return fs, ls, (np.hypot(ff, ll) <= radius) & (ff >= 0.02)  # half disc in front of the base


SYN_CFG = {
    "table": {"white": [0.20, 0.32], "band_width": 0.05, "overlay_thickness": 0.001},
    "robot": {"xyz": [0.0, -0.295, 0.0], "rpy": [0.0, 0.0, np.pi / 2]},
    "reach_check": {"grid_step": 0.01, "tip_height": 0.015, "base_shift_max": 0.05},
}
FP_LOCAL = np.array([[-0.031, -0.055], [0.065, -0.055], [0.065, 0.055], [-0.031, 0.055]])


def test_base_frame_conversions_round_trip():
    rb = SYN_CFG["robot"]
    f, lat = cr.to_base(np.array([[0.0, 0.0], [0.1, -0.295]]), rb)
    assert f == pytest.approx([0.295, 0.0], abs=1e-9)  # the arm points to +y
    assert lat == pytest.approx([0.0, -0.1], abs=1e-9)  # +x (far side) is the base's right
    xy = cr.to_table(f, lat, rb)
    assert xy == pytest.approx(np.array([[0.0, 0.0], [0.1, -0.295]]), abs=1e-9)
    fp = cr.to_table(FP_LOCAL[:, 0], FP_LOCAL[:, 1], rb)
    assert fp[:, 1].max() == pytest.approx(-0.23)  # 0.02 outside the band's outer edge
    assert not cr.overlaps_band(fp, SYN_CFG["table"])
    assert cr.overlaps_band(fp + [0, 0.03], SYN_CFG["table"])


def test_lookup_and_proposals_on_a_synthetic_map():
    fs, ls, reach = _synthetic_map(0.44)
    got = cr.lookup(fs, ls, reach, np.array([0.3, 0.5, 0.3, 0.435]), np.array([0, 0, 0.9, 0.0]))
    assert got.tolist() == [True, False, False, True]
    # between cells: 0.445 is reachable only if both 0.44 and 0.45 are (0.45 > radius 0.44)
    assert not cr.lookup(fs, ls, reach, np.array([0.445]), np.array([0.0]))[0]
    p = cr.proposals(SYN_CFG, fs, ls, reach, FP_LOCAL)
    # left corners: forward 0.455 - dy, lateral 0.10 -> need dy >= 0.455 - sqrt(.44^2 - .1^2)
    # = 0.0265; the conservative lookup (cells on both sides) gives one grid step more
    full = p["1_smallest_shift_reaching_all"]
    assert full["shift"] == pytest.approx([0.0, 0.04])
    assert full["coverage"] == 1.0 and all(full["corners"].values())
    assert full["overlaps_band"]  # 0.04 closer puts the footprint on the band
    best = p["1_best_shift_outside_band"]
    assert not best["overlaps_band"] and best["coverage"] < 1.0
    assert not best["corners"]["near-left"] and not best["corners"]["far-left"]


def test_work_strip_is_the_reachable_full_depth_part_next_to_the_base():
    fs, ls, reach = _synthetic_map(0.44)
    grid = cr.white_grid(SYN_CFG)[:, :2]
    ok = cr.reachable_at(SYN_CFG, fs, ls, reach, grid)
    strip = cr.work_strip(SYN_CFG, ok)
    # full depth (x = +-0.10) is reachable while 0.295 + y <= sqrt(.44^2 - .1^2) = 0.4285,
    # i.e. up to the column y = 0.13; the conservative lookup stops one column earlier
    assert strip["strip_y"] == pytest.approx([-0.16, 0.12])
    assert strip["fraction_of_white"] == pytest.approx(0.28 / 0.32, abs=1e-3)
