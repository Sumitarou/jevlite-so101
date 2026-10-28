"""M4a intruder: zones, path, clearance from the frame and block path, camera view (no ROS)."""

import math

import numpy as np
import pytest
import yaml

from jevlite.api import Choice
from scripts.sim.demo_plan import half_extent
from scripts.sim.fit_camera import project
from scripts.sim.intruder import (
    build_scenarios,
    choice_of,
    edge_distance,
    load_m4a,
    rect_circle_overlap,
    zone_of,
)
from scripts.sim.kinematics import SIM_YAML

SIM = yaml.safe_load(SIM_YAML.read_text())
M4A = load_m4a()
R = SIM["intruder"]["radius"]
WX, WY = SIM["table"]["white"][0] / 2, SIM["table"]["white"][1] / 2
B = SIM["table"]["band_width"]


def test_rect_circle_overlap_touching_counts():
    assert rect_circle_overlap(1, 1, 1.5, 0, 0.5)  # touches the side
    assert not rect_circle_overlap(1, 1, 1.5001, 0, 0.5)
    d = 0.5 / math.sqrt(2)
    assert rect_circle_overlap(1, 1, 1 + d, 1 + d, 0.5 + 1e-12)  # corner
    assert not rect_circle_overlap(1, 1, 1 + d + 1e-3, 1 + d + 1e-3, 0.5)


@pytest.mark.parametrize(
    "x, y, zone",
    [(0.0, 0.0, "center"),
     (0.0, WY + R, "center"),          # touching the white area: STOP side
     (0.0, WY + R + 1e-4, "edge"),
     (0.0, WY + B + R, "edge"),        # touching the band's outer edge
     (0.0, WY + B + R + 1e-4, "none"),
     (WX + B + R + 1e-4, 0.0, "none"),
     (WX + 0.5 * R, WY + 0.5 * R, "center"),  # near the corner, overlapping it
     (WX + R, WY + R, "edge")],        # diagonal from the corner: |d| = R * sqrt(2) > R
)  # fmt: skip
def test_zone_of_boundaries(x, y, zone):
    assert zone_of(x, y, SIM) == zone


def test_edge_distance_sign():
    assert edge_distance(0, 0, SIM) < 0
    assert edge_distance(0, WY + R, SIM) == pytest.approx(0, abs=1e-12)
    assert edge_distance(0, WY + R + 0.03, SIM) == pytest.approx(0.03)


def test_zone_choice_mapping_is_ordered_by_severity():
    assert [choice_of(z, M4A) for z in ("none", "edge", "center")] == [
        Choice.CONTINUE, Choice.SLOW, Choice.STOP]  # fmt: skip


def test_named_places_are_in_their_zones():
    p = M4A["path"]
    zones = {name: zone_of(p["x"], y, SIM) for name, y in p["places"].items()}
    assert zones == {"out": "none", "edge": "edge", "center": "center", "boundary": "edge"}


def test_path_keeps_clear_of_the_target_frame_and_the_block_path():
    """The rod's footprint never overlaps the blue frame or the area the block can sweep."""
    x_min = M4A["path"]["x"] - R
    tf = SIM["target_frame"]
    frame_half_outer = tf["inner"] / 2 + tf["line_width"]
    assert x_min > abs(tf["center"][0]) + frame_half_outer
    demo, size = SIM["demo"], SIM["block"]["size"]
    yaws = np.radians(np.linspace(-demo["start_yaw_deg"], demo["start_yaw_deg"], 61))
    block_x = max(abs(v) for v in demo["start_x"]) + max(half_extent(size, a) for a in yaws)
    bs = M4A["block_start"]
    block_x = max(block_x, abs(bs[0]) + max(half_extent(size, bs[2]), size[0] / 2))
    assert x_min > block_x + 0.005  # at least 5 mm apart
    assert M4A["path"]["x"] + R < WX  # still over the white area along x


def _uv(points):
    c = SIM["camera"]
    u, v, d = project(np.asarray(points, float), c["xyz"][0], c["xyz"][2], c["rpy"][1],
                      c["width"], c["height"], c["hfov"])  # fmt: skip
    return u[0], v[0], d[0]


def _rod_points(x, y, z_center):
    half = SIM["intruder"]["length"] / 2
    return [(x + dx, y + dy, z_center + dz) for dx in (-R, R) for dy in (-R, R)
            for dz in (-half, half)]  # fmt: skip


def test_out_and_park_are_out_of_view_and_edge_center_in_view():
    c = SIM["camera"]
    sc = build_scenarios(M4A, SIM)[0]
    for name in ("out",):
        u, v, d = _uv(_rod_points(*sc.points[name]))
        assert np.all((u < 0) | (u > c["width"]) | (v < 0) | (v > c["height"]) | (d <= 0)), name
    u, v, d = _uv(_rod_points(*sc.park))
    assert np.all((u < 0) | (u > c["width"]) | (v < 0) | (v > c["height"]) | (d <= 0))
    for name in ("edge", "center", "boundary"):
        x, y, z = sc.points[name]
        u, v, d = _uv([(x, y, z - SIM["intruder"]["length"] / 2)])  # the foot of the rod
        assert 0 < u[0] < c["width"] and 0 < v[0] < c["height"] and d[0] > 0, name


def test_scenarios_build_and_interpolate():
    scs = {s.id: s for s in build_scenarios(M4A, SIM)}
    assert {"base_1", "edge_1", "center_1", "stale_latency", "flicker"} <= set(scs)
    base = scs["base_1"]
    assert not base.has_intruder and base.position(5.0) == base.park and base.zone(5, SIM) == "none"
    c = scs["center_1"]
    t0, t1 = c.keys[0][0], c.keys[1][0]
    p0, p1 = np.array(c.points[c.keys[0][1]]), np.array(c.points[c.keys[1][1]])
    assert np.allclose(c.position((t0 + t1) / 2), (p0 + p1) / 2)
    assert c.position(-1) == c.points["out"] and c.position(99) == c.points["out"]
    assert c.zone(5.0, SIM) == "center" and c.zone(0.0, SIM) == "none"
    assert scs["stale_latency"].judge["latency_s"] == 0.15
    assert scs["stale_latency"].judge["mode"] == M4A["judge"]["mode"]  # defaults merged


def test_build_scenarios_rejects_bad_keys():
    bad = {**M4A, "scenarios": [{"id": "a", "keys": [[2, "out"], [1, "edge"]]}]}
    with pytest.raises(ValueError, match="increase"):
        build_scenarios(bad, SIM)
    bad = {**M4A, "scenarios": [{"id": "a", "keys": [[1, "nowhere"]]}]}
    with pytest.raises(ValueError, match="unknown"):
        build_scenarios(bad, SIM)
    bad = {**M4A, "scenarios": [{"id": "a"}, {"id": "a"}]}
    with pytest.raises(ValueError, match="duplicate"):
        build_scenarios(bad, SIM)


def test_world_rod_is_visual_only_without_gravity_and_publishes_its_pose():
    from pathlib import Path

    gz = Path(SIM_YAML).parents[1] / "sim_ws" / "src" / "so101_gazebo"
    world = (gz / "worlds" / "table.sdf.xacro").read_text()
    model = world[world.index("<model name=\"${itr['name']}\">") :]
    model = model[: model.index("</model>")]
    assert "<collision" not in model and "<gravity>false</gravity>" in model
    assert "<cylinder>" in model and "gz::sim::systems::PosePublisher" in model
    launch = (gz / "launch" / "sim.launch.py").read_text()
    assert "gz_intruder_pose" in launch and 'intruder["pose_topic"]' in launch
