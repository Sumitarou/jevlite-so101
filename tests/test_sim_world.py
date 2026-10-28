"""Arm colours, the block and the target frame (configs/sim.yaml, world xacro, launch; no ROS)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from scripts.sim import urdf_tools

ROOT = Path(__file__).resolve().parents[1]
SIM = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
URDF = ROOT / "sim_ws" / "src" / "so101_description" / "urdf" / "so101_new_calib.urdf"
GZ_PKG = ROOT / "sim_ws" / "src" / "so101_gazebo"
WORLD = (GZ_PKG / "worlds" / "table.sdf.xacro").read_text()


def _materials(xml: str) -> dict[str, list[str]]:
    root = ET.fromstring(xml)
    out = {}
    for link in root.findall("link"):
        out[link.get("name")] = [v.find("material").get("name") for v in link.findall("visual")]
    return out


def test_recolor_makes_the_base_grey_and_the_arm_white_and_keeps_the_motors():
    before = URDF.read_text()
    after = urdf_tools.recolor(before, SIM["appearance"])
    old, new = _materials(before), _materials(after)
    printed = SIM["appearance"]["printed_material"]
    for link, mats in old.items():
        for o, n in zip(mats, new[link], strict=True):
            if o != printed:
                assert n == o  # e.g. the black motors stay
            elif link == SIM["appearance"]["base_link"]:
                assert n == urdf_tools.BASE_MATERIAL
            else:
                assert n == urdf_tools.ARM_MATERIAL
    root = ET.fromstring(after)
    colors = {m.get("name"): m.find("color").get("rgba") for m in root.findall("material")}
    assert colors[urdf_tools.ARM_MATERIAL] == "0.92 0.92 0.92 1"
    assert colors[urdf_tools.BASE_MATERIAL] == "0.45 0.45 0.45 1"
    assert printed not in [n for mats in new.values() for n in mats]


def test_recolor_rejects_an_unknown_material_name():
    with pytest.raises(ValueError, match="no visual uses"):
        urdf_tools.recolor(URDF.read_text(), {**SIM["appearance"], "printed_material": "nope"})


def test_target_frame_is_visual_only_and_inside_the_white_area():
    macro = WORLD[WORLD.index('<xacro:macro name="frame_line"') :]
    macro = macro[: macro.index("</xacro:macro>")]
    assert "<visual" in macro and "<collision" not in macro
    tf, (wx, wy) = SIM["target_frame"], SIM["table"]["white"]
    half_outer = tf["inner"] / 2 + tf["line_width"]
    assert abs(tf["center"][0]) + half_outer <= wx / 2
    assert abs(tf["center"][1]) + half_outer <= wy / 2
    assert tf["center"][1] > 0  # left of the centre (seen from the camera)
    b, g, r = tf["rgba"][2], tf["rgba"][1], tf["rgba"][0]
    assert b > 0.6 and r < 0.3 and g < 0.5  # blue


def test_block_matches_the_real_one_and_has_physics():
    blk = SIM["block"]
    assert blk["size"] == [0.05, 0.05, 0.01]
    assert 0 < blk["mass"] < 0.2 and 0 < blk["mu"] <= 1
    model = WORLD[WORLD.index("<model name=\"${blk['name']}\">") :]
    model = model[: model.index("</model>")]
    for key in ("<inertial>", "<mass>${bm}</mass>", "<mu>${blk['mu']}</mu>",
                "gz::sim::systems::PosePublisher", "<collision"):  # fmt: skip
        assert key in model
    assert "${bm * (by * by + bz * bz) / 12}" in model  # solid box inertia


def test_launch_recolours_and_bridges_the_block_pose_and_set_pose():
    text = (GZ_PKG / "launch" / "sim.launch.py").read_text()
    assert 'recolor(robot, cfg["appearance"])' in text
    assert "/world/table/set_pose@ros_gz_interfaces/srv/SetEntityPose" in text
    assert "geometry_msgs/msg/PoseStamped[gz.msgs.Pose" in text
    assert "<exec_depend>ros_gz_interfaces</exec_depend>" in (GZ_PKG / "package.xml").read_text()
