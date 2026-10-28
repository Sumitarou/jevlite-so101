"""configs/sim.yaml, configs/sim_controllers.yaml and the so101_gazebo package (no ROS needed)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SIM = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
CTRL = yaml.safe_load((ROOT / "configs" / "sim_controllers.yaml").read_text())
URDF = ROOT / "sim_ws" / "src" / "so101_description" / "urdf" / "so101_new_calib.urdf"
GZ_PKG = ROOT / "sim_ws" / "src" / "so101_gazebo"


def movable_joints():
    r = ET.parse(URDF).getroot()
    return {
        j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
        for j in r.findall("joint")
        if j.get("type") != "fixed"
    }


def test_controllers_cover_every_movable_joint_once():
    arm = CTRL["arm_position_controller"]["ros__parameters"]["joints"]
    grip = CTRL["gripper_position_controller"]["ros__parameters"]["joints"]
    assert sorted(arm + grip) == sorted(movable_joints())
    assert len(arm + grip) == len(set(arm + grip))
    cm = CTRL["controller_manager"]["ros__parameters"]
    assert cm["use_sim_time"] is True
    for name in (
        "joint_state_broadcaster",
        "arm_position_controller",
        "gripper_position_controller",
    ):
        assert "type" in cm[name]


def test_position_gain_is_explicit():
    gz = CTRL["gz_ros2_control"]["ros__parameters"]
    assert 0 < gz["position_proportional_gain"] <= 1
    assert isinstance(gz["hold_joints"], bool)


def test_check_targets_are_within_limits_with_margin():
    limits = movable_joints()
    poses = {"targets": SIM["check"]["targets"], **SIM["check"]["poses"]}
    for pose, targets in poses.items():
        assert set(targets) == set(limits), pose
        for name, value in targets.items():
            lo, hi = limits[name]
            assert lo + 0.1 <= value <= hi - 0.1, (pose, name)


def test_ros2_control_block_lists_the_same_joints():
    text = (GZ_PKG / "urdf" / "so101_gazebo.urdf.xacro").read_text()
    for name in movable_joints():
        assert f'<xacro:position_joint name="{name}"/>' in text


def test_camera_is_c920_like_and_looks_down_at_an_angle():
    cam = SIM["camera"]
    assert (cam["width"], cam["height"]) == (640, 480)
    assert cam["update_rate"] == 30
    assert 1.1 < cam["hfov"] < 1.35  # about 70 degrees
    pitch = cam["rpy"][1]
    assert 0.5 < pitch < 1.3  # oblique, not straight down (pi/2) and not horizontal
    assert cam["xyz"][2] > 0  # above the table top


def test_table_geometry_is_consistent():
    t = SIM["table"]
    assert t["band_width"] == pytest.approx(0.05)
    assert t["white"][0] + 2 * t["band_width"] <= t["size"][0]
    assert t["white"][1] + 2 * t["band_width"] <= t["size"][1]


def test_launch_stops_with_a_fetch_hint_when_meshes_are_missing():
    text = (GZ_PKG / "launch" / "sim.launch.py").read_text()
    assert "scripts/sim/fetch_so101_assets.py" in text
    assert "missing_meshes" in text
    assert "-s -r --headless-rendering" in text  # server only, never the GUI
