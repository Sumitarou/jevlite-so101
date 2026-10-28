"""so101_description: the vendored URDF, the asset lock and the fetch script (no ROS needed)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from scripts.sim import fetch_so101_assets as fetch

PKG = Path(__file__).resolve().parents[1] / "sim_ws" / "src" / "so101_description"
URDF = PKG / "urdf" / "so101_new_calib.urdf"

# from the URDF at the pinned commit (docs: M1 plan)
JOINTS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}


@pytest.fixture(scope="module")
def robot():
    return ET.parse(URDF).getroot()


def test_movable_joints_and_limits(robot):
    movable = {
        j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
        for j in robot.findall("joint")
        if j.get("type") != "fixed"
    }
    assert movable == pytest.approx(JOINTS)


def test_mesh_uris_match_the_lock(robot):
    lock = json.loads((PKG / "assets.lock").read_text())
    uris = {m.get("filename") for m in robot.iter("mesh")}
    assert all(u.startswith(fetch.MESH_PREFIX) for u in uris)
    assert {u.removeprefix(fetch.MESH_PREFIX) for u in uris} == set(lock["meshes"])
    assert all(len(h) == 64 for h in lock["meshes"].values())
    assert lock["commit"] == fetch.COMMIT


def test_links_with_geometry_have_sane_inertia(robot):
    for link in robot.findall("link"):
        if not link.findall("visual"):
            continue  # frames without geometry are merged into their parent by gz sdf
        inertial = link.find("inertial")
        assert inertial is not None, link.get("name")
        mass = float(inertial.find("mass").get("value"))
        i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        m = np.array(
            [
                [i["ixx"], i["ixy"], i["ixz"]],
                [i["ixy"], i["iyy"], i["iyz"]],
                [i["ixz"], i["iyz"], i["izz"]],
            ]
        )
        ev = np.linalg.eigvalsh(m)
        assert mass >= 0.005, link.get("name")
        assert ev[0] > 1e-7, link.get("name")  # positive definite, not vanishingly small
        assert ev[0] + ev[1] >= ev[2] * (1 - 1e-6), link.get("name")  # triangle inequality


def test_rewrite_only_changes_mesh_paths():
    src = '<mesh filename="assets/a_v1.stl"/><mesh filename="assets/b.stl"/><x y="assets/"/>'
    out, names = fetch.rewrite_mesh_paths(src)
    assert names == ["a_v1.stl", "b.stl"]
    assert out == (
        f'<mesh filename="{fetch.MESH_PREFIX}a_v1.stl"/>'
        f'<mesh filename="{fetch.MESH_PREFIX}b.stl"/><x y="assets/"/>'
    )


def test_license_and_notice_record_the_source():
    assert "Apache License" in (PKG / "LICENSE").read_text()
    notice = (PKG / "NOTICE").read_text()
    assert fetch.COMMIT in notice and "TheRobotStudio/SO-ARM100" in notice


def test_package_declares_apache_license():
    pkg = ET.parse(PKG / "package.xml").getroot()
    assert pkg.find("name").text == "so101_description"
    assert pkg.find("license").text == "Apache-2.0"


def test_meshes_are_not_tracked_by_git():
    from scripts.capture_samples import is_git_ignored

    assert is_git_ignored(PKG / "meshes" / "base_so101_v2.stl")
