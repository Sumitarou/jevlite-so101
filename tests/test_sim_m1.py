"""M1 in headless Gazebo: runs scripts/sim/check_m1.py with the system Python and ROS 2.

Marker `sim` (excluded by default). ROS is never imported into the uv environment: the check
runs in a separate `bash -c "source ... && python3 ..."` with the virtualenv removed from the
environment, and only its JSON result is read here.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROS_SETUP = Path("/opt/ros/jazzy/setup.bash")
WS_SETUP = ROOT / "sim_ws" / "install" / "setup.bash"
RESULT = ROOT / "outputs" / "sim" / "m1_result.json"

pytestmark = pytest.mark.sim


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    venv = env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    if venv:
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep) if not p.startswith(venv)
        )
    return env


@pytest.fixture(scope="module")
def result():
    if not ROS_SETUP.is_file() or not WS_SETUP.is_file():
        pytest.skip("ROS 2 Jazzy or the built sim_ws is missing")
    RESULT.unlink(missing_ok=True)
    cmd = (
        f"source {ROS_SETUP} && source {WS_SETUP} && "
        "timeout --signal=INT --kill-after=20 180 /usr/bin/python3 scripts/sim/check_m1.py"
    )
    proc = subprocess.run(
        ["bash", "-c", cmd], cwd=ROOT, env=clean_env(), capture_output=True, text=True,
        timeout=240,
    )  # fmt: skip
    assert RESULT.is_file(), proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.returncode, json.loads(RESULT.read_text())


def test_check_m1_passes_and_leaves_nothing_running(result):
    code, r = result
    assert code == 0, r.get("error")
    assert r["ok"] and r["reached_all"]
    assert r["shutdown"]["leftover"] is False


def test_every_joint_reaches_its_target(result):
    _, r = result
    chk = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())["check"]
    assert set(r["joints"]) == set(chk["targets"])
    for name, j in r["joints"].items():
        assert j["err_end_rad"] < chk["tolerance"], name
        assert j["t95_sim_s"] is not None, name


def test_camera_frame_is_640x480_and_not_uniform(result):
    _, r = result
    cam = r["camera"]
    assert (cam["width"], cam["height"]) == (640, 480)
    assert cam["std"] > 5.0
    assert (ROOT / "outputs" / "sim" / "m1_camera.png").is_file()


def test_gz_ros2_control_parameters_are_the_configured_ones(result):
    _, r = result
    ctrl = yaml.safe_load((ROOT / "configs" / "sim_controllers.yaml").read_text())
    gain = ctrl["gz_ros2_control"]["ros__parameters"]["position_proportional_gain"]
    assert float(r["parameters"]["position_proportional_gain"]) == pytest.approx(gain)
