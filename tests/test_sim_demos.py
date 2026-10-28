"""Push demonstrations in headless Gazebo: scripts/sim/gen_demos.py (marker `sim`).

Runs one episode and its replay (same seed) with the system Python and ROS 2, outside the uv
environment, and checks the bags (topics, rates), episode.json and the reproducibility.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROS_SETUP = Path("/opt/ros/jazzy/setup.bash")
WS_SETUP = ROOT / "sim_ws" / "install" / "setup.bash"
SIM = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
SEED = 1

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


def ros(cmd: str, timeout: float) -> subprocess.CompletedProcess:
    full = f"source {ROS_SETUP} && source {WS_SETUP} && {cmd}"
    return subprocess.run(["bash", "-c", full], cwd=ROOT, env=clean_env(), capture_output=True,
                          text=True, timeout=timeout)  # fmt: skip


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not ROS_SETUP.is_file() or not WS_SETUP.is_file():
        pytest.skip("ROS 2 Jazzy or the built sim_ws is missing")
    out = {}
    for name, args in (("first", f"--episodes 1 --seed {SEED}"), ("replay", f"--replay {SEED}")):
        d = tmp_path_factory.mktemp(name)
        p = ros(f"timeout --signal=INT --kill-after=30 600 /usr/bin/python3 "
                f"scripts/sim/gen_demos.py {args} --out {d}", timeout=700)  # fmt: skip
        assert (d / "summary.json").is_file(), p.stdout[-2000:] + p.stderr[-2000:]
        out[name] = (p.returncode, d, json.loads((d / "summary.json").read_text()))
    return out


def _episode(d: Path) -> tuple[Path, dict]:
    bag = next(d.glob("ep_0000_seed*"))
    return bag, json.loads((bag / "episode.json").read_text())


def test_runs_finish_and_leave_nothing_running(runs):
    for code, _, summary in runs.values():
        assert code == 0, summary.get("error")
        assert summary["shutdown"]["leftover"] is False
        assert summary["episodes"][0]["recorder_stop"]["leftover"] is False


def test_bag_has_every_topic_at_its_rate(runs):
    bag, info = _episode(runs["first"][1])
    text = ros(f"ros2 bag info {bag}", timeout=60).stdout
    counts = {m.group(1): int(m.group(2))
              for m in re.finditer(r"Topic: (\S+) \| Type: \S+ \| Count: (\d+)", text)}  # fmt: skip
    assert set(SIM["demo"]["record"]["topics"]) <= set(counts)
    dur = float(re.search(r"Duration:\s+([0-9.]+)s", text).group(1))
    assert counts["/demo/action"] == info["ticks"]
    assert counts["/overhead_camera/image"] / dur == pytest.approx(30, rel=0.1)
    assert counts["/block/pose"] / dur == pytest.approx(30, rel=0.1)
    assert counts["/joint_states"] / dur == pytest.approx(100, rel=0.1)
    assert counts["/demo/episode"] == 2  # start and end


def test_episode_json_records_the_seed_result_and_phases(runs):
    _, info = _episode(runs["first"][1])
    assert info["seed"] == SEED and info["status"] == "done"
    assert set(info["result"]) >= {"success", "inside", "error_m", "yaw_deg", "tilt_deg"}
    assert list(info["phase_start_s"])[0] == "to_start" and "push" in info["phase_start_s"]
    assert info["bag_bytes"] > 0


def test_replay_reproduces_the_episode(runs):
    _, a = _episode(runs["first"][1])
    _, b = _episode(runs["replay"][1])
    assert a["block_start"] == b["block_start"] and a["ticks"] == b["ticks"]
    pa, pb = a["final_block_pose"], b["final_block_pose"]
    assert abs(pa[0] - pb[0]) < 0.005 and abs(pa[1] - pb[1]) < 0.005
    assert a["result"]["success"] == b["result"]["success"]
