"""M4a gate in headless Gazebo: scripts/sim/gate_demo_sim.py with the hold policy (marker `sim`).

The rod is moved by set_pose each tick (center_1: out -> band -> white area by t = 4.5 s); the
oracle judge turns that into STOP; the measured rod pose follows the script; nothing is left.
"""

import json

import pytest
from test_sim_demos import ROS_SETUP, WS_SETUP, ros

pytestmark = pytest.mark.sim


def test_gate_runner_moves_the_rod_stops_the_arm_and_cleans_up(tmp_path):
    if not ROS_SETUP.is_file() or not WS_SETUP.is_file():
        pytest.skip("ROS 2 Jazzy or the built sim_ws is missing")
    p = ros(f"timeout --signal=INT --kill-after=30 600 /usr/bin/python3 "
            f"scripts/sim/gate_demo_sim.py --fake-policy hold --duration 6 --only center_1 "
            f"--out-base {tmp_path}",
            timeout=700)  # fmt: skip
    outs = list(tmp_path.iterdir())
    assert len(outs) == 1, p.stdout[-2000:] + p.stderr[-2000:]
    out = outs[0]
    s = json.loads((out / "summary.json").read_text())
    assert p.returncode == 0, s.get("error")
    assert s["episodes"] == 1 and s["leftover_processes"] == []
    assert not s["shutdown"]["leftover"] and all(not x["leftover"] for x in s["server_stops"])
    ep = json.loads((out / "results.jsonl").read_text().splitlines()[0])
    assert ep["scenario"] == "center_1"
    assert ep["intruder_meas_err_max_m"] < 0.02  # the rod follows the script (within a tick)
    ticks = [json.loads(x) for x in (out / "bags" / ep["bag"] / "ticks.jsonl").read_text()
             .splitlines()]  # fmt: skip
    g = {round(t["t"], 2): t["gate"] for t in ticks}
    assert g[3.0]["state"] == "CONTINUE"  # resumed after the start hold, rod still outside
    assert g[5.0]["truth"] == "center" and g[5.0]["state"] == "STOP"
    assert any(t["gate"]["truth"] == "edge" and t["gate"]["state"] == "SLOW" for t in ticks)
    assert ep["metrics"]["resumes"] == 1  # only the start; the rod is still in at 6 s
    assert (out / "bags" / ep["bag"] / "judge.jsonl").is_file()
