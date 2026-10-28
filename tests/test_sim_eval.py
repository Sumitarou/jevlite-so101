"""Closed-loop evaluator in headless Gazebo: scripts/sim/eval_policy.py (marker `sim`).

Uses the fake policy server (`--fake-policy hold`: always commands the current joint state), so
no checkpoint and no GPU inference are needed (Gazebo's camera still renders on the GPU). The arm
never pushes, so the episode must be judged a failure; every process must be gone afterwards.
"""

import json

import pytest
from test_sim_demos import ROS_SETUP, WS_SETUP, ros

pytestmark = pytest.mark.sim


def test_hold_policy_episode_fails_and_cleans_up(tmp_path):
    if not ROS_SETUP.is_file() or not WS_SETUP.is_file():
        pytest.skip("ROS 2 Jazzy or the built sim_ws is missing")
    out = tmp_path / "eval"
    p = ros(f"timeout --signal=INT --kill-after=30 600 /usr/bin/python3 scripts/sim/eval_policy.py "
            f"--fake-policy hold --seeds 100 --duration 3 --out {out}", timeout=700)  # fmt: skip
    assert (out / "summary.json").is_file(), p.stdout[-2000:] + p.stderr[-2000:]
    s = json.loads((out / "summary.json").read_text())
    assert p.returncode == 0, s.get("error")
    assert s["server"]["policy"] == "fake:hold"
    assert s["episodes"] == 1 and s["successes"] == 0
    assert s["loop"]["ticks"] == 90  # 3 s at 30 Hz
    ep = json.loads((out / "results.jsonl").read_text().splitlines()[0])
    assert ep["seed"] == 100 and ep["result"]["success"] is False
    # the block stayed where it was put (the arm did not move it)
    x, y, _ = ep["block_start"]
    assert abs(ep["final_block_pose"][0] - x) < 0.003 and abs(ep["final_block_pose"][1] - y) < 0.003
    assert (out / "bags" / ep["bag"] / "ticks.jsonl").is_file()
    assert s["leftover_processes"] == []
    assert not s["server_stop"]["leftover"] and not s["shutdown"]["leftover"]


def test_robustness_runner_moves_the_block_and_cleans_up(tmp_path):
    """scripts/sim/robustness.py with the hold policy: both suites in one Gazebo, a policy server
    restart between n_action_steps settings, and the block moved by set_pose at t = 1.5 s."""
    if not ROS_SETUP.is_file() or not WS_SETUP.is_file():
        pytest.skip("ROS 2 Jazzy or the built sim_ws is missing")
    p = ros(f"timeout --signal=INT --kill-after=30 900 /usr/bin/python3 scripts/sim/robustness.py "
            f"--fake-policy hold --duration 3 --only center pre_x+2 --n-action-steps 100 10 "
            f"--no-video --out-base {tmp_path}", timeout=1000)  # fmt: skip
    outs = {d.name.split("_m3e_")[1]: d for d in tmp_path.iterdir()}
    assert set(outs) == {"start", "perturb"}, p.stdout[-2000:] + p.stderr[-2000:]
    s = {k: json.loads((d / "summary.json").read_text()) for k, d in outs.items()}
    assert p.returncode == 0, s["start"].get("error")
    assert s["start"]["episodes"] == 1 and s["perturb"]["episodes"] == 2
    assert s["perturb"]["server_starts"] == 2  # n 100 (start + perturb), then n 10
    assert s["perturb"]["leftover_processes"] == []
    assert all(not x["leftover"] for x in s["perturb"]["server_stops"])
    assert not s["perturb"]["shutdown"]["leftover"]
    eps = [json.loads(x) for x in (outs["perturb"] / "results.jsonl").read_text().splitlines()]
    assert [e["condition"] for e in eps] == ["pre_x+2_n100", "pre_x+2_n10"]
    for e in eps:
        pm = e["metrics"]["perturb"]
        assert pm["perturb_ok"] and pm["t_event"] == pytest.approx(1.5, abs=0.04)
        assert e["final_block_pose"][0] == pytest.approx(e["block_start"][0] + 0.02, abs=0.003)
        assert e["metrics"]["push_ticks"] == 0 and e["result"]["success"] is False
        assert not e["metrics"]["touched"]  # the set_pose jump is not the arm
    c = json.loads((outs["start"] / "results.jsonl").read_text().splitlines()[0])
    assert c["condition"] == "center" and not c["metrics"]["touched"]
    assert c["block_start"] == pytest.approx([0.0, -0.04, 0.0])
