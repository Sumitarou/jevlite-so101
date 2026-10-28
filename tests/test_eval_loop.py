"""scripts/sim/eval_loop.py: closed-loop run on sim time, with a fake simulator and policy."""

import numpy as np
import pytest

from scripts.sim.eval_loop import (
    Obs,
    pick_examples,
    run_closed_loop,
    summarize_episodes,
    summarize_ticks,
)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
NAMES = JOINTS + ["gripper"]


class FakeSim:
    """Sim clock that advances only while waiting (and by `cost` per act, to model latency)."""

    def __init__(self, image_period=1 / 30):
        self.t, self.sent, self.log = 10.0, [], []
        self.image_period = image_period

    def sim_now(self):
        return self.t

    def wait_until(self, t):
        self.t = max(self.t, t)
        return True

    def latest_obs(self):
        t_img = np.floor(self.t / self.image_period) * self.image_period
        q = {n: 0.1 * k for k, n in enumerate(NAMES)}
        return Obs(np.zeros((4, 6, 3), np.uint8), t_img, q)

    def send(self, names, q):
        self.sent.append((list(names), np.asarray(q)))
        self.log.append("send")


class FakeClient:
    def __init__(self, sim, cost_s=0.0, queue_after_reset=0):
        self.sim, self.cost, self.q_after = sim, cost_s, queue_after_reset
        self.n = 0

    def reset(self):
        self.sim.log.append("reset")
        return {"queue_len": self.q_after}

    def act(self, image, state):
        self.sim.log.append("act")
        self.sim.t += self.cost(self.n) if callable(self.cost) else self.cost
        inferred = self.n % 10 == 0
        self.n += 1
        return {"action": state, "command": np.append(state, -0.16), "names": NAMES,
                "infer_ms": 20.0 if inferred else 0.5, "inferred": inferred}  # fmt: skip


def test_reset_comes_before_the_first_observation_and_ticks_are_counted():
    sim = FakeSim()
    out = run_closed_loop(sim, FakeClient(sim), JOINTS, rate_hz=30, duration_s=1.0)
    assert sim.log[0] == "reset" and sim.log[1] == "act"
    assert sim.log.count("reset") == 1
    s = out["summary"]
    assert s["ticks"] == 30 and s["overruns"] == 0 and s["inferences"] == 3
    names, q = sim.sent[0]
    assert names == NAMES and q[-1] == pytest.approx(-0.16)
    np.testing.assert_allclose(q[:5], [0.0, 0.1, 0.2, 0.3, 0.4], atol=1e-6)
    assert s["infer_ms_p50"] == pytest.approx(20.0)


def test_state_is_taken_in_the_policy_joint_order():
    sim = FakeSim()
    order = list(reversed(JOINTS))
    run_closed_loop(sim, FakeClient(sim), order, rate_hz=30, duration_s=0.1)
    np.testing.assert_allclose(sim.sent[0][1][:5], [0.4, 0.3, 0.2, 0.1, 0.0], atol=1e-6)


def test_reset_that_leaves_queued_actions_is_an_error():
    sim = FakeSim()
    with pytest.raises(RuntimeError, match="queued"):
        run_closed_loop(sim, FakeClient(sim, queue_after_reset=3), JOINTS, 30, 1.0)
    assert "act" not in sim.log


def test_slow_inference_counts_overruns_and_obs_age():
    sim = FakeSim()
    # every 10th tick the network runs and takes 50 ms of sim time (> one 33 ms period)
    client = FakeClient(sim, cost_s=lambda n: 0.05 if n % 10 == 0 else 0.0)
    out = run_closed_loop(sim, client, JOINTS, rate_hz=30, duration_s=1.0)
    s = out["summary"]
    assert s["overruns"] == 3
    assert s["obs_age_ms_max"] >= 50.0
    late = [t for t in out["ticks"] if t["late_s"] > 0]
    assert late and late[0]["i"] == 1  # the tick after a slow one starts late


def test_summarize_ticks_empty():
    assert summarize_ticks([])["ticks"] == 0


def _ep(seed, ok, err):
    return {"seed": seed, "status": "done", "result": {"success": ok, "error_m": err},
            "_ticks": [{"infer_ms": 10.0, "inferred": True, "rtt_ms": 12.0, "obs_age_s": 0.02,
                        "late_s": 0.0, "overrun": False}],
            "rtf_loop": 0.38, "wall_duration_s": 40.0}  # fmt: skip


def test_pick_examples_and_summary():
    eps = [_ep(100, True, 0.02), _ep(101, False, 0.09), _ep(102, True, 0.01), _ep(103, False, 0.2)]
    ex = pick_examples(eps)
    assert ex["success"]["seed"] == 102 and ex["failure"]["seed"] == 101
    s = summarize_episodes(eps)
    assert s["success_rate"] == 0.5 and s["error_m_max"] == 0.2
    assert s["error_m_mean_success"] == pytest.approx(0.015)
    all_ok = [_ep(100, True, 0.02), _ep(101, True, 0.03)]
    assert set(pick_examples(all_ok)) == {"success", "worst"}
    assert summarize_episodes([])["episodes"] == 0


def test_on_tick_runs_before_the_observation_and_its_event_is_kept():
    sim = FakeSim()
    calls = []

    def on_tick(i, t):
        calls.append((i, round(t, 4)))
        sim.log.append(f"tick{i}")
        return {"moved": True} if i == 3 else None

    out = run_closed_loop(sim, FakeClient(sim), JOINTS, 30, 0.2, on_tick=on_tick,
                          extra=lambda obs: {"state": [obs.joints[j] for j in JOINTS]})  # fmt: skip
    assert calls == [(i, round(i / 30, 4)) for i in range(6)]
    assert sim.log[1:5] == ["tick0", "act", "send", "tick1"]
    ticks = out["ticks"]
    assert [("event" in k) for k in ticks] == [False, False, False, True, False, False]
    assert ticks[3]["event"] == {"moved": True}
    assert ticks[0]["state"] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_without_hooks_the_tick_record_is_unchanged():
    sim = FakeSim()
    tick = run_closed_loop(sim, FakeClient(sim), JOINTS, 30, 0.1)["ticks"][0]
    assert set(tick) == {"i", "t", "late_s", "obs_age_s", "infer_ms", "inferred", "rtt_ms",
                         "overrun", "action"}  # fmt: skip
