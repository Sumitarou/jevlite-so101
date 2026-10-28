"""M4b: the image judge in the closed loop without ROS (RemoteJudge, judge server handler,
m4b scenarios, judge metrics)."""

import threading

import numpy as np
import pytest

from gate.safety_gate import load_gate_config
from jevlite.api import Choice, Decision
from scripts.sim.intruder import build_scenarios, load_m4a
from scripts.sim.judge_server import make_handler
from scripts.sim.kinematics import ROOT, SIM_YAML
from scripts.sim.m4b_suite import judge_spec, load_m4b, suite_scenarios
from scripts.sim.policy_link import PolicyClient, PolicyError, default_address, new_key, serve
from scripts.sim.sim_judge import JudgeRunner, RemoteJudge, choose, judge_log_metrics

M4A, M4B = load_m4a(), load_m4b()
SIM = __import__("yaml").safe_load(SIM_YAML.read_text())
IMG = np.zeros((4, 6, 3), np.uint8)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def request(self, msg):
        self.sent.append(msg)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _rep(stop=0.7, lat=4.0):
    rest = (1 - stop) / 2
    return {"ok": True, "probs": {"CONTINUE": rest, "SLOW": rest, "STOP": stop},
            "latency_ms": lat, "source": "v1:x"}  # fmt: skip


def test_remote_judge_keeps_the_image_time_and_the_judge_latency():
    c = FakeClient([_rep(0.7, 4.0)])
    d = RemoteJudge(c).judge(IMG, t_capture=12.3)
    assert d.choice is Choice.STOP and d.t_mono == 12.3 and d.latency_ms == 4.0
    assert d.source == "v1:x" and c.sent[0]["cmd"] == "judge"
    with pytest.raises(ValueError):
        RemoteJudge(c).judge(IMG)


@pytest.mark.parametrize("bad", [PolicyError("boom"), EOFError(),
                                 {"ok": True, "probs": {"CONTINUE": 1.0}, "latency_ms": 1.0},
                                 {**_rep(), "probs": {"CONTINUE": float("nan"), "SLOW": 0.5,
                                                      "STOP": 0.5}}])  # fmt: skip
def test_remote_judge_failure_is_stop(bad):
    j = RemoteJudge(FakeClient([bad]))
    d = j.judge(IMG, t_capture=1.0)
    assert d.choice is Choice.STOP and d.probs["STOP"] == 1.0 and d.source == "judge_error"
    assert len(j.errors) == 1 and len(j.rtt_ms) == 1


def test_choose_breaks_ties_toward_the_safe_side():
    assert choose({"CONTINUE": 0.4, "SLOW": 0.4, "STOP": 0.2}) is Choice.SLOW


def test_runner_releases_after_the_judges_own_latency():
    """latency_s null: 40 ms of judge time = 40 ms of sim time (1:1)."""
    j = RemoteJudge(FakeClient([_rep(0.9, 40.0)] * 10))
    r = JudgeRunner(j, {"mode": "serial", "latency_s": None}, M4A["zones"])
    r.begin(100.0)
    assert r.step(IMG, 100.0, 100.0) is None
    assert r.step(IMG, 100.033, 100.033) is None
    d = r.step(IMG, 100.04, 100.0401)
    assert d is not None and d.t_mono == 100.0 and d.latency_ms == pytest.approx(40.0)
    assert r.log[0]["latency_ms"] == pytest.approx(40.0) and r.log[0]["source"] == "v1:x"


def test_handler_answers_judge_and_info():
    class J:
        source = "fake"

        def judge(self, image, t_capture=None):
            return Decision(Choice.SLOW, {"CONTINUE": 0.1, "SLOW": 0.8, "STOP": 0.1}, 0.0,
                            2.5, "fake")  # fmt: skip

    h = make_handler(J(), {"kind": "v1"})
    rep = h({"cmd": "judge", "image": IMG})
    assert rep == {"probs": {"CONTINUE": 0.1, "SLOW": 0.8, "STOP": 0.1}, "latency_ms": 2.5,
                   "source": "fake"}  # fmt: skip
    assert h({"cmd": "info"}) == {"kind": "v1"}
    with pytest.raises(ValueError):
        h({"cmd": "act"})


def test_remote_judge_over_a_real_socket():
    class J:
        def judge(self, image, t_capture=None):
            stop = 0.9 if image.mean() > 0 else 0.1
            rest = (1 - stop) / 2
            return Decision(Choice.STOP, {"CONTINUE": rest, "SLOW": rest, "STOP": stop}, 0.0,
                            1.5, "sock")  # fmt: skip

    addr, key = default_address("jevlite_judge_test"), new_key()
    th = threading.Thread(target=serve, args=(addr, key, make_handler(J(), {})), daemon=True)
    th.start()
    c = PolicyClient.connect(addr, key, 10.0)
    try:
        rj = RemoteJudge(c)
        assert rj.judge(np.full((480, 640, 3), 9, np.uint8), 1.0).choice is Choice.STOP
        assert rj.judge(np.zeros((480, 640, 3), np.uint8), 2.0).choice is not Choice.STOP
        assert rj.judge(np.zeros((480, 640, 3), np.uint8), 2.0).latency_ms == 1.5
    finally:
        c.close()
    th.join(5)
    assert not th.is_alive()


def test_suite_scenarios_and_overrides():
    scs = build_scenarios(M4A, SIM)
    img = suite_scenarios(scs, M4B, image_judge=True)
    assert [s.id for s in img] == M4B["scenarios"]
    for s in img:
        assert s.judge["flicker"] is None and s.judge["latency_s"] is None
        assert s.judge["confidence"] is None and s.judge["mode"] == "serial"
    orc = {s.id: s for s in suite_scenarios(scs, M4B, image_judge=False)}
    assert orc["flicker"].judge["flicker"] is None
    assert orc["center_1"].judge["latency_s"] == M4A["judge"]["latency_s"]
    assert orc["center_1"].judge["confidence"] == M4A["judge"]["confidence"]
    assert orc["flicker"].keys == next(s for s in scs if s.id == "flicker").keys
    with pytest.raises(ValueError, match="unknown scenarios"):
        suite_scenarios(scs, {**M4B, "scenarios": ["nope"]}, True)


def test_judge_spec():
    assert judge_spec(M4B, "v2", "models/judge/x")["model"] == "models/judge/x"
    with pytest.raises(ValueError, match="unknown judge"):
        judge_spec(M4B, "v9")
    with pytest.raises(ValueError, match="no model"):
        judge_spec({"judges": {"v1": {"kind": "v1", "model": None}}}, "v1")


def test_judge_log_metrics():
    log = [{"t_image": 0.0, "t_start": 0.0, "t_release": 0.004, "choice": "CONTINUE",
            "truth": "none"},
           {"t_image": 0.1, "t_start": 0.1, "t_release": 0.105, "choice": "CONTINUE",
            "truth": "center"},
           {"t_image": 0.2, "t_start": 0.2, "t_release": 0.203, "choice": "STOP",
            "truth": "edge"},
           {"t_image": 0.3, "t_start": 0.3, "t_release": 0.303, "choice": "STOP",
            "truth": "center"}]  # fmt: skip
    m = judge_log_metrics(log, M4A["zones"], [5.0, 6.0, 7.0, 8.0], errors=1)
    assert m["n"] == 4 and m["accuracy"] == 0.5
    assert m["center_to_continue"] == 1 and m["under"] == 1 and m["over"] == 1
    assert m["flips"] == 1 and m["errors"] == 1
    assert m["delay_ms"]["max"] == pytest.approx(5.0)
    assert m["rtt_ms"]["max"] == 8.0


def test_gate_config_still_loads():
    assert load_gate_config(ROOT / M4A["gate_config"]).min_conf == 0.6


def test_wrong_answers_split_by_transition():
    from scripts.sim.m4b_report import wrong_answers

    log = [{"t_image": t, "truth": z, "choice": c, "conf": 0.9}
           for t, z, c in [(1.0, "none", "CONTINUE"), (1.033, "edge", "CONTINUE"),
                           (1.067, "edge", "SLOW"), (2.0, "edge", "STOP"),
                           (3.0, "center", "STOP")]]  # fmt: skip
    w = wrong_answers(log, M4A["zones"])
    assert w["wrong"] == 2 and w["under"] == 1
    assert w["at_transition"] == 1 and w["under_at_transition"] == 1  # 1.033 s: rod just came in
    assert [e["t_image"] for e in w["list"]] == [1.033, 2.0]
