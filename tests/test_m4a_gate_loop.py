"""M4a gated closed loop (eval_loop + GatedCommand + JudgeRunner) with a fake sim and policy."""

import numpy as np
import pytest

from gate.safety_gate import load_gate_config
from scripts.sim.eval_loop import Obs, run_closed_loop
from scripts.sim.gate_loop import GatedCommand, gate_metrics, tcp_speed
from scripts.sim.intruder import ROOT, load_m4a
from scripts.sim.sim_judge import JudgeRunner, OracleJudge
from tests.test_eval_loop import JOINTS, NAMES, FakeClient, FakeSim

M4A = load_m4a()
GCFG = load_gate_config(ROOT / "configs" / "gate_sim.yaml")
RATE = 30


class TrackingSim(FakeSim):
    """FakeSim whose joints follow the last command exactly (no G9 follow error)."""

    def __init__(self):
        super().__init__()
        self.q = {n: 0.1 * k for k, n in enumerate(NAMES)}
        self.q["gripper"] = -0.16

    def latest_obs(self):
        o = super().latest_obs()
        return Obs(o.image, o.t_image, dict(self.q))

    def send(self, names, q):
        super().send(names, q)
        self.q.update(zip(names, map(float, q), strict=True))


def make_gate(sim, client, zone_rel, judge=None):
    holder = {}

    def zone_at(t_abs):
        return zone_rel(t_abs - holder["g"].t0)

    s = {**M4A["judge"], "latency_s": 0.0, **(judge or {})}
    runner = JudgeRunner(OracleJudge(zone_at, M4A["zones"]), s, M4A["zones"], truth=zone_at)
    g = GatedCommand(GCFG, runner, client, NAMES, truth=zone_at)
    holder["g"] = g
    return g


def run(zone_rel, seconds=3.0, judge=None):
    sim = TrackingSim()
    client = FakeClient(sim)
    g = make_gate(sim, client, zone_rel, judge)
    out = run_closed_loop(sim, client, JOINTS, rate_hz=RATE, duration_s=seconds, gate=g)
    return sim, client, g, out["ticks"]


def test_without_a_gate_the_records_are_unchanged():
    sim = FakeSim()
    out = run_closed_loop(sim, FakeClient(sim), JOINTS, rate_hz=RATE, duration_s=0.2)
    assert all("gate" not in t for t in out["ticks"])
    assert set(out["ticks"][0]) == {"i", "t", "late_s", "obs_age_s", "infer_ms", "inferred",
                                    "rtt_ms", "overrun", "action"}  # fmt: skip


def test_start_holds_in_stop_then_resumes_once_and_resets_the_policy():
    sim, client, g, ticks = run(lambda t: "none", seconds=1.5)
    states = [t["gate"]["state"] for t in ticks]
    hold = GCFG.resume_hold_s
    n_hold = int(round(hold * RATE))
    assert states[:n_hold] == ["STOP"] * n_hold  # G6: starts STOPped, waits resume_hold_s
    assert states[n_hold] == "CONTINUE"
    assert sim.log.count("reset") == 2  # episode start + on_resume (G7)
    assert len(g.resumes) == 1 and g.resumes[0]["queue_len"] == 0
    # the policy is not asked while STOPped, nor on the tick that leaves STOP
    called = [t["gate"]["policy_called"] for t in ticks]
    assert not any(called[: n_hold + 1]) and all(called[n_hold + 1 :])
    assert client.n == len(ticks) - n_hold - 1
    resumed = [k for k, t in enumerate(ticks) if t["gate"]["resumed"]]
    assert resumed == [n_hold]
    r = ticks[n_hold]["gate"]
    assert r["q_cmd"] == r["q_now"]  # no command from an old chunk on the resume tick
    assert ticks[0]["action"] == [] and ticks[-1]["action"]


def test_reset_comes_before_the_first_act_after_a_resume():
    sim, *_ = run(lambda t: "none", seconds=1.0)
    first_act = sim.log.index("act")
    assert [e for e in sim.log[:first_act] if e != "send"] == ["reset", "reset"]


def test_centre_stops_the_arm_and_it_resumes_after_the_hold():
    sim, client, g, ticks = run(lambda t: "center" if 1.0 <= t < 1.5 else "none", seconds=3.0)
    by_t = {round(t["t"], 4): t["gate"] for t in ticks}
    assert by_t[1.0]["state"] == "STOP" and by_t[1.0]["reason"] == "ok"
    stopped = [t for t in ticks if 1.0 <= t["t"] < 1.5]
    assert all(t["gate"]["state"] == "STOP" for t in stopped)
    assert all(not t["gate"]["policy_called"] for t in stopped[1:])
    assert all(t["gate"]["q_cmd"] == t["gate"]["q_now"] for t in stopped)
    resumes = [t["t"] for t in ticks if t["gate"]["resumed"]]
    assert len(resumes) == 2 and resumes[1] == pytest.approx(1.5 + GCFG.resume_hold_s, abs=1e-6)
    assert sim.log.count("reset") == 3


def test_edge_slows_and_flicker_never_lets_it_relax():
    flick = {"flicker": {"t": [1.0, 2.5], "period_s": 0.2, "alt": "none"}}
    _, _, _, ticks = run(lambda t: "edge" if 1.0 <= t < 2.5 else "none", 3.5, judge=flick)
    st = {round(t["t"], 4): t["gate"]["state"] for t in ticks}
    win = [s for t, s in st.items() if 1.0 <= t < 2.5]
    assert set(win) == {"SLOW"}  # requested flips SLOW/CONTINUE, but never for resume_hold_s
    req = {t["gate"]["requested"] for t in ticks if 1.0 <= t["t"] < 2.5}
    assert req == {"SLOW", "CONTINUE"}
    assert st[round(2.5 + GCFG.resume_hold_s, 4)] == "CONTINUE"


def test_stale_and_dropout_stop_through_g1():
    _, _, _, ticks = run(lambda t: "none", 3.0, judge={"dropouts": [[1.0, 2.0]]})
    reasons = {t["gate"]["reason"] for t in ticks if 1.3 <= t["t"] < 2.0}
    assert reasons == {"stale"}
    assert {t["gate"]["state"] for t in ticks if 1.3 <= t["t"] < 2.0} == {"STOP"}
    _, _, _, ticks = run(lambda t: "none", 3.0, judge={"latency_s": 0.15})
    assert "stale" in {t["gate"]["reason"] for t in ticks}


def test_low_confidence_is_one_level_safer():
    _, _, _, ticks = run(lambda t: "none", 2.0, judge={"confidence": 0.55})
    after = [t["gate"] for t in ticks if t["t"] >= 1.0]
    assert {g["reason"] for g in after} == {"low_conf"} and {g["state"] for g in after} == {"SLOW"}


def test_names_must_match():
    sim = FakeSim()
    g = make_gate(sim, FakeClient(sim), lambda t: "none")
    g.begin(0.0)
    with pytest.raises(ValueError):
        g.filter(sim.latest_obs(), 0.0, NAMES[::-1], None)


def test_tcp_speed_and_gate_metrics():
    t = np.array([0.0, 0.5, 1.0])
    assert np.allclose(tcp_speed(t, np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0, 0]])), [0, 0.2, 0])
    _, _, _, ticks = run(lambda t: "center" if 1.0 <= t < 1.5 else "none", seconds=3.0)
    m = gate_metrics(ticks, lambda q: np.asarray(q)[:, :3], react_s=0.1)
    assert m["resumes"] == 2 and m["intrusion"]["center_spans"][0][0] == pytest.approx(1.0)
    assert sum(m["state_s"].values()) == pytest.approx(3.0, abs=1e-6)
    assert m["intrusion"]["stop_after_center_s"] == pytest.approx(0.0)
    assert m["intrusion"]["tcp_speed_max_center_settled"] == pytest.approx(0.0)
    assert m["intrusion"]["resume_after_center_s"] == pytest.approx(
        GCFG.resume_hold_s + 1 / RATE, abs=1e-3)  # fmt: skip (tick times are rounded)
    assert m["policy_calls"] < len(ticks)


# ---------- M4a-S B: SLOW reads the chunk at time_scale ----------

STEP = 0.01  # each action moves every arm joint by this much


class SeqClient(FakeClient):
    """Action n = start + n * STEP on the arm joints (a ramp the arm can follow)."""

    def act(self, image, state):
        self.sim.log.append("act")
        a = np.array([0.0, 0.1, 0.2, 0.3, 0.4]) + self.n * STEP
        self.n += 1
        return {"action": a, "command": np.append(a, -0.16), "names": NAMES,
                "infer_ms": 0.5, "inferred": False}  # fmt: skip


def run_seq(zone_rel, seconds, time_scale, cfg=GCFG):
    sim = TrackingSim()
    client = SeqClient(sim)
    holder = {}

    def zone_at(t_abs):
        return zone_rel(t_abs - holder["g"].t0)

    s = {**M4A["judge"], "latency_s": 0.0}
    runner = JudgeRunner(OracleJudge(zone_at, M4A["zones"]), s, M4A["zones"], truth=zone_at)
    g = GatedCommand(cfg, runner, client, NAMES, truth=zone_at, time_scale=time_scale)
    holder["g"] = g
    ticks = run_closed_loop(sim, client, JOINTS, rate_hz=RATE, duration_s=seconds, gate=g)["ticks"]
    return client, g, ticks


def _edge(t):
    return "edge" if 1.0 <= t < 3.0 else "none"


def test_slow_reads_actions_at_the_time_scale_and_interpolates():
    client, g, ticks = run_seq(_edge, 4.0, 0.3)
    slow = [t for t in ticks if t["gate"]["plan"].startswith("slow")]
    calls = sum(t["gate"]["policy_called"] for t in slow)
    assert calls / len(slow) == pytest.approx(0.3, abs=0.05)
    tgt = np.array([t["gate"]["q_target"][0] for t in slow])
    # the target advances 0.3 action per tick: 0.3 * STEP (also over a read)
    assert np.allclose(np.diff(tgt), 0.3 * STEP, atol=1e-6)
    assert all(t["gate"]["phase"] is not None for t in slow)


def test_leaving_slow_has_no_catch_up_jump():
    _, _, ticks = run_seq(_edge, 4.0, 0.3)
    tgt = np.array([t["gate"]["q_target"][0] for t in ticks])
    plans = [t["gate"]["plan"] for t in ticks]
    k = plans.index("carry")
    assert all(p == "continue" for p in plans[k + 1 : k + 10])
    assert np.max(np.abs(np.diff(tgt[k - 3 : k + 10]))) <= STEP + 1e-9  # at most one action


def test_slow_keeps_the_chunk_from_running_ahead():
    """Over the same time, the plain gate uses up all actions; time scaling uses ~0.3 of them."""
    c1, _, _ = run_seq(_edge, 4.0, 1.0)
    c3, _, _ = run_seq(_edge, 4.0, 0.3)
    slow_ticks = (2.0 + GCFG.resume_hold_s) * RATE  # edge 1-3 s, then the G6 hold
    assert c1.n - c3.n == pytest.approx(0.7 * slow_ticks, abs=3)


def test_time_scale_one_is_the_plain_gate():
    _, _, a = run_seq(_edge, 4.0, 1.0)
    sim = TrackingSim()
    client = SeqClient(sim)
    g = make_gate(sim, client, _edge)
    b = run_closed_loop(sim, client, JOINTS, rate_hz=RATE, duration_s=4.0, gate=g)["ticks"]
    assert [t["gate"]["q_cmd"] for t in a] == [t["gate"]["q_cmd"] for t in b]


def test_stop_and_resume_drop_the_slow_state():
    def zone(t):
        return "edge" if 1.0 <= t < 2.0 else "center" if 2.0 <= t < 2.5 else "none"

    _, g, ticks = run_seq(zone, 4.0, 0.3)
    stop = [t for t in ticks if t["gate"]["state"] == "STOP" and t["t"] > 1.0]
    assert stop and all(t["gate"]["plan"] in ("stop", "slow_interp", "slow_fetch") for t in
                        stop[:1])  # fmt: skip
    after = [t["gate"] for t in ticks if t["t"] > 2.0 and t["gate"]["resumed"]]
    assert len(after) == 1
    k = next(i for i, t in enumerate(ticks) if t["t"] > 2.0 and t["gate"]["resumed"])
    nxt = ticks[k + 1]["gate"]
    assert nxt["plan"] == "continue" and nxt["policy_called"]  # no carry, no old action


def test_bad_time_scale():
    sim = TrackingSim()
    with pytest.raises(ValueError):
        GatedCommand(GCFG, None, FakeClient(sim), NAMES, time_scale=0.0)
