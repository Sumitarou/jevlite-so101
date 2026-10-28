"""M4a judges on sim time: oracle + JudgeRunner (latency, serial/pipelined, dropouts, flicker)."""

import math

import numpy as np
import pytest

from jevlite.api import Choice, Decision, Judge
from scripts.sim.intruder import load_m4a
from scripts.sim.sim_judge import JudgeRunner, OracleJudge, probs_for

ZONES = load_m4a()["zones"]
IMG = np.zeros((4, 4, 3), np.uint8)
DT = 1 / 32  # exact binary tick for exact time comparisons
T0 = 100.0


def zone_at(t):
    return "center" if 101.0 <= t < 102.0 else "none"


def runner(**s):
    base = {"mode": "serial", "latency_s": 0.0, "confidence": 0.9, "dropouts": [], "flicker": None}
    r = JudgeRunner(OracleJudge(zone_at, ZONES), {**base, **s}, ZONES, truth=zone_at)
    r.begin(T0)
    return r


def run(r, seconds, age=0.0):
    """Tick for `seconds`; the image of each tick is `age` old. Returns (now, decision) pairs."""
    out = []
    for i in range(int(seconds / DT)):
        now = T0 + i * DT
        out.append((now, r.step(IMG, now - age, now)))
    return out


def test_probs_for_sums_to_one_and_puts_confidence_on_the_choice():
    p = probs_for(Choice.SLOW, 0.7)
    assert math.isclose(sum(p.values()), 1.0) and p["SLOW"] == 0.7 and p["STOP"] == p["CONTINUE"]
    with pytest.raises(ValueError):
        probs_for(Choice.SLOW, 1.5)


def test_oracle_is_a_judge_and_uses_the_capture_time():
    j = OracleJudge(zone_at, ZONES, confidence=0.8)
    assert isinstance(j, Judge)
    d = j.judge(IMG, t_capture=101.5)
    assert d.choice is Choice.STOP and d.t_mono == 101.5 and d.source == "oracle"
    assert d.confidence == 0.8
    with pytest.raises(ValueError):
        j.judge(IMG)


def test_zero_latency_answers_in_the_same_step_with_the_image_time():
    r = runner()
    d = r.step(IMG, T0 - 0.02, T0)
    assert d.t_mono == T0 - 0.02 and d.latency_ms == 0.0 and d.choice is Choice.CONTINUE


def test_serial_latency_and_age():
    lat = 3 * DT
    r = runner(latency_s=lat)
    res = run(r, 1.0)
    assert all(d is None for now, d in res if now < T0 + lat)  # nothing before the first result
    first = next(now for now, d in res if d is not None)
    assert first == T0 + lat
    ages = [now - d.t_mono for now, d in res if d is not None]
    # the next job starts on the tick a result comes out: the age peaks one tick before that
    assert max(ages) == pytest.approx(2 * lat - DT)
    # one at a time: a new result every `lat` (results released at multiples of lat)
    rel = [e["t_release"] for e in r.log]
    assert np.allclose(np.diff(rel), lat, atol=1e-4)  # the log is rounded to 0.1 ms
    assert all(e["t_release"] - e["t_start"] == pytest.approx(lat, abs=1e-4) for e in r.log)
    assert res[-1][1].latency_ms == pytest.approx(lat * 1e3)


def test_serial_age_reaches_twice_the_latency_so_g1_can_fire():
    lat = 0.15
    r = runner(latency_s=lat)
    res = run(r, 2.0)
    ages = [now - d.t_mono for now, d in res if d is not None]
    assert max(ages) > 0.2 > lat  # stale_s 0.2 is exceeded at times with 150 ms
    assert max(ages) <= 2 * lat + 2 * DT


def test_pipelined_releases_every_tick_with_a_fixed_age():
    lat = 3 * DT
    r = runner(mode="pipelined", latency_s=lat)
    res = run(r, 1.0)
    ages = [now - d.t_mono for now, d in res if d is not None]
    assert np.allclose(ages, lat)
    assert len(r.log) == len(ages)


def test_the_truth_reaches_the_gate_after_the_latency():
    lat = 4 * DT
    r = runner(latency_s=lat, mode="pipelined")
    res = dict(run(r, 3.0))
    assert res[T0 + 1.0 + lat - DT].choice is Choice.CONTINUE
    assert res[T0 + 1.0 + lat].choice is Choice.STOP
    assert all(e["truth"] == ("center" if e["choice"] == "STOP" else "none") for e in r.log)


def test_dropout_stops_new_results_and_the_last_one_ages():
    r = runner(dropouts=[[0.5, 1.5]])
    res = run(r, 2.0)
    during = [(now, d) for now, d in res if T0 + 0.5 <= now < T0 + 1.5]
    assert len({d.t_mono for _, d in during}) == 1  # frozen on the last result before 0.5 s
    assert during[-1][0] - during[-1][1].t_mono > 0.9
    after = [d for now, d in res if now >= T0 + 1.5]
    assert after[0].t_mono == T0 + 1.5


def test_flicker_alternates_with_the_period():
    r = runner(flicker={"t": [0.0, 1.0], "period_s": 0.25, "alt": "edge"})
    res = run(r, 1.0)
    got = [(round(now - T0, 4), d.choice) for now, d in res]
    for t, c in got:
        k = int(t // 0.25)
        assert c is (Choice.SLOW if k % 2 == 1 else Choice.CONTINUE), t
    assert any(e["flicker"] for e in r.log)
    assert all(e["truth"] == "none" for e in r.log)  # the truth itself never changed


def test_confidence_override_and_measured_latency():
    class Slow:
        def judge(self, image, t_capture=None):
            return Decision(Choice.SLOW, probs_for(Choice.SLOW, 0.99), t_capture, 125.0, "model")

    r = JudgeRunner(Slow(), {"mode": "serial", "latency_s": None, "confidence": 0.55}, ZONES)
    r.begin(T0)
    assert r.step(IMG, T0, T0) is None
    d = r.step(IMG, T0 + 0.125, T0 + 0.125)
    assert d.latency_ms == 125.0 and d.confidence == 0.55 and d.source == "model"
    assert d.t_mono == T0


def test_bad_settings_and_missing_begin():
    with pytest.raises(ValueError):
        JudgeRunner(OracleJudge(zone_at, ZONES), {"mode": "batch"}, ZONES)
    with pytest.raises(ValueError):
        JudgeRunner(OracleJudge(zone_at, ZONES), {"mode": "serial", "latency_s": -1}, ZONES)
    r = JudgeRunner(OracleJudge(zone_at, ZONES), {"mode": "serial", "latency_s": 0}, ZONES)
    with pytest.raises(RuntimeError):
        r.step(IMG, 0, 0)
