"""Safety gate spec G1-G9 (docs/spec.md). Each spec change is tested here before implementing.

Interface assumed by these tests (to be implemented in gate/safety_gate.py):

    cfg = GateConfig(stale_s, min_conf, slow_alpha, resume_hold_s, future_tol_s,
                     max_speed, dt_max, max_follow_err, slow_speed_scale)
    cfg = load_gate_config(path=None)       # None -> configs/gate.yaml; missing key -> error
    gate = SafetyGate(cfg, on_resume=None)
    out = gate.step(decision: Decision | None, now: float, q_now, q_target) -> GateOutput
    out.q_cmd: np.ndarray (new array)   out.state: Choice (state actually applied)
    out.requested: Choice (after G1/G2/checks)   out.reason: Reason (why requested is what it is)
    out.rate_limited / out.clock_backwards / out.follow_error: bool (for logs)

Times use exact binary fractions (0.25, 0.5, ...) so `==` boundaries are tested exactly,
which is why CFG uses stale_s=0.25 / future_tol_s=0.125 instead of the defaults 0.2 / 0.001.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gate.safety_gate import GateConfig, Reason, SafetyGate, load_gate_config
from jevlite.api import Choice, Decision

# max_speed / max_follow_err are effectively unlimited here so G3-G5 tests see the pre-limit
# values and a non-moving Q_NOW is not a follow error; G8 / G9 tests use G8_CFG / G9_CFG.
CFG = GateConfig(
    stale_s=0.25,
    min_conf=0.6,
    slow_alpha=0.3,
    resume_hold_s=0.5,
    future_tol_s=0.125,
    max_speed=(1e12,) * 6,
    dt_max=1.0,
    max_follow_err=(1e12,) * 6,
    slow_speed_scale=1.0,
)
HOLD = CFG.resume_hold_s
EPS = 1e-9

Q_NOW = np.array([0.0, 10.0, -5.0, 20.0, 0.0, 30.0])
Q_TARGET = np.array([10.0, 0.0, 5.0, 20.0, -10.0, 40.0])
Q_SLOW = Q_NOW + CFG.slow_alpha * (Q_TARGET - Q_NOW)

C, S, X = Choice.CONTINUE, Choice.SLOW, Choice.STOP


def dec(choice, t: float, conf: float = 0.9) -> Decision:
    """Decision whose max prob is `conf` (on `choice`); the rest is split evenly."""
    rest = (1.0 - conf) / 2
    probs = {c.name: (conf if c is choice else rest) for c in Choice}
    return Decision(choice=choice, probs=probs, t_mono=t, latency_ms=1.0, source="test")


class ResumeCounter:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


def step(gate: SafetyGate, d: Decision | None, now: float):
    return gate.step(d, now, Q_NOW.copy(), Q_TARGET.copy())


def fresh(gate: SafetyGate, choice, now: float, conf: float = 0.9):
    """Step with a decision made at `now` (age 0)."""
    return step(gate, dec(choice, now, conf), now)


def running_gate(cfg: GateConfig = CFG, on_resume=None) -> tuple[SafetyGate, float]:
    """A gate that has already resumed into CONTINUE. Returns (gate, current time)."""
    gate = SafetyGate(cfg, on_resume=on_resume)
    fresh(gate, C, 0.0)
    out = fresh(gate, C, cfg.resume_hold_s)
    assert out.state is C
    return gate, cfg.resume_hold_s


def slow_gate(on_resume=None) -> tuple[SafetyGate, float]:
    """A gate that is running and has just been tightened to SLOW (no pending hold)."""
    gate, t = running_gate(on_resume=on_resume)
    t += 0.25
    assert fresh(gate, S, t).state is S
    return gate, t


# ---------------------------------------------------------------------------------------
# Config: every number comes from configs/gate.yaml
# ---------------------------------------------------------------------------------------


def test_cfg_repo_yaml_has_spec_defaults():
    cfg = load_gate_config()
    assert cfg.stale_s == pytest.approx(0.2)
    assert cfg.min_conf == pytest.approx(0.6)
    assert cfg.slow_alpha == pytest.approx(0.3)
    assert cfg.resume_hold_s == pytest.approx(0.5)
    assert cfg.future_tol_s == pytest.approx(0.001)
    assert len(cfg.max_speed) == 6  # SO-101: 5 joints + gripper (values are provisional)
    assert all(v > 0 for v in cfg.max_speed)
    assert cfg.dt_max == pytest.approx(0.1)
    assert cfg.max_follow_err == pytest.approx((15.0, 15.0, 15.0, 15.0, 15.0, 30.0))
    assert cfg.slow_speed_scale == pytest.approx(0.4)


FULL_YAML = (
    "stale_s: 0.1\nmin_conf: 0.7\nslow_alpha: 0.5\nresume_hold_s: 1.0\nfuture_tol_s: 0.002\n"
    "max_speed: [1.0, 2.0]\ndt_max: 0.05\nmax_follow_err: [3.0, 4.0]\nslow_speed_scale: 0.4\n"
)


def test_cfg_loads_values_from_given_yaml(tmp_path: Path):
    p = tmp_path / "gate.yaml"
    p.write_text(FULL_YAML)
    assert load_gate_config(p) == GateConfig(
        0.1, 0.7, 0.5, 1.0, 0.002, (1.0, 2.0), 0.05, (3.0, 4.0), 0.4
    )


@pytest.mark.parametrize(
    "missing",
    [
        "stale_s",
        "min_conf",
        "slow_alpha",
        "resume_hold_s",
        "future_tol_s",
        "max_speed",
        "dt_max",
        "max_follow_err",
        "slow_speed_scale",
    ],
)
def test_cfg_missing_key_is_error_not_silent_default(tmp_path: Path, missing):
    p = tmp_path / "gate.yaml"
    p.write_text("".join(ln + "\n" for ln in FULL_YAML.splitlines() if not ln.startswith(missing)))
    with pytest.raises((KeyError, ValueError)):
        load_gate_config(p)


@pytest.mark.parametrize(
    ("key", "listed"),
    [("max_speed", "[1.0, 2.0]"), ("max_follow_err", "[3.0, 4.0]")],
)
def test_cfg_per_joint_values_must_be_lists(tmp_path: Path, key, listed):
    p = tmp_path / "gate.yaml"
    p.write_text(FULL_YAML.replace(f"{key}: {listed}", f"{key}: 1.0"))
    with pytest.raises(ValueError):
        load_gate_config(p)


def test_cfg_per_joint_lists_must_have_the_same_length():
    with pytest.raises(ValueError):
        replace(CFG, max_follow_err=(1.0,) * 5)


@pytest.mark.parametrize(
    "kw",
    [
        {"stale_s": 0.0},
        {"stale_s": -0.1},
        {"min_conf": -0.1},
        {"min_conf": 1.1},
        {"slow_alpha": -0.1},
        {"slow_alpha": 1.1},
        {"resume_hold_s": -0.1},
        {"future_tol_s": -0.001},
        {"stale_s": float("nan")},
        {"resume_hold_s": float("inf")},
        {"future_tol_s": float("nan")},
        {"max_speed": (), "max_follow_err": ()},
        {"max_speed": (1.0,) * 5 + (0.0,)},
        {"max_speed": (1.0,) * 5 + (-1.0,)},
        {"max_speed": (1.0,) * 5 + (float("nan"),)},
        {"max_speed": (1.0,) * 5 + (float("inf"),)},
        {"max_follow_err": (1.0,) * 5 + (0.0,)},
        {"max_follow_err": (1.0,) * 5 + (-1.0,)},
        {"max_follow_err": (1.0,) * 5 + (float("nan"),)},
        {"max_follow_err": (1.0,) * 5 + (float("inf"),)},
        {"dt_max": 0.0},
        {"dt_max": -0.1},
        {"dt_max": float("nan")},
        {"dt_max": float("inf")},
        {"slow_speed_scale": 0.0},
        {"slow_speed_scale": -0.1},
        {"slow_speed_scale": 1.1},
        {"slow_speed_scale": float("nan")},
    ],
)
def test_cfg_invalid_values_rejected(kw):
    with pytest.raises(ValueError):
        replace(CFG, **kw)


# ---------------------------------------------------------------------------------------
# G1 stale: None or now - t_mono > stale_s -> STOP
# ---------------------------------------------------------------------------------------


def test_g1_none_decision_is_stop():
    gate, t = running_gate()
    out = step(gate, None, t + 0.25)
    assert out.state is X
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


def test_g1_none_on_first_step_is_stop():
    out = step(SafetyGate(CFG), None, 0.0)
    assert out.state is X


def test_g1_stale_confident_continue_is_stop():
    gate, t = running_gate()
    out = step(gate, dec(C, t, conf=1.0), t + 0.5)
    assert out.state is X
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


def test_g1_boundary_age_exactly_stale_s_is_fresh():
    gate, t = running_gate()
    out = step(gate, dec(C, t), t + CFG.stale_s)  # now - t_mono == stale_s -> not stale
    assert out.state is C


def test_g1_age_just_over_stale_s_is_stop():
    gate, t = running_gate()
    out = step(gate, dec(C, t), t + CFG.stale_s + EPS)
    assert out.state is X


def test_g1_stale_slow_is_stop():
    gate, t = running_gate()
    out = step(gate, dec(S, t), t + 1.0)
    assert out.state is X


def test_g1_stale_threshold_comes_from_config():
    cfg = replace(CFG, stale_s=1.0)
    gate, t = running_gate(cfg)
    assert step(gate, dec(C, t), t + 0.75).state is C  # stale under CFG, fresh under cfg


# ---------------------------------------------------------------------------------------
# G1b future timestamp: t_mono - now > future_tol_s -> STOP (clock mismatch, untrusted)
# ---------------------------------------------------------------------------------------


def test_g1b_future_t_mono_is_stop_with_reason():
    gate, t = running_gate()
    now = t + 0.25
    out = step(gate, dec(C, now + 1.0, conf=1.0), now)
    assert out.state is X
    assert out.requested is X
    assert out.reason is Reason.FUTURE
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


def test_g1b_boundary_exactly_future_tol_s_is_accepted():
    gate, t = running_gate()
    now = t + 0.25
    out = step(gate, dec(C, now + CFG.future_tol_s), now)  # t_mono - now == tol -> ok
    assert out.state is C
    assert out.reason is Reason.OK


def test_g1b_just_over_future_tol_s_is_stop():
    gate, t = running_gate()
    now = t + 0.25
    out = step(gate, dec(C, now + CFG.future_tol_s + EPS), now)
    assert out.state is X
    assert out.reason is Reason.FUTURE


def test_g1b_slightly_future_within_tol_is_accepted():
    gate, t = running_gate()
    now = t + 0.25
    assert step(gate, dec(C, now + CFG.future_tol_s / 2), now).state is C


def test_g1b_future_tolerance_comes_from_config():
    cfg = replace(CFG, future_tol_s=0.5)
    gate, t = running_gate(cfg)
    now = t + 0.25
    assert step(gate, dec(C, now + 0.25), now).state is C  # future under CFG, ok under cfg


def test_g1b_future_decision_interrupts_hold_and_does_not_resume():
    cb = ResumeCounter()
    gate = SafetyGate(CFG, on_resume=cb)
    fresh(gate, C, 0.0)
    assert step(gate, dec(C, 1.25), 0.25).state is X  # 1 s in the future: breaks the run
    assert fresh(gate, C, 0.5).state is X
    assert fresh(gate, C, 0.75).state is X
    assert fresh(gate, C, 1.0).state is C
    assert cb.calls == 1


def test_g1b_future_while_slow_is_immediate_stop():
    gate, t = slow_gate()
    out = step(gate, dec(S, t + 1.0), t + 0.25)
    assert out.state is X
    assert out.reason is Reason.FUTURE


def test_g1b_default_tolerance_is_one_millisecond():
    """With the repo config: 1 ms ahead is accepted, 2 ms ahead is STOP."""
    cfg = load_gate_config()
    gate = SafetyGate(replace(cfg, resume_hold_s=0.0))
    assert step(gate, dec(C, 10.0 + 0.0009), 10.0).state is C
    assert step(gate, dec(C, 10.0 + 0.002), 10.0).reason is Reason.FUTURE


# ---------------------------------------------------------------------------------------
# Reason: why `requested` is what it is (for logs)
# ---------------------------------------------------------------------------------------


def _with(choice=C, t=None, probs=None):
    base = dec(choice, 0.0)
    return Decision(
        choice=choice,
        probs=base.probs if probs is None else probs,
        t_mono=t,
        latency_ms=1.0,
        source="test",
    )


@pytest.mark.parametrize(
    ("make", "requested", "reason"),
    [
        (lambda now: dec(C, now), C, "OK"),
        (lambda now: dec(S, now), S, "OK"),
        (lambda now: dec(X, now), X, "OK"),
        (lambda now: None, X, "NO_DECISION"),
        (lambda now: dec(C, now - 1.0), X, "STALE"),
        (lambda now: dec(C, now + 1.0), X, "FUTURE"),
        (lambda now: _with(t=float("nan")), X, "NON_FINITE_TIME"),
        (lambda now: _with("FAST", t=now), X, "UNKNOWN_CHOICE"),
        (lambda now: _with(t=now, probs={"CONTINUE": float("nan")}), X, "NON_FINITE_PROBS"),
        (lambda now: dec(C, now, conf=0.5), S, "LOW_CONF"),
        (lambda now: dec(S, now, conf=0.5), X, "LOW_CONF"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_reason_explains_requested(make, requested, reason):
    gate, t = running_gate()
    now = t + 0.25
    out = step(gate, make(now), now)
    assert out.requested is requested
    assert out.reason is Reason[reason]


def test_reason_non_finite_target():
    gate, t = running_gate()
    q_target = Q_TARGET.copy()
    q_target[1] = float("nan")
    out = gate.step(dec(C, t + 0.25), t + 0.25, Q_NOW.copy(), q_target)
    assert out.requested is X
    assert out.reason is Reason.NON_FINITE_TARGET


def test_reason_is_ok_while_holding():
    """During a hold, requested/reason describe the decision; state shows the hold."""
    gate = SafetyGate(CFG)
    out = fresh(gate, C, 0.0)
    assert (out.state, out.requested, out.reason) == (X, C, Reason.OK)


def test_reason_is_a_readable_string():
    assert str(Reason.FUTURE) == "future"
    assert {str(r) for r in Reason} >= {"ok", "stale", "future", "low_conf"}


# ---------------------------------------------------------------------------------------
# G2 low confidence: max(probs) < min_conf -> one level safer
# ---------------------------------------------------------------------------------------


def test_g2_low_conf_continue_becomes_slow():
    gate, t = running_gate()
    out = fresh(gate, C, t + 0.25, conf=0.5)
    assert out.state is S


def test_g2_low_conf_slow_becomes_stop():
    gate, t = running_gate()
    out = fresh(gate, S, t + 0.25, conf=0.5)
    assert out.state is X


def test_g2_low_conf_stop_stays_stop():
    gate, t = running_gate()
    out = fresh(gate, X, t + 0.25, conf=0.4)
    assert out.state is X


def test_g2_boundary_conf_exactly_min_conf_is_not_lowered():
    gate, t = running_gate()
    out = fresh(gate, C, t + 0.25, conf=CFG.min_conf)  # max(probs) == min_conf -> keep
    assert out.state is C


def test_g2_boundary_slow_conf_exactly_min_conf_stays_slow():
    gate, t = running_gate()
    out = fresh(gate, S, t + 0.25, conf=CFG.min_conf)
    assert out.state is S


def test_g2_conf_just_below_min_conf_is_lowered():
    gate, t = running_gate()
    out = fresh(gate, C, t + 0.25, conf=CFG.min_conf - EPS)
    assert out.state is S


def test_g2_empty_probs_is_low_confidence():
    gate, t = running_gate()
    d = Decision(choice=C, probs={}, t_mono=t + 0.25, latency_ms=1.0, source="test")
    assert step(gate, d, t + 0.25).state is S


def test_g2_low_conf_continue_counts_as_slow_for_hold():
    """Low-confidence CONTINUE is a SLOW request: it can relax STOP->SLOW, never to CONTINUE."""
    gate = SafetyGate(CFG)
    assert fresh(gate, C, 0.0, conf=0.5).state is X
    assert fresh(gate, C, 0.25, conf=0.5).state is X
    assert fresh(gate, C, HOLD, conf=0.5).state is S
    assert fresh(gate, C, 1.0, conf=0.5).state is S
    assert fresh(gate, C, 2.0, conf=0.5).state is S


# ---------------------------------------------------------------------------------------
# G3 STOP: q_cmd = q_now
# ---------------------------------------------------------------------------------------


def test_g3_stop_holds_current_position():
    gate, t = running_gate()
    out = fresh(gate, X, t + 0.25)
    assert out.state is X
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


def test_g3_q_cmd_is_a_new_array():
    gate = SafetyGate(CFG)
    q_now = Q_NOW.copy()
    out = gate.step(dec(X, 0.0), 0.0, q_now, Q_TARGET.copy())
    q_now[:] = 999.0
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


# ---------------------------------------------------------------------------------------
# G4 SLOW: q_cmd = q_now + slow_alpha * (q_target - q_now)
# ---------------------------------------------------------------------------------------


def test_g4_slow_interpolates_toward_target():
    gate, t = running_gate()
    out = fresh(gate, S, t + 0.25)
    assert out.state is S
    np.testing.assert_allclose(out.q_cmd, [3.0, 7.0, -2.0, 20.0, -3.0, 33.0])


def test_g4_slow_alpha_comes_from_config():
    cfg = replace(CFG, slow_alpha=0.5)
    gate, t = running_gate(cfg)
    out = fresh(gate, S, t + 0.25)
    np.testing.assert_allclose(out.q_cmd, (Q_NOW + Q_TARGET) / 2)


# ---------------------------------------------------------------------------------------
# G5 CONTINUE: q_cmd = q_target
# ---------------------------------------------------------------------------------------


def test_g5_continue_passes_target_through():
    gate, t = running_gate()
    out = fresh(gate, C, t + 0.25)
    assert out.state is C
    np.testing.assert_array_equal(out.q_cmd, Q_TARGET)


# ---------------------------------------------------------------------------------------
# G6 hysteresis: relaxing transitions (STOP->SLOW, STOP->CONTINUE, SLOW->CONTINUE) need the
#    destination to be requested for resume_hold_s consecutively; tightening is immediate.
# ---------------------------------------------------------------------------------------


def test_g6_gate_starts_in_stop():
    out = fresh(SafetyGate(CFG), C, 0.0)
    assert out.state is X
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


def test_g6_stop_to_continue_boundary_exactly_resume_hold_s_relaxes():
    gate = SafetyGate(CFG)
    assert fresh(gate, X, 0.0).state is X
    assert fresh(gate, C, 1.0).state is X  # hold starts here
    assert fresh(gate, C, 1.25).state is X
    out = fresh(gate, C, 1.0 + HOLD)  # held exactly resume_hold_s -> relax
    assert out.state is C
    np.testing.assert_array_equal(out.q_cmd, Q_TARGET)


def test_g6_stop_to_continue_just_before_resume_hold_s_stays_stopped():
    gate = SafetyGate(CFG)
    fresh(gate, X, 0.0)
    fresh(gate, C, 1.0)
    out = fresh(gate, C, 1.0 + HOLD - EPS)
    assert out.state is X
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


def test_g6_stop_to_slow_boundary_exactly_resume_hold_s_relaxes():
    gate = SafetyGate(CFG)
    fresh(gate, X, 0.0)
    out = fresh(gate, S, 1.0)  # hold starts here
    assert out.state is X
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)
    assert fresh(gate, S, 1.25).state is X
    out = fresh(gate, S, 1.0 + HOLD)
    assert out.state is S
    np.testing.assert_allclose(out.q_cmd, Q_SLOW)


def test_g6_stop_to_slow_just_before_resume_hold_s_stays_stopped():
    gate = SafetyGate(CFG)
    fresh(gate, X, 0.0)
    fresh(gate, S, 1.0)
    assert fresh(gate, S, 1.0 + HOLD - EPS).state is X


def test_g6_slow_to_continue_boundary_and_stays_slow_meanwhile():
    gate, t = slow_gate()
    out = fresh(gate, C, t + 0.25)  # hold starts
    assert out.state is S
    np.testing.assert_allclose(out.q_cmd, Q_SLOW)
    assert fresh(gate, C, t + 0.25 + HOLD - EPS).state is S
    out = fresh(gate, C, t + 0.25 + HOLD)
    assert out.state is C
    np.testing.assert_array_equal(out.q_cmd, Q_TARGET)


def test_g6_stop_via_slow_to_continue_needs_two_holds():
    gate = SafetyGate(CFG)
    fresh(gate, X, 0.0)
    fresh(gate, S, 1.0)
    assert fresh(gate, S, 1.5).state is S  # first hold done
    fresh(gate, C, 1.75)  # second hold starts at the first CONTINUE
    assert fresh(gate, C, 2.0).state is S
    assert fresh(gate, C, 1.75 + HOLD).state is C


def test_g6_mixed_continue_and_slow_requests_do_not_relax_stop():
    """The destination itself must persist; alternating CONTINUE/SLOW keeps STOP."""
    gate = SafetyGate(CFG)
    fresh(gate, X, 0.0)
    for k, choice in enumerate([C, S, C, S, C, S, C, S]):
        assert fresh(gate, choice, 1.0 + 0.25 * k).state is X


def test_g6_interrupted_continue_resets_hold():
    gate = SafetyGate(CFG)
    fresh(gate, X, 0.0)
    fresh(gate, C, 1.0)
    fresh(gate, X, 1.25)  # interruption
    fresh(gate, C, 1.5)  # hold restarts here
    assert fresh(gate, C, 1.75).state is X
    assert fresh(gate, C, 2.0 - EPS).state is X
    assert fresh(gate, C, 2.0).state is C


def test_g6_slow_request_interrupts_continue_hold():
    gate = SafetyGate(CFG)
    fresh(gate, X, 0.0)
    fresh(gate, C, 1.0)
    fresh(gate, S, 1.25)  # different destination -> C hold reset
    fresh(gate, C, 1.5)
    assert fresh(gate, C, 1.75).state is X
    assert fresh(gate, C, 2.0).state is C


def test_g6_same_level_request_interrupts_hold():
    """In SLOW, a SLOW request between CONTINUEs breaks the CONTINUE run."""
    gate, t = slow_gate()
    fresh(gate, C, t + 0.25)  # run 1 starts
    fresh(gate, S, t + 0.5)  # same level as the state -> run 1 broken
    fresh(gate, C, t + 0.625)  # run 2 starts
    assert fresh(gate, C, t + 0.25 + HOLD).state is S  # run 1 would have completed here
    assert fresh(gate, C, t + 0.625 + HOLD).state is C


def test_g6_stale_decision_interrupts_hold():
    gate = SafetyGate(CFG)
    fresh(gate, X, 0.0)
    fresh(gate, C, 1.0)
    assert step(gate, None, 1.25).state is X
    fresh(gate, C, 1.5)
    assert fresh(gate, C, 1.75).state is X
    assert fresh(gate, C, 2.0).state is C


def test_g6_hold_uses_config_value():
    cfg = replace(CFG, resume_hold_s=1.0)
    gate = SafetyGate(cfg)
    fresh(gate, C, 0.0)
    assert fresh(gate, C, 0.5).state is X  # would relax under CFG (0.5)
    assert fresh(gate, C, 1.0).state is C


@pytest.mark.parametrize(
    ("before", "after"),
    [(C, S), (C, X), (S, X)],
    ids=["CONTINUE->SLOW", "CONTINUE->STOP", "SLOW->STOP"],
)
def test_g6_tightening_is_immediate(before, after):
    gate, t = running_gate() if before is C else slow_gate()
    out = fresh(gate, after, t + 0.25)
    assert out.state is after


def test_g6_tightening_during_hold_is_immediate():
    gate, t = slow_gate()
    fresh(gate, C, t + 0.25)  # holding toward CONTINUE
    assert fresh(gate, X, t + 0.5).state is X


def test_g6_continue_stays_continue_without_new_hold():
    gate, t = running_gate()
    for k in range(1, 5):
        assert fresh(gate, C, t + 0.25 * k).state is C


# ---------------------------------------------------------------------------------------
# G7 resume hook: on_resume called once at the moment the gate leaves STOP
#    (to SLOW or to CONTINUE)
# ---------------------------------------------------------------------------------------


def test_g7_on_resume_called_once_at_stop_to_continue():
    cb = ResumeCounter()
    gate = SafetyGate(CFG, on_resume=cb)
    fresh(gate, X, 0.0)
    fresh(gate, C, 1.0)
    fresh(gate, C, 1.25)
    assert cb.calls == 0  # not during the hold
    assert fresh(gate, C, 1.5).state is C
    assert cb.calls == 1  # exactly at the relaxing step
    fresh(gate, C, 1.75)
    fresh(gate, C, 2.0)
    assert cb.calls == 1  # not repeated while running


def test_g7_on_resume_called_once_at_stop_to_slow():
    cb = ResumeCounter()
    gate = SafetyGate(CFG, on_resume=cb)
    fresh(gate, X, 0.0)
    fresh(gate, S, 1.0)
    fresh(gate, S, 1.25)
    assert cb.calls == 0
    assert fresh(gate, S, 1.5).state is S
    assert cb.calls == 1
    fresh(gate, S, 1.75)
    assert cb.calls == 1


def test_g7_slow_to_continue_does_not_call_again():
    cb = ResumeCounter()
    gate = SafetyGate(CFG, on_resume=cb)
    fresh(gate, X, 0.0)
    fresh(gate, S, 0.25)
    fresh(gate, S, 0.75)  # left STOP -> 1 call
    fresh(gate, C, 1.0)
    assert fresh(gate, C, 1.5).state is C  # SLOW -> CONTINUE, not from STOP
    assert cb.calls == 1


def test_g7_continue_to_slow_and_back_does_not_call():
    cb = ResumeCounter()
    gate, t = running_gate(on_resume=cb)
    assert cb.calls == 1  # first start counts (gate starts in STOP)
    fresh(gate, S, t + 0.25)
    fresh(gate, C, t + 0.5)
    assert fresh(gate, C, t + 0.5 + HOLD).state is C
    assert cb.calls == 1


def test_g7_called_again_after_a_new_stop():
    cb = ResumeCounter()
    gate, t = running_gate(on_resume=cb)
    assert cb.calls == 1
    fresh(gate, X, t + 0.25)
    fresh(gate, C, t + 0.5)
    fresh(gate, C, t + 1.0)
    assert cb.calls == 2


def test_g7_not_called_while_staying_stopped():
    cb = ResumeCounter()
    gate = SafetyGate(CFG, on_resume=cb)
    fresh(gate, X, 0.0)
    step(gate, None, 0.25)  # G1
    fresh(gate, S, 0.5, conf=0.5)  # G2: SLOW -> STOP
    fresh(gate, C, 0.75)  # hold started, still STOP
    step(gate, dec(C, 0.75), 1.25)  # G1 stale -> STOP, hold reset
    assert cb.calls == 0


def test_g7_on_resume_is_optional():
    gate, t = running_gate(on_resume=None)
    fresh(gate, X, t + 0.25)
    fresh(gate, C, t + 0.5)
    assert fresh(gate, C, t + 1.0).state is C


# ---------------------------------------------------------------------------------------
# G8 rate limiter: relative to the previous output q_cmd_prev (reset to q_now on the first call
#    and while STOPped). Per-joint step limit L_j = max_speed[j] * dt, dt = now - previous now
#    capped at dt_max (first call / time going backwards -> dt = 0). The whole step vector is
#    scaled by one factor s = min(1, min_j L_j / |d_j|), so the direction is kept.
# ---------------------------------------------------------------------------------------

DT = 0.25  # every G8 step below is DT after the previous one unless noted
V = 2.0  # -> L = V * DT = 0.5 (exact in binary, so == boundaries are exact)
L = V * DT
G8_CFG = replace(CFG, max_speed=(V,) * 6, dt_max=1.0)
DIR = np.sign(Q_TARGET - Q_NOW)  # [1, -1, 1, 0, -1, 1]


def resuming_gate(choice=C, cfg: GateConfig = G8_CFG, on_resume=None):
    """A gate holding toward `choice` with q_cmd_prev == Q_NOW; the next fresh `choice`
    step at t + DT relaxes (hold met), so it is the first step that moves. Returns (gate, t)."""
    gate = SafetyGate(cfg, on_resume=on_resume)
    assert fresh(gate, choice, 0.0).state is X  # first call: q_cmd_prev = q_now
    assert fresh(gate, choice, HOLD - DT).state is X
    return gate, HOLD - DT


def _step_to(gate, choice, now, q_target, q_now=Q_NOW):
    return gate.step(dec(choice, now), now, q_now.copy(), q_target)


def _scaled(q_ref: np.ndarray, q_unlimited: np.ndarray, limits: np.ndarray) -> np.ndarray:
    d = q_unlimited - q_ref
    nz = d != 0
    s = min(1.0, float(np.min(limits[nz] / np.abs(d[nz])))) if nz.any() else 1.0
    return q_ref + s * d


def test_g8_resume_step_is_scaled_as_a_whole():
    gate, t = resuming_gate()
    out = fresh(gate, C, t + DT)
    assert out.state is C
    # d = [10, -10, 10, 0, -10, 10] from q_cmd_prev = Q_NOW -> s = 0.5 / 10 = 0.05
    np.testing.assert_allclose(out.q_cmd, [0.5, 9.5, -4.5, 20.0, -0.5, 30.5])


def test_g8_limits_command_change_not_distance_to_q_now():
    """(a): with the arm lagging (q_now fixed), the command still advances L per step."""
    gate, t = resuming_gate()
    for k in range(1, 4):
        out = fresh(gate, C, t + k * DT)  # q_now stays Q_NOW every step
        np.testing.assert_allclose(out.q_cmd, Q_NOW + k * L * DIR)
        assert out.rate_limited is True
    assert np.abs(out.q_cmd - Q_NOW).max() == pytest.approx(3 * L)  # > L: not tied to q_now


def test_g8_direction_is_preserved_not_clipped_per_joint():
    gate, t = resuming_gate()
    q_target = Q_NOW + np.array([4.0, 1.0, 0.0, 0.0, -2.0, 0.0])
    out = _step_to(gate, C, t + DT, q_target)
    # s = 0.5 / 4 = 0.125 -> every joint gets 1/8 of its step (per-joint clip would give 0.5)
    np.testing.assert_array_equal(out.q_cmd - Q_NOW, [0.5, 0.125, 0.0, 0.0, -0.25, 0.0])


def test_g8_each_joint_has_its_own_limit():
    cfg = replace(G8_CFG, max_speed=(V, V, V, V, V, 4 * V))  # gripper may move 4x faster
    gate, t = resuming_gate(cfg=cfg)
    q_target = Q_NOW + np.array([1.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    out = _step_to(gate, C, t + DT, q_target)
    # limits [0.5 ... 2.0]: joint 0 needs s = 0.5, gripper allows s = 1.0 -> s = 0.5
    np.testing.assert_array_equal(out.q_cmd - Q_NOW, [0.5, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_g8_slow_is_scaled_after_interpolation():
    gate, t = resuming_gate(S)
    out = fresh(gate, S, t + DT)
    assert out.state is S
    np.testing.assert_allclose(out.q_cmd, _scaled(Q_NOW, Q_SLOW, np.full(6, L)))  # 3.0 > L


# G8 in SLOW: the limit is slow_speed_scale * max_speed[j] * dt (the state after this step's
# G6 transition decides; tightening into SLOW is immediate, relaxing out of it uses the full limit)
SLOW_CFG = replace(G8_CFG, slow_speed_scale=0.5, slow_alpha=0.5)
LS = 0.5 * L  # 0.25


def test_g8_slow_uses_the_scaled_limit():
    gate, t = resuming_gate(S, cfg=SLOW_CFG)
    out = fresh(gate, S, t + DT)
    q_slow = Q_NOW + 0.5 * (Q_TARGET - Q_NOW)
    np.testing.assert_allclose(out.q_cmd, _scaled(Q_NOW, q_slow, np.full(6, LS)))
    assert out.rate_limited is True
    assert np.max(np.abs(out.q_cmd - Q_NOW)) == pytest.approx(LS)


@pytest.mark.parametrize(("extra", "limited"), [(0.0, False), (1e-6, True)])
def test_g8_slow_boundary_exactly_the_scaled_limit_is_not_scaled(extra, limited):
    gate, t = resuming_gate(S, cfg=SLOW_CFG)
    q_target = Q_NOW + 2 * (LS + extra) * DIR  # slow_alpha 0.5 -> step LS (+ extra)
    out = _step_to(gate, S, t + DT, q_target)
    assert out.rate_limited is limited
    assert np.max(np.abs(out.q_cmd - Q_NOW)) == pytest.approx(LS)


def test_g8_continue_limit_is_unchanged_by_slow_speed_scale():
    gate, t = resuming_gate(C, cfg=SLOW_CFG)
    out = fresh(gate, C, t + DT)
    np.testing.assert_allclose(out.q_cmd, [0.5, 9.5, -4.5, 20.0, -0.5, 30.5])


def test_g8_tightening_into_slow_uses_the_slow_limit_at_once():
    gate, t = resuming_gate(C, cfg=SLOW_CFG)
    prev = fresh(gate, C, t + DT).q_cmd
    out = fresh(gate, S, t + 2 * DT)
    assert out.state is S
    assert np.max(np.abs(out.q_cmd - prev)) <= LS + 1e-12
    assert out.rate_limited is True


def test_g8_relaxing_out_of_slow_uses_the_full_limit_on_that_step():
    gate, t = resuming_gate(S, cfg=SLOW_CFG)
    far = Q_NOW + 100.0 * DIR
    prev = _step_to(gate, S, t + DT, far).q_cmd
    now = t + DT
    steps = []
    while True:
        now += DT
        out = _step_to(gate, C, now, far)
        steps.append((out.state, np.max(np.abs(out.q_cmd - prev))))
        prev = out.q_cmd
        if out.state is C:
            break
    for state, step_ in steps[:-1]:  # still SLOW while holding toward CONTINUE
        assert state is S and step_ == pytest.approx(LS)
    assert steps[-1][1] == pytest.approx(L)


def test_g8_stop_is_not_limited_and_resets_reference():
    gate, t = resuming_gate()
    fresh(gate, C, t + DT)
    fresh(gate, C, t + 2 * DT)  # command is now 2L away from Q_NOW
    out = fresh(gate, X, t + 3 * DT)
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)  # STOP jumps back to q_now at once
    assert out.rate_limited is False


def test_g8_resume_starts_from_current_position_after_stop():
    """q_cmd_prev is reset to q_now while STOPped, so a moved arm resumes from where it is."""
    gate, t = resuming_gate()
    fresh(gate, C, t + DT)
    fresh(gate, X, t + 2 * DT)
    q_moved = Q_NOW + np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # pushed while stopped
    fresh_t = t + 2 * DT
    for k in (1, 2):  # hold toward CONTINUE (q_now = q_moved)
        out = _step_to(gate, C, fresh_t + k * DT, Q_TARGET, q_now=q_moved)
        assert out.state is X
        np.testing.assert_array_equal(out.q_cmd, q_moved)
    out = _step_to(gate, C, fresh_t + 3 * DT, Q_TARGET, q_now=q_moved)
    assert out.state is C
    np.testing.assert_allclose(out.q_cmd, _scaled(q_moved, Q_TARGET, np.full(6, L)))


def test_g8_small_steps_pass_through_unchanged():
    gate, t = resuming_gate()
    q_target = Q_NOW + np.array([0.25, -0.25, 0.0, 0.125, -0.5, 0.5])
    out = _step_to(gate, C, t + DT, q_target)
    np.testing.assert_array_equal(out.q_cmd, q_target)
    assert out.rate_limited is False


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["+", "-"])
def test_g8_boundary_step_exactly_limit_is_not_scaled(sign):
    gate, t = resuming_gate()
    q_target = Q_NOW + sign * L  # |d_j| == L_j on every joint
    out = _step_to(gate, C, t + DT, q_target)
    np.testing.assert_array_equal(out.q_cmd, q_target)
    assert out.rate_limited is False


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["+", "-"])
def test_g8_step_just_over_limit_is_scaled(sign):
    gate, t = resuming_gate()
    out = _step_to(gate, C, t + DT, Q_NOW + sign * (L + 1e-6))
    assert np.all(np.abs(out.q_cmd - Q_NOW) <= L)
    np.testing.assert_allclose(out.q_cmd, Q_NOW + sign * L, rtol=0, atol=1e-12)
    assert out.rate_limited is True


def test_g8_limit_scales_with_dt_between_calls():
    gate, t = resuming_gate()
    fresh(gate, C, t + DT)  # q_cmd_prev = Q_NOW + L * DIR
    out = fresh(gate, C, t + DT + DT / 2)  # half the time -> half the step
    np.testing.assert_allclose(out.q_cmd, Q_NOW + 1.5 * L * DIR)


def test_g8_dt_is_capped_by_dt_max():
    gate, t = resuming_gate(cfg=replace(G8_CFG, dt_max=DT))
    fresh(gate, C, t + DT)
    out = fresh(gate, C, t + DT + 0.2 + DT)  # 0.45 s later, but dt is capped at 0.25
    np.testing.assert_allclose(out.q_cmd, Q_NOW + 2 * L * DIR)


def test_g8_first_call_does_not_move():
    gate = SafetyGate(replace(G8_CFG, resume_hold_s=0.0))
    out = fresh(gate, C, 5.0)  # CONTINUE at once (no hold), but dt = 0 on the first call
    assert out.state is C
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)
    assert out.rate_limited is True


def test_g8_same_now_twice_holds_the_previous_command():
    gate, t = resuming_gate()
    first = fresh(gate, C, t + DT)
    out = fresh(gate, C, t + DT)  # dt = 0
    np.testing.assert_array_equal(out.q_cmd, first.q_cmd)


def test_g8_time_going_backwards_holds_the_previous_command():
    gate, t = resuming_gate()
    first = fresh(gate, C, t + DT)
    out = step(gate, dec(C, t), t)  # now went back by DT -> dt = 0
    np.testing.assert_array_equal(out.q_cmd, first.q_cmd)
    out = fresh(gate, C, t + DT)  # dt measured from the backward call
    np.testing.assert_allclose(out.q_cmd, Q_NOW + 2 * L * DIR)


def test_g8_clock_backwards_is_flagged_for_logs():
    gate, t = resuming_gate()
    assert fresh(gate, C, t + DT).clock_backwards is False
    out = step(gate, dec(C, t), t)  # now went back by DT
    assert out.clock_backwards is True
    assert out.reason is Reason.OK  # reason still describes the decision
    assert fresh(gate, C, t + DT).clock_backwards is False  # forward again
    assert fresh(gate, C, t + DT).clock_backwards is False  # same now is not backwards


def test_g8_first_call_is_not_clock_backwards():
    assert fresh(SafetyGate(G8_CFG), X, 0.0).clock_backwards is False


def test_g8_limit_comes_from_config():
    gate, t = resuming_gate(cfg=replace(G8_CFG, max_speed=(4 * V,) * 6))
    out = fresh(gate, C, t + DT)
    np.testing.assert_allclose(out.q_cmd - Q_NOW, [2.0, -2.0, 2.0, 0.0, -2.0, 2.0])


def test_g8_does_not_change_state_requested_or_reason():
    gate, t = resuming_gate()
    out = fresh(gate, C, t + DT)
    assert (out.state, out.requested, out.reason) == (C, C, Reason.OK)


def test_g8_max_speed_length_must_match_joints():
    gate = SafetyGate(replace(G8_CFG, max_speed=(V,) * 5, max_follow_err=(1e12,) * 5))
    with pytest.raises(ValueError):
        fresh(gate, C, 0.0)


def test_g8_resume_jump_is_smoothed():
    """After a long STOP, the first CONTINUE steps approach the far target gradually."""
    cb = ResumeCounter()
    gate = SafetyGate(G8_CFG, on_resume=cb)
    q_now = Q_NOW.copy()
    q_target = Q_NOW + np.array([10.0, -5.0, 2.5, 0.0, 10.0, -10.0])  # drifted while stopped
    moves = []
    for k in range(40):
        now = DT * k
        out = gate.step(dec(C, now), now, q_now, q_target)
        moves.append(out.q_cmd - q_now)
        q_now = out.q_cmd  # ideal servo: reaches q_cmd every step
    assert cb.calls == 1
    moves = np.array(moves)
    assert np.all(np.abs(moves) <= L)
    np.testing.assert_allclose(q_now, q_target)  # 10 / 0.5 = 20 steps after resuming
    d = q_target - Q_NOW
    for m in moves[np.any(moves != 0, axis=1)]:  # every move points straight at the target
        np.testing.assert_allclose(m / np.linalg.norm(m), d / np.linalg.norm(d))


def test_g8_invariants_hold_for_random_inputs():
    rng = np.random.default_rng(0)
    speeds = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 8.0])
    cfg = replace(G8_CFG, max_speed=tuple(speeds), dt_max=0.1, resume_hold_s=0.0)
    gate = SafetyGate(cfg)
    now, prev_now, prev_cmd = 0.0, None, None
    for k in range(500):
        now += rng.uniform(0.0, 0.2)
        q_now = rng.uniform(-3, 3, 6)
        q_target = q_now + rng.normal(0, 2, 6)
        d = dec([C, S, X][rng.integers(3)], now, conf=rng.uniform(0.3, 1.0))
        out = gate.step(d, now, q_now, q_target)
        dt = 0.0 if prev_now is None else min(now - prev_now, cfg.dt_max)
        ref = q_now if prev_cmd is None else prev_cmd  # q_cmd_prev
        prev_now, prev_cmd = now, out.q_cmd
        if out.state is X:
            np.testing.assert_array_equal(out.q_cmd, q_now)
            continue
        move = out.q_cmd - ref
        assert np.all(np.abs(move) <= speeds * dt + 1e-12), k
        base = {C: q_target, S: q_now + cfg.slow_alpha * (q_target - q_now)}[out.state]
        unlimited = base - ref
        nz = unlimited != 0
        if nz.any():  # move = s * unlimited with one s in [0, 1]
            s = move[nz] / unlimited[nz]
            assert np.ptp(s) < 1e-9 and -1e-12 <= s[0] <= 1 + 1e-12, k
        else:
            assert not np.any(move), k


def test_g8_rate_limited_flag():
    gate, t = resuming_gate()
    assert fresh(gate, C, t + DT).rate_limited is True
    prev = fresh(gate, C, t + 2 * DT).q_cmd
    assert _step_to(gate, C, t + 3 * DT, prev + L / 2).rate_limited is False
    assert fresh(gate, X, t + 4 * DT).rate_limited is False


def test_g8_unchanged_command_is_not_rate_limited():
    """Holding the same command while running (target == q_cmd_prev) is not a limited step."""
    gate, t = resuming_gate()
    prev = fresh(gate, C, t + DT).q_cmd
    out = _step_to(gate, C, t + 2 * DT, prev)
    np.testing.assert_array_equal(out.q_cmd, prev)
    assert out.rate_limited is False


# ---------------------------------------------------------------------------------------
# G9 follow error: per joint |q_cmd_prev - q_now| > max_follow_err[j] -> STOP this cycle
#    (reason follow_error, flag follow_error); leaving STOP follows G6.
# ---------------------------------------------------------------------------------------

FE = (1.0,) * 5 + (2.0,)  # gripper tolerates more
G9_CFG = replace(CFG, max_follow_err=FE)


def g9_running(on_resume=None):
    """Running gate whose last output (q_cmd_prev) is Q_TARGET (CFG speed is unlimited)."""
    gate, t = running_gate(G9_CFG, on_resume=on_resume)
    return gate, t


def _with_q_now(gate, choice, now, q_now, decision_t=None):
    d = dec(choice, now if decision_t is None else decision_t)
    return gate.step(d, now, np.asarray(q_now, dtype=float), Q_TARGET.copy())


def test_g9_arm_not_following_is_stop():
    gate, t = g9_running()
    out = _with_q_now(gate, C, t + 0.25, Q_NOW)  # commanded Q_TARGET, arm still at Q_NOW
    assert out.state is X
    assert out.requested is X
    assert out.reason is Reason.FOLLOW_ERROR
    assert out.follow_error is True
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


def test_g9_following_arm_is_ok():
    gate, t = g9_running()
    out = _with_q_now(gate, C, t + 0.25, Q_TARGET)
    assert out.state is C
    assert out.follow_error is False
    assert out.reason is Reason.OK


@pytest.mark.parametrize("joint", [0, 5], ids=["joint0", "gripper"])
def test_g9_boundary_exactly_max_follow_err_is_ok(joint):
    gate, t = g9_running()
    q_now = Q_TARGET.copy()
    q_now[joint] -= FE[joint]  # |q_cmd_prev - q_now| == max_follow_err -> not an error
    out = _with_q_now(gate, C, t + 0.25, q_now)
    assert out.state is C
    assert out.follow_error is False


@pytest.mark.parametrize("joint", [0, 5], ids=["joint0", "gripper"])
def test_g9_just_over_max_follow_err_is_stop(joint):
    gate, t = g9_running()
    q_now = Q_TARGET.copy()
    q_now[joint] += FE[joint] + 1e-6
    out = _with_q_now(gate, C, t + 0.25, q_now)
    assert out.state is X
    assert out.reason is Reason.FOLLOW_ERROR


def test_g9_threshold_is_per_joint():
    gate, t = g9_running()
    q_now = Q_TARGET + np.array([0, 0, 0, 0, 0, 1.5])  # 1.5 > 1.0 but gripper allows 2.0
    assert _with_q_now(gate, C, t + 0.25, q_now).state is C


def test_g9_applies_in_slow_too():
    gate, t = g9_running()
    out = _with_q_now(gate, S, t + 0.25, Q_TARGET)  # follows fine -> SLOW
    assert out.state is S
    q_cmd_slow = out.q_cmd
    out = _with_q_now(gate, S, t + 0.5, q_cmd_slow + 5.0)  # arm 5 off the SLOW command
    assert out.state is X
    assert out.reason is Reason.FOLLOW_ERROR


def test_g9_resume_follows_g6_hysteresis():
    cb = ResumeCounter()
    gate, t = g9_running(on_resume=cb)
    assert cb.calls == 1
    _with_q_now(gate, C, t + 0.25, Q_NOW)  # follow error -> STOP, q_cmd_prev = Q_NOW
    assert _with_q_now(gate, C, t + 0.5, Q_NOW).state is X  # hold starts (arm at Q_NOW)
    assert _with_q_now(gate, C, t + 0.75, Q_NOW).state is X
    out = _with_q_now(gate, C, t + 1.0, Q_NOW)  # held resume_hold_s
    assert out.state is C
    assert out.follow_error is False
    assert cb.calls == 2


def test_g9_follow_error_interrupts_hold():
    gate, t = g9_running()
    _with_q_now(gate, C, t + 0.25, Q_NOW)  # STOP by follow error
    _with_q_now(gate, C, t + 0.5, Q_NOW)  # hold starts
    out = _with_q_now(gate, C, t + 0.75, Q_NOW + 5.0)  # arm moved 5 while stopped
    assert out.follow_error is True
    _with_q_now(gate, C, t + 1.0, Q_NOW + 5.0)  # hold restarts here
    assert _with_q_now(gate, C, t + 1.25, Q_NOW + 5.0).state is X
    assert _with_q_now(gate, C, t + 1.5, Q_NOW + 5.0).state is C


def test_g9_first_call_is_not_a_follow_error():
    out = _with_q_now(SafetyGate(G9_CFG), C, 0.0, Q_NOW + 100.0)
    assert out.follow_error is False


def test_g9_uses_the_rate_limited_command_as_reference():
    """q_cmd_prev is what was actually sent (after G8), not the policy target."""
    cfg = replace(G8_CFG, max_follow_err=(1.0,) * 6)
    gate, t = resuming_gate(cfg=cfg)
    out = fresh(gate, C, t + DT)  # sent Q_NOW + L * DIR (L = 0.5), target is 10 away
    out = _with_q_now(gate, C, t + 2 * DT, Q_NOW)  # arm at Q_NOW: error 0.5 <= 1.0
    assert out.follow_error is False
    assert out.state is C


def test_g9_threshold_comes_from_config():
    gate, t = running_gate(replace(CFG, max_follow_err=(20.0,) * 6))
    assert _with_q_now(gate, C, t + 0.25, Q_NOW).state is C  # error 10 <= 20


def test_g9_reason_priority_over_decision_reasons():
    """When the decision is also bad, reason names follow_error; the flag is always set."""
    gate, t = g9_running()
    out = _with_q_now(gate, C, t + 0.25, Q_NOW, decision_t=t - 1.0)  # stale AND not following
    assert out.reason is Reason.FOLLOW_ERROR
    assert out.follow_error is True


def test_g9_non_finite_target_reason_wins_but_flag_is_set():
    gate, t = g9_running()
    q_target = Q_TARGET.copy()
    q_target[0] = float("nan")
    out = gate.step(dec(C, t + 0.25), t + 0.25, Q_NOW.copy(), q_target)
    assert out.reason is Reason.NON_FINITE_TARGET
    assert out.follow_error is True


# ---------------------------------------------------------------------------------------
# Defensive input checks (beyond G1-G7): bad values -> STOP; inconsistent shapes -> error
# ---------------------------------------------------------------------------------------

BAD_FLOATS = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("bad", BAD_FLOATS, ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("key", ["CONTINUE", "STOP"])
def test_def_non_finite_prob_is_stop(bad, key):
    gate, t = running_gate()
    d = dec(C, t + 0.25, conf=0.9)
    d.probs[key] = bad
    out = step(gate, d, t + 0.25)
    assert out.state is X
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


@pytest.mark.parametrize("bad", BAD_FLOATS, ids=["nan", "inf", "-inf"])
def test_def_non_finite_t_mono_is_stop(bad):
    gate, t = running_gate()
    d = Decision(choice=C, probs=dec(C, 0).probs, t_mono=bad, latency_ms=1.0, source="test")
    assert step(gate, d, t + 0.25).state is X


@pytest.mark.parametrize("bad", BAD_FLOATS, ids=["nan", "inf", "-inf"])
def test_def_non_finite_q_target_is_stop_and_holds_q_now(bad):
    gate, t = running_gate()
    q_target = Q_TARGET.copy()
    q_target[2] = bad
    out = gate.step(dec(C, t + 0.25), t + 0.25, Q_NOW.copy(), q_target)
    assert out.state is X
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


@pytest.mark.parametrize("bad", BAD_FLOATS, ids=["nan", "inf", "-inf"])
def test_def_non_finite_q_now_raises(bad):
    """STOP means 'send q_now'; a non-finite q_now cannot be sent, so refuse loudly."""
    gate, t = running_gate()
    q_now = Q_NOW.copy()
    q_now[0] = bad
    with pytest.raises(ValueError):
        gate.step(dec(C, t + 0.25), t + 0.25, q_now, Q_TARGET.copy())


@pytest.mark.parametrize("bad_choice", ["FAST", "CONTINUE", 0, 3, None], ids=repr)
def test_def_unknown_choice_is_stop(bad_choice):
    """Only Choice members are accepted (a bare "CONTINUE" string or 0 is not a Choice)."""
    gate, t = running_gate()
    d = Decision(
        choice=bad_choice, probs=dec(C, 0).probs, t_mono=t + 0.25, latency_ms=1.0, source="t"
    )
    out = step(gate, d, t + 0.25)
    assert out.state is X
    np.testing.assert_array_equal(out.q_cmd, Q_NOW)


@pytest.mark.parametrize(
    "q_target",
    [np.zeros(5), np.zeros(7), np.zeros((6, 1)), np.zeros(0)],
    ids=["shorter", "longer", "2d", "empty"],
)
def test_def_shape_mismatch_raises(q_target):
    gate = SafetyGate(CFG)
    with pytest.raises(ValueError):
        gate.step(dec(C, 0.0), 0.0, Q_NOW.copy(), q_target)


def test_def_bad_input_interrupts_hold_and_does_not_resume():
    cb = ResumeCounter()
    gate = SafetyGate(CFG, on_resume=cb)
    fresh(gate, C, 0.0)
    d = dec(C, 0.25)
    d.probs["SLOW"] = float("nan")
    assert step(gate, d, 0.25).state is X  # breaks the CONTINUE run
    assert fresh(gate, C, 0.5).state is X
    assert fresh(gate, C, 0.75).state is X
    assert fresh(gate, C, 1.0).state is C
    assert cb.calls == 1


def test_def_bad_input_while_running_is_immediate_stop():
    gate, t = running_gate()
    q_target = Q_TARGET.copy()
    q_target[0] = float("nan")
    out = gate.step(dec(C, t + 0.25), t + 0.25, Q_NOW.copy(), q_target)
    assert out.state is X
    assert fresh(gate, C, t + 0.5).state is X  # must hold again before CONTINUE
