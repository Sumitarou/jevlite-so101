"""Safety gate: limit the policy's joint command using the latest judge Decision.

Spec G1-G9 is in docs/spec.md; tests/test_gate.py is the executable version.

    gate = SafetyGate(load_gate_config(), on_resume=policy.reset)
    out = gate.step(latest_decision, time.monotonic(), q_now, q_target)
    robot.send(out.q_cmd)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import yaml

from jevlite.api import Choice, Decision

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "gate.yaml"


@dataclass(frozen=True)
class GateConfig:
    stale_s: float
    min_conf: float
    slow_alpha: float
    resume_hold_s: float
    future_tol_s: float
    max_speed: tuple[float, ...]  # G8: per joint, robot joint units per second
    dt_max: float  # G8: cap on the time between calls
    max_follow_err: tuple[float, ...]  # G9: per joint, robot joint units
    slow_speed_scale: float  # G8: in SLOW the limit is this times max_speed (0 < x <= 1)

    def __post_init__(self):
        names = ("stale_s", "min_conf", "slow_alpha", "resume_hold_s", "future_tol_s", "dt_max",
                 "slow_speed_scale")  # fmt: skip
        for name in names:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}")
        if self.stale_s <= 0:
            raise ValueError(f"stale_s must be > 0, got {self.stale_s}")
        if not 0.0 <= self.min_conf <= 1.0:
            raise ValueError(f"min_conf must be in [0, 1], got {self.min_conf}")
        if not 0.0 <= self.slow_alpha <= 1.0:
            raise ValueError(f"slow_alpha must be in [0, 1], got {self.slow_alpha}")
        if self.resume_hold_s < 0:
            raise ValueError(f"resume_hold_s must be >= 0, got {self.resume_hold_s}")
        if self.future_tol_s < 0:
            raise ValueError(f"future_tol_s must be >= 0, got {self.future_tol_s}")
        if self.dt_max <= 0:
            raise ValueError(f"dt_max must be > 0, got {self.dt_max}")
        if not 0.0 < self.slow_speed_scale <= 1.0:
            raise ValueError(f"slow_speed_scale must be in (0, 1], got {self.slow_speed_scale}")
        for name in ("max_speed", "max_follow_err"):
            values = tuple(float(v) for v in getattr(self, name))
            if not values:
                raise ValueError(f"{name} must list one value per joint")
            if not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError(f"{name} must be finite and > 0, got {values}")
            object.__setattr__(self, name, values)
        if len(self.max_speed) != len(self.max_follow_err):
            raise ValueError("max_speed and max_follow_err must list the same joints")


def load_gate_config(path: Path | str | None = None) -> GateConfig:
    """Read configs/gate.yaml (or `path`). Every key is required; no silent defaults."""
    with Path(path or DEFAULT_CONFIG).open() as f:
        raw = yaml.safe_load(f) or {}
    return GateConfig(
        stale_s=float(raw["stale_s"]),
        min_conf=float(raw["min_conf"]),
        slow_alpha=float(raw["slow_alpha"]),
        resume_hold_s=float(raw["resume_hold_s"]),
        future_tol_s=float(raw["future_tol_s"]),
        max_speed=_per_joint("max_speed", raw["max_speed"]),
        dt_max=float(raw["dt_max"]),
        max_follow_err=_per_joint("max_follow_err", raw["max_follow_err"]),
        slow_speed_scale=float(raw["slow_speed_scale"]),
    )


def _per_joint(name: str, value) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list (one value per joint), got {value!r}")
    return tuple(float(v) for v in value)


class Reason(StrEnum):
    """Why a step requested what it did (GateOutput.reason). Anything but OK/LOW_CONF -> STOP."""

    OK = "ok"
    NO_DECISION = "no_decision"  # G1: decision is None
    STALE = "stale"  # G1: now - t_mono > stale_s
    FUTURE = "future"  # t_mono - now > future_tol_s (clock mismatch)
    NON_FINITE_TIME = "non_finite_time"
    UNKNOWN_CHOICE = "unknown_choice"
    NON_FINITE_PROBS = "non_finite_probs"
    LOW_CONF = "low_conf"  # G2: one level safer
    FOLLOW_ERROR = "follow_error"  # G9: arm is not following the previous command
    NON_FINITE_TARGET = "non_finite_target"


@dataclass(frozen=True)
class GateOutput:
    q_cmd: np.ndarray
    state: Choice  # state actually applied this step
    requested: Choice  # what this step asked for after G1/G2 and input checks
    reason: Reason  # why `requested` is what it is
    rate_limited: bool  # G8 scaled the step down this time
    clock_backwards: bool  # `now` was earlier than on the previous call (treated as dt = 0)
    follow_error: bool  # G9 detected, even if another reason took priority


class SafetyGate:
    def __init__(self, cfg: GateConfig, on_resume: Callable[[], None] | None = None):
        self.cfg = cfg
        self.on_resume = on_resume
        self.state = Choice.STOP  # start stopped; leaving STOP needs a hold (G6)
        self._pending: Choice | None = None  # relaxed level being held toward
        self._pending_since = 0.0
        self._prev_now: float | None = None
        self._q_cmd_prev: np.ndarray | None = None  # last output; q_now while STOPped

    def requested_choice(self, decision: Decision | None, now: float) -> tuple[Choice, Reason]:
        """G1 + G2 + input checks: the level this decision asks for. Anything odd -> STOP."""
        if decision is None:
            return Choice.STOP, Reason.NO_DECISION
        if not isinstance(decision.choice, Choice):
            return Choice.STOP, Reason.UNKNOWN_CHOICE
        t = decision.t_mono
        if not math.isfinite(t):
            return Choice.STOP, Reason.NON_FINITE_TIME
        if now - t > self.cfg.stale_s:  # G1
            return Choice.STOP, Reason.STALE
        if t - now > self.cfg.future_tol_s:
            return Choice.STOP, Reason.FUTURE
        probs = list(decision.probs.values())
        if not all(math.isfinite(p) for p in probs):
            return Choice.STOP, Reason.NON_FINITE_PROBS
        conf = max(probs, default=0.0)
        if conf < self.cfg.min_conf:  # G2: one level safer
            return Choice(min(decision.choice + 1, Choice.STOP)), Reason.LOW_CONF
        return decision.choice, Reason.OK

    def _transition(self, requested: Choice, now: float) -> None:
        """G6 hysteresis and G7 resume hook."""
        if requested >= self.state:
            if requested > self.state:
                self.state = requested  # tightening: immediate
            self._pending = None  # same or tighter request breaks any hold
            return
        if requested != self._pending:
            self._pending, self._pending_since = requested, now
        if now - self._pending_since >= self.cfg.resume_hold_s:
            left_stop = self.state is Choice.STOP
            self.state, self._pending = requested, None
            if left_stop and self.on_resume is not None:
                self.on_resume()  # G7

    def _dt(self, now: float) -> tuple[float, bool]:
        """Time since the previous call, capped at dt_max. First call / backwards -> 0."""
        prev, self._prev_now = self._prev_now, now
        if prev is None:
            return 0.0, False
        if now < prev:
            return 0.0, True
        return min(now - prev, self.cfg.dt_max), False

    def _rate_limit(self, q_ref, q_cmd, dt: float) -> tuple[np.ndarray, bool]:
        """G8: scale the change from q_ref so that |change_j| <= max_speed[j] * dt
        (times slow_speed_scale while SLOW)."""
        d = q_cmd - q_ref
        moving = d != 0
        if not moving.any():
            return q_cmd, False
        scale = self.cfg.slow_speed_scale if self.state is Choice.SLOW else 1.0
        limits = np.asarray(self.cfg.max_speed) * (scale * dt)
        s = min(1.0, float(np.min(limits[moving] / np.abs(d[moving]))))
        if s >= 1.0:
            return q_cmd, False
        return q_ref + s * d, True

    def _follow_error(self, q_now: np.ndarray) -> bool:
        """G9: the arm is further from the previous command than max_follow_err allows."""
        if self._q_cmd_prev is None:
            return False
        err = np.abs(self._q_cmd_prev - q_now)
        return bool(np.any(err > np.asarray(self.cfg.max_follow_err)))

    def step(self, decision: Decision | None, now: float, q_now, q_target) -> GateOutput:
        q_now = np.array(q_now, dtype=np.float64)  # copies: q_cmd never aliases caller arrays
        q_target = np.array(q_target, dtype=np.float64)
        if q_now.shape != q_target.shape:
            raise ValueError(f"shape mismatch: q_now {q_now.shape} vs q_target {q_target.shape}")
        n_joints = len(self.cfg.max_speed)
        if q_now.shape != (n_joints,):
            raise ValueError(f"q_now has shape {q_now.shape}; max_speed lists {n_joints} joints")
        if not np.all(np.isfinite(q_now)):
            # STOP would send q_now; a non-finite position cannot be sent safely.
            raise ValueError(f"q_now must be finite, got {q_now}")
        dt, clock_backwards = self._dt(now)
        follow_error = self._follow_error(q_now)

        # reason priority: non_finite_target > follow_error > decision reasons
        requested, reason = self.requested_choice(decision, now)
        if follow_error:
            requested, reason = Choice.STOP, Reason.FOLLOW_ERROR
        if not np.all(np.isfinite(q_target)):
            requested, reason = Choice.STOP, Reason.NON_FINITE_TARGET
        self._transition(requested, now)

        rate_limited = False
        if self.state is Choice.STOP:  # G3, not rate limited; resets the G8/G9 reference
            q_cmd = q_now
        else:
            if self.state is Choice.SLOW:  # G4
                q_cmd = q_now + self.cfg.slow_alpha * (q_target - q_now)
            else:  # G5
                q_cmd = q_target
            q_ref = q_now if self._q_cmd_prev is None else self._q_cmd_prev
            q_cmd, rate_limited = self._rate_limit(q_ref, q_cmd, dt)  # G8, applied last
        self._q_cmd_prev = q_cmd
        return GateOutput(
            q_cmd=q_cmd,
            state=self.state,
            requested=requested,
            reason=reason,
            rate_limited=rate_limited,
            clock_backwards=clock_backwards,
            follow_error=follow_error,
        )
