"""M4a: the safety gate between the policy and the joint command, on sim time (no ROS).

GatedCommand is the `gate` of eval_loop.run_closed_loop. Each tick:
  1. the judge runner gets the newest image (sim_judge.JudgeRunner: latency, dropouts, ...)
  2. the policy is asked only when the gate is not STOPped (policy_wanted); while STOPped the
     target is the current position, so ACT's chunk is not used up during a stop
  3. SafetyGate.step(latest decision, sim now, q_now, q_target) -> q_cmd (gate/safety_gate.py,
     unchanged; configs/gate_sim.yaml)
G7: on_resume calls client.reset(), i.e. the policy server's begin_episode(), the one place that
calls policy.reset(): after a stop the policy infers a new chunk from the current observation.
The tick it leaves STOP sends q_now (the policy was not asked on that tick), so no command from
the chunk computed before the stop is ever sent.
All times are sim time; Decision.t_mono is the image stamp (sim), so G1 sees the image's age.

SLOW time scaling (M4a-S, `time_scale`, configs/m4a.yaml slow.time_scale): while the gate is
SLOW the policy's actions are taken out at time_scale per tick instead of one per tick, and the
target is interpolated between the last two (a_prev + phase * (a_next - a_prev)). ACT's chunk
then advances at time_scale of its speed, so the arm really slows down and does not fall behind
the chunk; leaving SLOW, the one action read ahead (a_next) is sent first, then one action per
tick again (no catch-up jump). STOP and on_resume drop this state. time_scale = 1 behaves like
the plain gate (one action per tick).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from gate.safety_gate import GateConfig, SafetyGate
from jevlite.api import Choice
from scripts.sim.sim_judge import JudgeRunner


class GatedCommand:
    def __init__(self, cfg: GateConfig, runner: JudgeRunner, client, names: list[str],
                 truth: Callable[[float], str] | None = None, time_scale: float = 1.0):  # fmt: skip
        if not 0.0 < time_scale <= 1.0:
            raise ValueError(f"time_scale must be in (0, 1], got {time_scale}")
        self.cfg, self.runner, self.client = cfg, runner, client
        self.names = list(names)  # joint order of q_now / q_target / q_cmd (6, gripper last)
        self.truth = truth
        self.time_scale = float(time_scale)
        self.gate: SafetyGate | None = None
        self.resumes: list[dict] = []
        self._resumed = False
        self._clear()

    def _clear(self) -> None:
        """Forget the SLOW interpolation and the last sent target."""
        self._a_prev = self._a_next = None
        self._phase = 0.0
        self._carry = None  # a_next read ahead in SLOW, still to be sent after leaving it
        self._last = None  # last target taken from the policy
        self._plan = "stop"

    def _on_resume(self):  # G7
        rep = self.client.reset()
        self._resumed = True
        self._clear()  # the policy starts a new chunk: no old action is interpolated or sent
        self.resumes.append({"queue_len": rep.get("queue_len"), "episode": rep.get("episode")})
        if rep.get("queue_len", 0) != 0:
            raise RuntimeError(f"policy reset on resume left {rep['queue_len']} queued actions")

    def begin(self, t0: float) -> None:
        self.t0 = t0
        self.gate = SafetyGate(self.cfg, on_resume=self._on_resume)
        self.runner.begin(t0)
        self.resumes = []
        self._clear()

    def policy_wanted(self) -> bool:
        """Plan this tick from the state before the step; True when an action is to be read."""
        st = self.gate.state
        if st is Choice.STOP:
            self._plan = "stop"
        elif st is Choice.CONTINUE:
            self._plan = "carry" if self._carry is not None else "continue"
        elif self._a_next is None:
            self._plan = "slow_enter"
        else:
            self._phase += self.time_scale
            self._plan = "slow_fetch" if self._phase >= 1.0 - 1e-9 else "slow_interp"
        return self._plan in ("continue", "slow_enter", "slow_fetch")

    def _target(self, q_now: np.ndarray, action) -> np.ndarray:
        plan = self._plan
        if plan == "stop":
            self._a_prev = self._a_next = self._carry = None
            self._phase = 0.0
            return q_now.copy()
        if plan == "continue":
            self._last = action
            return action
        if plan == "carry":
            target, self._carry = self._carry, None
            self._last = target
            return target
        if plan == "slow_enter":
            self._a_prev = self._last if self._last is not None else q_now.copy()
            self._a_next, self._phase = action, self.time_scale
        elif plan == "slow_fetch":
            self._a_prev, self._a_next = self._a_next, action
            self._phase -= 1.0
        return self._a_prev + self._phase * (self._a_next - self._a_prev)

    def _after(self, state: Choice) -> None:
        """Leaving SLOW for CONTINUE: send the read-ahead action next; STOP drops everything."""
        if self._plan.startswith("slow") and state is Choice.CONTINUE:
            self._carry = self._a_next if self._phase < 1.0 - 1e-9 else None
            self._last = self._a_next
            self._a_prev = self._a_next = None
            self._phase = 0.0
        elif state is Choice.STOP:
            self._a_prev = self._a_next = self._carry = None
            self._phase = 0.0

    def filter(self, obs, now: float, names: list[str], q_target) -> tuple[np.ndarray, dict]:
        if list(names) != self.names:
            raise ValueError(f"joint names {names} != {self.names}")
        q_now = np.array([obs.joints[n] for n in self.names], np.float64)
        called = q_target is not None
        plan = self._plan
        if called != (plan in ("continue", "slow_enter", "slow_fetch")):
            raise RuntimeError(f"policy called={called} does not match the plan {plan!r}")
        q_target = self._target(q_now, None if q_target is None else np.asarray(q_target, float))
        phase = self._phase
        d = self.runner.step(obs.image, obs.t_image, now)
        self._resumed = False
        out = self.gate.step(d, now, q_now, q_target)
        if not self._resumed:  # on_resume already cleared everything
            self._after(out.state)
        rec = {
            "now": round(now - self.t0, 4),
            "decision": None if d is None else {
                "choice": d.choice.name, "conf": round(d.confidence, 4),
                "age_s": round(now - d.t_mono, 4), "latency_ms": round(d.latency_ms, 2),
                "source": d.source,
            },
            "truth": self.truth(obs.t_image) if self.truth else None,
            "requested": out.requested.name, "state": out.state.name, "reason": str(out.reason),
            "rate_limited": out.rate_limited, "follow_error": out.follow_error,
            "clock_backwards": out.clock_backwards, "resumed": self._resumed,
            "policy_called": called, "plan": plan,
            "phase": round(phase, 4) if plan.startswith("slow") else None,
            "q_now": [round(float(v), 6) for v in q_now],
            "q_target": [round(float(v), 6) for v in q_target],
            "q_cmd": [round(float(v), 6) for v in out.q_cmd],
        }  # fmt: skip
        return out.q_cmd, rec


# ---------- metrics of one gated episode (numpy only) ----------


def _spans(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    out, start = [], None
    for k, m in enumerate(mask):
        if m and start is None:
            start = t[k]
        if not m and start is not None:
            out.append((start, t[k]))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def tcp_speed(t: np.ndarray, tcp: np.ndarray) -> np.ndarray:
    """|dTCP/dt| per tick [m/s] (backward difference; 0 at the first tick)."""
    v = np.zeros(len(t))
    if len(t) > 1:
        v[1:] = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)
    return v


def slow_stats(t, state, speed, q_target, q_now, rate_limited, exit_window_s: float = 0.5) -> dict:
    """SLOW in one episode: spans, TCP speed during SLOW (mean / max, same unit as `speed`), the
    max speed in exit_window_s after each SLOW -> CONTINUE exit, the arm's lag behind the target
    (max |q_target - q_now| over the arm joints) at the end of each span and during SLOW, and the
    rate_limited fraction during SLOW. q_* are (T, >= 5); only the first 5 (arm) joints count."""
    t, state, speed = np.asarray(t, float), np.asarray(state), np.asarray(speed, float)
    lag = np.abs(np.asarray(q_target, float)[:, :5] - np.asarray(q_now, float)[:, :5]).max(1)
    slow = state == "SLOW"
    if not slow.any():
        return {"slow_s": 0.0}
    spans, exits, lag_end = [], [], []
    k = 0
    while k < len(t):
        if not slow[k]:
            k += 1
            continue
        a = k
        while k < len(t) and slow[k]:
            k += 1
        spans.append([round(float(t[a]), 3), round(float(t[k - 1]), 3)])
        lag_end.append(float(lag[k - 1]))
        if k < len(t) and state[k] == "CONTINUE":
            exits.append(float(t[k - 1]))
    dt = np.diff(t, append=t[-1] + (t[1] - t[0] if len(t) > 1 else 0))
    exit_max = [float(speed[(t > e) & (t <= e + exit_window_s)].max()) for e in exits
                if ((t > e) & (t <= e + exit_window_s)).any()]  # fmt: skip
    return {
        "slow_s": round(float(dt[slow].sum()), 3),
        "spans": spans,
        "exits": exits,
        "speed_mean": round(float(speed[slow].mean()), 4),
        "speed_max": round(float(speed[slow].max()), 4),
        "exit_speed_max": round(max(exit_max), 4) if exit_max else None,
        "lag_end_max": round(max(lag_end), 4),
        "lag_max": round(float(lag[slow].max()), 4),
        "rate_limited_frac": round(float(np.mean(np.asarray(rate_limited, bool)[slow])), 4),
    }


def gate_metrics(ticks: list[dict], fk, react_s: float = 0.0) -> dict:
    """Summary of a gated episode from its ticks (each with `t`, `gate`).

    react_s: allowance after the truth turns `center` before the arm is expected to be still
    (judge latency + the gate + the arm settling), for tcp_speed_max_center_settled."""
    t = np.array([k["t"] for k in ticks], float)
    g = [k["gate"] for k in ticks]
    dt = np.diff(t, append=t[-1] + (t[1] - t[0] if len(t) > 1 else 0))
    state = np.array([x["state"] for x in g])
    truth = np.array([x["truth"] or "none" for x in g])
    q = np.array([x["q_now"][:5] for x in g], float)
    v = tcp_speed(t, fk(q))
    active = state != "STOP"
    reasons: dict[str, int] = {}
    for x in g:
        reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    ages = [x["decision"]["age_s"] for x in g if x["decision"]]
    out = {
        "state_s": {c.name: round(float(dt[state == c.name].sum()), 3) for c in Choice},
        "resumes": int(sum(x["resumed"] for x in g)),
        "resume_t": [float(t[k]) for k, x in enumerate(g) if x["resumed"]],
        "reasons": reasons,
        "rate_limited_ticks": int(sum(x["rate_limited"] for x in g)),
        "rate_limited_frac_active": round(
            float(np.mean([x["rate_limited"] for x, a in zip(g, active, strict=True) if a]))
            if active.any() else 0.0, 4),  # fmt: skip
        "follow_error_ticks": int(sum(x["follow_error"] for x in g)),
        "policy_calls": int(sum(x["policy_called"] for x in g)),
        "decision_age_s_max": round(max(ages), 4) if ages else None,
        "tcp_speed_max": round(float(v.max()), 4),
        "slow": slow_stats(t, state, v, [x["q_target"] for x in g], q,
                           [x["rate_limited"] for x in g]),  # fmt: skip
        "intrusion": None,
    }
    intr = truth != "none"
    if intr.any():
        center = truth == "center"
        info = {
            "spans": [[round(a, 3), round(b, 3)] for a, b in _spans(intr, t)],
            "center_spans": [[round(a, 3), round(b, 3)] for a, b in _spans(center, t)],
            "tcp_speed_max": round(float(v[intr].max()), 4),
            "tcp_speed_max_edge": round(float(v[truth == "edge"].max()), 4)
            if (truth == "edge").any() else None,
        }  # fmt: skip
        if center.any():
            c0 = t[np.argmax(center)]
            stops = t[(state == "STOP") & (t >= c0)]
            info["stop_after_center_s"] = round(float(stops[0] - c0), 4) if len(stops) else None
            settled = center & (t >= c0 + react_s)
            info["tcp_speed_max_center_settled"] = (
                round(float(v[settled].max()), 4) if settled.any() else None)  # fmt: skip
            c1 = t[len(center) - 1 - np.argmax(center[::-1])]  # last centre tick
            res = [rt for rt in out["resume_t"] if rt > c1]
            info["resume_after_center_s"] = round(res[0] - c1, 4) if res else None
        out["intrusion"] = info
    return out
