"""Closed-loop policy run on sim time (no ROS here: eval_policy.py supplies the environment).

Every 1/rate_hz of sim time: take the newest camera image and joint state, ask the policy
server for an action, send the 6-joint command it returns. Before the first tick the policy is
reset (its action queue emptied: one episode never continues the previous one's chunk).
Recorded per tick, and summarized:
  infer_ms   the server's own time for the step (preprocess + select_action + postprocess);
             large only on ticks that ran the network (a new chunk)
  rtt_ms     wall time of the whole request (encode, socket, server, decode)
  obs_age_s  sim time from the image stamp to sending the command (how old the picture is when
             the command based on it goes out; as the gate's G1 sees it)
  late_s     sim time the tick started after its due time
  overrun    the command went out after the next tick was due
Optional hooks (M3e): on_tick(i, t) runs at each tick before the observation is taken (e.g. to
move the block); a dict it returns is kept as the tick's `event`. extra(obs) returns fields added
to every tick record (e.g. the joint state and the block pose).
Optional gate (M4a, scripts/sim/gate_loop.GatedCommand): begin(t0) at the start; each tick the
policy is asked only if gate.policy_wanted() (not while the gate is STOPped: the chunk is not
used up and q_target is the current position), then gate.filter(obs, now, names, q_target)
turns the policy's command into the one sent, and its record is kept as the tick's `gate`.
With gate=None the loop and its records are exactly as before.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Obs:
    image: np.ndarray  # (H, W, 3) uint8 RGB
    t_image: float  # sim time of the image stamp [s]
    joints: dict[str, float]  # latest joint positions by name [rad]


class EvalEnv(Protocol):
    def sim_now(self) -> float: ...

    def wait_until(self, t: float) -> bool:
        """Spin until sim time >= t; False if the wall-clock limit ran out."""
        ...

    def latest_obs(self) -> Obs | None: ...

    def send(self, names: list[str], q: np.ndarray) -> None: ...


class Client(Protocol):
    def reset(self) -> dict: ...

    def act(self, image: np.ndarray, state: np.ndarray) -> dict: ...


def _pct(v: list[float], q: float) -> float | None:
    return float(np.percentile(v, q)) if v else None


def summarize_ticks(ticks: list[dict]) -> dict:
    inf = [t["infer_ms"] for t in ticks if t["inferred"]]
    rtt = [t["rtt_ms"] for t in ticks]
    age = [t["obs_age_s"] * 1e3 for t in ticks]
    return {
        "ticks": len(ticks),
        "inferences": len(inf),
        "overruns": int(sum(t["overrun"] for t in ticks)),
        "infer_ms_p50": _pct(inf, 50),
        "infer_ms_p95": _pct(inf, 95),
        "infer_ms_max": max(inf) if inf else None,
        "rtt_ms_p50": _pct(rtt, 50),
        "rtt_ms_p95": _pct(rtt, 95),
        "rtt_ms_max": max(rtt) if rtt else None,
        "obs_age_ms_p50": _pct(age, 50),
        "obs_age_ms_p95": _pct(age, 95),
        "obs_age_ms_max": max(age) if age else None,
        "late_ms_max": max(t["late_s"] for t in ticks) * 1e3 if ticks else None,
    }


def run_closed_loop(
    env: EvalEnv,
    client: Client,
    state_joints: list[str],
    rate_hz: float,
    duration_s: float,
    obs_timeout_s: float = 5.0,
    on_tick: Callable[[int, float], dict | None] | None = None,
    extra: Callable[[Obs], dict] | None = None,
    gate=None,
) -> dict:
    """Reset the policy, then run round(duration_s * rate_hz) ticks. Returns summary + ticks."""
    rep = client.reset()
    if rep.get("queue_len", 0) != 0:
        raise RuntimeError(f"policy reset left {rep['queue_len']} queued actions")
    t0 = env.sim_now()
    if not env.wait_until(t0):
        raise TimeoutError("sim clock not running")
    end = time.monotonic() + obs_timeout_s
    while env.latest_obs() is None:
        if time.monotonic() > end:
            raise TimeoutError("no camera image / joint state")
        env.wait_until(env.sim_now() + 1.0 / rate_hz)
    t0 = env.sim_now()
    if gate is not None:
        gate.begin(t0)
    ticks = []
    for i in range(int(round(duration_s * rate_hz))):
        due = t0 + i / rate_hz
        if not env.wait_until(due):
            raise TimeoutError(f"sim time did not reach tick {i}")
        late = env.sim_now() - due
        event = on_tick(i, due - t0) if on_tick is not None else None
        obs = env.latest_obs()
        state = np.array([obs.joints[j] for j in state_joints], np.float32)
        w = time.monotonic()
        r = client.act(obs.image, state) if gate is None or gate.policy_wanted() else None
        rtt_ms = (time.monotonic() - w) * 1e3
        if gate is None:
            env.send(list(r["names"]), np.asarray(r["command"], np.float64))
        else:
            names = list(r["names"]) if r is not None else list(gate.names)
            target = np.asarray(r["command"], np.float64) if r is not None else None
            q_cmd, grec = gate.filter(obs, env.sim_now(), names, target)
            env.send(names, q_cmd)
        t_sent = env.sim_now()
        rec = {
            "i": i,
            "t": round(due - t0, 4),
            "late_s": late,
            "obs_age_s": t_sent - obs.t_image,
            "infer_ms": float(r["infer_ms"]) if r is not None else 0.0,
            "inferred": bool(r["inferred"]) if r is not None else False,
            "rtt_ms": rtt_ms,
            "overrun": t_sent > t0 + (i + 1) / rate_hz,
            "action": [float(v) for v in np.asarray(r["action"]).ravel()] if r is not None else [],
        }
        if gate is not None:
            rec["gate"] = grec
        if extra is not None:
            rec.update(extra(obs))
        if event:
            rec["event"] = event
        ticks.append(rec)
    return {"summary": summarize_ticks(ticks), "ticks": ticks}


def pick_examples(eps: list[dict]) -> dict[str, dict]:
    """One success (smallest error) and one failure (first by seed); if there is no failure,
    the success with the largest error stands in as 'worst'."""
    done = [e for e in eps if e.get("status") == "done"]
    ok = [e for e in done if e["result"]["success"]]
    bad = [e for e in done if not e["result"]["success"]]
    out = {}
    if ok:
        out["success"] = min(ok, key=lambda e: e["result"]["error_m"])
    if bad:
        out["failure"] = min(bad, key=lambda e: e["seed"])
    elif len(ok) > 1:
        out["worst"] = max(ok, key=lambda e: e["result"]["error_m"])
    return out


def summarize_episodes(eps: list[dict]) -> dict:
    done = [e for e in eps if e.get("status") == "done"]
    if not done:
        return {"episodes": 0}
    err = [e["result"]["error_m"] for e in done]
    ok = [e for e in done if e["result"]["success"]]
    ticks = [t for e in done for t in e["_ticks"]]
    return {
        "episodes": len(done),
        "successes": len(ok),
        "success_rate": round(len(ok) / len(done), 3),
        "error_m_mean": round(float(np.mean(err)), 4),
        "error_m_max": round(float(np.max(err)), 4),
        "error_m_min": round(float(np.min(err)), 4),
        "error_m_mean_success": round(float(np.mean([e["result"]["error_m"] for e in ok])), 4)
        if ok
        else None,
        "loop": summarize_ticks(ticks),
        "rtf_loop_mean": round(float(np.mean([e["rtf_loop"] for e in done])), 3),
        "mean_wall_s": round(float(np.mean([e["wall_duration_s"] for e in done])), 1),
    }
