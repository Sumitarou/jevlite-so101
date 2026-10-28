"""M4a judges on sim time: the oracle (zone from the scripted intruder) and a delay wrapper.

The judge entrance is jevlite.api.Judge, judge(image, t_capture) -> Decision, so the oracle can
be swapped for an image model (M4b: SigLIP features + a small classifier) without touching the
loop. JudgeRunner wraps any such judge and models, on sim time, what a real judge adds:
  latency   a result comes out latency_s after judging started (null: use the judge's own
            Decision.latency_ms, e.g. a real model; the sim / wall-time ratio is then ignored)
  mode      serial: one at a time; when idle, take the newest image (one GPU). The image is up to
                    about 2 x latency old just before the next result.
            pipelined: every tick's image is judged; each result comes out latency_s later.
  dropouts  [[t0, t1], ...]: no judging starts in these windows (results stop coming)
  flicker   {t: [t0, t1], period_s, alt}: images taken in the window alternate, every period_s,
            between the judge's own answer and the zone `alt`
  confidence  probability put on the chosen choice; the other two share the rest
Times in the config are seconds from the episode start (begin(t0)); Decision.t_mono is the sim
time of the image the decision was made from (the image stamp), as the gate's G1 expects.

M4b: RemoteJudge asks the image judge server (scripts/sim/judge_server.py, uv env) over
policy_link and returns its probabilities with the judge's own latency_ms (wall time of preprocess +
inference). With latency_s null, JudgeRunner releases the result that many ms later on sim time:
the wall-time ms are used 1:1 as sim-time ms, i.e. as on the real robot, where sim time is real
time (scaling by the real-time factor ~0.36 would make the judge look ~2.8x faster than it is).
A failed request gives STOP (source "judge_error"), the safe side.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from jevlite.api import Choice, Decision, Judge

MODES = ("serial", "pipelined")


def probs_for(choice: Choice, confidence: float) -> dict[str, float]:
    """`confidence` on `choice`, the rest split evenly over the other two."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    rest = (1.0 - confidence) / (len(Choice) - 1)
    return {c.name: (confidence if c is choice else rest) for c in Choice}


def choose(probs: dict[str, float]) -> Choice:
    """The most probable choice; ties go to the more severe one (as jevlite.api)."""
    return max(Choice, key=lambda c: (probs[c.name], c.value))


class RemoteJudge:
    """Judge through the judge server. `client` has request(msg) -> reply (policy_link's
    PolicyClient); it raises on a failed request, which is answered with STOP here."""

    def __init__(self, client, source: str = "remote"):
        self.client, self.source = client, source
        self.rtt_ms: list[float] = []  # wall-time round trips (socket + judge), for the record
        self.errors: list[str] = []

    def judge(self, image: np.ndarray, t_capture: float | None = None) -> Decision:
        if t_capture is None:
            raise ValueError("RemoteJudge needs t_capture (the sim time of the image)")
        w = time.perf_counter()
        try:
            rep = self.client.request({"cmd": "judge", "image": image})
            probs = {c.name: float(rep["probs"][c.name]) for c in Choice}
            if not all(np.isfinite(v) for v in probs.values()):
                raise ValueError(f"probs not finite: {probs}")
            lat, src = float(rep["latency_ms"]), str(rep.get("source", self.source))
        except Exception as e:  # noqa: BLE001 -- any failure means no trustworthy answer
            self.errors.append(f"{type(e).__name__}: {e}")
            probs = probs_for(Choice.STOP, 1.0)
            lat, src = (time.perf_counter() - w) * 1e3, "judge_error"
        self.rtt_ms.append((time.perf_counter() - w) * 1e3)
        return Decision(choose(probs), probs, float(t_capture), lat, src)


class OracleJudge:
    """Judge from the truth: zone_at(t_capture) -> zone name, mapped by `zones` (m4a.yaml).
    The image is not looked at. Its own latency is 0 (JudgeRunner adds the modelled one)."""

    def __init__(self, zone_at: Callable[[float], str], zones: dict[str, str],
                 confidence: float = 1.0, source: str = "oracle"):  # fmt: skip
        self.zone_at, self.confidence, self.source = zone_at, confidence, source
        self.choices = {z: Choice.parse(c) for z, c in zones.items()}

    def judge(self, image: np.ndarray, t_capture: float | None = None) -> Decision:
        if t_capture is None:
            raise ValueError("the oracle needs t_capture (the sim time of the image)")
        choice = self.choices[self.zone_at(t_capture)]
        return Decision(choice, probs_for(choice, self.confidence), float(t_capture), 0.0,
                        self.source)  # fmt: skip


@dataclass
class _Job:
    t_image: float
    t_start: float
    t_release: float
    decision: Decision
    flickered: bool


class JudgeRunner:
    def __init__(self, judge: Judge, settings: dict, zones: dict[str, str],
                 truth: Callable[[float], str] | None = None):  # fmt: skip
        self.judge, self.s, self.truth = judge, dict(settings), truth
        if self.s["mode"] not in MODES:
            raise ValueError(f"judge mode must be one of {MODES}, got {self.s['mode']!r}")
        lat = self.s.get("latency_s")
        if lat is not None and lat < 0:
            raise ValueError(f"latency_s must be >= 0, got {lat}")
        self.zones = {z: Choice.parse(c) for z, c in zones.items()}
        self.t0: float | None = None
        self._jobs: deque[_Job] = deque()
        self._latest: Decision | None = None
        self.log: list[dict] = []  # one entry per released decision

    def begin(self, t0: float) -> None:
        self.t0 = t0
        self._jobs.clear()
        self._latest = None
        self.log = []

    def _in(self, window, t_rel: float) -> bool:
        return window[0] <= t_rel < window[1]

    def _dropped(self, t_rel: float) -> bool:
        return any(self._in(w, t_rel) for w in self.s.get("dropouts") or [])

    def _flicker(self, t_image_rel: float) -> Choice | None:
        f = self.s.get("flicker")
        if not f or not self._in(f["t"], t_image_rel):
            return None
        k = int(np.floor((t_image_rel - f["t"][0]) / f["period_s"]))
        return self.zones[f["alt"]] if k % 2 == 1 else None

    def _start(self, image, t_image: float, now: float) -> None:
        d = self.judge.judge(image, t_capture=t_image)
        lat = self.s.get("latency_s")
        lat_s = d.latency_ms / 1e3 if lat is None else float(lat)
        alt = self._flicker(t_image - self.t0)
        choice = alt if alt is not None else d.choice
        conf = self.s.get("confidence")
        probs = d.probs if conf is None and alt is None else probs_for(
            choice, d.confidence if conf is None else float(conf))  # fmt: skip
        d = replace(d, choice=choice, probs=probs, latency_ms=lat_s * 1e3)
        self._jobs.append(_Job(t_image, now, now + lat_s, d, alt is not None))

    def _release(self, now: float) -> None:
        while self._jobs and self._jobs[0].t_release <= now + 1e-9:
            j = self._jobs.popleft()
            self._latest = j.decision
            self.log.append({
                "t_image": round(j.t_image - self.t0, 4), "t_start": round(j.t_start - self.t0, 4),
                "t_release": round(j.t_release - self.t0, 4), "choice": j.decision.choice.name,
                "conf": round(j.decision.confidence, 4), "flicker": j.flickered,
                "latency_ms": round(j.decision.latency_ms, 3), "source": j.decision.source,
                "truth": self.truth(j.t_image) if self.truth else None,
            })  # fmt: skip

    def step(self, image: np.ndarray, t_image: float, now: float) -> Decision | None:
        """Release results due by `now`, maybe start judging this image; the latest result.
        A result with zero latency comes out in the same step."""
        if self.t0 is None:
            raise RuntimeError("begin(t0) first")
        self._release(now)
        idle = not self._jobs or self.s["mode"] == "pipelined"
        if idle and not self._dropped(now - self.t0):
            self._start(image, t_image, now)
            self._release(now)
        return self._latest


def _pct(v, q) -> float | None:
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def judge_log_metrics(log: list[dict], zones: dict[str, str], rtt_ms: list[float] | None = None,
                      errors: int = 0) -> dict:  # fmt: skip
    """Released decisions of one episode (JudgeRunner.log) against the truth at the image time:
    accuracy, center -> CONTINUE, milder / stricter than the truth, choice changes (flips), the
    modelled delay (t_release - t_start, sim) and the wall-time round trips."""
    rows = [e for e in log if e.get("truth") is not None]
    want = [Choice.parse(zones[e["truth"]]) for e in rows]
    got = [Choice.parse(e["choice"]) for e in rows]
    delay = [(e["t_release"] - e["t_start"]) * 1e3 for e in log]
    flips = sum(a["choice"] != b["choice"] for a, b in zip(log, log[1:], strict=False))
    return {
        "n": len(log),
        "accuracy": round(float(np.mean([g == w for g, w in zip(got, want, strict=True)])), 4)
        if rows else None,
        "center_to_continue": int(sum(e["truth"] == "center" and g is Choice.CONTINUE
                                      for e, g in zip(rows, got, strict=True))),
        "under": int(sum(g < w for g, w in zip(got, want, strict=True))),
        "over": int(sum(g > w for g, w in zip(got, want, strict=True))),
        "flips": int(flips),
        "delay_ms": {"p50": _pct(delay, 50), "p95": _pct(delay, 95),
                     "max": round(max(delay), 3) if delay else None},
        "rtt_ms": {"p50": _pct(rtt_ms or [], 50), "p95": _pct(rtt_ms or [], 95),
                   "max": round(max(rtt_ms), 3) if rtt_ms else None},
        "errors": int(errors),
    }  # fmt: skip
