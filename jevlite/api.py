"""Common types for all judges (v0 / teacher / student).

Jev-style output: a choice, its probabilities and the latency in ms. Nothing else.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable

import numpy as np


class Choice(IntEnum):
    """Ordered by severity: CONTINUE < SLOW < STOP."""

    CONTINUE = 0
    SLOW = 1
    STOP = 2

    @classmethod
    def parse(cls, name: str) -> Choice:
        return cls[name.strip().upper()]


@dataclass(frozen=True)
class Decision:
    choice: Choice
    probs: dict[str, float]
    # Capture time of the image this decision was made from, on the time.monotonic() clock
    # (camera input: the V4L2 buffer timestamp, jevlite.camera.Frame.t_capture, passed in as
    # judge(image, t_capture=...)). Without t_capture (e.g. a saved image) it is the time the
    # judge started, which is closer to the real capture than the finish time. It is never the
    # finish time. The gate's G1 uses now - t_mono as the age of the image.
    t_mono: float
    # Time the judge itself took (preprocess + inference). Not the age of the image:
    # age ~= wait from capture to judge start + latency_ms.
    latency_ms: float
    source: str

    @property
    def confidence(self) -> float:
        return max(self.probs.values()) if self.probs else 0.0


@dataclass(frozen=True)
class Score:
    """A scalar judgement in [0, 1] (reserved for later use)."""

    value: float
    t_mono: float
    latency_ms: float
    source: str


@dataclass(frozen=True)
class YesNo:
    """A binary judgement (reserved for later use)."""

    yes: bool
    p_yes: float
    t_mono: float
    latency_ms: float
    source: str
    extra: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Judge(Protocol):
    def judge(self, image: np.ndarray, t_capture: float | None = None) -> Decision:
        """Judge one RGB image. `t_capture` (time.monotonic() clock) becomes Decision.t_mono;
        without it, Decision.t_mono is the time judging started."""
        ...


class HasCaptureTime(Protocol):
    rgb: np.ndarray
    t_capture: float


@dataclass(frozen=True)
class JudgeStart:
    """Taken when judge() starts: t_mono for the Decision, perf0 for latency_ms."""

    t_mono: float
    perf0: float

    def latency_ms(self) -> float:
        return (time.perf_counter() - self.perf0) * 1000.0


def start_judging(t_capture: float | None = None) -> JudgeStart:
    """Call first thing in judge(). Decision.t_mono = t_capture if given (camera frames), else
    the judge start time (saved images: closer to the real capture than the finish time)."""
    t_mono = time.monotonic() if t_capture is None else float(t_capture)
    return JudgeStart(t_mono=t_mono, perf0=time.perf_counter())


def judge_frame(judge: Judge, frame: HasCaptureTime) -> Decision:
    """Judge a camera Frame, carrying its capture time into Decision.t_mono."""
    return judge.judge(frame.rgb, t_capture=frame.t_capture)


def decision_from_probs(probs: dict[str, float], source: str, start: JudgeStart) -> Decision:
    """Build a Decision from per-choice probabilities.

    `start` comes from start_judging() at the beginning of judge(): it gives t_mono and the
    reference for latency_ms (time spent judging). Ties are broken toward the more severe
    choice (safe side).
    """
    names = {c.name for c in Choice}
    if set(probs) != names:
        raise ValueError(f"probs keys must be {sorted(names)}, got {sorted(probs)}")
    total = sum(probs.values())
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        raise ValueError(f"probs must sum to 1, got {total}")
    best = max(Choice, key=lambda c: (probs[c.name], c.value))
    return Decision(
        choice=best,
        probs=dict(probs),
        t_mono=start.t_mono,
        latency_ms=start.latency_ms(),
        source=source,
    )
