"""Submission video (configs/video.yaml): config checks and the timeline, numpy only.

The renderer (scripts/video/make_video.py, train env because it reads bags with rosbags) uses
these. Times: segments are closed-loop sim seconds, captions / annotations are output seconds
from the start of their section."""

from __future__ import annotations

import bisect
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

KINDS = {"title", "still", "clip", "compare", "card"}
TIMED = {"clip", "compare"}  # sections whose length comes from segments


def validate(cfg: dict) -> None:
    """Raise ValueError on a malformed config (before any rendering)."""
    out = cfg["output"]
    if out["width"] <= 0 or out["height"] <= 0 or out["fps"] <= 0:
        raise ValueError("output size / fps must be positive")
    names = set()
    for sec in cfg["sections"]:
        name, kind = sec.get("name"), sec.get("kind")
        if not name or name in names:
            raise ValueError(f"section name missing or duplicated: {name!r}")
        names.add(name)
        if kind not in KINDS:
            raise ValueError(f"{name}: unknown kind {kind!r}")
        for src in ([sec["source"]] if "source" in sec else []) + sec.get("sources", []):
            if src not in cfg["sources"]:
                raise ValueError(f"{name}: unknown source {src!r}")
        if kind in TIMED:
            segs = sec.get("segments") or []
            if not segs:
                raise ValueError(f"{name}: segments are required")
            prev_to = None
            for s in segs:
                if not s["speed"] > 0:
                    raise ValueError(f"{name}: speed must be > 0")
                if not s["to"] > s["from"]:
                    raise ValueError(f"{name}: segment to must be after from")
                if prev_to is not None and s["from"] < prev_to - 1e-9:
                    raise ValueError(f"{name}: segments must not go back in time")
                prev_to = s["to"]
            if sec.get("pre_hold_s", 0) < 0 or sec.get("hold_s", 0) < 0:
                raise ValueError(f"{name}: holds must be >= 0")
            if kind == "compare" and len(sec.get("sources", [])) != 2:
                raise ValueError(f"{name}: compare needs two sources")
            if kind == "clip" and "source" not in sec:
                raise ValueError(f"{name}: clip needs a source")
        elif not sec.get("duration_s", 0) > 0:
            raise ValueError(f"{name}: duration_s must be > 0")
        dur = section_frames(sec, out["fps"]) / out["fps"]
        for c in sec.get("captions", []):
            if not c["until"] > c["at"] or c["at"] < 0:
                raise ValueError(f"{name}: caption times must satisfy 0 <= at < until")
            if c["until"] > dur + 1e-6:
                raise ValueError(f"{name}: caption ends at {c['until']} s after the section "
                                 f"({dur:.2f} s)")  # fmt: skip
    lo, hi = cfg["duration_limits_s"]
    total = total_duration_s(cfg)
    if not lo <= total <= hi:
        raise ValueError(f"total {total:.1f} s is outside {lo}-{hi} s")


def build_timeline(segments: list[dict], fps: float, pre_hold_s: float = 0.0,
                   hold_s: float = 0.0) -> list[tuple[float, float]]:  # fmt: skip
    """(sim_t, speed) for every output frame. speed 0 = paused (the holds before and after)."""
    frames: list[tuple[float, float]] = []
    frames += [(float(segments[0]["from"]), 0.0)] * round(pre_hold_s * fps)
    for s in segments:
        a, b, v = float(s["from"]), float(s["to"]), float(s["speed"])
        n = round((b - a) / v * fps)
        frames += [(a + k * v / fps, v) for k in range(n)]
    frames += [(float(segments[-1]["to"]), 0.0)] * round(hold_s * fps)
    return frames


def section_frames(sec: dict, fps: float) -> int:
    if sec["kind"] in TIMED:
        return len(build_timeline(sec["segments"], fps, sec.get("pre_hold_s", 0),
                                  sec.get("hold_s", 0)))  # fmt: skip
    return round(sec["duration_s"] * fps)


def total_duration_s(cfg: dict) -> float:
    fps = cfg["output"]["fps"]
    return sum(section_frames(s, fps) for s in cfg["sections"]) / fps


def speed_label(speed: float, labels: dict) -> str:
    """Text for the playback rate; every rate other than 1 is spelled out with its factor."""
    if speed == 0:
        return labels["paused"]
    if abs(speed - 1.0) < 1e-9:
        return labels["normal"]
    key = "fast" if speed > 1 else "slow"
    return labels[key].format(s=f"{speed:g}")


def frame_index(stamps, t: float) -> int:
    """Index of the last frame at or before t (the first frame if t is earlier)."""
    return max(0, bisect.bisect_right(stamps, t + 1e-6) - 1)


def last_at(times, t: float) -> int:
    """Index of the last entry at or before t, -1 if none."""
    return bisect.bisect_right(times, t + 1e-6) - 1


def active_captions(captions: list[dict], t: float) -> list[str]:
    return [c["text"] for c in captions if c["at"] <= t < c["until"]]


WORD_EXTRA = set("±≤≥≈×")  # signs that belong to the number next to them
NO_START = set("）」』】、。，．・ー…:)")  # never at the start of a line (hangs on the previous)
NO_END = set("（「『【(")  # never at the end of a line (moves to the next)


def _word_char(ch: str) -> bool:
    return ch != " " and (ch.isascii() or ch in WORD_EXTRA)


def _tokens(text: str) -> list[str]:
    """Japanese characters one by one; runs of ASCII letters / digits / signs stay together."""
    out: list[str] = []
    for ch in text:
        if out and _word_char(ch) and _word_char(out[-1][-1]):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def wrap_text(text: str, width: float, measure: Callable[[str], float]) -> list[str]:
    """Wrap so every line fits width (Japanese has no spaces); ASCII words such as 'SLOW' or
    '2.5' are not split unless one alone is too wide. Keeps '\\n'."""
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for tok in _tokens(para):
            if cur and measure(cur + tok) > width and tok[0] not in NO_START:
                carry = ""
                while len(cur) > 1 and cur[-1] in NO_END:
                    cur, carry = cur[:-1], cur[-1] + carry
                lines.append(cur.rstrip())
                cur = carry + tok.lstrip()
            else:
                cur += tok
        lines.append(cur)
    return lines


def new_stamp_dir(base: Path, now: datetime | None = None) -> Path:
    """Create base/<YYYYmmdd_HHMMSS>/ (with _2, _3, ... if it exists), never reusing one."""
    stem = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    path, i = base / stem, 1
    while path.exists():
        i += 1
        path = base / f"{stem}_{i}"
    path.mkdir(parents=True)
    return path
