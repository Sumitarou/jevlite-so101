"""Align bag topics to the action ticks (numpy only: used by the main env's tests and by train/).

Every frame of the dataset is one `/demo/action` message (time t = its header stamp, sim time).
- observation.state: `/joint_states` linearly interpolated at t, per joint. A frame outside the
  recorded range is dropped by default (the recorder may subscribe to /joint_states a little after
  the commands start, while the arm is already moving); with hold_state=True the end value is held
  and the frame is only marked.
- observation.images.*: the newest image with stamp <= t ("at or before the action"). A frame
  with no earlier image gets index -1 and is to be dropped.
Times are integer nanoseconds (header stamps) so "at or before" has no float rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stamp_ns(sec: int, nanosec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nanosec)


def _check_sorted(name: str, t: np.ndarray) -> None:
    if t.ndim != 1 or len(t) == 0:
        raise ValueError(f"{name}: need a non-empty 1-D array of times")
    if np.any(np.diff(t) < 0):
        raise ValueError(f"{name}: times are not sorted")


def interp_state(
    t_query: np.ndarray, t_src: np.ndarray, q_src: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint linear interpolation of q_src (N, J) at t_query (M,).

    Returns (q (M, J), outside (M,) bool): outside is True where t_query lies before the first or
    after the last source sample (the end value is held there).
    """
    t_query, t_src = np.asarray(t_query, np.int64), np.asarray(t_src, np.int64)
    q_src = np.asarray(q_src, np.float64)
    _check_sorted("t_src", t_src)
    if q_src.ndim != 2 or q_src.shape[0] != len(t_src):
        raise ValueError("q_src must be (len(t_src), J)")
    # interpolate relative to the first sample so float64 keeps ns resolution
    x, xp = (t_query - t_src[0]).astype(np.float64), (t_src - t_src[0]).astype(np.float64)
    q = np.stack([np.interp(x, xp, q_src[:, j]) for j in range(q_src.shape[1])], axis=1)
    outside = (t_query < t_src[0]) | (t_query > t_src[-1])
    return q, outside


def nearest_gap_ns(t_query: np.ndarray, t_src: np.ndarray) -> np.ndarray:
    """|t_query - nearest source sample| in ns (how far the interpolation reaches)."""
    t_query, t_src = np.asarray(t_query, np.int64), np.asarray(t_src, np.int64)
    _check_sorted("t_src", t_src)
    i = np.searchsorted(t_src, t_query)
    left = t_src[np.clip(i - 1, 0, len(t_src) - 1)]
    right = t_src[np.clip(i, 0, len(t_src) - 1)]
    return np.minimum(np.abs(t_query - left), np.abs(right - t_query))


def prev_index(t_query: np.ndarray, t_src: np.ndarray) -> np.ndarray:
    """Index of the newest source sample with t_src <= t_query, or -1 if there is none."""
    t_query, t_src = np.asarray(t_query, np.int64), np.asarray(t_src, np.int64)
    _check_sorted("t_src", t_src)
    return np.searchsorted(t_src, t_query, side="right") - 1


@dataclass
class Alignment:
    keep: np.ndarray  # (M,) bool: frame is used (has an image at or before it, and a state)
    state: np.ndarray  # (M, J) interpolated joint state
    state_outside: np.ndarray  # (M,) bool: t outside the /joint_states range
    state_gap_ns: np.ndarray  # (M,) distance to the nearest joint_states sample
    image_index: np.ndarray  # (M,) index into the image list, -1 = none
    image_age_ns: np.ndarray  # (M,) t_action - t_image (only meaningful where keep)

    def stats(self) -> dict:
        k = self.keep
        has_img = self.image_index >= 0
        age = self.image_age_ns[k] / 1e6
        gap = self.state_gap_ns[k] / 1e6
        outside = self.state_outside & k
        return {
            "frames_in": int(len(k)),
            "frames_kept": int(k.sum()),
            "dropped_no_image": int((~has_img).sum()),
            "dropped_no_state": int((has_img & ~k).sum()),
            "image_age_ms_max": float(age.max()) if age.size else None,
            "image_age_ms_mean": float(age.mean()) if age.size else None,
            "state_gap_ms_max": float(gap.max()) if gap.size else None,
            "state_gap_ms_mean": float(gap.mean()) if gap.size else None,
            "state_held_frames": int(outside.sum()),
            "state_held_ms_max": float(gap[outside[k]].max()) if outside.any() else 0.0,
            "image_reused_frames": int(k.sum() - len(np.unique(self.image_index[k]))),
        }


def align(
    t_action: np.ndarray,
    t_js: np.ndarray,
    q_js: np.ndarray,
    t_img: np.ndarray,
    hold_state: bool = False,
) -> Alignment:
    t_action = np.asarray(t_action, np.int64)
    _check_sorted("t_action", t_action)
    q, outside = interp_state(t_action, t_js, q_js)
    idx = prev_index(t_action, t_img)
    keep = (idx >= 0) & (hold_state | ~outside)
    t_img = np.asarray(t_img, np.int64)
    age = np.where(idx >= 0, t_action - t_img[np.clip(idx, 0, None)], 0)
    return Alignment(keep, q, outside, nearest_gap_ns(t_action, t_js), idx, age)


def reorder(names_src: list[str], names_dst: list[str]) -> list[int]:
    """Column indices that put names_src in the order names_dst (KeyError if one is missing)."""
    pos = {n: i for i, n in enumerate(names_src)}
    missing = [n for n in names_dst if n not in pos]
    if missing:
        raise KeyError(f"joints missing: {missing}")
    return [pos[n] for n in names_dst]
