"""scripts/sim/bag_align.py: aligning joint states and images to the action ticks."""

import numpy as np
import pytest

from scripts.sim.bag_align import (
    align,
    interp_state,
    nearest_gap_ns,
    prev_index,
    reorder,
    stamp_ns,
)

MS = 1_000_000


def test_stamp_ns():
    assert stamp_ns(3, 564_000_000) == 3_564_000_000


def test_interp_state_linear_per_joint_and_exact_at_samples():
    t = np.array([0, 10, 20]) * MS
    q = np.array([[0.0, 1.0], [1.0, 1.0], [3.0, -1.0]])
    out, outside = interp_state(np.array([0, 5, 10, 15, 20]) * MS, t, q)
    np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(out[:, 1], [1.0, 1.0, 1.0, 0.0, -1.0])
    assert not outside.any()


def test_interp_state_holds_end_values_outside_and_marks_them():
    t = np.array([10, 20]) * MS
    q = np.array([[1.0], [2.0]])
    out, outside = interp_state(np.array([0, 10, 20, 30]) * MS, t, q)
    np.testing.assert_allclose(out[:, 0], [1.0, 1.0, 2.0, 2.0])
    assert outside.tolist() == [True, False, False, True]


def test_interp_keeps_ns_resolution_at_large_stamps():
    t0 = 1_000_000 * 1_000_000_000  # 1e6 s
    t = np.array([t0, t0 + 10 * MS])
    out, _ = interp_state(np.array([t0 + 1]), t, np.array([[0.0], [10 * MS]]))
    assert out[0, 0] == pytest.approx(1.0)


def test_prev_index_is_at_or_before():
    t_img = np.array([0, 33, 66]) * MS
    q = np.array([-1, 0, 32, 33, 34, 100]) * MS
    assert prev_index(q, t_img).tolist() == [-1, 0, 0, 1, 1, 2]


def test_prev_index_monotone():
    rng = np.random.default_rng(0)
    t_img = np.sort(rng.integers(0, 10**9, 50))
    q = np.sort(rng.integers(0, 10**9, 200))
    idx = prev_index(q, t_img)
    assert np.all(np.diff(idx) >= 0)
    ok = idx >= 0
    assert np.all(t_img[idx[ok]] <= q[ok])
    nxt = idx[ok] + 1
    has_next = nxt < len(t_img)
    assert np.all(t_img[nxt[has_next]] > q[ok][has_next])


def test_nearest_gap():
    t = np.array([0, 10, 20]) * MS
    gap = nearest_gap_ns(np.array([-5, 3, 10, 18, 30]) * MS, t) // MS
    assert gap.tolist() == [5, 3, 0, 2, 10]


def test_align_drops_frames_without_image_and_reports_stats():
    t_act = np.array([0, 33, 66, 100]) * MS
    t_js = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) * MS
    q_js = np.arange(10, dtype=float)[:, None]
    t_img = np.array([30, 60, 90]) * MS
    a = align(t_act, t_js, q_js, t_img)
    assert a.keep.tolist() == [False, True, True, True]
    assert a.image_index.tolist() == [-1, 0, 1, 2]
    assert (a.image_age_ns[a.keep] // MS).tolist() == [3, 6, 10]
    assert a.state_outside.tolist() == [True, False, False, False]
    s = a.stats()
    assert s["frames_kept"] == 3 and s["dropped_no_image"] == 1 and s["dropped_no_state"] == 0
    assert s["image_age_ms_max"] == pytest.approx(10.0)
    assert s["state_held_frames"] == 0  # the held frame was dropped anyway
    assert s["image_reused_frames"] == 0


def test_align_drops_frames_before_the_first_state_by_default():
    t_act = np.array([5, 6, 7, 8]) * MS
    a = align(t_act, np.array([7, 8]) * MS, np.zeros((2, 1)), np.array([0]) * MS)
    assert a.keep.tolist() == [False, False, True, True]
    s = a.stats()
    assert s["dropped_no_state"] == 2 and s["dropped_no_image"] == 0 and s["frames_kept"] == 2
    assert s["state_held_frames"] == 0


def test_align_counts_reused_images_and_held_state():
    t_act = np.array([5, 6, 7]) * MS
    a = align(t_act, np.array([6, 7]) * MS, np.zeros((2, 1)), np.array([0]) * MS, hold_state=True)
    s = a.stats()
    assert s["image_reused_frames"] == 2
    assert s["state_held_frames"] == 1 and s["state_held_ms_max"] == pytest.approx(1.0)


def test_unsorted_or_empty_times_are_rejected():
    with pytest.raises(ValueError):
        prev_index(np.array([0]), np.array([2, 1]))
    with pytest.raises(ValueError):
        interp_state(np.array([0]), np.array([], dtype=np.int64), np.zeros((0, 1)))
    with pytest.raises(ValueError):
        align(np.array([2, 1]), np.array([0, 3]), np.zeros((2, 1)), np.array([0]))


def test_reorder():
    src = ["elbow_flex", "gripper", "shoulder_lift", "shoulder_pan", "wrist_flex", "wrist_roll"]
    dst = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    assert [src[i] for i in reorder(src, dst)] == dst
    with pytest.raises(KeyError):
        reorder(src[:-1], dst)
