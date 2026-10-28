import time

import numpy as np
import pytest

from jevlite.api import (
    Choice,
    Decision,
    Judge,
    decision_from_probs,
    judge_frame,
    start_judging,
)


def test_choice_order_is_by_severity():
    assert Choice.CONTINUE < Choice.SLOW < Choice.STOP


def test_choice_parse():
    assert Choice.parse(" slow ") is Choice.SLOW
    with pytest.raises(KeyError):
        Choice.parse("FAST")


def test_decision_from_probs_picks_argmax():
    d = decision_from_probs({"CONTINUE": 0.1, "SLOW": 0.7, "STOP": 0.2}, "v0", start_judging())
    assert d.choice is Choice.SLOW
    assert d.source == "v0"
    assert d.latency_ms >= 0.0
    assert d.confidence == pytest.approx(0.7)


def test_decision_tie_breaks_toward_safe_side():
    d = decision_from_probs({"CONTINUE": 0.4, "SLOW": 0.2, "STOP": 0.4}, "v0", start_judging())
    assert d.choice is Choice.STOP


@pytest.mark.parametrize(
    "probs",
    [
        {"CONTINUE": 0.5, "SLOW": 0.5},
        {"CONTINUE": 0.5, "SLOW": 0.5, "STOP": 0.5},
    ],
)
def test_decision_from_probs_rejects_bad_probs(probs):
    with pytest.raises(ValueError):
        decision_from_probs(probs, "v0", start_judging())


def test_decision_is_immutable():
    d = Decision(Choice.STOP, {"CONTINUE": 0, "SLOW": 0, "STOP": 1}, 0.0, 1.0, "x")
    with pytest.raises(AttributeError):
        d.choice = Choice.CONTINUE  # type: ignore[misc]


def test_judge_protocol():
    class Dummy:
        def judge(self, image, t_capture=None):
            return decision_from_probs(
                {"CONTINUE": 1.0, "SLOW": 0.0, "STOP": 0.0}, "dummy", start_judging(t_capture)
            )

    assert isinstance(Dummy(), Judge)


# --- t_mono: capture time if given, else the time judging started ---------------------

PROBS = {"CONTINUE": 1.0, "SLOW": 0.0, "STOP": 0.0}


def test_start_judging_uses_t_capture_when_given():
    assert start_judging(123.25).t_mono == 123.25


def test_start_judging_without_t_capture_is_the_start_time():
    before = time.monotonic()
    start = start_judging()
    time.sleep(0.05)  # judging takes a while
    d = decision_from_probs(PROBS, "x", start)
    after = time.monotonic()
    assert before <= d.t_mono <= before + 0.01  # start, not finish
    assert after - d.t_mono >= 0.05


def test_latency_ms_is_the_judging_time_not_the_image_age():
    start = start_judging(time.monotonic() - 10.0)  # a 10 s old frame
    time.sleep(0.02)
    d = decision_from_probs(PROBS, "x", start)
    assert 20.0 <= d.latency_ms < 200.0  # only the judging time


def test_judge_frame_passes_rgb_and_capture_time():
    seen = {}

    class Recorder:
        def judge(self, image, t_capture=None):
            seen["image"], seen["t_capture"] = image, t_capture
            return decision_from_probs(PROBS, "rec", start_judging(t_capture))

    class F:
        rgb = np.zeros((2, 2, 3), np.uint8)
        t_capture = 42.5

    d = judge_frame(Recorder(), F())
    assert seen["image"] is F.rgb and seen["t_capture"] == 42.5
    assert d.t_mono == 42.5
