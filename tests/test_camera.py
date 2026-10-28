"""jevlite.camera with a fake capture and fake v4l2-ctl; one `hw` test uses the real camera."""

import logging
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from jevlite.camera import (
    Camera,
    CameraConfig,
    CameraError,
    ClockMismatchError,
    V4L2Controls,
    fourcc_to_str,
    load_camera_config,
    ordered_controls,
    redact_device,
)
from jevlite.config import CONFIG_DIR

EXAMPLE = CONFIG_DIR / "camera.example.yaml"
SPEC_CONTROLS = {
    "power_line_frequency": 1,
    "exposure_dynamic_framerate": 0,
    "focus_automatic_continuous": 0,
    "focus_absolute": 30,
    "zoom_absolute": 100,
    "pan_absolute": 0,
    "tilt_absolute": 0,
}
DEV = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_ABCD1234-video-index0"


def cfg(**kw) -> CameraConfig:
    base = dict(
        device=DEV,
        width=640,
        height=480,
        fps=30.0,
        fourcc="MJPG",
        buffersize=2,
        clock_check_frames=3,
        max_clock_offset_s=0.5,
        controls=tuple(SPEC_CONTROLS.items()),
    )
    return CameraConfig(**{**base, **kw})


class FakeCapture:
    """Records set() calls in order; get() returns what was set unless overridden.

    read() delivers a frame every `period` s whose POS_MSEC is `ts_shift` s off the monotonic
    clock (a little in the past, like a real capture). It can also fail, raise, block or start
    drifting after a number of reads."""

    def __init__(self, device, api, opened=True, overrides=None, **behaviour):
        self.device, self.api, self.opened = device, api, opened
        self.calls: list[tuple[int, float]] = []
        self.props: dict[int, float] = {}
        self.overrides = overrides or {}
        self.released = False
        self.period = behaviour.get("period", 0.005)
        self.ts_shift = behaviour.get("ts_shift", 0.0)
        self.no_timestamp = behaviour.get("no_timestamp", False)
        self.fail_after = behaviour.get("fail_after")
        self.raise_after = behaviour.get("raise_after")
        self.drift_after = behaviour.get("drift_after")  # (n_reads, new ts_shift)
        self.block = behaviour.get("block")  # threading.Event: reads wait while it is clear
        self.n_reads = 0
        self.last_ts = 0.0

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.calls.append((prop, value))
        self.props[prop] = value
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return 0.0 if self.no_timestamp else self.last_ts * 1000.0
        return self.overrides.get(prop, self.props.get(prop, 0.0))

    def read(self):
        if self.block is not None:
            self.block.wait(1.0)
        time.sleep(self.period)
        self.n_reads += 1
        if self.raise_after is not None and self.n_reads > self.raise_after:
            raise OSError("USB disconnected")
        if self.fail_after is not None and self.n_reads > self.fail_after:
            return False, None
        if self.drift_after is not None and self.n_reads > self.drift_after[0]:
            self.ts_shift = self.drift_after[1]
        self.last_ts = time.monotonic() - 0.002 + self.ts_shift
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[..., 0] = 255  # blue in BGR
        return True, frame

    def release(self):
        self.released = True


class FakeControls:
    def __init__(self, overrides=None):
        self.set_calls: list[tuple[str, int]] = []
        self.values: dict[str, int] = {}
        self.overrides = overrides or {}

    def set(self, name, value):
        self.set_calls.append((name, value))
        self.values[name] = value

    def get(self, names):
        return {n: self.overrides.get(n, self.values.get(n)) for n in names}


_OPEN: list[Camera] = []


@pytest.fixture(autouse=True)
def _close_cameras():
    yield
    while _OPEN:
        _OPEN.pop().close()


def make(c=None, opened=True, cap_overrides=None, ctl_overrides=None, **behaviour):
    caps = []

    def factory(device, api):
        cap = FakeCapture(device, api, opened, cap_overrides, **behaviour)
        caps.append(cap)
        return cap

    ctl = FakeControls(ctl_overrides)
    cam = Camera(c or cfg(), capture_factory=factory, controls=ctl)
    _OPEN.append(cam)
    return cam, caps, ctl


# --- config ------------------------------------------------------------------------------


def test_example_config_has_the_agreed_values():
    c = load_camera_config(EXAMPLE)
    assert c.device == "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_<SERIAL>-video-index0"
    assert (c.width, c.height, c.fps, c.fourcc, c.buffersize) == (640, 480, 30.0, "MJPG", 2)
    assert (c.clock_check_frames, c.max_clock_offset_s) == (5, 0.5)
    assert dict(c.controls) == SPEC_CONTROLS
    names = dict(c.controls)
    assert not any(n.startswith(("white_balance", "auto_exposure", "exposure_time")) for n in names)


def test_real_config_is_ignored_by_git():
    res = subprocess.run(
        ["git", "check-ignore", "-q", str(CONFIG_DIR / "camera.yaml")],
        cwd=CONFIG_DIR.parent,
    )
    assert res.returncode == 0  # configs/camera.yaml (with the serial) is never committed


def test_missing_config_points_to_the_example(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="camera.example.yaml"):
        load_camera_config(tmp_path / "camera.yaml")


@pytest.mark.parametrize("key", ["buffersize", "clock_check_frames", "max_clock_offset_s"])
def test_keys_are_required(tmp_path: Path, key):
    p = tmp_path / "camera.yaml"
    p.write_text("".join(ln for ln in EXAMPLE.read_text().splitlines(True) if key not in ln))
    with pytest.raises(KeyError):
        load_camera_config(p)


@pytest.mark.parametrize(
    "kw",
    [
        {"fourcc": "MJP"},
        {"width": 0},
        {"fps": -1.0},
        {"buffersize": 0},
        {"clock_check_frames": 0},
        {"max_clock_offset_s": 0.0},
    ],
)
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        cfg(**kw)


def test_redact_device_hides_the_serial():
    assert redact_device(DEV) == (
        "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_<SERIAL>-video-index0"
    )
    assert redact_device("/dev/video0") == "/dev/video0"


# --- open(): order of settings ---------------------------------------------------------


def test_opens_with_v4l2_backend():
    cam, caps, _ = make()
    cam.open()
    assert caps[0].device == DEV
    assert caps[0].api == cv2.CAP_V4L2


def test_fourcc_is_set_before_frame_size_and_buffer_comes_from_config():
    cam, caps, _ = make()
    cam.open()
    props = [p for p, _ in caps[0].calls]
    assert props.index(cv2.CAP_PROP_FOURCC) < props.index(cv2.CAP_PROP_FRAME_WIDTH)
    assert props.index(cv2.CAP_PROP_FOURCC) < props.index(cv2.CAP_PROP_FRAME_HEIGHT)
    assert dict(caps[0].calls)[cv2.CAP_PROP_BUFFERSIZE] == 2
    cam1, caps1, _ = make(cfg(buffersize=1))
    cam1.open()
    assert dict(caps1[0].calls)[cv2.CAP_PROP_BUFFERSIZE] == 1
    assert fourcc_to_str(dict(caps[0].calls)[cv2.CAP_PROP_FOURCC]) == "MJPG"


def test_autofocus_is_switched_off_before_focus_absolute():
    # even if the YAML lists focus_absolute first
    c = cfg(controls=(("focus_absolute", 20), ("zoom_absolute", 100),
                      ("focus_automatic_continuous", 0)))  # fmt: skip
    cam, _, ctl = make(c)
    cam.open()
    names = [n for n, _ in ctl.set_calls]
    assert names.index("focus_automatic_continuous") < names.index("focus_absolute")
    assert names == ["focus_automatic_continuous", "focus_absolute", "zoom_absolute"]


def test_ordered_controls_keeps_yaml_order_for_the_rest():
    out = ordered_controls(list(SPEC_CONTROLS.items()))
    assert out[:2] == [("focus_automatic_continuous", 0), ("exposure_dynamic_framerate", 0)]
    assert [n for n, _ in out[2:]] == [
        "power_line_frequency", "focus_absolute", "zoom_absolute", "pan_absolute", "tilt_absolute"
    ]  # fmt: skip


def test_controls_are_applied_on_every_open():
    cam, caps, ctl = make()
    cam.open()
    cam.close()
    cam.open()
    assert len(ctl.set_calls) == 2 * len(SPEC_CONTROLS)
    assert caps[0].released and len(caps) == 2


def test_open_failure_raises_without_the_serial():
    cam, _, _ = make(opened=False)
    with pytest.raises(RuntimeError) as e:
        cam.open()
    assert "ABCD1234" not in str(e.value) and "<SERIAL>" in str(e.value)


# --- open(): read-back and warnings ----------------------------------------------------


def test_no_warning_when_everything_matches(caplog):
    cam, _, _ = make()
    with caplog.at_level(logging.WARNING, logger="jevlite.camera"):
        cam.open()
    assert cam.mismatches == []
    assert caplog.records == []
    assert cam.readback["fourcc"] == "MJPG" and cam.readback["controls"] == SPEC_CONTROLS


@pytest.mark.parametrize(
    ("cap_overrides", "ctl_overrides", "expect"),
    [
        ({cv2.CAP_PROP_FRAME_WIDTH: 1280}, None, "width is 1280"),
        ({cv2.CAP_PROP_FRAME_HEIGHT: 720}, None, "height is 720"),
        ({cv2.CAP_PROP_FPS: 15.0}, None, "fps is 15.0"),
        ({cv2.CAP_PROP_FOURCC: cv2.VideoWriter.fourcc(*"YUYV")}, None, "fourcc is 'YUYV'"),
        ({cv2.CAP_PROP_BUFFERSIZE: 4}, None, "buffersize is 4, expected 2"),
        (None, {"focus_absolute": 80}, "control focus_absolute is 80, expected 30"),
        (None, {"focus_automatic_continuous": 1}, "control focus_automatic_continuous is 1"),
    ],
)
def test_mismatch_is_warned(caplog, cap_overrides, ctl_overrides, expect):
    cam, _, _ = make(cap_overrides=cap_overrides, ctl_overrides=ctl_overrides)
    with caplog.at_level(logging.WARNING, logger="jevlite.camera"):
        cam.open()
    assert any(expect in m for m in cam.mismatches)
    assert any(expect in r.getMessage() for r in caplog.records)
    assert all("ABCD1234" not in r.getMessage() for r in caplog.records)


def test_fps_within_tolerance_is_not_a_mismatch():
    cam, _, _ = make(cap_overrides={cv2.CAP_PROP_FPS: 30.3})
    cam.open()
    assert cam.mismatches == []


# --- reader thread: latest frame, seq, waiting --------------------------------------


def test_read_returns_latest_frame_with_capture_time_and_seq():
    cam, _, _ = make()
    cam.open()
    f = cam.read()
    assert f.rgb.shape == (48, 64, 3) and f.rgb.dtype == np.uint8
    assert f.rgb[0, 0].tolist() == [0, 0, 255]  # BGR blue -> RGB blue
    assert f.seq >= 1
    assert 0.0 <= time.monotonic() - f.t_capture < 0.1  # capture time, not read-out time


def test_slow_consumer_gets_the_newest_frame_not_a_queued_one():
    cam, _, _ = make(period=0.005)
    cam.open()
    first = cam.read()
    time.sleep(0.1)  # a slow judge
    f = cam.read()
    assert f.seq > first.seq + 5  # skipped the frames it did not need
    assert time.monotonic() - f.t_capture < 0.03  # still recent


def test_wait_new_returns_a_newer_frame():
    cam, _, _ = make()
    cam.open()
    f1 = cam.read()
    f2 = cam.wait_new(f1.seq, timeout=1.0)
    assert f2.seq > f1.seq and f2.t_capture > f1.t_capture


def test_wait_new_times_out_when_no_new_frame_arrives():
    gate = threading.Event()
    gate.set()
    cam, _, _ = make(block=gate)
    cam.open()
    f = cam.read()
    gate.clear()  # camera stalls (reads now wait)
    time.sleep(0.05)
    latest = cam.read()  # returns the last frame without waiting...
    with pytest.raises(TimeoutError):
        cam.wait_new(latest.seq, timeout=0.1)  # ...but waiting for a new one times out
    assert latest.seq >= f.seq
    gate.set()


def test_reader_exception_is_reported_not_hidden():
    cam, _, _ = make(raise_after=10)  # 3 reads for the clock check, then a few frames
    cam.open()
    time.sleep(0.2)
    with pytest.raises(CameraError) as e:
        cam.read()
    assert isinstance(e.value.__cause__, OSError)
    with pytest.raises(CameraError):
        cam.wait_new(0, timeout=0.1)


def test_failed_read_stops_the_reader_and_is_reported():
    cam, _, _ = make(fail_after=8)
    cam.open()
    time.sleep(0.2)
    with pytest.raises(CameraError, match="failed to read"):
        cam.read()


def test_close_stops_the_thread_and_releases():
    cam, caps, _ = make()
    cam.open()
    thread = cam._thread
    assert thread.is_alive()
    cam.close()
    assert not thread.is_alive()
    assert caps[0].released
    with pytest.raises(CameraError):
        cam.read()
    cam.close()  # idempotent


def test_read_before_open_raises():
    cam, _, _ = make()
    with pytest.raises(CameraError):
        cam.read()


def test_context_manager_closes():
    cam, caps, _ = make()
    with cam:
        cam.read()
    assert caps[0].released


def test_measure_fps_uses_capture_timestamps():
    cam, _, _ = make(period=0.01)
    cam.open()
    assert 50 < cam.measure_fps(n_frames=10) < 110  # ~1 / 0.01 s


# --- clock: capture time must be on the time.monotonic() clock ------------------------


def test_clock_mismatch_at_open_is_an_error_and_nothing_starts():
    cam, caps, _ = make(ts_shift=-1.0)  # timestamps 1 s behind time.monotonic()
    with pytest.raises(ClockMismatchError, match="uvcvideo"):
        cam.open()
    assert caps[0].released
    assert cam._thread is None


def test_capture_time_in_the_future_at_open_is_an_error():
    cam, _, _ = make(ts_shift=+1.0)
    with pytest.raises(ClockMismatchError):
        cam.open()


def test_missing_timestamp_is_an_error():
    cam, _, _ = make(no_timestamp=True)  # POS_MSEC = 0: not a monotonic timestamp
    with pytest.raises(ClockMismatchError):
        cam.open()


def test_clock_offsets_are_recorded():
    cam, _, _ = make()
    cam.open()
    assert len(cam.clock_offsets) == 3
    assert all(0 <= o < 0.05 for o in cam.clock_offsets)


def test_runtime_drift_only_warns_and_keeps_the_capture_time(caplog):
    """No fallback to the read-out time: it would make frames look fresher than they are."""
    cam, _, _ = make(drift_after=(6, -2.0))  # after a few frames timestamps lag by 2 s
    with caplog.at_level(logging.WARNING, logger="jevlite.camera"):
        cam.open()
        time.sleep(0.1)
        f = cam.read()
    assert time.monotonic() - f.t_capture > 1.5  # still the (old) capture time -> G1 stops
    assert cam.clock_warnings > 0
    assert any("G1" in r.getMessage() for r in caplog.records)


# --- v4l2-ctl wrapper ---------------------------------------------------------------------


def test_v4l2ctl_commands_and_parsing():
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        out = "focus_absolute: 20\nfocus_automatic_continuous: 0\npan_absolute: -3600\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out if "-C" in cmd else "")

    ctl = V4L2Controls(DEV, run=run)
    ctl.set("focus_absolute", 20)
    got = ctl.get(["focus_absolute", "focus_automatic_continuous", "pan_absolute"])
    assert calls[0] == ["v4l2-ctl", "-d", DEV, "-c", "focus_absolute=20"]
    assert calls[1][3:] == ["-C", "focus_absolute,focus_automatic_continuous,pan_absolute"]
    assert got == {"focus_absolute": 20, "focus_automatic_continuous": 0, "pan_absolute": -3600}


# --- real camera (excluded by default; run with -m hw) -----------------------------------


@pytest.mark.hw
def test_real_camera_opens_with_the_configured_settings():
    with Camera(load_camera_config()) as cam:
        f = cam.read()
        assert cam.mismatches == []
        assert f.rgb.shape == (480, 640, 3)
        assert all(abs(o) < 0.5 for o in cam.clock_offsets)
        time.sleep(0.1)
        assert time.monotonic() - cam.read().t_capture < 0.1  # latest frame, not a queued one
