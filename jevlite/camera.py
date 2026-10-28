"""USB UVC camera input (OpenCV V4L2 backend + v4l2-ctl for controls).

Settings come from configs/camera.yaml (template: configs/camera.example.yaml). v4l2 controls
reset when the camera is unplugged or the PC restarts, so they are applied on every open().
After opening, format and controls are read back; mismatches are logged as warnings and kept
in `Camera.mismatches`.

A reader thread (same process) reads the camera continuously and keeps only the latest frame,
so a slow judge always gets a recent image instead of a queued, stale one. Each frame carries
its capture time (the V4L2 buffer timestamp, CLOCK_MONOTONIC = the clock of time.monotonic())
and a sequence number so the same frame is not judged twice.

    with Camera(load_camera_config()) as cam:
        f = cam.read()                  # latest Frame (rgb, t_capture, seq)
        f = cam.wait_new(f.seq, 0.5)    # wait until a newer frame arrives

If the reader thread stops or fails, read()/wait_new() raise CameraError instead of returning
the last frame forever. Runtime clock drift is only warned about; the gate's G1 (stale/future)
turns it into STOP.

Quick check (no window, no file written; prints the read-back, fps and frame age):

    uv run python -m jevlite.camera
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from jevlite.config import CONFIG_DIR

log = logging.getLogger(__name__)

# Automatic modes must be switched off before their manual values are written, otherwise
# the camera ignores the manual value (e.g. focus_absolute while autofocus is on).
AUTO_FIRST = (
    "focus_automatic_continuous",
    "white_balance_automatic",
    "auto_exposure",
    "exposure_dynamic_framerate",
)
FPS_TOLERANCE = 0.5  # frames/s; CAP_PROP_FPS may report e.g. 30.0 for a 30 fps request


@dataclass(frozen=True)
class CameraConfig:
    device: str
    width: int
    height: int
    fps: float
    fourcc: str
    buffersize: int  # driver buffers; 1 = freshest frame but drops to ~18 fps on the C920
    clock_check_frames: int  # frames compared with time.monotonic() at open()
    max_clock_offset_s: float  # |time.monotonic() - capture time| above this is an error/warning
    controls: tuple[tuple[str, int], ...]  # in the order written in the YAML

    def __post_init__(self):
        if len(self.fourcc) != 4:
            raise ValueError(f"fourcc must be 4 characters, got {self.fourcc!r}")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width, height and fps must be positive")
        if self.buffersize < 1:
            raise ValueError(f"buffersize must be >= 1, got {self.buffersize}")
        if self.clock_check_frames < 1 or not self.max_clock_offset_s > 0:
            raise ValueError("clock_check_frames must be >= 1 and max_clock_offset_s > 0")


def load_camera_config(path: Path | str | None = None) -> CameraConfig:
    path = Path(path) if path else CONFIG_DIR / "camera.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy configs/camera.example.yaml to configs/camera.yaml and "
            "replace <SERIAL> (see `ls /dev/v4l/by-id/`)."
        )
    raw = yaml.safe_load(path.read_text()) or {}
    controls = raw.get("controls") or {}
    return CameraConfig(
        device=str(raw["device"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        fps=float(raw["fps"]),
        fourcc=str(raw["fourcc"]),
        buffersize=int(raw["buffersize"]),
        clock_check_frames=int(raw["clock_check_frames"]),
        max_clock_offset_s=float(raw["max_clock_offset_s"]),
        controls=tuple((str(k), int(v)) for k, v in controls.items()),
    )


def redact_device(device: str) -> str:
    """Hide the serial number in a /dev/v4l/by-id name (for logs and reports)."""
    return re.sub(r"(usb-[0-9a-f]{4}_[^/]*?_)[^_/-]+(-video-index\d+)$", r"\1<SERIAL>\2", device)


def ordered_controls(controls: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
    """Automatic switches first (AUTO_FIRST order), then the rest in their given order."""
    controls = list(controls)
    auto = sorted((c for c in controls if c[0] in AUTO_FIRST), key=lambda c: AUTO_FIRST.index(c[0]))
    return auto + [c for c in controls if c[0] not in AUTO_FIRST]


def fourcc_to_str(code: float) -> str:
    code = int(code)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


class CameraError(RuntimeError):
    """The camera cannot deliver frames (not open, reader thread stopped or failed)."""


class ClockMismatchError(CameraError):
    """Capture timestamps are not on the time.monotonic() clock (checked at open())."""


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray  # HxWx3 uint8
    t_capture: float  # capture time [s], same clock as time.monotonic() (V4L2 buffer timestamp)
    seq: int  # 1, 2, 3, ... per open(); a judge can skip frames it has already seen


class V4L2Controls:
    """Thin wrapper around `v4l2-ctl` (no udev or permission changes; needs the video group)."""

    def __init__(self, device: str, run: Callable[..., Any] = subprocess.run):
        self.device = device
        self._run = run

    def _ctl(self, *args: str) -> str:
        res = self._run(
            ["v4l2-ctl", "-d", self.device, *args], capture_output=True, text=True, check=True
        )
        return res.stdout

    def set(self, name: str, value: int) -> None:
        self._ctl("-c", f"{name}={value}")

    def get(self, names: Iterable[str]) -> dict[str, int]:
        out = self._ctl("-C", ",".join(names))
        values = {}
        for line in out.splitlines():
            m = re.match(r"\s*(\w+):\s*(-?\d+)", line)
            if m:
                values[m.group(1)] = int(m.group(2))
        return values


class Camera:
    def __init__(
        self,
        cfg: CameraConfig,
        capture_factory: Callable[[str, int], Any] | None = None,
        controls: V4L2Controls | None = None,
    ):
        self.cfg = cfg
        self._factory = capture_factory or cv2.VideoCapture
        self._controls = controls or V4L2Controls(cfg.device)
        self._cap = None
        self.mismatches: list[str] = []
        self.readback: dict[str, Any] = {}
        self.clock_offsets: list[float] = []  # time.monotonic() - t_capture at open()
        self.clock_warnings = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._error: BaseException | None = None

    def open(self) -> Camera:
        cap = self._factory(self.cfg.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise CameraError(f"cannot open camera {redact_device(self.cfg.device)}")
        # FOURCC before the frame size: the driver picks sizes per pixel format.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self.cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cfg.buffersize)  # few buffers: no stale queue
        self._cap = cap
        try:
            for name, value in ordered_controls(self.cfg.controls):
                self._controls.set(name, value)
            self.mismatches = self._verify()
            for msg in self.mismatches:
                log.warning("camera %s: %s", redact_device(self.cfg.device), msg)
            self._check_clock()
        except BaseException:
            self._release()
            raise
        self._start_reader()
        return self

    def _grab(self) -> tuple[np.ndarray, float]:
        """One blocking read: (BGR frame, capture time in seconds)."""
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise CameraError("failed to read a frame")
        return bgr, self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def _check_clock(self) -> None:
        """Refuse to start if capture times are not on the time.monotonic() clock.

        There is deliberately no fallback to the read-out time: it is later than the real
        capture time, so frames would look fresher than they are (the unsafe direction)."""
        self.clock_offsets = []
        for _ in range(self.cfg.clock_check_frames):
            _, t_capture = self._grab()
            self.clock_offsets.append(time.monotonic() - t_capture)
        worst = max(self.clock_offsets, key=abs)
        if abs(worst) > self.cfg.max_clock_offset_s:
            raise ClockMismatchError(
                f"capture timestamps are {worst:+.3f} s from time.monotonic() "
                f"(limit {self.cfg.max_clock_offset_s} s); is /sys/module/uvcvideo/parameters/"
                "clock CLOCK_MONOTONIC?"
            )

    def _start_reader(self) -> None:
        self._stop.clear()
        with self._cond:
            self._latest, self._error = None, None
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        seq = 0
        try:
            while not self._stop.is_set():
                bgr, t_capture = self._grab()
                offset = time.monotonic() - t_capture
                if abs(offset) > self.cfg.max_clock_offset_s:
                    self.clock_warnings += 1
                    if self.clock_warnings == 1 or self.clock_warnings % 300 == 0:
                        log.warning(
                            "camera capture time is %+.3f s from time.monotonic() (%d frames so "
                            "far); the gate's G1 will treat such frames as stale/future",
                            offset,
                            self.clock_warnings,
                        )
                seq += 1
                frame = Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_capture, seq)
                with self._cond:
                    self._latest = frame
                    self._cond.notify_all()
        except BaseException as e:  # reported to the caller by read()/wait_new()
            with self._cond:
                self._error = e
                self._cond.notify_all()
            log.error("camera reader stopped: %s", e)

    def _check_alive(self) -> None:
        """Raise unless the reader is running. Call with self._cond held."""
        if self._error is not None:
            raise CameraError(f"camera reader failed: {self._error}") from self._error
        if self._thread is None or not self._thread.is_alive():
            raise CameraError("camera is not open (reader thread is not running)")

    def read(self, timeout: float = 1.0) -> Frame:
        """Latest frame (waits up to `timeout` for the first one after open())."""
        return self.wait_new(0, timeout)

    def wait_new(self, after_seq: int, timeout: float = 1.0) -> Frame:
        """Wait until a frame newer than `after_seq` is available and return the latest one.

        Raises TimeoutError if none arrives within `timeout`, CameraError if the reader is
        not running or has failed."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                self._check_alive()  # first: never hand out the last frame of a dead reader
                if self._latest is not None and self._latest.seq > after_seq:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no camera frame newer than seq {after_seq} in {timeout} s")
                self._cond.wait(remaining)

    def measure_fps(self, n_frames: int = 60) -> float:
        """Frames per second delivered by the reader over n_frames new frames."""
        f = self.wait_new(0)
        t0, seq0 = f.t_capture, f.seq
        while f.seq < seq0 + n_frames:
            f = self.wait_new(f.seq)
        return (f.seq - seq0) / (f.t_capture - t0)

    def close(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(join_timeout)
            if self._thread.is_alive():
                log.warning("camera reader did not stop within %.1f s", join_timeout)
            self._thread = None
        self._release()

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def _verify(self) -> list[str]:
        cap = self._cap
        rb: dict[str, Any] = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            "buffersize": int(cap.get(cv2.CAP_PROP_BUFFERSIZE)),
        }
        wanted = dict(self.cfg.controls)
        rb["controls"] = self._controls.get(wanted) if wanted else {}
        self.readback = rb
        problems = []
        for key in ("width", "height"):
            if rb[key] != getattr(self.cfg, key):
                problems.append(f"{key} is {rb[key]}, expected {getattr(self.cfg, key)}")
        if abs(rb["fps"] - self.cfg.fps) > FPS_TOLERANCE:
            problems.append(f"fps is {rb['fps']}, expected {self.cfg.fps}")
        if rb["fourcc"] != self.cfg.fourcc:
            problems.append(f"fourcc is {rb['fourcc']!r}, expected {self.cfg.fourcc!r}")
        if rb["buffersize"] != self.cfg.buffersize:
            problems.append(f"buffersize is {rb['buffersize']}, expected {self.cfg.buffersize}")
        for name, value in wanted.items():
            got = rb["controls"].get(name)
            if got != value:
                problems.append(f"control {name} is {got}, expected {value}")
        return problems


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_camera_config()
    with Camera(cfg) as cam:
        frame = cam.read()
        fps = cam.measure_fps()
        ages = []
        for _ in range(30):  # a slow reader: 100 ms between reads
            time.sleep(0.1)
            f = cam.read()
            ages.append(time.monotonic() - f.t_capture)
        print(f"device   {redact_device(cfg.device)}")
        print(f"readback {cam.readback}")
        print(f"clock    offsets at open {[round(o, 4) for o in cam.clock_offsets]} s")
        print(f"frame    {frame.rgb.shape} {frame.rgb.dtype} seq={frame.seq}")
        print(f"fps      {fps:.1f} (capture timestamps over 60 frames)")
        print(f"age      100 ms reader: mean {1000 * np.mean(ages):.1f} ms, "
              f"max {1000 * np.max(ages):.1f} ms")  # fmt: skip
        print("mismatches: " + ("none" if not cam.mismatches else "; ".join(cam.mismatches)))
    return 1 if cam.mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
