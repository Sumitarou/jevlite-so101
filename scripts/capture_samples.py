"""Capture labelled camera images for training / evaluation (run it in your own terminal).

    uv run python scripts/capture_samples.py --label none [--session NAME] [--preview]
    uv run python scripts/capture_samples.py --label edge --burst 50 --interval 0.5 --preview

Images go to samples/private/<session>/<label>/<NNNNN>.jpg and one row per image is appended
to samples/private/<session>/metadata.csv (file, label, teacher choice, seq, t_capture, frame
age, camera settings; never the camera serial). The output root must be git-ignored, because
images may show people and rooms. No window is opened: with --preview the last saved image is
also written to outputs/preview_latest.jpg (keep it open in VS Code).

Modes:
  default   press Enter to save one image, q (or Ctrl+D) to quit
  --burst N --interval S   save N images, one every S seconds (Ctrl+C stops early)

Labels (configs/capture.yaml) correspond to the teacher's 3 choices:
  none = A (なし), edge = B (端にある), center = C (中央にある)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from jevlite.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_COLUMNS = [
    "file",
    "label",
    "teacher_choice",
    "seq",
    "t_capture",
    "frame_age_ms",
    "saved_at",
    "width",
    "height",
    "fourcc",
    "fps",
    "buffersize",
    "controls",
]


def is_git_ignored(path: Path, repo_root: Path = REPO_ROOT) -> bool:
    res = subprocess.run(
        ["git", "check-ignore", "-q", str(path)], cwd=repo_root, capture_output=True
    )
    return res.returncode == 0


def camera_summary(readback: dict[str, Any]) -> dict[str, Any]:
    """Settings actually applied (from Camera.readback). No device path, so no serial."""
    controls = readback.get("controls", {})
    return {
        "width": readback.get("width"),
        "height": readback.get("height"),
        "fourcc": readback.get("fourcc"),
        "fps": readback.get("fps"),
        "buffersize": readback.get("buffersize"),
        "controls": ";".join(f"{k}={v}" for k, v in controls.items()),
    }


def next_index(label_dir: Path, digits: int) -> int:
    """1 + the largest existing NNNNN.jpg, so a session can be continued later."""
    pattern = re.compile(rf"^(\d{{{digits}}})\.jpg$")
    numbers = [int(m[1]) for p in label_dir.glob("*.jpg") if (m := pattern.match(p.name))]
    return max(numbers, default=0) + 1


class SampleWriter:
    def __init__(
        self,
        session_dir: Path,
        label: str,
        cfg: dict[str, Any],
        camera: dict[str, Any],
        preview_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if label not in cfg["labels"]:
            raise ValueError(f"unknown label {label!r}; choose from {list(cfg['labels'])}")
        self.session_dir = session_dir
        self.label = label
        self.teacher = cfg["labels"][label]["teacher"]
        self.quality = int(cfg["jpeg_quality"])
        self.digits = int(cfg["filename_digits"])
        self.camera = camera
        self.preview_path = preview_path
        self.clock = clock
        self.label_dir = session_dir / label
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = session_dir / "metadata.csv"
        self.index = next_index(self.label_dir, self.digits)
        self.saved: list[Path] = []

    def save(self, frame) -> Path:
        path = self.label_dir / f"{self.index:0{self.digits}d}.jpg"
        if path.exists():  # never overwrite an existing sample
            raise FileExistsError(path)
        age_ms = (self.clock() - frame.t_capture) * 1000.0
        Image.fromarray(frame.rgb).save(path, quality=self.quality)
        self._append_row(
            {
                "file": path.relative_to(self.session_dir).as_posix(),
                "label": self.label,
                "teacher_choice": self.teacher,
                "seq": frame.seq,
                "t_capture": f"{frame.t_capture:.6f}",
                "frame_age_ms": f"{age_ms:.1f}",
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                **self.camera,
            }
        )
        if self.preview_path is not None:
            self.preview_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.preview_path.with_suffix(".tmp.jpg")
            Image.fromarray(frame.rgb).save(tmp, quality=self.quality)
            os.replace(tmp, self.preview_path)  # atomic: the viewer never sees half a file
        self.index += 1
        self.saved.append(path)
        return path

    def _append_row(self, row: dict[str, Any]) -> None:
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)


def run_interactive(cam, writer: SampleWriter, timeout: float, input_fn=input, out=print) -> int:
    """Enter saves one new frame, q quits. Each save waits for a frame not saved before."""
    last_seq = 0
    while True:
        try:
            key = input_fn(f"[{writer.label}] Enter = save, q = quit > ").strip().lower()
        except EOFError:
            break
        if key == "q":
            break
        if key:
            out(f"unknown key {key!r} (Enter = save, q = quit)")
            continue
        frame = cam.wait_new(last_seq, timeout)
        last_seq = frame.seq
        out(f"saved {writer.save(frame)} (seq {frame.seq})")
    return len(writer.saved)


def run_burst(
    cam,
    writer: SampleWriter,
    n: int,
    interval: float,
    timeout: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    out=print,
) -> int:
    """Save n new frames, one every `interval` seconds (on a fixed schedule, no drift)."""
    last_seq = 0
    t0 = clock()
    try:
        for i in range(n):
            sleep(max(0.0, t0 + i * interval - clock()))
            frame = cam.wait_new(last_seq, timeout)
            last_seq = frame.seq
            path = writer.save(frame)
            out(f"[{i + 1}/{n}] saved {path} (seq {frame.seq})")
    except KeyboardInterrupt:
        out("stopped by Ctrl+C")
    return len(writer.saved)


def build_parser(labels: list[str]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, choices=labels)
    ap.add_argument("--session", help="session name (default: date-time, e.g. 20261010-2240)")
    ap.add_argument("--burst", type=int, metavar="N", help="save N images automatically")
    ap.add_argument("--interval", type=float, metavar="S", help="seconds between burst images")
    ap.add_argument("--preview", action="store_true", help="also write outputs/preview_latest.jpg")
    return ap


def main(
    argv: list[str] | None = None,
    camera_factory: Callable | None = None,
    input_fn=input,
    repo_root: Path = REPO_ROOT,
    check_ignored: Callable[[Path], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    out=print,
) -> int:
    cfg = load_config("capture.yaml")
    args = build_parser(list(cfg["labels"])).parse_args(argv)
    if (args.burst is None) != (args.interval is None):
        out("error: --burst and --interval must be given together")
        return 2
    if args.burst is not None and (args.burst < 1 or args.interval < 0):
        out("error: --burst must be >= 1 and --interval >= 0")
        return 2
    session = args.session
    if session is None:
        session = datetime.now().strftime(cfg["session_format"])
    if not re.fullmatch(r"[A-Za-z0-9._-]+", session):
        out(f"error: session name {session!r} may only contain letters, digits, . _ -")
        return 2
    out_root = repo_root / cfg["out_root"]
    check_ignored = check_ignored or (lambda p: is_git_ignored(p, repo_root))
    if not check_ignored(out_root / session):
        out(f"error: {cfg['out_root']} is not git-ignored; refusing to save camera images there")
        return 2
    preview = repo_root / cfg["preview_path"] if args.preview else None
    if preview is not None and not check_ignored(preview):
        out(f"error: {cfg['preview_path']} is not git-ignored")
        return 2

    if camera_factory is None:
        from jevlite.camera import Camera, load_camera_config

        def camera_factory():
            return Camera(load_camera_config())

    timeout = float(cfg["frame_timeout_s"])
    with camera_factory() as cam:
        writer = SampleWriter(
            out_root / session, args.label, cfg, camera_summary(cam.readback), preview, clock
        )
        out(f"session {session}, label {args.label} ({writer.teacher}: "
            f"{cfg['labels'][args.label]['ja']}) -> {writer.label_dir.relative_to(repo_root)}, "
            f"next {writer.index:0{writer.digits}d}.jpg")  # fmt: skip
        if args.burst is not None:
            n = run_burst(cam, writer, args.burst, args.interval, timeout, sleep, clock, out)
        else:
            n = run_interactive(cam, writer, timeout, input_fn, out)
    out(f"{n} image(s) saved; metadata: {writer.csv_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
