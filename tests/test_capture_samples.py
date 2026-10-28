"""scripts/capture_samples.py with a fake camera, scripted key input and a fake clock."""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import capture_samples as cap

SERIAL = "ABCD1234"


class FakeFrame:
    def __init__(self, seq, t_capture):
        self.seq, self.t_capture = seq, t_capture
        self.rgb = np.full((48, 64, 3), seq % 256, np.uint8)


class FakeCamera:
    """Each wait_new() returns a frame newer than after_seq; records the calls."""

    def __init__(self, clock):
        self.clock = clock
        self.seq = 0
        self.wait_calls: list[int] = []
        self.readback = {
            "width": 640, "height": 480, "fps": 30.0, "fourcc": "MJPG", "buffersize": 2,
            "controls": {"focus_automatic_continuous": 0, "focus_absolute": 20},
        }  # fmt: skip
        self.device = f"/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_{SERIAL}-video-index0"
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def wait_new(self, after_seq, timeout=1.0):
        self.wait_calls.append(after_seq)
        self.seq = max(self.seq, after_seq) + 1
        return FakeFrame(self.seq, self.clock() - 0.04)


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


@pytest.fixture
def env(tmp_path: Path):
    clock = FakeClock()
    camera = FakeCamera(clock)
    messages: list[str] = []

    def run(argv, keys=(), ignored=True):
        keys = list(keys)

        def input_fn(prompt):
            if not keys:
                raise EOFError
            return keys.pop(0)

        return cap.main(
            argv,
            camera_factory=lambda: camera,
            input_fn=input_fn,
            repo_root=tmp_path,
            check_ignored=lambda p: ignored,
            sleep=clock.sleep,
            clock=clock,
            out=messages.append,
        )

    return {"run": run, "root": tmp_path, "camera": camera, "clock": clock, "msgs": messages}


def rows(session_dir: Path):
    with (session_dir / "metadata.csv").open() as f:
        return list(csv.DictReader(f))


# --- interactive -------------------------------------------------------------------------


def test_enter_saves_one_image_each_and_q_quits(env):
    rc = env["run"](["--label", "none", "--session", "s1"], keys=["", "", "x", "", "q", ""])
    assert rc == 0
    d = env["root"] / "samples" / "private" / "s1"
    assert sorted(p.name for p in (d / "none").iterdir()) == ["00001.jpg", "00002.jpg", "00003.jpg"]
    assert any("unknown key 'x'" in m for m in env["msgs"])
    assert env["camera"].closed


def test_each_save_is_a_new_frame(env):
    env["run"](["--label", "edge", "--session", "s1"], keys=["", "", ""])
    seqs = [int(r["seq"]) for r in rows(env["root"] / "samples" / "private" / "s1")]
    assert seqs == sorted(set(seqs))  # strictly increasing, no frame saved twice
    assert env["camera"].wait_calls == [0, seqs[0], seqs[1]]


def test_ctrl_d_ends_the_session(env):
    assert env["run"](["--label", "none", "--session", "s1"], keys=[""]) == 0  # then EOF


# --- burst -------------------------------------------------------------------------------


def test_burst_saves_n_images_on_a_fixed_schedule(env):
    rc = env["run"](["--label", "center", "--session", "b", "--burst", "4", "--interval", "0.5"])
    assert rc == 0
    d = env["root"] / "samples" / "private" / "b" / "center"
    assert len(list(d.glob("*.jpg"))) == 4
    assert env["clock"].sleeps == [0.0, 0.5, 0.5, 0.5]


def test_burst_needs_interval(env):
    assert env["run"](["--label", "none", "--burst", "3"]) == 2
    assert any("--burst and --interval" in m for m in env["msgs"])


def test_ctrl_c_stops_a_burst_and_keeps_what_was_saved(env):
    calls = {"n": 0}
    real = env["camera"].wait_new

    def wait_new(after_seq, timeout=1.0):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(after_seq, timeout)

    env["camera"].wait_new = wait_new
    env["run"](["--label", "none", "--session", "b", "--burst", "10", "--interval", "0.1"])
    assert len(rows(env["root"] / "samples" / "private" / "b")) == 2
    assert any("Ctrl+C" in m for m in env["msgs"])


# --- files and metadata -----------------------------------------------------------------


def test_layout_and_metadata_csv(env):
    env["run"](["--label", "edge", "--session", "s1"], keys=["", ""])
    d = env["root"] / "samples" / "private" / "s1"
    r = rows(d)
    assert list(r[0]) == cap.CSV_COLUMNS
    assert [x["file"] for x in r] == ["edge/00001.jpg", "edge/00002.jpg"]
    assert {x["label"] for x in r} == {"edge"} and {x["teacher_choice"] for x in r} == {"B"}
    assert r[0]["fourcc"] == "MJPG" and r[0]["width"] == "640" and r[0]["buffersize"] == "2"
    assert r[0]["controls"] == "focus_automatic_continuous=0;focus_absolute=20"
    assert float(r[0]["frame_age_ms"]) == pytest.approx(40.0)
    assert float(r[0]["t_capture"]) == pytest.approx(1000.0 - 0.04)
    img = Image.open(d / "edge" / "00001.jpg")
    assert img.size == (64, 48) and img.format == "JPEG"


def test_no_serial_anywhere_in_the_output(env):
    env["run"](["--label", "none", "--session", "s1", "--preview"], keys=["", ""])
    for p in env["root"].rglob("*.csv"):
        assert SERIAL not in p.read_text()
    assert all(SERIAL not in m for m in env["msgs"])


def test_numbering_continues_in_an_existing_session_and_labels_share_one_csv(env):
    env["run"](["--label", "none", "--session", "s1"], keys=["", ""])
    env["run"](["--label", "none", "--session", "s1"], keys=[""])
    env["run"](["--label", "center", "--session", "s1"], keys=[""])
    d = env["root"] / "samples" / "private" / "s1"
    assert sorted(p.name for p in (d / "none").glob("*.jpg")) == [
        "00001.jpg", "00002.jpg", "00003.jpg"
    ]  # fmt: skip
    assert [x["file"] for x in rows(d)] == [
        "none/00001.jpg", "none/00002.jpg", "none/00003.jpg", "center/00001.jpg"
    ]  # fmt: skip
    assert (d / "metadata.csv").read_text().count("file,label") == 1  # one header


def test_existing_file_is_never_overwritten(tmp_path: Path):
    cfg = cap.load_config("capture.yaml")
    w = cap.SampleWriter(tmp_path / "s", "none", cfg, {}, clock=lambda: 1.0)
    (w.label_dir / f"{w.index:05d}.jpg").write_text("keep")
    with pytest.raises(FileExistsError):
        w.save(FakeFrame(1, 0.9))


def test_default_session_name_is_the_date_time(env):
    env["run"](["--label", "none"], keys=[""])
    (session,) = [p.name for p in (env["root"] / "samples" / "private").iterdir()]
    assert len(session) == 13 and session[8] == "-" and session.replace("-", "").isdigit()


def test_preview_is_overwritten_with_the_last_image(env):
    env["run"](["--label", "none", "--session", "s1", "--preview"], keys=["", "", ""])
    preview = env["root"] / "outputs" / "preview_latest.jpg"
    last = env["root"] / "samples" / "private" / "s1" / "none" / "00003.jpg"
    assert np.array_equal(np.asarray(Image.open(preview)), np.asarray(Image.open(last)))
    assert not list((env["root"] / "outputs").glob("*.tmp.jpg"))


def test_no_preview_without_the_flag(env):
    env["run"](["--label", "none", "--session", "s1"], keys=[""])
    assert not (env["root"] / "outputs").exists()


# --- safety checks -----------------------------------------------------------------------


def test_refuses_to_save_where_git_would_track_the_images(env):
    rc = env["run"](["--label", "none", "--session", "s1"], keys=[""], ignored=False)
    assert rc == 2
    assert not (env["root"] / "samples").exists()
    assert any("not git-ignored" in m for m in env["msgs"])


@pytest.mark.parametrize("name", ["../escape", "a/b", "sp ace", ""])
def test_bad_session_names_are_rejected(env, name):
    assert env["run"](["--label", "none", "--session", name]) == 2


def test_unknown_label_is_rejected(env):
    with pytest.raises(SystemExit):
        env["run"](["--label", "hand"])


def test_labels_match_the_teacher_choices():
    labels = cap.load_config("capture.yaml")["labels"]
    assert {k: v["teacher"] for k, v in labels.items()} == {"none": "A", "edge": "B", "center": "C"}


def test_repo_output_locations_are_git_ignored():
    cfg = cap.load_config("capture.yaml")
    assert cap.is_git_ignored(cap.REPO_ROOT / cfg["out_root"] / "20261010-2240" / "none" / "1.jpg")
    assert cap.is_git_ignored(cap.REPO_ROOT / cfg["out_root"] / "x" / "metadata.csv")
    assert cap.is_git_ignored(cap.REPO_ROOT / cfg["preview_path"])
