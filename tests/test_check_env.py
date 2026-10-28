import json
import subprocess
from pathlib import Path

import httpx
import pytest

from scripts import check_env

SMI_OUT = "NVIDIA RTX 2000 Ada Generation, 580.126.09, 16380, 512\n"
TAGS = {
    "models": [
        {
            "name": "gemma3:4b",
            "digest": "abc123",
            "size": 3350272403,
            "details": {"quantization_level": "Q4_K_M"},
            "capabilities": ["completion", "vision"],
        }
    ]
}


TRAIN_PROBE_OUT = (
    '{"lerobot": "0.6.1", "torch": "2.11.0+cu128", "cuda_available": true, "cuda": "12.8", '
    '"backbone_weights": {"resnet18-f37072fd.pth": "ab"}}\n'
)


def fake_run(cmd, **kwargs):
    if cmd[:3] == ["uv", "run", "--project"]:
        return subprocess.CompletedProcess(cmd, 0, stdout=TRAIN_PROBE_OUT)
    if cmd[0] == "uv":
        return subprocess.CompletedProcess(cmd, 0, stdout="uv 0.12.24\n")
    if cmd[0] == "nvidia-smi":
        return subprocess.CompletedProcess(cmd, 0, stdout=SMI_OUT)
    raise FileNotFoundError(cmd[0])


def fake_http_get(url, timeout):
    req = httpx.Request("GET", url)
    if url.endswith("/api/version"):
        return httpx.Response(200, json={"version": "0.12.0"}, request=req)
    if url.endswith("/api/tags"):
        return httpx.Response(200, json=TAGS, request=req)
    return httpx.Response(404, request=req)


def failing_http_get(url, timeout):
    raise httpx.ConnectError("connection refused")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "teacher.yaml").write_text(
        "ollama_url: http://localhost:11434\nmodels: [gemma3:4b]\n"
    )
    (tmp_path / "uv.lock").write_text("lock")
    return tmp_path


def _collect(repo, **kw):
    kw.setdefault("run", fake_run)
    kw.setdefault("http_get", fake_http_get)
    kw.setdefault("video_glob", lambda: [])
    return check_env.collect(repo_root=repo, hub_cache=repo / "hub", **kw)


def test_report_structure_and_json_serializable(repo, monkeypatch):
    monkeypatch.setattr(check_env.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    report = _collect(repo)
    keys = ("system", "packages", "train_env", "gpu", "torch", "ollama", "hf", "disk", "camera")
    for key in keys:
        assert "status" in report[key], key
    json.dumps(report)
    assert report["gpu"]["gpus"][0]["driver"] == "580.126.09"
    assert report["ollama"]["status"] == "ok"
    assert report["ollama"]["models"]["gemma3:4b"]["digest"] == "abc123"
    assert len(report["packages"]["uv_lock_sha256"]) == 64


def test_camera_not_connected_is_skipped(repo):
    report = _collect(repo, video_glob=lambda: [])
    assert report["camera"]["status"] == "skipped"
    assert report["camera"]["reason"] == "not connected"
    assert "camera    skipped  not connected" in check_env.summarize(report)


# --- camera ------------------------------------------------------------------------------

SERIAL = "ABCD1234"
SPEC_YAML = """device: {dev}
width: 640
height: 480
fps: 30
fourcc: MJPG
buffersize: 2
clock_check_frames: 3
max_clock_offset_s: 0.5
controls:
  focus_automatic_continuous: 0
  focus_absolute: 20
"""


class FakeFrame:
    def __init__(self, seq, t_capture):
        self.seq, self.t_capture = seq, t_capture
        self.rgb = None


class FakeCam:
    """Stands in for jevlite.camera.Camera: 30 fps frames that are 40 ms old when read."""

    def __init__(self, cfg, mismatches=(), fail=None):
        self.cfg, self.fail = cfg, fail
        self.mismatches = list(mismatches)
        self.readback = {
            "width": 640, "height": 480, "fps": 30.0, "fourcc": "MJPG", "buffersize": 2,
            "controls": {"focus_automatic_continuous": 0, "focus_absolute": 20},
        }  # fmt: skip
        self.clock_offsets = [0.0296, 0.0281, 0.0333]
        self.seq = 0

    def __enter__(self):
        if self.fail:
            raise self.fail
        return self

    def __exit__(self, *exc):
        pass

    def _next(self):
        self.seq += 1
        return FakeFrame(self.seq, check_env.time.monotonic() - 0.040)

    def read(self):
        return self._next()

    def wait_new(self, after_seq, timeout=1.0):
        return self._next()


@pytest.fixture
def cam_env(tmp_path: Path, repo: Path):
    """A fake /sys and /dev/v4l/by-id with a C920 on a USB 2.0 port, and configs/camera.yaml."""
    sys_root = tmp_path / "sys"
    usb_dev = sys_root / "devices" / "pci0000:00" / "usb1" / "1-13"
    (usb_dev / "1-13:1.0").mkdir(parents=True)
    (usb_dev / "speed").write_text("480\n")
    v4l = sys_root / "class" / "video4linux" / "video0"
    v4l.mkdir(parents=True)
    (v4l / "device").symlink_to(usb_dev / "1-13:1.0")
    clock = sys_root / "module" / "uvcvideo" / "parameters"
    clock.mkdir(parents=True)
    (clock / "clock").write_text("CLOCK_MONOTONIC\n")
    dev = tmp_path / "dev"
    (dev / "v4l" / "by-id").mkdir(parents=True)
    (dev / "video0").write_text("")
    by_id = dev / "v4l" / "by-id" / f"usb-046d_HD_Pro_Webcam_C920_{SERIAL}-video-index0"
    by_id.symlink_to(dev / "video0")
    (repo / "configs" / "camera.yaml").write_text(SPEC_YAML.format(dev=by_id))
    return {"repo": repo, "sys_root": sys_root, "by_id": by_id}


def run_with_v4l2(cmd, **kwargs):
    if cmd[0] == "v4l2-ctl":
        return subprocess.CompletedProcess(cmd, 0, stdout="v4l2-ctl 1.26.1\n")
    return fake_run(cmd, **kwargs)


def _camera(env, factory=FakeCam, sensitive=None):
    report = _collect(
        env["repo"],
        run=run_with_v4l2,
        video_glob=lambda: ["/dev/video0", "/dev/video1"],
        camera_factory=factory,
        sys_root=env["sys_root"],
        sensitive=sensitive if sensitive is not None else [(SERIAL, "<SERIAL>")],
    )
    return report["camera"], json.dumps(report)


def test_camera_records_model_usb_settings_and_measurements(cam_env):
    cam, text = _camera(cam_env)
    assert cam["status"] == "ok"
    assert cam["vendor_id"] == "046d" and cam["model"] == "HD_Pro_Webcam_C920"
    assert cam["usb_speed_mbps"] == 480
    assert cam["device"].endswith("HD_Pro_Webcam_C920_<SERIAL>-video-index0")
    assert cam["requested"] == {
        "fourcc": "MJPG", "width": 640, "height": 480, "fps": 30.0, "buffersize": 2,
        "controls": {"focus_automatic_continuous": 0, "focus_absolute": 20},
    }  # fmt: skip
    assert cam["actual"]["fourcc"] == "MJPG" and cam["actual"]["buffersize"] == 2
    assert cam["actual"]["controls"] == {"focus_automatic_continuous": 0, "focus_absolute": 20}
    assert cam["mismatches"] == []
    assert cam["clock_check"] == {"offsets_ms": [29.6, 28.1, 33.3], "limit_ms": 500.0}
    m = cam["measured"]
    assert m["frames"] == 60 and m["fps"] > 0
    assert 39.0 <= m["frame_age_ms_mean"] <= m["frame_age_ms_max"] < 200.0
    assert cam["uvcvideo_clock"] == "CLOCK_MONOTONIC"
    assert cam["versions"]["v4l2-ctl"] == "v4l2-ctl 1.26.1"
    assert cam["versions"]["opencv"]
    assert SERIAL not in text


def test_camera_identity_strips_the_serial():
    ident = check_env.camera_identity(
        "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_ABCD1234-video-index0"
    )
    assert ident == {"vendor_id": "046d", "model": "HD_Pro_Webcam_C920"}
    assert check_env.camera_identity("/dev/video0") == {"vendor_id": None, "model": None}


def test_camera_without_config_is_skipped_but_recorded(cam_env):
    (cam_env["repo"] / "configs" / "camera.yaml").unlink()
    cam, _ = _camera(cam_env)
    assert cam["status"] == "skipped"
    assert "camera.yaml not found" in cam["reason"]
    assert cam["devices"] == ["/dev/video0", "/dev/video1"]
    assert cam["uvcvideo_clock"] == "CLOCK_MONOTONIC"
    assert "measured" not in cam


def test_camera_mismatch_is_an_error_with_details(cam_env):
    cam, _ = _camera(cam_env, factory=lambda c: FakeCam(c, ["fps is 15.0, expected 30.0"]))
    assert cam["status"] == "error"
    assert "fps is 15.0" in cam["error"]
    assert cam["measured"]["frames"] == 60


def test_camera_open_failure_is_recorded_not_raised(cam_env):
    def factory(c):
        return FakeCam(c, fail=RuntimeError(f"cannot open camera {cam_env['by_id']}"))

    cam, text = _camera(cam_env, factory=factory)
    assert cam["status"] == "error"
    assert cam["model"] == "HD_Pro_Webcam_C920"  # what was known before opening is kept
    assert SERIAL not in text  # the error message was scrubbed


def test_scrub_replaces_home_host_user_and_serial():
    pairs = [("/home/alice", "~"), ("myhost", "<HOST>"), ("alice", "<USER>"), (SERIAL, "<SERIAL>")]
    obj = {"a": ["/home/alice/dev/x", "on myhost by alice"], f"k{SERIAL}": f"cam_{SERIAL}", "n": 3}
    out = check_env.scrub(obj, sorted(pairs, key=lambda p: -len(p[0])))
    text = json.dumps(out)
    for value in ("/home/alice", "myhost", "alice", SERIAL):
        assert value not in text
    assert out["a"][0] == "~/dev/x" and out["n"] == 3


def test_sensitive_strings_include_home_and_host():
    values = [v for v, _ in check_env._sensitive_strings()]
    assert str(Path.home()) in values
    assert check_env.socket.gethostname() in values


def test_ollama_down_is_error_not_exception(repo):
    report = _collect(repo, http_get=failing_http_get)
    assert report["ollama"]["status"] == "error"
    assert "ConnectError" in report["ollama"]["error"]


def test_ollama_missing_model_is_error(repo):
    (repo / "configs" / "teacher.yaml").write_text("models: [gemma3:4b, qwen2.5vl:7b]\n")
    report = _collect(repo)
    assert report["ollama"]["status"] == "error"
    assert report["ollama"]["missing"] == ["qwen2.5vl:7b"]


def test_hf_skipped_without_zeroshot_config(repo):
    assert _collect(repo)["hf"]["status"] == "skipped"


def test_hf_pinned_revision_cached(repo):
    (repo / "configs" / "zeroshot.yaml").write_text("model_id: org/m\nrevision: deadbeef\n")
    (repo / "hub" / "models--org--m" / "snapshots" / "deadbeef").mkdir(parents=True)
    hf = _collect(repo)["hf"]
    assert hf["pinned_revision_cached"] is True
    assert hf["cached_revisions"] == ["deadbeef"]


def test_report_has_no_hostname_or_home(repo):
    text = json.dumps(_collect(repo))
    assert check_env.platform.node() not in text
    assert str(Path.home()) not in text


TRAIN_LOCK = """version = 1

[[package]]
name = "lerobot"
version = "0.6.1"

[[package]]
name = "numpy"
version = "2.2.6"
"""


def test_train_env_skipped_without_lock(repo):
    assert _collect(repo)["train_env"]["status"] == "skipped"


def test_train_env_versions_from_lock_and_probe(repo):
    (repo / "train").mkdir()
    (repo / "train" / "uv.lock").write_text(TRAIN_LOCK)
    sec = _collect(repo)["train_env"]
    assert sec["status"] == "ok"
    assert sec["versions"]["lerobot"] == "0.6.1" and sec["versions"]["numpy"] == "2.2.6"
    assert sec["versions"]["rosbags"] is None
    assert len(sec["uv_lock_sha256"]) == 64
    assert sec["probe"]["cuda_available"] is True
    assert "resnet18-f37072fd.pth" in sec["probe"]["backbone_weights"]


def test_train_env_probe_failure_is_recorded(repo):
    (repo / "train").mkdir()
    (repo / "train" / "uv.lock").write_text(TRAIN_LOCK)

    def run(cmd, **kw):
        if cmd[:3] == ["uv", "run", "--project"]:
            raise subprocess.CalledProcessError(1, cmd)
        return fake_run(cmd, **kw)

    sec = _collect(repo, run=run)["train_env"]
    assert sec["status"] == "ok" and sec["probe"]["status"] == "error"
