"""Record the execution environment to docs/env_report.json.

Each section is collected independently; a failure is recorded as status "error" and the
script keeps going. No hostname, user name, absolute home paths or camera serial numbers are
written (the report is committed to a public repository); scrub() also replaces them in any
error message as a second line of defence.

The camera section opens the camera with configs/camera.yaml for about 2 s (no window, no
image saved) to record the settings actually applied, the measured fps and frame age.

    uv run python scripts/check_env.py
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = [
    "numpy",
    "pillow",
    "pyyaml",
    "httpx",
    "torch",
    "transformers",
    "opencv-python-headless",
    "pytest",
    "ruff",
]
GIB = 1024**3
# the separate training env (train/, lerobot): versions read from train/uv.lock
TRAIN_PACKAGES = ["lerobot", "torch", "torchvision", "torchcodec", "numpy", "rosbags", "av"]
TRAIN_PROBE = """
import glob, hashlib, importlib.metadata as m, json, os, torch
d = os.path.join(torch.hub.get_dir(), "checkpoints")
w = {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
     for p in sorted(glob.glob(os.path.join(d, "resnet*")))}
print(json.dumps({"lerobot": m.version("lerobot"), "torch": torch.__version__,
                  "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
                  "backbone_weights": w}))
"""

Section = dict[str, Any]


def _section(fn: Callable[[], Section]) -> Section:
    try:
        out = fn()
    except Exception as e:  # every failure is reported, never raised
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    out.setdefault("status", "ok")
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def check_system(run: Callable = subprocess.run) -> Section:
    uv = run(["uv", "--version"], capture_output=True, text=True, check=True).stdout.strip()
    os_name = platform.platform()
    try:
        osr = platform.freedesktop_os_release()
        os_name = osr.get("PRETTY_NAME", os_name)
    except OSError:
        pass
    return {
        "os": os_name,
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "uv": uv,
    }


def check_packages(repo_root: Path = REPO_ROOT) -> Section:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    lock = repo_root / "uv.lock"
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    return {"versions": versions, "uv_lock_sha256": lock_sha}


def check_train_env(repo_root: Path = REPO_ROOT, run: Callable = subprocess.run) -> Section:
    """Versions pinned in train/uv.lock, and a probe run inside that env (CUDA, weights)."""
    lock = repo_root / "train" / "uv.lock"
    if not lock.exists():
        return {"status": "skipped", "reason": "train/uv.lock not found"}
    pkgs = {p["name"]: p.get("version") for p in tomllib.loads(lock.read_text()).get("package", [])}
    out: Section = {
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "versions": {n: pkgs.get(n) for n in TRAIN_PACKAGES},
    }
    try:
        res = run(
            ["uv", "run", "--project", str(repo_root / "train"), "python", "-c", TRAIN_PROBE],
            capture_output=True, text=True, check=True, timeout=300,
        )  # fmt: skip
        out["probe"] = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception as e:  # the env may not be synced yet: record, do not fail the report
        out["probe"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    return out


def check_gpu_smi(run: Callable = subprocess.run) -> Section:
    if shutil.which("nvidia-smi") is None:
        return {"status": "skipped", "reason": "nvidia-smi not found"}
    q = "name,driver_version,memory.total,memory.used"
    res = run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    gpus = []
    for line in res.stdout.strip().splitlines():
        name, driver, total, used = (s.strip() for s in line.split(","))
        gpus.append(
            {
                "name": name,
                "driver": driver,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
            }
        )
    return {"gpus": gpus}


def check_torch() -> Section:
    if importlib.util.find_spec("torch") is None:
        return {"status": "skipped", "reason": "torch not installed"}
    import torch

    out: Section = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if out["cuda_available"]:
        out["device"] = torch.cuda.get_device_name(0)
        out["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        out["cudnn"] = torch.backends.cudnn.version()
    else:
        out["status"] = "error"
        out["error"] = "torch installed but CUDA not available"
    return out


def check_ollama(teacher_cfg: dict[str, Any], http_get: Callable = httpx.get) -> Section:
    url = teacher_cfg.get("ollama_url", "http://localhost:11434").rstrip("/")
    wanted = list(teacher_cfg.get("models", []))
    version = http_get(f"{url}/api/version", timeout=3.0)
    version.raise_for_status()
    tags = http_get(f"{url}/api/tags", timeout=3.0)
    tags.raise_for_status()
    installed = {m["name"]: m for m in tags.json().get("models", [])}
    models = {
        name: {
            "digest": installed[name].get("digest"),
            "size_bytes": installed[name].get("size"),
            "quantization": installed[name].get("details", {}).get("quantization_level"),
            "capabilities": installed[name].get("capabilities"),
        }
        for name in wanted
        if name in installed
    }
    missing = [n for n in wanted if n not in installed]
    out: Section = {"version": version.json().get("version"), "models": models, "missing": missing}
    if missing:
        out["status"] = "error"
        out["error"] = f"missing models: {missing}"
    return out


def _hf_hub_cache() -> Path:
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"])
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub"


def check_hf(zeroshot_cfg: dict[str, Any], hub_cache: Path | None = None) -> Section:
    model_id = zeroshot_cfg.get("model_id")
    if not model_id:
        return {"status": "skipped", "reason": "configs/zeroshot.yaml has no model_id yet"}
    revision = zeroshot_cfg.get("revision")
    hub_cache = hub_cache or _hf_hub_cache()
    snap = hub_cache / f"models--{model_id.replace('/', '--')}" / "snapshots"
    cached = sorted(p.name for p in snap.iterdir()) if snap.is_dir() else []
    return {
        "model_id": model_id,
        "revision": revision,
        "cached_revisions": cached,
        "pinned_revision_cached": bool(revision) and revision in cached,
    }


def check_disk(repo_root: Path = REPO_ROOT, hub_cache: Path | None = None) -> Section:
    hub_cache = hub_cache or _hf_hub_cache()
    target = hub_cache if hub_cache.exists() else Path.home()
    out = {}
    for label, p in (("repo", repo_root), ("hf_cache", target)):
        u = shutil.disk_usage(p)
        out[label] = {"free_gib": round(u.free / GIB, 1), "total_gib": round(u.total / GIB, 1)}
    return out


BY_ID = re.compile(
    r"usb-(?P<vendor>[0-9a-fA-F]{4})_(?P<model>.+)_(?P<serial>[^_/]+)-video-index\d+$"
)


def camera_identity(device: str) -> dict[str, str | None]:
    """Vendor id and model from a /dev/v4l/by-id name, without the serial number."""
    m = BY_ID.search(device)
    if not m:
        return {"vendor_id": None, "model": None}
    return {"vendor_id": m["vendor"].lower(), "model": m["model"]}


def usb_speed_mbps(device: str, sys_root: Path = Path("/sys")) -> int | None:
    """USB link speed of the camera (e.g. 480 for USB 2.0 high speed) from sysfs."""
    node = Path(os.path.realpath(device)).name  # e.g. video0
    d = (sys_root / "class" / "video4linux" / node / "device").resolve()
    for candidate in (d, *d.parents):
        speed = candidate / "speed"
        if speed.is_file():
            return int(float(speed.read_text().strip()))
        if candidate == sys_root:
            break
    return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _v4l2ctl_version(run: Callable) -> str | None:
    try:
        out = run(["v4l2-ctl", "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def measure_camera(cam, n_frames: int = 60) -> dict[str, float]:
    """fps from capture timestamps and frame age (read-out - capture) over n new frames."""
    first = f = cam.read()
    ages = []
    for _ in range(n_frames):
        f = cam.wait_new(f.seq, timeout=1.0)
        ages.append(time.monotonic() - f.t_capture)
    return {
        "frames": n_frames,
        "fps": round((f.seq - first.seq) / (f.t_capture - first.t_capture), 2),
        "frame_age_ms_mean": round(1000 * sum(ages) / len(ages), 1),
        "frame_age_ms_max": round(1000 * max(ages), 1),
    }


def check_camera(
    repo_root: Path = REPO_ROOT,
    video_glob: Callable[[], list[str]] = lambda: glob.glob("/dev/video*"),
    camera_factory: Callable | None = None,
    run: Callable = subprocess.run,
    sys_root: Path = Path("/sys"),
) -> Section:
    try:
        import cv2

        opencv = cv2.__version__
    except ImportError:
        opencv = None
    out: Section = {
        "versions": {"opencv": opencv, "v4l2-ctl": _v4l2ctl_version(run)},
        "uvcvideo_clock": _read_text(sys_root / "module" / "uvcvideo" / "parameters" / "clock"),
        "devices": sorted(video_glob()),
    }
    if not out["devices"]:
        return {**out, "status": "skipped", "reason": "not connected"}
    cfg_path = repo_root / "configs" / "camera.yaml"
    if not cfg_path.exists():
        reason = "configs/camera.yaml not found (copy configs/camera.example.yaml)"
        return {**out, "status": "skipped", "reason": reason}

    from jevlite.camera import Camera, load_camera_config, redact_device

    cfg = load_camera_config(cfg_path)
    out["device"] = redact_device(cfg.device)
    out.update(camera_identity(cfg.device))
    out["usb_speed_mbps"] = usb_speed_mbps(cfg.device, sys_root)
    out["requested"] = {
        "fourcc": cfg.fourcc,
        "width": cfg.width,
        "height": cfg.height,
        "fps": cfg.fps,
        "buffersize": cfg.buffersize,
        "controls": dict(cfg.controls),
    }
    try:
        with (camera_factory or Camera)(cfg) as cam:
            rb = dict(cam.readback)
            out["actual"] = {k: rb[k] for k in ("fourcc", "width", "height", "fps", "buffersize")}
            out["actual"]["controls"] = rb.get("controls", {})
            out["mismatches"] = list(cam.mismatches)
            out["clock_check"] = {
                "offsets_ms": [round(1000 * o, 1) for o in cam.clock_offsets],
                "limit_ms": round(1000 * cfg.max_clock_offset_s, 1),
            }
            out["measured"] = measure_camera(cam)
    except Exception as e:  # recorded, never raised; keep what was collected so far
        return {**out, "status": "error", "error": f"{type(e).__name__}: {e}"}
    if out["mismatches"]:
        out["status"] = "error"
        out["error"] = "; ".join(out["mismatches"])
    return out


def _sensitive_strings() -> list[tuple[str, str]]:
    """(value, placeholder) pairs that must never appear in the report."""
    pairs = [(str(Path.home()), "~")]
    for value, tag in ((socket.gethostname(), "<HOST>"), (os.environ.get("USER", ""), "<USER>")):
        if value:
            pairs.append((value, tag))
    try:
        import getpass

        pairs.append((getpass.getuser(), "<USER>"))
    except (KeyError, OSError):
        pass
    for name in glob.glob("/dev/v4l/by-id/*"):
        m = BY_ID.search(name)
        if m:
            pairs.append((m["serial"], "<SERIAL>"))
    # longest first so a home path is replaced before the user name inside it
    return sorted({p for p in pairs if p[0]}, key=lambda p: -len(p[0]))


def scrub(obj: Any, pairs: list[tuple[str, str]] | None = None) -> Any:
    """Replace sensitive strings in every string of a JSON-like object."""
    pairs = _sensitive_strings() if pairs is None else pairs
    if isinstance(obj, str):
        for value, tag in pairs:
            obj = obj.replace(value, tag)
        return obj
    if isinstance(obj, dict):
        return {scrub(k, pairs): scrub(v, pairs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, pairs) for v in obj]
    return obj


def collect(
    repo_root: Path = REPO_ROOT,
    run: Callable = subprocess.run,
    http_get: Callable = httpx.get,
    video_glob: Callable[[], list[str]] | None = None,
    hub_cache: Path | None = None,
    camera_factory: Callable | None = None,
    sys_root: Path | None = None,
    sensitive: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    teacher_cfg = _load_yaml(repo_root / "configs" / "teacher.yaml")
    zeroshot_cfg = _load_yaml(repo_root / "configs" / "zeroshot.yaml")
    cam_kwargs: dict[str, Any] = {"repo_root": repo_root, "run": run}
    if video_glob:
        cam_kwargs["video_glob"] = video_glob
    if camera_factory:
        cam_kwargs["camera_factory"] = camera_factory
    if sys_root:
        cam_kwargs["sys_root"] = sys_root
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "system": _section(lambda: check_system(run)),
        "packages": _section(lambda: check_packages(repo_root)),
        "train_env": _section(lambda: check_train_env(repo_root, run)),
        "gpu": _section(lambda: check_gpu_smi(run)),
        "torch": _section(check_torch),
        "ollama": _section(lambda: check_ollama(teacher_cfg, http_get)),
        "hf": _section(lambda: check_hf(zeroshot_cfg, hub_cache)),
        "disk": _section(lambda: check_disk(repo_root, hub_cache)),
        "camera": _section(lambda: check_camera(**cam_kwargs)),
    }
    return scrub(report, sensitive)


def summarize(report: dict[str, Any]) -> str:
    lines = []
    for key, sec in report.items():
        if not isinstance(sec, dict):
            continue
        status = sec.get("status", "?")
        detail = sec.get("error") or sec.get("reason") or ""
        lines.append(f"{key:<9} {status:<8} {detail}".rstrip())
    return "\n".join(lines)


def main() -> int:
    report = collect()
    out = REPO_ROOT / "docs" / "env_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(summarize(report))
    print(f"-> {out.relative_to(REPO_ROOT)}")
    errors = [k for k, v in report.items() if isinstance(v, dict) and v.get("status") == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
