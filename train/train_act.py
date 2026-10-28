"""Train ACT on the converted dataset with lerobot's own training loop (lerobot_train.train()).

    uv run --project train python train/train_act.py --name trial      # configs/train_act.yaml
    uv run --project train python train/train_act.py --name trial --steps 300 --batch-size 8
    uv run --project train python train/train_act.py --resume models/act/<run>   # checkpoints/last
    ... --resume models/act/<run> --steps 600   # continue to a larger total

Run folder models/act/<stamp>_<name>/ (git-ignored):
  run.json          settings, the lerobot command line, versions, git commit, GPU and Ollama state
  progress.jsonl    one line per log_freq steps: step, loss, grad_norm, lr, step time (data/update),
                    VRAM (torch max allocated / reserved), elapsed, ETA; and one line per checkpoint
  lerobot/          lerobot's output_dir: checkpoints/<step>/ (weights, optimizer, scheduler, step),
                    checkpoints/last -> newest
Before training, Ollama's /api/ps must list no loaded model (it would share the 16 GB of VRAM);
--allow-ollama goes on anyway. The Ollama service settings are not touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LAST_CONFIG = Path("lerobot/checkpoints/last/pretrained_model/train_config.json")


def ollama_loaded(url: str, timeout: float = 3.0) -> list[dict] | None:
    """Models Ollama has in memory (/api/ps), or None if Ollama does not answer."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/ps", timeout=timeout) as r:
            return [
                {"name": m.get("name"), "size_vram": m.get("size_vram")}
                for m in json.load(r).get("models", [])
            ]
    except OSError:
        return None


def gpu_memory() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.split(",")  # fmt: skip
        return {"used_mib": int(out[0]), "total_mib": int(out[1])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def lerobot_args(ds: dict, tr: dict, run: Path, steps: int, batch: int) -> list[str]:
    args = [
        f"--dataset.repo_id={ds['repo_id']}",
        f"--dataset.root={ROOT / ds['root']}",
        f"--policy.type={tr['policy']}",
        f"--policy.device={tr['device']}",
        "--policy.push_to_hub=false",
        f"--output_dir={run / 'lerobot'}",
        f"--job_name={run.name}",
        f"--seed={tr['seed']}",
        f"--steps={steps}",
        f"--batch_size={batch}",
        f"--num_workers={tr['num_workers']}",
        f"--log_freq={tr['log_freq']}",
        f"--save_freq={tr['save_freq']}",
        "--wandb.enable=false",
    ]
    args += [f"--policy.{k}={v}" for k, v in tr.get("policy_args", {}).items()]
    return args


def resume_args(run: Path, steps: int | None) -> list[str]:
    cfg = run / LAST_CONFIG
    if not cfg.exists():
        raise FileNotFoundError(f"no checkpoint to resume from: {cfg}")
    args = [f"--config_path={cfg}", "--resume=true"]
    return args + ([f"--steps={steps}"] if steps else [])


class ProgressLog(logging.Handler):
    """Turns lerobot's `logging.info(train_tracker)` and checkpoint messages into JSON lines."""

    def __init__(self, path: Path, total_steps: int, log_freq: int):
        super().__init__(logging.INFO)
        self.path, self.total, self.t0 = path, total_steps, time.monotonic()
        self.log_freq = log_freq
        self.first_step: int | None = None

    def _write(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def emit(self, record: logging.LogRecord) -> None:
        from lerobot.utils.logging_utils import MetricsTracker

        try:
            msg = record.msg
            elapsed = time.monotonic() - self.t0
            if isinstance(msg, MetricsTracker):
                m = {k: float(v.avg) for k, v in msg.metrics.items()}
                step = int(msg.steps)
                if self.first_step is None:
                    # steps done before this process (> 0 on resume); the first log comes
                    # log_freq steps after the start
                    self.first_step = step - self.log_freq
                done = step - self.first_step
                eta = elapsed / done * (self.total - step) if done > 0 else None
                row = {"step": step, "epoch": float(msg.epochs), **m}
                row.update(self._vram())
                row.update({"elapsed_s": round(elapsed, 1), "eta_s": eta and round(eta, 1)})
                row["t"] = datetime.now().isoformat(timespec="seconds")
                self._write(row)
            elif isinstance(msg, str) and msg.startswith("Checkpoint policy after step"):
                step = int(msg.rsplit(" ", 1)[-1])
                self._write({"event": "checkpoint", "step": step, "elapsed_s": round(elapsed, 1)})
        except Exception:  # logging must never stop training
            self.handleError(record)

    @staticmethod
    def _vram() -> dict:
        import torch

        if not torch.cuda.is_available():
            return {}
        mib = 1024**2
        return {
            "vram_max_alloc_mib": round(torch.cuda.max_memory_allocated() / mib),
            "vram_reserved_mib": round(torch.cuda.memory_reserved() / mib),
        }


def versions() -> dict:
    import importlib.metadata as md

    out = {n: md.version(n) for n in ("lerobot", "torch", "torchvision", "torchcodec", "numpy")}
    try:
        out["git_commit"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        out["git_commit"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "train_act.yaml"))
    ap.add_argument("--name", default="act", help="run folder suffix")
    ap.add_argument("--steps", type=int, help="override train.steps (with --resume: the new total)")
    ap.add_argument("--batch-size", type=int, help="override train.batch_size")
    ap.add_argument("--resume", metavar="RUN_DIR", help="continue a run from its last checkpoint")
    ap.add_argument("--allow-ollama", action="store_true", help="train even if a model is loaded")
    args = ap.parse_args()

    full = yaml.safe_load(Path(args.config).read_text())
    ds, tr = full["dataset"], full["train"]

    loaded = ollama_loaded(tr["ollama_url"])
    gpu = gpu_memory()
    print(f"ollama /api/ps: {loaded}   GPU memory: {gpu}", flush=True)
    if loaded and not args.allow_ollama:
        names = ", ".join(m["name"] for m in loaded)
        print(
            f"Ollama has {names} in VRAM. Unload it (`ollama stop <model>`) or wait for its "
            "keep_alive to expire, or pass --allow-ollama.",
            file=sys.stderr,
        )
        return 3

    if args.resume:
        run = Path(args.resume).resolve()
        prev = json.loads((run / "run.json").read_text())
        steps = args.steps or prev["steps"]
        cli = resume_args(run, args.steps)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = ROOT / tr["out_dir"] / f"{stamp}_{args.name}"
        run.mkdir(parents=True)  # lerobot creates run/lerobot itself (it must not exist)
        steps = args.steps or tr["steps"]
        cli = lerobot_args(ds, tr, run, steps, args.batch_size or tr["batch_size"])

    info_path = run / ("run.json" if not args.resume else f"resume_{datetime.now():%H%M%S}.json")
    info_path.write_text(
        json.dumps(
            {
                "steps": steps,
                "resume": bool(args.resume),
                "lerobot_argv": cli,
                "config": full,
                "versions": versions(),
                "gpu_before": gpu,
                "ollama_loaded_before": loaded,
                "started": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    print(f"run: {run}", flush=True)

    from lerobot.scripts import lerobot_train
    from lerobot.utils.import_utils import register_third_party_plugins

    progress = ProgressLog(run / "progress.jsonl", steps, tr["log_freq"])
    orig_init_logging = lerobot_train.init_logging

    def init_logging(*a, **kw):  # lerobot clears the root handlers here: add ours after it
        orig_init_logging(*a, **kw)
        logging.getLogger().addHandler(progress)

    lerobot_train.init_logging = init_logging
    register_third_party_plugins()
    sys.argv = ["lerobot_train", *cli]
    t0 = time.monotonic()
    try:
        lerobot_train.train()
    finally:
        progress._write(
            {"event": "end", "elapsed_s": round(time.monotonic() - t0, 1), **ProgressLog._vram()}
        )
    print(f"done: {run}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
