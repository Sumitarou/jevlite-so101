"""train/train_act.py and train/plot_progress.py helpers (no GPU, no training)."""

import json
import logging
from pathlib import Path

import pytest
import yaml

import plot_progress
import train_act

ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    full = yaml.safe_load((ROOT / "configs" / "train_act.yaml").read_text())
    return full["dataset"], full["train"]


def test_lerobot_args_come_from_the_config(tmp_path):
    ds, tr = _cfg()
    args = train_act.lerobot_args(ds, tr, tmp_path / "run", steps=123, batch=16)
    assert f"--dataset.root={ROOT / ds['root']}" in args
    assert "--policy.type=act" in args and "--policy.push_to_hub=false" in args
    assert "--steps=123" in args and "--batch_size=16" in args
    assert f"--output_dir={tmp_path / 'run' / 'lerobot'}" in args
    assert "--wandb.enable=false" in args
    assert f"--policy.chunk_size={tr['policy_args']['chunk_size']}" in args


def test_resume_args_need_a_last_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_act.resume_args(tmp_path, None)
    cfg = tmp_path / train_act.LAST_CONFIG
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}")
    assert train_act.resume_args(tmp_path, None) == [f"--config_path={cfg}", "--resume=true"]
    assert train_act.resume_args(tmp_path, 600)[-1] == "--steps=600"


def test_ollama_loaded_none_when_down():
    assert train_act.ollama_loaded("http://127.0.0.1:9", timeout=0.2) is None


def test_main_stops_when_ollama_has_a_model(monkeypatch, tmp_path):
    monkeypatch.setattr(train_act, "ollama_loaded", lambda url: [{"name": "x", "size_vram": 1}])
    monkeypatch.setattr(train_act, "gpu_memory", lambda: None)
    monkeypatch.setattr("sys.argv", ["train_act.py", "--name", "t"])
    assert train_act.main() == 3


def _tracker(step, loss):
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    class _M(MetricsTracker):
        def __init__(self):  # skip the dataset-dependent constructor
            pass

    t = _M()
    t.__dict__.update({"steps": step, "epochs": step / 100, "samples": step * 8, "episodes": 0})
    m = {"loss": AverageMeter("loss", ":.3f"), "update_s": AverageMeter("u", ":.3f"),
         "dataloading_s": AverageMeter("d", ":.3f")}  # fmt: skip
    m["loss"].update(loss)
    m["update_s"].update(0.15)
    m["dataloading_s"].update(0.001)
    t.__dict__["metrics"] = m
    return t


def test_progress_log_rows_eta_and_checkpoints(tmp_path):
    path = tmp_path / "progress.jsonl"
    h = train_act.ProgressLog(path, total_steps=300, log_freq=10)
    log = logging.getLogger("test_progress")
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    log.info(_tracker(210, 3.0))  # first log after a resume from step 200
    log.info("Checkpoint policy after step 300")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows[0]["step"] == 210 and rows[0]["loss"] == pytest.approx(3.0)
    assert h.first_step == 200
    assert rows[0]["eta_s"] is not None  # 10 done, 90 to go
    assert rows[1] == {"event": "checkpoint", "step": 300, "elapsed_s": rows[1]["elapsed_s"]}


def test_summarize_uses_medians_and_lists_checkpoints():
    rows = [
        {"step": s, "loss": 10 - s / 100, "update_s": u, "dataloading_s": 0.001}
        for s, u in [(10, 1.0), (20, 0.15), (30, 0.15), (40, 0.15), (50, 1.0)]
    ]
    s = plot_progress.summarize(rows, [{"event": "checkpoint", "step": 50}])
    assert s["update_s_median"] == pytest.approx(0.15)
    assert s["loss_first"] > s["loss_last"]
    assert s["checkpoints"] == [50]
