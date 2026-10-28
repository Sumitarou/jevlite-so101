"""train/policy_server.py and train/units.py (fake policy, CPU only)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_runtime import hold_runtime
from policy_server import make_handler
from scripts.sim.policy_link import KEY_ENV, PolicyClient, new_key
from units import ARM6, sim_command

ROOT = Path(__file__).resolve().parents[2]
DS = yaml.safe_load((ROOT / "configs" / "train_act.yaml").read_text())["dataset"]


def test_sim_command_orders_names_and_adds_fixed():
    q = sim_command([1, 2, 3, 4, 5], DS["joints"], DS["fixed_joints"])
    assert q.tolist() == [1, 2, 3, 4, 5, -0.16]
    q = sim_command([1, 2, 3, 4, 5], DS["joints"], DS["fixed_joints"], names=ARM6[::-1])
    assert q.tolist() == [-0.16, 5, 4, 3, 2, 1]


@pytest.mark.parametrize(
    "action, fixed, exc",
    [([1, 2, 3, 4], {"gripper": 0}, ValueError),  # wrong length
     ([1, 2, 3, float("nan"), 5], {"gripper": 0}, ValueError),  # not finite
     ([1, 2, 3, 4, 5], {}, KeyError),  # gripper missing
     ([1, 2, 3, 4, 5], {"gripper": 0, "wrist_roll": 0}, ValueError)],  # learned and fixed
)  # fmt: skip
def test_sim_command_rejects(action, fixed, exc):
    with pytest.raises(exc):
        sim_command(action, DS["joints"], fixed)


def test_handler_reset_act_info():
    rt = hold_runtime(DS)
    info = {"command_names": ARM6, "joints": rt.joints}
    h = make_handler(rt, info)
    assert h({"cmd": "reset"})["queue_len"] == 0
    r = h({"cmd": "act", "image": np.zeros((8, 8, 3), np.uint8), "state": np.ones(5)})
    assert r["names"] == ARM6 and r["command"][-1] == pytest.approx(-0.16)
    assert h({"cmd": "info"}) is info
    with pytest.raises(ValueError):
        h({"cmd": "bogus"})


def test_server_process_end_to_end(tmp_path):
    addr, key = str(tmp_path / "s.sock"), new_key()
    env = {**os.environ, KEY_ENV: key, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "train" / "policy_server.py"), "--fake", "hold",
         "--address", addr, "--image-shape", "48", "64", "3"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    try:
        c = PolicyClient.connect(addr, key, timeout_s=60, alive=lambda: proc.poll() is None)
        info = c.info()
        assert info["policy"] == "fake:hold" and info["joints"] == DS["joints"]
        assert c.reset() == {"ok": True, "queue_len": 0, "episode": 1}
        s = np.array([0.1, -0.2, 0.3, 1.2, -1.5], np.float32)
        r = c.act(np.zeros((48, 64, 3), np.uint8), s)
        np.testing.assert_allclose(r["command"], [*s, -0.16], atol=1e-6)
        c.close()
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        out = proc.stdout.read()
    assert "READY" in out
    assert not Path(addr).exists()  # the listener removed its socket file


def test_server_needs_the_key(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != KEY_ENV}
    r = subprocess.run(
        [sys.executable, str(ROOT / "train" / "policy_server.py"), "--fake", "hold",
         "--address", str(tmp_path / "s.sock")],
        env=env, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert r.returncode == 2 and KEY_ENV in r.stderr
