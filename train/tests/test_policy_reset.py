"""train/policy_runtime.py: begin_episode() empties the action queue (no GPU).

Without a reset, ACT keeps executing the chunk it predicted in the previous episode, whatever it
sees now. begin_episode() is the one entry point that resets the policy (also to be used by the
safety gate's on_resume, G7).
"""

import ast
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from policy_runtime import PolicyRuntime, hold_runtime, identity

ROOT = Path(__file__).resolve().parents[2]
DS = yaml.safe_load((ROOT / "configs" / "train_act.yaml").read_text())["dataset"]
IMG = np.zeros((32, 48, 3), np.uint8)


class ChunkPolicy:
    """Behaves like ACT's queue: when empty, predict n actions from the current state."""

    def __init__(self, n=4):
        self.n = n
        self.config = type("Cfg", (), {"temporal_ensemble_coeff": None})()
        self._action_queue = deque([], maxlen=n)
        self.predictions = 0

    def reset(self):
        self._action_queue = deque([], maxlen=self.n)

    def select_action(self, batch):
        if not self._action_queue:
            s = batch["observation.state"]
            self._action_queue.extend(s + 0.01 * k for k in range(self.n))
            self.predictions += 1
        return self._action_queue.popleft()


def _rt(policy):
    return PolicyRuntime(policy, identity, identity, DS, "cpu")


def test_without_reset_the_previous_chunk_comes_out():
    rt = _rt(ChunkPolicy())
    rt.begin_episode()
    s1, s2 = np.full(5, 0.5, np.float32), np.full(5, -0.7, np.float32)
    a1 = rt.act(IMG, s1)["action"]
    np.testing.assert_allclose(a1, s1, atol=1e-6)
    # "next episode" without a reset: the old chunk continues, the new state is ignored
    a2 = rt.act(IMG, s2)["action"]
    np.testing.assert_allclose(a2, s1 + 0.01, atol=1e-6)
    assert rt.policy.predictions == 1


def test_with_reset_the_new_observation_is_used():
    rt = _rt(ChunkPolicy())
    rt.begin_episode()
    s1, s2 = np.full(5, 0.5, np.float32), np.full(5, -0.7, np.float32)
    rt.act(IMG, s1)
    assert rt.queue_len() == 3
    assert rt.begin_episode() == 0
    r = rt.act(IMG, s2)
    np.testing.assert_allclose(r["action"], s2, atol=1e-6)
    assert r["inferred"] and rt.policy.predictions == 2 and rt.episodes == 2


def _tiny_act():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    torch.manual_seed(0)
    cfg = ACTConfig(
        device="cpu", chunk_size=4, n_action_steps=4, dim_model=32, n_heads=2,
        dim_feedforward=64, n_encoder_layers=1, n_vae_encoder_layers=1, latent_dim=4,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (5,)),
            DS["image_key"]: PolicyFeature(FeatureType.VISUAL, (3, 32, 48)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (5,))},
    )  # fmt: skip
    p = ACTPolicy(cfg)
    p.eval()
    return p


def test_real_act_queue_without_and_with_reset():
    """The same with lerobot's ACT (tiny, random weights, CPU): the queue is the thing reset."""
    rt = _rt(_tiny_act())
    rt.begin_episode()
    rng = np.random.default_rng(0)
    img_a = rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)
    img_b = rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)
    s_a, s_b = np.full(5, 0.3, np.float32), np.full(5, -2.0, np.float32)

    first = rt.act(img_a, s_a)
    assert first["inferred"] and rt.queue_len() == 3
    # without a reset, a very different observation still gets the queued action
    leftover = rt.act(img_b, s_b)
    assert not leftover["inferred"] and rt.queue_len() == 2
    # what the network would say for the new observation
    rt.begin_episode()
    fresh = rt.act(img_b, s_b)
    assert fresh["inferred"] and rt.queue_len() == 3
    assert not np.allclose(leftover["action"], fresh["action"])
    # and a reset really starts over: same observation -> same first action
    rt.begin_episode()
    np.testing.assert_allclose(rt.act(img_b, s_b)["action"], fresh["action"], atol=1e-6)


def test_command_adds_the_fixed_gripper():
    rt = _rt(ChunkPolicy())
    rt.begin_episode()
    r = rt.act(IMG, np.arange(5, dtype=np.float32))
    assert r["command"].shape == (6,) and r["command"][-1] == pytest.approx(-0.16)
    np.testing.assert_allclose(r["command"][:5], np.arange(5), atol=1e-6)


def test_hold_runtime_commands_the_current_state():
    rt = hold_runtime(DS)
    assert rt.begin_episode() == 0
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5], np.float32)
    np.testing.assert_allclose(rt.act(IMG, s)["action"], s, atol=1e-6)


def test_bad_inputs_are_rejected():
    rt = _rt(ChunkPolicy())
    rt.begin_episode()
    with pytest.raises(ValueError):
        rt.act(IMG, np.zeros(6, np.float32))
    with pytest.raises(ValueError):
        rt.act(IMG.astype(np.float32), np.zeros(5, np.float32))


def _reset_calls(path: Path) -> list[str]:
    """Functions in which `<...>.policy.reset()` or `policy.reset()` is called."""
    tree = ast.parse(path.read_text())
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(fn):
            f = getattr(node, "func", None)
            if isinstance(node, ast.Call) and isinstance(f, ast.Attribute) and f.attr == "reset":
                v = f.value
                name = v.attr if isinstance(v, ast.Attribute) else getattr(v, "id", "")
                if name == "policy":
                    found.append(fn.name)
    return found


def test_policy_reset_is_called_only_in_begin_episode():
    """One entry point: no other code under train/ or scripts/sim/ resets the policy."""
    files = [*(ROOT / "train").glob("*.py"), *(ROOT / "scripts" / "sim").glob("*.py")]
    hits = [(f.name, fn) for f in files for fn in _reset_calls(f)]
    assert hits == [("policy_runtime.py", "begin_episode")], hits
