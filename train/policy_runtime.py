"""A loaded policy with its pre/post processors, stepped one observation at a time.

begin_episode() is the only place that calls policy.reset(): it empties ACT's action queue (or
the temporal ensembler) so a new episode never continues the previous episode's chunk. The
safety gate's on_resume (G7: leaving STOP) is to call the same begin_episode(), so that a policy
resumed after a stop also starts from a fresh chunk.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from units import sim_command  # noqa: E402


class PolicyRuntime:
    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        dataset_cfg: dict,
        device: str = "cpu",
    ):
        self.policy, self.pre, self.post = policy, preprocessor, postprocessor
        self.joints = list(dataset_cfg["joints"])
        self.fixed = dict(dataset_cfg.get("fixed_joints", {}))
        self.image_key, self.task = dataset_cfg["image_key"], dataset_cfg["task"]
        self.device = device
        self.episodes = 0

    @classmethod
    def from_checkpoint(
        cls,
        pretrained: Path,
        dataset_cfg: dict,
        device: str = "cuda",
        n_action_steps: int | None = None,
        temporal_ensemble_coeff: float | None = None,
    ) -> PolicyRuntime:
        """Load pretrained_model/ (weights, config, normalizer stats). The execution settings
        (n_action_steps, temporal ensembling) can be changed without retraining."""
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(pretrained)
        cfg.device = device
        if temporal_ensemble_coeff is not None:
            cfg.temporal_ensemble_coeff = temporal_ensemble_coeff
            cfg.n_action_steps = 1
        elif n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        policy = get_policy_class(cfg.type).from_pretrained(pretrained, config=cfg)
        policy.to(device).eval()
        pre, post = make_pre_post_processors(
            cfg,
            pretrained_path=str(pretrained),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        return cls(policy, pre, post, dataset_cfg, device)

    def queue_len(self) -> int:
        q = getattr(self.policy, "_action_queue", None)
        return len(q) if q is not None else 0

    def begin_episode(self) -> int:
        """Start a new episode (or resume after a stop): forget all queued actions."""
        self.policy.reset()
        self.episodes += 1
        return self.queue_len()

    def act(self, image: np.ndarray, state: np.ndarray) -> dict:
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        state = np.asarray(state, np.float32).ravel()
        if state.shape != (len(self.joints),):
            raise ValueError(f"state has shape {state.shape}, expected ({len(self.joints)},)")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be HxWx3 uint8, got {image.dtype} {image.shape}")
        inferred = (
            getattr(self.policy.config, "temporal_ensemble_coeff", None) is not None
            or self.queue_len() == 0
        )
        t = time.perf_counter()
        obs = {"observation.state": state.copy(), self.image_key: np.ascontiguousarray(image)}
        obs = prepare_observation_for_inference(obs, torch.device(self.device), self.task)
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(obs)))
        action = action.detach().to("cpu").numpy().astype(np.float64).reshape(-1)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t) * 1e3
        return {
            "action": action,
            "command": sim_command(action, self.joints, self.fixed),
            "infer_ms": infer_ms,
            "inferred": inferred,
            "queue_len": self.queue_len(),
        }


class HoldPolicy:
    """Stand-in policy: always commands the current joint state (the arm stays where it is).

    Used for the evaluator's own tests (an episode that must be judged a failure). Same queue
    interface as ACT so begin_episode() and queue_len() work unchanged."""

    def __init__(self):
        from collections import deque

        self._action_queue = deque([], maxlen=1)
        self.config = type("Cfg", (), {"temporal_ensemble_coeff": None})()

    def reset(self):
        self._action_queue.clear()

    def select_action(self, batch):
        return batch["observation.state"].clone()


def identity(x):
    return x


def hold_runtime(dataset_cfg: dict) -> PolicyRuntime:
    return PolicyRuntime(HoldPolicy(), identity, identity, dataset_cfg, "cpu")
