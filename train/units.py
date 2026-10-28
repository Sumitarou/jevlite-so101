"""The one place where the policy's output becomes a robot command.

The dataset and the policy use the 5 arm joints in rad (configs/train_act.yaml dataset.joints).
Commands for the arm are 6 joints: the arm joints plus dataset.fixed_joints (the gripper, which
never moves in the demos, -0.16 rad). Gazebo takes rad as is. The real SO-101 through LeRobot
takes degrees or -100..100 for the arm and 0..100 for the gripper; that conversion will be added
here too (after the arm is calibrated), so no other code handles grippers or units.
"""

from __future__ import annotations

import numpy as np

ARM6 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def sim_command(
    action: np.ndarray, joints: list[str], fixed: dict[str, float], names: list[str] = ARM6
) -> np.ndarray:
    """Policy action (len(joints),) in rad -> command for `names` in rad (Gazebo)."""
    action = np.asarray(action, np.float64).ravel()
    if action.shape != (len(joints),):
        raise ValueError(f"action has shape {action.shape}, expected ({len(joints)},)")
    if not np.all(np.isfinite(action)):
        raise ValueError("action is not finite")
    by_name = dict(zip(joints, action, strict=True))
    overlap = set(by_name) & set(fixed)
    if overlap:
        raise ValueError(f"joints both learned and fixed: {sorted(overlap)}")
    by_name.update({k: float(v) for k, v in fixed.items()})
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"no value for joints {missing}")
    return np.array([by_name[n] for n in names], np.float64)
