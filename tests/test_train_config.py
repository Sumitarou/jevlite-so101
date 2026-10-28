"""configs/train_act.yaml: dataset joints and fixed (left-out) joints agree with sim.yaml."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_joints_and_fixed_joints():
    ds = yaml.safe_load((ROOT / "configs" / "train_act.yaml").read_text())["dataset"]
    ik = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())["reach_check"]
    joints, fixed = ds["joints"], ds["fixed_joints"]
    assert not set(joints) & set(fixed)
    # the policy's joints are the IK joints; the gripper is sent as the demos' constant value
    assert joints == ik["ik_joints"]
    assert fixed == ik["fixed"]
