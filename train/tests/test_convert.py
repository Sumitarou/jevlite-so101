"""train/convert_bags.py and train/bagio.py on a small synthetic bag (no ROS, no GPU).

uv run --project train pytest train/tests
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

import bagio
import convert_bags

ROOT = Path(__file__).resolve().parents[2]
TS = get_typestore(Stores.ROS2_JAZZY)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JS_ORDER = ["elbow_flex", "gripper", "shoulder_lift", "shoulder_pan", "wrist_flex", "wrist_roll"]
MS = 1_000_000
H, W = 64, 96


def _header(t_ns):
    T = TS.types
    return T["std_msgs/msg/Header"](
        stamp=T["builtin_interfaces/msg/Time"](sec=t_ns // 10**9, nanosec=t_ns % 10**9),
        frame_id="",
    )


def _joint_state(t_ns, names, q):
    return TS.types["sensor_msgs/msg/JointState"](
        header=_header(t_ns),
        name=list(names),
        position=np.asarray(q, np.float64),
        velocity=np.zeros(0),
        effort=np.zeros(0),
    )


def _image(t_ns, value):
    data = np.full((H, W, 3), value, np.uint8)
    return TS.types["sensor_msgs/msg/Image"](
        header=_header(t_ns), height=H, width=W, encoding="rgb8", is_bigendian=0,
        step=W * 3, data=data.reshape(-1),
    )  # fmt: skip


def write_bag(d: Path, seed: int, n_ticks=12, js_start_ms=1000 + 40):
    """Actions every 1/30 s from t=1 s; joint_states (shuffled names) at 100 Hz from js_start_ms."""
    d.mkdir(parents=True)
    bag = d / "bag"
    t_act = [10**9 + round(k * 1e9 / 30) for k in range(n_ticks)]
    with Writer(bag, version=9, storage_plugin=StoragePlugin.MCAP) as w:
        ca = w.add_connection("/demo/action", "sensor_msgs/msg/JointState", typestore=TS)
        cs = w.add_connection("/joint_states", "sensor_msgs/msg/JointState", typestore=TS)
        ci = w.add_connection("/overhead_camera/image", "sensor_msgs/msg/Image", typestore=TS)
        msgs = []
        for k, t in enumerate(t_act):
            msgs.append((t, ca, _joint_state(t, JOINTS, [k * 0.01 * (j + 1) for j in range(6)])))
        for t in range(js_start_ms * MS, t_act[-1] + 50 * MS, 10 * MS):
            q = {n: (t / 1e9) * (j + 1) for j, n in enumerate(JOINTS)}  # q_j(t) = t * (j+1)
            msgs.append((t, cs, _joint_state(t, JS_ORDER, [q[n] for n in JS_ORDER])))
        for k in range(n_ticks + 1):  # images 5 ms before each tick
            t = t_act[0] - 5 * MS + round(k * 1e9 / 30)
            msgs.append((t, ci, _image(t, 10 * k)))
        for t, c, m in sorted(msgs, key=lambda x: x[0]):
            w.write(c, t, TS.serialize_cdr(m, c.msgtype))
    # the bag folder is what gen_demos.py writes (episode.json next to the mcap)
    (bag / "episode.json").write_text(json.dumps({"seed": seed, "phases": []}))
    return bag, t_act


@pytest.fixture
def demo_src(tmp_path):
    src = tmp_path / "demos"
    rows = []
    for i, ok in enumerate([True, False, True]):
        bag, _ = write_bag(src / f"tmp{i}", seed=i)
        bag.rename(src / f"ep_{i:04d}_seed{i}")
        (src / f"tmp{i}").rmdir()
        rows.append({"bag": f"ep_{i:04d}_seed{i}", "result": {"success": ok}})
    (src / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return src


def test_read_episode_orders_joints_and_decodes_images(demo_src):
    cfg = yaml.safe_load((ROOT / "configs" / "train_act.yaml").read_text())["dataset"]
    e = bagio.read_episode(demo_src / "ep_0000_seed0", cfg["topics"], JOINTS)
    assert e.action.shape == (12, 6) and e.state.shape[1] == 6
    t = e.t_state[0] / 1e9
    np.testing.assert_allclose(e.state[0], [t * (j + 1) for j in range(6)])
    assert e.images[3].shape == (H, W, 3) and int(e.images[3][0, 0, 0]) == 30


def test_check_ticks():
    t = np.array([0, 33, 67, 100]) * MS
    assert convert_bags.check_ticks(t, 30, 0.002, "x") < 0.001
    with pytest.raises(ValueError):
        convert_bags.check_ticks(np.array([0, 40]) * MS, 30, 0.002, "x")


def test_convert_end_to_end(demo_src, tmp_path, monkeypatch):
    root = tmp_path / "ds"
    monkeypatch.setattr(
        sys, "argv", ["convert_bags.py", "--src", str(demo_src), "--root", str(root)]
    )
    assert convert_bags.main() == 0
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/test", root=root, return_uint8=True)
    side = [json.loads(x) for x in (root / "jevlite_episodes.jsonl").read_text().splitlines()]
    # the failed demo (seed 1) is skipped; ticks before the first joint_states (t < 1.040 s:
    # ticks 0 and 1 at 1.000 and 1.033 s) are dropped
    assert [r["seed"] for r in side] == [0, 2]
    assert ds.meta.total_episodes == 2 and ds.meta.total_frames == 2 * 10
    assert all(r["first_action_tick"] == 2 and r["align"]["dropped_no_state"] == 2 for r in side)
    it = ds[0]
    assert it["task"] == "push the yellow block into the blue frame"
    # dataset.joints: the 5 arm joints (the constant gripper is left out)
    np.testing.assert_allclose(it["action"].numpy(), [0.02 * (j + 1) for j in range(5)], atol=1e-6)
    t2 = (10**9 + round(2e9 / 30)) / 1e9
    np.testing.assert_allclose(
        it["observation.state"].numpy(), [t2 * (j + 1) for j in range(5)], atol=1e-5
    )
    assert ds.meta.features["action"]["names"] == JOINTS[:5]
    img = it["observation.images.overhead"].numpy()
    assert img.shape == (3, H, W)
    assert abs(int(np.median(img)) - 20) <= 3  # image 2 (value 20), 5 ms before tick 2
    conv = json.loads((root / "jevlite_convert.json").read_text())
    assert conv["image_age_ms_max"] == pytest.approx(5.0, abs=0.01)


def test_convert_refuses_existing_root(demo_src, tmp_path, monkeypatch):
    root = tmp_path / "ds"
    root.mkdir()
    monkeypatch.setattr(
        sys, "argv", ["convert_bags.py", "--src", str(demo_src), "--root", str(root)]
    )
    assert convert_bags.main() == 2
