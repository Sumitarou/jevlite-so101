"""M1 check: SO-101 in headless Gazebo reaches joint targets; one overhead camera frame is saved.

    source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
      timeout 180 python3 scripts/sim/check_m1.py

Runs with the system Python (rclpy), not uv. Starts `ros2 launch so101_gazebo sim.launch.py`
in its own process group, sends the targets in configs/sim.yaml to the position controllers,
measures the step response of every joint, saves outputs/sim/m1_camera.png and writes
outputs/sim/m1_result.json. The launch is always stopped (SIGINT, then SIGKILL) and the
script checks that none of its processes are left. Exit code 0 only if every joint reached
its target within the tolerance and a camera frame was saved.

`--pose NAME` sends `check.poses.NAME` instead of `check.targets` and writes
m1_camera_NAME.png / m1_result_NAME.json / m1_launch_NAME.log (e.g. `--pose reach_white`).
`--engine ENGINE` overrides configs/sim.yaml physics.engine (launch argument physics_engine) and
adds `_ENGINE` to the output names (e.g. `--engine bullet-featherstone`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "sim"
ERR_CHECKPOINTS_S = [0.05, 0.1, 0.2, 0.5, 1.0]  # |target - q| at these sim times after the command


class Probe(Node):
    def __init__(self, camera_topic: str):
        super().__init__("m1_check", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.samples: list[tuple[float, float, dict[str, float]]] = []  # (wall, sim, positions)
        self.clock_samples: list[tuple[float, float]] = []
        self.image: Image | None = None
        self.want_image = False
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 100)
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(Image, camera_topic, self._on_image, 1)
        self.arm_pub = self.create_publisher(
            Float64MultiArray, "/arm_position_controller/commands", 10
        )
        self.grip_pub = self.create_publisher(
            Float64MultiArray, "/gripper_position_controller/commands", 10
        )

    # callback names must not shadow rclpy.Node attributes (Node has its own _clock)
    def _on_joint_states(self, msg: JointState):
        sim = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.samples.append(
            (time.monotonic(), sim, dict(zip(msg.name, msg.position, strict=False)))
        )

    def _on_clock(self, msg: Clock):
        self.clock_samples.append((time.monotonic(), msg.clock.sec + msg.clock.nanosec * 1e-9))

    def _on_image(self, msg: Image):
        if self.want_image:
            self.image = msg
            self.want_image = False


def spin_until(node: Node, cond, timeout: float, step: float = 0.01) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=step)
        if cond():
            return True
    return False


def get_param(node: Node, target: str, name: str):
    out = subprocess.run(["ros2", "param", "get", target, name], capture_output=True, text=True,
                         timeout=15)  # fmt: skip
    m = re.search(r"value is: (.+)", out.stdout)
    return m.group(1).strip() if m else None


def gz_rtf() -> float | None:
    try:
        out = subprocess.run(["gz", "topic", "-e", "-t", "/world/table/stats", "-n", "1"],
                             capture_output=True, text=True, timeout=10).stdout  # fmt: skip
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"real_time_factor:\s*([0-9.eE+-]+)", out)
    return float(m.group(1)) if m else None


def step_response(samples, t0_wall, t0_sim, q0, target, frac, tol):
    delta = target - q0
    rows = [(w - t0_wall, s - t0_sim, q) for w, s, q in samples if w >= t0_wall]
    res = {"q0": q0, "target": target, "delta": delta}
    t95 = next(((sw, ss) for sw, ss, q in rows if abs(q - q0) >= frac * abs(delta)), None)
    res["t95_sim_s"], res["t95_wall_s"] = (t95[1], t95[0]) if t95 else (None, None)
    tin = next(((sw, ss) for sw, ss, q in rows if abs(q - target) <= tol), None)
    res["t_within_tol_sim_s"], res["t_within_tol_wall_s"] = (
        (tin[1], tin[0]) if tin else (None, None)
    )
    sign = 1.0 if delta >= 0 else -1.0
    over = max([(q - target) * sign for _, _, q in rows] + [0.0])
    res["overshoot_rad"] = over
    res["overshoot_pct"] = 100.0 * over / abs(delta) if delta else 0.0
    res["err_at_sim_s"] = {}
    for c in ERR_CHECKPOINTS_S:
        q = next((q for _, ss, q in rows if ss >= c), None)
        res["err_at_sim_s"][str(c)] = None if q is None else abs(target - q)
    return res


def save_image(msg: Image, path: Path) -> dict:
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(msg.encoding)
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    arr = arr.reshape(msg.height, msg.step)[:, : msg.width * channels]
    arr = arr.reshape(msg.height, msg.width, channels)
    if msg.encoding.startswith("bgr"):
        arr = arr[..., [2, 1, 0]]
    PILImage.fromarray(arr[..., :3] if channels >= 3 else arr[..., 0]).save(path)
    return {"width": msg.width, "height": msg.height, "encoding": msg.encoding,
            "mean": round(float(arr.mean()), 1), "std": round(float(arr.std()), 1)}  # fmt: skip


def stop(proc: subprocess.Popen, timeout: float) -> dict:
    pgid = proc.pid  # started with start_new_session=True, so pgid == pid
    info = {"sigint": False, "sigkill": False, "leftover": False}
    try:
        os.killpg(pgid, signal.SIGINT)
        info["sigint"] = True
        proc.wait(timeout=timeout)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        info["sigkill"] = True
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            os.killpg(pgid, 0)  # any process of the group still alive?
        except ProcessLookupError:
            return info
        time.sleep(0.2)
    os.killpg(pgid, signal.SIGKILL)
    info["sigkill"] = True
    time.sleep(1)
    try:
        os.killpg(pgid, 0)
        info["leftover"] = True
    except ProcessLookupError:
        pass
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M1 check in headless Gazebo")
    ap.add_argument("--pose", help="name under check.poses in configs/sim.yaml")
    ap.add_argument("--engine", help="physics engine (default: configs/sim.yaml physics.engine)")
    args = ap.parse_args(argv)
    sfx = (f"_{args.pose}" if args.pose else "") + (f"_{args.engine}" if args.engine else "")
    sim = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    ctrl = yaml.safe_load((ROOT / "configs" / "sim_controllers.yaml").read_text())
    chk = sim["check"]
    arm_joints = ctrl["arm_position_controller"]["ros__parameters"]["joints"]
    grip_joints = ctrl["gripper_position_controller"]["ros__parameters"]["joints"]
    gz_cfg = ctrl["gz_ros2_control"]["ros__parameters"]
    OUT.mkdir(parents=True, exist_ok=True)
    log = (OUT / f"m1_launch{sfx}.log").open("w")
    engine = args.engine or sim["physics"].get("engine", "dartsim")
    result: dict = {"ok": False, "pose": args.pose or "targets", "physics_engine": engine}
    t_start = time.monotonic()
    proc = node = None
    try:
        # everything after the launch starts is inside try, so `finally` always stops it
        proc = subprocess.Popen(["ros2", "launch", "so101_gazebo", "sim.launch.py",
                                 f"physics_engine:={engine}"], stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)  # fmt: skip
        rclpy.init()
        node = Probe(sim["camera"]["topic"])
        ready = spin_until(
            node,
            lambda: (
                node.samples
                and node.arm_pub.get_subscription_count() > 0
                and node.grip_pub.get_subscription_count() > 0
            ),
            chk["startup_timeout_s"],
        )
        result["startup_wall_s"] = round(time.monotonic() - t_start, 2)
        if not ready:
            result["error"] = "controllers or /joint_states not ready in time"
            return 1
        spin_until(node, lambda: False, 1.0)  # let the initial state settle
        q0 = dict(node.samples[-1][2])
        result["initial_positions"] = q0
        targets = chk["poses"][args.pose] if args.pose else chk["targets"]
        t0_wall = time.monotonic()
        t0_sim = node.samples[-1][1]
        node.arm_pub.publish(Float64MultiArray(data=[float(targets[j]) for j in arm_joints]))
        node.grip_pub.publish(Float64MultiArray(data=[float(targets[j]) for j in grip_joints]))

        def all_within():
            q = node.samples[-1][2]
            return all(abs(q[j] - targets[j]) <= chk["tolerance"] for j in targets)

        reached = spin_until(node, all_within, chk["settle_timeout_s"])
        spin_until(node, lambda: False, chk["hold_s"])
        hold = [s for s in node.samples if s[0] >= time.monotonic() - chk["hold_s"]]
        result["reached_all"] = reached
        result["joints"] = {}
        for j in targets:
            r = step_response(
                [(w, s, q[j]) for w, s, q in node.samples], t0_wall, t0_sim, q0[j], targets[j],
                chk["response_fraction"], chk["tolerance"],
            )  # fmt: skip
            # the hold window starts when every joint is within tolerance, so it still contains
            # the end of the approach; err_end_rad (last sample) is the steady-state error
            r["final_err_rad"] = float(np.mean([abs(q[j] - targets[j]) for _, _, q in hold]))
            r["max_err_during_hold_rad"] = float(max(abs(q[j] - targets[j]) for _, _, q in hold))
            r["err_end_rad"] = abs(node.samples[-1][2][j] - targets[j])
            r["hold_sim_s"] = hold[-1][1] - hold[0][1]
            result["joints"][j] = r

        node.want_image = True
        got = spin_until(node, lambda: node.image is not None, 10.0)
        result["camera"] = save_image(node.image, OUT / f"m1_camera{sfx}.png") if got else None

        c = node.clock_samples
        if len(c) >= 2:
            result["rtf_from_clock"] = round((c[-1][1] - c[0][1]) / (c[-1][0] - c[0][0]), 3)
        result["rtf_gz_stats"] = gz_rtf()
        cm_node = "/controller_manager"
        # the GazeboSimROS2ControlPlugin's node is named gz_ros_control (not gz_ros2_control)
        gz_nodes = [n for n in node.get_node_names_and_namespaces() if n[0] == "gz_ros_control"]
        gz_node = (gz_nodes[0][1].rstrip("/") + "/" + gz_nodes[0][0]) if gz_nodes else None
        result["parameters"] = {
            "position_proportional_gain": get_param(node, gz_node, "position_proportional_gain")
            if gz_node else None,
            "hold_joints": get_param(node, gz_node, "hold_joints") if gz_node else None,
            "gz_ros2_control_node": gz_node,
            "controller_update_rate_hz": get_param(node, cm_node, "update_rate"),
            "physics_max_step_size_s": sim["physics"]["max_step_size"],
            "configured_gain": gz_cfg["position_proportional_gain"],
        }  # fmt: skip
        result["ok"] = bool(reached and result["camera"])
        return 0 if result["ok"] else 1
    except Exception as e:  # recorded in the result; the launch is still stopped below
        result["error"] = f"{type(e).__name__}: {e}"
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if proc is not None:
            result["shutdown"] = stop(proc, chk["shutdown_timeout_s"])
        result["total_wall_s"] = round(time.monotonic() - t_start, 2)
        log.close()
        (OUT / f"m1_result{sfx}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
