"""Generate push demonstrations in headless Gazebo and record each episode with rosbag2 (mcap).

    source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
      timeout --signal=INT --kill-after=30 1800 python3 scripts/sim/gen_demos.py --episodes 3

    ... --replay 7     # run the episode with seed 7 again
    ... --seeds 3 10 34 --engine bullet-featherstone   # given seeds, another physics engine

Runs with the system Python (rclpy), like check_m1.py. Settings: configs/sim.yaml `demo`.
Starts `ros2 launch so101_gazebo sim.launch.py` once in its own process group, then for each
episode (seed = seed_base + i, or the --replay seed):
  1. plan the episode (scripts/sim/demo_plan.py; open loop, joint commands at rate_hz)
  2. move the arm to the home pose (joint space), put the block at its start (set_pose)
  3. start `ros2 bag record` (own process group), publish the episode info on /demo/episode
  4. stream the commands on sim time (both controllers and /demo/action)
  5. judge success from /block/pose at the end of the settle phase, publish the result,
     stop the recorder, write episode.json next to the bag and a line to episodes.jsonl
All processes are stopped at the end (SIGINT, then SIGKILL) and checked to be gone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import yaml  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from ros_gz_interfaces.msg import Entity  # noqa: E402
from ros_gz_interfaces.srv import SetEntityPose  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float64MultiArray, String  # noqa: E402

from scripts.sim.demo_plan import Planner, judge, surface_z  # noqa: E402

LOG_DIR = ROOT / "outputs" / "sim"


class Runner(Node):
    def __init__(self, cfg: dict, arm_joints, grip_joints):
        super().__init__("demo_runner", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.cfg = cfg
        self.arm_joints, self.grip_joints = arm_joints, grip_joints
        self.joints: dict[str, float] = {}
        self.block: PoseStamped | None = None
        self.create_subscription(JointState, "/joint_states", self._on_joints, 50)
        self.create_subscription(PoseStamped, cfg["block"]["pose_topic"], self._on_block, 10)
        self.arm_pub = self.create_publisher(
            Float64MultiArray, "/arm_position_controller/commands", 10
        )
        self.grip_pub = self.create_publisher(
            Float64MultiArray, "/gripper_position_controller/commands", 10
        )
        self.action_pub = self.create_publisher(JointState, "/demo/action", 10)
        latched = QoSProfile(depth=4, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)  # fmt: skip
        self.episode_pub = self.create_publisher(String, "/demo/episode", latched)
        self.set_pose = self.create_client(SetEntityPose, "/world/table/set_pose")

    def _on_joints(self, msg: JointState):
        self.joints.update(zip(msg.name, msg.position, strict=False))

    def _on_block(self, msg: PoseStamped):
        self.block = msg

    def sim_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_for(self, sim_s: float, wall_limit: float = 60.0):
        end, wall_end = self.sim_now() + sim_s, time.monotonic() + wall_limit
        while self.sim_now() < end and time.monotonic() < wall_end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def spin_until(self, cond, wall_limit: float) -> bool:
        end = time.monotonic() + wall_limit
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
            if cond():
                return True
        return False

    def send(self, names, q):
        """Publish one command (6 joints in `names` order) to both controllers and /demo/action."""
        cmd = dict(zip(names, (float(v) for v in q), strict=True))
        self.arm_pub.publish(Float64MultiArray(data=[cmd[j] for j in self.arm_joints]))
        self.grip_pub.publish(Float64MultiArray(data=[cmd[j] for j in self.grip_joints]))
        msg = JointState(name=list(names), position=[cmd[j] for j in names])
        msg.header.stamp = self.get_clock().now().to_msg()
        self.action_pub.publish(msg)

    def stream(self, names, q, rate, on_tick=None, wall_limit=600.0) -> bool:
        """Send q[i] at sim time t0 + i / rate. on_tick(i) is called after each send."""
        t0 = self.sim_now()
        wall_end = time.monotonic() + wall_limit
        i = 0
        while i < len(q):
            if time.monotonic() > wall_end:
                return False
            if self.sim_now() >= t0 + i / rate:
                self.send(names, q[i])
                if on_tick:
                    on_tick(i)
                i += 1
            rclpy.spin_once(self, timeout_sec=0.002)
        return True

    def place_block(self, x, y, yaw, wall_limit=10.0) -> bool:
        t = self.cfg["table"]
        z = t["height"] + surface_z(self.cfg) + self.cfg["block"]["size"][2] / 2
        req = SetEntityPose.Request()
        req.entity = Entity(name=self.cfg["block"]["name"], type=Entity.MODEL)
        req.pose.position.x, req.pose.position.y, req.pose.position.z = x, y, z
        req.pose.orientation.z, req.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        fut = self.set_pose.call_async(req)
        end = time.monotonic() + wall_limit
        while not fut.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
        return fut.done() and fut.result() is not None and fut.result().success

    def block_pose_table(self):
        """Latest block pose (x, y, z above the table top, qx, qy, qz, qw)."""
        p = self.block.pose
        return (p.position.x, p.position.y, p.position.z - self.cfg["table"]["height"],
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)  # fmt: skip


def stop_group(proc: subprocess.Popen, timeout: float) -> dict:
    """SIGINT the process group, wait, SIGKILL if needed; report leftovers."""
    pgid = proc.pid
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
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return info
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
        info["sigkill"] = True
        time.sleep(1)
        os.killpg(pgid, 0)
        info["leftover"] = True
    except ProcessLookupError:
        pass
    return info


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def run_episode(
    node: Runner, planner: Planner, seed: int, index: int, out: Path, demo: dict
) -> dict:
    wall0 = time.monotonic()
    plan = planner.plan(seed)
    names, rate = plan.joint_names, plan.rate_hz
    rec = demo["record"]
    info = {"episode": index, **plan.summary(), "status": "started"}

    # 1. home (joint space, from wherever the arm is) and the block at its start
    cur = np.array([node.joints.get(j, 0.0) for j in names])
    n = max(1, int(round(demo["home_move_s"] * rate)))
    s = (np.arange(1, n + 1) / n)[:, None]
    node.stream(names, cur + s * (plan.q[0] - cur), rate)
    x, y, yaw = plan.block_start
    if not node.place_block(x, y, yaw):
        raise RuntimeError("set_pose failed")
    node.spin_for(demo["reset_settle_s"])
    bp = node.block_pose_table()
    if math.hypot(bp[0] - x, bp[1] - y) > 0.003:
        raise RuntimeError(f"block not at its start after set_pose: {bp[:2]} vs {(x, y)}")

    # 2. recorder
    bag = out / f"ep_{index:04d}_seed{seed}"
    cmd = ["ros2", "bag", "record", "-o", str(bag), "-s", rec["storage"], "--use-sim-time",
           "--storage-preset-profile", rec["preset"], "--topics", *rec["topics"]]  # fmt: skip
    log = (out / f"ep_{index:04d}_record.log").open("w")
    recorder = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        ok = node.spin_until(
            lambda: (
                node.action_pub.get_subscription_count() > 0
                and node.count_subscribers(node.cfg["camera"]["topic"]) > 0
            ),
            30.0,
        )
        if not ok:
            raise RuntimeError("the recorder did not subscribe in time")
        node.spin_for(0.3)
        t_start = node.sim_now()
        info["sim_start"] = round(t_start, 3)
        node.episode_pub.publish(String(data=json.dumps(info)))

        # 3. stream, snapshot the block at the end of the settle phase
        ph = {name: (a, b) for name, a, b in plan.phases}
        phase_t = {}
        snap = {}

        def on_tick(i):
            for name, (a, _) in ph.items():
                if i == a:
                    phase_t[name] = round(node.sim_now() - t_start, 3)
            if i == ph["settle"][1] - 1 and node.block is not None:
                snap["pose"] = node.block_pose_table()

        limit = demo["episode_timeout_s"] - (time.monotonic() - wall0)
        w_stream, s_stream = time.monotonic(), node.sim_now()
        if not node.stream(names, plan.q, rate, on_tick, wall_limit=limit):
            raise RuntimeError("episode timed out")
        # real time factor while the commands were streamed (sim time / wall time)
        info["rtf_stream"] = round(
            (node.sim_now() - s_stream) / max(time.monotonic() - w_stream, 1e-9), 3
        )
        node.spin_for(demo["tail_s"])
        result = judge(node.cfg, snap["pose"]) if "pose" in snap else {"success": False}
        info.update(status="done", result=result, phase_start_s=phase_t,
                    sim_duration_s=round(node.sim_now() - t_start, 3),
                    final_block_pose=[round(v, 5) for v in snap.get("pose", [])])  # fmt: skip
        node.episode_pub.publish(String(data=json.dumps(info)))
        node.spin_for(0.3)
    finally:
        info["recorder_stop"] = stop_group(recorder, 20.0)
        log.close()
    info["wall_duration_s"] = round(time.monotonic() - wall0, 2)
    info["bag"] = bag.name
    info["bag_bytes"] = dir_size(bag) if bag.exists() else 0
    (bag / "episode.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="push demonstrations in headless Gazebo")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, help="seed_base (default: demo.seed_base)")
    ap.add_argument("--replay", type=int, help="run only the episode with this seed")
    ap.add_argument("--seeds", type=int, nargs="+", help="run the episodes with these seeds")
    ap.add_argument("--engine", help="physics engine (default: configs/sim.yaml physics.engine)")
    ap.add_argument("--out", help="output directory (default: demo.record.out_dir/<time>)")
    ap.add_argument("--preset", help="mcap storage preset (default: demo.record.preset)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    ctrl = yaml.safe_load((ROOT / "configs" / "sim_controllers.yaml").read_text())
    demo = cfg["demo"]
    if args.preset:
        demo["record"]["preset"] = args.preset
    base_seed = demo["seed_base"] if args.seed is None else args.seed
    if args.replay is not None:
        seeds = [args.replay]
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = [base_seed + i for i in range(args.episodes)]
    engine = args.engine or cfg["physics"].get("engine", "dartsim")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / demo["record"]["out_dir"] / stamp
    out.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"out": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
                     "physics_engine": engine, "seeds": seeds, "episodes": []}  # fmt: skip
    t_wall = time.monotonic()
    launch_log = (out / "launch.log").open("w")
    proc = node = None
    code = 1
    try:
        proc = subprocess.Popen(["ros2", "launch", "so101_gazebo", "sim.launch.py",
                                 f"physics_engine:={engine}"], stdout=launch_log,
                                stderr=subprocess.STDOUT, start_new_session=True)  # fmt: skip
        rclpy.init()
        node = Runner(cfg, ctrl["arm_position_controller"]["ros__parameters"]["joints"],
                      ctrl["gripper_position_controller"]["ros__parameters"]["joints"])  # fmt: skip
        ready = node.spin_until(
            lambda: node.joints and node.block is not None
            and node.arm_pub.get_subscription_count() > 0
            and node.grip_pub.get_subscription_count() > 0
            and node.set_pose.service_is_ready(),
            cfg["check"]["startup_timeout_s"],
        )  # fmt: skip
        summary["startup_wall_s"] = round(time.monotonic() - t_wall, 2)
        if not ready:
            raise RuntimeError("simulation not ready in time")
        planner = Planner(cfg)
        with (out / "episodes.jsonl").open("a") as jl:
            for i, seed in enumerate(seeds):
                info = run_episode(node, planner, seed, i, out, demo)
                jl.write(json.dumps(info) + "\n")
                jl.flush()
                summary["episodes"].append(info)
                r = info.get("result", {})
                print(f"episode {i} seed {seed}: success={r.get('success')} "
                      f"error={r.get('error_m')} m, sim {info.get('sim_duration_s')} s, "
                      f"wall {info['wall_duration_s']} s, {info['bag_bytes'] / 1e6:.1f} MB",
                      flush=True)  # fmt: skip
        code = 0
    except Exception as e:  # recorded; everything is still stopped below
        summary["error"] = f"{type(e).__name__}: {e}"
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if proc is not None:
            summary["shutdown"] = stop_group(proc, cfg["check"]["shutdown_timeout_s"])
        launch_log.close()
        eps = summary["episodes"]
        if eps:
            ok = [e for e in eps if e.get("result", {}).get("success")]
            summary["success_rate"] = round(len(ok) / len(eps), 3)
            summary["mean_sim_s"] = round(float(np.mean([e["sim_duration_s"] for e in eps])), 2)
            summary["mean_wall_s"] = round(float(np.mean([e["wall_duration_s"] for e in eps])), 2)
            summary["mean_bag_mb"] = round(float(np.mean([e["bag_bytes"] for e in eps])) / 1e6, 1)
        summary["total_wall_s"] = round(time.monotonic() - t_wall, 1)
        brief = {k: v for k, v in summary.items() if k != "episodes"}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(brief, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
