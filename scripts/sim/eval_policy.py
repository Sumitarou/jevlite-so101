"""Evaluate a trained policy in headless Gazebo: closed loop at sim-time 30 Hz on new seeds.

    source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
      timeout --signal=INT --kill-after=30 3600 python3 scripts/sim/eval_policy.py \
        --checkpoint models/act/<run>/lerobot/checkpoints/030000 --seeds 100 101 102 103 104

    ... --checkpoint ... --n-action-steps 30          # execution settings, no retraining
    ... --fake-policy hold --seeds 100 --duration 3   # evaluator self-test (must be a failure)

Runs with the system Python (rclpy), like gen_demos.py. The policy runs in a second process in
the train/ env (train/policy_server.py, started here, own process group), reached through a local
socket (scripts/sim/policy_link.py). Settings: configs/sim.yaml `eval` (seeds, length) and `demo`
(rate, home pose, recording). For each seed:
  1. the block start is drawn exactly as for the demos (demo_plan.Planner); the arm goes home
  2. `ros2 bag record` starts (same topics as the demos)
  3. the policy is reset, then run_closed_loop (scripts/sim/eval_loop.py) for eval.duration_s:
     newest image + joint state -> server -> 6-joint command (train/units.py adds the gripper)
  4. after eval.tail_s the block pose is judged with demo_plan.judge (same rule as the demos)
Output: outputs/sim/<stamp>_m3d_eval_<name>/ with results.jsonl, summary.json, bags/, and
mp4s of one success and one failure. All processes are stopped and checked to be gone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import yaml  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from scripts.sim.demo_plan import ARM, Planner, judge, push_geometry  # noqa: E402
from scripts.sim.eval_loop import (  # noqa: E402
    Obs,
    pick_examples,
    run_closed_loop,
    summarize_episodes,
)
from scripts.sim.gen_demos import Runner, dir_size, stop_group  # noqa: E402
from scripts.sim.outputs import new_run_dir  # noqa: E402
from scripts.sim.policy_link import KEY_ENV, PolicyClient, default_address, new_key  # noqa: E402


def image_to_array(msg: Image) -> np.ndarray:
    if msg.encoding != "rgb8":
        raise ValueError(f"image encoding {msg.encoding} (need rgb8)")
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    return a[:, : msg.width * 3].reshape(msg.height, msg.width, 3)


class EvalNode(Runner):
    """gen_demos' Runner plus the camera, as the environment of eval_loop.run_closed_loop."""

    def __init__(self, cfg, arm_joints, grip_joints):
        super().__init__(cfg, arm_joints, grip_joints)
        self.image: Image | None = None
        self.create_subscription(Image, cfg["camera"]["topic"], self._on_image, 5)

    def _on_image(self, msg: Image):
        self.image = msg

    def _drain(self, n: int = 200):
        for _ in range(n):  # handle what is queued (also when a tick is already late)
            rclpy.spin_once(self, timeout_sec=0)

    def wait_until(self, t: float, wall_limit: float = 30.0) -> bool:
        self._drain()
        end = time.monotonic() + wall_limit
        while self.sim_now() < t:
            if time.monotonic() > end:
                return False
            rclpy.spin_once(self, timeout_sec=0.002)
        return True

    def latest_obs(self) -> Obs | None:
        if self.image is None or not self.joints:
            return None
        s = self.image.header.stamp
        return Obs(image_to_array(self.image), s.sec + s.nanosec * 1e-9, dict(self.joints))


def start_server(
    args, ecfg: dict, out: Path, address: str, key: str, log_name: str = "policy_server.log"
) -> subprocess.Popen:
    cmd = ["uv", "run", "--project", str(ROOT / "train"), "python",
           str(ROOT / "train" / "policy_server.py"), "--address", address]  # fmt: skip
    if args.fake_policy:
        cmd += ["--fake", args.fake_policy]
    else:
        cmd += ["--checkpoint", str(args.checkpoint), "--device", args.device or ecfg["device"]]
        if args.n_action_steps:
            cmd += ["--n-action-steps", str(args.n_action_steps)]
        if args.temporal_ensemble_coeff is not None:
            cmd += ["--temporal-ensemble-coeff", str(args.temporal_ensemble_coeff)]
    log = (out / log_name).open("w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, KEY_ENV: key}, cwd=ROOT)  # fmt: skip


def wait_ready(proc: subprocess.Popen, log: Path, timeout_s: float) -> dict:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        for line in log.read_text().splitlines():
            if line.startswith("READY "):
                return json.loads(line[len("READY ") :])
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited ({proc.returncode}); see {log}")
        time.sleep(0.5)
    raise TimeoutError("policy server not ready in time")


def run_episode(node, planner, client, server, seed, index, out, cfg, args, start=None,
                on_tick=None, extra=None, tag=None, meta=None, gate=None) -> dict:  # fmt: skip
    """One episode. The block start comes from the seed (as for the demos) or, with `start`
    (x, y, yaw), is given directly (M3e; seed may then be None). on_tick / extra / gate go to
    run_closed_loop; `tag` names the bag (default seed<seed>); `meta` is added to the info."""
    demo, ecfg = cfg["demo"], cfg["eval"]
    wall0 = time.monotonic()
    if start is None:
        plan = planner.plan(seed)  # only the block start and the home pose are used
        names, q_first = plan.joint_names, plan.q[0]
        block_start, target = plan.block_start, plan.target
    else:  # same home pose definition: the first command of the plan for this start
        names, q_first = ARM + ["gripper"], planner.first_command(start)
        block_start, target = tuple(float(v) for v in start), push_geometry(cfg, start)["target"]
    rate = float(demo["rate_hz"])
    info = {"episode": index, "seed": seed, **(meta or {}),
            "block_start": [round(v, 5) for v in block_start],
            "target": [float(v) for v in target], "policy": server["policy"],
            "status": "started"}  # fmt: skip

    cur = np.array([node.joints.get(j, 0.0) for j in names])
    n = max(1, int(round(demo["home_move_s"] * rate)))
    s = (np.arange(1, n + 1) / n)[:, None]
    node.stream(names, cur + s * (q_first - cur), rate)
    x, y, yaw = block_start
    if not node.place_block(x, y, yaw):
        raise RuntimeError("set_pose failed")
    node.spin_for(demo["reset_settle_s"])
    bp = node.block_pose_table()
    if math.hypot(bp[0] - x, bp[1] - y) > 0.003:
        raise RuntimeError(f"block not at its start after set_pose: {bp[:2]} vs {(x, y)}")

    rec = demo["record"]
    bag = out / "bags" / f"ep_{index:04d}_{tag or f'seed{seed}'}"
    cmd = ["ros2", "bag", "record", "-o", str(bag), "-s", rec["storage"], "--use-sim-time",
           "--storage-preset-profile", rec["preset"], "--topics", *rec["topics"]]  # fmt: skip
    log = (out / "bags" / f"ep_{index:04d}_record.log").open("w")
    recorder = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        ok = node.spin_until(
            lambda: (
                node.action_pub.get_subscription_count() > 0
                and node.count_subscribers(cfg["camera"]["topic"]) > 1  # us + the recorder
            ),
            30.0,
        )
        if not ok:
            raise RuntimeError("the recorder did not subscribe in time")
        node.spin_for(0.3)
        t_start = node.sim_now()
        info["sim_start"] = round(t_start, 3)
        node.episode_pub.publish(String(data=json.dumps(info)))
        w = time.monotonic()
        loop = run_closed_loop(node, client, server["joints"], rate,
                               args.duration or ecfg["duration_s"], on_tick=on_tick,
                               extra=extra, gate=gate)  # fmt: skip
        info["rtf_loop"] = round((node.sim_now() - t_start) / max(time.monotonic() - w, 1e-9), 3)
        node.spin_for(ecfg["tail_s"])
        pose = node.block_pose_table()
        result = judge(cfg, pose)
        info.update(status="done", result=result, loop=loop["summary"],
                    sim_duration_s=round(node.sim_now() - t_start, 3),
                    final_block_pose=[round(v, 5) for v in pose])  # fmt: skip
        node.episode_pub.publish(String(data=json.dumps(info)))
        node.spin_for(0.3)
    finally:
        info["recorder_stop"] = stop_group(recorder, 20.0)
        log.close()
    info["wall_duration_s"] = round(time.monotonic() - wall0, 2)
    info["bag"] = bag.name
    info["bag_bytes"] = dir_size(bag) if bag.exists() else 0
    if bag.exists():
        (bag / "episode.json").write_text(json.dumps(info, indent=2) + "\n")
        (bag / "ticks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in loop["ticks"]))
    info["_ticks"] = loop["ticks"]
    return info


def bag_to_mp4(bag: Path, path: Path, topic: str, fps: int, label: str, overlay=None) -> int:
    """Write the camera images of a bag as an mp4 with a small caption. Returns the frame count.
    overlay(img_bgr, stamp_s) may draw more on each frame (M4a: the gate state band)."""
    import cv2
    import rosbag2_py
    from rclpy.serialization import deserialize_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))  # fmt: skip
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    writer, n, t0 = None, 0, None
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, Image)
        img = cv2.cvtColor(image_to_array(msg), cv2.COLOR_RGB2BGR)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        t0 = t if t0 is None else t0
        cv2.putText(img, f"{label}  t={t - t0:5.2f} s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip
        if overlay is not None:
            overlay(img, t)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(img)
        n += 1
    if writer is not None:
        writer.release()
    return n


def leftover_processes() -> list[str]:
    pats = ["gz sim", "ros_gz_bridge", "ros2 bag record", "policy_server.py"]
    found = []
    for p in pats:
        r = subprocess.run(["pgrep", "-af", p], capture_output=True, text=True)
        found += [ln for ln in r.stdout.splitlines() if "pgrep" not in ln]
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="closed-loop policy evaluation in headless Gazebo")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path, help="…/checkpoints/<step> (or its pretrained_model)")
    g.add_argument("--fake-policy", choices=["hold"], help="no model (evaluator self-test)")
    ap.add_argument("--seeds", type=int, nargs="+", help="default: configs/sim.yaml eval.seeds")
    ap.add_argument("--duration", type=float, help="override eval.duration_s (sim s)")
    ap.add_argument("--n-action-steps", type=int, help="actions executed per chunk")
    ap.add_argument("--temporal-ensemble-coeff", type=float, help="ACT temporal ensembling")
    ap.add_argument("--device", help="override eval.device")
    ap.add_argument("--engine", help="physics engine (default: configs/sim.yaml physics.engine)")
    ap.add_argument("--name", default="run", help="output folder suffix")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", type=Path, help="output folder (default: a new outputs/sim/ folder)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    ctrl = yaml.safe_load((ROOT / "configs" / "sim_controllers.yaml").read_text())
    ecfg = cfg["eval"]
    seeds = args.seeds or list(range(ecfg["seeds"][0], ecfg["seeds"][1] + 1))
    engine = args.engine or cfg["physics"].get("engine", "dartsim")
    out = args.out or new_run_dir(f"m3d_eval_{args.name}")
    (out / "bags").mkdir(parents=True)
    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    summary: dict = {"out": str(rel), "seeds": seeds, "physics_engine": engine,
                     "args": {k: str(v) for k, v in vars(args).items()}}  # fmt: skip
    t_wall = time.monotonic()
    address, key = default_address(), new_key()
    server = launch = node = client = None
    launch_log = (out / "launch.log").open("w")
    eps: list[dict] = []
    code = 1
    try:
        server = start_server(args, ecfg, out, address, key)
        launch = subprocess.Popen(["ros2", "launch", "so101_gazebo", "sim.launch.py",
                                   f"physics_engine:={engine}"], stdout=launch_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)  # fmt: skip
        sinfo = wait_ready(server, out / "policy_server.log", ecfg["server_start_timeout_s"])
        client = PolicyClient.connect(address, key, 30.0, alive=lambda: server.poll() is None)
        summary["server"] = sinfo
        print(f"policy server: {json.dumps(sinfo)}", flush=True)
        rclpy.init()
        joints = {k: ctrl[f"{k}_position_controller"]["ros__parameters"]["joints"]
                  for k in ("arm", "gripper")}  # fmt: skip
        node = EvalNode(cfg, joints["arm"], joints["gripper"])
        ready = node.spin_until(
            lambda: node.joints and node.block is not None and node.image is not None
            and node.arm_pub.get_subscription_count() > 0
            and node.grip_pub.get_subscription_count() > 0
            and node.set_pose.service_is_ready(),
            cfg["check"]["startup_timeout_s"],
        )  # fmt: skip
        summary["startup_wall_s"] = round(time.monotonic() - t_wall, 2)
        if not ready:
            raise RuntimeError("simulation not ready in time")
        planner = Planner(cfg)
        with (out / "results.jsonl").open("a") as jl:
            for i, seed in enumerate(seeds):
                info = run_episode(node, planner, client, sinfo, seed, i, out, cfg, args)
                eps.append(info)
                jl.write(json.dumps({k: v for k, v in info.items() if k != "_ticks"}) + "\n")
                jl.flush()
                r, lp = info["result"], info["loop"]
                print(f"episode {i} seed {seed}: success={r['success']} error={r['error_m']} m, "
                      f"infer p95 {lp['infer_ms_p95']} ms, overruns {lp['overruns']}, "
                      f"wall {info['wall_duration_s']} s", flush=True)  # fmt: skip
        code = 0
    except Exception as e:  # recorded; everything is still stopped below
        summary["error"] = f"{type(e).__name__}: {e}"
        print(summary["error"], file=sys.stderr)
    finally:
        if client is not None:
            client.close()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if server is not None:
            summary["server_stop"] = stop_group(server, 20.0)
        if launch is not None:
            summary["shutdown"] = stop_group(launch, cfg["check"]["shutdown_timeout_s"])
        launch_log.close()
        summary.update(summarize_episodes(eps))
        if not args.no_video:
            videos = {}
            for kind, e in pick_examples(eps).items():
                path = out / f"{kind}_seed{e['seed']}.mp4"
                label = f"seed {e['seed']} {kind} err {e['result']['error_m'] * 100:.1f} cm"
                try:
                    n = bag_to_mp4(out / "bags" / e["bag"], path, cfg["camera"]["topic"],
                                   ecfg["video_fps"], label)  # fmt: skip
                    videos[kind] = {"file": path.name, "seed": e["seed"], "frames": n}
                except Exception as err:  # the results are kept even if a video fails
                    videos[kind] = {"seed": e["seed"], "error": f"{type(err).__name__}: {err}"}
            summary["videos"] = videos
        summary["leftover_processes"] = leftover_processes()
        summary["total_wall_s"] = round(time.monotonic() - t_wall, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"-> {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
