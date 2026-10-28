"""M4a: the safety gate in the closed loop of headless Gazebo, with a scripted intruder (a rod).

    source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
      timeout --signal=INT --kill-after=30 3600 python3 scripts/sim/gate_demo_sim.py

    ... --dry-run                          # list the scenarios and the estimated time only
    ... --only center_1 base_1             # a subset (scenario ids)
    ... --fake-policy hold --duration 4    # self-test (no checkpoint)
    ... --check-intruder                   # put the rod at each named place, save one frame each
    ... --judge v1                         # M4b: the image judge v1 on the m4b scenarios
    ... --judge oracle --m4b               # M4b: the oracle on the same scenarios (comparison)

Settings: configs/m4a.yaml (policy, scenarios, judge, zones), configs/gate_sim.yaml (the gate),
configs/sim.yaml (intruder model; `eval` / `demo` as in M3d). Gazebo and the policy server
(train/policy_server.py) start once. Each scenario reuses eval_policy.run_episode (home pose,
block start, bag, closed loop, success judge) with three additions:
  on_tick  moves the rod with set_pose to the scenario's scripted position (each tick)
  gate     scripts/sim/gate_loop.GatedCommand: oracle judge (zone of the scripted rod at the image
           time) behind a JudgeRunner (latency, dropouts, flicker, confidence), the unchanged
           SafetyGate, and on_resume -> the policy server's reset (G7)
  extra    joint state, block pose, the rod's scripted and measured positions
M4b (--judge <name>, configs/m4b.yaml): the oracle is replaced by an image judge running in its
own server (scripts/sim/judge_server.py, uv env), asked through sim_judge.RemoteJudge; the
JudgeRunner releases each result after the judge's own latency (wall ms used as sim ms). The
scenarios and judge settings come from m4b.yaml (scripts/sim/m4b_suite.py); results.jsonl gets
`judge` (accuracy against the truth, center -> CONTINUE, delay, round trips) per scenario.
The bag also records the rod's pose (/intruder/pose), so frames + labels can be exported later
(train/export_intruder_frames.py). Output: outputs/sim/<stamp>_m4a_gate/ with results.jsonl (per
scenario, with gate metrics), summary.json (config copies, git commit), bags/ (ticks.jsonl with
the gate record per tick, judge.jsonl) and one mp4 per scenario with the gate state drawn on it
(--videos-from <run> rewrites them). All processes are stopped and checked to be gone.
Figures and tables: scripts/sim/gate_demo_report.py (uv env).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import yaml  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from ros_gz_interfaces.msg import Entity  # noqa: E402
from ros_gz_interfaces.srv import SetEntityPose  # noqa: E402

from gate.safety_gate import load_gate_config  # noqa: E402
from scripts.sim.demo_plan import ARM, Planner, surface_z  # noqa: E402
from scripts.sim.eval_loop import summarize_episodes  # noqa: E402
from scripts.sim.eval_policy import (  # noqa: E402
    EvalNode,
    bag_to_mp4,
    image_to_array,
    leftover_processes,
    run_episode,
    wait_ready,
)
from scripts.sim.gate_loop import GatedCommand, gate_metrics  # noqa: E402
from scripts.sim.gate_video import make_overlay  # noqa: E402
from scripts.sim.gen_demos import stop_group  # noqa: E402
from scripts.sim.intruder import M4A_YAML, build_scenarios, load_m4a, zone_of  # noqa: E402
from scripts.sim.m4b_suite import (  # noqa: E402
    M4B_YAML,
    ORACLE,
    judge_spec,
    load_m4b,
    suite_scenarios,
)
from scripts.sim.outputs import BASE, new_run_dir  # noqa: E402
from scripts.sim.policy_link import KEY_ENV, PolicyClient, default_address, new_key  # noqa: E402
from scripts.sim.robustness import Server, block_xyyaw, git_state  # noqa: E402
from scripts.sim.robustness_core import TcpFK, estimate_wall_s  # noqa: E402
from scripts.sim.sim_judge import (  # noqa: E402
    JudgeRunner,
    OracleJudge,
    RemoteJudge,
    judge_log_metrics,
)


class GateNode(EvalNode):
    """EvalNode plus the intruder: its measured pose and set_pose without waiting."""

    def __init__(self, cfg, arm_joints, grip_joints):
        super().__init__(cfg, arm_joints, grip_joints)
        self.intruder: PoseStamped | None = None
        self.create_subscription(PoseStamped, cfg["intruder"]["pose_topic"], self._on_intr, 10)
        self._pending = []  # set_pose futures not yet answered

    def _on_intr(self, msg: PoseStamped):
        self.intruder = msg

    def _intruder_request(self, x, y, z) -> SetEntityPose.Request:
        """(x, y, z): rod centre, table frame, z above the white area's top surface."""
        req = SetEntityPose.Request()
        req.entity = Entity(name=self.cfg["intruder"]["name"], type=Entity.MODEL)
        z_world = self.cfg["table"]["height"] + surface_z(self.cfg) + z
        req.pose.position.x, req.pose.position.y, req.pose.position.z = x, y, z_world
        req.pose.orientation.w = 1.0
        return req

    def place_intruder(self, x, y, z, wall_limit=10.0) -> bool:
        fut = self.set_pose.call_async(self._intruder_request(x, y, z))
        end = time.monotonic() + wall_limit
        while not fut.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
        return fut.done() and fut.result() is not None and fut.result().success

    def move_intruder(self, x, y, z) -> int:
        """Ask for the move without waiting (each tick). Returns the number of failed answers
        collected since the previous call."""
        self._pending.append(self.set_pose.call_async(self._intruder_request(x, y, z)))
        failed, keep = 0, []
        for f in self._pending:
            if f.done():
                failed += int(f.result() is None or not f.result().success)
            else:
                keep.append(f)
        self._pending = keep
        return failed

    def intruder_table(self) -> list[float] | None:
        """Latest measured rod centre (x, y, z above the white area), or None."""
        if self.intruder is None:
            return None
        p = self.intruder.pose.position
        return [p.x, p.y, p.z - self.cfg["table"]["height"] - surface_z(self.cfg)]


def start_launch(cfg, out: Path):
    log = (out / "launch.log").open("w")
    engine = cfg["physics"].get("engine", "dartsim")
    proc = subprocess.Popen(["ros2", "launch", "so101_gazebo", "sim.launch.py",
                             f"physics_engine:={engine}"], stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)  # fmt: skip
    return proc, log


def make_node(cfg, cls=None) -> GateNode:
    ctrl = yaml.safe_load((ROOT / "configs" / "sim_controllers.yaml").read_text())
    joints = {k: ctrl[f"{k}_position_controller"]["ros__parameters"]["joints"]
              for k in ("arm", "gripper")}  # fmt: skip
    rclpy.init()
    node = (cls or GateNode)(cfg, joints["arm"], joints["gripper"])
    ready = node.spin_until(
        lambda: node.joints and node.block is not None and node.image is not None
        and node.intruder is not None
        and node.arm_pub.get_subscription_count() > 0
        and node.grip_pub.get_subscription_count() > 0
        and node.set_pose.service_is_ready(),
        cfg["check"]["startup_timeout_s"],
    )  # fmt: skip
    if not ready:
        raise RuntimeError("simulation not ready in time (is the intruder in the world?)")
    return node


def check_intruder(node: GateNode, cfg, m4a, out: Path) -> dict:
    from PIL import Image as PILImage

    sc = build_scenarios(m4a, cfg)[0]
    places = {**sc.points, "park": sc.park}
    res = {}
    for name, (x, y, z) in places.items():
        ok = node.place_intruder(x, y, z)
        node.spin_for(0.5)
        t_req = node.sim_now()
        node.spin_until(lambda t=t_req: node.image is not None and _stamp(node.image) > t, 10.0)
        meas = node.intruder_table()
        PILImage.fromarray(image_to_array(node.image)).save(out / f"intruder_{name}.png")
        err = float(np.linalg.norm(np.subtract(meas, (x, y, z)))) if meas else None
        res[name] = {"requested": [x, y, z], "measured": meas, "set_pose_ok": bool(ok),
                     "error_m": err, "zone": zone_of(x, y, cfg)}  # fmt: skip
        print(f"{name}: zone {res[name]['zone']}, set_pose {ok}, error {err}", flush=True)
    return res


class JudgeServer:
    """The image judge server (scripts/sim/judge_server.py, uv env), one per run."""

    def __init__(self, spec: dict, m4b: dict, out: Path):
        self.spec, self.m4b, self.out = spec, m4b, out
        self.proc = self.client = self.info = None
        self.stops: list[dict] = []

    def start(self) -> dict:
        address, key = default_address("jevlite_judge"), new_key()
        cmd = ["uv", "run", "python", str(ROOT / "scripts" / "sim" / "judge_server.py"),
               "--address", address, "--kind", self.spec["kind"], "--device", self.m4b["device"]]
        if self.spec.get("model"):
            cmd += ["--model", str(ROOT / self.spec["model"])]
        log = self.out / "judge_server.log"
        with log.open("w") as f:
            self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                                         start_new_session=True,
                                         env={**os.environ, KEY_ENV: key})  # fmt: skip
        self.info = wait_ready(self.proc, log, self.m4b["server_start_timeout_s"])
        self.client = PolicyClient.connect(address, key, 30.0,
                                           alive=lambda: self.proc.poll() is None)  # fmt: skip
        print(f"judge server: {json.dumps(self.info)}", flush=True)
        return self.info

    def stop(self) -> None:
        if self.client is not None:
            self.client.close()
        if self.proc is not None:
            self.stops.append(stop_group(self.proc, 20.0))
        self.proc = self.client = None


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def record_cfg(cfg: dict) -> dict:
    """sim.yaml with the rod's pose topic added to the recorded topics (M3d topics unchanged)."""
    c = copy.deepcopy(cfg)
    topics = c["demo"]["record"]["topics"]
    if cfg["intruder"]["pose_topic"] not in topics:
        topics.append(cfg["intruder"]["pose_topic"])
    return c


def make_hooks(node: GateNode, sc, gated: GatedCommand):
    """on_tick: move the rod to its scripted position. extra: joints, block, rod (both)."""

    def on_tick(i, t):
        if not sc.has_intruder:
            return None
        failed = node.move_intruder(*sc.position(t))
        return {"set_pose_failed": failed} if failed else None

    def extra(obs):
        meas = node.intruder_table()
        return {"state": [float(obs.joints[j]) for j in ARM],
                "block": [round(v, 6) for v in block_xyyaw(node)],
                "intruder": [round(v, 5) for v in sc.position(obs.t_image - gated.t0)],
                "intruder_meas": [round(v, 5) for v in meas] if meas else None}  # fmt: skip

    return on_tick, extra


def run_scenarios(scs, node, planner, server, fk, cfg, m4a, gcfg, args, out, eps: list,
                  judge_server: JudgeServer | None = None) -> None:  # fmt: skip
    sinfo = server.ensure(m4a["policy"]["n_action_steps"], out)
    rcfg = record_cfg(cfg)
    block_start = tuple(float(v) for v in m4a["block_start"])
    with (out / "results.jsonl").open("a") as jl:
        for i, sc in enumerate(scs):
            holder = {}

            def zone_at(t_abs, sc=sc, holder=holder):
                return sc.zone(t_abs - holder["gate"].t0, cfg)

            if judge_server is None:
                judge = OracleJudge(zone_at, m4a["zones"], confidence=sc.judge["confidence"])
            else:
                judge = RemoteJudge(judge_server.client, judge_server.info.get("source", "remote"))
            runner = JudgeRunner(judge, sc.judge, m4a["zones"], truth=zone_at)
            gated = GatedCommand(gcfg, runner, server.client, sinfo["command_names"], zone_at,
                                 time_scale=m4a["slow"]["time_scale"])  # fmt: skip
            holder["gate"] = gated
            if not node.place_intruder(*sc.position(0.0)):
                raise RuntimeError("set_pose of the intruder failed")
            on_tick, extra = make_hooks(node, sc, gated)
            ep_args = SimpleNamespace(duration=args.duration or sc.duration_s or m4a["duration_s"])
            info = run_episode(node, planner, server.client, sinfo, None, i, out, rcfg, ep_args,
                               start=block_start, on_tick=on_tick, extra=extra, tag=sc.id,
                               meta=sc.meta(), gate=gated)  # fmt: skip
            node.place_intruder(*sc.park)
            react = (sc.judge.get("latency_s") or 0.0) + m4a["metrics"]["settle_s"]
            info["metrics"] = gate_metrics(info["_ticks"], fk, react_s=react)
            info["resume_calls"] = gated.resumes
            info["judge_name"] = args.judge
            info["judge"] = judge_log_metrics(runner.log, m4a["zones"],
                                              getattr(judge, "rtt_ms", None),
                                              len(getattr(judge, "errors", [])))  # fmt: skip
            info["loop_t0"] = gated.t0  # sim time of tick 0 (for the video overlay, labels)
            meas = [(k["intruder"], k["intruder_meas"]) for k in info["_ticks"]
                    if sc.has_intruder and k.get("intruder_meas")]  # fmt: skip
            info["intruder_meas_err_max_m"] = round(max(
                (float(np.linalg.norm(np.subtract(a, b))) for a, b in meas), default=0.0), 5)
            bag = out / "bags" / info["bag"]
            if bag.exists():
                (bag / "judge.jsonl").write_text("".join(json.dumps(e) + "\n" for e in runner.log))
            eps.append(info)
            jl.write(json.dumps({k: v for k, v in info.items() if k != "_ticks"}) + "\n")
            jl.flush()
            m, r = info["metrics"], info["result"]
            intr = m["intrusion"] or {}
            print(f"[{i + 1}/{len(scs)}] {sc.id}: success={r['success']} error={r['error_m']} m, "
                  f"state_s {m['state_s']}, resumes {m['resumes']}, reasons {m['reasons']}, "
                  f"tcp max {m['tcp_speed_max']} m/s, in intrusion {intr.get('tcp_speed_max')}, "
                  f"centre settled {intr.get('tcp_speed_max_center_settled')}, "
                  f"wall {info['wall_duration_s']} s", flush=True)  # fmt: skip


def write_videos(out: Path, cfg: dict, m4a: dict) -> dict:
    """One mp4 per episode of a run folder, with the gate band (scripts/sim/gate_video.py)."""
    videos = {}
    for line in (out / "results.jsonl").read_text().splitlines():
        e = json.loads(line)
        bag = out / "bags" / e["bag"]
        ticks = [json.loads(x) for x in (bag / "ticks.jsonl").read_text().splitlines()]
        path = out / f"{e['scenario']}.mp4"
        label = f"{e['scenario']}  {'success' if e['result']['success'] else 'failure'}"
        try:
            n = bag_to_mp4(bag, path, cfg["camera"]["topic"], m4a["video"]["fps"], label,
                           overlay=make_overlay(ticks, e["loop_t0"], e["scenario"]))  # fmt: skip
            videos[e["scenario"]] = {"file": path.name, "frames": n}
        except Exception as err:  # the results are kept even if a video fails
            videos[e["scenario"]] = {"error": f"{type(err).__name__}: {err}"}
        print(f"video {path.name}: {videos[e['scenario']]}", flush=True)
    return videos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M4a: safety gate in the Gazebo closed loop")
    ap.add_argument("--config", type=Path, default=M4A_YAML)
    ap.add_argument("--only", nargs="+", help="scenario ids to run (default: all)")
    ap.add_argument("--fake-policy", choices=["hold"], help="no model (self-test)")
    ap.add_argument("--duration", type=float, help="override duration_s (sim s)")
    ap.add_argument("--device", help="override policy.device")
    ap.add_argument("--check-intruder", action="store_true", help="place the rod, save frames")
    ap.add_argument("--out-base", type=Path, default=BASE, help="parent of the output folder")
    ap.add_argument("--name", default="gate", help="output folder suffix (m4a_<name>)")
    ap.add_argument("--dry-run", action="store_true", help="list the scenarios, run nothing")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--videos-from", type=Path, help="only (re)write the mp4s of a run folder")
    ap.add_argument("--judge", default=ORACLE, help="oracle or an image judge of m4b.yaml")
    ap.add_argument("--judge-model", help="override the judge's model folder")
    ap.add_argument("--m4b", action="store_true", help="the m4b scenarios (implied by --judge)")
    ap.add_argument("--m4b-config", type=Path, default=M4B_YAML)
    args = ap.parse_args(argv)
    m4a = load_m4a(args.config)
    cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    if args.videos_from:
        print(json.dumps(write_videos(args.videos_from, cfg, m4a), indent=2))
        return 0
    scs = build_scenarios(m4a, cfg)
    image_judge = args.judge != ORACLE
    m4b = spec = None
    if image_judge or args.m4b:
        m4b = load_m4b(args.m4b_config)
        scs = suite_scenarios(scs, m4b, image_judge)
        if image_judge:
            try:
                spec = judge_spec(m4b, args.judge, args.judge_model)
            except ValueError as e:
                ap.error(str(e))
    if args.only:
        unknown = set(args.only) - {s.id for s in scs}
        if unknown:
            ap.error(f"unknown scenarios {sorted(unknown)}")
        scs = [s for s in scs if s.id in args.only]
    if not args.check_intruder:
        durs = [args.duration or sc.duration_s or m4a["duration_s"] for sc in scs]
        print(f"{len(scs)} scenarios: " + ", ".join(s.id for s in scs))
        est = estimate_wall_s(len(scs), sum(durs) / len(scs)) if scs else 0.0
        print(f"estimated wall time {est / 60:.0f} min", flush=True)
        if args.dry_run:
            return 0
    gcfg = load_gate_config(ROOT / m4a["gate_config"])
    if args.check_intruder:
        stem = "m4a_intruder_check"
    elif m4b is not None:
        stem = f"m4b_{args.judge}" + ("" if args.name == "gate" else f"_{args.name}")
    else:
        stem = f"m4a_{args.name}"
    out = new_run_dir(stem, base=args.out_base)
    t_wall = time.monotonic()
    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    summary: dict = {"out": str(rel), "config": m4a, "gate_config": asdict(gcfg),
                     "git": git_state(), "args": {k: str(v) for k, v in vars(args).items()},
                     "physics_engine": cfg["physics"].get("engine", "dartsim"),
                     "m4b": m4b, "judge_spec": spec}  # fmt: skip
    launch = log = node = None
    server = jserver = None
    eps: list[dict] = []
    code = 1
    try:
        launch, log = start_launch(cfg, out)
        node = make_node(cfg)
        if args.check_intruder:
            summary["places"] = check_intruder(node, cfg, m4a, out)
        else:
            (out / "bags").mkdir()
            sargs = SimpleNamespace(fake_policy=args.fake_policy, device=args.device)
            server = Server(sargs, m4a, cfg["eval"], out)
            if spec is not None:
                jserver = JudgeServer(spec, m4b, out)
                summary["judge_server"] = jserver.start()
            run_scenarios(scs, node, Planner(cfg), server, TcpFK(cfg), cfg, m4a, gcfg, args,
                          out, eps, jserver)  # fmt: skip
        code = 0
    except Exception as e:  # recorded; everything is still stopped below
        summary["error"] = f"{type(e).__name__}: {e}"
        print(summary["error"], file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if server is not None:
            server.stop()
            summary["server_stops"] = server.stops
        if jserver is not None:
            jserver.stop()
            summary["judge_server_stops"] = jserver.stops
        if launch is not None:
            summary["shutdown"] = stop_group(launch, cfg["check"]["shutdown_timeout_s"])
        if log is not None:
            log.close()
        if not args.check_intruder:
            summary.update(summarize_episodes(eps))
            if eps and not args.no_video:
                summary["videos"] = write_videos(out, cfg, m4a)
            summary["scenarios"] = {e["scenario"]: {"success": e["result"]["success"],
                                                    "metrics": e["metrics"],
                                                    "judge": e.get("judge")}
                                    for e in eps}  # fmt: skip
        summary["leftover_processes"] = leftover_processes()
        summary["total_wall_s"] = round(time.monotonic() - t_wall, 1)
        name = "check.json" if args.check_intruder else "summary.json"
        (out / name).write_text(json.dumps(summary, indent=2, default=_json) + "\n")
        brief = {k: summary.get(k) for k in ("episodes", "successes", "leftover_processes",
                                              "error", "total_wall_s")}  # fmt: skip
        print(json.dumps(brief, indent=2, default=_json))
        print(f"-> {out}")
    return code


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o).__name__)


if __name__ == "__main__":
    sys.exit(main())
