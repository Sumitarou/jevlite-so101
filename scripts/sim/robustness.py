"""M3e: robustness of a trained policy in headless Gazebo (evaluation only, no retraining).

    source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
      timeout --signal=INT --kill-after=60 5400 python3 scripts/sim/robustness.py

    ... --dry-run                          # list the episodes and the estimated time only
    ... --suites perturb --only push_x+2   # a subset (names or ids)
    ... --fake-policy hold --duration 3    # evaluator self-test (no checkpoint)

Settings: configs/robustness.yaml (conditions, policy, metrics), configs/sim.yaml (`eval`,
`demo`: home pose, recording, success judge, all as in M3d). Suites:
  start    the block starts outside the demo range (x, yaw, y), n_action_steps fixed
  perturb  the block is moved with set_pose at a set sim time (during or before the push),
           for each n_action_steps (changed at inference time, no retraining)
Gazebo starts once; the policy server (train/policy_server.py) is restarted only when
n_action_steps changes. Each episode reuses eval_policy.run_episode (home pose, set_pose, bag,
closed loop, judge); here the start is given directly, and every tick also records the joint
state and the block pose (x, y, yaw), from which robustness_core computes the TCP metrics.
Output per suite: outputs/sim/<stamp>_m3e_<suite>/ with results.jsonl (one line per episode,
with its metrics), summary.json (with a copy of the config and the git commit), bags/
(ticks.jsonl), and the mp4s named in the config's `videos`. Processes are checked to be gone.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import yaml  # noqa: E402

from scripts.sim.demo_plan import ARM, Planner  # noqa: E402
from scripts.sim.eval_loop import summarize_episodes  # noqa: E402
from scripts.sim.eval_policy import (  # noqa: E402
    EvalNode,
    bag_to_mp4,
    leftover_processes,
    run_episode,
    start_server,
    wait_ready,
)
from scripts.sim.gen_demos import stop_group  # noqa: E402
from scripts.sim.outputs import BASE, git_state, new_run_dir  # noqa: E402, F401
from scripts.sim.policy_link import PolicyClient, default_address, new_key  # noqa: E402
from scripts.sim.robustness_core import (  # noqa: E402
    ROBUSTNESS_YAML,
    SUITES,
    TcpFK,
    build_conditions,
    episode_metrics,
    estimate_wall_s,
    perturbed_pose,
    x_slopes,
    yaw_of,
)


def block_xyyaw(node) -> list[float]:
    x, y, _, qx, qy, qz, qw = node.block_pose_table()
    return [x, y, yaw_of(qx, qy, qz, qw)]


def make_hooks(node, cond, cfg, rcfg):
    """on_tick: move the block once, at the perturbation's time. extra: state + block pose."""
    p = cond.perturbation
    done = []

    def on_tick(i, t):
        if p is None or done or t < p["t_s"] - 1e-9:
            return None
        done.append(i)
        cur = block_xyyaw(node)
        clear = rcfg["perturb_suite"]["y_clearance"]
        new, dy = perturbed_pose(cur, p["dx"], math.radians(p["dyaw_deg"]),
                                 cfg["block"]["size"], clear)  # fmt: skip
        ok = node.place_block(*new)
        return {"name": p["name"], "t": round(t, 4), "dx": p["dx"], "dyaw_deg": p["dyaw_deg"],
                "dy": round(dy, 5), "from": [round(v, 5) for v in cur],
                "to": [round(v, 5) for v in new], "set_pose_ok": bool(ok)}  # fmt: skip

    def extra(obs):
        return {"state": [float(obs.joints[j]) for j in ARM],
                "block": [round(v, 6) for v in block_xyyaw(node)]}  # fmt: skip

    return on_tick, extra


class Server:
    """The policy server for one n_action_steps; restarted when it changes."""

    def __init__(self, args, rcfg, ecfg, out):
        self.args, self.rcfg, self.ecfg, self.out = args, rcfg, ecfg, out
        self.proc = self.client = self.info = None
        self.n = None
        self.starts = 0
        self.stops: list[dict] = []

    def ensure(self, n: int, out: Path) -> dict:
        if self.proc is not None and self.n == n:
            return self.info
        self.stop()
        self.starts += 1
        address = default_address(f"jevlite_policy_m3e_{self.starts}")
        key = new_key()
        sargs = SimpleNamespace(
            fake_policy=self.args.fake_policy,
            checkpoint=ROOT / self.rcfg["policy"]["checkpoint"],
            device=self.args.device or self.rcfg["policy"]["device"],
            n_action_steps=n, temporal_ensemble_coeff=None,
        )  # fmt: skip
        log = f"policy_server_n{n}.log"
        self.proc = start_server(sargs, self.ecfg, out, address, key, log_name=log)
        self.info = wait_ready(self.proc, out / log, self.ecfg["server_start_timeout_s"])
        self.client = PolicyClient.connect(
            address, key, 30.0, alive=lambda: self.proc.poll() is None
        )
        self.n = n
        print(f"policy server (n_action_steps {n}): {json.dumps(self.info)}", flush=True)
        return self.info

    def stop(self):
        if self.client is not None:
            self.client.close()
        if self.proc is not None:
            self.stops.append({"n_action_steps": self.n, **stop_group(self.proc, 20.0)})
        self.proc = self.client = None


def run_suite(suite, conds, node, planner, server, fk, cfg, rcfg, args, out, eps: list) -> None:
    """Run the episodes in order, appending each finished one to eps (kept if a later fails)."""
    ep_args = SimpleNamespace(duration=args.duration or rcfg["duration_s"])
    with (out / "results.jsonl").open("a") as jl:
        for i, c in enumerate(conds):
            sinfo = server.ensure(c.n_action_steps, out)
            on_tick, extra = make_hooks(node, c, cfg, rcfg)
            info = run_episode(node, planner, server.client, sinfo, None, i, out, cfg, ep_args,
                               start=c.start, on_tick=on_tick, extra=extra, tag=c.id,
                               meta=c.meta())  # fmt: skip
            info["server_n_action_steps"] = sinfo.get("n_action_steps")
            info["metrics"] = episode_metrics(info["_ticks"], fk, cfg, rcfg["metrics"])
            eps.append(info)
            jl.write(json.dumps({k: v for k, v in info.items() if k != "_ticks"}) + "\n")
            jl.flush()
            m, r = info["metrics"], info["result"]
            pm = m.get("perturb", {})
            print(f"[{suite} {i + 1}/{len(conds)}] {c.id}: success={r['success']} "
                  f"error={r['error_m']} m, |x gap| push {m['gap_abs_mean']}, "
                  f"touched={m['touched']}, moved {m['block_moved_m']} m"
                  + (f", reaim {pm.get('reaim_s')} s, perturb_ok {pm.get('perturb_ok')}"
                     if pm else "")
                  + f", wall {info['wall_duration_s']} s", flush=True)  # fmt: skip


def write_videos(eps, names, out, cfg) -> dict:
    videos = {}
    by_id = {e["condition"]: e for e in eps if e.get("status") == "done"}
    for cid in names:
        e = by_id.get(cid)
        if e is None:
            continue
        r = e["result"]
        label = f"{cid} {'success' if r['success'] else 'failure'} err {r['error_m'] * 100:.1f} cm"
        path = out / f"{cid}.mp4"
        try:
            n = bag_to_mp4(out / "bags" / e["bag"], path, cfg["camera"]["topic"],
                           cfg["eval"]["video_fps"], label)  # fmt: skip
            videos[cid] = {"file": path.name, "frames": n}
        except Exception as err:  # the results are kept even if a video fails
            videos[cid] = {"error": f"{type(err).__name__}: {err}"}
    return videos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3e robustness evaluation in headless Gazebo")
    ap.add_argument("--config", type=Path, default=ROBUSTNESS_YAML)
    ap.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    ap.add_argument("--only", nargs="+", help="condition names or ids to run (default: all)")
    ap.add_argument("--n-action-steps", type=int, nargs="+", help="only these settings")
    ap.add_argument("--fake-policy", choices=["hold"], help="no model (evaluator self-test)")
    ap.add_argument("--duration", type=float, help="override duration_s (sim s)")
    ap.add_argument("--device", help="override policy.device")
    ap.add_argument("--out-base", type=Path, default=BASE, help="parent of the output folders")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the episodes, run nothing")
    args = ap.parse_args(argv)
    rcfg = yaml.safe_load(args.config.read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    ctrl = yaml.safe_load((ROOT / "configs" / "sim_controllers.yaml").read_text())
    plans = {}
    for suite in args.suites:
        conds = build_conditions(rcfg, cfg, suite)
        if args.only:
            conds = [c for c in conds if c.name in args.only or c.id in args.only]
        if args.n_action_steps:
            conds = [c for c in conds if c.n_action_steps in args.n_action_steps]
        if conds:
            plans[suite] = conds
    n_eps = sum(len(v) for v in plans.values())
    n_srv = sum(len({c.n_action_steps for c in v}) for v in plans.values())
    est = estimate_wall_s(n_eps, args.duration or rcfg["duration_s"], n_servers=n_srv)
    for suite, conds in plans.items():
        print(f"{suite}: {len(conds)} episodes: " + ", ".join(c.id for c in conds))
    print(f"total {n_eps} episodes, estimated wall time {est / 60:.0f} min", flush=True)
    if args.dry_run or not plans:
        return 0 if plans else 1

    t_wall = time.monotonic()
    ecfg = cfg["eval"]
    outs = {s: new_run_dir(f"m3e_{s}", base=args.out_base) for s in plans}
    for out in outs.values():
        (out / "bags").mkdir()
    first = next(iter(outs.values()))
    meta = {"config": rcfg, "config_path": str(args.config), "git": git_state(),
            "args": {k: str(v) for k, v in vars(args).items()},
            "physics_engine": cfg["physics"].get("engine", "dartsim")}  # fmt: skip
    summaries = {s: {"suite": s, "out": str(o.relative_to(ROOT) if o.is_relative_to(ROOT) else o),
                     **meta} for s, o in outs.items()}  # fmt: skip
    results: dict[str, list] = {s: [] for s in plans}
    launch = node = None
    server = Server(args, rcfg, ecfg, first)
    launch_log = (first / "launch.log").open("w")
    code = 1
    try:
        launch = subprocess.Popen(["ros2", "launch", "so101_gazebo", "sim.launch.py",
                                   f"physics_engine:={meta['physics_engine']}"],
                                  stdout=launch_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)  # fmt: skip
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
        if not ready:
            raise RuntimeError("simulation not ready in time")
        planner, fk = Planner(cfg), TcpFK(cfg)
        for suite, conds in plans.items():
            t0 = time.monotonic()
            run_suite(
                suite,
                conds,
                node,
                planner,
                server,
                fk,
                cfg,
                rcfg,
                args,
                outs[suite],
                results[suite],
            )
            summaries[suite]["suite_wall_s"] = round(time.monotonic() - t0, 1)
        code = 0
    except Exception as e:  # recorded; everything is still stopped below
        for s in summaries.values():
            s["error"] = f"{type(e).__name__}: {e}"
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        server.stop()
        shutdown = stop_group(launch, cfg["check"]["shutdown_timeout_s"]) if launch else None
        launch_log.close()
        left = leftover_processes()
        for suite, out in outs.items():
            eps = results[suite]
            s = summaries[suite]
            s.update(summarize_episodes(eps))
            s["server_starts"], s["server_stops"] = server.starts, server.stops
            s["shutdown"], s["leftover_processes"] = shutdown, left
            if suite == "start":
                s["x_slopes"] = x_slopes(eps, cfg["demo"])
            if not args.no_video:
                s["videos"] = write_videos(eps, rcfg.get("videos", {}).get(suite, []), out, cfg)
            s["total_wall_s"] = round(time.monotonic() - t_wall, 1)
            (out / "summary.json").write_text(json.dumps(s, indent=2, default=_json) + "\n")
            print(f"-> {out}")
        print(json.dumps({k: {kk: v.get(kk) for kk in ("episodes", "successes", "x_slopes",
                                                       "leftover_processes", "error")}
                          for k, v in summaries.items()}, indent=2, default=_json))  # fmt: skip
    return code


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


if __name__ == "__main__":
    sys.exit(main())
