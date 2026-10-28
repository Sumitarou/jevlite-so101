"""M4b: compare gate_demo_sim.py runs of the same m4b scenarios with different judges.

    uv run python scripts/sim/m4b_report.py outputs/sim/<oracle run> <v1 run> <v2 run>

Per scenario and judge (from results.jsonl and each bag's judge.jsonl): success and error, STOP /
SLOW time, resumes, time from the truth entering center to STOP, from leaving it to the resume,
the TCP speed max while the rod is in (and after settling in center), the judge's accuracy
against the truth at the image time, center -> CONTINUE, milder / stricter answers, how many of
the wrong answers came within transition_s of a change of the truth (the rod moving across a
boundary), the judge delay (sim) and round trip (wall) p95, and the oldest decision the gate saw.
The first run is the reference (the oracle). Output: outputs/sim/<stamp>_m4b_report/ with
compare.md and compare.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jevlite.api import Choice  # noqa: E402
from scripts.sim.intruder import load_m4a  # noqa: E402
from scripts.sim.outputs import git_state, new_run_dir  # noqa: E402

TRANSITION_S = 0.1


def load_run(run: Path) -> list[dict]:
    eps = [json.loads(x) for x in (run / "results.jsonl").read_text().splitlines()]
    for e in eps:
        text = (run / "bags" / e["bag"] / "judge.jsonl").read_text()
        e["_log"] = [json.loads(x) for x in text.splitlines()]
    return eps


def wrong_answers(log: list[dict], zones: dict, transition_s: float = TRANSITION_S) -> dict:
    """Wrong answers, split by whether the truth changed within transition_s of the image."""
    changes = [b["t_image"] for a, b in zip(log, log[1:], strict=False)
               if a["truth"] != b["truth"]]  # fmt: skip
    out = {"wrong": 0, "at_transition": 0, "under": 0, "under_at_transition": 0, "list": []}
    for e in log:
        want = Choice.parse(zones[e["truth"]])
        got = Choice.parse(e["choice"])
        if got == want:
            continue
        near = any(abs(e["t_image"] - c) <= transition_s + 1e-9 for c in changes)
        out["wrong"] += 1
        out["at_transition"] += int(near)
        out["under"] += int(got < want)
        out["under_at_transition"] += int(near and got < want)
        out["list"].append({"t_image": e["t_image"], "truth": e["truth"], "choice": e["choice"],
                            "conf": e["conf"], "at_transition": near})  # fmt: skip
    return out


def row(e: dict, zones: dict) -> dict:
    m, j, r = e["metrics"], e["judge"], e["result"]
    i = m["intrusion"] or {}
    w = wrong_answers(e["_log"], zones)
    return {
        "success": r["success"], "error_cm": round(r["error_m"] * 100, 2),
        "stop_s": m["state_s"]["STOP"], "slow_s": m["state_s"]["SLOW"], "resumes": m["resumes"],
        "center_to_stop_s": i.get("stop_after_center_s"),
        "resume_after_center_s": i.get("resume_after_center_s"),
        "tcp_max_in_intrusion": i.get("tcp_speed_max"),
        "tcp_max_center_settled": i.get("tcp_speed_max_center_settled"),
        "judge_n": j["n"], "accuracy": j["accuracy"], "center_to_continue": j["center_to_continue"],
        "under": j["under"], "over": j["over"], "wrong_at_transition": w["at_transition"],
        "under_at_transition": w["under_at_transition"], "wrong_list": w["list"],
        "delay_p95_ms": j["delay_ms"]["p95"], "rtt_p95_ms": j["rtt_ms"]["p95"],
        "errors": j["errors"], "age_max_s": m["decision_age_s_max"], "reasons": m["reasons"],
    }  # fmt: skip


def _f(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "○" if v else "×"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def compare_md(res: dict) -> str:
    names = res["judges"]
    lines = ["# M4b 閉ループ：判定器の比較（同じシナリオ・同じ方策・同じゲート）", "",
             "実行：" + "、".join(f"{n} = `{p}`" for n, p in zip(names, res["runs"], strict=True)),
             "", "判定の誤りのうち「切り替わり」は、正解が変わった時刻から "
             f"{TRANSITION_S} s 以内の画像（棒が境目を横切る途中）。", ""]  # fmt: skip
    head = ("| シナリオ | 判定器 | 成否（誤差 cm） | STOP s | SLOW s | 再開 | center → STOP s "
            "| center の後 → 再開 s | 侵入中の手先 最大 cm/s | 判定の正解率 | center→CONTINUE "
            "| 緩い / 厳しい（うち切り替わり） | 遅れ p95 ms（往復） | 古さ 最大 ms |")  # fmt: skip
    lines += [head, "|" + "---|" * 14]
    for sc, per in res["scenarios"].items():
        for n in names:
            r = per[n]
            tcp = None if r["tcp_max_in_intrusion"] is None else r["tcp_max_in_intrusion"] * 100
            lines.append(
                f"| {sc} | {n} | {_f(r['success'])}（{r['error_cm']}） | {_f(r['stop_s'])} "
                f"| {_f(r['slow_s'])} | {r['resumes']} | {_f(r['center_to_stop_s'], 3)} "
                f"| {_f(r['resume_after_center_s'], 3)} | {_f(tcp)} | {_f(r['accuracy'], 4)} "
                f"| **{r['center_to_continue']}** | {r['under']} / {r['over']}"
                f"（{r['wrong_at_transition']}） | {_f(r['delay_p95_ms'], 1)}"
                f"（{_f(r['rtt_p95_ms'], 1)}） | {_f(r['age_max_s'] * 1000, 0)} |")  # fmt: skip
    lines += ["", "## まとめ（6 本の合計）", "",
              "| 判定器 | 成功 | 判定の回数 | 正解率 | center→CONTINUE | 緩い | 厳しい "
              "| 誤りのうち切り替わり | 遅れ p95 の最大 ms | 古さ 最大 ms |",
              "|" + "---|" * 10]  # fmt: skip
    for n, t in res["totals"].items():
        lines.append(f"| {n} | {t['success']}/{t['episodes']} | {t['judge_n']} "
                     f"| {t['accuracy']:.4f} | **{t['center_to_continue']}** | {t['under']} "
                     f"| {t['over']} | {t['wrong_at_transition']}/{t['under'] + t['over']} "
                     f"| {t['delay_p95_ms_max']} | {t['age_max_ms']:.0f} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def totals(per_judge_rows: list[dict]) -> dict:
    n = sum(r["judge_n"] for r in per_judge_rows)
    acc = sum(r["accuracy"] * r["judge_n"] for r in per_judge_rows) / n
    return {
        "episodes": len(per_judge_rows), "success": sum(r["success"] for r in per_judge_rows),
        "judge_n": n, "accuracy": round(acc, 4),
        **{k: sum(r[k] for r in per_judge_rows)
           for k in ("center_to_continue", "under", "over", "wrong_at_transition")},
        "delay_p95_ms_max": max(r["delay_p95_ms"] for r in per_judge_rows),
        "age_max_ms": max(r["age_max_s"] for r in per_judge_rows) * 1000,
    }  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M4b: compare closed-loop runs by judge")
    ap.add_argument("runs", type=Path, nargs="+", help="the first one is the reference")
    args = ap.parse_args(argv)
    zones = load_m4a()["zones"]
    runs = [load_run(r) for r in args.runs]
    names = [r[0]["judge_name"] for r in runs]
    ids = [e["scenario"] for e in runs[0]]
    for n, r in zip(names, runs, strict=True):
        if [e["scenario"] for e in r] != ids:
            raise SystemExit(f"{n}: scenarios differ from the reference")
    rows = {n: [row(e, zones) for e in r] for n, r in zip(names, runs, strict=True)}
    res = {"runs": [str(p) for p in args.runs], "judges": names, "git": git_state(),
           "scenarios": {sc: {n: rows[n][k] for n in names} for k, sc in enumerate(ids)},
           "totals": {n: totals(rows[n]) for n in names}}  # fmt: skip
    out = new_run_dir("m4b_report")
    (out / "compare.json").write_text(json.dumps(res, indent=2) + "\n")
    md = compare_md(res)
    (out / "compare.md").write_text(md)
    print(md)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
