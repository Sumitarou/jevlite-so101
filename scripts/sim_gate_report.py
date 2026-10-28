"""Compare the four G8/G9 settings on the sim_gate_demo scenario for the report.

    uv run python scripts/sim_gate_report.py [--out docs/figures] [--no-figures] [--sweep]

Writes summary.csv and summary.md (and the four figures unless --no-figures) to --out.
--sweep also prints how max_follow_err trades detection time against false stops.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from gate.safety_gate import GateConfig, Reason, load_gate_config
from jevlite.config import load_config
from scripts.sim_gate_demo import (
    compute_metrics,
    plot,
    simulate,
    without_follow_check,
    without_rate_limit,
)


@dataclass(frozen=True)
class Variant:
    key: str
    rate_limit: bool
    follow_check: bool
    note: str

    @property
    def figure(self) -> str:
        g8 = "g8on" if self.rate_limit else "g8off"
        g9 = "g9on" if self.follow_check else "g9off"
        return f"sim_gate_demo_{self.key}_{g8}_{g9}.png"

    def gate_cfg(self, base: GateConfig) -> GateConfig:
        cfg = base if self.rate_limit else without_rate_limit(base)
        return cfg if self.follow_check else without_follow_check(cfg)


VARIANTS = [
    Variant("1", False, False, "G8・G9 なし"),
    Variant("2", True, False, "G8 の効果だけを見る"),
    Variant("3", True, True, "本番の設定"),
    Variant("4", False, True, "考察用：再開時の跳びで G9 が作動"),
]

COLUMNS = [
    "variant",
    "g8",
    "g9",
    "max_cmd_step",
    "steady_track_err",
    "obstacle_detected",
    "detect_time_s",
    "unnecessary_stops",
    "restops_within_window",
    "continue_ratio",
    "figure",
]


def run_variants(base: GateConfig, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for v in VARIANTS:
        res = simulate(v.gate_cfg(base), scenario)
        rows.append(
            {
                "variant": v.key,
                "g8": v.rate_limit,
                "g9": v.follow_check,
                "note": v.note,
                "figure": v.figure,
                **compute_metrics(res, scenario),
                "_res": res,
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["g8"], out["g9"] = int(r["g8"]), int(r["g9"])
            out["obstacle_detected"] = int(r["obstacle_detected"])
            out["detect_time_s"] = "" if r["detect_time_s"] is None else f"{r['detect_time_s']:.2f}"
            for k in ("max_cmd_step", "steady_track_err", "continue_ratio"):
                out[k] = f"{r[k]:.3f}"
            w.writerow(out)


def _onoff(b: bool) -> str:
    return "あり" if b else "なし"


def write_summary_md(rows: list[dict[str, Any]], scenario: dict[str, Any], path: Path) -> None:
    m = scenario["metrics"]
    w0, w1 = m["steady_window"]
    lines = [
        "# G8 / G9 の比較（sim_gate_demo の合成シナリオ）",
        "",
        "`uv run python scripts/sim_gate_report.py` で生成（`summary.csv` と同じ内容）。"
        "閾値は `configs/gate.yaml`（仮の値）、シナリオは `configs/sim_demo.yaml`。",
        "",
        "| 図 | G8 | G9 | 指令の1ステップあたりの最大変化 | 定常区間の平均追従誤差 "
        "| 障害物の検出 | 検出までの時間 [s] | 不要なSTOP | CONTINUE の時間割合 | 備考 |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        det = "あり" if r["obstacle_detected"] else "なし"
        dt = "—" if r["detect_time_s"] is None else f"{r['detect_time_s']:.2f}"
        lines.append(
            f"| [{'①②③④'[int(r['variant']) - 1]}]({r['figure']}) | {_onoff(r['g8'])} "
            f"| {_onoff(r['g9'])} | {r['max_cmd_step']:.2f} | {r['steady_track_err']:.2f} "
            f"| {det} | {dt} | {r['unnecessary_stops']} | {r['continue_ratio'] * 100:.1f} % "
            f"| {r['note']} |"
        )
    lines += [
        "",
        "## 図の見方",
        "- ③の 1.82 s に見える指令（q_cmd）の縦の段差は、G9 の STOP で指令を実際の位置（q_now）"
        "に戻したもので、腕は動いていない（障害物で止められたまま）。",
        "",
        "## 指標の定義",
        "- **指令の1ステップあたりの最大変化**：STOP 以外の周期で、隣り合う周期の q_cmd の差"
        "（全関節の最大、単位は関節値）。STOP で q_now に戻る変化は意図した動作なので除く。"
        "G8 の上限は関節ごとに max_speed × 周期（50 Hz で 5 関節 1.2、グリッパー 2.0）なので、"
        "②の 1.53 はグリッパーの値で、上限内。",
        f"- **定常区間の平均追従誤差**：{w0}〜{w1} s の |q_target − q_now| の平均（全関節）。"
        "最後の再開 7.9 s の直後にある、意図した追いつき動作は含めない。",
        "- **障害物の検出／検出までの時間**：障害物の区間内で reason = follow_error になった最初の"
        "周期と、区間開始との差。",
        f"- **不要なSTOP**：再開から {m['restop_window_s']} s 以内の再停止のうち、判定器の"
        "有効な STOP 判定（reason = ok）によるものを除いた回数。",
        "  再開から 1 s 以内の再停止の総数（判定器の STOP を含む）は CSV の "
        "`restops_within_window` にある（③では 3.0 s の手の侵入による 1 回がここに入る）。",
        "- **CONTINUE の時間割合**：10 s のうちゲート状態が CONTINUE だった周期の割合。",
        "",
    ]
    path.write_text("\n".join(lines))


def follow_err_sweep(
    base: GateConfig, scenario: dict[str, Any], thresholds: list[float]
) -> list[dict[str, Any]]:
    """G8+G9 with every joint's max_follow_err set to each threshold (gripper keeps 2x)."""
    obs = scenario["obstacles"]
    out = []
    for th in thresholds:
        n = len(base.max_follow_err)
        cfg = replace(base, max_follow_err=(th,) * (n - 1) + (2 * th,))
        res = simulate(cfg, scenario)
        m = compute_metrics(res, scenario)
        outside = [
            float(t)
            for k, t in enumerate(res.t)
            if res.reason[k] is Reason.FOLLOW_ERROR
            and not any(o["start"] <= t < o["end"] for o in obs)
        ]
        out.append(
            {
                "max_follow_err": th,
                "detect_time_s": m["detect_time_s"],
                "false_follow_error_cycles": len(outside),
                "unnecessary_stops": m["unnecessary_stops"],
                "continue_ratio": m["continue_ratio"],
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument("--lang", choices=["ja", "en"], default="ja")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="print the max_follow_err trade-off")
    args = ap.parse_args(argv)

    scenario = load_config("sim_demo.yaml")
    base = load_gate_config()
    rows = run_variants(base, scenario)
    args.out.mkdir(parents=True, exist_ok=True)
    write_summary_csv(rows, args.out / "summary.csv")
    write_summary_md(rows, scenario, args.out / "summary.md")
    print(f"-> {args.out / 'summary.csv'}\n-> {args.out / 'summary.md'}")
    if not args.no_figures:
        variants = {v.key: v for v in VARIANTS}
        for r in rows:
            v = variants[r["variant"]]
            png = plot(r["_res"], scenario, args.out / v.figure, args.lang, v.rate_limit,
                       v.follow_check)  # fmt: skip
            print(f"-> {png}")
    if args.sweep:
        print("\nmax_follow_err  detect[s]  false_FE_cycles  unnecessary_stops  continue%")
        for s in follow_err_sweep(base, scenario, [1, 1.5, 2, 3, 5, 8, 10, 15, 20, 30, 40]):
            d = "-" if s["detect_time_s"] is None else f"{s['detect_time_s']:.2f}"
            print(f"{s['max_follow_err']:>14}  {d:>9}  {s['false_follow_error_cycles']:>15}  "
                  f"{s['unnecessary_stops']:>17}  {s['continue_ratio'] * 100:8.1f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
