# jevlite-so101

A fast local judge for the SO-101 robot arm (LeRobot). It looks at a camera image and returns one of
`CONTINUE` / `SLOW` / `STOP`, with probabilities and latency in ms. A safety gate uses that decision
to limit the commands sent to the arm. All results below are from a Gazebo simulation (README in Japanese).

> [!WARNING]
> **カメラ画像による判定は、安全規格（ISO 13849・IEC 61508 など）でいう安全機能ではありません。**
> 実際のロボットで使うときは、認証された安全装置（非常停止、安全柵・ライトカーテン、安全 PLC など）と必ず併用してください。
> このリポジトリのゲートは、それらを置き換えるものではなく、その内側で指令を絞る補助です。

> [!IMPORTANT]
> **ここに載せた結果は、すべてシミュレーション（Gazebo Harmonic）でのものです。実機の SO-101・実カメラでは確かめていません。**
> 判定器 v1・v2 は、棒の色・太さ・照明が 1 通りのシミュレータの画像だけで学習・評価しています。

## 目的

Physical AI 講座 応用編1 最終課題の「＋α（α2）」です。

- 学習した方策（LeRobot の ACT）がロボットアームを動かしている間に、カメラ画像から作業領域への侵入を
  「継続／減速／停止」の 3 択で高速に判定する（Jev 型：選択肢・確率・遅延 ms だけを返す）。
- その判定で、方策の指令をセーフティ・ゲートが絞る（止める・遅くする・ゆっくり再開する）。
- 教育演習の雛形として使えるよう、再現性・テスト・記録を重視する（経過は下の「開発の段階」）。

提出動画は別途提出しています（このリポジトリには含めていません）。

## 構成

```
 カメラ画像 ──► 判定器 (v0 / v1 / v2) ──► Decision {choice, probs, t_mono, latency_ms}
                                                  │
 ACT の方策 ──► 目標の関節角 q_target ──► セーフティ・ゲート G1〜G9 ──► 送る関節角 q_cmd ──► SO-101（Gazebo）
                                                  ▲
                                     現在の関節角 q_now
```

### 方策：ACT（LeRobot 0.6.1）
- タスク：作業台の上のブロックを +y にまっすぐ押して、目標の枠に入れる（Gazebo、上からのカメラ 1 台）。
- お手本：計画した軌道（ROS 不要の計画 → Gazebo で実行・rosbag2 に記録）50 本。グリッパーを除いた 5 関節。
- 学習：100,000 ステップ（RTX 2000 Ada で約 4 時間 25 分）。

### セーフティ・ゲート（[gate/safety_gate.py](gate/safety_gate.py)、設定は `configs/gate.yaml`・`configs/gate_sim.yaml`）
選択肢は重大度順 `CONTINUE < SLOW < STOP`。起動直後は STOP。

| | 内容 |
|---|---|
| G1 期限切れ | Decision が無い、または画像が古い（`now - t_mono > stale_s`、既定 0.2 s）、未来の時刻 → STOP |
| G2 低確信 | `max(probs) < min_conf`（既定 0.6）→ 一段階安全側（CONTINUE→SLOW、SLOW→STOP） |
| G3 STOP | `q_cmd = q_now`（その場で保持） |
| G4 SLOW | `q_cmd = q_now + slow_alpha · (q_target − q_now)`（既定 0.3） |
| G5 CONTINUE | `q_cmd = q_target` |
| G6 ヒステリシス | 制限を緩める遷移は、遷移先の判定が `resume_hold_s`（既定 0.5 s）続いてから。強める遷移は即時 |
| G7 再開フック | STOP から抜けた瞬間に `on_resume` を 1 回呼ぶ（方策の `reset()` 用） |
| G8 レートリミッタ | 前回の指令からの 1 周期の変化を関節ごとの上限で縮める（方向を保つ）。SLOW 中は上限を `slow_speed_scale` 倍 |
| G9 追従誤差 | `\|q_cmd_prev − q_now\|` が関節ごとの上限を超えたら STOP |

NaN・inf・選択肢以外の値は STOP として扱います。詳しい仕様は [docs/spec.md](docs/spec.md)、テストは [tests/test_gate.py](tests/test_gate.py)。

### 画像の判定器（共通 API：`judge(image, t_capture=None) -> Decision`、[jevlite/api.py](jevlite/api.py)）
| | 中身 | 学習 |
|---|---|---|
| v0 | SigLIP（`google/siglip-base-patch16-224`）の画像‐テキスト類似度によるゼロショット | なし |
| 教師 | Ollama 上の VLM（gemma3:4b / qwen2.5vl:7b / qwen3.5:27b）。属性を答えさせ、選択肢はコードで決める | なし（オフラインのラベル付け用） |
| v1 | SigLIP の特徴（pooled ＋ 2×2 grid）＋ ロジスティック回帰 | シミュレータの画像 |
| v2 | MobileNetV3-Small（ImageNet の重みから微調整） | シミュレータの画像 |

シミュレータでは、侵入物は 1 本の棒です。棒の足元が白い面から離れていれば `none`→CONTINUE、
白い面の周りの帯（緑のテープ）にかかれば `edge`→SLOW、白い面にかかれば `center`→STOP が正解です。

### ディレクトリ
| | |
|---|---|
| `jevlite/` | 判定器（v0・教師・v1/v2）、Decision、カメラ入力 |
| `gate/` | セーフティ・ゲート |
| `scripts/` | 環境の記録、判定器の評価・学習、実カメラの撮影 |
| `scripts/sim/` | Gazebo の実行（お手本の生成、方策の評価、ゲート付きの閉ループ、判定サーバー）と図 |
| `scripts/video/` | 提出動画の下書き |
| `train/` | ACT の学習環境（LeRobot。メインの uv 環境とは別の uv プロジェクト） |
| `sim_ws/` | ROS 2 のワークスペース（SO-101 の URDF、Gazebo のワールド・launch） |
| `configs/` | 閾値・選択肢・プロンプト・シナリオなどの調整値（コードには書かない） |
| `tests/`・`train/tests/` | pytest（カメラ・GPU・Ollama・シミュレータなしで通る） |
| `docs/` | 仕様（[spec.md](docs/spec.md)）、作業台の配置、ラベルの定義、実機の手順（[real_robot_plan.md](docs/real_robot_plan.md)）、ゲートのデモの図 |

## 主な結果（シミュレーション）

数値は開発中の作業記録（このリポジトリには含めていません）から、そのまま載せています。

### 方策（ACT、100k ステップ、新しい種 100〜119 の 20 本）
| 成功 | 誤差（枠の中心から）平均 / 最大 | 推論の遅れ p50 / p95 | 刻み（30 Hz）に間に合わなかった回数 |
|---|---|---|---|
| 20/20 | 1.61 / 2.42 cm | 9.6 / 10.9 ms | 0 / 9,000 |

頑健さ（学習し直しなし）：お手本の範囲（x ±2.5 cm）の中ではブロックの x に合わせて押す位置を変える（傾き 0.90）が、
範囲の外では端で頭打ちになる（傾き ≈ 0）。初期位置を変えた 17 本で成功 11/17。押している途中にブロックをずらしても、
押す位置は直さない（n_action_steps 100・30 で成功 8/8 は、ずれても枠の内側に残ったため）。

### ゲート（正解の判定＋模擬の遅れ 50 ms、10 シナリオ）
- 成功 9/10。白い面に棒が入ってから 0.07〜0.10 s で STOP、棒が出てから約 0.8 s で再開し、方策を reset して押し切った。
- 失敗の 1 本（判定の遅れ 150 ms の模擬）は、G1 で 20 s ずっと STOP（安全側に止まり続けた正しい挙動）。
  止まらずに動ける判定の遅れは約 60 ms までで、これが判定器の目標 p95 ≤ 30 ms の根拠です。
- SLOW の直し（G8 の SLOW 中の上限と、SLOW 中は ACT の chunk を 0.3 倍の速さで取り出す）：SLOW 中の手先の速さは基準の
  0.29〜0.43 倍（直す前は 0.53〜0.90 倍）、SLOW を抜けた直後の跳び（直す前は基準の最大 2.3 倍）はなくなった。

### 画像の判定器（シミュレータの画像 9,896 枚。test 1,729 枚、エピソード単位で分割）
| 判定器 | 正解率 | 境目付近（1 cm 以内） | **center→CONTINUE** | 遅れ p50 / p95 |
|---|---|---|---|---|
| v0（ゼロショット、手向けのプロンプト） | 0.348 | 0.347 | 229 | 4.0 / 4.3 ms |
| v0（ゼロショット、棒向けのプロンプト） | 0.392 | 0.325 | 541 | 4.0 / 4.3 ms |
| v1（SigLIP grid ＋ LR、center の logit +1.5） | 0.917 | 0.825 | **0** | 3.0 / 3.4 ms |
| v2（MobileNetV3-Small） | **0.982** | **0.959** | **0** | 2.0 / 2.0 ms |

合格基準（推論 p95 ≤ 30 ms、center→CONTINUE 0 件）を v1・v2 とも満たしました。誤りはほぼすべて境目から 1 cm 以内です。
確信度だけでは誤りを止められません（危険側の誤りのうち G2 に拾われたのは v2 で 10 件中 1 件）。

### 閉ループ（画像の判定器をゲートに入れて同じ 6 シナリオ）
| 判定器 | 成功 | 判定の回数 | 閉ループの中の正解率 | **center→CONTINUE** | 判定の遅れ p95 | 判定の古さ 最大 |
|---|---|---|---|---|---|---|
| 正解（遅れ 50 ms を模擬） | 6/6 | 1,794 | 1.000 | 0 | 50 ms | 140 ms |
| v1 | 6/6 | 3,594 | 0.998 | **0** | 7.2 ms | 74 ms |
| v2 | 6/6 | 3,594 | 0.998 | **0** | 6.3 ms | 73 ms |

画像の判定器でも、正解の判定とほぼ同じ動き（STOP・SLOW の時間、再開までの時間、成否）になりました。
判定の誤りは、すべて棒が境目を横切る途中の 1〜2 枚でした。閉ループの中の遅れが単体（2〜3.4 ms）より大きいのは、
GPU を Gazebo の描画・ACT と分け合うためです。

### 実カメラ（参考、36 枚の小さなセッション）
実カメラ（Logitech C920n）で人の手を撮った画像では、v0 は使えず（正解率 0.24〜0.30）、教師の VLM も最良で 0.806
（qwen3.5:27b、参照画像つきの属性モード、p95 約 9 s）でした。この段階は一時停止しており、v1・v2 は実カメラ画像では学習・評価していません。
収録した画像はリポジトリに含めていません（`samples/private/` は git の管理外）。

## 開発の段階

| 段階 | 内容 | 主な結果 |
|---|---|---|
| Step 0〜1 | 雛形、v0（ゼロショット）、教師の VLM の疎通、セーフティ・ゲート（テスト先行。G1〜G7 の後に G8・G9 を追加） | ゲートの仕様をテストで固定 |
| 段階 2〜4 | 実カメラの入力（撮影時刻つき）、撮影、v0 と教師の評価 | 実カメラの 36 枚で最良 0.806（qwen3.5:27b）。ここで一時停止 |
| M1 | Gazebo の土台（SO-101 の URDF、作業台、カメラの位置） | 各関節が目標に届く・カメラ画像が出る |
| M2 | お手本の計画と生成（ROS 不要の計画 → Gazebo で実行・記録） | 50/50 成功 |
| M3 | LeRobotDataset への変換、ACT の学習（100k）、評価 | 新しい種で 20/20 |
| M3e | 頑健さ（範囲の外の初期位置、押している途中のずれ） | 11/17。範囲の外では頭打ち、途中のずれは直さない |
| M4a | ゲートを閉ループに組み込む（正解の判定＋模擬の遅れ） | 9/10（失敗の 1 本は遅れ 150 ms で止まり続けた） |
| M4a-S | SLOW を本当に遅くする（G8 の SLOW 中の上限、chunk を 0.3 倍の速さで取り出す） | SLOW 中は基準の 0.29〜0.43 倍、抜けた後の跳びなし |
| M4b | 画像の判定器 v1・v2（シミュレータの画像 9,896 枚）、閉ループ | test で center→CONTINUE 0、閉ループ 18/18 |
| M5 | 提出動画（別途提出） | – |
| 実機の準備 | `lerobot[feetech]`、接続設定の雛形、手順書（[docs/real_robot_plan.md](docs/real_robot_plan.md)） | モーターの ID 設定まで。キャリブレーション以降は未実施 |

## 限界と未確認事項

- **結果はすべてシミュレーションでのもので、実機の SO-101・実カメラでは確かめていません。** 実機の手順は [docs/real_robot_plan.md](docs/real_robot_plan.md) にあります。
- 判定器 v1・v2 は、棒が 1 種類・照明が 1 通りのシミュレータの画像だけで学習・評価しました。人の手・影・照明の変化には未対応です。
  実カメラでは、v0 も教師の VLM も十分な正解率が出ていません（最良 0.806、p95 約 9 s）。
- 誤りは境目付近に集まり、確信度（G2）ではほとんど拾えません。v2 は test で帯（edge）を CONTINUE と答えた誤りが 8 件ありました。
  帯を見落としても白い面に入れば STOP になりますが、帯の上で減速しないまま白い面に近づく時間ができます。
- 判定の遅れが約 60 ms を超えると（serial の方式）、G1 で止まり続けて動けません。pipelined の方式は試していません。
- 方策（ACT）は、お手本の範囲の外の初期位置に合わせられず、押している途中のずれも直しません。直すには、お手本を足して学習し直す必要があります。
- SLOW 中の G8 の上限は、シミュレーションでは一度も効きませんでした（テストでのみ確認）。SLOW 中の行動の補間が、実機のサーボの分解能に対して十分に細かいかは未確認です。
- `configs/gate_sim.yaml` の `max_speed`・`max_follow_err` はシミュレータ用の値です（追従誤差の上限 0.48 rad は大きい）。実機の値は、実機の追従を測ってから決めます。
- 閉ループで試した棒の通り道は 1 本の線だけです。
- 再現の手順のうち、新しいフォルダで最初から確かめたのは、`uv sync`（メインと `train/`）・メッシュの取得・既定のテスト
  （メイン 627 件、`train/` 30 件）・ruff までです。メッシュを取得する前は、メッシュを使う 16 件のテストが飛ばされます（失敗はしません）。
  Gazebo・GPU・実機を使う手順（ビルド、お手本・データの生成、学習、評価、閉ループ、動画）は、新しいフォルダでは通していません
  （記録とスクリプトの引数から書いたものです）。

## 動作環境

確かめた環境：

- Ubuntu 24.04（x86_64）、RAM 32 GB
- NVIDIA RTX 2000 Ada 16 GB（Driver 580、CUDA 13.0。PyTorch は cu128 のホイール）。Gazebo のカメラの描画にも GPU を使います
- [uv](https://docs.astral.sh/uv/)（Python 3.12。依存の版は `uv.lock`・`train/uv.lock` で固定）
- ROS 2 Jazzy（`/opt/ros/jazzy`）、Gazebo Harmonic、ros_gz、gz_ros2_control、ros2_controllers、xacro、robot_state_publisher、
  rosbag2 の mcap ストレージ
- ffmpeg と日本語のフォント（`fc-match` で見つかるもの。動画の下書きだけ）
- v4l-utils（`v4l2-ctl`。実カメラを使うときだけ）
- Ollama（教師の VLM を使うときだけ。localhost:11434）

ROS の部分は、例えば次のパッケージで揃います（環境に合わせて読み替えてください。sudo が要ります）。

```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-gz-ros2-control ros-jazzy-ros2-controllers \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-rosbag2-storage-mcap \
  python3-colcon-common-extensions ffmpeg fonts-noto-cjk v4l-utils
```

`~/.bashrc` には ROS の source を書かず、ROS を使うコマンドは毎回 `source /opt/ros/jazzy/setup.bash && …` で実行します
（uv の環境と混ぜないため）。

## セットアップ

```bash
uv sync                    # メインの環境（判定器・ゲート・評価・図）
uv sync --project train    # ACT の学習環境（LeRobot。動画の下書きもここ）
bash scripts/install_hooks.sh   # 秘密情報のチェック（下の節）
python3 scripts/sim/fetch_so101_assets.py   # SO-101 のメッシュ（再現の手順 1。無いとテストの一部が飛ばされる）
uv run ruff check .
uv run pytest -q           # hw / ollama / gpu / sim のテストは既定で除外
(cd train && uv run pytest -q)   # train の環境のテスト
uv run python scripts/check_env.py   # 環境の記録 → docs/env_report.json
```

マーカーつきのテスト：`uv run pytest -q -m gpu`、`-m ollama`（Ollama が必要）、`-m hw`（カメラ）、`-m sim`（Gazebo）。

## 学習済みの重み・データについて

**学習済みの重み・データ・実行の記録・動画は配布していません。** `data/`・`models/`・`outputs/` は git の管理外で、
SO-ARM100 のメッシュも取得スクリプトで取り寄せます。下の手順で、すべて作り直せます。

- お手本・棒の置き方・データの分割の乱数の種は `configs/` にあり、同じ設定なら同じ計画になります。
  ただし Gazebo の物理と描画の時間の揺れ、GPU の非決定性があるため、数値は完全には一致しません。
- 実行のフォルダは `<日時>_<名前>` で作られ、上書きしません。次の段階で使うフォルダは、`configs/` の該当の項目
  （下の手順に書いた場所）を、自分の実行のフォルダに書き換えてください。

## 再現の手順

どの環境で動かすかを各手順に書いています。

- **ROS**：`source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && …`（システムの Python と rclpy）
- **uv**：`uv run python …`（メインの環境）
- **train**：`uv run --project train python …`（LeRobot の環境）

Gazebo はすべて headless（GUI なし）で動きます。長い実行は tmux などで、`timeout` を付けて流してください。
所要時間は上の環境での目安です（シミュレータは実時間の約 0.38 倍で進みます）。

### 1. メッシュの取得（uv 不要）
SO-ARM100 の固定コミットから 13 個（約 15 MB）を取り寄せ、sha256 を照合します。メッシュが無いままシミュレーションを起動すると、
launch がこのコマンドを表示して止まります。

```bash
python3 scripts/sim/fetch_so101_assets.py
```

### 2. ビルド（ROS）

```bash
source /opt/ros/jazzy/setup.bash && cd sim_ws && colcon build --symlink-install
```

### 3. シミュレータの確認（ROS、3 分以内）
各関節が目標に届くこと・カメラ画像が出ることを確かめます。結果は `outputs/sim/m1_result.json` と `outputs/sim/m1_camera.png`。

```bash
source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
  timeout 180 python3 scripts/sim/check_m1.py
```

同じ確認を pytest から行うには `uv run pytest -q -m sim`。作業台とロボットの配置は [docs/sim_layout.md](docs/sim_layout.md)、
届く範囲は `uv run python scripts/sim/check_reach.py`、カメラの位置の計算は `uv run python scripts/sim/fit_camera.py` です
（どちらも ROS 不要。結果は `configs/sim.yaml` にすでに書いてあります）。

### 4. お手本の生成（ROS、50 本・約 32 分・約 280 MB）

```bash
source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
  timeout --signal=INT --kill-after=30 3600 python3 scripts/sim/gen_demos.py \
    --episodes 50 --out data/sim_demos/m2_50eps
```

設定は `configs/sim.yaml` の `demo`（種 0〜49、物理エンジン dartsim）。出力の場所は `configs/train_act.yaml` の `dataset.src` と同じです。

### 5. LeRobotDataset への変換（train、数分）

```bash
uv run --project train python train/convert_bags.py      # → data/lerobot/so101_sim_push_m2_arm5（push しない）
uv run --project train python train/check_dataset.py     # 件数・正規化の統計・図 → outputs/sim/<日時>_m3a_check/
```

### 6. ACT の学習（train、100k ステップ・約 4.5 時間）

```bash
uv run --project train python train/train_act.py --name m3c_100k
uv run --project train python train/plot_progress.py models/act/<日時>_m3c_100k   # 損失の図
```

設定は `configs/train_act.yaml` の `train`。止まったときは `--resume models/act/<日時>_m3c_100k` で最後のチェックポイントから再開します。
学習が終わったら、`configs/m4a.yaml` と `configs/robustness.yaml` の `policy.checkpoint` を
`models/act/<日時>_m3c_100k/lerobot/checkpoints/100000` に書き換えます。

### 7. 方策の評価（ROS、20 本・約 17 分／頑健さ 40 本・約 35 分）

```bash
source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
  timeout --signal=INT --kill-after=30 3600 python3 scripts/sim/eval_policy.py \
    --checkpoint models/act/<日時>_m3c_100k/lerobot/checkpoints/100000 --name ck100k

source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
  timeout --signal=INT --kill-after=60 5400 python3 scripts/sim/robustness.py
uv run python scripts/sim/robustness_report.py outputs/sim/<日時>_m3e_start outputs/sim/<日時>_m3e_perturb
```

方策は train の環境の別のプロセス（`train/policy_server.py`）で動き、評価のスクリプトが起動・停止します。
種は `configs/sim.yaml` の `eval`、頑健さの条件は `configs/robustness.yaml`。

### 8. ゲート付きの閉ループ（正解の判定、ROS、10 シナリオ・約 12 分）

```bash
source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
  timeout --signal=INT --kill-after=30 3600 python3 scripts/sim/gate_demo_sim.py
uv run python scripts/sim/gate_demo_report.py outputs/sim/<日時>_m4a_gate
```

シナリオは `configs/m4a.yaml`、ゲートの設定は `configs/gate_sim.yaml`。`--dry-run` で一覧と見込みの時間だけを表示します。
ハードウェアもシミュレータも使わないゲートのデモは `uv run python scripts/sim_gate_demo.py` です。

### 9. 判定器のデータと学習（データ：ROS、120 本・約 75 分・約 400 MB／学習・比較：uv、GPU、数分）

```bash
source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
  timeout --signal=INT --kill-after=30 7200 python3 scripts/sim/gen_intruder_data.py --name m4b

uv run python scripts/train_student.py v1 --data data/sim_intruder/<日時>_m4b --name grid
uv run python scripts/train_student.py v2 --data data/sim_intruder/<日時>_m4b
uv run python scripts/eval_sim_judges.py --data data/sim_intruder/<日時>_m4b \
  --v1 models/judge/<日時>_v1_grid --v2 models/judge/<日時>_v2 --split test
```

設定は `configs/judge_sim.yaml`（データの数・分割、切り出し、v1・v2）。学習は train、モデルの選択は val を使い、test は比較で 1 回だけ使います。
v1 の center の logit +1.5（`logit_bias.v1`）は、今回のデータの val で「center→none が 0 になる最小の値」として決めたものです。
作り直したデータでは、val の結果（`models/judge/<日時>_v1_grid/meta.json`）を見て決め直してください。
学習したモデルのフォルダを `configs/m4b.yaml` の `judges.v1.model`・`judges.v2.model` に書きます。

### 10. 閉ループ（画像の判定器、ROS、3 回 × 約 7 分）

```bash
source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
  timeout --signal=INT --kill-after=30 3600 python3 scripts/sim/gate_demo_sim.py --judge oracle --m4b
source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
  timeout --signal=INT --kill-after=30 3600 python3 scripts/sim/gate_demo_sim.py --judge v1
source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
  timeout --signal=INT --kill-after=30 3600 python3 scripts/sim/gate_demo_sim.py --judge v2

uv run python scripts/sim/m4b_report.py outputs/sim/<日時>_m4b_oracle outputs/sim/<日時>_m4b_v1 outputs/sim/<日時>_m4b_v2
uv run python scripts/sim/gate_demo_report.py outputs/sim/<日時>_m4b_v2 --compare outputs/sim/<日時>_m4b_oracle
```

画像の判定器は、判定サーバー（`scripts/sim/judge_server.py`、メインの uv 環境）で動き、`gate_demo_sim.py` が起動・停止します。
判定器自身の遅れ（実時間）を、そのまま sim 時間の遅れとして使います。

### 11. 動画の下書き（train、約 2.5 分）

`configs/video.yaml` の `sources`（素材の実行フォルダと bag）と、図の区間の `chart.report`（頑健さの report のフォルダ）を、自分の実行のフォルダに書き換えてから：

```bash
uv run --project train python scripts/video/make_video.py   # → outputs/video/<日時>/draft.mp4
```

## 実機（SO-101）の準備

`train/` の環境に `lerobot[feetech]` を入れています。届いた日の手順（ポートの確認、モーターの ID、キャリブレーション、テレオペ）、
LeRobot 0.6.1 の関節値の単位、キャリブレーションの後にやること（rad ↔ 度の符号とオフセット、ゲートの値の決め方）は
[docs/real_robot_plan.md](docs/real_robot_plan.md) にあります。接続の設定は `configs/so101_real.example.yaml` を
`configs/so101_real.yaml` にコピーして埋めます（ポートの名前にシリアル番号が入るため、`so101_real.yaml` は git の管理外）。
**モーターを動かす・トルクをかける・モーターに書き込むコマンドは、作業者が自分のターミナルで実行します**（開発に使った AI のツールには実行させていません）。

## 実カメラで試した段階（参考）

`configs/camera.example.yaml` を `configs/camera.yaml` にコピーして、自分のカメラに合わせます（`camera.yaml` は git の管理外）。

```bash
uv run python scripts/capture_samples.py --label none --preview    # 撮影 → samples/private/（git の管理外）
uv run python scripts/run_zeroshot_on_images.py --dir samples/     # v0 の判定と遅れ
uv run python scripts/eval_samples.py --session <セッション>        # v0 と教師の比較（Ollama）
```

ラベルの定義は [docs/labeling.md](docs/labeling.md)。教師の呼び出し方とプロンプトは `configs/teacher.yaml`・`configs/eval.yaml`。

## 秘密情報のチェック（最初のコミットの前に）

Git のフックは `git clone` で複製されないため、クローンごとに pre-commit フックを入れます。

```bash
bash scripts/install_hooks.sh
```

`scripts/hooks/pre-commit` を `.git/hooks/` にコピーし、このマシンで見つけた値（ホスト名、ユーザー名、IP アドレス、
カメラのシリアル番号）を `.git/info/secret-patterns`（git の管理外）に書きます。値は画面に出しません。
ほかの値は 1 行に `ラベル<TAB>値` で書き足します。

```bash
printf 'personal-email\t%s\n' 'you@example.com' >> .git/info/secret-patterns
```

ステージした変更にこれらの値があるとコミットを拒み、ラベル・件数・ファイルだけを表示します。`192.168.` や `/home/` のような
一般的な文字列は、文書やテストに出てくるため、わざと入れていません。チェックと `git commit` を 1 つのコマンドにつながず、結果を見てからコミットしてください。

## AI の利用について

このリポジトリは、Anthropic の **Claude Code**（AI のコーディング支援）を使って開発しました。
コード・テスト・文書の多くは Claude Code が書き、作者が方針の指示・計画の承認・結果の確認を行いました。
長時間の実行（お手本・データの生成、学習、閉ループのバッチ）と GUI の操作は作者が行っています。
Claude Code には進め方のルール（sudo や実機のモーターの操作は作者が行う、ゲートの変更はテストを先に書く、調整値は `configs/` にだけ置く、など）を与えて作業しました。
[docs/spec.md](docs/spec.md) の仕様は、そのときの指示書から文言を変えずに移したものです。

## ライセンス

[Apache License 2.0](LICENSE)。第三者の素材（SO-ARM100 の URDF・メッシュ、LeRobot、SigLIP、torchvision の重み、教師の VLM など）の
出典とライセンスは [NOTICE](NOTICE) にまとめています。ロボットのモデルは TheRobotStudio/SO-ARM100（Apache-2.0）の
`so101_new_calib.urdf` で、変更点は [sim_ws/src/so101_description/NOTICE](sim_ws/src/so101_description/NOTICE) にあります。
