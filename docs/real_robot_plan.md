# SO-101 実機の準備と最初の手順

LeRobot 0.6.1（`train/` の環境、`lerobot[feetech]`）で SO-101 の leader / follower を動かすまでの手順と、
その後にシミュレータの成果（ACT の方策・ゲート）を実機に合わせるためにやることをまとめる。

**モーターを動かす・トルクをかける・モーターに書き込むコマンドは、すべてユーザーが自分のターミナルで実行する**（開発に使った AI のツール（Claude Code）には実行させない）。
コマンドは `train/` で `uv run …` の形で実行する。`<FOLLOWER_PORT>` などは `configs/so101_real.yaml`（git の管理外）に控えた値に置き換える。

## 0. 設定ファイル

```bash
cp configs/so101_real.example.yaml configs/so101_real.yaml   # ポートとカメラを埋める。git の管理外
```

- ポートは `/dev/ttyACM*` ではなく `ls -l /dev/serial/by-id/` の名前を使う（抜き差しで番号が入れ替わるため）。by-id の名前にはシリアル番号が含まれるので、リポジトリ・docs・ログに書かない。
- by-id の名前のシリアル番号の部分を、pre-commit フックの値の一覧（`.git/info/secret-patterns`）に追加する（ユーザーが行う。値を画面に出さない）。
- `/dev/ttyACM*` を開く権限がない場合は `dialout` グループへの追加が必要（sudo なのでユーザーが行う）。

## 1. 届いた日の手順

1. **電源とモーターのラベル貼り**
   - leader と follower の電源アダプタは電圧が違うことがある（キットによる）。取り違えないよう、アダプタとコントローラ基板にラベルを貼る。
   - モーターは組み立て前に、関節名（shoulder_pan=1 … gripper=6）と leader / follower のラベルを貼る。leader 用と follower 用でギア比が違うモーターが混ざっている場合があるので、キットの説明どおりに分ける。
2. **ポートを調べる**：`uv run lerobot-find-port`
   - 一覧を出した後、「USB を抜いて Enter」と言われる。抜いて Enter を押すと、消えたポートが表示される。leader・follower それぞれの基板で 1 回ずつ行い、`configs/so101_real.yaml` に by-id の名前で書く。
3. **モーターの ID とボーレートの設定（follower → leader の順）**
   ```bash
   uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=<FOLLOWER_PORT>
   uv run lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>
   ```
   - gripper → wrist_roll → … → shoulder_pan の順に、「その 1 個だけを基板につないで Enter」を繰り返す（デイジーチェーンにつながず 1 個ずつ）。
4. **組み立て**（キットの説明に従う。各モーターがラベルの関節に付いていることを確かめる）
5. **キャリブレーション**（follower → leader）
   ```bash
   uv run lerobot-calibrate --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=follower_arm
   uv run lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=leader_arm
   ```
   - トルクが切れた状態で、腕を「各関節の可動範囲の中央」に置いて Enter（その位置がモーターの 2047 になるよう homing の補正値が書かれる）。
   - 次に wrist_roll 以外の関節を、端から端までゆっくり動かして Enter（その間の最小・最大を記録する。wrist_roll は 0〜4095 に固定）。
   - `id` はキャリブレーションファイルの名前になる。以降のコマンドでも同じ `id` を使う。
6. **テレオペ**
   ```bash
   uv run lerobot-teleoperate \
     --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=follower_arm \
     --robot.max_relative_target=5.0 \
     --teleop.type=so101_leader --teleop.port=<LEADER_PORT> --teleop.id=leader_arm
   ```
   - 最初は `max_relative_target=5.0`（度/刻み。最初の動作確認用の仮の値）で、leader をゆっくり動かす。gripper も同じ 5（0〜100 の単位）で切られる。
   - 整数の `5` ではなく `5.0` と書く（0.6.1 は float か dict しか受け付けず、int だと TypeError）。
   - 止めるときは Ctrl+C（`disable_torque_on_disconnect: true` で follower のトルクが切れて腕が落ちるので、手で支えるか、低い姿勢で止める）。

## 2. LeRobot 0.6.1 の関節値の単位

- 既定は `use_degrees=True`（leader・follower とも）。
  - 腕の 5 関節：**度**。`(生の値 − 中央) × 360 / 4095`。中央 = キャリブレーションで記録した可動範囲の (最小 + 最大) / 2。
  - gripper：**0〜100**（記録した範囲の最小 = 0、最大 = 100）。
- 0° は URDF のゼロではなく、関節ごとの「記録した範囲の中央」。キャリブレーションで中央に置いた姿勢とも、範囲が左右非対称なら一致しない。
- 度のモードでは drive_mode（向きの反転）が使われないので、URDF と向きが逆の関節があり得る。
- 度のモードでは LeRobot はソフトウェアで値を切り詰めない。範囲を守るのは、キャリブレーションでモーターに書かれた Min/Max_Position_Limit。

## 3. キャリブレーションの後にやること

### 3.1 関節ごとの符号とオフセット（rad ↔ 度）
方策とシミュレータは rad（URDF の関節角）、実機は上の度で動く。`q_deg = sign_j × rad2deg(q_rad) + offset_j` の `sign_j`・`offset_j` を実機で求める。

1. シミュレータの初期姿勢（`configs/sim.yaml` の `demo.home_tcp` から IK で解いた姿勢。`scripts/sim/demo_plan.py` の `Planner.q_home`）：

   | 関節 | rad | 度 |
   |---|---|---|
   | shoulder_pan | +0.0224 | +1.28 |
   | shoulder_lift | −0.2341 | −13.42 |
   | elbow_flex | +0.3102 | +17.77 |
   | wrist_flex | +1.4947 | +85.64 |
   | wrist_roll | −1.4997 | −85.93 |
   | gripper | −0.16（固定、閉じたまま） | — |

   （2026-10-13 に現在の `configs/sim.yaml` から計算した値。sim.yaml を変えたら計算し直す）
2. トルクを切った follower を手でこの姿勢に合わせ、各関節の度の値を記録する（読み取りだけのスクリプトを用意し、ユーザーが実行する）。合わせ方は、シミュレータの同じ姿勢の画像と見比べる。
3. 符号は、関節を 1 つずつ URDF の + 向きに少し動かして、度の値が増えるか減るかで決める。オフセットは `offset_j = q_deg(初期姿勢) − sign_j × rad2deg(q_home_j)`。手合わせの誤差を減らすため、初期姿勢のほかにもう 1〜2 姿勢（例：URDF のゼロに近い姿勢）で確かめる。
4. `train/units.py` に実機の変換（rad → 度、度 → rad）を追加する。**テストを先に書く**（往復で元に戻る、符号・オフセットが configs の値どおり、NaN・範囲外の扱い）。符号とオフセットは configs の YAML に置く（コードに直書きしない）。

### 3.2 グリッパー
sim の −0.16 rad（閉じた状態。押す動作の間は動かさない）に対応する 0〜100 の値を実機で決める。閉じた状態で爪の先の隙間が sim と同じ約 2 mm になる値を目安にし、握り込んでモーターに負荷がかかり続けない値にする。

### 3.3 ゲートの max_speed・max_follow_err
`configs/gate.yaml` の `max_speed`（度/秒、gripper は 0〜100/秒）と `max_follow_err`（度）は仮の値。テレオペや低速の指令で、関節ごとの追従の遅れ（指令 − 現在位置）と速さを記録し、sim と同じ決め方（`configs/gate_sim.yaml` のコメント）で決め直す。

## 4. キャリブレーションのファイル

- 保存場所：`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`、`…/teleoperators/so_leader/<id>.json`（環境変数 `HF_LEROBOT_CALIBRATION` で変えられる）。リポジトリの外なので git には入らない。
- 中身：関節ごとの homing の補正値と、記録した範囲の最小・最大。補正値と範囲はモーター自身にも書かれている。
- **消えた場合**：LeRobot は接続時にキャリブレーションをやり直させる。やり直すと範囲の中央（＝0°）が変わるので、3.1 の符号・オフセット、3.2 のグリッパーの値、実機で集めたデータの関節値がずれる。
  - 防ぐため、キャリブレーションの直後に json を `data/real/calibration/`（git の管理外）にコピーしておく。消えたらそこから戻す。
  - やり直した場合は、3.1 と 3.2 を測り直し、それ以前の実機のデータには、どのキャリブレーションのものかを記録に残す。

## キャリブレーション手順（2026-10 版）

### 前提
- モーター12個の ID 設定は完了（フォロワー F.1〜F.6、リーダー L.1〜L.6）。フォロワー（白）は組み立て済み、リーダー（黒）は未組み立てなら先に組む
- フォロワー：STS3215 C047（12V・1/345）×6、12V・2.0A のアダプター（F.12V）、F 基板
- リーダー：C046（1/147）ID 4・5・6、C044（1/191）ID 1・3、C001（1/345）ID 2、5V・4.0A のアダプター（L.5V）、L 基板
- **12V をリーダーにつながない**（7.4V のモーターを壊す）
- 基板：Seeed Bus Servo Driver Board for XIAO V1.0。端子台は裏面の印字で左が −（黒）、右が ＋（赤）。TTL Mode ジャンパーはキャップあり
- モーターを動かす・書き込むコマンドはユーザーが自分のターミナルで実行する（AI のツールには実行させない）

### 1. 場所と固定
- 台座の回転軸を中心に半径約45cm、机から高さ約60cmに物を置かない（腕をまっすぐ伸ばして実測し、5〜10cm 足すのが確実）
- 関節を1つずつ動かすときは、他の関節を折りたたむと半径30cm程度に収まる
- 台座をクランプ2個で机に固定する
- 最終位置にする場合：白い面（32×20cm）の右辺の外、緑の帯の外縁から1cm。カメラのスタンドやブロックは範囲の外へ
- 電源・USB のケーブルは腕に巻き込まれないよう台座の後ろへ逃がす

### 2. 組み立ての確認（電源を入れる前）
- どの関節も、部品を動かすとモーターの軸が回ること
- 空回りする場合の原因：ホーンがスプラインにかみ合っていない／中心のネジ M3×6 がない／部品がホーンに止まっていない／出力軸側と受け側を逆に付けた
- ケーブルが 1→2→…→6 と全部つながり、1 の空いている口から基板へ1本出ている

### 3. フォロワーのキャリブレーション
1. F 基板に F.1 からのケーブル → F.12V をコンセントへ（10秒様子を見る）→ USB-C で HP Z2 へ（ハブ不可、リーダー基板は抜く）
2. リポジトリの根から `cd train && uv run lerobot-find-port`（USB だけ抜いて Enter → ポートを控えて差し直す。詳しくは「1. 届いた日の手順」の 2）
3. `uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=jevlite_follower`
   - 操作は「1. 届いた日の手順」の 5 と同じ（すべての関節を可動範囲のおおよそ真ん中に置いて Enter → wrist_roll 以外を1つずつ端から端まで動かす → Enter）
   - 終了後もトルクは切れたまま。腕を支えて下ろす
4. 控え（理由は「4. キャリブレーションのファイル」）：
   ```bash
   mkdir -p ../data/real/calibration   # train/ にいるので、リポジトリの根の data/（git の管理外）は ../data/
   cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/jevlite_follower.json ../data/real/calibration/
   ```

### 4. リーダーのキャリブレーション
1. F 基板の USB と電源を抜く → L 基板に L.1 からのケーブル → L.5V をコンセントへ → USB-C で HP Z2 へ
2. `uv run lerobot-find-port` でポートを確認
3. `uv run lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=jevlite_leader`（手順はフォロワーと同じ）
4. 控え：`cp ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/jevlite_leader.json ../data/real/calibration/`（train/ で実行）

### 5. 両方つないだときのポートの固定
- 同時につなぐと ttyACM0/1 が入れ替わることがあるので `/dev/serial/by-id/usb-1a86_USB_Single_Serial_<SERIAL>-if00` で指定する
- どちらの by-id がどちらの基板かは、片方ずつ抜き差しして確かめる
- 値の書き場所（configs/so101_real.yaml、git の対象外）と、シリアル番号を .git/info/secret-patterns にユーザー自身が追加することは「0. 設定ファイル」を参照

### 6. テレオペで動作確認
- follower の max_relative_target は 5.0 から始める（float で書く理由は「1. 届いた日の手順」の 6）
- 例：
  ```bash
  uv run lerobot-teleoperate \
    --robot.type=so101_follower --robot.port=<F の by-id> --robot.id=jevlite_follower --robot.max_relative_target=5.0 \
    --teleop.type=so101_leader --teleop.port=<L の by-id> --teleop.id=jevlite_leader
  ```
- 最初はリーダーをゆっくり小さく動かす。手を動作範囲に入れない。電源プラグをすぐ抜ける位置に置く
- Ctrl+C で止めるとフォロワーのトルクが切れて腕が落ちる → 止める前に下ろした姿勢にするか手で支える

### 7. その後
- 実機をシミュレータの初期姿勢（「3.1 関節ごとの符号とオフセット」の表）に合わせて度の値を記録 → 関節ごとの符号とオフセットを求め、train/units.py にテスト先行で変換を追加（手順は 3.1）
- configs/gate.yaml の max_speed・max_follow_err を度・度/秒で実測して決め直す（起動時と通常時の追従誤差を測る。決め方は 3.3）

### トラブルの切り分け
| 症状 | 確かめること |
|---|---|
| モーターが見つからない | 電源、3ピンのケーブル、ジャンパー、基板裏の USB 用パッド |
| 通信エラーが続く・動作中に止まる | 12V・2.0A の電源の容量不足の可能性（公式は 12V なら 5A 以上を推奨） |
| ある関節の値が動かない | その関節のホーンの空回り、ケーブルの抜け |
| ポートが変わる | by-id で指定する |
| キャリブレーションをやり直した | 0° の位置が変わるので、符号とオフセットを測り直し、控えも取り直す（「4. キャリブレーションのファイル」） |
