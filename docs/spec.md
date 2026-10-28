# 仕様（Decision・判定器の API・セーフティ・ゲート G1〜G9・教師 VLM の呼び出し規約）

この仕様は開発時の指示書から、見出しの作業用の注記を除き、文言を変えずに移したものです。

## インタフェース
- 選択肢は重大度順：`CONTINUE(継続) < SLOW(減速) < STOP(停止)`
- `Decision`: `choice`, `probs: dict[str,float]`, `t_mono: float`, `latency_ms: float`, `source: str`（"v0"/"teacher:<model>"/"v1" 等）
  - `t_mono` は**判定に使った画像の撮影時刻**（`time.monotonic()` と同じ時計。カメラ入力では V4L2 バッファのタイムスタンプ＝`Frame.t_capture` を `judge(image, t_capture=...)` で渡す）。`t_capture` を渡さない場合（保存済みの画像など）は**判定開始時刻**（終了時刻より実際の撮影に近い側）。判定が終わった時刻にはしない。ゲートの G1 は `now - t_mono` で「画像の古さ」を見る
  - `latency_ms` は判定処理（前処理＋推論）にかかった時間で、画像の古さとは別物（古さ ≒ 撮影から判定開始までの待ち＋latency_ms）
- 判定器の共通API: `judge(image: np.ndarray, t_capture: float | None = None) -> Decision`（カメラの `Frame` は `judge_frame(judge, frame)` で撮影時刻ごと渡せる）

## セーフティ・ゲート仕様（Step 1：テスト先行で実装）
入力: 最新の Decision（またはNone）、現在時刻、現在関節角 q_now、方策の目標関節角 q_target
出力: 実際に送る関節角 q_cmd と、ゲート状態
- G1 期限切れ: Decision が None、または `now - t_mono > stale_s`（既定 0.2 s）なら STOP 扱い。`t_mono - now > future_tol_s`（既定 0.001 s、時計の不整合）も STOP 扱い
- G2 低確信: `max(probs) < min_conf`（既定 0.6）なら、一段階安全側（CONTINUE→SLOW、SLOW→STOP）
- G3 STOP: `q_cmd = q_now`（その場で保持）
- G4 SLOW: `q_cmd = q_now + slow_alpha * (q_target - q_now)`（既定 slow_alpha 0.3）
- G5 CONTINUE: `q_cmd = q_target`
- G6 ヒステリシス: 制限を緩める遷移（STOP→SLOW、STOP→CONTINUE、SLOW→CONTINUE）は、遷移先の判定（G1・G2適用後）が `resume_hold_s`（既定 0.5 s）連続することを条件とする。別の判定・期限切れ・不正入力が挟まれば計測し直し、境界は「ちょうど resume_hold_s で遷移」。制限を強める遷移は即時。起動直後の状態は STOP
- G7 再開フック: STOP から抜けた瞬間（遷移先が SLOW でも CONTINUE でも）に `on_resume` コールバックを1回呼ぶ（方策の `policy.reset()` 用）。SLOW→CONTINUE では呼ばない
- G8 レートリミッタ: すべての状態で、G3〜G5 の結果に最後に適用する。基準は前回出力した指令 `q_cmd_prev`（初回呼び出し時と STOP 中は `q_cmd_prev = q_now` にリセットし、再開は必ず現在位置から始める）。1 ステップの関節ごとの上限 `L_j = max_speed[j] × dt`（`max_speed` は関節ごとの配列、単位はロボットの関節値の単位/秒。ゲートは単位非依存）。`dt` は前回呼び出しからの `now` の差分を `dt_max` で頭打ちにしたもの（初回と時刻の逆行は dt=0 で動かさない。逆行は `GateOutput.clock_backwards = True` で記録）。差分ベクトル `d = q_cmd - q_cmd_prev` は全体を同じ比率 `s = min(1, min_j L_j/|d_j|)` で縮め、方向を保つ。SLOW 中は上限を `L_j = slow_speed_scale × max_speed[j] × dt`（`0 < slow_speed_scale ≤ 1`、既定 0.4）にする（どちらの上限を使うかは、その周期の G6 の遷移の後の状態で決める）。制限がかかったら `GateOutput.rate_limited = True`。STOP（`q_cmd = q_now`）は制限しない
- G9 追従誤差の監視: 関節ごとに `|q_cmd_prev - q_now| > max_follow_err[j]`（`configs/gate.yaml` の関節ごとの配列）なら、その周期は STOP 扱い（`reason = follow_error`、`GateOutput.follow_error = True`）。STOP からの復帰は G6 のヒステリシスに従う
- 入力防御: probs・t_mono・q_target の NaN/inf、Choice 以外の choice は STOP 扱い。q_now の NaN/inf と shape 不一致は ValueError。STOP/SLOW の理由は `GateOutput.reason` に残す
- 数値はすべて `configs/gate.yaml` から読む

## 教師VLM（Ollama）の呼び出し規約
- `/api/chat`、`stream: false`、`think: false`（対応モデルのみ）、`options.temperature: 0`
- `format` にJSON Schema（`choice` を enum で限定）を指定し、選択肢以外を返させない
- 失敗・パース不能時は例外にせず `choice=STOP, source="teacher_error"` を返す（安全側）
- **属性モード**（`OllamaAttrTeacher`、`configs/eval.yaml` の `teacher_modes` で `mode: attr`）：選択肢を答えさせず、`format` の JSON Schema で `reason`（文字列）→ `object_present`（bool）→ `touches_white`（bool）の順に制約する。choice はコードの `decide_from_attributes` で決める（object_present=false → CONTINUE、true かつ touches_white=false → SLOW、touches_white=true → STOP）。object_present=false なのに touches_white=true のような矛盾は STOP 側に倒し、`contradiction` として記録する。参照画像（何もない状態）を指定した場合は、1 枚目に参照、2 枚目に判定対象を送る。属性が欠ける・型が違う応答は `teacher_error`（STOP）
