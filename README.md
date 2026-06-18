# Screen Eye Tracking

RetinaFaceまたはDEIMv2 Wholebody49の目位置検出と gaze ONNX モデルを使い、ディスプレイ上の視線ヒット位置に赤い点を表示するデスクトップアプリケーションです。

構成は Electron + React の透明オーバーレイと、Python/ONNX Runtime GPU の推論バックエンドです。

## セットアップ

Python は 3.10.x 固定です。このリポジトリでは `.python-version` を `3.10.12` にしています。

```bash
uv sync
source .venv/bin/activate
pnpm install
```

必要なモデルは `models/` に配置します。

```text
models/retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx
models/generalizing_gaze_estimation_with_weak_supervision_from_synthetic_views_Nx3x160x160.onnx
```

DEIMv2を検出器として使う場合は次のモデルも配置します。

```text
models/deimv2_dinov3_s_wholebody49_ins_s08_maskhead256x3_center_1240query_masks.onnx
```

## 起動

デフォルトは TensorRT です。TensorRT が使えない場合は警告を表示して CUDA、CPU の順にfallbackします。

```bash
pnpm dev -- --backend tensorrt
```

CUDA/CPU を明示する場合:

```bash
pnpm dev -- --backend cuda
pnpm dev -- --backend cpu
```

ビルドして起動する場合:

```bash
pnpm build
pnpm start -- --backend tensorrt
```

## 主なオプション

```bash
pnpm dev -- \
--backend tensorrt \
--detector retinaface \
--display-index 0 \
--display-size-inch 31.5 \
--camera 0 \
--score-threshold 0.50 \
--preview-fps 8
```

- `--backend tensorrt|cuda|cpu`: ONNX Runtime の実行バックエンド。デフォルトは `tensorrt`。
- `--detector retinaface|deim`: 目位置の検出器。デフォルトは `retinaface`。
- `--retinaface-model`: RetinaFaceモデルパス。デフォルトは `models/retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx`。
- `--deim-model`: DEIMv2モデルパス。`--detector deim` のときに使います。
- `--detector-model`: 選択中の検出器モデルパスを直接上書きします。
- `--display-index`: 点を表示する対象モニタ番号。Electron が取得したモニタ一覧の並び順です。
- `--debug-overlay`: 透明オーバーレイではなく通常の不透明ウィンドウで起動し、DevToolsも開きます。
- `--shape-overlay`: Linux/Windowsで透明ウィンドウの形状を可視部品周辺だけに絞ります。通常のクリック透過が効かない環境向けのfallbackです。視線マーカーがちらつく場合は使わないでください。
- `--display-size-inch`: 対象モニタの対角インチ。`18, 19, ..., 31, 31.5, 32` から選択します。デフォルトは `31.5`。
- `--camera`: OpenCV のカメラ番号または動画パス。デフォルトは `0`。
- `--score-threshold`: Head/Eye 検出のscore閾値。
- `--calibration-file`: 5点補正結果の保存先。デフォルトは `.gaze_calibration.json`。
- `--calibrate`: 5点簡易キャリブレーションを実行します。
- `--preview-fps`: 右上のPiPカメラプレビュー更新FPS。デフォルトは `8`。
- `--hide-preview`: PiPカメラプレビューを非表示にします。
- `--no-flip-x`: 視線点の左右反転補正を無効化します。デフォルトでは画面x座標を反転補正します。
- `--no-flip-y`: gazeモデルのpitch成分に対する上下反転補正を無効化します。顔/目のカメラ内Y位置による平行移動補正は反転しません。
- `--camera-screen-x`: カメラ設置位置の画面内X座標。左端 `0.0`、中央 `0.5`、右端 `1.0`。デフォルトは `0.5`。
- `--camera-screen-y`: カメラ設置位置の画面内Y座標。上端 `0.0`、中央 `0.5`、下端 `1.0`。デフォルトは `0.0`。
- `--eye-position-weight-x`: 顔/目bboxのX位置による平行移動補正の重み。デフォルトは `1.0`。
- `--eye-position-weight-y`: 顔/目bboxのY位置による平行移動補正の重み。デフォルトは `0.25`。姿勢変化で上下に張り付く場合は小さくします。
- `--retinaface-head-face-ratio`: RetinaFace利用時にFace幅をHead幅相当に補正する静的比率。デフォルトは `1.545`。

右上のPiPにはカメラ映像と検出結果を表示します。Head/Face相当のbboxは緑、Eyeは黄で描画され、`Head OK / Eyes 2` になっていれば検出と目の選択が視線推定に使える状態です。

RetinaFaceを使う場合、距離計算にはDEIMv2のHead幅を基準にした `16cm` 仮定が必要です。RetinaFaceのFace幅はHead幅より狭いため、ソースコード内の `RETINAFACE_HEAD_FACE_WIDTH_RATIO = 1.545` を静的補正係数として使い、Face幅をHead幅相当に補正して距離計算に反映します。現在の比率は右下ステータスとPiP内に表示されます。

DEIMv2のEye検出を使う場合:

```bash
pnpm dev -- --backend cuda --detector deim
```

## 画面に何も表示されない場合

`pnpm dev` は Vite の5173番ポートを使います。既存のdev serverが残っている場合は誤接続を避けるため起動を止めるので、先に古いプロセスを終了してください。

透明ウィンドウのため、rendererのロード失敗や別モニタ表示が見えにくい場合があります。まず通常ウィンドウで確認してください。

```bash
pnpm dev -- --backend tensorrt --debug-overlay
```

通常の透明オーバーレイではマウスイベントを背面アプリへ透過します。●、PiP、ステータス表示、カメラ位置マーカーが見えていても、その領域をクリックした操作は背面アプリへ届きます。通常は視線マーカーのちらつきを避けるため、OSのウィンドウ形状は更新しません。クリック透過が効かない環境だけ `--shape-overlay` を試してください。`--debug-overlay` のときだけ通常ウィンドウとして操作を受け取ります。

起動ログには `Display 0`, `Display 1` のようにモニタ一覧が出ます。表示先が違う場合は `--display-index` を指定します。`--display-index` を省略した場合はプライマリディスプレイを使います。

## 5点キャリブレーション

キャリブレーションなしでも起動できます。補正したい場合は `--calibrate` を付けて起動します。

```bash
pnpm dev -- --backend cuda --calibrate
```

画面中央に `3`, `2`, `1` のカウントダウンが出た後、ターゲットが順番に表示されます。中央ターゲットでは左右2本の赤い矢印で `→ ○ ←` のように丸を挟み、外周ターゲットでは画面中央にターゲット方向を示す赤い矢印を表示します。表示されたターゲットを見てください。アプリが各点を自動でサンプリングし、5点分のraw推定値とターゲットから2D affine補正を計算して `.gaze_calibration.json` に保存します。次回以降は同じファイルが自動で読み込まれます。

`--calibrate` を付けた起動では、ターゲットを見やすくするため5点キャリブレーション中だけPiPカメラプレビューは非表示になります。キャリブレーション完了後は再び表示されます。

キャリブレーション後に顔の上下位置を変えると●が画面端に張り付く場合、古い `.gaze_calibration.json` を削除して再キャリブレーションしてください。新しいキャリブレーションファイルにはraw入力範囲が保存され、範囲外への強い外挿を抑制します。

## 推定の前提

- カメラ入力は `640x480`、水平FOVは `90°` として扱います。
- カメラは対象ディスプレイの上中央にある前提です。
- 上下方向は顔/目bboxのY座標からカメラ中心に対する目の高さを推定し、カメラが画面上端中央にあるものとして画面座標へ投影します。顔が画角上へ移動した場合は点も上へ、下へ移動した場合は点も下へ動く向きです。
- 成人の平均的な頭部横幅を `16cm` と仮定し、検出器の顔/頭部bbox横幅から目とディスプレイの距離を推定します。
- RetinaFace利用時は、静的なHead/Face幅比率 `1.545` でFace幅をHead幅相当に補正します。
- デフォルトでは RetinaFace の左右目ランドマークを視線モデル用cropの中心計算に使います。
- `--detector deim` の場合は DEIMv2 の classid `17` Eye の上位2件を使います。

## 既知の制約

- カメラが画面上中央から大きくずれる場合、キャリブレーションなしの絶対位置精度は落ちます。
- 頭部横幅16cmの仮定から外れるほど距離推定がずれます。
- 眼鏡、強い逆光、暗い環境、顔の大きな回転では Eye/Head 検出またはgaze推定が不安定になります。
- 複数人が映る場合は、scoreが最も高いHeadを対象にします。

## 検証

```bash
uv run python -m compileall src
pnpm build
uv run python -m screen_eye_tracking.backend --help
```
