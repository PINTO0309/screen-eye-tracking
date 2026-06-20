# Screen Eye Tracking

A desktop application that estimates where the user is looking on the selected display and renders a red gaze marker at the estimated hit position. It uses RetinaFace or DEIMv2 Wholebody49 for eye position detection and an ONNX gaze model for gaze estimation.

The application is built as an Electron + React transparent overlay. It can run inference through the original Python / ONNX Runtime GPU backend, or fully inside Electron renderer with onnxruntime-web or LiteRT.js.

- Tested on a 31.5-inch display - The display size can be changed using CLI parameters

  https://github.com/user-attachments/assets/7eff8143-2417-4459-8e63-de97868e3ff0

- Webcam and dGPU/iGPU only

  <img width="843" height="458" alt="image" src="https://github.com/user-attachments/assets/c3e192f0-9cb2-4ab8-ae31-9c93ce0647eb" />

## Setup

Python is pinned to the 3.10.x series. This repository uses `.python-version` set to `3.10.12`.

```bash
########## This step is not necessary if you are not using Python
# Installing uv
## Linux / Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
## Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Starting a virtual environment
uv sync
source .venv/bin/activate
########## This step is not necessary if you are not using Python

# Installing pnpm
## Linux / Mac
curl -fsSL https://get.pnpm.io/install.sh | sh -
## Windows
Invoke-WebRequest https://get.pnpm.io/install.ps1 -UseBasicParsing | Invoke-Expression

# Installing npm packages
pnpm install
```

Download the model files from the [`onnx` release](https://github.com/PINTO0309/screen-eye-tracking/releases/tag/onnx), then place the required models under `public/models/`.

```text
public/models/retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx
public/models/gaze_Nx3x160x160.onnx
```

For `--runtime onnxweb`, also place:

```text
public/models/gaze_1x3x160x160.onnx
```

For `--runtime litert`, also place:

```text
public/models/retinaface_mbn025_wo_postprocess_480x640_float32.tflite
public/models/gaze_1x3x160x160_float32.tflite
```

The LiteRT RetinaFace model has its dynamic postprocess tail removed. It must output `loc`, `conf_logits`, and `landms`; the renderer decodes boxes, scores, landmarks, and applies the lightweight selection step in JavaScript.

If you use DEIMv2 as the detector, also place this model under `public/models/`.

```text
public/models/deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query_masks.onnx
```

## Run

The default ONNX Runtime backend is TensorRT. If TensorRT is unavailable, the backend emits a warning and falls back to CUDA, then CPU.

```bash
# Recommended
pnpm dev -- --backend cuda --calibrate --gaze-projection-mode binocular-screen
# Selecting the gaze estimation mode
pnpm dev -- --backend cuda --calibrate
pnpm dev -- --backend cuda --calibrate --gaze-projection-mode legacy
pnpm dev -- --backend cuda --calibrate --gaze-projection-mode binocular-screen
pnpm dev -- --backend cuda --calibrate --gaze-projection-mode binocular-convergence
```

To explicitly use CUDA or CPU:

```bash
pnpm dev -- --backend tensorrt --calibrate
pnpm dev -- --backend cpu --calibrate
```

To run inference fully in Electron without starting Python:

```bash
# onnxruntime-web
pnpm dev -- --runtime onnxweb --calibrate --gaze-projection-mode binocular-screen
# LiteRT.js
pnpm dev -- --runtime litert --calibrate --gaze-projection-mode binocular-screen
```

The web runtimes try WebGPU first. If model loading fails, they reload both models with wasm.
When a web runtime is selected, Electron uses the selected model files in `public/models/` during dev runs and copies them to `dist/models/` for production runs. It also copies the required runtime wasm assets from `node_modules/` to `public/` during dev runs, or to `dist/` for production runs. The renderer loads both models and wasm assets as normal public assets.

To build and run:

```bash
pnpm build
pnpm start -- --backend tensorrt --calibrate
```

## Main Options

```bash
pnpm dev -- \
--backend tensorrt \
--runtime python \
--detector retinaface \
--display-index 0 \
--display-size-inch 31.5 \
--camera 0 \
--score-threshold 0.50 \
--preview-fps 8
```

- `--runtime python|onnxweb|litert`: Inference runtime. Default: `python`. `onnxweb` and `litert` never start the Python process.
- `--backend tensorrt|cuda|cpu`: Python ONNX Runtime execution backend. Default: `tensorrt`. Ignored by web runtimes.
- `--detector retinaface|deim`: Eye position detector. Default: `retinaface`.
- `--retinaface-model`: RetinaFace model path. Default: `public/models/retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx`.
- `--deim-model`: DEIMv2 model path. Used when `--detector deim` is selected.
- `--detector-model`: Directly overrides the selected detector model path.
- `--display-index`: Target monitor index for the overlay marker. This uses the display order reported by Electron.
- `--debug-overlay`: Starts as a normal opaque window instead of a transparent overlay and opens DevTools.
- `--shape-overlay`: On Linux/Windows, restricts the transparent window shape to the visible overlay elements. This is a fallback for environments where normal click-through does not work. Do not use it if the gaze marker flickers.
- `--display-size-inch`: Target monitor diagonal size in inches. Any positive finite value is accepted. Default: `31.5`.
- `--camera`: Python runtime uses an OpenCV camera index or video path. Web runtimes use a browser video input index or `deviceId`. Default: `0`.
- `--camera-fov`: Horizontal camera FOV in degrees. Must be greater than `0` and less than `180`. Default: `90`.
- `--score-threshold`: Head/Eye detection score threshold.
- `--calibration-file`: Path for the 5-point calibration result. Default: `.gaze_calibration.json`.
- `--calibrate`: Runs 5-point calibration.
- `--smoothing-alpha`: Horizontal gaze marker smoothing. Larger values are steadier but slower. Default: `0.65`.
- `--smoothing-alpha-y`: Vertical gaze marker smoothing. Larger values are steadier but slower. Default: `0.45`.
- `--preview-fps`: PiP camera preview update FPS. Default: `8`.
- `--hide-preview`: Hides the PiP camera preview.
- `--no-flip-x`: Disables horizontal gaze point flip correction. By default the screen x coordinate is flipped.
- `--no-flip-y`: Disables vertical pitch flip correction for the gaze model output. The parallel translation correction from the face/eye camera-space Y position is not flipped.
- `--camera-screen-x`: Camera position on the target screen, normalized horizontally. Left `0.0`, center `0.5`, right `1.0`. Default: `0.5`.
- `--camera-screen-y`: Camera position on the target screen, normalized vertically. Top `0.0`, center `0.5`, bottom `1.0`. Default: `0.0`.
- `--eye-position-weight-x`: Weight for the parallel translation correction from the face/eye bbox X position. Default: `1.0`.
- `--eye-position-weight-y`: Weight for the parallel translation correction from the face/eye bbox Y position. Default: `0.25`. Lower this if posture changes make the marker stick to the top or bottom edge.
- `--retinaface-head-face-ratio`: Static ratio used with RetinaFace to convert Face width to Head-equivalent width. Default: `1.545`.
- `--gaze-projection-mode legacy|binocular-screen|binocular-convergence`: Screen projection mode. Default: `legacy`.

If vertical tracking feels too slow while horizontal tracking is acceptable, lower `--smoothing-alpha-y`, for example `--smoothing-alpha-y 0.30`. If vertical movement is too small rather than too slow, try increasing `--eye-position-weight-y` gradually, for example `0.35` or `0.45`.

The PiP preview in the upper-right corner shows the camera image and detection state. The Head/Face-equivalent bbox is drawn in green and Eye detections are drawn in yellow. When gaze estimation succeeds, green line segments are drawn from both eye centers toward the estimated gaze direction. `Head OK / Eyes 2` means the current detection is usable for gaze estimation.

When RetinaFace is used, distance estimation still needs the `16cm` Head-width assumption used by DEIMv2 Head detection. RetinaFace Face width is narrower than Head width, so the static source constant `RETINAFACE_HEAD_FACE_WIDTH_RATIO = 1.545` converts Face width to Head-equivalent width before distance estimation. The current ratio is shown in the lower-right status area and the PiP preview.

Experimental binocular projection modes can be selected with `--gaze-projection-mode`. `legacy` keeps the original behavior, averaging the left/right eye gaze angles before projection. `binocular-screen` projects each eye separately to the screen plane and averages the two hit points. `binocular-convergence` estimates the closest point between the left/right gaze rays and uses that point as the screen hit position; if the rays are unstable, it falls back to `legacy`.

These binocular modes are approximations from the model's per-eye yaw/pitch and 2D eye positions, not a true optical vergence measurement. Because the raw point distribution can change between modes, use a separate calibration file when comparing them:

```bash
pnpm dev -- --gaze-projection-mode binocular-screen --calibration-file .gaze_calibration.binocular-screen.json
pnpm dev -- --gaze-projection-mode binocular-convergence --calibration-file .gaze_calibration.binocular-convergence.json
```

To use DEIMv2 Eye detection:

```bash
pnpm dev -- --backend cuda --detector deim
```

DEIMv2 is supported only by `--runtime python`. The web runtimes currently support RetinaFace only.

## If Nothing Appears

`pnpm dev` uses Vite on port `5173`. If an old dev server is still running, startup is stopped to avoid connecting to the wrong renderer. Stop the old process first.

Because the default window is transparent, renderer load failures or display selection mistakes can be hard to see. First verify with a normal window:

```bash
pnpm dev -- --backend tensorrt --debug-overlay
```

In the normal transparent overlay, mouse events pass through to the application behind it. Clicks pass through even when the red gaze marker, PiP, status panel, or camera position marker is visible. The camera position marker is shown as an upward arrow with the `Camera position` label. To avoid gaze-marker flicker, the OS window shape is not updated by default. Try `--shape-overlay` only in environments where normal click-through does not work. `--debug-overlay` is the only mode where the overlay receives normal window input.

Startup logs include display entries such as `Display 0` and `Display 1`. If the overlay appears on the wrong monitor, specify `--display-index`. When `--display-index` is omitted, the primary display is used.

## 5-Point Calibration

The app can run without calibration. To calibrate, start it with `--calibrate`.

```bash
pnpm dev -- --backend cuda --calibrate
```

After a centered `3`, `2`, `1` countdown, calibration targets appear in sequence. For the center target, two red arrows bracket the target as `-> O <-`. For outer targets, a red arrow at the screen center points toward the target direction. The inner circle expands from a small red circle to yellow and then green, fitting the outer circle when the point display completes. Look at the displayed target. The app samples each point automatically, computes a 2D affine correction from the 5 raw estimates and target points, and saves it to `.gaze_calibration.json`. The same file is loaded automatically on later runs.

When `--calibrate` is used, the PiP camera preview and normal red gaze marker are hidden during the 5-point calibration targets so the targets are easier to see. The PiP preview and gaze marker return after calibration completes.

If the marker sticks to a screen edge after calibration when your face moves up or down, delete the old `.gaze_calibration.json` and recalibrate. New calibration files store raw input bounds and suppress strong extrapolation outside the calibrated range.

## Estimation Assumptions

- Camera input is treated as `640x480`. Horizontal FOV defaults to `90°` and can be changed with `--camera-fov`.
- The camera is assumed to be mounted at the top center of the target display.
- For the vertical direction, the app estimates eye height relative to the camera center from the face/eye bbox Y position and projects it to screen coordinates assuming the camera is at the top center of the display. If the face moves upward in the camera frame, the marker moves upward; if the face moves downward, the marker moves downward.
- Adult average Head width is assumed to be `16cm`. The eye-to-display distance is estimated from the detected face/head bbox width.
- With RetinaFace, Face width is converted to Head-equivalent width using the static Head/Face ratio `1.545`.
- By default, RetinaFace left/right eye landmarks are used to compute the gaze-model crop center.
- `--gaze-projection-mode binocular-screen` and `binocular-convergence` use left/right gaze angles separately; they are experimental approximations and should be calibrated separately from `legacy`.
- With `--detector deim`, the top two DEIMv2 class id `17` Eye detections are used.

## Known Limitations

- Absolute position accuracy degrades without calibration if the camera is far from the top center of the display.
- Distance estimation drifts when the user's actual Head width differs from the `16cm` assumption.
- Glasses, strong backlight, dark scenes, or large face rotation can destabilize Eye/Head detection or gaze estimation.
- When multiple people are visible, the highest-score Head is selected.

## Verification

```bash
uv run python -m compileall src
pnpm build
uv run python -m screen_eye_tracking.backend --help
```
