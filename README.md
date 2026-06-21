# Screen Eye Tracking

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20771714.svg)](https://doi.org/10.5281/zenodo.20771713)

A desktop application that estimates where the user is looking on the selected display and renders a red gaze marker at the estimated hit position. It uses RetinaFace or DEIMv2 Wholebody49 for eye position detection and an ONNX gaze model for gaze estimation.

The application is built as an Electron + React transparent overlay. It can run inference through the original Python / ONNX Runtime GPU backend, or fully inside Electron renderer with onnxruntime-web or LiteRT.js.

MediaPipe dependency-free.

- Tested on a 31.5-inch display - The display size can be changed using CLI parameters

  https://github.com/user-attachments/assets/7eff8143-2417-4459-8e63-de97868e3ff0

- Webcam and dGPU/iGPU only

  <img width="843" height="458" alt="image" src="https://github.com/user-attachments/assets/c3e192f0-9cb2-4ab8-ae31-9c93ce0647eb" />

## Arch

<img width="1600" height="730" alt="framework-architecture" src="https://github.com/user-attachments/assets/133e5969-55bc-40c2-89cb-207ace82677c" />

## Setup

```bash
########## This step is not necessary if you are not using Python ##########
# Installing uv
## Linux / Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
## Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Starting a virtual environment
uv python install 3.10.12
uv sync
source .venv/bin/activate
########## This step is not necessary if you are not using Python ##########

# Installing pnpm
## Linux / Mac
curl -fsSL https://get.pnpm.io/install.sh | sh -
## Windows
Invoke-WebRequest https://get.pnpm.io/install.ps1 -UseBasicParsing | Invoke-Expression

# Installing npm packages
pnpm install
```

Download the model files from the [`onnx` release](https://github.com/PINTO0309/screen-eye-tracking/releases/tag/onnx), then place the required models under `public/models/`.

For `--runtime python`:

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

An archive of CoreML models is also available from the same release, but I don't have demo code because I don't own an iPhone.

- [coreml_retinaface.tar.gz](https://github.com/PINTO0309/screen-eye-tracking/releases/download/onnx/coreml_retinaface.tar.gz)
- [coreml_gaze.tar.gz](https://github.com/PINTO0309/screen-eye-tracking/releases/download/onnx/coreml_gaze.tar.gz)

## Run

### 1. Python ver

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

To explicitly use TensorRT or CPU:

```bash
pnpm dev -- --backend tensorrt --calibrate
pnpm dev -- --backend cpu --calibrate
```

### 2. Web component only ver (Python independent)

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
- `--camera-resolution`: Camera capture resolution preset or `WIDTHxHEIGHT`. Default: `VGA` (`640x480`). Accepted presets are `QQVGA`, `QVGA`, `VGA`, `SVGA`, `XGA`, `HD`/`720p`, `SXGA`, `UXGA`, `Full HD`/`1080p`, `3MP`, `QHD`/`WQHD`/`1440p`, `5MP`, `6MP`, `4K UHD`, `DCI 4K`, `12MP`, `5K`, `6K`, `8K UHD`, and `12K`. `2MP`, `4MP`, and `8MP` aliases are rejected; use `1920x1080`, `2560x1440`, or `3840x2160` instead.
- `--camera-fov`: Horizontal camera FOV in degrees. Must be greater than `0` and less than `180`. Default: `90`.
- `--score-threshold`: Head/Eye detection score threshold.
- `--calibration-file`: Path for the 5-point calibration result. Default: `.gaze_calibration.json`.
- `--calibrate`: Runs 5-point calibration.
- `--smoothing-alpha`: Horizontal gaze marker smoothing. Larger values are steadier but slower. Default: `0.65`.
- `--smoothing-alpha-y`: Vertical gaze marker smoothing. Larger values are steadier but slower. Default: `0.45`.
- `--preview-fps`: PiP camera preview update FPS. Default: `8`.
- `--external-api`: Starts a read-only local HTTP/WebSocket API for external applications. Disabled by default.
- `--external-api-host`: Host for `--external-api`. Default: `127.0.0.1`.
- `--external-api-port`: Port for `--external-api`. Default: `47892`.
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

## External Read-Only API

External API access is disabled unless `--external-api` is passed. When enabled, the Electron main process exposes the latest internal state for local applications:

```bash
pnpm dev -- --backend cuda --external-api --calibrate --gaze-projection-mode binocular-screen
pnpm dev -- --runtime onnxweb --external-api --calibrate --gaze-projection-mode binocular-screen
pnpm dev -- --runtime litert --external-api --calibrate --gaze-projection-mode binocular-screen
```

HTTP endpoints:

```text
GET http://127.0.0.1:47892/health
GET http://127.0.0.1:47892/snapshot
GET http://127.0.0.1:47892/snapshot/gaze
GET http://127.0.0.1:47892/snapshot/display
GET http://127.0.0.1:47892/snapshot/camera
GET http://127.0.0.1:47892/snapshot/calibration
GET http://127.0.0.1:47892/snapshot/runtime
GET http://127.0.0.1:47892/snapshot/models
GET http://127.0.0.1:47892/snapshot/status
```

`gaze` sample.

```json
{
  "x_norm": 0.00046604398934387703,
  "y_norm": 0.958942713790909,
  "raw_x_norm": 0,
  "raw_y_norm": 0.27114625325864034,
  "x_px": 2561.1930726127202,
  "y_px": 1380.877507858909,
  "confidence": 0.9944654703140259,
  "distance_m": 0.4017150043171776,
  "head_face_width_ratio": 1.545,
  "eye_position_weight_x": 1,
  "eye_position_weight_y": 0.25,
  "gaze_projection_mode": "binocular-screen",
  "yaw_deg": 45.64941204243133,
  "pitch_deg": -12.357516307166726,
  "updated_at": "2026-06-21T13:47:46.399Z"
}
```

WebSocket updates are available at:

```text
ws://127.0.0.1:47892/events
```

Testing how to retrieve all status information in real time using WebSocket.

```bash
websocat ws://127.0.0.1:47892/events | jq .
```

Result sample.

```json
{
  "type": "update",
  "changed": [
    "preview"
  ],
  "snapshot": {
    "schema_version": 1,
    "started_at": "2026-06-21T13:47:11.321Z",
    "updated_at": "2026-06-21T13:49:54.535Z",
    "runtime": {
      "name": "onnxweb",
      "backend": "tensorrt",
      "updated_at": "2026-06-21T13:47:13.297Z",
      "accelerator": "webgpu"
    },
    "display": {
      "display_index": 1,
      "requested_display_index": 1,
      "display_count": 2,
      "bounds": {
        "x": 2560,
        "y": 0,
        "width": 2560,
        "height": 1440
      },
      "invalid_display": false,
      "display_size_inch": 31.5,
      "display_width": 2560,
      "display_height": 1440,
      "updated_at": "2026-06-21T13:47:11.515Z"
    },
    "camera": {
      "camera": "0",
      "camera_resolution_name": "VGA",
      "camera_width": 640,
      "camera_height": 480,
      "camera_fov_deg": 90,
      "camera_screen_x": 0.5,
      "camera_screen_y": 0,
      "eye_position_weight_x": 1,
      "eye_position_weight_y": 0.25,
      "updated_at": "2026-06-21T13:47:13.297Z"
    },
    "gaze": {
      "x_norm": 0.17188336409460786,
      "y_norm": 0.5195849356182066,
      "raw_x_norm": 0.40830238024351473,
      "raw_y_norm": 0.06745025281012713,
      "x_px": 3000.021412082196,
      "y_px": 748.2023072902175,
      "confidence": 0.9993162155151367,
      "distance_m": 0.35452647209083504,
      "head_face_width_ratio": 1.545,
      "eye_position_weight_x": 1,
      "eye_position_weight_y": 0.25,
      "gaze_projection_mode": "binocular-screen",
      "yaw_deg": -0.5973023779724806,
      "pitch_deg": -3.642409505603852,
      "updated_at": "2026-06-21T13:49:54.453Z"
    },
    "calibration": {
      "path": "/home/b920405/git/screen-eye-tracking/.gaze_calibration.json",
      "affine": [
        [
          4.7702737155679955,
          -0.16037419549669352
        ],
        [
          -0.1617891579496593,
          8.48930676701609
        ],
        [
          -1.7831292403376189,
          0.03883337028363698
        ]
      ],
      "source_bounds": {
        "min": [
          0.3911222626423193,
          0.01679723168253273
        ],
        "max": [
          0.5708666378962844,
          0.1091656248711355
        ],
        "margin": 0.08
      },
      "samples": [
        {
          "raw": [
            0.47425021214696883,
            0.06281074167161842
          ],
          "target": [
            0.5,
            0.5
          ]
        },
        {
          "raw": [
            0.3911222626423193,
            0.01679723168253273
          ],
          "target": [
            0.12,
            0.12
          ]
        },
        {
          "raw": [
            0.5708666378962844,
            0.02074244710003301
          ],
          "target": [
            0.88,
            0.12
          ]
        },
        {
          "raw": [
            0.5491286581713097,
            0.1091656248711355
          ],
          "target": [
            0.88,
            0.88
          ]
        },
        {
          "raw": [
            0.4184644907591354,
            0.10751170352506421
          ],
          "target": [
            0.12,
            0.88
          ]
        }
      ],
      "updated_at": "2026-06-21T13:47:24.147Z",
      "status": "saved",
      "saved_path": "/home/b920405/git/screen-eye-tracking/.gaze_calibration.json"
    },
    "models": {
      "detector": {
        "runtime": "onnxweb",
        "accelerator": "webgpu",
        "detector": "retinaface",
        "model": "/home/b920405/git/screen-eye-tracking/public/models/retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx",
        "providers": [
          "webgpu"
        ],
        "updated_at": "2026-06-21T13:47:13.297Z"
      },
      "gaze": {
        "runtime": "onnxweb",
        "accelerator": "webgpu",
        "providers": [
          "webgpu"
        ],
        "updated_at": "2026-06-21T13:47:13.297Z"
      }
    },
    "status": {
      "level": "info",
      "message": "Models loaded",
      "updated_at": "2026-06-21T13:47:13.297Z"
    },
    "preview": {
      "head_detected": true,
      "eye_count": 2,
      "width_ratio": 1.545,
      "updated_at": "2026-06-21T13:49:54.535Z"
    }
  }
}
```


The WebSocket sends a `snapshot` event on connect and `update` events after state changes. The snapshot uses `schema_version: 1` and includes latest-known `runtime`, `display`, `camera`, `gaze`, `calibration`, `models`, `status`, and preview metadata. Preview image data is intentionally omitted from the external snapshot.

Snapshot fields are latest-known values. Slices can be `null` before the app has enough data, and fields that are not known yet are omitted from JSON responses.

```ts
type ExternalSnapshot = {
  schema_version: 1;
  started_at: string; // ISO timestamp
  updated_at: string; // ISO timestamp
  runtime: RuntimeState | null;
  display: DisplayState | null;
  camera: CameraState | null;
  gaze: GazeState | null;
  calibration: CalibrationState | null;
  models: {
    detector: ModelState | null;
    gaze: ModelState | null;
  };
  status: StatusState | null;
  preview: PreviewState | null;
};

type RuntimeState = {
  name?: "python" | "onnxweb" | "litert";
  backend?: "tensorrt" | "cuda" | "cpu" | string;
  accelerator?: "tensorrt" | "cuda" | "cpu" | "webgpu" | "wasm" | string;
  updated_at: string;
};

type DisplayState = {
  display_index?: number;
  requested_display_index?: number;
  display_count?: number;
  bounds?: { x?: number; y?: number; width?: number; height?: number };
  invalid_display: boolean;
  display_size_inch?: number;
  display_width?: number;
  display_height?: number;
  updated_at: string;
};

type CameraState = {
  camera?: string;
  camera_resolution_name?: string;
  camera_width?: number;
  camera_height?: number;
  camera_fov_deg?: number;
  camera_screen_x?: number;
  camera_screen_y?: number;
  eye_position_weight_x?: number;
  eye_position_weight_y?: number;
  updated_at: string;
};

type GazeState = {
  x_norm?: number;
  y_norm?: number;
  raw_x_norm?: number;
  raw_y_norm?: number;
  x_px?: number;
  y_px?: number;
  confidence?: number;
  distance_m?: number;
  head_face_width_ratio?: number;
  eye_position_weight_x?: number;
  eye_position_weight_y?: number;
  gaze_projection_mode?:
    | "legacy"
    | "binocular-screen"
    | "binocular-convergence";
  yaw_deg?: number;
  pitch_deg?: number;
  updated_at: string;
};

type CalibrationState = {
  path?: string;
  status?: string;
  count?: number;
  saved_path?: string;
  message?: string;
  affine?: number[][];
  source_bounds?: {
    min?: [number, number];
    max?: [number, number];
    margin?: number;
  };
  samples?: Array<{
    raw: [number, number];
    target: [number, number];
  }>;
  read_error?: string;
  updated_at: string;
};

type ModelState = {
  runtime?: "python" | "onnxweb" | "litert";
  accelerator?: "tensorrt" | "cuda" | "cpu" | "webgpu" | "wasm" | string;
  detector?: "retinaface" | "deim" | string;
  model?: string;
  providers?: string[];
  updated_at: string;
};

type StatusState = {
  level?: "info" | "warning" | "error";
  message?: string;
  updated_at: string;
};

type PreviewState = {
  head_detected: boolean;
  eye_count?: number;
  width_ratio?: number;
  updated_at: string;
};
```

`x_norm` and `y_norm` are normalized gaze-marker coordinates on the selected display. `x_px` and `y_px` are desktop pixel coordinates computed from the selected display bounds, so multi-monitor offsets are included. `raw_x_norm` and `raw_y_norm` are the projection result before calibration correction and smoothing. The `preview` slice is available in the full `/snapshot` response and WebSocket events, but there is no `/snapshot/preview` endpoint.

WebSocket message shapes:

```ts
type SnapshotEvent = {
  type: "snapshot";
  snapshot: ExternalSnapshot;
};

type UpdateEvent = {
  type: "update";
  changed: Array<
    | "runtime"
    | "display"
    | "camera"
    | "gaze"
    | "calibration"
    | "models"
    | "status"
    | "preview"
  >;
  snapshot: ExternalSnapshot;
};
```

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

- Camera input defaults to `VGA` (`640x480`) and can be changed with `--camera-resolution`. Horizontal FOV defaults to `90°` and can be changed with `--camera-fov`.
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

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{katsuya_hyodo_2026_20771714,
  author    = {Katsuya Hyodo},
  title     = {screen-eye-tracking},
  year      = {2026},
  month     = {jun},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.20771714},
  url       = {https://github.com/PINTO0309/screen-eye-tracking},
  abstract  = {A desktop application that estimates where the user is looking on the selected display and renders a red gaze marker at the estimated hit position.},
}
```
