import { createRequire } from "node:module";
import http from "node:http";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const { createRequestHandler } = require("./externalApi.cjs");
const { GazeStateStore } = require("./externalStore.cjs");

function createStore() {
  return new GazeStateStore({
    startedAt: "2026-06-21T00:00:00.000Z",
    now: () => "2026-06-21T00:00:01.000Z",
    readCalibrationFile: () =>
      JSON.stringify({
        affine: [
          [1, 0],
          [0, 1],
          [0, 0]
        ],
        source_bounds: {
          min: [0.1, 0.2],
          max: [0.8, 0.9],
          margin: 0.08
        },
        samples: [{ raw: [0.5, 0.5], target: [0.5, 0.5] }]
      })
  });
}

function initializeStore(store = createStore()) {
  store.initialize(
    {
      runtime: "python",
      displayIndex: 1,
      requestedDisplayIndex: 1,
      displayCount: 2,
      displayBounds: { x: 100, y: 200, width: 1920, height: 1080 },
      invalidDisplay: false,
      cameraScreenX: 0.5,
      cameraScreenY: 0,
      webInference: null
    },
    {
      backend: "cuda",
      camera: "0",
      cameraResolutionName: "VGA",
      cameraWidth: 640,
      cameraHeight: 480,
      cameraFov: 90,
      displaySizeInch: 31.5,
      calibrationFile: "/tmp/.gaze_calibration.json",
      eyePositionWeightX: 1,
      eyePositionWeightY: 0.25
    }
  );
  return store;
}

function getJson(path, handler) {
  return new Promise((resolve, reject) => {
    const server = http.createServer(handler);
    server.listen(0, "127.0.0.1", async () => {
      try {
        const address = server.address();
        const response = await fetch(`http://127.0.0.1:${address.port}${path}`);
        const body = await response.json();
        resolve({ status: response.status, body });
      } catch (error) {
        reject(error);
      } finally {
        server.close();
      }
    });
  });
}

describe("GazeStateStore", () => {
  it("initializes display, camera, runtime, and calibration slices from config", () => {
    const snapshot = initializeStore().getSnapshot();

    expect(snapshot.runtime).toMatchObject({ name: "python", backend: "cuda" });
    expect(snapshot.display).toMatchObject({
      display_index: 1,
      display_count: 2,
      bounds: { x: 100, y: 200, width: 1920, height: 1080 },
      display_size_inch: 31.5
    });
    expect(snapshot.camera).toMatchObject({
      camera: "0",
      camera_resolution_name: "VGA",
      camera_width: 640,
      camera_height: 480,
      camera_fov_deg: 90
    });
    expect(snapshot.calibration).toMatchObject({
      path: "/tmp/.gaze_calibration.json",
      affine: [
        [1, 0],
        [0, 1],
        [0, 0]
      ]
    });
  });

  it("updates gaze normalized and display pixel coordinates", () => {
    const store = initializeStore();

    store.updateFromBackendMessage({
      type: "gaze",
      x_norm: 0.25,
      y_norm: 0.5,
      raw_x_norm: 0.2,
      raw_y_norm: 0.45,
      confidence: 0.9,
      distance_m: 0.7,
      yaw_deg: 3,
      pitch_deg: -2,
      gaze_projection_mode: "binocular-screen",
      detect_inference_ms: 4.2,
      gaze_inference_ms: 12.5,
      inference_ms: 16.7
    });

    expect(store.getSnapshot().gaze).toMatchObject({
      x_norm: 0.25,
      y_norm: 0.5,
      raw_x_norm: 0.2,
      raw_y_norm: 0.45,
      x_px: 580,
      y_px: 740,
      confidence: 0.9,
      distance_m: 0.7,
      yaw_deg: 3,
      pitch_deg: -2,
      detect_inference_ms: 4.2,
      gaze_inference_ms: 12.5,
      inference_ms: 16.7
    });
  });

  it("updates status, camera, and model runtime fields from status messages", () => {
    const store = initializeStore();

    store.updateFromBackendMessage({
      type: "status",
      level: "info",
      message: "Models loaded",
      runtime: "python",
      detector: "retinaface",
      detector_model: "retinaface.onnx",
      detector_providers: ["CUDAExecutionProvider"],
      gaze_providers: ["CPUExecutionProvider"],
      camera_width: 1280,
      camera_height: 720,
      camera_fov_deg: 80,
      eye_position_weight_x: 0.8,
      eye_position_weight_y: 0.3
    });

    const snapshot = store.getSnapshot();
    expect(snapshot.status).toMatchObject({ level: "info", message: "Models loaded" });
    expect(snapshot.camera).toMatchObject({
      camera_width: 1280,
      camera_height: 720,
      camera_fov_deg: 80,
      eye_position_weight_x: 0.8,
      eye_position_weight_y: 0.3
    });
    expect(snapshot.models.detector).toMatchObject({
      detector: "retinaface",
      model: "retinaface.onnx",
      accelerator: "cuda",
      providers: ["CUDAExecutionProvider"]
    });
    expect(snapshot.models.gaze).toMatchObject({
      accelerator: "cpu",
      providers: ["CPUExecutionProvider"]
    });
  });

  it("updates calibration state and excludes preview image data", () => {
    const store = initializeStore();

    store.updateFromBackendMessage({ type: "calibration", status: "saved", count: 5, path: "/tmp/calibration.json" });
    store.updateFromBackendMessage({
      type: "preview",
      image: "data:image/jpeg;base64,large",
      head_detected: true,
      eye_count: 2,
      width_ratio: 1.5
    });

    const snapshot = store.getSnapshot();
    expect(snapshot.calibration).toMatchObject({
      path: "/tmp/calibration.json",
      status: "saved",
      count: 5,
      saved_path: "/tmp/calibration.json"
    });
    expect(snapshot.preview).toEqual({
      head_detected: true,
      eye_count: 2,
      width_ratio: 1.5,
      updated_at: "2026-06-21T00:00:01.000Z"
    });
    expect(snapshot.preview.image).toBeUndefined();
  });
});

describe("external API request handler", () => {
  it("serves health and snapshot JSON", async () => {
    const store = initializeStore();
    const handler = createRequestHandler(store);

    await expect(getJson("/health", handler)).resolves.toMatchObject({
      status: 200,
      body: { ok: true, schema_version: 1, started_at: "2026-06-21T00:00:00.000Z" }
    });
    await expect(getJson("/snapshot/display", handler)).resolves.toMatchObject({
      status: 200,
      body: {
        display_index: 1,
        bounds: { x: 100, y: 200, width: 1920, height: 1080 }
      }
    });
  });

  it("returns 404 for unknown paths", async () => {
    const handler = createRequestHandler(initializeStore());

    await expect(getJson("/missing", handler)).resolves.toMatchObject({
      status: 404,
      body: { error: "not_found" }
    });
  });
});
