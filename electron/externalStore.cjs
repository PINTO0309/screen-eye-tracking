const SCHEMA_VERSION = 1;

function defaultNow() {
  return new Date().toISOString();
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function integer(value) {
  const number = finiteNumber(value);
  return number === undefined ? undefined : Math.trunc(number);
}

function stringValue(value) {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function compactObject(value) {
  const output = {};
  for (const [key, item] of Object.entries(value)) {
    if (item !== undefined) {
      output[key] = item;
    }
  }
  return output;
}

function displayBounds(bounds) {
  if (!bounds || typeof bounds !== "object") {
    return undefined;
  }
  return compactObject({
    x: integer(bounds.x),
    y: integer(bounds.y),
    width: integer(bounds.width),
    height: integer(bounds.height)
  });
}

function validBounds(bounds) {
  return bounds && Number.isFinite(bounds.x) && Number.isFinite(bounds.y) && bounds.width > 0 && bounds.height > 0;
}

function readCalibrationPayload(path, readCalibrationFile) {
  if (!path || typeof readCalibrationFile !== "function") {
    return {};
  }
  try {
    const text = readCalibrationFile(path);
    if (!text) {
      return {};
    }
    const payload = JSON.parse(text);
    const affine = Array.isArray(payload.affine) ? payload.affine : undefined;
    const sourceBounds = payload.source_bounds && typeof payload.source_bounds === "object" ? payload.source_bounds : undefined;
    const samples = Array.isArray(payload.samples) ? payload.samples : undefined;
    return compactObject({
      affine,
      source_bounds: sourceBounds,
      samples
    });
  } catch (error) {
    return {
      read_error: error instanceof Error ? error.message : String(error)
    };
  }
}

function createInitialSnapshot(startedAt) {
  return {
    schema_version: SCHEMA_VERSION,
    started_at: startedAt,
    updated_at: startedAt,
    runtime: null,
    display: null,
    camera: null,
    gaze: null,
    calibration: null,
    models: {
      detector: null,
      gaze: null
    },
    status: null,
    preview: null
  };
}

class GazeStateStore {
  constructor(options = {}) {
    this.now = options.now || defaultNow;
    this.readCalibrationFile = options.readCalibrationFile;
    const startedAt = options.startedAt || this.now();
    this.state = createInitialSnapshot(startedAt);
    this.listeners = new Set();
  }

  getSnapshot() {
    return clone(this.state);
  }

  getSlice(name) {
    if (!Object.prototype.hasOwnProperty.call(this.state, name)) {
      return undefined;
    }
    return clone(this.state[name]);
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  initialize(config, options = {}) {
    const updatedAt = this.now();
    const webConfig = config?.webInference || null;
    const bounds = displayBounds(config?.displayBounds);
    const calibrationFile = stringValue(webConfig?.calibrationFile) || stringValue(options.calibrationFile);
    const displayWidth = integer(webConfig?.displayWidth) || bounds?.width;
    const displayHeight = integer(webConfig?.displayHeight) || bounds?.height;

    this.state.runtime = compactObject({
      name: stringValue(config?.runtime) || stringValue(webConfig?.runtime),
      backend: stringValue(webConfig?.backend) || stringValue(options.backend),
      accelerator: undefined,
      updated_at: updatedAt
    });
    this.state.display = compactObject({
      display_index: integer(config?.displayIndex),
      requested_display_index: integer(config?.requestedDisplayIndex),
      display_count: integer(config?.displayCount),
      bounds,
      invalid_display: Boolean(config?.invalidDisplay),
      display_size_inch: finiteNumber(webConfig?.displaySizeInch) ?? finiteNumber(options.displaySizeInch),
      display_width: displayWidth,
      display_height: displayHeight,
      updated_at: updatedAt
    });
    this.state.camera = compactObject({
      camera: stringValue(webConfig?.camera) || stringValue(options.camera),
      camera_resolution_name: stringValue(webConfig?.cameraResolutionName) || stringValue(options.cameraResolutionName),
      camera_width: integer(webConfig?.cameraWidth) ?? integer(options.cameraWidth),
      camera_height: integer(webConfig?.cameraHeight) ?? integer(options.cameraHeight),
      camera_fov_deg: finiteNumber(webConfig?.cameraFov) ?? finiteNumber(options.cameraFov),
      camera_screen_x: finiteNumber(config?.cameraScreenX) ?? finiteNumber(webConfig?.cameraScreenX),
      camera_screen_y: finiteNumber(config?.cameraScreenY) ?? finiteNumber(webConfig?.cameraScreenY),
      eye_position_weight_x: finiteNumber(webConfig?.eyePositionWeightX) ?? finiteNumber(options.eyePositionWeightX),
      eye_position_weight_y: finiteNumber(webConfig?.eyePositionWeightY) ?? finiteNumber(options.eyePositionWeightY),
      updated_at: updatedAt
    });
    this.state.calibration = compactObject({
      path: calibrationFile,
      ...readCalibrationPayload(calibrationFile, this.readCalibrationFile),
      updated_at: updatedAt
    });
    this.state.updated_at = updatedAt;
    this.notify(["runtime", "display", "camera", "calibration"]);
  }

  updateFromBackendMessage(payload) {
    if (!payload || typeof payload !== "object" || typeof payload.type !== "string") {
      return [];
    }
    const updatedAt = this.now();
    const changed = new Set();

    if (payload.type === "gaze") {
      const xNorm = finiteNumber(payload.x_norm);
      const yNorm = finiteNumber(payload.y_norm);
      const bounds = this.state.display?.bounds;
      const pixels =
        xNorm !== undefined && yNorm !== undefined && validBounds(bounds)
          ? {
              x_px: bounds.x + xNorm * bounds.width,
              y_px: bounds.y + yNorm * bounds.height
            }
          : {};
      this.state.gaze = compactObject({
        x_norm: xNorm,
        y_norm: yNorm,
        raw_x_norm: finiteNumber(payload.raw_x_norm),
        raw_y_norm: finiteNumber(payload.raw_y_norm),
        ...pixels,
        confidence: finiteNumber(payload.confidence),
        distance_m: finiteNumber(payload.distance_m),
        head_face_width_ratio: finiteNumber(payload.head_face_width_ratio),
        eye_position_weight_x: finiteNumber(payload.eye_position_weight_x),
        eye_position_weight_y: finiteNumber(payload.eye_position_weight_y),
        gaze_projection_mode: stringValue(payload.gaze_projection_mode),
        detect_inference_ms: finiteNumber(payload.detect_inference_ms),
        gaze_inference_ms: finiteNumber(payload.gaze_inference_ms),
        inference_ms: finiteNumber(payload.inference_ms),
        yaw_deg: finiteNumber(payload.yaw_deg),
        pitch_deg: finiteNumber(payload.pitch_deg),
        updated_at: updatedAt
      });
      changed.add("gaze");
    } else if (payload.type === "status") {
      this.state.status = compactObject({
        level: stringValue(payload.level),
        message: stringValue(payload.message),
        updated_at: updatedAt
      });
      this.mergeRuntimeFromStatus(payload, updatedAt);
      this.mergeCameraFromStatus(payload, updatedAt);
      this.mergeModelsFromStatus(payload, updatedAt);
      changed.add("status");
      changed.add("runtime");
      if (this.state.camera) {
        changed.add("camera");
      }
      if (payload.detector_providers || payload.gaze_providers || payload.detector_model) {
        changed.add("models");
      }
    } else if (payload.type === "calibration") {
      const path = stringValue(payload.path) || this.state.calibration?.path;
      this.state.calibration = compactObject({
        ...(this.state.calibration || {}),
        path,
        status: stringValue(payload.status),
        count: integer(payload.count),
        saved_path: stringValue(payload.path),
        message: stringValue(payload.message),
        ...readCalibrationPayload(path, this.readCalibrationFile),
        updated_at: updatedAt
      });
      changed.add("calibration");
    } else if (payload.type === "preview") {
      this.state.preview = compactObject({
        head_detected: Boolean(payload.head_detected),
        eye_count: integer(payload.eye_count),
        width_ratio: finiteNumber(payload.width_ratio),
        updated_at: updatedAt
      });
      changed.add("preview");
    }

    if (changed.size === 0) {
      return [];
    }
    this.state.updated_at = updatedAt;
    const changedList = Array.from(changed);
    this.notify(changedList);
    return changedList;
  }

  mergeRuntimeFromStatus(payload, updatedAt) {
    this.state.runtime = compactObject({
      ...(this.state.runtime || {}),
      name: stringValue(payload.runtime) || this.state.runtime?.name,
      accelerator: stringValue(payload.accelerator) || this.state.runtime?.accelerator,
      updated_at: updatedAt
    });
  }

  mergeCameraFromStatus(payload, updatedAt) {
    const cameraUpdate = compactObject({
      camera_resolution_name: stringValue(payload.camera_resolution_name),
      camera_width: integer(payload.camera_width),
      camera_height: integer(payload.camera_height),
      camera_fov_deg: finiteNumber(payload.camera_fov_deg),
      camera_screen_x: finiteNumber(payload.camera_screen_x),
      camera_screen_y: finiteNumber(payload.camera_screen_y),
      eye_position_weight_x: finiteNumber(payload.eye_position_weight_x),
      eye_position_weight_y: finiteNumber(payload.eye_position_weight_y)
    });
    if (Object.keys(cameraUpdate).length === 0) {
      return;
    }
    this.state.camera = compactObject({
      ...(this.state.camera || {}),
      ...cameraUpdate,
      updated_at: updatedAt
    });
  }

  mergeModelsFromStatus(payload, updatedAt) {
    if (payload.detector_providers || payload.detector_model || payload.detector) {
      this.state.models.detector = compactObject({
        runtime: stringValue(payload.runtime) || this.state.runtime?.name,
        accelerator: stringValue(payload.accelerator) || providerToAccelerator(payload.detector_providers?.[0]),
        detector: stringValue(payload.detector),
        model: stringValue(payload.detector_model),
        providers: Array.isArray(payload.detector_providers) ? payload.detector_providers.map(String) : undefined,
        updated_at: updatedAt
      });
    }
    if (payload.gaze_providers) {
      this.state.models.gaze = compactObject({
        runtime: stringValue(payload.runtime) || this.state.runtime?.name,
        accelerator: stringValue(payload.accelerator) || providerToAccelerator(payload.gaze_providers?.[0]),
        providers: Array.isArray(payload.gaze_providers) ? payload.gaze_providers.map(String) : undefined,
        updated_at: updatedAt
      });
    }
  }

  notify(changed) {
    const event = {
      type: "update",
      changed,
      snapshot: this.getSnapshot()
    };
    for (const listener of this.listeners) {
      listener(event);
    }
  }
}

function providerToAccelerator(provider) {
  if (!provider) {
    return undefined;
  }
  const normalized = String(provider).toLowerCase();
  if (normalized.includes("tensorrt")) {
    return "tensorrt";
  }
  if (normalized.includes("cuda")) {
    return "cuda";
  }
  if (normalized.includes("cpu")) {
    return "cpu";
  }
  if (normalized.includes("webgpu")) {
    return "webgpu";
  }
  if (normalized.includes("wasm")) {
    return "wasm";
  }
  return String(provider).replace(/ExecutionProvider$/u, "");
}

module.exports = {
  GazeStateStore,
  SCHEMA_VERSION,
  providerToAccelerator
};
