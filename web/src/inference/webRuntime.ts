import type { BackendMessage, WebInferenceConfig } from "../global";
import { logStatus } from "../logger";
import {
  Calibration,
  CAMERA_HEIGHT,
  CAMERA_WIDTH,
  DisplayGeometry,
  ScreenProjector,
  clamp01,
  createPreviewImage,
  type Detection
} from "./core";
import { createRuntimeModels, type RuntimeModels } from "./runtimes";

export interface WebRuntimeController {
  captureCalibration(target: [number, number]): Promise<void>;
  stop(): void;
}

export async function startWebRuntime(
  config: WebInferenceConfig,
  emit: (payload: BackendMessage) => void
): Promise<WebRuntimeController> {
  let stopped = false;
  let latestRaw: [number, number] | null = null;
  let smoothed: [number, number] | null = null;
  let stream: MediaStream | null = null;
  let models: RuntimeModels | null = null;
  let lastStatus = 0;
  let lastPreview = 0;
  let lastProjectionWarning = 0;
  let fallbackInProgress = false;

  const calibration = new Calibration(config.calibrationFile, emit);
  await calibration.load();

  if (config.detector !== "retinaface") {
    emit({
      type: "status",
      level: "error",
      message: `${config.runtime} runtime supports only RetinaFace detector`,
      runtime: config.runtime
    });
    return {
      captureCalibration: (target) => captureCalibration(calibration, latestRaw, target, emit, config),
      stop: () => {
        stopped = true;
      }
    };
  }

  try {
    try {
      models = await createRuntimeModels(config, "webgpu");
    } catch (error) {
      const payload: BackendMessage = {
        type: "status",
        level: "warning",
        message: `WebGPU model load failed: ${error}; falling back to wasm`,
        runtime: config.runtime,
        accelerator: "webgpu",
        logged: true
      };
      logStatus(payload, error);
      emit(payload);
      models = await createRuntimeModels(config, "wasm");
    }
    emit({
      type: "status",
      level: "info",
      message: "Models loaded",
      runtime: config.runtime,
      accelerator: models.accelerator,
      detector: config.detector,
      detector_model: config.retinafaceModel,
      detector_providers: models.detectorProviders,
      head_face_width_ratio: config.retinafaceHeadFaceRatio,
      camera_screen_x: config.cameraScreenX,
      camera_screen_y: config.cameraScreenY,
      eye_position_weight_x: config.eyePositionWeightX,
      eye_position_weight_y: config.eyePositionWeightY,
      gaze_projection_mode: config.gazeProjectionMode,
      gaze_providers: models.gazeProviders
    });

    const camera = await openCamera(config.camera);
    stream = camera.stream;
    const video = camera.video;
    const canvas = document.createElement("canvas");
    canvas.width = CAMERA_WIDTH;
    canvas.height = CAMERA_HEIGHT;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      throw new Error("2D canvas context is not available");
    }

    const projector = new ScreenProjector(
      new DisplayGeometry(config.displayWidth, config.displayHeight, config.displaySizeInch),
      config.flipX,
      config.flipY,
      config.cameraScreenX,
      config.cameraScreenY,
      config.eyePositionWeightX,
      config.eyePositionWeightY
    );
    const previewInterval = 1000 / Math.max(0.5, config.previewFps);

    const loop = async () => {
      if (stopped || models === null) {
        return;
      }
      try {
        ctx.drawImage(video, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
        const frame = ctx.getImageData(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
        const { head, eyes } = await models.detect(frame);
        const now = performance.now();
        const shouldEmitPreview = !config.hidePreview && now - lastPreview >= previewInterval;
        if (head === null || eyes.length < 2) {
          if (shouldEmitPreview) {
            const message = head === null ? "Head not detected" : `Eyes detected: ${eyes.length}`;
            emitPreview(frame, head, eyes, message, null, config.retinafaceHeadFaceRatio, emit);
            lastPreview = now;
          }
          throw new Error(head === null ? "Head was not detected" : "Two eyes were not detected");
        }

        const gaze = await models.estimate(frame, head, eyes);
        if (shouldEmitPreview) {
          emitPreview(frame, head, eyes, null, [gaze.yawDeg, gaze.pitchDeg], config.retinafaceHeadFaceRatio, emit);
          lastPreview = now;
        }
        const distanceM = projector.distanceFromHead(head, config.retinafaceHeadFaceRatio);
        const projection = projector.projectEstimate(config.gazeProjectionMode, eyes, gaze, distanceM);
        if (projection.fallbackReason && now - lastProjectionWarning > 2000) {
          const payload: BackendMessage = {
            type: "status",
            level: "warning",
            message: `${projection.fallbackReason}; falling back to legacy projection`,
            runtime: config.runtime,
            accelerator: models.accelerator,
            gaze_projection_mode: config.gazeProjectionMode,
            logged: true
          };
          logStatus(payload);
          emit(payload);
          lastProjectionWarning = now;
        }
        latestRaw = projection.point;
        const corrected = calibration.apply(latestRaw);
        if (smoothed === null) {
          smoothed = corrected;
        } else {
          const alphaX = Math.max(0, Math.min(0.95, config.smoothingAlpha));
          const alphaY = Math.max(0, Math.min(0.95, config.smoothingAlphaY));
          smoothed = [
            alphaX * smoothed[0] + (1 - alphaX) * corrected[0],
            alphaY * smoothed[1] + (1 - alphaY) * corrected[1]
          ];
        }
        const confidence = Math.min(head.score, (eyes[0].score + eyes[1].score) * 0.5);
        emit({
          type: "gaze",
          x_norm: clamp01(smoothed[0]),
          y_norm: clamp01(smoothed[1]),
          raw_x_norm: latestRaw[0],
          raw_y_norm: latestRaw[1],
          confidence,
          distance_m: distanceM,
          head_face_width_ratio: config.retinafaceHeadFaceRatio,
          eye_position_weight_x: projector.eyePositionWeightX,
          eye_position_weight_y: projector.eyePositionWeightY,
          gaze_projection_mode: config.gazeProjectionMode,
          yaw_deg: gaze.yawDeg,
          pitch_deg: gaze.pitchDeg
        });
      } catch (error) {
        const now = performance.now();
        if (models?.accelerator === "webgpu" && !fallbackInProgress && isRecoverableInferenceError(error)) {
          fallbackInProgress = true;
          const payload: BackendMessage = {
            type: "status",
            level: "warning",
            message: `WebGPU inference failed: ${error instanceof Error ? error.message : String(error)}; falling back to wasm`,
            runtime: config.runtime,
            accelerator: "webgpu",
            logged: true
          };
          logStatus(payload, error);
          emit(payload);
          try {
            models.dispose();
            models = await createRuntimeModels(config, "wasm");
            emit({
              type: "status",
              level: "info",
              message: "Models reloaded with wasm",
              runtime: config.runtime,
              accelerator: models.accelerator,
              detector: config.detector,
              detector_model: config.retinafaceModel,
              detector_providers: models.detectorProviders,
              head_face_width_ratio: config.retinafaceHeadFaceRatio,
              camera_screen_x: config.cameraScreenX,
              camera_screen_y: config.cameraScreenY,
              eye_position_weight_x: config.eyePositionWeightX,
              eye_position_weight_y: config.eyePositionWeightY,
              gaze_projection_mode: config.gazeProjectionMode,
              gaze_providers: models.gazeProviders
            });
          } catch (fallbackError) {
            const fallbackPayload: BackendMessage = {
              type: "status",
              level: "error",
              message: `Failed to reload models with wasm: ${fallbackError instanceof Error ? fallbackError.message : String(fallbackError)}`,
              runtime: config.runtime,
              accelerator: "wasm",
              logged: true
            };
            logStatus(fallbackPayload, fallbackError);
            emit(fallbackPayload);
          } finally {
            fallbackInProgress = false;
          }
        } else if (fallbackInProgress) {
          // Suppress repeated frame errors while runtime fallback is rebuilding sessions.
        } else {
          if (now - lastStatus > 1000) {
            const payload: BackendMessage = {
              type: "status",
              level: "warning",
              message: error instanceof Error ? error.message : String(error),
              runtime: config.runtime,
              accelerator: models?.accelerator,
              logged: true
            };
            logStatus(payload, error);
            emit(payload);
            lastStatus = now;
          }
        }
      } finally {
        if (!stopped) {
          window.setTimeout(loop, 33);
        }
      }
    };
    void loop();
  } catch (error) {
    const payload: BackendMessage = {
      type: "status",
      level: "error",
      message: error instanceof Error ? error.message : String(error),
      runtime: config.runtime,
      accelerator: models?.accelerator,
      logged: true
    };
    logStatus(payload, error);
    emit(payload);
  }

  return {
    captureCalibration: (target) => captureCalibration(calibration, latestRaw, target, emit, config),
    stop: () => {
      stopped = true;
      models?.dispose();
      for (const track of stream?.getTracks() ?? []) {
        track.stop();
      }
    }
  };
}

function isRecoverableInferenceError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("OrtRun") ||
    message.includes("bad_array_new_length") ||
    message.includes("failed to call") ||
    message.includes("ERROR_CODE")
  );
}

async function captureCalibration(
  calibration: Calibration,
  raw: [number, number] | null,
  target: [number, number],
  emit: (payload: BackendMessage) => void,
  config: WebInferenceConfig
): Promise<void> {
  console.info("Capturing calibration sample", JSON.stringify({ raw, target }));
  try {
    await calibration.capture(raw, target);
  } catch (error) {
    const payload: BackendMessage = {
      type: "status",
      level: "error",
      message: `Calibration capture failed: ${error instanceof Error ? error.message : String(error)}`,
      runtime: config.runtime,
      logged: true
    };
    logStatus(payload, error);
    emit(payload);
  }
}

async function openCamera(camera: string): Promise<{ stream: MediaStream; video: HTMLVideoElement }> {
  const constraints = await cameraConstraints(camera);
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      ...constraints,
      width: { ideal: CAMERA_WIDTH },
      height: { ideal: CAMERA_HEIGHT }
    },
    audio: false
  });
  const video = document.createElement("video");
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();
  await waitForVideo(video);
  return { stream, video };
}

async function cameraConstraints(camera: string): Promise<MediaTrackConstraints> {
  const index = Number.parseInt(camera, 10);
  if (Number.isFinite(index) && String(index) === camera.trim()) {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput");
    const selected = devices[index];
    return selected?.deviceId ? { deviceId: { exact: selected.deviceId } } : {};
  }
  if (camera) {
    return { deviceId: { exact: camera } };
  }
  return {};
}

function waitForVideo(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Camera video did not become ready"));
    }, 5000);
    const onReady = () => {
      cleanup();
      resolve();
    };
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", onReady);
    };
    video.addEventListener("loadeddata", onReady, { once: true });
  });
}

function emitPreview(
  frame: ImageData,
  head: Detection | null,
  eyes: Detection[],
  message: string | null,
  gazeAngles: [number, number] | null,
  widthRatio: number,
  emit: (payload: BackendMessage) => void
): void {
  const image = createPreviewImage(frame, head, eyes, message, widthRatio, gazeAngles ?? undefined);
  if (!image) {
    return;
  }
  emit({
    type: "preview",
    image,
    head_detected: head !== null,
    eye_count: eyes.length,
    width_ratio: widthRatio
  });
}
