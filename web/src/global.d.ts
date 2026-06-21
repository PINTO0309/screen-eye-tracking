export {};

declare global {
  interface Window {
    gazeBridge: {
      getConfig: () => Promise<RendererConfig>;
      readCalibration: (path: string) => Promise<{ ok: true; text: string | null } | { ok: false; error: string }>;
      writeCalibration: (
        path: string,
        payload: unknown
      ) => Promise<{ ok: true; path: string } | { ok: false; error: string }>;
      ready: () => void;
      setOverlayRegions: (regions: OverlayRegion[]) => void;
      publishBackendMessage: (payload: BackendMessage) => void;
      onBackendMessage: (callback: (payload: BackendMessage) => void) => () => void;
      sendCalibrationCapture: (target: [number, number]) => void;
    };
  }
}

export interface OverlayRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface RendererConfig {
  calibrate: boolean;
  displayIndex: number;
  requestedDisplayIndex: number;
  displayCount: number;
  displayBounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  invalidDisplay: boolean;
  cameraScreenX: number;
  cameraScreenY: number;
  runtime: RuntimeName;
  webInference: WebInferenceConfig | null;
}

export type RuntimeName = "python" | "onnxweb" | "litert";
export type GazeProjectionMode = "legacy" | "binocular-screen" | "binocular-convergence";
export type AcceleratorName = "webgpu" | "wasm";
export type WebAccelerator = AcceleratorName;

export interface WebInferenceConfig {
  runtime: Exclude<RuntimeName, "python">;
  detector: "retinaface" | "deim";
  backend: string;
  camera: string;
  cameraResolutionName?: string;
  cameraWidth: number;
  cameraHeight: number;
  cameraFov: number;
  scoreThreshold: number;
  displaySizeInch: number;
  displayWidth: number;
  displayHeight: number;
  calibrationFile: string;
  retinafaceModel: string;
  gazeModel: string;
  retinafaceModelUrl: string;
  gazeModelUrl: string;
  onnxWasmBaseUrl: string;
  liteRtWasmBaseUrl: string;
  smoothingAlpha: number;
  smoothingAlphaY: number;
  previewFps: number;
  hidePreview: boolean;
  flipX: boolean;
  flipY: boolean;
  cameraScreenX: number;
  cameraScreenY: number;
  eyePositionWeightX: number;
  eyePositionWeightY: number;
  retinafaceHeadFaceRatio: number;
  gazeProjectionMode: GazeProjectionMode;
}

export type BackendMessage =
  | {
      type: "gaze";
      x_norm: number;
      y_norm: number;
      raw_x_norm?: number;
      raw_y_norm?: number;
      confidence: number;
      distance_m: number;
      head_face_width_ratio?: number;
      eye_position_weight_x?: number;
      eye_position_weight_y?: number;
      gaze_projection_mode?: GazeProjectionMode;
      yaw_deg: number;
      pitch_deg: number;
    }
  | {
      type: "status";
      level: "info" | "warning" | "error";
      message: string;
      runtime?: RuntimeName;
      accelerator?: AcceleratorName;
      logged?: boolean;
      detector?: string;
      detector_model?: string;
      detector_providers?: string[];
      head_face_width_ratio?: number;
      camera_fov_deg?: number;
      camera_resolution_name?: string;
      camera_width?: number;
      camera_height?: number;
      camera_screen_x?: number;
      camera_screen_y?: number;
      eye_position_weight_x?: number;
      eye_position_weight_y?: number;
      gaze_projection_mode?: GazeProjectionMode;
      gaze_providers?: string[];
    }
  | {
      type: "preview";
      image: string;
      head_detected: boolean;
      eye_count: number;
      width_ratio?: number;
    }
  | {
      type: "calibration";
      status: string;
      count?: number;
      path?: string;
      message?: string;
    };
