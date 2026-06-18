export {};

declare global {
  interface Window {
    gazeBridge: {
      getConfig: () => Promise<RendererConfig>;
      ready: () => void;
      setOverlayRegions: (regions: OverlayRegion[]) => void;
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
      yaw_deg: number;
      pitch_deg: number;
    }
  | {
      type: "status";
      level: "info" | "warning" | "error";
      message: string;
      detector?: string;
      detector_model?: string;
      detector_providers?: string[];
      head_face_width_ratio?: number;
      camera_screen_x?: number;
      camera_screen_y?: number;
      eye_position_weight_x?: number;
      eye_position_weight_y?: number;
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
