import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { AcceleratorName, BackendMessage, OverlayRegion, RendererConfig, RuntimeName } from "./global";
import { startWebRuntime, type WebRuntimeController } from "./inference/webRuntime";
import { logStatus } from "./logger";
import "./styles.css";

type GazeState = Extract<BackendMessage, { type: "gaze" }> & {
  receivedAt: number;
};

type PreviewState = Extract<BackendMessage, { type: "preview" }> & {
  receivedAt: number;
};

type ModelRuntimeState = {
  detector?: ModelRuntimeInfo;
  gaze?: ModelRuntimeInfo;
};

type ModelRuntimeInfo = {
  runtime: RuntimeName;
  accelerator: string;
};

type ClickEffectState = {
  id: number;
  effect: "single" | "double";
};

const calibrationPoints: [number, number][] = [
  [0.5, 0.5],
  [0.12, 0.12],
  [0.88, 0.12],
  [0.88, 0.88],
  [0.12, 0.88]
];
const calibrationTargetDurationMs = 1900;
const calibrationCaptureDelayMs = 1800;

function providerToAccelerator(provider: string | undefined): string {
  if (!provider) {
    return "unknown";
  }
  const normalized = provider.toLowerCase();
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
  return provider.replace(/ExecutionProvider$/u, "");
}

function statusRuntime(payload: Extract<BackendMessage, { type: "status" }>, config: RendererConfig | null): RuntimeName {
  return payload.runtime ?? config?.runtime ?? "python";
}

function statusAccelerator(
  payload: Extract<BackendMessage, { type: "status" }>,
  providers: string[] | undefined
): AcceleratorName | string {
  return payload.accelerator ?? providerToAccelerator(providers?.[0]);
}

function formatMs(value: number | undefined): string | null {
  return value !== undefined && Number.isFinite(value) ? `${value.toFixed(1)}ms` : null;
}

function App() {
  const [config, setConfig] = useState<RendererConfig | null>(null);
  const [gaze, setGaze] = useState<GazeState | null>(null);
  const [status, setStatus] = useState("starting");
  const [statusLevel, setStatusLevel] = useState<"info" | "warning" | "error">("info");
  const [preview, setPreview] = useState<PreviewState | null>(null);
  const [modelRuntime, setModelRuntime] = useState<ModelRuntimeState>({});
  const [calibrationCountdown, setCalibrationCountdown] = useState<number | null>(null);
  const [calibrationIndex, setCalibrationIndex] = useState<number | null>(null);
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [clickEffect, setClickEffect] = useState<ClickEffectState | null>(null);
  const cameraMarkerRef = useRef<HTMLDivElement>(null);
  const gazeDotRef = useRef<HTMLDivElement>(null);
  const calibrationCountdownRef = useRef<HTMLDivElement>(null);
  const calibrationArrowRef = useRef<HTMLDivElement>(null);
  const calibrationCenterArrowsRef = useRef<HTMLDivElement>(null);
  const calibrationTargetRef = useRef<HTMLDivElement>(null);
  const previewRef = useRef<HTMLElement>(null);
  const statusRef = useRef<HTMLElement>(null);
  const webRuntimeRef = useRef<WebRuntimeController | null>(null);
  const webRuntimeStartingRef = useRef(false);
  const configRef = useRef<RendererConfig | null>(null);

  useEffect(() => {
    configRef.current = config;
  }, [config]);

  const handleBackendMessage = useCallback((payload: BackendMessage) => {
    if (payload.type === "gaze") {
      setGaze({ ...payload, receivedAt: Date.now() });
      setStatus("tracking");
      setStatusLevel("info");
    } else if (payload.type === "status") {
      if (!payload.logged) {
        logStatus(payload);
      }
      setStatus(payload.message);
      setStatusLevel(payload.level);
      if (payload.detector_providers || payload.gaze_providers) {
        setModelRuntime((current) => ({
          detector: payload.detector_providers
            ? {
                runtime: statusRuntime(payload, configRef.current),
                accelerator: statusAccelerator(payload, payload.detector_providers)
              }
            : current.detector,
          gaze: payload.gaze_providers
            ? {
                runtime: statusRuntime(payload, configRef.current),
                accelerator: statusAccelerator(payload, payload.gaze_providers)
              }
            : current.gaze
        }));
      }
    } else if (payload.type === "preview") {
      setPreview({ ...payload, receivedAt: Date.now() });
    } else if (payload.type === "lip_click_effect") {
      setClickEffect((current) => ({
        id: (current?.id ?? 0) + 1,
        effect: payload.effect
      }));
    } else if (payload.type === "calibration") {
      setStatus(`calibration: ${payload.status}${payload.count ? ` ${payload.count}/5` : ""}`);
      setStatusLevel(payload.status === "saved" ? "info" : "warning");
      if (payload.status === "saved") {
        setCalibrationDone(true);
        setCalibrationCountdown(null);
        setCalibrationIndex(null);
      }
    }
  }, []);

  const handleRendererRuntimeMessage = useCallback(
    (payload: BackendMessage) => {
      window.gazeBridge.publishBackendMessage(payload);
      handleBackendMessage(payload);
    },
    [handleBackendMessage]
  );

  useEffect(() => {
    window.gazeBridge.getConfig().then(setConfig);
    const unsubscribe = window.gazeBridge.onBackendMessage(handleBackendMessage);
    window.gazeBridge.ready();
    return unsubscribe;
  }, [handleBackendMessage]);

  useEffect(() => {
    if (!config?.webInference || webRuntimeRef.current || webRuntimeStartingRef.current) {
      return;
    }
    let cancelled = false;
    webRuntimeStartingRef.current = true;
    startWebRuntime(config.webInference, handleRendererRuntimeMessage)
      .then((runtime) => {
        webRuntimeStartingRef.current = false;
        if (cancelled) {
          runtime.stop();
          return;
        }
        webRuntimeRef.current = runtime;
      })
      .catch((error) => {
        webRuntimeStartingRef.current = false;
        console.error("Failed to start web runtime", error);
        handleRendererRuntimeMessage({
          type: "status",
          level: "error",
          message: error instanceof Error ? error.message : String(error),
          runtime: config.webInference?.runtime
        });
      });
    return () => {
      cancelled = true;
      webRuntimeStartingRef.current = false;
      webRuntimeRef.current?.stop();
      webRuntimeRef.current = null;
    };
  }, [config?.webInference, handleRendererRuntimeMessage]);

  useEffect(() => {
    if (!config?.calibrate || calibrationDone) {
      return;
    }
    setCalibrationIndex(null);
    setCalibrationCountdown(3);
  }, [config?.calibrate, calibrationDone]);

  useEffect(() => {
    if (calibrationCountdown === null || calibrationDone) {
      return;
    }
    const countdownTimer = window.setTimeout(() => {
      if (calibrationCountdown > 1) {
        setCalibrationCountdown(calibrationCountdown - 1);
      } else {
        setCalibrationCountdown(null);
        setCalibrationIndex(0);
      }
    }, 1000);
    return () => window.clearTimeout(countdownTimer);
  }, [calibrationCountdown, calibrationDone]);

  useEffect(() => {
    if (calibrationIndex === null || calibrationIndex >= calibrationPoints.length) {
      return;
    }
    console.info("Calibration target shown", JSON.stringify({
      index: calibrationIndex,
      target: calibrationPoints[calibrationIndex]
    }));
    const captureTimer = window.setTimeout(() => {
      console.info("Calibration capture timer fired", JSON.stringify({
        index: calibrationIndex,
        target: calibrationPoints[calibrationIndex]
      }));
      if (config?.runtime === "python") {
        window.gazeBridge.sendCalibrationCapture(calibrationPoints[calibrationIndex]);
      } else {
        webRuntimeRef.current?.captureCalibration(calibrationPoints[calibrationIndex]).catch((error) => {
          console.error("Calibration capture failed", error);
          handleRendererRuntimeMessage({
            type: "status",
            level: "error",
            message: error instanceof Error ? error.message : String(error),
            runtime: config?.runtime
          });
        });
      }
    }, calibrationCaptureDelayMs);
    const nextTimer = window.setTimeout(() => {
      if (calibrationIndex + 1 < calibrationPoints.length) {
        console.info("Calibration target advancing", JSON.stringify({
          from: calibrationIndex,
          to: calibrationIndex + 1
        }));
        setCalibrationIndex(calibrationIndex + 1);
      }
    }, calibrationTargetDurationMs);
    return () => {
      window.clearTimeout(captureTimer);
      window.clearTimeout(nextTimer);
    };
  }, [calibrationIndex, config?.runtime, handleRendererRuntimeMessage]);

  const dotStyle = useMemo(() => {
    if (!gaze || Date.now() - gaze.receivedAt > 1500) {
      return { opacity: 0 };
    }
    return {
      transform: `translate(${gaze.x_norm * window.innerWidth}px, ${gaze.y_norm * window.innerHeight}px)`,
      opacity: Math.max(0.35, Math.min(1, gaze.confidence))
    };
  }, [gaze]);

  useEffect(() => {
    if (!clickEffect) {
      return;
    }
    const timeout = window.setTimeout(() => {
      setClickEffect((current) => (current?.id === clickEffect.id ? null : current));
    }, clickEffect.effect === "double" ? 1240 : 840);
    return () => window.clearTimeout(timeout);
  }, [clickEffect]);

  const target = calibrationIndex === null || calibrationCountdown !== null ? null : calibrationPoints[calibrationIndex];
  const calibrationActive = calibrationCountdown !== null || calibrationIndex !== null;
  const isCenterCalibrationTarget =
    target !== null && Math.hypot(target[0] - 0.5, target[1] - 0.5) < 0.05;
  const calibrationArrowStyle = useMemo(() => {
    if (!target) {
      return null;
    }
    const deltaX = target[0] - 0.5;
    const deltaY = target[1] - 0.5;
    if (Math.hypot(deltaX, deltaY) < 0.05) {
      return null;
    }
    return {
      transform: `translate(0, -50%) rotate(${Math.atan2(deltaY, deltaX)}rad)`
    };
  }, [target]);

  useEffect(() => {
    const collectRegions = () => {
      const regions: OverlayRegion[] = [];
      const addElement = (element: HTMLElement | null, padding: number) => {
        if (!element) {
          return;
        }
        const rect = element.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) {
          return;
        }
        regions.push({
          x: rect.left - padding,
          y: rect.top - padding,
          width: rect.width + padding * 2,
          height: rect.height + padding * 2
        });
      };
      addElement(cameraMarkerRef.current, 18);
      if (!calibrationActive && gaze && Date.now() - gaze.receivedAt <= 1500) {
        addElement(gazeDotRef.current, 56);
      }
      addElement(calibrationCountdownRef.current, 32);
      addElement(calibrationArrowRef.current, 28);
      addElement(calibrationCenterArrowsRef.current, 28);
      addElement(calibrationTargetRef.current, 18);
      addElement(previewRef.current, 24);
      if (!calibrationActive) {
        addElement(statusRef.current, 18);
      }
      window.gazeBridge.setOverlayRegions(regions);
    };

    collectRegions();
    const intervalId = window.setInterval(collectRegions, 250);
    window.addEventListener("resize", collectRegions);
    return () => {
      window.clearInterval(intervalId);
      window.removeEventListener("resize", collectRegions);
    };
  }, [config, gaze, preview, status, statusLevel, calibrationActive, calibrationCountdown, calibrationIndex, target]);

  return (
    <main className="overlay">
      <div
        ref={cameraMarkerRef}
        className="camera-marker"
        aria-label="camera position marker"
        style={{
          transform: `translate(${(config?.cameraScreenX ?? 0.5) * window.innerWidth}px, ${(config?.cameraScreenY ?? 0) * window.innerHeight}px)`
        }}
      >
        <div className="camera-arrow-shape" />
        <div className="camera-label">Camera position</div>
      </div>
      {!calibrationActive && (
        <div
          key={clickEffect?.id ?? 0}
          ref={gazeDotRef}
          className={`gaze-dot${clickEffect ? ` ${clickEffect.effect}-click-effect` : ""}`}
          style={dotStyle}
        >
          {clickEffect && <span className="gaze-click-label">{clickEffect.effect === "double" ? "Double" : "Single"}</span>}
        </div>
      )}
      {calibrationCountdown !== null && (
        <div ref={calibrationCountdownRef} className="calibration-countdown">
          {calibrationCountdown}
        </div>
      )}
      {calibrationArrowStyle && (
        <div ref={calibrationArrowRef} className="calibration-arrow" style={calibrationArrowStyle}>
          <div className="calibration-arrow-shape" />
        </div>
      )}
      {isCenterCalibrationTarget && (
        <div ref={calibrationCenterArrowsRef} className="calibration-center-arrows">
          <div className="calibration-arrow calibration-center-arrow-left">
            <div className="calibration-arrow-shape" />
          </div>
          <div className="calibration-arrow calibration-center-arrow-right">
            <div className="calibration-arrow-shape" />
          </div>
        </div>
      )}
      {target && (
        <div
          key={calibrationIndex}
          ref={calibrationTargetRef}
          className="calibration-target"
          style={{
            transform: `translate(${target[0] * window.innerWidth}px, ${target[1] * window.innerHeight}px)`
          }}
        >
          <div />
        </div>
      )}
      {preview && !calibrationActive && (
        <aside ref={previewRef} className="preview">
          <img src={preview.image} alt="camera detection preview" />
          <div className={preview.head_detected && preview.eye_count >= 2 ? "ok" : "warn"}>
            Head {preview.head_detected ? "OK" : "NO"} / Eyes {preview.eye_count}
          </div>
        </aside>
      )}
      {!calibrationActive && (
        <section ref={statusRef} className={`status ${statusLevel}`}>
          <div>{status}</div>
          {gaze && (
            <div>
              d {gaze.distance_m.toFixed(2)}m / yaw {gaze.yaw_deg.toFixed(1)} / pitch {gaze.pitch_deg.toFixed(1)}
            </div>
          )}
          {gaze?.head_face_width_ratio && (
            <div>
              Head/Face {gaze.head_face_width_ratio.toFixed(3)}x
            </div>
          )}
          {gaze?.eye_position_weight_y !== undefined && (
            <div>
              Eye pos weight x {gaze.eye_position_weight_x?.toFixed(2)} / y {gaze.eye_position_weight_y.toFixed(2)}
            </div>
          )}
          {gaze?.gaze_projection_mode && <div>Projection {gaze.gaze_projection_mode}</div>}
          {formatMs(gaze?.inference_ms) && (
            <div>
              Infer {formatMs(gaze?.inference_ms)} / det {formatMs(gaze?.detect_inference_ms) ?? "-"} / gaze{" "}
              {formatMs(gaze?.gaze_inference_ms) ?? "-"}
            </div>
          )}
          {modelRuntime.detector && (
            <div>
              Detector {modelRuntime.detector.runtime} / {modelRuntime.detector.accelerator}
            </div>
          )}
          {modelRuntime.gaze && (
            <div>
              Gaze {modelRuntime.gaze.runtime} / {modelRuntime.gaze.accelerator}
            </div>
          )}
          {config && (
            <div>
              display {config.displayIndex + 1}/{config.displayCount}
            </div>
          )}
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
