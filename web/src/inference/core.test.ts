import { describe, expect, it, vi } from "vitest";
import { Calibration, DisplayGeometry, ScreenProjector, estimateGazeFromModelOutput, parseRetinaFaceOutput } from "./core";
import type { GazeEstimate } from "./core";

describe("ScreenProjector", () => {
  const display = new DisplayGeometry(1920, 1080, 31.5);
  const projector = new ScreenProjector(display, false, false, 0.5, 0.5);
  const leftEye = { classId: 17, score: 1, x1: 296, y1: 236, x2: 304, y2: 244 };
  const rightEye = { classId: 17, score: 1, x1: 336, y1: 236, x2: 344, y2: 244 };
  const eyes = [leftEye, rightEye];
  const distanceM = 0.6;

  it("matches the legacy projection formula", () => {
    expect(projector.project([320, 240], 0, 0, distanceM)).toEqual([0.5, 0.5]);
  });

  it("uses camera FOV when computing focal length", () => {
    const narrowFovProjector = new ScreenProjector(display, true, true, 0.5, 0, 1, 0.25, 60);
    expect(narrowFovProjector.cameraFovDeg).toBe(60);
    expect(narrowFovProjector.focalPx).toBeGreaterThan(projector.focalPx);
  });

  it("matches legacy when binocular-screen eye angles match", () => {
    const estimate: GazeEstimate = {
      yawDeg: 5,
      pitchDeg: 2,
      leftYawDeg: 5,
      leftPitchDeg: 2,
      rightYawDeg: 5,
      rightPitchDeg: 2
    };
    const legacy = projector.project([320, 240], estimate.yawDeg, estimate.pitchDeg, distanceM);
    const result = projector.projectEstimate("binocular-screen", eyes, estimate, distanceM);
    expect(result.fallbackReason).toBeUndefined();
    expect(result.point[0]).toBeCloseTo(legacy[0], 7);
    expect(result.point[1]).toBeCloseTo(legacy[1], 7);
  });

  it("falls back for parallel binocular-convergence rays", () => {
    const estimate: GazeEstimate = {
      yawDeg: 0,
      pitchDeg: 0,
      leftYawDeg: 0,
      leftPitchDeg: 0,
      rightYawDeg: 0,
      rightPitchDeg: 0
    };
    const result = projector.projectEstimate("binocular-convergence", eyes, estimate, distanceM);
    expect(result.fallbackReason).toBeTruthy();
    expect(result.point).toEqual(projector.project([320, 240], 0, 0, distanceM));
  });
});

describe("RetinaFace parser", () => {
  it("selects the highest scoring face and derives sorted eye boxes", () => {
    const rows = new Float32Array(17 * 2);
    rows.set([0, 0, 0.4, 10, 20, 110, 120, 80, 50, 40, 50, 0, 0, 0, 0, 0, 0], 0);
    rows.set([0, 0, 0.9, 100, 120, 300, 340, 240, 180, 160, 180, 0, 0, 0, 0, 0, 0], 17);
    const result = parseRetinaFaceOutput(rows, 0.5);
    expect(result.head?.score).toBeCloseTo(0.9);
    expect(result.eyes).toHaveLength(2);
    expect(result.eyes[0].x1).toBeLessThan(result.eyes[1].x1);
  });
});

describe("Gaze output parser", () => {
  it("accepts [1, 962, 3] flattened output and splits both eyes", () => {
    const output = new Float32Array(962 * 3);
    for (let i = 0; i < 962; i += 1) {
      output[i * 3] = i % 481;
      output[i * 3 + 1] = i % 17;
      output[i * 3 + 2] = i % 31;
    }
    const estimate = estimateGazeFromModelOutput(output, { input: new Float32Array(1), inverseScale: 1, inverseTx: 0, inverseTy: 0 });
    expect(Number.isFinite(estimate.yawDeg)).toBe(true);
    expect(Number.isFinite(estimate.pitchDeg)).toBe(true);
  });
});

describe("Calibration", () => {
  it("writes affine-compatible calibration payload", async () => {
    const writeCalibration = vi.fn().mockResolvedValue({ ok: true, path: ".gaze_calibration.json" });
    vi.stubGlobal("window", {
      gazeBridge: {
        readCalibration: vi.fn().mockResolvedValue({ ok: true, text: null }),
        writeCalibration
      }
    });
    const emit = vi.fn();
    const calibration = new Calibration(".gaze_calibration.json", emit);
    await calibration.capture([0.5, 0.5], [0.5, 0.5]);
    await calibration.capture([0.1, 0.1], [0.12, 0.12]);
    await calibration.capture([0.9, 0.1], [0.88, 0.12]);
    await calibration.capture([0.9, 0.9], [0.88, 0.88]);
    await calibration.capture([0.1, 0.9], [0.12, 0.88]);
    expect(writeCalibration).toHaveBeenCalledOnce();
    const payload = writeCalibration.mock.calls[0][1];
    expect(payload.affine).toHaveLength(3);
    expect(payload.source_bounds.min).toEqual([0.1, 0.1]);
    expect(payload.source_bounds.max).toEqual([0.9, 0.9]);
    vi.unstubAllGlobals();
  });
});
