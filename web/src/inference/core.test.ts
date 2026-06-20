import { describe, expect, it, vi } from "vitest";
import {
  Calibration,
  DisplayGeometry,
  ScreenProjector,
  estimateGazeFromModelOutput,
  parseRetinaFaceOutput,
  parseRetinaFacePreNmsOutput,
  parseRetinaFaceRawOutput
} from "./core";
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

  it("parses LiteRT pre-NMS outputs and keeps the highest scoring face", () => {
    const boxes = new Float32Array(4 * 2);
    const scores = new Float32Array(2);
    const landms = new Float32Array(10 * 2);
    boxes.set([10, 20, 110, 120], 0);
    boxes.set([100, 120, 300, 340], 4);
    scores.set([0.8, 0.95]);
    landms.set([80, 50, 40, 50, 0, 0, 0, 0, 0, 0], 0);
    landms.set([240, 180, 160, 180, 0, 0, 0, 0, 0, 0], 10);

    const result = parseRetinaFacePreNmsOutput(boxes, scores, landms, 0.5);
    expect(result.head?.score).toBeCloseTo(0.95);
    expect(result.head?.x1).toBeCloseTo(100);
    expect(result.eyes).toHaveLength(2);
    expect(result.eyes[0].x1).toBeLessThan(result.eyes[1].x1);
  });

  it("keeps LiteRT pre-NMS threshold compatible with the embedded 0.70 model threshold", () => {
    const result = parseRetinaFacePreNmsOutput(
      new Float32Array([10, 20, 110, 120]),
      new Float32Array([0.69]),
      new Float32Array([80, 50, 40, 50, 0, 0, 0, 0, 0, 0]),
      0.5
    );
    expect(result.head).toBeNull();
    expect(result.eyes).toHaveLength(0);
  });

  it("returns no LiteRT pre-NMS detection for invalid boxes or incomplete landmarks", () => {
    const invalidBox = parseRetinaFacePreNmsOutput(
      new Float32Array([110, 120, 10, 20]),
      new Float32Array([0.95]),
      new Float32Array([80, 50, 40, 50, 0, 0, 0, 0, 0, 0]),
      0.5
    );
    const incompleteLandmarks = parseRetinaFacePreNmsOutput(
      new Float32Array([10, 20, 110, 120]),
      new Float32Array([0.95]),
      new Float32Array([80, 50, 40, 50]),
      0.5
    );
    expect(invalidBox.head).toBeNull();
    expect(incompleteLandmarks.head).toBeNull();
  });

  it("decodes LiteRT raw loc, logits, and landmarks", () => {
    const loc = new Float32Array(4 * 2);
    const logits = new Float32Array([0, 2, 0, 4]);
    const landms = new Float32Array(10 * 2);
    const result = parseRetinaFaceRawOutput(loc, logits, landms, 0.5);

    expect(result.head?.score).toBeGreaterThan(0.98);
    expect(result.head?.x1).toBeCloseTo(0);
    expect(result.head?.y1).toBeCloseTo(0);
    expect(result.head?.x2).toBeCloseTo(20);
    expect(result.head?.y2).toBeCloseTo(20);
    expect(result.eyes).toHaveLength(2);
  });

  it("keeps LiteRT raw threshold compatible with the embedded 0.70 model threshold", () => {
    const result = parseRetinaFaceRawOutput(
      new Float32Array(4),
      new Float32Array([0, Math.log(0.69 / 0.31)]),
      new Float32Array(10),
      0.5
    );
    expect(result.head).toBeNull();
  });

  it("returns no LiteRT raw detection for invalid decoded boxes", () => {
    const loc = new Float32Array([Number.NaN, 0, 0, 0]);
    const result = parseRetinaFaceRawOutput(loc, new Float32Array([0, 4]), new Float32Array(10), 0.5);
    expect(result.head).toBeNull();
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
