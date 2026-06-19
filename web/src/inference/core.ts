import type { BackendMessage, GazeProjectionMode } from "../global";

export const CAMERA_WIDTH = 640;
export const CAMERA_HEIGHT = 480;
export const GAZE_INPUT_SIZE = 160;
export const HEAD_CLASS_ID = 7;
export const EYE_CLASS_ID = 17;
export const AVERAGE_HEAD_WIDTH_M = 0.16;
export const CAMERA_HORIZONTAL_FOV_DEG = 90;
const IRIS_IDX_481 = [248, 252, 224, 228, 232, 236, 240, 244];

export interface Detection {
  classId: number;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface GazeEstimate {
  yawDeg: number;
  pitchDeg: number;
  leftYawDeg: number;
  leftPitchDeg: number;
  rightYawDeg: number;
  rightPitchDeg: number;
}

export interface ProjectionResult {
  point: [number, number];
  fallbackReason?: string;
}

export interface CalibrationPayload {
  affine: number[][];
  source_bounds?: {
    min: number[];
    max: number[];
    margin?: number;
  };
  samples?: Array<{ raw: [number, number]; target: [number, number] }>;
}

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function validCameraFovDeg(value: number, fallback = CAMERA_HORIZONTAL_FOV_DEG): number {
  return Number.isFinite(value) && value > 0 && value < 180 ? value : fallback;
}

export function center(det: Detection): [number, number] {
  return [(det.x1 + det.x2) * 0.5, (det.y1 + det.y2) * 0.5];
}

export function width(det: Detection): number {
  return Math.max(0, det.x2 - det.x1);
}

export function height(det: Detection): number {
  return Math.max(0, det.y2 - det.y1);
}

function hypot(x: number, y: number): number {
  return Math.hypot(x, y);
}

export function createRetinaFaceInput(frame: ImageData): Float32Array {
  const input = new Float32Array(1 * 3 * CAMERA_HEIGHT * CAMERA_WIDTH);
  const plane = CAMERA_HEIGHT * CAMERA_WIDTH;
  const data = frame.data;
  for (let y = 0; y < CAMERA_HEIGHT; y += 1) {
    for (let x = 0; x < CAMERA_WIDTH; x += 1) {
      const src = (y * frame.width + x) * 4;
      const dst = y * CAMERA_WIDTH + x;
      input[dst] = data[src] - 104;
      input[plane + dst] = data[src + 1] - 117;
      input[plane * 2 + dst] = data[src + 2] - 123;
    }
  }
  return input;
}

export function createRetinaFaceInputNhwc(frame: ImageData): Float32Array {
  const input = new Float32Array(1 * CAMERA_HEIGHT * CAMERA_WIDTH * 3);
  const data = frame.data;
  for (let y = 0; y < CAMERA_HEIGHT; y += 1) {
    for (let x = 0; x < CAMERA_WIDTH; x += 1) {
      const src = (y * frame.width + x) * 4;
      const dst = (y * CAMERA_WIDTH + x) * 3;
      input[dst] = data[src] - 104;
      input[dst + 1] = data[src + 1] - 117;
      input[dst + 2] = data[src + 2] - 123;
    }
  }
  return input;
}

export function parseRetinaFaceOutput(output: ArrayLike<number>, scoreThreshold: number): {
  head: Detection | null;
  eyes: Detection[];
} {
  const rowSize = 17;
  let bestOffset = -1;
  let bestScore = -Infinity;
  for (let offset = 0; offset + rowSize <= output.length; offset += rowSize) {
    const score = Number(output[offset + 2]);
    if (score >= scoreThreshold && score > bestScore) {
      bestScore = score;
      bestOffset = offset;
    }
  }
  if (bestOffset < 0) {
    return { head: null, eyes: [] };
  }

  const x1 = clamp(Number(output[bestOffset + 3]), 0, CAMERA_WIDTH - 1);
  const y1 = clamp(Number(output[bestOffset + 4]), 0, CAMERA_HEIGHT - 1);
  const x2 = clamp(Number(output[bestOffset + 5]), 0, CAMERA_WIDTH - 1);
  const y2 = clamp(Number(output[bestOffset + 6]), 0, CAMERA_HEIGHT - 1);
  if (x2 <= x1 || y2 <= y1) {
    return { head: null, eyes: [] };
  }
  const head: Detection = { classId: HEAD_CLASS_ID, score: bestScore, x1, y1, x2, y2 };
  const rightEye: [number, number] = [Number(output[bestOffset + 7]), Number(output[bestOffset + 8])];
  const leftEye: [number, number] = [Number(output[bestOffset + 9]), Number(output[bestOffset + 10])];
  const eyeBoxSize = Math.max(10, width(head) * 0.08);
  const eyes = [eyeDetection(leftEye, eyeBoxSize, bestScore), eyeDetection(rightEye, eyeBoxSize, bestScore)].sort(
    (a, b) => center(a)[0] - center(b)[0]
  );
  return { head, eyes };
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function eyeDetection(point: [number, number], size: number, score: number): Detection {
  const half = size * 0.5;
  return {
    classId: EYE_CLASS_ID,
    score,
    x1: point[0] - half,
    y1: point[1] - half,
    x2: point[0] + half,
    y2: point[1] + half
  };
}

export interface GazeCrop {
  input: Float32Array;
  inverseScale: number;
  inverseTx: number;
  inverseTy: number;
}

export function createGazeInput(frame: ImageData, head: Detection, eyes: Detection[]): GazeCrop {
  if (eyes.length < 2) {
    throw new Error("Two eye detections are required");
  }
  const left = center(eyes[0]);
  const right = center(eyes[1]);
  const eyeCenter: [number, number] = [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5];
  const eyeDistance = hypot(right[0] - left[0], right[1] - left[1]);
  const cropSize = Math.max(width(head) / 1.5, eyeDistance) * 1.5;
  const scale = GAZE_INPUT_SIZE / Math.max(1, cropSize);
  const inverseScale = 1 / scale;
  const inverseTx = eyeCenter[0] - GAZE_INPUT_SIZE * 0.5 * inverseScale;
  const inverseTy = eyeCenter[1] - GAZE_INPUT_SIZE * 0.5 * inverseScale;
  const input = new Float32Array(1 * 3 * GAZE_INPUT_SIZE * GAZE_INPUT_SIZE);
  const plane = GAZE_INPUT_SIZE * GAZE_INPUT_SIZE;
  for (let y = 0; y < GAZE_INPUT_SIZE; y += 1) {
    for (let x = 0; x < GAZE_INPUT_SIZE; x += 1) {
      const srcX = x * inverseScale + inverseTx;
      const srcY = y * inverseScale + inverseTy;
      const [r, g, b] = sampleRgb(frame, srcX, srcY);
      const dst = y * GAZE_INPUT_SIZE + x;
      input[dst] = r / 127.5 - 1;
      input[plane + dst] = g / 127.5 - 1;
      input[plane * 2 + dst] = b / 127.5 - 1;
    }
  }
  return { input, inverseScale, inverseTx, inverseTy };
}

export function createGazeInputNhwc(frame: ImageData, head: Detection, eyes: Detection[]): GazeCrop {
  if (eyes.length < 2) {
    throw new Error("Two eye detections are required");
  }
  const left = center(eyes[0]);
  const right = center(eyes[1]);
  const eyeCenter: [number, number] = [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5];
  const eyeDistance = hypot(right[0] - left[0], right[1] - left[1]);
  const cropSize = Math.max(width(head) / 1.5, eyeDistance) * 1.5;
  const scale = GAZE_INPUT_SIZE / Math.max(1, cropSize);
  const inverseScale = 1 / scale;
  const inverseTx = eyeCenter[0] - GAZE_INPUT_SIZE * 0.5 * inverseScale;
  const inverseTy = eyeCenter[1] - GAZE_INPUT_SIZE * 0.5 * inverseScale;
  const input = new Float32Array(1 * GAZE_INPUT_SIZE * GAZE_INPUT_SIZE * 3);
  for (let y = 0; y < GAZE_INPUT_SIZE; y += 1) {
    for (let x = 0; x < GAZE_INPUT_SIZE; x += 1) {
      const srcX = x * inverseScale + inverseTx;
      const srcY = y * inverseScale + inverseTy;
      const [r, g, b] = sampleRgb(frame, srcX, srcY);
      const dst = (y * GAZE_INPUT_SIZE + x) * 3;
      input[dst] = r / 127.5 - 1;
      input[dst + 1] = g / 127.5 - 1;
      input[dst + 2] = b / 127.5 - 1;
    }
  }
  return { input, inverseScale, inverseTx, inverseTy };
}

function sampleRgb(frame: ImageData, x: number, y: number): [number, number, number] {
  if (x < 0 || y < 0 || x > frame.width - 1 || y > frame.height - 1) {
    return [0, 0, 0];
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(frame.width - 1, x0 + 1);
  const y1 = Math.min(frame.height - 1, y0 + 1);
  const wx = x - x0;
  const wy = y - y0;
  const c00 = pixel(frame, x0, y0);
  const c10 = pixel(frame, x1, y0);
  const c01 = pixel(frame, x0, y1);
  const c11 = pixel(frame, x1, y1);
  return [0, 1, 2].map((channel) => {
    const top = c00[channel] * (1 - wx) + c10[channel] * wx;
    const bottom = c01[channel] * (1 - wx) + c11[channel] * wx;
    return top * (1 - wy) + bottom * wy;
  }) as [number, number, number];
}

function pixel(frame: ImageData, x: number, y: number): [number, number, number] {
  const offset = (y * frame.width + x) * 4;
  return [frame.data[offset], frame.data[offset + 1], frame.data[offset + 2]];
}

export function estimateGazeFromModelOutput(output: ArrayLike<number>, crop: GazeCrop): GazeEstimate {
  if (output.length < 962 * 3) {
    throw new Error(`Unexpected gaze output length: ${output.length}`);
  }
  const points = new Float32Array(962 * 3);
  for (let i = 0; i < 962; i += 1) {
    const src = i * 3;
    const x = Number(output[src]) * crop.inverseScale + crop.inverseTx;
    const y = Number(output[src + 1]) * crop.inverseScale + crop.inverseTy;
    const z = Number(output[src + 2]) * crop.inverseScale;
    points[src] = y;
    points[src + 1] = x;
    points[src + 2] = z;
  }
  const [thetaXL, thetaYL] = anglesAndVecFromEye(points, 0);
  const [thetaXR, thetaYR] = anglesAndVecFromEye(points, 481);
  const leftYawDeg = (thetaYL * 180) / Math.PI;
  const leftPitchDeg = (-thetaXL * 180) / Math.PI;
  const rightYawDeg = (thetaYR * 180) / Math.PI;
  const rightPitchDeg = (-thetaXR * 180) / Math.PI;
  return {
    yawDeg: (leftYawDeg + rightYawDeg) * 0.5,
    pitchDeg: (leftPitchDeg + rightPitchDeg) * 0.5,
    leftYawDeg,
    leftPitchDeg,
    rightYawDeg,
    rightPitchDeg
  };
}

function anglesAndVecFromEye(points: Float32Array, pointOffset: number): [number, number] {
  const mean = [0, 0, 0];
  for (let i = 0; i < 32; i += 1) {
    const src = (pointOffset + i) * 3;
    mean[0] += points[src];
    mean[1] += points[src + 1];
    mean[2] += points[src + 2];
  }
  mean[0] /= 32;
  mean[1] /= 32;
  mean[2] /= 32;
  const vec = [0, 0, 0];
  for (const index of IRIS_IDX_481) {
    const src = (pointOffset + index) * 3;
    vec[0] += points[src] - mean[0];
    vec[1] += points[src + 1] - mean[1];
    vec[2] += points[src + 2] - mean[2];
  }
  vec[0] /= IRIS_IDX_481.length;
  vec[1] /= IRIS_IDX_481.length;
  vec[2] /= IRIS_IDX_481.length;
  const norm = Math.hypot(vec[0], vec[1], vec[2]);
  if (norm <= 1e-6) {
    throw new Error("Invalid gaze vector norm");
  }
  return anglesFromVec([vec[0] / norm, vec[1] / norm, vec[2] / norm]);
}

function anglesFromVec(vec: [number, number, number]): [number, number] {
  const x = -vec[2];
  const y = vec[1];
  const z = -vec[0];
  const theta = Math.atan2(y, x);
  const phi = Math.atan2(Math.sqrt(x * x + y * y), z) - Math.PI / 2;
  return [phi, theta];
}

export class DisplayGeometry {
  constructor(
    readonly widthPx: number,
    readonly heightPx: number,
    readonly diagonalInch: number
  ) {}

  get sizeM(): [number, number] {
    const diagonalM = this.diagonalInch * 0.0254;
    const pixelDiagonal = Math.hypot(this.widthPx, this.heightPx);
    return [(diagonalM * this.widthPx) / pixelDiagonal, (diagonalM * this.heightPx) / pixelDiagonal];
  }
}

export class ScreenProjector {
  readonly cameraScreenX: number;
  readonly cameraScreenY: number;
  readonly eyePositionWeightX: number;
  readonly eyePositionWeightY: number;
  readonly cameraFovDeg: number;
  readonly focalPx: number;

  constructor(
    readonly display: DisplayGeometry,
    readonly flipX = true,
    readonly flipY = true,
    cameraScreenX = 0.5,
    cameraScreenY = 0,
    eyePositionWeightX = 1,
    eyePositionWeightY = 0.25,
    cameraFovDeg = CAMERA_HORIZONTAL_FOV_DEG
  ) {
    this.cameraScreenX = clamp01(cameraScreenX);
    this.cameraScreenY = clamp01(cameraScreenY);
    this.eyePositionWeightX = clamp(eyePositionWeightX, 0, 1);
    this.eyePositionWeightY = clamp(eyePositionWeightY, 0, 1);
    this.cameraFovDeg = validCameraFovDeg(cameraFovDeg);
    this.focalPx = CAMERA_WIDTH / (2 * Math.tan((this.cameraFovDeg * Math.PI) / 180 * 0.5));
  }

  distanceFromHead(head: Detection, widthRatio = 1): number {
    const correctedWidthPx = Math.max(1, width(head) * widthRatio);
    return (AVERAGE_HEAD_WIDTH_M * this.focalPx) / correctedWidthPx;
  }

  project(eyeCenterPx: [number, number], yawDeg: number, pitchDeg: number, distanceM: number): [number, number] {
    return this.normalizeHitM(this.screenHitM(eyeCenterPx, yawDeg, pitchDeg, distanceM));
  }

  projectEstimate(mode: GazeProjectionMode, eyes: Detection[], estimate: GazeEstimate, distanceM: number): ProjectionResult {
    const eyeCenter: [number, number] = [
      (center(eyes[0])[0] + center(eyes[1])[0]) * 0.5,
      (center(eyes[0])[1] + center(eyes[1])[1]) * 0.5
    ];
    const legacy = this.project(eyeCenter, estimate.yawDeg, estimate.pitchDeg, distanceM);
    if (mode === "legacy") {
      return { point: legacy };
    }
    if (mode === "binocular-screen") {
      const leftHit = this.screenHitM(center(eyes[0]), estimate.leftYawDeg, estimate.leftPitchDeg, distanceM);
      const rightHit = this.screenHitM(center(eyes[1]), estimate.rightYawDeg, estimate.rightPitchDeg, distanceM);
      const hit: [number, number] = [(leftHit[0] + rightHit[0]) * 0.5, (leftHit[1] + rightHit[1]) * 0.5];
      if (Number.isFinite(hit[0]) && Number.isFinite(hit[1])) {
        return { point: this.normalizeHitM(hit) };
      }
      return { point: legacy, fallbackReason: "binocular-screen produced a non-finite hit point" };
    }
    if (mode === "binocular-convergence") {
      const convergence = this.convergenceHitM(
        center(eyes[0]),
        estimate.leftYawDeg,
        estimate.leftPitchDeg,
        center(eyes[1]),
        estimate.rightYawDeg,
        estimate.rightPitchDeg,
        distanceM
      );
      if (typeof convergence === "string") {
        return { point: legacy, fallbackReason: convergence };
      }
      return { point: this.normalizeHitM(convergence) };
    }
    return { point: legacy, fallbackReason: `Unsupported gaze projection mode: ${mode}` };
  }

  private eyeOriginM(eyeCenterPx: [number, number], distanceM: number): [number, number] {
    const [displayWM, displayHM] = this.display.sizeM;
    const eyeXM = ((eyeCenterPx[0] - CAMERA_WIDTH * 0.5) * distanceM) / this.focalPx * this.eyePositionWeightX;
    const eyeYM = ((eyeCenterPx[1] - CAMERA_HEIGHT * 0.5) * distanceM) / this.focalPx * this.eyePositionWeightY;
    return [displayWM * this.cameraScreenX + eyeXM, displayHM * this.cameraScreenY + eyeYM];
  }

  private screenHitM(eyeCenterPx: [number, number], yawDeg: number, pitchDeg: number, distanceM: number): [number, number] {
    const [eyeXM, eyeYM] = this.eyeOriginM(eyeCenterPx, distanceM);
    const hitXM = eyeXM + Math.tan((yawDeg * Math.PI) / 180) * distanceM;
    let pitchYM = Math.tan((pitchDeg * Math.PI) / 180) * distanceM;
    if (this.flipY) {
      pitchYM = -pitchYM;
    }
    return [hitXM, eyeYM + pitchYM];
  }

  private normalizeHitM(hitM: [number, number]): [number, number] {
    const [displayWM, displayHM] = this.display.sizeM;
    let xNorm = clamp01(hitM[0] / displayWM);
    if (this.flipX) {
      xNorm = 1 - xNorm;
    }
    return [xNorm, clamp01(hitM[1] / displayHM)];
  }

  private gazeDirection(yawDeg: number, pitchDeg: number): [number, number, number] {
    let pitchTan = Math.tan((pitchDeg * Math.PI) / 180);
    if (this.flipY) {
      pitchTan = -pitchTan;
    }
    const direction: [number, number, number] = [Math.tan((yawDeg * Math.PI) / 180), pitchTan, 1];
    const norm = Math.hypot(direction[0], direction[1], direction[2]);
    if (!Number.isFinite(norm) || norm <= 1e-9) {
      throw new Error("Invalid gaze direction");
    }
    return [direction[0] / norm, direction[1] / norm, direction[2] / norm];
  }

  private convergenceHitM(
    leftCenterPx: [number, number],
    leftYawDeg: number,
    leftPitchDeg: number,
    rightCenterPx: [number, number],
    rightYawDeg: number,
    rightPitchDeg: number,
    distanceM: number
  ): [number, number] | string {
    let leftOrigin: [number, number];
    let rightOrigin: [number, number];
    let d1: [number, number, number];
    let d2: [number, number, number];
    try {
      leftOrigin = this.eyeOriginM(leftCenterPx, distanceM);
      rightOrigin = this.eyeOriginM(rightCenterPx, distanceM);
      d1 = this.gazeDirection(leftYawDeg, leftPitchDeg);
      d2 = this.gazeDirection(rightYawDeg, rightPitchDeg);
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
    const p1: [number, number, number] = [leftOrigin[0], leftOrigin[1], 0];
    const p2: [number, number, number] = [rightOrigin[0], rightOrigin[1], 0];
    const b = dot(d1, d2);
    const denom = 1 - b * b;
    if (denom <= 1e-6) {
      return "binocular-convergence rays are nearly parallel";
    }
    const w0: [number, number, number] = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
    const d = dot(d1, w0);
    const e = dot(d2, w0);
    const t = (b * e - d) / denom;
    const u = (e - b * d) / denom;
    if (t <= 0 || u <= 0) {
      return "binocular-convergence intersection is behind the eye plane";
    }
    const closestLeft: [number, number, number] = [p1[0] + t * d1[0], p1[1] + t * d1[1], p1[2] + t * d1[2]];
    const closestRight: [number, number, number] = [p2[0] + u * d2[0], p2[1] + u * d2[1], p2[2] + u * d2[2]];
    const closestDistance = Math.hypot(
      closestLeft[0] - closestRight[0],
      closestLeft[1] - closestRight[1],
      closestLeft[2] - closestRight[2]
    );
    if (closestDistance > 0.2) {
      return "binocular-convergence rays do not meet closely";
    }
    const midpoint: [number, number, number] = [
      (closestLeft[0] + closestRight[0]) * 0.5,
      (closestLeft[1] + closestRight[1]) * 0.5,
      (closestLeft[2] + closestRight[2]) * 0.5
    ];
    if (!midpoint.every(Number.isFinite)) {
      return "binocular-convergence produced a non-finite point";
    }
    if (midpoint[2] <= 0 || midpoint[2] > distanceM * 3) {
      return "binocular-convergence depth is outside the expected range";
    }
    return [midpoint[0], midpoint[1]];
  }
}

function dot(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export class Calibration {
  matrix: number[][] | null = null;
  sourceMin: [number, number] | null = null;
  sourceMax: [number, number] | null = null;
  readonly sourceMargin = 0.08;
  readonly samples: Array<[[number, number], [number, number]]> = [];

  constructor(
    readonly path: string,
    readonly emit: (payload: BackendMessage) => void
  ) {}

  async load(): Promise<void> {
    const response = await window.gazeBridge.readCalibration(this.path);
    if (!response.ok) {
      this.emit({ type: "status", level: "warning", message: `Failed to load calibration: ${response.error}` });
      return;
    }
    if (response.text === null) {
      return;
    }
    try {
      const payload = JSON.parse(response.text) as CalibrationPayload;
      if (Array.isArray(payload.affine) && payload.affine.length === 3 && payload.affine.every((row) => row.length === 2)) {
        this.matrix = payload.affine.map((row) => [Number(row[0]), Number(row[1])]);
        const bounds = payload.source_bounds;
        if (bounds?.min?.length === 2 && bounds.max?.length === 2) {
          this.sourceMin = [Number(bounds.min[0]), Number(bounds.min[1])];
          this.sourceMax = [Number(bounds.max[0]), Number(bounds.max[1])];
        }
        this.emit({ type: "status", level: "info", message: `Loaded calibration: ${this.path}` });
      }
    } catch (error) {
      this.emit({ type: "status", level: "warning", message: `Failed to load calibration: ${error}` });
    }
  }

  apply(raw: [number, number]): [number, number] {
    if (this.matrix === null) {
      return raw;
    }
    let [x, y] = raw;
    if (this.sourceMin && this.sourceMax) {
      const spanX = Math.max(this.sourceMax[0] - this.sourceMin[0], 0.05);
      const spanY = Math.max(this.sourceMax[1] - this.sourceMin[1], 0.05);
      x = clamp(x, this.sourceMin[0] - spanX * this.sourceMargin, this.sourceMax[0] + spanX * this.sourceMargin);
      y = clamp(y, this.sourceMin[1] - spanY * this.sourceMargin, this.sourceMax[1] + spanY * this.sourceMargin);
    }
    const outX = x * this.matrix[0][0] + y * this.matrix[1][0] + this.matrix[2][0];
    const outY = x * this.matrix[0][1] + y * this.matrix[1][1] + this.matrix[2][1];
    return [clamp01(outX), clamp01(outY)];
  }

  async capture(raw: [number, number] | null, target: [number, number]): Promise<void> {
    if (raw === null) {
      this.emit({ type: "calibration", status: "no_sample", message: "No gaze sample is available yet" });
      return;
    }
    this.samples.push([raw, target]);
    console.info("Calibration sample accepted", JSON.stringify({ count: this.samples.length, raw, target }));
    this.emit({ type: "calibration", status: "sampled", count: this.samples.length });
    if (this.samples.length < 5) {
      return;
    }
    const recent = this.samples.slice(-5);
    this.matrix = solveAffine(recent);
    const rawXs = recent.map(([sample]) => sample[0]);
    const rawYs = recent.map(([sample]) => sample[1]);
    this.sourceMin = [Math.min(...rawXs), Math.min(...rawYs)];
    this.sourceMax = [Math.max(...rawXs), Math.max(...rawYs)];
    const payload: CalibrationPayload = {
      affine: this.matrix,
      source_bounds: {
        min: this.sourceMin,
        max: this.sourceMax,
        margin: this.sourceMargin
      },
      samples: recent.map(([sample, sampleTarget]) => ({ raw: sample, target: sampleTarget }))
    };
    const response = await window.gazeBridge.writeCalibration(this.path, payload);
    if (response.ok) {
      console.info("Calibration saved", JSON.stringify({ path: response.path }));
      this.emit({ type: "calibration", status: "saved", path: response.path });
    } else {
      this.emit({ type: "status", level: "error", message: `Failed to save calibration: ${response.error}` });
    }
  }
}

function solveAffine(samples: Array<[[number, number], [number, number]]>): number[][] {
  const ata = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  const atb = [
    [0, 0],
    [0, 0],
    [0, 0]
  ];
  for (const [[x, y], [tx, ty]] of samples) {
    const row = [x, y, 1];
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        ata[i][j] += row[i] * row[j];
      }
      atb[i][0] += row[i] * tx;
      atb[i][1] += row[i] * ty;
    }
  }
  const inv = invert3x3(ata);
  return inv.map((row) => [
    row[0] * atb[0][0] + row[1] * atb[1][0] + row[2] * atb[2][0],
    row[0] * atb[0][1] + row[1] * atb[1][1] + row[2] * atb[2][1]
  ]);
}

function invert3x3(m: number[][]): number[][] {
  const a = m[0][0];
  const b = m[0][1];
  const c = m[0][2];
  const d = m[1][0];
  const e = m[1][1];
  const f = m[1][2];
  const g = m[2][0];
  const h = m[2][1];
  const i = m[2][2];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) <= 1e-12) {
    throw new Error("Calibration samples are singular");
  }
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]
  ];
}

export function createPreviewImage(
  frame: ImageData,
  head: Detection | null,
  eyes: Detection[],
  message: string | null,
  widthRatio: number | null,
  gazeAngles?: [number, number]
): string {
  const source = document.createElement("canvas");
  source.width = CAMERA_WIDTH;
  source.height = CAMERA_HEIGHT;
  const sourceCtx = source.getContext("2d");
  if (!sourceCtx) {
    return "";
  }
  sourceCtx.putImageData(frame, 0, 0);
  if (head) {
    drawBox(sourceCtx, head, "#14dc5a", "Head");
  }
  for (const eye of eyes) {
    drawBox(sourceCtx, eye, "#00d2ff", "Eye");
    const [cx, cy] = center(eye);
    sourceCtx.fillStyle = "#00d2ff";
    sourceCtx.beginPath();
    sourceCtx.arc(cx, cy, 4, 0, Math.PI * 2);
    sourceCtx.fill();
  }
  if (gazeAngles && eyes.length >= 2) {
    drawGazeLines(sourceCtx, eyes, gazeAngles[0], gazeAngles[1]);
  }
  if (message) {
    strokeText(sourceCtx, message, 12, 28, 22);
  }
  if (widthRatio !== null) {
    strokeText(sourceCtx, `Head/Face ${widthRatio.toFixed(3)}x`, 12, CAMERA_HEIGHT - 14, 18);
  }
  const preview = document.createElement("canvas");
  preview.width = 320;
  preview.height = 240;
  const previewCtx = preview.getContext("2d");
  if (!previewCtx) {
    return "";
  }
  previewCtx.drawImage(source, 0, 0, preview.width, preview.height);
  return preview.toDataURL("image/jpeg", 0.72);
}

function drawBox(ctx: CanvasRenderingContext2D, det: Detection, color: string, label: string): void {
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.strokeRect(det.x1, det.y1, width(det), height(det));
  strokeText(ctx, `${label} ${det.score.toFixed(2)}`, det.x1, Math.max(18, det.y1 - 6), 16, color);
}

function strokeText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  size: number,
  fill = "#ffffff"
): void {
  ctx.font = `${size}px sans-serif`;
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#000000";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = fill;
  ctx.fillText(text, x, y);
}

function drawGazeLines(ctx: CanvasRenderingContext2D, eyes: Detection[], yawDeg: number, pitchDeg: number): void {
  const diag = Math.sqrt(CAMERA_WIDTH * CAMERA_HEIGHT);
  const length = 0.4 * diag;
  const dx = length * Math.sin((yawDeg * Math.PI) / 180);
  const dy = length * Math.sin((-pitchDeg * Math.PI) / 180);
  for (const eye of eyes.slice(0, 2)) {
    const [cx, cy] = center(eye);
    ctx.lineCap = "round";
    ctx.lineWidth = 7;
    ctx.strokeStyle = "#000000";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + dx, cy + dy);
    ctx.stroke();
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#00ff00";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + dx, cy + dy);
    ctx.stroke();
  }
}
