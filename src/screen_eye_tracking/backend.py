from __future__ import annotations

import argparse
import base64
import json
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEIM_MODEL = REPO_ROOT / "public" / "models" / "deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query_masks.onnx"
DEFAULT_RETINAFACE_MODEL = REPO_ROOT / "public" / "models" / "retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx"
DEFAULT_GAZE_MODEL = REPO_ROOT / "public" / "models" / "gaze_Nx3x160x160.onnx"
DEFAULT_CALIBRATION_FILE = REPO_ROOT / ".gaze_calibration.json"

HEAD_CLASS_ID = 7
EYE_CLASS_ID = 17
AVERAGE_HEAD_WIDTH_M = 0.16
RETINAFACE_HEAD_FACE_WIDTH_RATIO = 1.545
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_HORIZONTAL_FOV_DEG = 90.0
GAZE_INPUT_SIZE = 160
IRIS_IDX_481 = np.asarray([248, 252, 224, 228, 232, 236, 240, 244], dtype=np.int64)


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)


@dataclass(frozen=True)
class DisplayGeometry:
    width_px: int
    height_px: int
    diagonal_inch: float

    @property
    def size_m(self) -> tuple[float, float]:
        diagonal_m = self.diagonal_inch * 0.0254
        pixel_diagonal = math.hypot(self.width_px, self.height_px)
        return (
            diagonal_m * self.width_px / pixel_diagonal,
            diagonal_m * self.height_px / pixel_diagonal,
        )


@dataclass(frozen=True)
class GazeEstimate:
    yaw_deg: float
    pitch_deg: float
    left_yaw_deg: float
    left_pitch_deg: float
    right_yaw_deg: float
    right_pitch_deg: float


@dataclass(frozen=True)
class ProjectionResult:
    point: tuple[float, float]
    fallback_reason: str | None = None


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def build_providers(backend: str, model_path: Path) -> list[Any]:
    available = set(ort.get_available_providers())
    requested = backend.lower()

    def cuda_or_cpu(reason: str) -> list[Any]:
        if "CUDAExecutionProvider" in available:
            emit({"type": "status", "level": "warning", "message": f"{reason}; falling back to CUDAExecutionProvider"})
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        emit({"type": "status", "level": "warning", "message": f"{reason}; falling back to CPUExecutionProvider"})
        return ["CPUExecutionProvider"]

    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return cuda_or_cpu("CUDAExecutionProvider is not available")
    if requested == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            return cuda_or_cpu("TensorrtExecutionProvider is not available")
        providers: list[Any] = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(model_path.parent),
                    "trt_fp16_enable": True,
                    "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
                },
            )
        ]
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    raise ValueError(f"Unsupported backend: {backend}")


def session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return options


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def valid_camera_fov_deg(value: float, fallback: float = CAMERA_HORIZONTAL_FOV_DEG) -> float:
    return value if math.isfinite(value) and 0.0 < value < 180.0 else fallback


def draw_box(image: np.ndarray, detection: Detection, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in (detection.x1, detection.y1, detection.x2, detection.y2)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {detection.score:.2f}"
    y = max(18, y1 - 6)
    cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_gaze_lines(image: np.ndarray, eyes: list[Detection], yaw_deg: float, pitch_deg: float) -> None:
    diag = math.sqrt(float(image.shape[0] * image.shape[1]))
    length = 0.4 * diag
    dx = length * math.sin(math.radians(yaw_deg))
    dy = length * math.sin(math.radians(-pitch_deg))
    for eye in eyes[:2]:
        start = (int(round(eye.center[0])), int(round(eye.center[1])))
        end = (int(round(eye.center[0] + dx)), int(round(eye.center[1] + dy)))
        cv2.line(image, start, end, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.line(image, start, end, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.circle(image, start, 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, start, 3, (0, 255, 0), -1, cv2.LINE_AA)


def emit_preview(
    frame: np.ndarray,
    head: Detection | None,
    eyes: list[Detection],
    message: str | None = None,
    width_ratio: float | None = None,
    gaze_angles: tuple[float, float] | None = None,
) -> None:
    preview = frame.copy()
    if head is not None:
        draw_box(preview, head, (20, 220, 90), "Head")
    for eye in eyes:
        draw_box(preview, eye, (0, 210, 255), "Eye")
        cx, cy = [int(round(v)) for v in eye.center]
        cv2.circle(preview, (cx, cy), 4, (0, 210, 255), -1)
    if gaze_angles is not None and len(eyes) >= 2:
        draw_gaze_lines(preview, eyes, gaze_angles[0], gaze_angles[1])
    if message:
        cv2.putText(preview, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if width_ratio is not None:
        ratio_text = f"Head/Face {width_ratio:.3f}x"
        cv2.putText(preview, ratio_text, (12, preview.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, ratio_text, (12, preview.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    preview = cv2.resize(preview, (320, 240), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return
    data = base64.b64encode(encoded).decode("ascii")
    emit(
        {
            "type": "preview",
            "image": f"data:image/jpeg;base64,{data}",
            "head_detected": head is not None,
            "eye_count": len(eyes),
            "width_ratio": width_ratio,
        }
    )


def angles_from_vec(vec: np.ndarray) -> tuple[float, float]:
    x, y, z = -vec[2], vec[1], -vec[0]
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z) - np.pi / 2
    return float(phi), float(theta)


def angles_and_vec_from_eye(eye: np.ndarray) -> tuple[float, float, np.ndarray]:
    p_iris = eye[IRIS_IDX_481] - eye[:32].mean(axis=0)
    vec = p_iris.mean(axis=0)
    norm = np.linalg.norm(vec, axis=0)
    if norm <= 1.0e-6:
        raise ValueError("Invalid gaze vector norm")
    vec = vec / norm
    theta_x, theta_y = angles_from_vec(vec)
    return theta_x, theta_y, vec


def similarity_crop(image: np.ndarray, center: tuple[float, float], crop_size: float) -> tuple[np.ndarray, np.ndarray]:
    scale = GAZE_INPUT_SIZE / max(1.0, crop_size)
    matrix = np.asarray(
        [
            [scale, 0.0, GAZE_INPUT_SIZE * 0.5 - center[0] * scale],
            [0.0, scale, GAZE_INPUT_SIZE * 0.5 - center[1] * scale],
        ],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(image, matrix, (GAZE_INPUT_SIZE, GAZE_INPUT_SIZE), borderValue=0.0)
    return crop, matrix


def transform_points3d(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = np.zeros_like(points, dtype=np.float32)
    scale = math.sqrt(float(matrix[0, 0] * matrix[0, 0] + matrix[0, 1] * matrix[0, 1]))
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    xy1 = np.concatenate([points[:, :2].astype(np.float32), ones], axis=1)
    transformed[:, :2] = xy1 @ matrix.T
    transformed[:, 2] = points[:, 2] * scale
    return transformed


class DeimV2EyeDetector:
    def __init__(self, model_path: Path, backend: str, score_threshold: float) -> None:
        self.model_path = model_path
        self.score_threshold = score_threshold
        providers = build_providers(backend, model_path)
        self.session = ort.InferenceSession(str(model_path), sess_options=session_options(), providers=providers)
        self.input = self.session.get_inputs()[0]
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.providers = self.session.get_providers()
        self._validate_model()

    def _validate_model(self) -> None:
        if self.input.name != "images" or list(self.input.shape) != [1, 3, 640, 640]:
            raise RuntimeError(f"Unexpected DEIMv2 input: {self.input.name} {self.input.shape}")
        if "label_xyxy_score" not in self.output_names:
            raise RuntimeError(f"DEIMv2 output label_xyxy_score is required, got {self.output_names}")

    def detect(self, frame: np.ndarray) -> tuple[Detection | None, list[Detection]]:
        image_h, image_w = frame.shape[:2]
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        chw = ((rgb - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
        output = self.session.run(["label_xyxy_score"], {"images": chw})[0][0]

        detections: list[Detection] = []
        for row in output:
            class_id = int(row[0])
            if class_id not in {HEAD_CLASS_ID, EYE_CLASS_ID}:
                continue
            score = float(row[5])
            if score < self.score_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in row[1:5]]
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                x1, x2 = x1 * image_w, x2 * image_w
                y1, y2 = y1 * image_h, y2 * image_h
            else:
                x1, x2 = x1 * image_w / 640.0, x2 * image_w / 640.0
                y1, y2 = y1 * image_h / 640.0, y2 * image_h / 640.0
            x1 = max(0.0, min(float(image_w - 1), x1))
            x2 = max(0.0, min(float(image_w - 1), x2))
            y1 = max(0.0, min(float(image_h - 1), y1))
            y2 = max(0.0, min(float(image_h - 1), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(class_id, score, x1, y1, x2, y2))

        heads = sorted((det for det in detections if det.class_id == HEAD_CLASS_ID), key=lambda det: det.score, reverse=True)
        eyes = [det for det in detections if det.class_id == EYE_CLASS_ID]
        if not heads:
            return None, []
        head = heads[0]
        selected = self._select_eyes(head, eyes)
        return head, selected

    @staticmethod
    def _select_eyes(head: Detection, eyes: Iterable[Detection]) -> list[Detection]:
        margin_x = head.width * 0.20
        margin_y = head.height * 0.20
        candidates = []
        for eye in eyes:
            cx, cy = eye.center
            if head.x1 - margin_x <= cx <= head.x2 + margin_x and head.y1 - margin_y <= cy <= head.y2 + margin_y:
                candidates.append(eye)
        candidates.sort(key=lambda det: det.score, reverse=True)
        candidates = candidates[:6]
        if len(candidates) <= 2:
            return sorted(candidates, key=lambda det: det.center[0])
        best_pair = max(
            ((a, b) for idx, a in enumerate(candidates) for b in candidates[idx + 1 :]),
            key=lambda pair: abs(pair[0].center[0] - pair[1].center[0]) * (pair[0].score + pair[1].score),
        )
        return sorted(best_pair, key=lambda det: det.center[0])


class RetinaFaceEyeDetector:
    def __init__(self, model_path: Path, backend: str, score_threshold: float) -> None:
        self.model_path = model_path
        self.score_threshold = score_threshold
        providers = build_providers(backend, model_path)
        self.session = ort.InferenceSession(str(model_path), sess_options=session_options(), providers=providers)
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.providers = self.session.get_providers()
        self.mean = np.asarray([104.0, 117.0, 123.0], dtype=np.float32)
        self._validate_model()

    def _validate_model(self) -> None:
        if self.input.name != "input" or list(self.input.shape) != [1, 3, 480, 640]:
            raise RuntimeError(f"Unexpected RetinaFace input: {self.input.name} {self.input.shape}")
        if self.output.name != "batchno_classid_score_x1y1x2y2_landms":
            raise RuntimeError(f"Unexpected RetinaFace output: {self.output.name} {self.output.shape}")

    def detect(self, frame: np.ndarray) -> tuple[Detection | None, list[Detection]]:
        image_h, image_w = frame.shape[:2]
        resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        input_tensor = resized[..., ::-1].astype(np.float32)
        input_tensor = (input_tensor - self.mean).transpose(2, 0, 1)[None, ...].astype(np.float32)
        output = self.session.run([self.output.name], {self.input.name: input_tensor})[0]
        if len(output) == 0:
            return None, []

        faces = [row for row in output if float(row[2]) >= self.score_threshold]
        if not faces:
            return None, []
        face = max(faces, key=lambda row: float(row[2]))
        score = float(face[2])
        x1 = max(0.0, min(float(image_w - 1), float(face[3]) * image_w / 640.0))
        y1 = max(0.0, min(float(image_h - 1), float(face[4]) * image_h / 480.0))
        x2 = max(0.0, min(float(image_w - 1), float(face[5]) * image_w / 640.0))
        y2 = max(0.0, min(float(image_h - 1), float(face[6]) * image_h / 480.0))
        if x2 <= x1 or y2 <= y1:
            return None, []

        head = Detection(HEAD_CLASS_ID, score, x1, y1, x2, y2)
        right_eye = (float(face[7]) * image_w / 640.0, float(face[8]) * image_h / 480.0)
        left_eye = (float(face[9]) * image_w / 640.0, float(face[10]) * image_h / 480.0)
        eye_box_size = max(10.0, head.width * 0.08)
        eyes = [
            self._eye_detection(left_eye, eye_box_size, score),
            self._eye_detection(right_eye, eye_box_size, score),
        ]
        return head, sorted(eyes, key=lambda det: det.center[0])

    @staticmethod
    def _eye_detection(center: tuple[float, float], size: float, score: float) -> Detection:
        cx, cy = center
        half = size * 0.5
        return Detection(EYE_CLASS_ID, score, cx - half, cy - half, cx + half, cy + half)


class GazeEstimator:
    def __init__(self, model_path: Path, backend: str) -> None:
        self.model_path = model_path
        providers = build_providers(backend, model_path)
        self.session = ort.InferenceSession(str(model_path), sess_options=session_options(), providers=providers)
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.providers = self.session.get_providers()
        self._validate_model()

    def _validate_model(self) -> None:
        if self.input.name != "input" or list(self.input.shape[1:]) != [3, 160, 160]:
            raise RuntimeError(f"Unexpected gaze model input: {self.input.name} {self.input.shape}")
        if self.output.name != "output" or list(self.output.shape[1:]) != [962, 3]:
            raise RuntimeError(f"Unexpected gaze model output: {self.output.name} {self.output.shape}")

    def estimate(self, frame: np.ndarray, head: Detection, eyes: list[Detection]) -> GazeEstimate:
        if len(eyes) < 2:
            raise ValueError("Two eye detections are required")
        left_eye, right_eye = eyes[0], eyes[1]
        left_center = np.asarray(left_eye.center, dtype=np.float32)
        right_center = np.asarray(right_eye.center, dtype=np.float32)
        eye_center = tuple(((left_center + right_center) * 0.5).tolist())
        eye_distance = float(np.linalg.norm(right_center - left_center))
        crop_size = max(head.width / 1.5, eye_distance) * 1.5
        crop, matrix = similarity_crop(frame, eye_center, crop_size)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None, ...]
        input_tensor = (input_tensor / 255.0 - 0.5) / 0.5
        prediction = self.session.run([self.output.name], {self.input.name: input_tensor})[0][0]
        inverse_matrix = cv2.invertAffineTransform(matrix).astype(np.float32)
        points = transform_points3d(prediction.astype(np.float32), inverse_matrix)
        eye_l = points[:481, :].copy()
        eye_r = points[481:, :].copy()
        for eye in (eye_l, eye_r):
            eye[:, [0, 1]] = eye[:, [1, 0]]
        theta_x_l, theta_y_l, _ = angles_and_vec_from_eye(eye_l)
        theta_x_r, theta_y_r, _ = angles_and_vec_from_eye(eye_r)
        left_yaw_deg = theta_y_l * 180.0 / math.pi
        left_pitch_deg = -theta_x_l * 180.0 / math.pi
        right_yaw_deg = theta_y_r * 180.0 / math.pi
        right_pitch_deg = -theta_x_r * 180.0 / math.pi
        yaw_deg = (left_yaw_deg + right_yaw_deg) * 0.5
        pitch_deg = (left_pitch_deg + right_pitch_deg) * 0.5
        return GazeEstimate(
            yaw_deg=float(yaw_deg),
            pitch_deg=float(pitch_deg),
            left_yaw_deg=float(left_yaw_deg),
            left_pitch_deg=float(left_pitch_deg),
            right_yaw_deg=float(right_yaw_deg),
            right_pitch_deg=float(right_pitch_deg),
        )


class Calibration:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.matrix: np.ndarray | None = None
        self.source_min: np.ndarray | None = None
        self.source_max: np.ndarray | None = None
        self.source_margin = 0.08
        self.samples: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            matrix = np.asarray(payload.get("affine"), dtype=np.float32)
            if matrix.shape == (3, 2):
                self.matrix = matrix
                source_bounds = payload.get("source_bounds")
                if isinstance(source_bounds, dict):
                    source_min = np.asarray(source_bounds.get("min"), dtype=np.float32)
                    source_max = np.asarray(source_bounds.get("max"), dtype=np.float32)
                    if source_min.shape == (2,) and source_max.shape == (2,):
                        self.source_min = source_min
                        self.source_max = source_max
                emit({"type": "status", "level": "info", "message": f"Loaded calibration: {self.path}"})
        except Exception as exc:
            emit({"type": "status", "level": "warning", "message": f"Failed to load calibration: {exc}"})

    def apply(self, raw: tuple[float, float]) -> tuple[float, float]:
        if self.matrix is None:
            return raw
        x, y = raw
        if self.source_min is not None and self.source_max is not None:
            source = np.asarray([x, y], dtype=np.float32)
            span = np.maximum(self.source_max - self.source_min, 0.05)
            lower = self.source_min - span * self.source_margin
            upper = self.source_max + span * self.source_margin
            x, y = np.clip(source, lower, upper).tolist()
        out = np.asarray([x, y, 1.0], dtype=np.float32) @ self.matrix
        return clamp01(float(out[0])), clamp01(float(out[1]))

    def capture(self, raw: tuple[float, float] | None, target: tuple[float, float]) -> None:
        if raw is None:
            emit({"type": "calibration", "status": "no_sample", "message": "No gaze sample is available yet"})
            return
        self.samples.append((raw, target))
        emit({"type": "calibration", "status": "sampled", "count": len(self.samples)})
        if len(self.samples) >= 5:
            source = np.asarray([[x, y, 1.0] for (x, y), _ in self.samples[-5:]], dtype=np.float32)
            target_matrix = np.asarray([target for _, target in self.samples[-5:]], dtype=np.float32)
            matrix, *_ = np.linalg.lstsq(source, target_matrix, rcond=None)
            self.matrix = matrix.astype(np.float32)
            source_xy = source[:, :2]
            self.source_min = source_xy.min(axis=0).astype(np.float32)
            self.source_max = source_xy.max(axis=0).astype(np.float32)
            payload = {
                "affine": self.matrix.tolist(),
                "source_bounds": {
                    "min": self.source_min.tolist(),
                    "max": self.source_max.tolist(),
                    "margin": self.source_margin,
                },
                "samples": [
                    {"raw": [float(raw_x), float(raw_y)], "target": [float(target_x), float(target_y)]}
                    for (raw_x, raw_y), (target_x, target_y) in self.samples[-5:]
                ],
            }
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            emit({"type": "calibration", "status": "saved", "path": str(self.path)})


class CommandReader(threading.Thread):
    def __init__(self, commands: queue.Queue[dict[str, Any]]) -> None:
        super().__init__(daemon=True)
        self.commands = commands

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                self.commands.put(json.loads(line))
            except json.JSONDecodeError as exc:
                emit({"type": "status", "level": "warning", "message": f"Invalid command JSON: {exc}"})


class ScreenProjector:
    def __init__(
        self,
        display: DisplayGeometry,
        flip_x: bool = True,
        flip_y: bool = True,
        camera_screen_x: float = 0.5,
        camera_screen_y: float = 0.0,
        eye_position_weight_x: float = 1.0,
        eye_position_weight_y: float = 0.25,
        camera_fov_deg: float = CAMERA_HORIZONTAL_FOV_DEG,
    ) -> None:
        self.display = display
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.camera_screen_x = clamp01(camera_screen_x)
        self.camera_screen_y = clamp01(camera_screen_y)
        self.eye_position_weight_x = max(0.0, min(1.0, eye_position_weight_x))
        self.eye_position_weight_y = max(0.0, min(1.0, eye_position_weight_y))
        self.camera_fov_deg = valid_camera_fov_deg(camera_fov_deg)
        self.focal_px = CAMERA_WIDTH / (2.0 * math.tan(math.radians(self.camera_fov_deg) * 0.5))

    def distance_from_head(self, head: Detection, width_ratio: float = 1.0) -> float:
        corrected_width_px = max(1.0, head.width * width_ratio)
        return (AVERAGE_HEAD_WIDTH_M * self.focal_px) / corrected_width_px

    def project(
        self,
        eye_center_px: tuple[float, float],
        yaw_deg: float,
        pitch_deg: float,
        distance_m: float,
    ) -> tuple[float, float]:
        return self._normalize_hit_m(self._screen_hit_m(eye_center_px, yaw_deg, pitch_deg, distance_m))

    def project_estimate(
        self,
        mode: str,
        eyes: list[Detection],
        estimate: GazeEstimate,
        distance_m: float,
    ) -> ProjectionResult:
        eye_center = (
            (eyes[0].center[0] + eyes[1].center[0]) * 0.5,
            (eyes[0].center[1] + eyes[1].center[1]) * 0.5,
        )
        legacy = self.project(eye_center, estimate.yaw_deg, estimate.pitch_deg, distance_m)
        if mode == "legacy":
            return ProjectionResult(legacy)
        if mode == "binocular-screen":
            left_hit = self._screen_hit_m(eyes[0].center, estimate.left_yaw_deg, estimate.left_pitch_deg, distance_m)
            right_hit = self._screen_hit_m(eyes[1].center, estimate.right_yaw_deg, estimate.right_pitch_deg, distance_m)
            hit = ((left_hit[0] + right_hit[0]) * 0.5, (left_hit[1] + right_hit[1]) * 0.5)
            if math.isfinite(hit[0]) and math.isfinite(hit[1]):
                return ProjectionResult(self._normalize_hit_m(hit))
            return ProjectionResult(legacy, "binocular-screen produced a non-finite hit point")
        if mode == "binocular-convergence":
            convergence = self._convergence_hit_m(
                eyes[0].center,
                estimate.left_yaw_deg,
                estimate.left_pitch_deg,
                eyes[1].center,
                estimate.right_yaw_deg,
                estimate.right_pitch_deg,
                distance_m,
            )
            if isinstance(convergence, str):
                return ProjectionResult(legacy, convergence)
            return ProjectionResult(self._normalize_hit_m(convergence))
        return ProjectionResult(legacy, f"Unsupported gaze projection mode: {mode}")

    def _eye_origin_m(self, eye_center_px: tuple[float, float], distance_m: float) -> tuple[float, float]:
        display_w_m, display_h_m = self.display.size_m
        eye_x_m = (eye_center_px[0] - CAMERA_WIDTH * 0.5) * distance_m / self.focal_px * self.eye_position_weight_x
        eye_y_m = (eye_center_px[1] - CAMERA_HEIGHT * 0.5) * distance_m / self.focal_px * self.eye_position_weight_y
        return display_w_m * self.camera_screen_x + eye_x_m, display_h_m * self.camera_screen_y + eye_y_m

    def _screen_hit_m(
        self,
        eye_center_px: tuple[float, float],
        yaw_deg: float,
        pitch_deg: float,
        distance_m: float,
    ) -> tuple[float, float]:
        eye_x_m, eye_y_m = self._eye_origin_m(eye_center_px, distance_m)
        hit_x_m = eye_x_m + math.tan(math.radians(yaw_deg)) * distance_m
        pitch_y_m = math.tan(math.radians(pitch_deg)) * distance_m
        if self.flip_y:
            pitch_y_m = -pitch_y_m
        hit_y_m = eye_y_m + pitch_y_m
        return hit_x_m, hit_y_m

    def _normalize_hit_m(self, hit_m: tuple[float, float]) -> tuple[float, float]:
        display_w_m, display_h_m = self.display.size_m
        hit_x_m, hit_y_m = hit_m
        x_norm = clamp01(hit_x_m / display_w_m)
        if self.flip_x:
            x_norm = 1.0 - x_norm
        return x_norm, clamp01(hit_y_m / display_h_m)

    def _gaze_direction(self, yaw_deg: float, pitch_deg: float) -> np.ndarray:
        pitch_tan = math.tan(math.radians(pitch_deg))
        if self.flip_y:
            pitch_tan = -pitch_tan
        direction = np.asarray([math.tan(math.radians(yaw_deg)), pitch_tan, 1.0], dtype=np.float64)
        norm = np.linalg.norm(direction)
        if not math.isfinite(float(norm)) or norm <= 1.0e-9:
            raise ValueError("Invalid gaze direction")
        return direction / norm

    def _convergence_hit_m(
        self,
        left_center_px: tuple[float, float],
        left_yaw_deg: float,
        left_pitch_deg: float,
        right_center_px: tuple[float, float],
        right_yaw_deg: float,
        right_pitch_deg: float,
        distance_m: float,
    ) -> tuple[float, float] | str:
        try:
            left_origin_xy = self._eye_origin_m(left_center_px, distance_m)
            right_origin_xy = self._eye_origin_m(right_center_px, distance_m)
            p1 = np.asarray([left_origin_xy[0], left_origin_xy[1], 0.0], dtype=np.float64)
            p2 = np.asarray([right_origin_xy[0], right_origin_xy[1], 0.0], dtype=np.float64)
            d1 = self._gaze_direction(left_yaw_deg, left_pitch_deg)
            d2 = self._gaze_direction(right_yaw_deg, right_pitch_deg)
        except ValueError as exc:
            return str(exc)

        b = float(np.dot(d1, d2))
        denom = 1.0 - b * b
        if denom <= 1.0e-6:
            return "binocular-convergence rays are nearly parallel"

        w0 = p1 - p2
        d = float(np.dot(d1, w0))
        e = float(np.dot(d2, w0))
        t = (b * e - d) / denom
        u = (e - b * d) / denom
        if t <= 0.0 or u <= 0.0:
            return "binocular-convergence intersection is behind the eye plane"

        closest_left = p1 + t * d1
        closest_right = p2 + u * d2
        closest_distance = float(np.linalg.norm(closest_left - closest_right))
        max_closest_distance = 0.20
        if closest_distance > max_closest_distance:
            return "binocular-convergence rays do not meet closely"

        midpoint = (closest_left + closest_right) * 0.5
        if not np.all(np.isfinite(midpoint)):
            return "binocular-convergence produced a non-finite point"
        if midpoint[2] <= 0.0 or midpoint[2] > distance_m * 3.0:
            return "binocular-convergence depth is outside the expected range"
        return float(midpoint[0]), float(midpoint[1])


def display_size_arg(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("display size must be a positive finite inch value")
    return parsed


def camera_fov_arg(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed >= 180.0:
        raise argparse.ArgumentTypeError("camera FOV must be greater than 0 and less than 180 degrees")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["retinaface", "deim"], default="retinaface")
    parser.add_argument("--backend", choices=["tensorrt", "cuda", "cpu"], default="tensorrt")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--camera-fov", type=camera_fov_arg, default=CAMERA_HORIZONTAL_FOV_DEG)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--display-size-inch", type=display_size_arg, default=31.5)
    parser.add_argument("--display-width", type=int, default=1920)
    parser.add_argument("--display-height", type=int, default=1080)
    parser.add_argument("--calibration-file", type=Path, default=DEFAULT_CALIBRATION_FILE)
    parser.add_argument("--detector-model", type=Path, default=None)
    parser.add_argument("--retinaface-model", type=Path, default=DEFAULT_RETINAFACE_MODEL)
    parser.add_argument("--deim-model", type=Path, default=DEFAULT_DEIM_MODEL)
    parser.add_argument("--gaze-model", type=Path, default=DEFAULT_GAZE_MODEL)
    parser.add_argument("--smoothing-alpha", type=float, default=0.65)
    parser.add_argument("--smoothing-alpha-y", type=float, default=0.45)
    parser.add_argument("--preview-fps", type=float, default=8.0)
    parser.add_argument("--hide-preview", action="store_true")
    parser.add_argument("--no-flip-x", action="store_true")
    parser.add_argument("--no-flip-y", action="store_true")
    parser.add_argument("--camera-screen-x", type=float, default=0.5)
    parser.add_argument("--camera-screen-y", type=float, default=0.0)
    parser.add_argument("--eye-position-weight-x", type=float, default=1.0)
    parser.add_argument("--eye-position-weight-y", type=float, default=0.25)
    parser.add_argument("--retinaface-head-face-ratio", type=float, default=RETINAFACE_HEAD_FACE_WIDTH_RATIO)
    parser.add_argument(
        "--gaze-projection-mode",
        choices=["legacy", "binocular-screen", "binocular-convergence"],
        default="legacy",
    )
    return parser.parse_args()


def camera_arg(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def handle_commands(
    commands: queue.Queue[dict[str, Any]],
    calibration: Calibration,
    latest_raw: tuple[float, float] | None,
) -> None:
    while True:
        try:
            command = commands.get_nowait()
        except queue.Empty:
            return
        if command.get("type") == "calibration_capture":
            target = command.get("target")
            if not isinstance(target, list) or len(target) != 2:
                emit({"type": "calibration", "status": "invalid_target"})
                continue
            calibration.capture(latest_raw, (clamp01(float(target[0])), clamp01(float(target[1]))))


def main() -> None:
    args = parse_args()
    display = DisplayGeometry(args.display_width, args.display_height, args.display_size_inch)
    calibration = Calibration(args.calibration_file)
    commands: queue.Queue[dict[str, Any]] = queue.Queue()
    CommandReader(commands).start()

    detector_model = args.detector_model
    if detector_model is None:
        detector_model = args.retinaface_model if args.detector == "retinaface" else args.deim_model
    if args.detector == "retinaface":
        detector = RetinaFaceEyeDetector(detector_model, args.backend, args.score_threshold)
    else:
        detector = DeimV2EyeDetector(detector_model, args.backend, args.score_threshold)
    head_face_width_ratio = args.retinaface_head_face_ratio if args.detector == "retinaface" else 1.0
    gaze = GazeEstimator(args.gaze_model, args.backend)
    projector = ScreenProjector(
        display,
        flip_x=not args.no_flip_x,
        flip_y=not args.no_flip_y,
        camera_screen_x=args.camera_screen_x,
        camera_screen_y=args.camera_screen_y,
        eye_position_weight_x=args.eye_position_weight_x,
        eye_position_weight_y=args.eye_position_weight_y,
        camera_fov_deg=args.camera_fov,
    )
    emit(
        {
            "type": "status",
            "level": "info",
            "message": "Models loaded",
            "detector": args.detector,
            "detector_model": str(detector_model),
            "detector_providers": detector.providers,
            "head_face_width_ratio": head_face_width_ratio,
            "camera_fov_deg": projector.camera_fov_deg,
            "camera_screen_x": projector.camera_screen_x,
            "camera_screen_y": projector.camera_screen_y,
            "eye_position_weight_x": projector.eye_position_weight_x,
            "eye_position_weight_y": projector.eye_position_weight_y,
            "gaze_projection_mode": args.gaze_projection_mode,
            "gaze_providers": gaze.providers,
        }
    )

    cap = cv2.VideoCapture(camera_arg(args.camera))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        emit({"type": "status", "level": "error", "message": f"Camera could not be opened: {args.camera}"})
        while True:
            handle_commands(commands, calibration, None)
            time.sleep(0.25)

    smoothed: tuple[float, float] | None = None
    latest_raw: tuple[float, float] | None = None
    last_status = 0.0
    last_preview = 0.0
    last_projection_warning = 0.0
    preview_interval = 1.0 / max(0.5, args.preview_fps)

    try:
        while True:
            handle_commands(commands, calibration, latest_raw)
            ok, frame = cap.read()
            if not ok:
                now = time.monotonic()
                if now - last_status > 1.0:
                    emit({"type": "status", "level": "warning", "message": "Camera frame is not available"})
                    last_status = now
                time.sleep(0.05)
                continue
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)
            try:
                head, eyes = detector.detect(frame)
                now = time.monotonic()
                should_emit_preview = not args.hide_preview and now - last_preview >= preview_interval
                if should_emit_preview and (head is None or len(eyes) < 2):
                    preview_message = None
                    if head is None:
                        preview_message = "Head not detected"
                    else:
                        preview_message = f"Eyes detected: {len(eyes)}"
                    emit_preview(
                        frame,
                        head,
                        eyes,
                        preview_message,
                        head_face_width_ratio if args.detector == "retinaface" else None,
                    )
                    last_preview = now
                if head is None:
                    raise ValueError("Head was not detected")
                if len(eyes) < 2:
                    raise ValueError("Two eyes were not detected")
                gaze_estimate = gaze.estimate(frame, head, eyes)
                if should_emit_preview:
                    emit_preview(
                        frame,
                        head,
                        eyes,
                        None,
                        head_face_width_ratio if args.detector == "retinaface" else None,
                        (gaze_estimate.yaw_deg, gaze_estimate.pitch_deg),
                    )
                    last_preview = now
                distance_m = projector.distance_from_head(head, width_ratio=head_face_width_ratio)
                projection = projector.project_estimate(args.gaze_projection_mode, eyes, gaze_estimate, distance_m)
                if projection.fallback_reason is not None and now - last_projection_warning > 2.0:
                    emit(
                        {
                            "type": "status",
                            "level": "warning",
                            "message": f"{projection.fallback_reason}; falling back to legacy projection",
                            "gaze_projection_mode": args.gaze_projection_mode,
                        }
                    )
                    last_projection_warning = now
                raw = projection.point
                latest_raw = raw
                corrected = calibration.apply(raw)
                if smoothed is None:
                    smoothed = corrected
                else:
                    alpha_x = max(0.0, min(0.95, args.smoothing_alpha))
                    alpha_y = max(0.0, min(0.95, args.smoothing_alpha_y))
                    smoothed = (
                        alpha_x * smoothed[0] + (1.0 - alpha_x) * corrected[0],
                        alpha_y * smoothed[1] + (1.0 - alpha_y) * corrected[1],
                    )
                confidence = min(head.score, (eyes[0].score + eyes[1].score) * 0.5)
                emit(
                    {
                        "type": "gaze",
                        "x_norm": clamp01(smoothed[0]),
                        "y_norm": clamp01(smoothed[1]),
                        "raw_x_norm": raw[0],
                        "raw_y_norm": raw[1],
                        "confidence": confidence,
                        "distance_m": distance_m,
                        "head_face_width_ratio": head_face_width_ratio,
                        "eye_position_weight_x": projector.eye_position_weight_x,
                        "eye_position_weight_y": projector.eye_position_weight_y,
                        "gaze_projection_mode": args.gaze_projection_mode,
                        "yaw_deg": gaze_estimate.yaw_deg,
                        "pitch_deg": gaze_estimate.pitch_deg,
                    }
                )
            except Exception as exc:
                now = time.monotonic()
                if now - last_status > 1.0:
                    emit({"type": "status", "level": "warning", "message": str(exc)})
                    last_status = now
            time.sleep(0.001)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
