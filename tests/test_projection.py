import argparse
import math
import unittest
from pathlib import Path

import numpy as np

from screen_eye_tracking.backend import Detection, DisplayGeometry, GazeEstimate, LipMotionEstimator, ScreenProjector, camera_resolution_arg, crop_detection, parse_yolo_output


class ScreenProjectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.display = DisplayGeometry(1920, 1080, 31.5)
        self.projector = ScreenProjector(
            self.display,
            flip_x=False,
            flip_y=False,
            camera_screen_x=0.5,
            camera_screen_y=0.5,
        )
        self.left_eye = Detection(17, 1.0, 296.0, 236.0, 304.0, 244.0)
        self.right_eye = Detection(17, 1.0, 336.0, 236.0, 344.0, 244.0)
        self.eyes = [self.left_eye, self.right_eye]
        self.distance_m = 0.6

    def test_legacy_matches_existing_projection_formula(self) -> None:
        point = self.projector.project((320.0, 240.0), 0.0, 0.0, self.distance_m)

        self.assertEqual(point, (0.5, 0.5))

    def test_camera_fov_controls_focal_length(self) -> None:
        narrow_fov_projector = ScreenProjector(self.display, camera_fov_deg=60.0)

        self.assertEqual(narrow_fov_projector.camera_fov_deg, 60.0)
        self.assertGreater(narrow_fov_projector.focal_px, self.projector.focal_px)

    def test_full_hd_camera_center_projects_to_screen_center(self) -> None:
        projector = ScreenProjector(
            self.display,
            flip_x=False,
            flip_y=False,
            camera_screen_x=0.5,
            camera_screen_y=0.5,
            camera_width=1920,
            camera_height=1080,
        )

        self.assertEqual(projector.project((960.0, 540.0), 0.0, 0.0, self.distance_m), (0.5, 0.5))

    def test_binocular_screen_matches_legacy_when_eye_angles_match(self) -> None:
        estimate = GazeEstimate(
            yaw_deg=5.0,
            pitch_deg=2.0,
            left_yaw_deg=5.0,
            left_pitch_deg=2.0,
            right_yaw_deg=5.0,
            right_pitch_deg=2.0,
        )

        legacy = self.projector.project((320.0, 240.0), estimate.yaw_deg, estimate.pitch_deg, self.distance_m)
        result = self.projector.project_estimate("binocular-screen", self.eyes, estimate, self.distance_m)

        self.assertIsNone(result.fallback_reason)
        self.assertAlmostEqual(result.point[0], legacy[0], places=7)
        self.assertAlmostEqual(result.point[1], legacy[1], places=7)

    def test_binocular_convergence_returns_finite_point_for_meeting_rays(self) -> None:
        yaw = math.degrees(math.atan((20.0 * self.distance_m / self.projector.focal_px) / self.distance_m))
        estimate = GazeEstimate(
            yaw_deg=0.0,
            pitch_deg=0.0,
            left_yaw_deg=yaw,
            left_pitch_deg=0.0,
            right_yaw_deg=-yaw,
            right_pitch_deg=0.0,
        )

        result = self.projector.project_estimate("binocular-convergence", self.eyes, estimate, self.distance_m)

        self.assertIsNone(result.fallback_reason)
        self.assertTrue(math.isfinite(result.point[0]))
        self.assertTrue(math.isfinite(result.point[1]))
        self.assertAlmostEqual(result.point[0], 0.5, places=7)
        self.assertAlmostEqual(result.point[1], 0.5, places=7)

    def test_binocular_convergence_falls_back_for_parallel_rays(self) -> None:
        estimate = GazeEstimate(
            yaw_deg=0.0,
            pitch_deg=0.0,
            left_yaw_deg=0.0,
            left_pitch_deg=0.0,
            right_yaw_deg=0.0,
            right_pitch_deg=0.0,
        )

        result = self.projector.project_estimate("binocular-convergence", self.eyes, estimate, self.distance_m)

        self.assertIsNotNone(result.fallback_reason)
        self.assertEqual(result.point, self.projector.project((320.0, 240.0), 0.0, 0.0, self.distance_m))


class CameraResolutionArgTest(unittest.TestCase):
    def test_accepts_presets_and_dimensions(self) -> None:
        self.assertEqual((camera_resolution_arg("VGA").width, camera_resolution_arg("VGA").height), (640, 480))
        self.assertEqual(
            (camera_resolution_arg("Full HD").width, camera_resolution_arg("Full HD").height),
            (1920, 1080),
        )
        self.assertEqual((camera_resolution_arg("1080p").width, camera_resolution_arg("1080p").height), (1920, 1080))
        self.assertEqual(
            (camera_resolution_arg("1280x720").width, camera_resolution_arg("1280x720").height),
            (1280, 720),
        )

    def test_rejects_duplicate_aliases_and_invalid_values(self) -> None:
        for value in ("2MP", "not-a-size", "1280x0"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    camera_resolution_arg(value)


class YoloWholeBody28ParserTest(unittest.TestCase):
    @staticmethod
    def output(candidate_count: int) -> np.ndarray:
        return np.zeros((1, 32, candidate_count), dtype=np.float32)

    @staticmethod
    def set_box(output: np.ndarray, index: int, cx: float, cy: float, width: float, height: float) -> None:
        output[0, 0, index] = cx
        output[0, 1, index] = cy
        output[0, 2, index] = width
        output[0, 3, index] = height

    @staticmethod
    def set_score(output: np.ndarray, index: int, class_id: int, score: float) -> None:
        output[0, 4 + class_id, index] = score

    def test_maps_classes_and_scales_direct_resize_boxes(self) -> None:
        output = self.output(3)
        self.set_box(output, 0, 320.0, 240.0, 200.0, 220.0)
        self.set_score(output, 0, 7, 0.8)
        self.set_box(output, 1, 270.0, 230.0, 30.0, 20.0)
        self.set_score(output, 1, 17, 0.21)
        self.set_box(output, 2, 370.0, 230.0, 30.0, 20.0)
        self.set_score(output, 2, 17, 0.22)

        result = parse_yolo_output(output, 0.75, image_w=1280, image_h=720)
        head, eyes = result.head, result.eyes

        self.assertIsNotNone(head)
        assert head is not None
        self.assertEqual(head.class_id, 7)
        self.assertAlmostEqual(head.x1, 440.0)
        self.assertAlmostEqual(head.y1, 195.0)
        self.assertAlmostEqual(head.x2, 840.0)
        self.assertAlmostEqual(head.y2, 525.0)
        self.assertEqual([eye.class_id for eye in eyes], [17, 17])
        self.assertLess(eyes[0].x1, eyes[1].x1)

    def test_head_uses_score_threshold_and_eye_uses_fixed_threshold(self) -> None:
        output = self.output(3)
        self.set_box(output, 0, 320.0, 240.0, 200.0, 220.0)
        self.set_score(output, 0, 7, 0.79)
        self.set_box(output, 1, 270.0, 230.0, 30.0, 20.0)
        self.set_score(output, 1, 17, 0.19)
        self.set_box(output, 2, 370.0, 230.0, 30.0, 20.0)
        self.set_score(output, 2, 17, 0.20)

        no_detection = parse_yolo_output(output, 0.8)
        detection = parse_yolo_output(output, 0.79)
        head, eyes = detection.head, detection.eyes

        self.assertIsNone(no_detection.head)
        self.assertEqual(no_detection.eyes, [])
        self.assertIsNotNone(head)
        self.assertEqual(len(eyes), 1)
        self.assertAlmostEqual(eyes[0].score, 0.20, places=6)

    def test_applies_class_specific_nms(self) -> None:
        output = self.output(5)
        self.set_box(output, 0, 320.0, 240.0, 220.0, 220.0)
        self.set_score(output, 0, 7, 0.9)
        self.set_box(output, 1, 322.0, 242.0, 220.0, 220.0)
        self.set_score(output, 1, 7, 0.85)
        self.set_box(output, 2, 260.0, 230.0, 30.0, 20.0)
        self.set_score(output, 2, 17, 0.7)
        self.set_box(output, 3, 262.0, 231.0, 30.0, 20.0)
        self.set_score(output, 3, 17, 0.6)
        self.set_box(output, 4, 380.0, 230.0, 30.0, 20.0)
        self.set_score(output, 4, 17, 0.65)

        result = parse_yolo_output(output, 0.5)
        head, eyes = result.head, result.eyes

        self.assertIsNotNone(head)
        assert head is not None
        self.assertAlmostEqual(head.score, 0.9)
        self.assertEqual([round(eye.score, 2) for eye in eyes], [0.7, 0.65])

    def test_selects_mouth_with_head_threshold_inside_head(self) -> None:
        low_output = self.output(2)
        self.set_box(low_output, 0, 320.0, 240.0, 220.0, 220.0)
        self.set_score(low_output, 0, 7, 0.9)
        self.set_box(low_output, 1, 320.0, 300.0, 40.0, 20.0)
        self.set_score(low_output, 1, 19, 0.49)

        self.assertIsNone(parse_yolo_output(low_output, 0.5).mouth)

        output = self.output(5)
        self.set_box(output, 0, 320.0, 240.0, 220.0, 220.0)
        self.set_score(output, 0, 7, 0.9)
        self.set_box(output, 1, 320.0, 300.0, 40.0, 20.0)
        self.set_score(output, 1, 19, 0.19)
        self.set_box(output, 2, 320.0, 300.0, 40.0, 20.0)
        self.set_score(output, 2, 19, 0.6)
        self.set_box(output, 3, 322.0, 301.0, 40.0, 20.0)
        self.set_score(output, 3, 19, 0.5)
        self.set_box(output, 4, 600.0, 450.0, 40.0, 20.0)
        self.set_score(output, 4, 19, 0.95)

        result = parse_yolo_output(output, 0.5)

        self.assertIsNotNone(result.mouth)
        assert result.mouth is not None
        self.assertEqual(result.mouth.class_id, 19)
        self.assertAlmostEqual(result.mouth.score, 0.6)


class LipMotionEstimatorTest(unittest.TestCase):
    def test_mouth_crop_uses_vsdlm_margins(self) -> None:
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
        crop = crop_detection(
            image,
            Detection(19, 1.0, 1.0, 1.0, 3.0, 3.0),
            4,
            4,
            margin_top=2,
            margin_bottom=6,
            margin_left=2,
            margin_right=2,
        )

        self.assertEqual(crop.shape, (4, 4, 3))
        np.testing.assert_array_equal(crop, image)

    def test_onnx_model_accepts_zero_mouth_crop(self) -> None:
        model_path = Path("public/models/vsdlm_l.onnx")
        if not model_path.exists():
            self.skipTest(f"{model_path} is not available")
        estimator = LipMotionEstimator(model_path, "cpu")
        probability = estimator.estimate(np.zeros((30, 48, 3), dtype=np.uint8), Detection(19, 1.0, 0.0, 0.0, 48.0, 30.0))

        self.assertTrue(math.isfinite(probability))


if __name__ == "__main__":
    unittest.main()
