import argparse
import math
import unittest

from screen_eye_tracking.backend import Detection, DisplayGeometry, GazeEstimate, ScreenProjector, camera_resolution_arg


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


if __name__ == "__main__":
    unittest.main()
