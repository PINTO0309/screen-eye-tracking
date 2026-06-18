import math
import unittest

from screen_eye_tracking.backend import Detection, DisplayGeometry, GazeEstimate, ScreenProjector


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


if __name__ == "__main__":
    unittest.main()
