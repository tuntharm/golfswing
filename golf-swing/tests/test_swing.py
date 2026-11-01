"""Checks the plane sign, the shaft detector, and shallow versus steep."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from swinglab.analyze import (
    above_plane_deg,
    analyze_frames,
    classify_shallowing,
    detect_shaft,
    elevation_deg,
    shaft_angle_deg,
)
from swinglab.render import draw_swing_fan, overlay_catalog
from swinglab.synthetic import design_swing, paint_swing


def _angle_error(actual: float, expected: float) -> float:
    return abs((actual - expected + 180) % 360 - 180)


class PlaneMathTest(unittest.TestCase):
    def test_address_line_is_zero_and_sky_side_is_positive(self) -> None:
        grip = (420.0, 210.0)
        head = (300.0, 390.0)
        self.assertAlmostEqual(above_plane_deg(grip, head, grip, head), 0.0, places=5)
        midpoint = ((grip[0] + head[0]) / 2, (grip[1] + head[1]) / 2)
        raised = (grip[0], grip[1] - 40)
        raised_head = (head[0], head[1] - 40)
        self.assertGreater(above_plane_deg(raised, raised_head, grip, head), 5)
        lowered = (grip[0], grip[1] + 40)
        lowered_head = (head[0], head[1] + 40)
        self.assertLess(above_plane_deg(lowered, lowered_head, grip, head), -5)
        self.assertGreater(midpoint[1], 0)

    def test_horizontal_shaft_has_low_elevation(self) -> None:
        self.assertLess(elevation_deg((0, 0), (100, 2)), 5)
        self.assertGreater(elevation_deg((0, 0), (0, 100)), 80)

    def test_not_shallowing_is_the_steep_delivery(self) -> None:
        self.assertEqual(classify_shallowing(22, -10, 0.9, 0.0)[0], "shallowed")
        self.assertEqual(classify_shallowing(22, 20, 0.9, 0.0)[0], "not_shallowing")
        self.assertEqual(classify_shallowing(10, 24, 0.9, 0.0)[0], "steepened")
        self.assertEqual(classify_shallowing(22, -10, 0.2, 0.0)[0], "unclear")


class ShaftDetectorTest(unittest.TestCase):
    def test_recovers_a_dark_shaft_and_a_bright_shaft(self) -> None:
        for color, background in ((20, 170), (230, 40)):
            for angle in (20, 125, 230, 340):
                image = np.full((480, 640), background, np.uint8)
                grip = (320.0, 240.0)
                length = 190.0
                radians = math.radians(angle)
                head = (grip[0] + length * math.cos(radians), grip[1] + length * math.sin(radians))
                cv2.line(image, (int(grip[0]), int(grip[1])), (int(head[0]), int(head[1])), color, 3)
                detected = detect_shaft(image, grip)
                self.assertLess(
                    _angle_error(detected["angle"], shaft_angle_deg(grip, head)),
                    6,
                    msg=f"color {color} angle {angle} got {detected['angle']}",
                )
                self.assertGreater(detected["confidence"], 0.5)


class SwingReadingTest(unittest.TestCase):
    def test_shallow_and_steep_cartoons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind, expected in (("shallow", "shallowed"), ("steep", "not_shallowing")):
                samples = design_swing(kind)
                report = analyze_frames(
                    paint_swing(samples),
                    fps=30,
                    poses=[sample["pose"] for sample in samples],
                    playback_speed=0.25,
                    out_dir=root / kind,
                )
                self.assertEqual(report["reading"], expected, msg=report["shallowing"])
                self.assertEqual(report["playbackSpeed"], 0.25)
                self.assertIsNone(report["delivery"]["clubPathDegrees"])
                video = Path(report["videoPath"])
                self.assertTrue(video.is_file())
                duration = float(
                    subprocess.check_output(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=nw=1:nk=1",
                            str(video),
                        ],
                        text=True,
                    )
                )
                self.assertGreater(duration, 5.5)
                self.assertLess(duration, 7.5)
                still = cv2.imread(report["p6StillPath"])
                self.assertIsNotNone(still)
                self.assertGreater(float(still.std()), 10)

    def test_left_handed_flag_mirrors_a_right_handed_shallow_swing(self) -> None:
        samples = design_swing("shallow")
        frames = [cv2.flip(frame, 1) for frame in paint_swing(samples)]
        width = frames[0].shape[1]

        def flip_pose(pose: dict) -> dict:
            flipped = {}
            for key, value in pose.items():
                if isinstance(value, tuple):
                    flipped[key] = (width - 1 - value[0], value[1])
                else:
                    flipped[key] = value
            return flipped

        poses = [flip_pose(sample["pose"]) for sample in samples]
        report = analyze_frames(frames, fps=30, poses=poses, handed="left", playback_speed=0.25)
        self.assertEqual(report["reading"], "shallowed")


class OverlayElementTest(unittest.TestCase):
    def test_perpendicular_shaft_stays_in_the_data_and_off_the_picture(self) -> None:
        frame = np.zeros((200, 200, 3), np.uint8)
        samples = [
            {"grip": (20.0, 180.0), "head": (180.0, 40.0), "downswing": False},
            {"grip": (100.0, 160.0), "head": (100.0, 30.0), "downswing": False},
        ]
        hidden = draw_swing_fan(frame, samples, 1, show_legend=False)
        shown = draw_swing_fan(
            frame,
            samples,
            1,
            show_legend=False,
            elements={"perpendicular": True},
        )
        self.assertLess(int(hidden[50, 100].max()), 30)
        self.assertGreater(int(shown[50, 100, 0]), 200)
        self.assertGreater(int(hidden[:, :, 0].max()), 200)
        catalog = overlay_catalog(samples)
        self.assertFalse(catalog["perpendicular"]["visible"])
        self.assertTrue(catalog["shaft_fan"]["visible"])
        self.assertEqual(len(catalog["perpendicular"]["lines"]), 1)
        self.assertEqual(len(catalog["shaft_fan"]["lines"]), 1)
        self.assertEqual(catalog["perpendicular"]["lines"][0]["head"], [100.0, 30.0])


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
class ModuleLoadTest(unittest.TestCase):
    def test_package_imports(self) -> None:
        import swinglab

        self.assertTrue(swinglab.__version__)


if __name__ == "__main__":
    unittest.main()
