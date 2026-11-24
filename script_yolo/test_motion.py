#!/usr/bin/env python3

from __future__ import annotations

import unittest
from collections import deque

import cv2
import numpy as np

from infer_video import alert_ready
from motion import MotionAnalyzer, MotionConfig


class MotionAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 150, size=(120, 160), dtype=np.uint8)
        self.background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.config = MotionConfig(
            analysis_width=160,
            frame_gap=1,
            diff_pixel_threshold=18,
            diff_ratio_threshold=0.02,
            flow_threshold=0.75,
            min_flow_points=4,
        )

    def test_rejects_global_camera_translation(self) -> None:
        analyzer = MotionAnalyzer(self.config)
        analyzer.analyze(self.background, [[10, 10, 150, 110]])
        shifted = cv2.warpAffine(
            self.background,
            np.float32([[1, 0, 4], [0, 1, 3]]),
            (160, 120),
            borderMode=cv2.BORDER_REFLECT,
        )

        metric = analyzer.analyze(shifted, [[10, 10, 150, 110]])[0]

        self.assertFalse(metric.confirmed)
        self.assertLess(metric.flow_magnitude, self.config.flow_threshold)

    def test_accepts_local_motion_on_static_background(self) -> None:
        analyzer = MotionAnalyzer(self.config)
        previous = self.background.copy()
        current = self.background.copy()
        texture = np.zeros((36, 36, 3), dtype=np.uint8)
        texture[::4, :] = 255
        texture[:, ::4] = 255
        previous[40:76, 40:76] = texture
        current[40:76, 40:76] = self.background[40:76, 40:76]
        current[40:76, 50:86] = texture
        box = [35, 35, 92, 82]
        analyzer.analyze(previous, [box])

        metric = analyzer.analyze(current, [box])[0]

        self.assertTrue(metric.confirmed)
        self.assertGreaterEqual(metric.diff_ratio, self.config.diff_ratio_threshold)
        self.assertGreaterEqual(metric.flow_magnitude, self.config.flow_threshold)

    def test_frame_difference_fallback_for_textureless_motion(self) -> None:
        analyzer = MotionAnalyzer(self.config)
        previous = np.zeros((120, 160, 3), dtype=np.uint8)
        current = previous.copy()
        current[40:80, 50:90] = 255
        box = [45, 35, 95, 85]
        analyzer.analyze(previous, [box])

        metric = analyzer.analyze(current, [box])[0]

        self.assertFalse(metric.flow_available)
        self.assertTrue(metric.diff_pass)
        self.assertTrue(metric.confirmed)

    def test_alert_requires_6_votes_in_full_10_frame_window(self) -> None:
        self.assertFalse(alert_ready(deque([True] * 9, maxlen=10), 10, 6))
        self.assertFalse(
            alert_ready(deque([True] * 5 + [False] * 5, maxlen=10), 10, 6)
        )
        self.assertTrue(
            alert_ready(deque([True] * 6 + [False] * 4, maxlen=10), 10, 6)
        )


if __name__ == "__main__":
    unittest.main()
