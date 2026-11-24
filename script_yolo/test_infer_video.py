#!/usr/bin/env python3

from __future__ import annotations

import unittest

from infer_video import TemporalAlert


class TemporalAlertTest(unittest.TestCase):
    def test_activates_after_six_hits_in_full_ten_frame_window(self) -> None:
        state = TemporalAlert(window=10, activate_hits=6, clear_misses=20)
        transitions = [state.update(hit)[1] for hit in [True] * 6 + [False] * 4]
        self.assertEqual(transitions[-1], "activated")
        self.assertTrue(state.active)

    def test_does_not_activate_on_short_burst(self) -> None:
        state = TemporalAlert(window=10, activate_hits=6, clear_misses=20)
        for hit in [True] * 5 + [False] * 10:
            state.update(hit)
        self.assertFalse(state.active)

    def test_hysteresis_keeps_alert_during_brief_detector_dropout(self) -> None:
        state = TemporalAlert(window=10, activate_hits=6, clear_misses=20)
        for hit in [True] * 10:
            state.update(hit)
        for hit in [False] * 4:
            alert, transition = state.update(hit)
        self.assertTrue(alert)
        self.assertIsNone(transition)

    def test_clears_after_detection_disappears(self) -> None:
        state = TemporalAlert(window=10, activate_hits=6, clear_misses=20)
        for hit in [True] * 10:
            state.update(hit)
        transitions = [state.update(False)[1] for _ in range(20)]
        self.assertIn("cleared", transitions)
        self.assertFalse(state.active)


if __name__ == "__main__":
    unittest.main()
