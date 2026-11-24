from __future__ import annotations

import unittest
from pathlib import Path

from prepare_dataset import Sample, select_calibration_samples


class CalibrationSelectionTest(unittest.TestCase):
    @staticmethod
    def sample(name: str, positive: bool) -> Sample:
        return Sample(
            source="positive" if positive else "negative",
            path=Path(name),
            positive=positive,
            split="train",
        )

    def test_selects_requested_balanced_count(self) -> None:
        samples = [self.sample(f"p{i}.jpg", True) for i in range(300)]
        samples += [self.sample(f"n{i}.jpg", False) for i in range(300)]
        selected = select_calibration_samples(samples, 500, 0.5, 42)
        self.assertEqual(len(selected), 500)
        self.assertEqual(sum(item.positive for item in selected), 250)
        self.assertEqual(sum(not item.positive for item in selected), 250)
        self.assertEqual(
            [item.path for item in selected],
            [item.path for item in select_calibration_samples(samples, 500, 0.5, 42)],
        )

    def test_rejects_insufficient_class_samples(self) -> None:
        samples = [self.sample(f"p{i}.jpg", True) for i in range(10)]
        samples += [self.sample(f"n{i}.jpg", False) for i in range(300)]
        with self.assertRaisesRegex(ValueError, "正样本不足"):
            select_calibration_samples(samples, 500, 0.5, 42)


if __name__ == "__main__":
    unittest.main()
