"""Camera-compensated frame-difference and sparse optical-flow analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    analysis_width: int = 640
    frame_gap: int = 5
    diff_pixel_threshold: int = 18
    diff_ratio_threshold: float = 0.003
    flow_threshold: float = 0.20
    max_camera_motion: float = 3.0
    min_flow_points: int = 4
    max_corners: int = 500
    quality_level: float = 0.01
    min_distance: int = 7


@dataclass
class MotionMetrics:
    available: bool = False
    confirmed: bool = False
    diff_ratio: float = 0.0
    flow_magnitude: float = 0.0
    flow_points: int = 0
    global_points: int = 0
    camera_dx: float = 0.0
    camera_dy: float = 0.0
    affine_inlier_ratio: float = 0.0
    camera_motion: float = 0.0
    camera_stable: bool = True
    diff_pass: bool = False
    flow_available: bool = False
    flow_pass: bool = False

    def to_dict(self) -> dict[str, bool | float | int]:
        values = asdict(self)
        return {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in values.items()
        }


class MotionAnalyzer:
    """Compare each frame with its predecessor after compensating camera motion."""

    def __init__(self, config: MotionConfig) -> None:
        self.config = config
        self.history: deque[np.ndarray] = deque(maxlen=max(2, config.frame_gap + 1))

    def reset(self) -> None:
        self.history.clear()

    def analyze(
        self, frame: np.ndarray, boxes_xyxy: list[list[float]]
    ) -> list[MotionMetrics]:
        gray, scale = self._prepare(frame)
        if self.history and self.history[-1].shape != gray.shape:
            self.history.clear()
        self.history.append(gray)
        if len(self.history) <= self.config.frame_gap:
            return [MotionMetrics() for _ in boxes_xyxy]

        previous = self.history[0]
        previous_points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=7,
        )
        tracked_previous, tracked_current = self._track(previous, gray, previous_points)
        affine, inlier_ratio = self._estimate_camera_motion(
            tracked_previous, tracked_current
        )

        height, width = gray.shape
        previous_blurred = cv2.GaussianBlur(previous, (5, 5), 0)
        current_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        aligned_previous = cv2.warpAffine(
            previous_blurred,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.warpAffine(
            np.full_like(previous, 255),
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        difference = cv2.absdiff(current_blurred, aligned_previous)
        changed = cv2.threshold(
            difference,
            self.config.diff_pixel_threshold,
            255,
            cv2.THRESH_BINARY,
        )[1]
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        residuals = self._flow_residuals(tracked_previous, tracked_current, affine)
        metrics: list[MotionMetrics] = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = self._scaled_box(box, scale, width, height)
            if x2 <= x1 or y2 <= y1:
                metrics.append(MotionMetrics(available=True))
                continue
            roi_valid = valid[y1:y2, x1:x2] > 0
            valid_pixels = int(np.count_nonzero(roi_valid))
            if valid_pixels:
                roi_changed = changed[y1:y2, x1:x2] > 0
                diff_ratio = float(np.count_nonzero(roi_changed & roi_valid) / valid_pixels)
            else:
                diff_ratio = 0.0

            if len(tracked_current):
                point_mask = (
                    (tracked_current[:, 0] >= x1)
                    & (tracked_current[:, 0] < x2)
                    & (tracked_current[:, 1] >= y1)
                    & (tracked_current[:, 1] < y2)
                )
                roi_residuals = residuals[point_mask]
            else:
                roi_residuals = np.empty((0,), dtype=np.float32)
            flow_points = int(len(roi_residuals))
            flow_magnitude = (
                float(np.percentile(roi_residuals, 75)) if flow_points else 0.0
            )
            diff_pass = diff_ratio >= self.config.diff_ratio_threshold
            flow_available = flow_points >= self.config.min_flow_points
            flow_pass = flow_magnitude >= self.config.flow_threshold
            camera_motion = float(np.hypot(affine[0, 2], affine[1, 2]))
            camera_stable = (
                self.config.max_camera_motion < 0
                or camera_motion <= self.config.max_camera_motion
            )
            confirmed = (
                camera_stable
                and diff_pass
                and (flow_pass if flow_available else True)
            )
            metrics.append(
                MotionMetrics(
                    available=True,
                    confirmed=confirmed,
                    diff_ratio=diff_ratio,
                    flow_magnitude=flow_magnitude,
                    flow_points=flow_points,
                    global_points=int(len(tracked_current)),
                    camera_dx=float(affine[0, 2]),
                    camera_dy=float(affine[1, 2]),
                    affine_inlier_ratio=inlier_ratio,
                    camera_motion=camera_motion,
                    camera_stable=camera_stable,
                    diff_pass=diff_pass,
                    flow_available=flow_available,
                    flow_pass=flow_pass,
                )
            )
        return metrics

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        scale = min(1.0, self.config.analysis_width / float(width))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale

    @staticmethod
    def _track(
        previous: np.ndarray,
        current: np.ndarray,
        previous_points: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if previous_points is None or len(previous_points) < 4:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or status is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty
        mask = status.reshape(-1).astype(bool)
        return previous_points.reshape(-1, 2)[mask], current_points.reshape(-1, 2)[mask]

    @staticmethod
    def _estimate_camera_motion(
        previous_points: np.ndarray, current_points: np.ndarray
    ) -> tuple[np.ndarray, float]:
        identity = np.float32([[1, 0, 0], [0, 1, 0]])
        if len(previous_points) < 8:
            return identity, 0.0
        affine, inliers = cv2.estimateAffinePartial2D(
            previous_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None:
            displacement = np.median(current_points - previous_points, axis=0)
            identity[:, 2] = displacement
            return identity, 0.0
        ratio = float(np.mean(inliers)) if inliers is not None else 0.0
        return affine.astype(np.float32), ratio

    @staticmethod
    def _flow_residuals(
        previous_points: np.ndarray,
        current_points: np.ndarray,
        affine: np.ndarray,
    ) -> np.ndarray:
        if not len(previous_points):
            return np.empty((0,), dtype=np.float32)
        homogeneous = np.column_stack(
            (previous_points, np.ones(len(previous_points), dtype=np.float32))
        )
        predicted = homogeneous @ affine.T
        return np.linalg.norm(current_points - predicted, axis=1)

    @staticmethod
    def _scaled_box(
        box: list[float], scale: float, width: int, height: int
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            max(0, min(width, int(x1 * scale))),
            max(0, min(height, int(y1 * scale))),
            max(0, min(width, int(np.ceil(x2 * scale)))),
            max(0, min(height, int(np.ceil(y2 * scale)))),
        )
