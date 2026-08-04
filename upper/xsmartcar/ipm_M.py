# Copyright (c) 2026 清影/123twtd
# 模块化重构贡献：高志禹

"""目标检测框到鸟瞰坐标的快速 IPM 变换。

接口与 :class:`xsmartcar.ipm_utils.FastIPM` 的使用方式一致：程序启动时
加载一次矩阵，主循环中逐帧调用 ``process()``。本模块只处理 human、car
和 gold，其他检测类别仍保留在原图坐标中交给各自模块使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


TARGET_LABELS = frozenset({"human", "car", "gold"})


@dataclass(frozen=True, slots=True)
class IPMDetection:
    """一个目标在鸟瞰图中的检测结果，坐标单位为像素。"""

    class_name: str
    score: float
    center_x_px: float
    center_y_px: float
    left_x_px: float
    right_x_px: float

    @property
    def box_xyxy(self) -> list[float]:
        """提供兼容绘图代码的鸟瞰框格式。"""

        return [self.left_x_px, self.center_y_px, self.right_x_px, self.center_y_px]


class FastDetectionIPM:
    """检测目标 IPM 处理器。

    人和车把检测框底边中心看作地面接触点。金币也按贴近地面的目标处理：
    使用底边中心做目标位置，底边左右点形成横向范围。
    """

    calibration_size = (640, 480)

    def __init__(
        self,
        npz_path,
        out_w: int = 160,
        out_h: int = 120,
        max_targets: int = 10,
        outside_margin_px: float = 20.0,
    ) -> None:
        self.out_size = (int(out_w), int(out_h))
        self.max_targets = max(1, int(max_targets))
        self.outside_margin_px = max(0.0, float(outside_margin_px))

        npz_file = Path(npz_path)
        if not npz_file.exists():
            raise FileNotFoundError(f"找不到标定文件: {npz_file}")

        with np.load(str(npz_file)) as data:
            if "M" not in data:
                raise KeyError(f"文件 {npz_file} 中没有找到矩阵 'M'")
            self.M = self._validate_matrix(data["M"], "M")

    @classmethod
    def from_ipm(
        cls,
        ipm,
        *,
        max_targets: int = 10,
        outside_margin_px: float = 20.0,
    ) -> "FastDetectionIPM":
        """Build a detection projector from an existing image IPM object."""

        if not hasattr(ipm, "M") or not hasattr(ipm, "out_size"):
            raise TypeError("ipm must provide M and out_size")

        out_w, out_h = ipm.out_size
        instance = cls.__new__(cls)
        instance.out_size = (int(out_w), int(out_h))
        instance.max_targets = max(1, int(max_targets))
        instance.outside_margin_px = max(0.0, float(outside_margin_px))
        instance.M = cls._validate_matrix(ipm.M, "M")
        return instance

    @staticmethod
    def _validate_matrix(matrix, name: str) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} 必须是有限数值组成的 3x3 矩阵")
        return matrix

    def process(self, dets, frame_shape=None) -> tuple[list[IPMDetection], bool]:
        """把一帧检测结果变换到鸟瞰坐标。

        参数 ``frame_shape`` 应传入原始图像的 ``frame.shape``。检测坐标会先
        缩放到矩阵标定使用的 640x480，再执行透视变换。返回值为
        ``(targets, success)``；检测结果尚未就绪或数量异常时 ``success=False``。
        """

        if dets is None:
            return [], False

        targets = [
            det for det in dets
            if getattr(det, "class_name", None) in TARGET_LABELS
        ]
        if len(targets) > self.max_targets:
            return [], False

        scale_x, scale_y = self._source_scale(frame_shape)
        results: list[IPMDetection] = []
        for det in targets:
            result = self._project_detection(det, scale_x, scale_y)
            if result is not None:
                results.append(result)
        return results, True

    def _source_scale(self, frame_shape) -> tuple[float, float]:
        if frame_shape is None:
            return 1.0, 1.0
        frame_h, frame_w = frame_shape[:2]
        calibration_w, calibration_h = self.calibration_size
        return (
            calibration_w / max(int(frame_w), 1),
            calibration_h / max(int(frame_h), 1),
        )

    def _project_detection(self, det, scale_x: float, scale_y: float) -> IPMDetection | None:
        try:
            x1, y1, x2, y2 = (float(value) for value in det.box_xyxy)
            score = float(det.score)
        except (AttributeError, TypeError, ValueError):
            return None

        left = min(x1, x2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        center_x = 0.5 * (left + right)
        class_name = det.class_name

        if class_name in {"human", "car"}:
            source_points = np.array(
                [[[center_x * scale_x, bottom * scale_y]]],
                dtype=np.float32,
            )
            projected = cv2.perspectiveTransform(source_points, self.M)[0]
            if not np.isfinite(projected).all():
                return None
            bird_x, bird_y = projected[0]
            if not self._within_extended_bounds(bird_x, bird_y):
                return None
            bird_x = self._clamp_x(bird_x)
            bird_y = self._clamp_y(bird_y)
            return IPMDetection(
                class_name=class_name,
                score=score,
                center_x_px=bird_x,
                center_y_px=bird_y,
                left_x_px=bird_x,
                right_x_px=bird_x,
            )

        source_points = np.array(
            [[
                [center_x * scale_x, bottom * scale_y],
                [left * scale_x, bottom * scale_y],
                [right * scale_x, bottom * scale_y],
            ]],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(source_points, self.M)[0]
        if not np.isfinite(projected).all():
            return None

        center_point = projected[0]
        if not self._within_extended_bounds(center_point[0], center_point[1]):
            return None

        range_x = projected[:, 0]
        center_x_px = self._clamp_x(center_point[0])
        center_y_px = self._clamp_y(center_point[1])
        left_x_px = self._clamp_x(float(np.min(range_x)))
        right_x_px = self._clamp_x(float(np.max(range_x)))
        return IPMDetection(
            class_name=class_name,
            score=score,
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            left_x_px=min(left_x_px, right_x_px),
            right_x_px=max(left_x_px, right_x_px),
        )

    def _within_extended_bounds(self, x: float, y: float) -> bool:
        w, h = self.out_size
        margin = self.outside_margin_px
        return -margin <= float(x) <= (w - 1 + margin) and -margin <= float(y) <= (h - 1 + margin)

    def _clamp_x(self, value: float) -> float:
        w, _ = self.out_size
        return float(max(0.0, min(float(w - 1), float(value))))

    def _clamp_y(self, value: float) -> float:
        _, h = self.out_size
        return float(max(0.0, min(float(h - 1), float(value))))
