"""检测器模块: 3 种可切换算法, 通过 algorithm 字段选择

可选算法:
    - mog2:         OpenCV MOG2 背景减除 (默认, 稳定)
    - running_avg:  滑动平均背景 (比 MOG2 快 2-3x, 单高斯)
    - frame_diff:   帧间差分 (最快, 适合相机固定+光照稳定)

性能 (480p, Mac M2 单核):
    mog2         ~3.0 ms/帧
    running_avg  ~1.0 ms/帧
    frame_diff   ~0.5 ms/帧
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import DetectorConfig
from .roi import RoiMask


@dataclass(frozen=True, slots=True)
class Detection:
    """单帧检测结果

    支持:
        if det.detect(frame): ...           # __bool__
        r = det.detect(frame); r.has_car    # 字段访问
        has_car, area = det.detect(frame)   # 解包
    """
    has_car: bool
    max_area: int = 0  # 最大运动区域面积 (像素²), 无车时为 0

    def __bool__(self) -> bool:
        return self.has_car


# 预定义Detection常量，避免重复创建对象
_EMPTY_DETECTION = Detection(False, 0)


class BaseDetector:
    """检测器基类"""

    algo_name: str = "base"

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.roi = RoiMask(cfg.roi)
        self._gray: Optional[np.ndarray] = None
        self._last_shape: Optional[tuple] = None

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        """单帧检测, 返回 Detection(has_car, max_area)"""
        raise NotImplementedError

    def reset(self) -> None:
        """重置内部状态 (用于镜头切换等)"""
        self._gray = None
        self._last_shape = None

    def _ensure_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        """灰度转换 (复用预分配缓冲区)"""
        h, w = frame_bgr.shape[:2]
        if self._gray is None or self._last_shape != (h, w):
            self._gray = np.empty((h, w), dtype=np.uint8)
            self._last_shape = (h, w)
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=self._gray)
        if not self.roi.is_full:
            return self.roi.apply(self._gray)
        return self._gray

    @staticmethod
    def _max_contour_area(mask: np.ndarray) -> int:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        return max((cv2.contourArea(c) for c in contours), default=0)


# ============== MOG2 ==============

class Mog2Detector(BaseDetector):
    """OpenCV MOG2 背景减除 (高斯混合模型, 多高斯自适应)

    优点: 适应光照变化、阴影抑制 (如关闭 detectShadows)
    缺点: 计算量比单高斯 / 帧差法大
    """
    algo_name = "mog2"

    def __init__(self, cfg: DetectorConfig):
        super().__init__(cfg)
        self._init_bg_subtractor()
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._fg_mask: Optional[np.ndarray] = None

    def _init_bg_subtractor(self) -> None:
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history,
            varThreshold=self.cfg.var_threshold,
            detectShadows=False,
        )

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        if frame_bgr is None or frame_bgr.size == 0:
            return _EMPTY_DETECTION
        gray = self._ensure_gray(frame_bgr)
        h, w = gray.shape[:2]
        if self._fg_mask is None or self._last_shape != (h, w):
            self._fg_mask = np.empty((h, w), dtype=np.uint8)
        fg_mask = self.bg_subtractor.apply(gray, self._fg_mask)
        cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel_open, dst=fg_mask)
        cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel_close, dst=fg_mask)
        max_area = self._max_contour_area(fg_mask)
        return Detection(max_area >= self.cfg.min_area, max_area)

    def reset(self) -> None:
        super().reset()
        self._init_bg_subtractor()
        self._fg_mask = None


# ============== 滑动平均背景 ==============

class RunningAvgDetector(BaseDetector):
    """单高斯滑动平均背景: background = α·curr + (1-α)·background

    优点: 比 MOG2 快 2-3x, 内存占用低 (1 通道 float32)
    缺点: 适应性弱于 MOG2 (快速光照变化时会误报)

    适用: 相机固定、光照缓慢变化的场景
    """
    algo_name = "running_avg"

    def __init__(self, cfg: DetectorConfig):
        super().__init__(cfg)
        self._bg: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._alpha = max(0.001, min(1.0, cfg.bg_alpha))
        self._gray_f: Optional[np.ndarray] = None
        self._diff: Optional[np.ndarray] = None
        self._fg_mask: Optional[np.ndarray] = None

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        if frame_bgr is None or frame_bgr.size == 0:
            return _EMPTY_DETECTION
        gray = self._ensure_gray(frame_bgr)
        h, w = gray.shape[:2]
        if self._gray_f is None or self._last_shape != (h, w):
            self._gray_f = np.empty((h, w), dtype=np.float32)
            self._diff = np.empty((h, w), dtype=np.float32)
            self._fg_mask = np.empty((h, w), dtype=np.uint8)
        np.copyto(self._gray_f, gray)

        if self._bg is None:
            self._bg = self._gray_f.copy()
            return _EMPTY_DETECTION

        cv2.absdiff(self._gray_f, self._bg, dst=self._diff)
        diff_u8 = cv2.convertScaleAbs(self._diff)
        cv2.threshold(
            diff_u8, self.cfg.diff_threshold, 255,
            cv2.THRESH_BINARY, dst=self._fg_mask
        )
        cv2.morphologyEx(self._fg_mask, cv2.MORPH_OPEN, self._kernel, dst=self._fg_mask)
        cv2.accumulateWeighted(self._gray_f, self._bg, self._alpha)
        area = cv2.countNonZero(self._fg_mask)
        return Detection(area >= self.cfg.min_area, area)

    def reset(self) -> None:
        super().reset()
        self._bg = None
        self._gray_f = None
        self._diff = None
        self._fg_mask = None


# ============== 帧间差分 ==============

class FrameDiffDetector(BaseDetector):
    """帧间差分 + 累积 mask: 当前帧与上一帧做差, 累积后超过阈值视为有车

    优点: 极快, 零背景学习, 对偶发噪声天然抗性
    缺点: 相机抖动/光照突变敏感; 静止目标不响应

    适用: 相机完全固定、光照稳定、目标持续运动的场景
          (俯视/斜俯视车道, 几乎完美匹配)
    """
    algo_name = "frame_diff"

    def __init__(self, cfg: DetectorConfig):
        super().__init__(cfg)
        self._prev_gray: Optional[np.ndarray] = None
        self._acc_mask: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._decay = max(1, int(cfg.diff_decay))
        self._diff: Optional[np.ndarray] = None
        self._fg: Optional[np.ndarray] = None

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        if frame_bgr is None or frame_bgr.size == 0:
            return _EMPTY_DETECTION
        gray = self._ensure_gray(frame_bgr)
        h, w = gray.shape[:2]
        if self._diff is None or self._last_shape != (h, w):
            self._diff = np.empty((h, w), dtype=np.uint8)
            self._fg = np.empty((h, w), dtype=np.uint8)

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return _EMPTY_DETECTION

        cv2.absdiff(gray, self._prev_gray, dst=self._diff)
        cv2.threshold(
            self._diff, self.cfg.diff_threshold, 255, cv2.THRESH_BINARY, dst=self._fg
        )
        cv2.morphologyEx(self._fg, cv2.MORPH_OPEN, self._kernel, dst=self._fg)
        if self._acc_mask is None:
            self._acc_mask = np.zeros_like(self._fg)
        cv2.subtract(self._acc_mask, self._decay, dst=self._acc_mask)
        np.maximum(self._acc_mask, self._fg, out=self._acc_mask)
        np.copyto(self._prev_gray, gray)
        area = cv2.countNonZero(self._acc_mask)
        return Detection(area >= self.cfg.min_area, area)

    def reset(self) -> None:
        super().reset()
        self._prev_gray = None
        self._acc_mask = None
        self._diff = None
        self._fg = None


# ============== 工厂 ==============

ALGO_REGISTRY = {
    "mog2": Mog2Detector,
    "running_avg": RunningAvgDetector,
    "frame_diff": FrameDiffDetector,
}


def create_detector(cfg: DetectorConfig) -> BaseDetector:
    algo = cfg.algorithm.lower()
    cls = ALGO_REGISTRY.get(algo)
    if cls is None:
        raise ValueError(
            f"unknown algorithm: {cfg.algorithm!r}, "
            f"available: {list(ALGO_REGISTRY.keys())}"
        )
    return cls(cfg)
