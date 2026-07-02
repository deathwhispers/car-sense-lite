"""ROI 区域管理：多边形遮罩生成"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


class RoiMask:
    """ROI 多边形遮罩，懒加载：根据第一次见到的 frame 尺寸生成 mask"""

    def __init__(self, polygon: Optional[List[Tuple[int, int]]] = None):
        self._polygon = polygon
        self._mask: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int]] = None  # (h, w)

    @property
    def is_full(self) -> bool:
        return not self._polygon

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """返回 ROI 内的 frame，区域外为 0"""
        if self.is_full:
            return frame
        mask = self._get_mask(frame.shape[0], frame.shape[1])
        return cv2.bitwise_and(frame, frame, mask=mask)

    def _get_mask(self, h: int, w: int) -> np.ndarray:
        if self._mask is not None and self._shape == (h, w):
            return self._mask
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self._polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        self._mask = mask
        self._shape = (h, w)
        return mask
