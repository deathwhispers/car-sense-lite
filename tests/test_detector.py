"""核心检测器冒烟测试: 验证工厂能正确创建各算法检测器

(详细的算法测试在 test_algorithms.py)
"""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.config import DetectorConfig
from src.detector import create_detector


def make_static_frame(h: int = 480, w: int = 640, value: int = 100) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestDetectorSmoke(unittest.TestCase):
    """所有算法的冒烟测试: 验证基本能跑"""

    def _make(self, algorithm: str, **kw) -> DetectorConfig:
        cfg = DetectorConfig(
            algorithm=algorithm, downsample=1.0, frame_skip=1,
            min_area=500, trigger_frames=1, cooldown_sec=0,
            warmup_sec=0, history=30, var_threshold=16,
            diff_threshold=25, bg_alpha=0.05, diff_decay=5,
            **kw,
        )
        return cfg

    def test_all_algorithms_create(self):
        for algo in ("mog2", "running_avg", "frame_diff"):
            det = create_detector(self._make(algo))
            self.assertIsNotNone(det)
            self.assertEqual(det.algo_name, algo)

    def test_algorithm_dispatch(self):
        """工厂能根据 algorithm 字段选择正确的实现"""
        for algo in ("mog2", "running_avg", "frame_diff"):
            det = create_detector(self._make(algo))
            self.assertEqual(det.algo_name, algo)

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            create_detector(self._make("unknown_algo"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
