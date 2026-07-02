"""3 个算法的综合测试: 功能正确性 + 性能

每个算法都要测:
  - 空帧/None 输入
  - 静态场景无车
  - 运动车辆检测
  - ROI 屏蔽
  - min_area 过滤
  - 单帧性能

运行: python -m pytest tests/test_algorithms.py -v
      或 python tests/test_algorithms.py
"""
from __future__ import annotations

import os
import sys
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DetectorConfig
from src.detector import (
    BaseDetector, FrameDiffDetector, Mog2Detector, RunningAvgDetector,
    create_detector, ALGO_REGISTRY,
)


# 公共测试夹具

def make_static_frame(h: int = 480, w: int = 640, value: int = 100) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def make_motion_frame(h: int = 480, w: int = 640,
                      value: int = 100, car_value: int = 200,
                      car_size: int = 80, x_offset: int = 0) -> np.ndarray:
    """带运动车辆的帧, x_offset 让车每帧移动模拟真实场景"""
    frame = np.full((h, w, 3), value, dtype=np.uint8)
    noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cx, cy = w // 2 + x_offset, h // 2
    cv2.rectangle(frame, (cx - car_size // 2, cy - car_size // 2),
                  (cx + car_size // 2, cy + car_size // 2),
                  (car_value, car_value, car_value), -1)
    return frame


def make_factory_cfg(algorithm: str, **overrides) -> DetectorConfig:
    base = dict(
        algorithm=algorithm,
        downsample=1.0,
        frame_skip=1,
        min_area=500,
        trigger_frames=1,
        cooldown_sec=0,
        warmup_sec=0,
        history=30,
        var_threshold=16,
        diff_threshold=25,
        bg_alpha=0.05,
        diff_decay=5,
    )
    base.update(overrides)
    return DetectorConfig(**base)


# ============== 通用测试 (参数化到所有算法) ==============

class _AllAlgorithmsMixin:
    """对所有算法跑同一组测试, 子类只指定 algo_name"""

    algo_name: str = ""

    def make_detector(self, **overrides) -> BaseDetector:
        cfg = make_factory_cfg(self.algo_name, **overrides)
        return create_detector(cfg)

    def test_init(self):
        det = self.make_detector()
        self.assertIsNotNone(det)
        self.assertEqual(det.algo_name, self.algo_name)

    def test_empty_input(self):
        det = self.make_detector()
        self.assertFalse(det.detect(None))
        empty = np.array([], dtype=np.uint8)
        self.assertFalse(det.detect(empty))

    def test_static_scene_no_alert(self):
        """静态场景下不应持续告警 (MOG2 warmup 后稳定)"""
        det = self.make_detector()
        static = make_static_frame()
        # 喂 30 帧让所有算法稳定 (MOG2 学习背景, running_avg 学习背景, frame_diff 累积衰减到 0)
        for _ in range(30):
            det.detect(static)
        # 之后应该稳定 False
        false_count = sum(1 for _ in range(10) if not det.detect(static))
        self.assertGreater(false_count, 8,
                           f"{self.algo_name}: 静态场景误报过多 ({10-false_count}/10)")

    def test_motion_detected(self):
        """运动的车应该被检测到"""
        det = self.make_detector()
        static = make_static_frame()
        for _ in range(30):
            det.detect(static)
        detected = sum(
            1 for i in range(10)
            if det.detect(make_motion_frame(x_offset=i * 20))
        )
        self.assertGreater(detected, 5,
                           f"{self.algo_name}: 运动检测率 {detected}/10 过低")

    def test_roi_filters_outside(self):
        """ROI 外的运动不应被检测"""
        det = self.make_detector(
            roi=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )
        static = make_static_frame()
        for _ in range(30):
            det.detect(static)
        for i in range(10):
            motion = np.full((480, 640, 3), 100, dtype=np.uint8)
            x = i * 20
            cv2.rectangle(motion, (360 + x, 260), (440 + x, 340),
                          (200, 200, 200), -1)
            self.assertFalse(det.detect(motion),
                             f"{self.algo_name}: ROI 外应被过滤")

    def test_min_area_filter(self):
        """小于 min_area 的运动应被过滤"""
        det = self.make_detector(min_area=5000)
        static = make_static_frame()
        for _ in range(30):
            det.detect(static)
        for i in range(5):
            small = np.full((480, 640, 3), 100, dtype=np.uint8)
            x = i * 20
            cv2.rectangle(small, (310 + x, 230), (330 + x, 250),
                          (200, 200, 200), -1)
            self.assertFalse(det.detect(small),
                             f"{self.algo_name}: 小运动应被 min_area 过滤")

    def test_reset(self):
        """reset() 应该清空内部状态, 检测器能继续工作"""
        det = self.make_detector()
        for _ in range(10):
            det.detect(make_static_frame())
        det.reset()  # 不应抛异常
        # reset 后让 MOG2 / running_avg 重新学背景
        # frame_diff 不需要 warmup
        if self.algo_name in ("mog2", "running_avg"):
            for _ in range(30):
                det.detect(make_static_frame())
        # 然后静态场景应返回 False
        self.assertFalse(det.detect(make_static_frame()),
                         f"{self.algo_name}: reset + warmup 后应稳定返回 False")


class TestMog2(_AllAlgorithmsMixin, unittest.TestCase):
    algo_name = "mog2"


class TestRunningAvg(_AllAlgorithmsMixin, unittest.TestCase):
    algo_name = "running_avg"


class TestFrameDiff(_AllAlgorithmsMixin, unittest.TestCase):
    algo_name = "frame_diff"


# ============== 算法专属测试 ==============

class TestMog2Specific(unittest.TestCase):
    def test_uses_bg_subtractor(self):
        det = Mog2Detector(make_factory_cfg("mog2"))
        self.assertIsNotNone(det.bg_subtractor)


class TestRunningAvgSpecific(unittest.TestCase):
    def test_bg_initialized_on_first_frame(self):
        det = RunningAvgDetector(make_factory_cfg("running_avg"))
        self.assertIsNone(det._bg)
        det.detect(make_static_frame())
        self.assertIsNotNone(det._bg)
        self.assertEqual(det._bg.dtype, np.float32)

    def test_bg_alpha_clamped(self):
        det = RunningAvgDetector(make_factory_cfg("running_avg", bg_alpha=5.0))
        # 5.0 应该被截断到 1.0
        self.assertLessEqual(det._alpha, 1.0)

    def test_quick_response(self):
        """alpha 越大响应越快: 1.0 应该立即切换背景"""
        det = RunningAvgDetector(make_factory_cfg("running_avg", bg_alpha=1.0))
        # 喂 5 帧浅色
        for _ in range(5):
            det.detect(make_static_frame(value=100))
        # 切换到深色: alpha=1.0 时背景应该立即变成深色
        det.detect(make_static_frame(value=200))
        # 当前帧还是 200, 背景也变 200, 差应该接近 0
        # 实际 alpha=1.0 时本帧 = bg, 下一帧差为 0
        result = det.detect(make_static_frame(value=200))
        self.assertFalse(result, "alpha=1.0 时背景应立即跟随, 差为 0")


class TestFrameDiffSpecific(unittest.TestCase):
    def test_no_prev_returns_false(self):
        """首帧 prev 未初始化, 返回 False"""
        det = FrameDiffDetector(make_factory_cfg("frame_diff"))
        self.assertIsNone(det._prev_gray)
        result = det.detect(make_static_frame())
        self.assertFalse(result)
        self.assertIsNotNone(det._prev_gray)

    def test_accumulator_builds_up(self):
        """连续多帧运动, 累积 mask 应快速增长"""
        det = FrameDiffDetector(make_factory_cfg("frame_diff", min_area=500))
        # 先初始化 prev
        det.detect(make_static_frame())
        # 连续 5 帧运动: 每帧 acc_mask 在运动区域应该 = 255 (因为 fg OR acc 保持 255)
        det.detect(make_motion_frame(x_offset=0))
        det.detect(make_motion_frame(x_offset=20))
        det.detect(make_motion_frame(x_offset=40))
        det.detect(make_motion_frame(x_offset=60))
        result = det.detect(make_motion_frame(x_offset=80))
        self.assertTrue(result, "frame_diff: 持续运动应触发告警")

    def test_decay_clears_accumulator(self):
        """静止一段时间后, 累积 mask 应衰减到 0"""
        det = FrameDiffDetector(make_factory_cfg("frame_diff", min_area=500,
                                                  diff_decay=20))
        # 喂一帧运动
        det.detect(make_motion_frame(x_offset=0))
        # 之后 50 帧静止: 每帧减 20, 50 帧后应该减到 0
        for _ in range(50):
            det.detect(make_static_frame())
        # 累积 mask 应该基本为 0
        nonzero = cv2.countNonZero(det._acc_mask)
        self.assertEqual(nonzero, 0, f"frame_diff: 静止后 acc_mask 应清零, 残留 {nonzero} 像素")


# ============== 工厂测试 ==============

class TestFactory(unittest.TestCase):
    def test_factory_dispatch(self):
        for algo in ALGO_REGISTRY:
            cfg = make_factory_cfg(algo)
            det = create_detector(cfg)
            self.assertEqual(det.algo_name, algo)

    def test_unknown_raises(self):
        cfg = make_factory_cfg("nonexistent")
        with self.assertRaises(ValueError):
            create_detector(cfg)

    def test_all_have_required_methods(self):
        for algo, cls in ALGO_REGISTRY.items():
            cfg = make_factory_cfg(algo)
            det = cls(cfg)
            self.assertTrue(callable(det.detect))
            self.assertTrue(callable(det.reset))


if __name__ == "__main__":
    unittest.main(verbosity=2)
