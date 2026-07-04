"""ROI 遮罩测试"""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.roi import RoiMask


class TestRoiMask(unittest.TestCase):

    def test_full_roi_passthrough(self):
        roi = RoiMask(polygon=None)
        self.assertTrue(roi.is_full)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
        out = roi.apply(frame)
        np.testing.assert_array_equal(out, frame)

    def test_polygon_masks_outside(self):
        roi = RoiMask(polygon=[(10, 10), (50, 10), (50, 50), (10, 50)])
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        out = roi.apply(frame)
        # ROI 内 (30, 30) 应保持 200
        self.assertEqual(out[30, 30, 0], 200)
        # ROI 外 (5, 5) 应为 0
        self.assertEqual(out[5, 5, 0], 0)
        self.assertEqual(out[5, 5, 1], 0)
        self.assertEqual(out[5, 5, 2], 0)
        # ROI 外 (80, 80) 应为 0
        self.assertEqual(out[80, 80, 0], 0)

    def test_mask_caches_by_shape(self):
        roi = RoiMask(polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        roi.apply(frame)
        first_mask_id = id(roi._mask)
        # 同样尺寸应复用 mask
        roi.apply(frame)
        self.assertEqual(id(roi._mask), first_mask_id)
        # 改变尺寸应重新生成
        frame2 = np.zeros((30, 30, 3), dtype=np.uint8)
        roi.apply(frame2)
        self.assertNotEqual(id(roi._mask), first_mask_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
