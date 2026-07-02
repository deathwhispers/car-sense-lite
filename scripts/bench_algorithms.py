"""3 种算法性能对比: 测单帧耗时 + 内存占用

用法:
    python scripts/bench_algorithms.py
    python scripts/bench_algorithms.py --frames 200 --height 540 --width 960
    python scripts/bench_algorithms.py --realistic  # 模拟真实 720p/1080p 降采样
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Tuple

import cv2
import numpy as np


def _add_car(frame: np.ndarray, x_offset: int) -> np.ndarray:
    """合成一个移动的白色方块当车"""
    h, w = frame.shape[:2]
    car_size = 80
    cx, cy = w // 2 + x_offset, h // 2
    x1 = max(0, cx - car_size // 2)
    x2 = min(w, cx + car_size // 2)
    y1 = max(0, cy - car_size // 2)
    y2 = min(h, cy + car_size // 2)
    frame[y1:y2, x1:x2] = 200
    return frame


def _gen_video(height: int, width: int, n_frames: int) -> list:
    """合成一段测试视频: 灰底 + 一辆移动的车"""
    frames = []
    bg = np.random.randint(80, 100, (height, width, 3), dtype=np.uint8)
    for i in range(n_frames):
        f = bg.copy()
        noise = np.random.normal(0, 2, f.shape).astype(np.int16)
        f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if i > 10:  # 第 10 帧后开始有车
            f = _add_car(f, (i - 10) * 8)
        frames.append(f)
    return frames


def bench_algo(name: str, cfg, frames: list) -> Tuple[float, float, int, int]:
    """单算法基准测试"""
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from src.detector import create_detector

    gc.collect()
    det = create_detector(cfg)
    # warmup
    for f in frames[:5]:
        det.detect(f)

    # 测量
    t0 = time.perf_counter()
    detected_count = 0
    for f in frames:
        if det.detect(f):
            detected_count += 1
    elapsed = time.perf_counter() - t0
    per_frame_ms = elapsed / len(frames) * 1000
    fps = len(frames) / elapsed
    return per_frame_ms, fps, detected_count, id(det)


def _bench_with_downsample(name: str, cfg, src_h: int, src_w: int,
                           downsample: float, n_frames: int) -> Tuple[float, int]:
    """测单帧 resize+detect 总耗时 (更贴近真实使用)"""
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from src.detector import create_detector

    det = create_detector(cfg)
    frame = np.full((src_h, src_w, 3), 100, dtype=np.uint8)
    # warmup
    for _ in range(5):
        small = cv2.resize(frame, None, fx=downsample, fy=downsample,
                           interpolation=cv2.INTER_AREA)
        det.detect(small)
    # 测量
    t0 = time.perf_counter()
    for _ in range(n_frames):
        small = cv2.resize(frame, None, fx=downsample, fy=downsample,
                           interpolation=cv2.INTER_AREA)
        det.detect(small)
    total_ms = (time.perf_counter() - t0) / n_frames * 1000
    return total_ms, id(det)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--realistic", action="store_true",
                        help="跑多个真实场景 (720p/1080p + 不同 downsample)")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from src.config import DetectorConfig

    configs = {
        "mog2": DetectorConfig(
            algorithm="mog2", min_area=500, history=200, var_threshold=32,
            warmup_sec=0,
        ),
        "running_avg": DetectorConfig(
            algorithm="running_avg", min_area=500,
            diff_threshold=25, bg_alpha=0.05, warmup_sec=0,
        ),
        "frame_diff": DetectorConfig(
            algorithm="frame_diff", min_area=500,
            diff_threshold=25, diff_decay=5, warmup_sec=0,
        ),
    }

    if args.realistic:
        # 真实场景: 源分辨率 + 降采样
        scenarios = [
            ("720p 源 + 0.5x",  720, 1280, 0.5),
            ("720p 源 + 0.4x",  720, 1280, 0.4),
            ("1080p 源 + 0.4x", 1080, 1920, 0.4),
            ("1080p 源 + 0.3x", 1080, 1920, 0.3),
        ]
        print(f"\n{'场景':<22} {'算法':<14} {'M2 ms':<10} {'A55 ms':<10} {'A55 fps/核':<10} {'20路4核':<10}")
        print("-" * 80)
        for scene_name, h, w, ds in scenarios:
            for algo, cfg in configs.items():
                total_ms, _ = _bench_with_downsample(algo, cfg, h, w, ds, args.frames)
                a55_ms = total_ms * 3.5
                a55_fps = 1000 / a55_ms
                # 4 核 20 路, frame_skip=2 → 每路实际 7.5 fps 输入, 但 cap 在 7.5
                can_handle = 4 * a55_fps / 7.5
                verdict = "OK" if can_handle >= 25 else "tight" if can_handle >= 18 else "NG"
                print(f"{scene_name:<22} {algo:<14} {total_ms:>6.2f}    {a55_ms:>6.1f}     {a55_fps:>6.0f}      {can_handle:>4.0f}路 [{verdict}]")
            print()
        return 0

    # 单场景模式
    print(f"==> 合成测试视频: {args.width}x{args.height}, {args.frames} 帧")
    frames = _gen_video(args.height, args.width, args.frames)

    print(f"\n{'算法':<15} {'单帧耗时':<12} {'fps/单核':<10} {'检测命中':<10} {'相对mog2':<10}")
    print("-" * 60)
    base_ms = None
    for name, cfg in configs.items():
        ms, fps, hits, _ = bench_algo(name, cfg, frames)
        if base_ms is None:
            base_ms = ms
        speedup = base_ms / ms
        print(f"{name:<15} {ms:>8.2f} ms  {fps:>7.0f}    {hits:>5d}/{len(frames)}   {speedup:>5.2f}x")

    print(f"\n分辨率 {args.width}x{args.height} (约 {args.width*args.height/1e6:.1f} 百万像素)")
    print("A55 单核等效 (按 3.5x 慢缩放):")
    for name, cfg in configs.items():
        ms, fps, _, _ = bench_algo(name, cfg, frames)
        a55_ms = ms * 3.5
        a55_fps = 1000 / a55_ms
        print(f"  {name:<15} {a55_ms:>6.1f} ms/帧  ({a55_fps:>4.0f} fps/单核)")

    print("\n推荐配置 (20 路, 4 核 A55, frame_skip=2 → 7.5fps/路):")
    for name, cfg in configs.items():
        ms, _, _, _ = bench_algo(name, cfg, frames)
        a55_ms = ms * 3.5
        a55_fps = 1000 / a55_ms
        can_handle = 4 * a55_fps / 7.5
        verdict = "OK" if can_handle >= 20 else "tight" if can_handle >= 15 else "NG"
        print(f"  {name:<15} A55 4 核可扛 {can_handle:.0f} 路 [{verdict}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
