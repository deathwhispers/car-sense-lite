"""生成测试视频: 模拟俯视车道, 一辆'车'从左到右通过

用于:
    1. 本地压测 (用文件循环代替 RTSP)
    2. 离线回归测试

用法:
    python scripts/gen_sample_video.py --output ./test_videos --count 5
    python scripts/gen_sample_video.py --output ./test_videos --count 1 --duration 30 --cars 3
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def gen_one(output_dir: Path, idx: int, width: int = 1920, height: int = 1080,
            fps: int = 15, duration_sec: int = 15, num_cars: int = 2) -> Path:
    """生成一个测试视频, 含 num_cars 辆车通过"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"test_{idx:02d}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter: {output_path}")

    total_frames = duration_sec * fps
    print(f"[{idx}] generating {output_path} "
          f"({width}x{height} @ {fps}fps, {duration_sec}s, {num_cars} cars)")

    # 预先生成 num_cars 辆车的进入时间 + 速度
    car_schedules = []
    for _ in range(num_cars):
        enter_at = random.randint(int(fps * 0.5), total_frames - int(fps * 5))
        speed = random.uniform(8, 25)  # 像素/帧
        car_w = random.randint(80, 140)
        car_h = random.randint(60, 100)
        lane_y = random.randint(height // 3, height * 2 // 3)
        car_schedules.append((enter_at, speed, car_w, car_h, lane_y))

    bg = np.random.randint(80, 100, (height, width, 3), dtype=np.uint8)
    # 画一条横线模拟车道
    cv2.line(bg, (0, height // 2 + 50), (width, height // 2 + 50),
             (60, 60, 60), 2)

    for frame_idx in range(total_frames):
        frame = bg.copy()
        # 加噪声模拟摄像头噪声
        noise = np.random.normal(0, 2, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        for enter_at, speed, car_w, car_h, lane_y in car_schedules:
            x = (frame_idx - enter_at) * speed
            if -car_w < x < width:
                x1 = int(x)
                y1 = int(lane_y - car_h // 2)
                x2 = x1 + car_w
                y2 = y1 + car_h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 2)

        writer.write(frame)

    writer.release()
    print(f"[{idx}] done: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成测试视频")
    parser.add_argument("--output", default="./test_videos", help="输出目录")
    parser.add_argument("--count", type=int, default=5, help="生成视频数量")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration", type=int, default=15, help="单视频时长(秒)")
    parser.add_argument("--cars", type=int, default=2, help="单视频中车辆数")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    for i in range(1, args.count + 1):
        gen_one(output_dir, i, args.width, args.height, args.fps,
                args.duration, args.cars)
    print(f"\nall done. {args.count} videos in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
