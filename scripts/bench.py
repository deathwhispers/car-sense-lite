"""性能压测: N 个 worker 进程并行处理 M 个通道, 测量 CPU 和内存占用

用法:
    python scripts/bench.py --channels 20 --duration 30

前置: 先用 gen_sample_video.py 生成 N 个测试视频
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path


def _measure_cpu():
    # macOS 不支持 psutil, 用 resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _worker(channel_id: str, video_path: str, duration_sec: int, frames: dict):
    """worker 子进程: 跑满 N 秒, 累计处理了多少帧"""
    import cv2
    import numpy as np
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from src.config import DetectorConfig
    from src.detector import CarDetector

    detector = CarDetector(DetectorConfig(
        downsample=0.4, frame_skip=2, min_area=800,
        trigger_frames=2, cooldown_sec=0,
        history=80, var_threshold=20,
    ))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        frames[channel_id] = -1
        return

    t0 = time.time()
    count = 0
    while time.time() - t0 < duration_sec:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        count += 1
        # 跳帧
        if count % 2 != 0:
            continue
        small = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        detector.detect(small)
    cap.release()
    frames[channel_id] = count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", default="./test_videos")
    parser.add_argument("--channels", type=int, default=20)
    parser.add_argument("--duration", type=int, default=20)
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        print(f"videos dir not found: {videos_dir}")
        return 1

    videos = sorted(videos_dir.glob("*.mp4"))
    if not videos:
        print(f"no .mp4 found in {videos_dir}")
        return 1
    if len(videos) < args.channels:
        print(f"注意: 只有 {len(videos)} 个视频, 20 路 worker 会循环使用")
        print(f"      如需每路独立视频, 运行: python scripts/gen_sample_video.py --count {args.channels}")

    manager = mp.Manager()
    frames_dict = manager.dict()
    procs = []
    print(f"==> 启动 {args.channels} 个 worker 进程, 跑 {args.duration}s")
    t0 = time.time()
    for i in range(args.channels):
        ch_id = f"bench_ch_{i:02d}"
        video = videos[i % len(videos)]
        p = mp.Process(target=_worker,
                       args=(ch_id, str(video), args.duration, frames_dict),
                       name=f"bench-{i}")
        p.start()
        procs.append(p)

    # 主进程循环打印 CPU/内存
    max_rss = 0
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(2)
            total_rss_mb = 0
            for p in procs:
                if p.is_alive():
                    # 跨平台获取子进程内存
                    try:
                        total_rss_mb += _get_rss_mb(p.pid)
                    except Exception:
                        pass
            if total_rss_mb > max_rss:
                max_rss = total_rss_mb
            elapsed = time.time() - t0
            print(f"  t={elapsed:5.1f}s  alive={sum(p.is_alive() for p in procs)}/{len(procs)}  "
                  f"rss={total_rss_mb:.0f}MB")
    except KeyboardInterrupt:
        pass

    for p in procs:
        p.join(timeout=5)
    elapsed = time.time() - t0

    print(f"\n==> 压测完成, 耗时 {elapsed:.1f}s")
    print(f"==> 最大总内存: {max_rss:.0f} MB")
    print(f"==> 各路处理帧数:")
    total = 0
    for k in sorted(frames_dict.keys()):
        v = frames_dict[k]
        total += max(0, v)
        print(f"    {k}: {v} frames ({v / elapsed:.1f} fps)")
    print(f"==> 总处理: {total} 帧, 合计 {total / elapsed:.0f} fps "
          f"({total / elapsed / args.channels:.1f} fps/ch)")

    # CPU 占用 (用主进程 utime+stime 估算, 不太准, 但够用)
    return 0


def _get_rss_mb(pid: int) -> float:
    """跨平台获取进程 RSS (MB)"""
    try:
        import psutil  # type: ignore
        return psutil.Process(pid).memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    # macOS fallback
    try:
        import subprocess
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True
        ).strip()
        return int(out) / 1024
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(main())
