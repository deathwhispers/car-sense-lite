"""单路处理 worker：拉流 → 降采样 → 跳帧 → 检测 → 去抖 → 告警

去抖状态机：
    counter:       连续命中次数
    cooldown_until: 下次可告警时间戳
    - 有车: counter += 1
    - 无车: counter = max(0, counter-1)  (缓慢衰减, 防止抖动)
    - counter >= trigger_frames 且不在冷却中: 触发告警, 进入冷却
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import ChannelConfig
from .detector import create_detector
from .notifier import AlertEvent, HttpNotifier
from .stream import StreamReader


logger = logging.getLogger("car-sense.worker")


class ChannelWorker:
    """单路通道 worker，运行在自己的线程里 (主进程内)

    一路一线程即可，因为 OpenCV 的解码和图像处理都是 C 代码
    会释放 GIL，线程间能真正并行。20 路 < CPU 核数 也不会成瓶颈。
    """

    def __init__(self, ch: ChannelConfig, notifier: HttpNotifier,
                 stats_interval: float = 30.0):
        self.ch = ch
        self.notifier = notifier
        self.detector = create_detector(ch.detector)
        self.stream = StreamReader(ch.source, ch.id)
        self.stats_interval = stats_interval
        # 预分配 resize 输出 buffer (省每帧 ~0.3ms numpy 分配)
        # 真实形状在拿到首帧后确定
        self._resize_dst: Optional[np.ndarray] = None
        self._resize_shape: Optional[Tuple[int, int]] = None  # (h, w)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ts: float = 0.0  # worker 实际启动时间, 用于 warmup

        # 去抖状态
        self._counter = 0
        self._cooldown_until = 0.0
        # 性能统计
        self._frame_seq = 0
        self._last_stats_ts = time.time()
        self._frames_processed = 0
        self._alerts_sent = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"worker-{self.ch.id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.stream.stop()

    def _run(self) -> None:
        self.stream.start()
        self._start_ts = time.time()
        logger.info("[%s] worker started, source=%s", self.ch.id,
                    self._sanitize_source(self.ch.source))
        if self.ch.detector.warmup_sec > 0:
            logger.info("[%s] warmup: %.1fs (MOG2 learning background)",
                        self.ch.id, self.ch.detector.warmup_sec)

        # 等待首次连接
        if not self.stream._connected.wait(timeout=10):
            logger.warning("[%s] stream not connected at start, will retry in loop",
                           self.ch.id)

        # 预计算跳帧模数
        frame_skip = self.ch.detector.frame_skip
        skip_mod = frame_skip if frame_skip > 1 else 0

        while not self._stop_event.is_set():
            frame = self.stream.read()
            if frame is None:
                time.sleep(0.005)
                continue

            self._frame_seq += 1
            # 跳帧 - 使用位运算优化（如果frame_skip是2的幂）
            if skip_mod > 0:
                if skip_mod & (skip_mod - 1) == 0:  # 2的幂
                    if self._frame_seq & (skip_mod - 1) != 0:
                        continue
                else:
                    if self._frame_seq % skip_mod != 0:
                        continue

            # 降采样 (复用预分配 buffer, 省 numpy 分配开销)
            if self.ch.detector.downsample < 1.0:
                h, w = frame.shape[:2]
                ds = self.ch.detector.downsample
                target_h = int(h * ds)
                target_w = int(w * ds)
                if (self._resize_shape != (target_h, target_w)
                        or self._resize_dst is None):
                    self._resize_dst = np.empty(
                        (target_h, target_w, 3), dtype=frame.dtype
                    )
                    self._resize_shape = (target_h, target_w)
                cv2.resize(frame, (target_w, target_h),
                           dst=self._resize_dst,
                           fx=ds, fy=ds,
                           interpolation=cv2.INTER_AREA)
                small = self._resize_dst
            else:
                small = frame

            # 检测
            result = self.detector.detect(small)
            self._on_detection(result.has_car, result.max_area)
            self._frames_processed += 1
            self._maybe_log_stats()

        logger.info("[%s] worker stopped, processed=%d alerts=%d",
                    self.ch.id, self._frames_processed, self._alerts_sent)

    def _on_detection(self, has_car: bool, max_area: int) -> None:
        # warmup 期内只让检测器跑, 不告警 (等 MOG2 学到稳定背景)
        if self._start_ts > 0 and \
                time.time() - self._start_ts < self.ch.detector.warmup_sec:
            if has_car:
                self._counter = 0  # warmup 期不累积
            return

        if has_car:
            self._counter += 1
        else:
            if self._counter > 0:
                self._counter -= 1
            return

        if self._counter < self.ch.detector.trigger_frames:
            return

        now = time.time()
        if now < self._cooldown_until:
            return  # 冷却中

        # 触发告警
        event = AlertEvent(
            channel_id=self.ch.id,
            channel_name=self.ch.name,
            frame_seq=self._frame_seq,
            max_area=max_area,
        )
        if self.notifier.put_event(event):
            self._alerts_sent += 1
            logger.info("[%s] ALERT seq=%d counter=%d", self.ch.id,
                        self._frame_seq, self._counter)
        self._counter = 0
        self._cooldown_until = now + self.ch.detector.cooldown_sec

    def _maybe_log_stats(self) -> None:
        now = time.time()
        if now - self._last_stats_ts < self.stats_interval:
            return
        elapsed = now - self._last_stats_ts
        fps = self._frames_processed / elapsed if elapsed > 0 else 0
        logger.info("[%s] stats: %.1f fps processed, %d alerts, counter=%d",
                    self.ch.id, fps, self._alerts_sent, self._counter)
        self._frames_processed = 0
        self._last_stats_ts = now

    @staticmethod
    def _sanitize_source(s: str) -> str:
        if "@" in s:
            scheme, _, tail = s.partition("://")
            _, _, host_part = tail.rpartition("@")
            return f"{scheme}://***@{host_part}"
        return s
