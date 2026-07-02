"""RTSP / 本地文件 拉流封装：自动重连、低缓冲、读取线程解耦
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger("car-sense.stream")


def _is_local_file(source: str) -> bool:
    if source.startswith("file://"):
        return True
    if "://" in source:
        return False
    return os.path.isfile(source)


class StreamReader:
    """单路视频源读取器，支持 RTSP/HTTP/本地文件：
    - FFMPEG 后端 + 低缓冲
    - 内部读线程，主线程调用 read() 拿最新帧
    - 断流自动重连
    - 本地文件读到 EOF 时自动循环
    """

    def __init__(self, source: str, channel_id: str, reconnect_sec: float = 2.0):
        self.source = source
        self.channel_id = channel_id
        self.reconnect_sec = reconnect_sec
        self._is_local = _is_local_file(source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"stream-{self.channel_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._release()

    def read(self) -> Optional[np.ndarray]:
        """获取最新一帧（无新帧时返回 None）"""
        with self._lock:
            return self._frame

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def _run(self) -> None:
        """后台读帧循环"""
        while not self._stop_event.is_set():
            try:
                self._open()
                self._connected.set()
                logger.info("[%s] stream connected: %s (%s)",
                            self.channel_id, self._source_label(),
                            "local file loop" if self._is_local else "rtsp/http")
                self._read_loop()
            except Exception as e:
                logger.warning("[%s] stream error: %s, reconnecting in %.1fs",
                               self.channel_id, e, self.reconnect_sec)
                self._connected.clear()
            finally:
                self._release()
            if not self._stop_event.is_set():
                self._stop_event.wait(self.reconnect_sec)

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._is_local:
                    # 本地文件 EOF: 循环回到开头
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                self._consecutive_failures += 1
                if self._consecutive_failures >= 30:
                    raise IOError("too many consecutive read failures")
                time.sleep(0.01)
                continue
            self._consecutive_failures = 0
            with self._lock:
                self._frame = frame

    def _open(self) -> None:
        if self._is_local:
            path = self.source[7:] if self.source.startswith("file://") else self.source
            cap = cv2.VideoCapture(path)
        else:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise IOError(f"failed to open stream: {self._source_label()}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._consecutive_failures = 0

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _source_label(self) -> str:
        s = self.source
        if "@" in s:
            scheme, _, tail = s.partition("://")
            _, _, host_part = tail.rpartition("@")
            return f"{scheme}://***@{host_part}"
        return s
