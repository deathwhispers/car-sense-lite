"""HTTP 告警：异步 + 重试 + 失败丢弃

告警事件:
    {
        "channel_id": str,
        "channel_name": str,
        "event": "car_detected",
        "timestamp": float,         # unix ts
        "iso_time": str,            # ISO8601
        "frame_seq": int,
        "max_area": int
    }
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from .config import NotifierConfig


logger = logging.getLogger("car-sense.notifier")


class AlertEvent:
    __slots__ = ("channel_id", "channel_name", "timestamp", "iso_time",
                 "frame_seq", "max_area")

    def __init__(self, channel_id: str, channel_name: str,
                 frame_seq: int, max_area: int):
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.timestamp = time.time()
        self.iso_time = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)\
            .astimezone().isoformat(timespec="milliseconds")
        self.frame_seq = frame_seq
        self.max_area = max_area

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "event": "car_detected",
            "timestamp": self.timestamp,
            "iso_time": self.iso_time,
            "frame_seq": self.frame_seq,
            "max_area": self.max_area,
        }


class HttpNotifier:
    """HTTP POST 告警器：内部一个发送线程，调用方 put_event 非阻塞"""
    QUEUE_MAX = 1024

    def __init__(self, cfg: NotifierConfig):
        self.cfg = cfg
        self._queue: "queue.Queue[AlertEvent]" = queue.Queue(maxsize=self.QUEUE_MAX)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0
        self._sent = 0
        self._failed = 0

    def start(self) -> None:
        if not self.cfg.url:
            logger.warning("notifier url is empty, alerts will be logged only")
            return
        self._thread = threading.Thread(
            target=self._run, name="notifier", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def put_event(self, event: AlertEvent) -> bool:
        """非阻塞提交事件，队列满则丢弃并计数"""
        if not self.cfg.url:
            logger.info("[ALERT] %s seq=%d area=%d", event.channel_id,
                        event.frame_seq, event.max_area)
            return True
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            self._dropped += 1
            if self._dropped % 50 == 1:
                logger.warning("notifier queue full, dropped %d events", self._dropped)
            return False

    def _run(self) -> None:
        session = requests.Session()
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._send_with_retry(session, event)

    def _send_with_retry(self, session: requests.Session, event: AlertEvent) -> None:
        payload = json.dumps(event.to_dict(), ensure_ascii=False)
        headers = {"Content-Type": "application/json", **self.cfg.headers}
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.retry + 1):
            try:
                resp = session.post(
                    self.cfg.url, data=payload, headers=headers,
                    timeout=self.cfg.timeout,
                )
                if 200 <= resp.status_code < 300:
                    self._sent += 1
                    return
                last_err = RuntimeError(f"http {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                last_err = e
            if attempt < self.cfg.retry:
                time.sleep(0.2 * (attempt + 1))
        self._failed += 1
        logger.error("[%s] notify failed after %d retries: %s",
                     event.channel_id, self.cfg.retry, last_err)
