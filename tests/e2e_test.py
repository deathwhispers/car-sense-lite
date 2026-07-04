"""端到端测试: 读本地视频 → worker 检测 → HTTP 告警 → mock webhook

启动 mock webhook → 启动 worker (消费本地视频循环) → 验证收到告警

用法:
    python tests/e2e_test.py                  # 默认 mog2
    python tests/e2e_test.py --algo frame_diff
    python tests/e2e_test.py --algo running_avg
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from src.config import ChannelConfig, DetectorConfig, NotifierConfig
from src.detector import create_detector
from src.notifier import HttpNotifier


RECEIVED_EVENTS: list = []
WEBHOOK_PORT = 18999


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n).decode("utf-8") if n else ""
        RECEIVED_EVENTS.append({"ts": time.time(), "body": body})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, *a, **kw):
        return


def start_webhook() -> HTTPServer:
    server = HTTPServer(("127.0.0.1", WEBHOOK_PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------- local file loop driver ----------

class FileLoopDriver:
    """直接调用 detector (不走 RTSP), 用本地文件循环测试"""

    def __init__(self, video_path: str, channel: ChannelConfig):
        self.video_path = video_path
        self.channel = channel
        self.detector = create_detector(channel.detector)
        self._counter = 0
        self._cooldown_until = 0.0
        self._frame_seq = 0
        self._alerts = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"file-loop-{channel.id}", daemon=True
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[FAIL] cannot open {self.video_path}")
            return
        print(f"[{self.channel.id}] algo={self.channel.detector.algorithm} "
              f"reading {self.video_path} ...")
        # 喂首帧让检测器初始化
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        small = cv2.resize(frame, None,
                           fx=self.channel.detector.downsample,
                           fy=self.channel.detector.downsample,
                           interpolation=cv2.INTER_AREA)
        self.detector.detect(small)  # 接受 Detection 返回值, 丢弃用于初始化背景
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self._frame_seq += 1
            if self._frame_seq % self.channel.detector.frame_skip != 0:
                continue
            small = cv2.resize(frame, None,
                               fx=self.channel.detector.downsample,
                               fy=self.channel.detector.downsample,
                               interpolation=cv2.INTER_AREA)
            has_car = self.detector.detect(small).has_car
            if has_car:
                self._counter += 1
            else:
                if self._counter > 0:
                    self._counter -= 1
                continue
            if self._counter < self.channel.detector.trigger_frames:
                continue
            now = time.time()
            if now < self._cooldown_until:
                continue
            from src.notifier import AlertEvent
            ev = AlertEvent(
                channel_id=self.channel.id,
                channel_name=self.channel.name,
                frame_seq=self._frame_seq,
                max_area=0,
            )
            RECEIVED_EVENTS.append({"ts": time.time(), "body": str(ev.to_dict())})
            self._alerts += 1
            print(f"[{self.channel.id}] ALERT seq={self._frame_seq} "
                  f"counter={self._counter} algo={self.channel.detector.algorithm}")
            self._counter = 0
            self._cooldown_until = now + self.channel.detector.cooldown_sec

        cap.release()
        print(f"[{self.channel.id}] done. total alerts: {self._alerts}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="mog2",
                        choices=["mog2", "running_avg", "frame_diff"])
    parser.add_argument("--duration", type=int, default=12)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    video_path = os.path.abspath("./test_videos/test_01.mp4")
    if not os.path.isfile(video_path):
        print(f"test video not found: {video_path}")
        print("run: python scripts/gen_sample_video.py --output ./test_videos --count 1")
        return 1

    RECEIVED_EVENTS.clear()

    print(f"==> 启动 mock webhook, 测试算法: {args.algo}")
    start_webhook()
    time.sleep(0.3)

    notifier = HttpNotifier(NotifierConfig(
        url=f"http://127.0.0.1:{WEBHOOK_PORT}/webhook",
        timeout=3, retry=2,
        headers={"X-Source": "e2e-test", "X-Algo": args.algo},
    ))
    notifier.start()

    ch = ChannelConfig(
        id=f"e2e_{args.algo}",
        name=f"e2e test ({args.algo})",
        source=video_path,
        enabled=True,
        detector=DetectorConfig(
            algorithm=args.algo,
            downsample=0.4,
            frame_skip=2,
            min_area=800,
            trigger_frames=2,
            cooldown_sec=2.0,
            warmup_sec=1.0,
            # 算法专属参数
            history=80, var_threshold=20,    # mog2
            diff_threshold=25, bg_alpha=0.05,  # running_avg
            diff_decay=5,                     # frame_diff
        ),
    )
    driver = FileLoopDriver(video_path, ch)
    driver.start()
    time.sleep(args.duration)
    driver.stop()
    notifier.stop()
    time.sleep(1)

    n_alerts = len(RECEIVED_EVENTS)
    print(f"\n==> 结果: webhook 收到 {n_alerts} 个告警 (算法: {args.algo})")
    if n_alerts == 0:
        print(f"[FAIL] 没收到告警")
        return 1
    print(f"[OK] 端到端测试通过 ({args.algo})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
