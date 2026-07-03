"""多进程 supervisor：N 个 worker 进程分摊多路通道

进程模型：
    - 主进程: 启停 worker、信号处理、监控健康
    - N 个 worker 进程 (N = 物理核数): 每进程管 (channels/N) 路
    - 每个 worker 进程内独立 HttpNotifier (HTTP POST 在子进程里做)
    - 这样设计最简单, 也避免了跨进程 queue 的开销
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import time
from typing import List, Optional

from .config import AppConfig, ChannelConfig
from .worker import ChannelWorker


logger = logging.getLogger("car-sense.supervisor")


# --------- worker 子进程入口 ---------

def _worker_entry(channels: List[ChannelConfig],
                  notifier_cfg_dict: dict,
                  stop_event: mp.Event,
                  log_level: str) -> None:
    """worker 进程入口: 在子进程里跑一组通道 + 独立 HTTP notifier"""
    import cv2
    # 多 worker 进程模型下, 限制 OpenCV 内部线程数避免与进程调度抢 CPU
    # 4 worker 进程时, 每个 worker 已被 OS 限制到 1 核, OpenCV 多线程无意义且有害
    cv2.setNumThreads(1)

    from .logger import setup_logger
    setup_logger("car-sense", log_level)

    from .config import NotifierConfig
    from .notifier import HttpNotifier

    notifier_cfg = NotifierConfig(**notifier_cfg_dict)
    notifier = HttpNotifier(notifier_cfg)
    notifier.start()

    workers: List[ChannelWorker] = []
    for ch in channels:
        w = ChannelWorker(ch, notifier)
        w.start()
        workers.append(w)

    try:
        # 子进程阻塞直到 stop_event
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        for w in workers:
            w.stop()
        notifier.stop()


# --------- 主进程 supervisor ---------

class Supervisor:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._stop_event = mp.Event()
        self._procs: List[mp.Process] = []
        self._stopping = False

    def start(self) -> None:
        self._setup_signals()
        self._spawn_workers()
        n_ch = len(self.cfg.channels)
        logger.info("supervisor started: %d workers, %d channels",
                    len(self._procs), n_ch)
        try:
            self._wait()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        logger.info("shutting down...")
        self._stop_event.set()
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                logger.warning("worker %s not exit, terminating", p.name)
                p.terminate()
                p.join(timeout=2)
        logger.info("supervisor stopped")

    # ----- 内部 -----

    def _spawn_workers(self) -> None:
        n_workers = max(1, self.cfg.supervisor.workers)
        chunks = self._split_channels(n_workers, self.cfg.channels)
        notifier_dict = {
            "url": self.cfg.notifier.url,
            "timeout": self.cfg.notifier.timeout,
            "retry": self.cfg.notifier.retry,
            "headers": self.cfg.notifier.headers,
        }
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            p = mp.Process(
                target=_worker_entry,
                args=(chunk, notifier_dict, self._stop_event,
                      self.cfg.supervisor.log_level),
                name=f"car-worker-{i}",
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            logger.info("spawned worker %d (pid=%d): channels=%s",
                        i, p.pid, [c.id for c in chunk])

    @staticmethod
    def _split_channels(n: int, channels: List[ChannelConfig]) -> List[List[ChannelConfig]]:
        """尽量均匀地把通道分到 N 个 worker (按顺序轮询分配)"""
        chunks: List[List[ChannelConfig]] = [[] for _ in range(n)]
        for i, ch in enumerate(channels):
            chunks[i % n].append(ch)
        return chunks

    def _wait(self) -> None:
        # 主进程阻塞, 同时做健康检查
        check_interval = 10
        last_check = 0.0
        while not self._stop_event.is_set():
            time.sleep(0.5)
            now = time.time()
            if now - last_check < check_interval:
                continue
            last_check = now
            for i, p in enumerate(self._procs):
                if not p.is_alive() and not self._stop_event.is_set():
                    logger.warning("worker %s (pid=%d) exited unexpectedly (code=%s)",
                                   p.name, p.pid, p.exitcode)

    def _setup_signals(self) -> None:
        def _handler(signum, frame):
            logger.info("received signal %d, shutting down", signum)
            self._stop_event.set()
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
