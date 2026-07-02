"""配置加载：YAML 解析 + 校验 + 默认值"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class DetectorConfig:
    # 算法选择: "mog2" / "running_avg" / "frame_diff"
    algorithm: str = "mog2"

    # 通用
    downsample: float = 0.4
    frame_skip: int = 2
    min_area: int = 1500
    trigger_frames: int = 3
    cooldown_sec: float = 5.0
    warmup_sec: float = 5.0  # 启动后前 N 秒不告警 (让背景模型稳定)
    roi: Optional[List[Tuple[int, int]]] = None  # 多边形，None=全屏

    # mog2 参数
    history: int = 200
    var_threshold: int = 32

    # running_avg / frame_diff 参数
    diff_threshold: int = 25  # 像素差阈值 (灰度)

    # running_avg 专属
    bg_alpha: float = 0.05  # 背景学习率 (越小越稳定, 越大越敏感)

    # frame_diff 专属
    diff_decay: int = 5  # 累积 mask 每帧衰减量 (255/该值 = 衰减到 0 的帧数)


@dataclass
class ChannelConfig:
    id: str
    name: str
    source: str
    enabled: bool = True
    detector: DetectorConfig = field(default_factory=DetectorConfig)


@dataclass
class NotifierConfig:
    url: str = ""
    timeout: float = 3.0
    retry: int = 2
    headers: dict = field(default_factory=dict)


@dataclass
class SupervisorConfig:
    workers: int = 2
    log_level: str = "INFO"
    log_file: str = ""


@dataclass
class AppConfig:
    supervisor: SupervisorConfig
    notifier: NotifierConfig
    channels: List[ChannelConfig]


def _parse_roi(raw) -> Optional[List[Tuple[int, int]]]:
    if not raw:
        return None
    return [(int(p[0]), int(p[1])) for p in raw]


def load_config(path: str) -> AppConfig:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sup_raw = raw.get("supervisor", {}) or {}
    # 默认 2 个 worker, 适配 4G 边缘设备 (OpenCV 单 worker 约 560MB 物理内存)
    # 8G+ 设备可调到 4, 4G 设备推荐 2
    supervisor = SupervisorConfig(
        workers=int(sup_raw.get("workers", 2)),
        log_level=str(sup_raw.get("log_level", "INFO")),
        log_file=str(sup_raw.get("log_file", "")),
    )

    notif_raw = raw.get("notifier", {}).get("http", {}) or {}
    notifier = NotifierConfig(
        url=str(notif_raw.get("url", "")),
        timeout=float(notif_raw.get("timeout", 3)),
        retry=int(notif_raw.get("retry", 2)),
        headers=dict(notif_raw.get("headers", {}) or {}),
    )

    channels: List[ChannelConfig] = []
    for ch_raw in raw.get("channels", []) or []:
        if not ch_raw.get("enabled", True):
            continue
        det_raw = ch_raw.get("detector", {}) or {}
        detector = DetectorConfig(
            algorithm=str(det_raw.get("algorithm", "mog2")),
            downsample=float(det_raw.get("downsample", 0.4)),
            frame_skip=int(det_raw.get("frame_skip", 2)),
            min_area=int(det_raw.get("min_area", 1500)),
            trigger_frames=int(det_raw.get("trigger_frames", 3)),
            cooldown_sec=float(det_raw.get("cooldown_sec", 5.0)),
            warmup_sec=float(det_raw.get("warmup_sec", 5.0)),
            history=int(det_raw.get("history", 200)),
            var_threshold=int(det_raw.get("var_threshold", 32)),
            diff_threshold=int(det_raw.get("diff_threshold", 25)),
            bg_alpha=float(det_raw.get("bg_alpha", 0.05)),
            diff_decay=int(det_raw.get("diff_decay", 5)),
            roi=_parse_roi(det_raw.get("roi")),
        )
        channels.append(
            ChannelConfig(
                id=str(ch_raw["id"]),
                name=str(ch_raw.get("name", ch_raw["id"])),
                source=str(ch_raw["source"]),
                enabled=True,
                detector=detector,
            )
        )

    if not channels:
        raise ValueError("config.channels is empty (or all disabled)")

    return AppConfig(supervisor=supervisor, notifier=notifier, channels=channels)
