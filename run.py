"""car-sense-lite 启动入口

用法:
    python run.py -c config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.logger import setup_logger
from src.supervisor import Supervisor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="car-sense-lite: 边缘设备来车检测器"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="仅校验配置和连通性, 不启动 worker"
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"[FATAL] failed to load config: {e}", file=sys.stderr)
        return 2

    logger = setup_logger("car-sense", cfg.supervisor.log_level, cfg.supervisor.log_file)
    logger.info("=" * 60)
    logger.info("car-sense-lite starting")
    logger.info("config: %s", Path(args.config).resolve())
    logger.info("workers: %d, channels: %d",
                cfg.supervisor.workers, len(cfg.channels))
    for ch in cfg.channels:
        logger.info("  - %s [%s] source=%s",
                    ch.id, ch.name, _sanitize(ch.source))
    logger.info("notifier: url=%s timeout=%.1fs retry=%d",
                cfg.notifier.url or "(log only)",
                cfg.notifier.timeout, cfg.notifier.retry)

    if args.check:
        logger.info("--check mode: config valid, exiting")
        return 0

    supervisor = Supervisor(cfg)
    try:
        supervisor.start()
    except KeyboardInterrupt:
        logger.info("interrupted by user")
    return 0


def _sanitize(s: str) -> str:
    if "@" in s:
        scheme, _, tail = s.partition("://")
        _, _, host_part = tail.rpartition("@")
        return f"{scheme}://***@{host_part}"
    return s


if __name__ == "__main__":
    sys.exit(main())
