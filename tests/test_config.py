"""配置加载测试"""
from __future__ import annotations

import os
import tempfile
import unittest

import yaml

from src.config import load_config


VALID_CONFIG = {
    "supervisor": {
        "workers": 2,
        "log_level": "DEBUG",
    },
    "notifier": {
        "http": {
            "url": "http://example.com/webhook",
            "timeout": 5,
            "retry": 3,
            "headers": {"X-Token": "abc"},
        }
    },
    "channels": [
        {
            "id": "ch1",
            "name": "Channel 1",
            "source": "rtsp://192.168.1.1:554/stream",
            "enabled": True,
            "detector": {
                "min_area": 1000,
                "trigger_frames": 5,
                "cooldown_sec": 10,
                "roi": [[0, 0], [100, 0], [100, 100], [0, 100]],
            },
        },
        {
            "id": "ch2",
            "name": "Channel 2 (disabled)",
            "source": "rtsp://192.168.1.2:554/stream",
            "enabled": False,
        },
    ],
}


class TestConfigLoading(unittest.TestCase):

    def test_load_valid_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(VALID_CONFIG, f)
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg.supervisor.workers, 2)  # 显式配置覆盖默认
            self.assertEqual(cfg.supervisor.log_level, "DEBUG")
            self.assertEqual(cfg.notifier.url, "http://example.com/webhook")
            self.assertEqual(cfg.notifier.timeout, 5.0)
            self.assertEqual(cfg.notifier.retry, 3)
            self.assertEqual(cfg.notifier.headers.get("X-Token"), "abc")
            # disabled channel 应被过滤
            self.assertEqual(len(cfg.channels), 1)
            ch = cfg.channels[0]
            self.assertEqual(ch.id, "ch1")
            self.assertEqual(ch.detector.min_area, 1000)
            self.assertEqual(ch.detector.trigger_frames, 5)
            self.assertEqual(ch.detector.cooldown_sec, 10.0)
            self.assertEqual(len(ch.detector.roi), 4)
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_empty_channels_raises(self):
        cfg = {"supervisor": {}, "notifier": {}, "channels": []}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_config(path)
        finally:
            os.unlink(path)

    def test_defaults_applied(self):
        cfg = {
            "supervisor": {},
            "notifier": {},
            "channels": [
                {"id": "ch1", "name": "x", "source": "rtsp://x", "enabled": True}
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            path = f.name
        try:
            loaded = load_config(path)
            # 默认值检查 (4G 边缘设备友好: 默认 2 worker)
            self.assertEqual(loaded.supervisor.workers, 2)
            self.assertEqual(loaded.notifier.timeout, 3.0)
            ch = loaded.channels[0]
            self.assertEqual(ch.detector.min_area, 1500)
            self.assertEqual(ch.detector.downsample, 0.4)
            self.assertEqual(ch.detector.warmup_sec, 5.0)
            self.assertIsNone(ch.detector.roi)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
