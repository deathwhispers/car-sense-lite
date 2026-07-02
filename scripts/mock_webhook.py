"""模拟告警接收端 (HTTP Webhook) - 用于本地调试

启动一个简单的 HTTP server, 接收 car-sense-lite 的 POST 请求并打印

用法:
    python scripts/mock_webhook.py --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock


_count_lock = Lock()
_count = 0
_start_ts = time.time()


class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global _count
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else ""
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {"_raw": body}

        with _count_lock:
            _count += 1
            n = _count

        ts = time.strftime("%H:%M:%S")
        channel = data.get("channel_id", "?")
        seq = data.get("frame_seq", "?")
        area = data.get("max_area", "?")
        print(f"[{ts}] #{n:04d} channel={channel} seq={seq} area={area} "
              f"body={json.dumps(data, ensure_ascii=False)}",
              flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, fmt, *args):
        # 屏蔽默认 access log, 避免刷屏
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), WebhookHandler)
    print(f"mock webhook listening on http://{args.host}:{args.port}")
    print("waiting for car-sense-lite alerts... (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        elapsed = time.time() - _start_ts
        with _count_lock:
            print(f"\nstopped. total received: {_count} in {elapsed:.1f}s")
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
