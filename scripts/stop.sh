#!/usr/bin/env bash
# 停止 car-sense-lite 进程
# 用法: ./scripts/stop.sh

set -e

PIDFILE="${PIDFILE:-/var/run/car-sense-lite.pid}"

if [[ -f "$PIDFILE" ]]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "stopping car-sense-lite (pid=$PID)..."
        kill -TERM "$PID"
        for i in {1..20}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "stopped."
                rm -f "$PIDFILE"
                exit 0
            fi
            sleep 0.5
        done
        echo "force killing..."
        kill -KILL "$PID" 2>/dev/null || true
        rm -f "$PIDFILE"
    else
        echo "pid $PID not running, cleaning pidfile"
        rm -f "$PIDFILE"
    fi
else
    # 回退: 用 pgrep 找
    PIDS=$(pgrep -f "python.*run\.py.*-c.*config\.yaml" || true)
    if [[ -n "$PIDS" ]]; then
        echo "killing pids: $PIDS"
        kill -TERM $PIDS
    else
        echo "no car-sense-lite process found"
    fi
fi
