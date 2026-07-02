#!/usr/bin/env bash
# 安装为 systemd 服务
# 用法: sudo ./scripts/install_systemd.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$SCRIPT_DIR/car-sense-lite.service"
INSTALL_DIR="/opt/car-sense-lite"

if [[ $EUID -ne 0 ]]; then
    echo "this script requires root. run with sudo." >&2
    exit 1
fi

echo "==> 安装目录: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

echo "==> 复制项目文件"
rsync -a --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='test_videos' \
    --exclude='config.yaml' \
    --exclude='*.log' \
    "$PROJECT_DIR/" "$INSTALL_DIR/"

if [[ ! -f "$INSTALL_DIR/config.yaml" ]]; then
    echo "==> 创建 config.yaml (从模板)"
    cp "$INSTALL_DIR/config.example.yaml" "$INSTALL_DIR/config.yaml"
    echo "    请编辑 $INSTALL_DIR/config.yaml 填入 RTSP 地址"
fi

echo "==> 安装 Python 依赖"
cd "$INSTALL_DIR"
python3 -m pip install -r requirements.txt

echo "==> 安装 systemd service"
cp "$SERVICE_FILE" /etc/systemd/system/car-sense-lite.service
systemctl daemon-reload
systemctl enable car-sense-lite.service
echo "    启动:  systemctl start car-sense-lite"
echo "    日志:  journalctl -u car-sense-lite -f"
echo "    状态:  systemctl status car-sense-lite"
