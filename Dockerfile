# ---------- 运行时镜像 ----------
FROM python:3.11-slim

# OpenCV + FFmpeg 运行时依赖
# - libsm6/libxext6/libxrender1: OpenCV 图像编解码
# - ffmpeg: RTSP/HTTP 拉流后端 (cv2.CAP_FFMPEG)
# - libgl1: 部分 OpenCV 图像处理需要
# - tzdata: 时区支持 (TZ 环境变量)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# 时区 (默认 Asia/Shanghai, 可在 docker-compose.yml 覆盖)
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# 先装 Python 依赖 (利用 Docker 缓存: 源码变更不重装依赖)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY run.py .

# 默认配置 (用户挂载自己的 config.yaml 时会被覆盖)
COPY config.example.yaml /app/config.example.yaml

# 创建日志目录
RUN mkdir -p /app/logs

# 健康检查: 进程在 + config 存在
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD pgrep -f "run.py -c /app/config.yaml" > /dev/null || exit 1

# 启动
CMD ["python", "-u", "run.py", "-c", "/app/config.yaml"]
