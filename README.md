# car-sense-lite

> 边缘设备来车检测器 · 4 核 4G 稳定扛 20 路 RTSP · 纯 CPU 推理最快 0.7ms/帧

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20ARM64%20%7C%20x86_64-lightgrey)](#)

为俯视/斜俯视车道场景设计的轻量级运动检测系统。基于 OpenCV 传统算法，**无需 GPU、无需深度学习模型**，
单个 worker 进程仅占 ~250MB 物理内存 (Linux)，在 4 核 4G 边缘设备上稳定处理 20 路 720p 视频流。

```
┌─────────┐  RTSP   ┌──────────────┐  HTTP POST   ┌──────────┐
│ 摄像头  │ ──────→ │ car-sense    │ ──────────→ │ 业务服务  │
│  × 20   │  H.264  │   -lite      │   JSON      │  webhook │
└─────────┘         └──────────────┘             └──────────┘
   720p             3 种算法可选                  告警 + 截图
                    多 worker 并行
```

## 目录

- [为什么用传统算法](#为什么用传统算法)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [3 种算法对比](#3-种算法对比)
- [配置详解](#配置详解)
- [告警接口 (Webhook)](#告警接口-webhook)
- [部署到边缘设备](#部署到边缘设备)
  - [Docker Compose (推荐)](#方式-1-docker-compose-推荐-一键部署)
- [调参指南](#调参指南)
- [性能基准](#性能基准)
- [开发指南](#开发指南)
- [FAQ](#faq)
- [License](#license)

## 为什么用传统算法

| 维度 | MOG2 / 帧差法 (本项目) | YOLOv5n (深度学习) |
|------|------------------------|---------------------|
| 单帧 CPU 耗时 (480p) | **0.2 - 3 ms** | 60 - 100 ms |
| 20 路所需算力 | **1-2 核** | 8+ 核 (需 GPU) |
| 内存占用 | **< 1 GB** | 2-6 GB |
| "是否有车"准确率 | 85-95% | 95%+ |
| **俯视/斜俯视场景** | **✓ 完美适配** | 通用 |
| 模型依赖 | 无 | 100MB+ 模型文件 |
| 启动延迟 | 5s warmup | 10-30s 加载 |

**用户场景**：只需要"有/无" + 容忍误报 + 俯视/斜俯视 = 传统算法是更优解。

## 核心特性

- **3 种可切换算法**：`frame_diff` / `running_avg` / `mog2`，按场景选最优
- **极致性能**：A55 + 720p 源 + `frame_diff`，单核 1498 fps
- **算法可混搭**：每个通道独立配置算法，门口用 mog2 防树影，停车场用 frame_diff 极致性能
- **告警去抖三重防护**：`warmup_sec` (启动学习) + `trigger_frames` (连续命中) + `cooldown_sec` (冷却)
- **HTTP 告警 + 异步队列**：不阻塞检测，失败重试，队列削峰
- **多进程模型**：N worker × M 通道，4 核设备默认 2 worker
- **断流自动重连**：RTSP 异常时自动恢复
- **本地文件回放**：支持视频文件循环 (压测 / 调试)
- **小而美**：核心 ~1200 行，单镜像 < 500MB

## 快速开始

### 1. 系统依赖 (仅 Linux 边缘设备需要)

```bash
# Debian / Ubuntu
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg libsm6 libxext6
```

> ffmpeg 是 RTSP 解码后端，libsm6/libxext6 是 OpenCV 图像编解码依赖。
> Mac/Windows 桌面环境 OpenCV wheel 自带这些依赖，无需额外安装。

### 2. 克隆并安装

```bash
git clone https://github.com/yourname/car-sense-lite.git
cd car-sense-lite

# 推荐用 venv 隔离环境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 验证安装
python run.py --help
```

### 3. 配置通道

```bash
cp config.example.yaml config.yaml
vim config.yaml
```

最小化配置示例（只填 RTSP 地址和告警 URL）：

```yaml
supervisor:
  workers: 2              # 4G 设备推荐 2, 8G+ 可用 4

notifier:
  http:
    url: "http://192.168.1.100:8000/webhook/car"

channels:
  - id: "gate_1"
    name: "大门入口"
    source: "rtsp://admin:admin123@192.168.1.10:554/Streaming/Channels/102"
    # ^ 用 720p 辅码流 (如 sub), 性能最佳
    detector:
      algorithm: frame_diff
      downsample: 0.5
      frame_skip: 2
      min_area: 1500
```

### 4. 启动

**方式 A：本地直接运行**（开发调试推荐）

```bash
# 前台运行 (Ctrl+C 停止)
python run.py -c config.yaml

# 仅校验配置
python run.py -c config.yaml --check
```

**方式 B：Docker Compose**（生产部署推荐）

```bash
docker compose up -d --build
docker compose logs -f
```

**方式 C：systemd**（裸机部署）

```bash
sudo scripts/install_systemd.sh
sudo systemctl start car-sense-lite
```

**预期输出**：

```
2026-07-02 14:00:00 [INFO] [MainProcess] car-sense: ============================================================
2026-07-02 14:00:00 [INFO] [MainProcess] car-sense: car-sense-lite starting
2026-07-02 14:00:00 [INFO] [MainProcess] car-sense: config: /path/to/config.yaml
2026-07-02 14:00:00 [INFO] [MainProcess] car-sense: workers: 2, channels: 20
2026-07-02 14:00:00 [INFO] [MainProcess] car-sense:   - gate_1 [大门入口] source=rtsp://***@192.168.1.10:554/...
2026-07-02 14:00:00 [INFO] [MainProcess] car-sense.supervisor: spawned worker 0 (pid=1234): channels=['gate_1', ...]
2026-07-02 14:00:00 [INFO] [MainProcess] car-sense.supervisor: spawned worker 1 (pid=1235): channels=['...', ...]
2026-07-02 14:00:00 [INFO] [car-worker-0] car-sense.worker: [gate_1] worker started
2026-07-02 14:00:00 [INFO] [car-worker-0] car-sense.worker: [gate_1] warmup: 5.0s (MOG2 learning background)
2026-07-02 14:00:00 [INFO] [car-worker-0] car-sense.stream: [gate_1] stream connected: rtsp://***@...
2026-07-02 14:00:08 [INFO] [car-worker-0] car-sense.worker: [gate_1] ALERT seq=576 counter=3
```

**5 秒后**（warmup 完成）开始正常告警。

### 5. 测试告警链路

启动一个 mock webhook 接收告警：

```bash
# 终端 1: 启动 mock webhook
python scripts/mock_webhook.py --port 8000

# 终端 2: 启动 car-sense (notifier.url 改成 http://127.0.0.1:8000/webhook)
python run.py -c config.yaml
```

mock webhook 会打印收到的每个告警 JSON。

### 6. (可选) 端到端测试

不需要真实摄像头，用合成视频验证 pipeline：

```bash
# 生成 1 个测试视频
python scripts/gen_sample_video.py --output ./test_videos --count 1 --duration 10

# 测试 3 个算法 (需要先启动 mock_webhook)
for algo in mog2 running_avg frame_diff; do
    python tests/e2e_test.py --algo $algo --duration 8
done
```

## 3 种算法对比

### 算法选择速查表

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 相机完全固定 + 光照稳定 + 车持续移动 | **`frame_diff`** ⭐ | 最快，0 背景学习 |
| 相机固定 + 白天/夜晚切换 | `running_avg` | 背景慢慢适应光照 |
| 复杂光照 (树影/云影/车阴影) | `mog2` | 多高斯模型自适应 |
| 相机微抖动 (风/振动) | `mog2` | 抗噪能力强 |
| 单核 1GHz ARM 极端低算力 | `frame_diff` + 大 `frame_skip` | 最低算力需求 |

### 各算法原理

#### `frame_diff` — 帧间差分 (最快, ~0.2ms/帧)

```
当前帧 ──┐
         ├── absdiff → 阈值化 → 形态学开 → 累积 mask (滑动衰减)
上一帧 ──┘                                          │
                                                    ├─ countNonZero ≥ min_area?
                                                    │   ↓
                                            有车 → 连续 N 帧 → 告警
```

**适用**：相机完全固定、光照恒定、目标持续移动
**优势**：极快 (0.2ms)、无背景学习、抗偶发噪声
**劣势**：对相机抖动敏感，静止目标不响应

#### `running_avg` — 滑动平均背景 (平衡, ~0.5ms/帧)

```
新帧 ──────────────┐
                   ├── accumulateWeighted (α) → 背景更新
历史背景 ──────────┘
新帧 ──┐
       ├── absdiff → 阈值化 → countNonZero
背景 ──┘
```

**适用**：相机固定、光照缓慢变化的场景
**优势**：比 MOG2 快 2-3 倍，背景自适应
**劣势**：快速光照变化时短暂误报

#### `mog2` — 高斯混合背景 (最稳, ~1.5ms/帧)

```
新帧 → MOG2.apply (多高斯模型自适应) → 形态学开+闭 → findContours → max(area)
```

**适用**：复杂光照、阴影干扰、相机微抖动
**优势**：MOG2 自适应阴影抑制、对光照鲁棒
**劣势**：计算量最大

### 性能对比 (实测)

`python scripts/bench_algorithms.py --realistic` 输出 (Mac M2 → A55 估算 ×3.5)：

```
720p 源 + 0.5x (360p 检测):
  mog2         0.97 ms  →  A55 3.4 ms  (294 fps/核)  20路4核 = 157 路
  running_avg  0.47 ms  →  A55 1.6 ms  (609 fps/核)  20路4核 = 325 路
  frame_diff   0.19 ms  →  A55 0.7 ms  (1498 fps/核) 20路4核 = 799 路
```

> 实际瓶颈是 **RTSP 解码** 而非检测算法。1080p 软解 ~30-50ms/帧/A55 单核。
> 强烈建议用 **720p 辅码流** + `frame_diff` 算法。

## 配置详解

完整配置见 `config.example.yaml`，所有字段都带注释。

### supervisor (全局)

```yaml
supervisor:
  workers: 2              # worker 进程数
                          # 4G 设备: 2 (推荐)
                          # 8G 设备: 4
                          # 16G+ 服务器: 4-8
  log_level: INFO         # DEBUG / INFO / WARNING / ERROR
  log_file: ""            # 留空=stdout, 填路径=写文件
```

### notifier (告警下发)

```yaml
notifier:
  http:
    url: "http://your-server/webhook"   # 告警接收地址, 留空=只写日志
    timeout: 3                          # HTTP 超时 (秒)
    retry: 2                            # 失败重试次数
    headers:                            # 自定义 header
      Authorization: "Bearer your-token"
      X-Source: car-sense-lite
```

### channels (通道列表)

每路通道独立配置：

```yaml
channels:
  - id: "gate_1"                       # 唯一 ID, 用于日志和告警
    name: "大门入口"                     # 人类可读名
    source: "rtsp://user:pass@host/path"  # RTSP / HTTP / 本地文件
    enabled: true                      # false = 跳过此通道
    detector:
      algorithm: frame_diff            # mog2 / running_avg / frame_diff

      # 通用
      downsample: 0.5                  # 降采样比例 (1.0=不降采样, 0.5=一半尺寸)
      frame_skip: 2                    # 跳帧: 每 N 帧处理 1 次 (15fps源→7.5fps)
      min_area: 1500                   # 最小运动面积 (像素²)
      trigger_frames: 3                # 连续 N 帧检测到才告警
      cooldown_sec: 5                  # 告警后冷却秒数 (一辆车只告警一次)
      warmup_sec: 5                    # 启动后预热秒数 (让背景模型稳定)
      roi:                             # 多边形 ROI, 留空=全屏
        - [x1, y1]
        - [x2, y2]
        - [x3, y3]
        - [x4, y4]

      # mog2 专属
      history: 200                     # MOG2 背景历史帧数
      var_threshold: 32                # MOG2 方差阈值 (越大越宽松)

      # running_avg / frame_diff 专属
      diff_threshold: 25               # 像素差阈值 (灰度, 越大越宽松)

      # running_avg 专属
      bg_alpha: 0.05                   # 背景学习率 (0.01-0.1, 越小越稳定)

      # frame_diff 专属
      diff_decay: 5                    # 累积 mask 每帧衰减量
```

### ROI 多边形配置

俯视/斜俯视下车道通常是不规则四边形。坐标系：左上角 (0,0)。

```yaml
# 假设 1080p 画面, 车道在中央 1/3 区域
roi:
  - [400, 200]      # 左上
  - [1480, 200]     # 右上
  - [1700, 880]     # 右下
  - [180, 880]      # 左下
```

调试 ROI 配置的小技巧：在主进程加一行 `cv2.imwrite('roi_debug.jpg', frame_with_roi)` 看下叠加效果。

## 告警接口 (Webhook)

### 请求格式

`POST <notifier.url>` `Content-Type: application/json`

```json
{
  "channel_id": "gate_1",
  "channel_name": "大门入口",
  "event": "car_detected",
  "timestamp": 1782972062.65,
  "iso_time": "2026-07-02T14:01:02.649+08:00",
  "frame_seq": 576,
  "max_area": 3200
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `channel_id` | string | 通道唯一 ID |
| `channel_name` | string | 通道名称 |
| `event` | string | 固定 `"car_detected"` |
| `timestamp` | float | Unix 时间戳 (秒, 浮点) |
| `iso_time` | string | ISO 8601 带时区 |
| `frame_seq` | int | 触发告警时的帧序号 |
| `max_area` | int | 最大运动轮廓面积 (像素²) |

### 业务侧处理示例 (Python / Flask)

```python
from flask import Flask, request

app = Flask(__name__)

@app.post("/webhook/car")
def receive():
    data = request.get_json()
    print(f"[{data['channel_name']}] 来车: seq={data['frame_seq']}")
    # TODO: 推送到钉钉/企微/邮件
    return {"ok": True}

if __name__ == "__main__":
    app.run(port=8000)
```

### 自定义 Header

通过 `notifier.http.headers` 添加鉴权 header：

```yaml
notifier:
  http:
    url: "https://api.your-server.com/cars"
    headers:
      Authorization: "Bearer eyJhbGc..."
      X-Tenant: factory-a
```

### 失败处理

- HTTP 2xx 视为成功
- 非 2xx 或超时: 指数退避重试 `retry` 次 (默认 2 次, 0.2s / 0.4s 间隔)
- 全部失败: 写 ERROR 日志, 事件丢弃
- 队列满 (1024 积压): 丢弃新事件, 每 50 个打印一次警告

## 部署到边缘设备

### 方式 1: Docker Compose (推荐, 一键部署)

最简单的方式，所有依赖、配置、开机启动都在容器里：

```bash
# 1. 准备配置
cp config.example.yaml config.yaml
vim config.yaml          # 填入 RTSP 地址和 webhook URL

# 2. 构建并启动 (后台)
docker compose up -d --build

# 3. 查看日志
docker compose logs -f

# 4. 停止
docker compose down
```

**docker-compose.yml 已配置**：
- `network_mode: host` — 直连局域网摄像头 (RTSP 不需要端口映射)
- 资源限制: 3.8 CPU + 2.5GB 内存 (4 核 4G 设备)
- 日志轮转: 50MB × 5 文件
- 健康检查: 每 30s 检测进程
- 自动重启: `unless-stopped`

**常用命令**：

```bash
docker compose ps            # 查看状态
docker compose restart       # 重启
docker compose top           # 看进程 CPU/内存
docker compose pull          # 拉取最新镜像 (如果用了 registry)
docker compose down -v       # 停止并删除日志目录
```

**自定义资源限制**（编辑 `docker-compose.yml` 的 `deploy.resources`）：

```yaml
# 8G 设备配置
deploy:
  resources:
    limits:
      cpus: '7.5'
      memory: 6G
```

### 方式 2: systemd (直接装在系统上, 不走容器)

```bash
# 一键安装 (在边缘设备上)
sudo scripts/install_systemd.sh

# 之后管理服务
sudo systemctl start car-sense-lite
sudo systemctl status car-sense-lite
sudo journalctl -u car-sense-lite -f
```

安装脚本会:
1. 复制项目到 `/opt/car-sense-lite`
2. 安装 Python 依赖
3. 部署 systemd unit (CPU 380% / 内存 2.5G 上限)
4. 设置开机自启

### 方式 3: nohup 后台运行 (临时调试)

```bash
nohup python run.py -c config.yaml > /var/log/car-sense.log 2>&1 &
echo $! > /var/run/car-sense-lite.pid

# 停止
./scripts/stop.sh
```

### Docker 镜像信息

- 基础镜像: `python:3.11-slim` (Debian Bookworm)
- 最终大小: ~600MB (含 OpenCV + FFmpeg)
- 架构: `linux/amd64`, `linux/arm64` (A55 设备用 arm64)

## 调参指南

### 误报太多？

| 措施 | 效果 |
|------|------|
| 调大 `min_area` (1500 → 3000) | 过滤小目标 (虫子/光斑) |
| 调大 `trigger_frames` (3 → 5) | 要求更多连续帧 |
| 调紧 `roi` (缩小多边形) | 排除干扰区域 |
| 改用 `mog2` + 调大 `var_threshold` (32 → 50) | 抗光照波动 |
| 调大 `diff_threshold` (25 → 40) | 提高像素差阈值 |

### 漏报太多？

| 措施 | 效果 |
|------|------|
| 调小 `min_area` (1500 → 800) | 敏感小目标 |
| 调大 `downsample` (0.4 → 0.6) | 提高检测分辨率 |
| 调小 `frame_skip` (2 → 1) | 检测更频繁 |
| 调大 `diff_threshold` (25 → 50) | frame_diff 需要更明显运动 |
| 调大 `bg_alpha` (0.05 → 0.1) | running_avg 更快适应 |
| 调小 `var_threshold` (32 → 16) | mog2 更敏感 |

### CPU 占用太高？

1. 调大 `frame_skip` (2 → 4)
2. 调小 `downsample` (0.5 → 0.3)
3. 减小 `workers` 数
4. 改用 `frame_diff`

### 内存吃紧？

- 减小 `workers` 数 (4G 设备 1-2 worker)
- 检查 RTSP 流是否占用过多 buffer (网络慢时 cv2.VideoCapture 会缓存)

## 性能基准

### 实测平台

- **测试机**: MacBook Pro M2, 8 核, 16GB RAM
- **目标机**: 4 核 ARM Cortex-A55 @ 2GHz, 4GB RAM (Linaro Debian)
- **缩放系数**: A55 性能 ≈ M2 的 1/3.5 (IPC 差异 4× + 频率差异 0.6×)

### 各场景性能

```
720p 源 + 0.5x 降采样到 360p:
  mog2         A55 3.4 ms/帧  (294 fps/单核)  20路4核容量 157路
  running_avg  A55 1.6 ms/帧  (609 fps/单核)  20路4核容量 325路
  frame_diff   A55 0.7 ms/帧  (1498 fps/单核) 20路4核容量 799路

1080p 源 + 0.4x 降采样到 432p:
  mog2         A55 18.9 ms/帧 (53 fps/单核)   20路4核容量 28路
  running_avg  A55 16.5 ms/帧 (61 fps/单核)   20路4核容量 32路
  frame_diff   A55 12.8 ms/帧 (78 fps/单核)   20路4核容量 42路
```

### 资源占用

| 设备 | workers | 物理内存 (Linux) | 备注 |
|------|---------|------------------|------|
| 4G 边缘 | 2 | ~500 MB | 默认推荐 |
| 8G 设备 | 4 | ~1 GB | 性能更稳 |
| 16G+ 服务器 | 4-8 | ~2 GB | 单 worker 抗更多路 |

> 注: macOS 测的进程 RSS 偏大 (含 OpenCV shared lib), Linux 上数字会小很多。

## 开发指南

### 项目结构

```
car-sense-lite/
├── run.py                    # 启动入口
├── config.example.yaml       # 配置模板
├── requirements.txt
├── src/
│   ├── config.py             # YAML 解析 + 校验
│   ├── logger.py             # 统一日志
│   ├── roi.py                # 多边形 ROI 懒加载
│   ├── stream.py             # RTSP/文件 拉流 + 断流重连
│   ├── detector.py           # 3 种算法的 BaseDetector + 工厂
│   ├── notifier.py           # HTTP POST 告警 + 异步队列
│   ├── worker.py             # 单路 pipeline
│   └── supervisor.py         # 多进程模型
├── tests/
│   ├── test_*.py             # 单元测试
│   └── e2e_test.py           # 端到端 pipeline 验证
└── scripts/
    ├── gen_sample_video.py   # 合成测试视频
    ├── mock_webhook.py       # 模拟告警接收
    ├── bench.py              # 多 worker 压测
    ├── bench_algorithms.py   # 3 种算法对比
    ├── stop.sh               # 停止脚本
    ├── install_systemd.sh    # 部署
    └── car-sense-lite.service
```

### 添加新算法

继承 `BaseDetector` 并注册到工厂：

```python
# src/detector.py
class MyDetector(BaseDetector):
    algo_name = "my_algo"
    
    def __init__(self, cfg: DetectorConfig):
        super().__init__(cfg)
        # 初始化你的算法
    
    def detect(self, frame_bgr: np.ndarray) -> bool:
        # 返回 True = 有车, False = 无车
        ...
    
    def reset(self) -> None:
        # 重置内部状态
        ...

ALGO_REGISTRY["my_algo"] = MyDetector
```

加测试 (`tests/test_algorithms.py`)，跑通后即可在 `config.yaml` 选用。

### 跑测试

```bash
# 全部单元测试
python -m unittest discover -s tests -v

# 端到端 (3 个算法)
for algo in mog2 running_avg frame_diff; do
    python tests/e2e_test.py --algo $algo --duration 8
done

# 算法性能对比
python scripts/bench_algorithms.py --realistic

# 多 worker 压测
python scripts/bench.py --channels 20 --duration 30
```

### 日志解读

```
[INFO] [car-worker-0] car-sense.worker: [gate_1] ALERT seq=576 counter=3
                                  └─ 通道  └─ 告警  └─ 帧序号  └─ 连续命中次数

[INFO] [car-worker-0] car-sense.worker: [gate_1] stats: 7.5 fps processed, 12 alerts, counter=0
                                  └─ 每 30s 打印一次统计

[ERROR] [car-worker-0] car-sense.notifier: [gate_1] notify failed after 2 retries: ...
                                  └─ 告警 HTTP 推送失败
```

## FAQ

### Q1: 启动后没有任何告警？

1. 检查 RTSP 地址是否正确: `ffplay rtsp://...` 验证
2. 检查 `--check` 模式配置: `python run.py -c config.yaml --check`
3. 查看 worker 日志是否 `stream connected`
4. 调小 `min_area` (默认 1500) 试试
5. 临时把 `cooldown_sec: 0` 看是否触发

### Q2: 误报很多？

通常是光照变化或树影。优先:
1. 收紧 `roi` 到车道区域
2. 改用 `mog2` 算法
3. 调大 `min_area` 和 `trigger_frames`
4. 调大 `var_threshold` (mog2) 或 `diff_threshold` (其他)

### Q3: 想在画面叠加 ROI / 检测框调试？

可以在 `worker.py` `_on_detection` 后加:

```python
if self.ch.detector.roi and not self.detector.roi.is_full:
    cv2.polylines(small, [np.array(self.ch.detector.roi)], True, (0, 255, 0), 2)
cv2.imwrite(f"/tmp/debug_{self.ch.id}.jpg", small)
```

### Q4: 能检测车离开吗？

可以，调小 `cooldown_sec` 或加一个 `cooldown_sec_departure` 字段。当前架构只检测"出现"，离开靠 cooldown 自然结束。

### Q5: 能不能检测车速度？

`max_area` 字段反映了车在画面中的大小，结合帧间隔可估算。但当前不输出速度，需要改 detector 输出额外字段。

### Q6: Docker 部署摄像头访问问题？

摄像头在内网时用 `--network host`；需要跨主机用 `--add-host` 或 `docker network`。

### Q7: Windows 能不能跑？

能，但 OpenCV 的 `cv2.CAP_FFMPEG` 在 Windows 上行为略不同。推荐 Windows 上用 `algorithm: mog2` (最稳定)。

### Q8: 多个通道能不能用同一路 RTSP？

能，配置两条 `channels` 共享 `source` URL 即可。会建立两个独立 RTSP 连接 (OpenCV 不支持多消费者共享 capture)。

## License

MIT
