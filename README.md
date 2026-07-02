# gm-ai-server — VLM & Perception AI 专用服务器

GM-SafePick Layer 3（VLM 推理增强层）的远端推理引擎。本仓库为本地源码副本，实际服务运行在远端 GPU 服务器上，经 SSH 隧道供 Isaac Sim 节点消费。

## 拓扑

```
┌─────────────────────────────────┐      SSH 隧道        ┌──────────────────────────────┐
│  Isaac Sim 节点                  │  ←───────────────→  │  gm-ai-server                 │
│                                  │                     │                               │
│  configs/vlm_client.yaml        │  localhost:18080     │  :8080  → vlm-service         │
│    base_url: 127.0.0.1:18080    │     ─────────→      │    Qwen2.5-VL-7B (4-bit NF4)  │
│                                  │                     │                               │
│  configs/perception_client.yaml │  localhost:18082     │  :8082  → perception-service  │
│    base_url: 127.0.0.1:18082    │     ─────────→      │    GDINO-base + SAM2-hiera-s  │
└─────────────────────────────────┘                     └──────────────────────────────┘
```

---

## 硬件要求

| 项 | 最低 | 推荐（当前） |
|:---|:-----|:-----------|
| GPU | NVIDIA 24GB VRAM | **L40S 48GB** |
| 驱动 | ≥535 | **580.126.09** |
| RAM | 32GB | **1 TB** |
| 磁盘 | 50GB（模型缓存） | `/root/gpufree-data`（49GB+） |
| OS | Ubuntu 22.04 | Ubuntu 22.04 (gpufree-container) |
| Python | 3.11 | 3.11 (conda) |

## GPU 显存占用

| 组件 | 端口 | 模型 | 常驻显存 |
|:-----|:----:|:-----|:---------|
| VLM | `:8080` | Qwen2.5-VL-7B (bitsandbytes 4-bit NF4) | ~6.5 GB |
| Perception | `:8082` | GDINO-base + SAM2-hiera-small | ~2.1 GB |
| **合计** | | | **~8.6 GB / 48 GB** |

---

## 目录结构

```
/root/gpufree-data/                    # 远端服务器工作目录
├── conda-envs/vlm/                    # Conda 环境 (Python 3.11)
├── huggingface/                       # HF 模型缓存 (HF_HOME)
├── vlm-service/
│   ├── app.py                         # VLM FastAPI (/health, /analyze)
│   ├── start.sh                       # 启动脚本
│   ├── smoke_test.py                  # 离线烟测
│   ├── sample.jpg                     # 测试图片
│   └── supervisor.out.log            # supervisord 日志
└── perception-service/
    ├── app.py                         # 感知 FastAPI (/health, /ground, /track)
    ├── start.sh                       # 启动脚本
    ├── smoke_test.py                  # 离线烟测
    ├── checkpoints/
    │   └── sam2.1_hiera_small.pt      # SAM2 权重 (~149 MB)
    └── supervisor.out.log            # supervisord 日志
```

本仓库 (`/root/gm-ai-server/`) 为**源码副本**，修改后 `scp` 到远端服务器部署：

```
/root/gm-ai-server/          # 本机（Isaac 节点）源码
├── vlm-service/
├── perception-service/
├── supervisord/
└── README.md                # 本文档
```

---

## 一、安装（全新服务器部署）

### 1.1 环境准备

```bash
# SSH 登录远端 AI 服务器
ssh -p <PORT> root@<HOST>

# 确认 GPU 可见
nvidia-smi

# 创建目录结构
mkdir -p /root/gpufree-data/{conda-envs,huggingface,vlm-service,perception-service/checkpoints}
```

### 1.2 创建 Conda 环境

```bash
source /opt/conda/etc/profile.d/conda.sh
conda create -y -p /root/gpufree-data/conda-envs/vlm python=3.11 pip
conda activate /root/gpufree-data/conda-envs/vlm
```

### 1.3 安装 PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 1.4 安装 VLM 依赖

```bash
pip install transformers accelerate qwen-vl-utils pillow bitsandbytes \
  fastapi uvicorn[standard] httpx pydantic
```

### 1.5 安装感知依赖

```bash
pip install opencv-python-headless supervision sam2
```

### 1.6 设置 HF 镜像（国内服务器必需）

```bash
export HF_HOME=/root/gpufree-data/huggingface
export HF_ENDPOINT=https://hf-mirror.com
```

> **重要**：服务器直连 `huggingface.co` 偶发 `Network unreachable`，HF 镜像可避免。这两个变量已写入 `start.sh` 启动脚本。

### 1.7 下载 SAM2 权重

```bash
conda activate /root/gpufree-data/conda-envs/vlm
export HF_HOME=/root/gpufree-data/huggingface
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="facebook/sam2.1-hiera-small",
    filename="sam2.1_hiera_small.pt",
    local_dir="/root/gpufree-data/perception-service/checkpoints",
)
PY
```

### 1.8 部署源码到远端

```bash
# 在 Isaac 节点执行（scp 到远端）
scp -P <PORT> -r /root/gm-ai-server/vlm-service/* \
  root@<HOST>:/root/gpufree-data/vlm-service/
scp -P <PORT> -r /root/gm-ai-server/perception-service/* \
  root@<HOST>:/root/gpufree-data/perception-service/
```

### 1.9 烟测验证

```bash
# 在远端服务器执行
conda activate /root/gpufree-data/conda-envs/vlm
export HF_HOME=/root/gpufree-data/huggingface
export HF_ENDPOINT=https://hf-mirror.com

# VLM 烟测（首次下载 Qwen2.5-VL-7B 约 7 分钟，缓存后 ~15s）
python /root/gpufree-data/vlm-service/smoke_test.py

# 感知烟测（首次下载 GDINO 约 2-5 分钟）
python /root/gpufree-data/perception-service/smoke_test.py
# 期望输出末尾: smoke_ok
```

---

## 二、服务启动

### 2.1 supervisord 常驻（推荐）

gpufree 容器使用 supervisord 管理进程，容器**重启后自动拉起**（`autostart=true`, `autorestart=true`）。

**安装 supervisord 配置**：

```bash
# 在远端服务器上
cp /root/gm-ai-server/supervisord/*.conf /.gpufree/
```

> 配置落盘后需**重启 gpufree 容器**（或等待容器重建）以让 supervisord 读入新配置。`ctl reload` 不会为新增 `/.gpufree/*.conf` 拉起进程。

**启停命令**（在远端服务器上）：

```bash
# 启动
/data/supervisord ctl -c /opt/supervisord.yaml start vlm-service
/data/supervisord ctl -c /opt/supervisord.yaml start perception-service

# 停止
/data/supervisord ctl -c /opt/supervisord.yaml stop vlm-service
/data/supervisord ctl -c /opt/supervisord.yaml stop perception-service

# 查看状态
/data/supervisord ctl -c /opt/supervisord.yaml status
```

> **警告**：不要对 supervisord 主进程 `kill`（会连带 sshd 短暂不可用）。仅操作具体 program。

### 2.2 手工启动（调试用，非首选）

```bash
# VLM
nohup /root/gpufree-data/vlm-service/start.sh \
  >> /root/gpufree-data/vlm-service/server.log 2>&1 &

# 感知
nohup /root/gpufree-data/perception-service/start.sh \
  >> /root/gpufree-data/perception-service/server.log 2>&1 &
```

### 2.3 启动时间

| 阶段 | 耗时 |
|:-----|:-----|
| VLM 加载权重（缓存后） | 15–90s 后 `/health` 返回 `ok` |
| VLM 首次下载模型 | ~7 分钟（HF 镜像） |
| 感知懒加载（首次调用触发） | ~2–5 分钟（GDINO 下载 + 加载） |

---

## 三、健康检查

### 远端本机验证

```bash
# VLM
curl -s http://127.0.0.1:8080/health | python3 -m json.tool
# 期望: {"status":"ok","model_id":"Qwen/Qwen2.5-VL-7B-Instruct","gpu":"NVIDIA L40S"}

# 感知
curl -s http://127.0.0.1:8082/health | python3 -m json.tool
# 期望: {"status":"ok"|"warming","models_loaded":true|false,...}

# 同时检查
curl -s http://127.0.0.1:8080/health && curl -s http://127.0.0.1:8082/health
```

### 端口占用检查

```bash
# 确认端口绑定（应监听 0.0.0.0:8080 和 127.0.0.1:8082）
ss -tlnp | grep -E '808[02]'

# 查找孤儿进程
pgrep -af "python app.py"
```

---

## 四、HTTP API

### 4.1 VLM `/analyze`（端口 `:8080`）

**请求**：

```bash
# 通过文件路径
curl -s -X POST http://127.0.0.1:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Is a human hand blocking the bin?",
    "image_path": "/root/gpufree-data/vlm-service/sample.jpg"
  }' | python3 -m json.tool

# 通过 base64
curl -s -X POST http://127.0.0.1:8080/analyze \
  -H "Content-Type: application/json" \
  -d "{
    \"image_b64\": \"$(base64 -w0 /root/gpufree-data/vlm-service/sample.jpg)\"
  }" | python3 -m json.tool
```

**响应字段**（对齐 Layer 3 `vlm_*` CSV 列）：

```json
{
  "model_id": "Qwen2.5-VL-7B-Instruct-4bit-nf4",
  "latency_ms": 850.3,
  "text": "{\"risk_type\": \"none\", ...}",
  "vlm_keywords": ["robot arm", "container", "parts"],
  "vlm_risk_type": "none",
  "vlm_risk_confidence": 0.9,
  "vlm_suggested_action": "continue",
  "vlm_explanation": "Normal pick-and-place, no human near the robot."
}
```

| 字段 | 类型 | 说明 |
|:-----|:-----|:-----|
| `model_id` | string | 模型标识 |
| `latency_ms` | float | 推理耗时（毫秒） |
| `vlm_keywords` | list | 场景关键词，可喂给感知服务 |
| `vlm_risk_type` | string | `static` / `dynamic` / `functional` / `none` |
| `vlm_risk_confidence` | float | 0.0–1.0 |
| `vlm_suggested_action` | string | `continue` / `slow_down` / `replan` / `stop` |
| `vlm_explanation` | string | 自然语言安全解释 |

推理延迟：**850–1250 ms**（4-bit NF4，`max_new_tokens=256`）。

### 4.2 感知 `/ground`（端口 `:8082`）

```bash
curl -s -X POST http://127.0.0.1:8082/ground \
  -H "Content-Type: application/json" \
  -d '{
    "text_prompt": "gloved hand . robot gripper . container",
    "image_path": "/root/gpufree-data/vlm-service/sample.jpg",
    "box_threshold": 0.25,
    "run_sam2": true
  }' | python3 -m json.tool
```

**响应**：

```json
{
  "gdino_model_id": "IDEA-Research/grounding-dino-base",
  "sam2_checkpoint": "sam2.1_hiera_small.pt",
  "latency_ms": 750.5,
  "detections": [
    {
      "label": "hand",
      "score": 0.85,
      "box_xyxy": [120, 80, 400, 350],
      "mask_area": 82340,
      "sam2_score": 0.97
    }
  ]
}
```

首帧延迟 ~7.5s（含 GDINO 首次下载/加载），缓存后 ~150–300ms。

### 4.3 感知 `/track`（端口 `:8082`）

视频时序追踪：首帧 `init`（GDINO 检测 + SAM2 掩码），后续帧 `step`（SAM2 时序传播，10–20ms/帧）。

```bash
# init — 建追踪会话
curl -s -X POST http://127.0.0.1:8082/track \
  -H "Content-Type: application/json" \
  -d '{
    "action": "init",
    "frame_index": 0,
    "image_b64": "<PNG base64>",
    "init": {
      "target_label": "hand",
      "text_prompt": "gloved hand . robot gripper",
      "box_threshold": 0.25,
      "re_detect_every_n": 100
    },
    "meta": {"step": 0}
  }'

# step — 逐帧追踪
curl -s -X POST http://127.0.0.1:8082/track \
  -H "Content-Type: application/json" \
  -d '{
    "action": "step",
    "session_id": "<session_id from init>",
    "frame_index": 1,
    "image_b64": "<PNG base64>",
    "meta": {"step": 1}
  }'
```

**`/track` 响应**（含运动信息）：

```json
{
  "session_id": "uuid-...",
  "frame_index": 1,
  "re_detected": false,
  "latency_ms": 42.0,
  "tracks": [
    {
      "track_id": 0,
      "label": "hand",
      "box_xyxy": [118, 78, 402, 355],
      "center_xy": [260.0, 216.5],
      "velocity_xy_px_s": [-18.5, 12.3],
      "speed_px_s": 22.2,
      "direction_deg": 146.3,
      "mask_area": 82100,
      "sam2_score": 0.98
    }
  ]
}
```

---

## 五、SSH 隧道（Isaac Sim 节点连接）⚠️ 修改主机名的位置

### 5.1 隧道命令

VLM 和感知服务**仅监听 AI 服务器本机**（`127.0.0.1`），Isaac 节点通过 SSH 端口转发访问。

在 **Isaac 仿真节点**上执行：

```bash
# ============================================================
# ⚠️ 部署到新服务器时，修改下面两处：
# ============================================================

ssh -f -N -o ServerAliveInterval=60 \
  -L 18080:127.0.0.1:8080 \
  -L 18082:127.0.0.1:8082 \
  -p 30481 root@120.209.70.195
#     ↑↑↑↑↑  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
#     端口    主机地址
```

**各段含义**：

| 参数 | 当前值 | 说明 | 部署新服务器时 |
|:-----|:------|:-----|:--------------|
| `-p <PORT>` | `30481` | AI 服务器 SSH 端口 | **✅ 改这里** |
| `root@<HOST>` | `root@120.209.70.195` | AI 服务器地址 | **✅ 改这里** |
| `-L 18080:127.0.0.1:8080` | — | 本机 18080 → 远端 8080 (VLM) | 通常不改 |
| `-L 18082:127.0.0.1:8082` | — | 本机 18082 → 远端 8082 (感知) | 通常不改 |
| `-o ServerAliveInterval=60` | — | 心跳保活，防止空闲断开 | 建议保留 |

> **凭证**：SSH 密码/密钥不要写入任何仓库文件或 Markdown。存储在 Isaac 节点的 `/root/.github_token`（`chmod 600`）末尾。

### 5.2 Isaac Sim 侧配置文件（不需改）

隧道建立后，以下配置文件保持 `127.0.0.1` 即可（它们走的是本机隧道端口，不直连远端）：

| 配置文件 | 内容 | 说明 |
|:---------|:-----|:-----|
| `configs/vlm_client.yaml` | `base_url: http://127.0.0.1:18080` | 隧道转发到远端 `:8080` |
| `configs/perception_client.yaml` | `base_url: http://127.0.0.1:18082` | 隧道转发到远端 `:8082` |

> **只要远端 SSH 地址/端口不变，这两个配置文件不需要修改。**

### 5.3 隧道验证

```bash
# 在 Isaac 节点执行
curl -s http://127.0.0.1:18080/health | python3 -m json.tool
# 期望: {"status":"ok","model_id":"Qwen/Qwen2.5-VL-7B-Instruct","gpu":"NVIDIA L40S"}

curl -s http://127.0.0.1:18082/health | python3 -m json.tool
# 期望: {"status":"ok"|"warming",...}
```

### 5.4 隧道断开重连

```bash
# 查找并杀掉旧隧道
pgrep -af "ssh.*-L.*18080"   # 记下 PID
kill <PID>

# 重新建立
ssh -f -N -o ServerAliveInterval=60 \
  -L 18080:127.0.0.1:8080 \
  -L 18082:127.0.0.1:8082 \
  -p 30481 root@120.209.70.195
```

### 5.5 Isaac Sim 仿真命令（带 VLM + 感知）

```bash
source /root/activate_isaaclab.sh
cd /root/GMRobot

# VLM only
python scripts/gm_state_machine_agent.py \
  --task=gm --headless --enable_cameras --enable_safety --enable_vlm

# VLM + 感知
python scripts/gm_state_machine_agent.py \
  --task=gm --headless --enable_cameras --enable_safety \
  --enable_vlm --enable_perception

# 500 步联调短测
python scripts/gm_state_machine_agent.py \
  --task=gm --headless --enable_cameras --enable_safety \
  --max_steps=500 --progress_interval=100 --enable_vlm
```

---

## 六、日常维护

### 6.1 查看日志

```bash
# supervisord 日志（远端服务器）
tail -f /root/gpufree-data/vlm-service/supervisor.out.log
tail -f /root/gpufree-data/vlm-service/supervisor.err.log
tail -f /root/gpufree-data/perception-service/supervisor.out.log

# 手工启动的日志
tail -f /root/gpufree-data/vlm-service/server.log
```

### 6.2 更新代码部署

```bash
# 在 Isaac 节点，scp 推送源码到远端
scp -P 30481 /root/gm-ai-server/vlm-service/app.py \
  root@120.209.70.195:/root/gpufree-data/vlm-service/
scp -P 30481 /root/gm-ai-server/perception-service/app.py \
  root@120.209.70.195:/root/gpufree-data/perception-service/

# 重启服务（在远端服务器上）
/data/supervisord ctl -c /opt/supervisord.yaml stop vlm-service
/data/supervisord ctl -c /opt/supervisord.yaml start vlm-service
```

### 6.3 查看 GPU 使用

```bash
# 远端服务器
nvidia-smi
watch -n 1 nvidia-smi
```

---

## 七、模型升级清单

| 组件 | 当前（生产） | 备选升级 | 影响 |
|:-----|:-----------|:---------|:-----|
| VLM 量化 | bitsandbytes 4-bit NF4 | vLLM AWQ | 延迟 ↓、吞吐 ↑；需验证 Qwen VL 兼容性 |
| GDINO | `grounding-dino-base` | `grounding-dino-large` | 精度 ↑、显存 ↑、延迟 ↑ |
| SAM2 | `sam2.1-hiera-small` | `sam2.1-hiera-large` | 精度 ↑、显存 ↑、延迟 ↑ |

> **AWQ 尝试记录（2026-06）**：Qwen2.5-VL-7B gptqmodel Marlin 内核要求 `out_features % 64 == 0`，但 Qwen vocab_size=3420 不满足，放弃 AWQ。当前统一用 bitsandbytes NF4。

---

## 八、故障排查

| 症状 | 检查 |
|:-----|:-----|
| `/health` 返回 `warming` | 感知服务未首次调用 `/ground` 加载模型 |
| `/health` 返回 `503` / 无响应 | `ss -tlnp \| grep 808` 查端口，`pgrep -af "python app.py"` 查进程 |
| `/analyze` 返回 503 | 模型未加载完成，等待 15-90s |
| 推理 OOM | `nvidia-smi` 确认显存，检查是否有孤儿进程占用 |
| SSH 隧道断开 | `pgrep -af "ssh.*-L"` 确认，重新执行隧道命令 |
| HF 下载 429 / Network unreachable | 确认 `HF_ENDPOINT=https://hf-mirror.com` 已设置 |
| supervisord `ctl start` 显示 `not started` | 以 `pgrep` / `curl` 实际状态为准，见 §二 |

---

## 相关文档

| 文档 | 说明 |
|:-----|:-----|
| [GM-SafePick_AI服务器部署.md](../GMRobot/source/GMRobot/docs/GM-SafePick_AI服务器部署.md) | 原部署文档（本文档源） |
| [GM-SafePick_Layer3_VLM推理增强层.md](../GMRobot/source/GMRobot/docs/GM-SafePick_Layer3_VLM推理增强层.md) | Layer 3 架构与 VLM 职责 |
| [GM-SafePick_架构总览.md](../GMRobot/source/GMRobot/docs/GM-SafePick_架构总览.md) | 三层安全架构 |
| [GM-SafePick_项目进展与遗留问题.md](../GMRobot/source/GMRobot/docs/GM-SafePick_项目进展与遗留问题.md) | 跨层进度看板 |
| [GM-SafePick_远程运行指南.md](../GMRobot/source/GMRobot/docs/GM-SafePick_远程运行指南.md) | Isaac Sim headless/VNC 运行 |
