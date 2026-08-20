# gm-ai-server

GMrobot 的 GPU 推理服务：

- VLM：Qwen2.5-VL-7B-Instruct-AWQ，提供 /health、/analyze
- Perception：Grounding DINO base + SAM2.1 Hiera Small，提供 /health、/ground、/track

默认 Docker 部署只绑定宿主机回环地址：

| GMrobot 地址 | 容器地址 | 服务 |
|---|---|---|
| 127.0.0.1:18080 | vlm:8080 | VLM |
| 127.0.0.1:18082 | perception:8082 | GDINO/SAM2 |

语义层是咨询层，不是认证安全功能。服务输出不能替代几何硬约束，也不能形成真机安全认证结论。

## 1. 从空服务器恢复

### 1.1 前置条件

- Linux x86-64
- NVIDIA GPU，建议显存至少 24 GiB
- NVIDIA 驱动支持 CUDA 12.8（本项目验证基线为 580 系列）
- Docker Engine、Compose v2+、NVIDIA Container Toolkit
- 至少 50 GiB 可用磁盘

先验证容器能看到 GPU：

~~~bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
~~~

### 1.2 克隆和配置

~~~bash
git clone https://github.com/OVERLORD799/gm-ai-server.git
cd gm-ai-server
cp .env.example .env
chmod 600 .env
~~~

.env 只包含本机端口、数据目录和可选 Hugging Face 镜像；不要把 token、SSH 密钥或密码放入仓库。

默认持久化目录为 ./data，已被 Git 忽略：

~~~text
data/
├── huggingface/          # Qwen/GDINO snapshot cache
├── checkpoints/         # SAM2 checkpoint
├── cache/
├── torch/
└── triton/
~~~

### 1.3 构建镜像

~~~bash
docker compose build
~~~

镜像基线：

- pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime（固定 digest 7db0e1bf…）
- Python 3.11
- Transformers 4.51.3
- AutoAWQ 0.2.9
- SAM2 source commit 2b90b9f5ceec907a1c18123530e92e794ad901a4

AutoAWQ 上游已停止维护，因此这里把它和 Transformers 版本一起固定为当前已验证组合；不要在灾备时单独升级，后续迁移到其他 AWQ 运行时应作为独立兼容性变更。

RTX 5090（sm_120）上，AutoAWQ 0.2.9 的 Triton int4 GEMM 无法完成 lowering。本恢复栈显式使用 AutoAWQ 自带的 `torch_fallback`，以可运行和确定失败为优先；它比优化内核慢。不得在未做真实模型 smoke/时延验证时切换 `VLM_AWQ_BACKEND=triton`。后续如需高吞吐，应独立评审 vLLM/Marlin 迁移，而不是在灾备任务中静默更换推理语义。

### 1.4 幂等预取模型

~~~bash
docker compose run --rm model-init
~~~

预取固定 revision：

| 模型 | Revision |
|---|---|
| Qwen/Qwen2.5-VL-7B-Instruct-AWQ | 536a35794df8831aa814970ee8f89eff577e7718 |
| IDEA-Research/grounding-dino-base | 12bdfa3120f3e7ec7b434d90674b3396eccf88eb |
| facebook/sam2.1-hiera-small | ee5bba1d82bb8749febdf90f45e84b687142ba03 |

下载器会校验 Qwen 两个 safetensors 分片（`4f75e3de…d1052`、`dae4128b…b8f0c`）、GDINO safetensors（`5548f844…ed21`）以及 SAM2 checkpoint（`6d1aa6f3…4d38`）的完整 SHA-256；精确值以 `scripts/download_models.py` 和 Compose 中的固定配置为准。

中断后可安全重复执行；下载缓存和 checkpoint 不进入 Git。
大权重默认由 aria2 以 8 条有界连接续传，完成后逐文件核对固定 SHA-256，再由 Hugging Face Hub 补齐 snapshot 元数据；网络异常最多重试 30 次并线性退避。可用 `MODEL_DOWNLOAD_CONNECTIONS` 调整到 1–16，诊断时可设置 `MODEL_DOWNLOAD_TRANSPORT=hub` 回退到 Hub 标准下载器。

### 1.5 启动

~~~bash
docker compose up -d --build vlm perception
docker compose ps
~~~

Compose 会先执行一次 model-init；固定 revision 下载和 SAM2 哈希校验成功后才启动两个常驻服务。VLM 随后加载 AWQ 权重，感知服务启动时预热 GDINO/SAM2。首次启动取决于磁盘和网络，可能需要数分钟。

`supervisord/` 只用于没有 Docker 的受管 GPU 容器，不能和 Compose 同时启用，以免重复占用端口与显存。

### 1.6 验证

~~~bash
python3 scripts/healthcheck.py --url http://127.0.0.1:18080/health --service vlm
python3 scripts/healthcheck.py --url http://127.0.0.1:18082/health --service perception --require-loaded
python3 vlm-service/smoke_test.py
python3 perception-service/smoke_test.py
~~~

查看日志：

~~~bash
docker compose logs --tail=200 vlm perception
~~~

停止服务但保留模型数据：

~~~bash
docker compose stop
~~~

## 2. 远程 GPU 服务器

推荐在 GPU 服务器上同样使用 Compose，但把远端宿主端口设为 8080/8082：

~~~dotenv
VLM_HOST_PORT=8080
PERCEPTION_HOST_PORT=8082
GM_AI_DATA_DIR=/srv/gm-ai-server-data
~~~

Compose 仍只绑定远端 127.0.0.1，避免服务暴露公网。在 Isaac/GMrobot 节点建立隧道：

若受管 GPU 容器没有 Docker，可使用平台自带的 Conda 与 supervisord。Python 环境、模型、SAM2 源码、缓存和日志都应放在数据盘；仓库本身放在 `/root/gm-ai-server`：

~~~bash
conda create -y --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  -p /root/gpufree-data/conda-envs/gm-ai python=3.11 pip

PY=/root/gpufree-data/conda-envs/gm-ai/bin/python
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0
"$PY" -m pip install -r requirements/vlm.txt -r requirements/perception.txt
"$PY" -m pip install --no-deps --no-build-isolation /root/gpufree-data/sam2
~~~

设置 `HF_HOME=/root/gpufree-data/huggingface`、可信 `HF_ENDPOINT` 和固定模型环境变量后，运行 `scripts/download_models.py --all`。模型全部校验完成后，将两个服务片段安装到平台的 supervisord include 目录。平台主进程不能在线重载时，可用仓库内的独立配置立即启动；实例下次重启后再由平台主进程接管：

~~~bash
install -m 0644 supervisord/vlm-service.conf /.gpufree/vlm-service.conf
install -m 0644 supervisord/perception-service.conf /.gpufree/perception-service.conf
/data/supervisord -d -c /root/gm-ai-server/supervisord/gm-ai-services.conf
~~~

当前 gpufree 配置针对 L40S 使用 Triton AWQ；RTX 5090 Docker 配置仍使用兼容的 torch fallback。无论哪条路径，都必须先核对端口只监听 127.0.0.1，再执行两个真实 smoke。避免同时运行独立 supervisor 与已加载同一片段的平台主 supervisor。

两种部署方式都通过以下隧道访问：

~~~bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -L 18080:127.0.0.1:8080 \
  -L 18082:127.0.0.1:8082 \
  -p "$GM_AI_SSH_PORT" "$GM_AI_SSH_USER@$GM_AI_SSH_HOST"
~~~

主机、端口、用户名和私钥必须由部署环境提供，不得提交到 Git。隧道成功后，GMrobot 配置继续使用 127.0.0.1:18080/18082。

## 3. API 身份与兼容性

### 3.1 VLM health

~~~json
{
  "status": "ok",
  "model_id": "Qwen2.5-VL-7B-Instruct-awq",
  "backend_model_id": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
  "quantization": "awq",
  "awq_backend": "torch_fallback",
  "model_loaded": true,
  "contract_mode": "canonical_v0a"
}
~~~

model_id 是 GMrobot 冻结的 API 身份；backend_model_id 和 revision 记录真实公开权重。服务拒绝请求中不匹配的 model/prompt/schema 身份，不会回显任意客户端标签冒充模型。

### 3.2 VLM analyze

POST /analyze 接受 base64 图片和 GMrobot canonical V0-A 元数据。成功响应包含：

- ok/request_id/frame_id
- scene_summary/keywords/risk_type/risk_confidence
- affected_entities/predicted_consequence/prediction_horizon_s/time_to_risk_s
- explanation/suggested_action/spatial_hint
- prompt_version/schema_version/model_id/latency_ms

为了旧客户端兼容，响应还保留 text 和 vlm_* 别名。模型输出缺字段、枚举非法或非 JSON 时返回 502，禁止静默合成风险语义。

### 3.3 Perception health

~~~json
{
  "status": "ok",
  "gdino_model_id": "IDEA-Research/grounding-dino-base",
  "sam2_model_id": "sam2.1_hiera_small.pt",
  "models_loaded": true,
  "contract_mode": "canonical_v0a"
}
~~~

/ground 与 /track 都返回相同的 model_versions，供 causal trace 校验。Tracking session 有固定上限、frame index 必须严格递增，服务进程只允许一个 tracking 操作在途。

## 4. 安全默认值

- 端口只绑定 127.0.0.1
- 容器 drop 全部 Linux capabilities 并启用 no-new-privileges
- 无 repo secret、无默认 SSH 地址、无默认凭据
- 默认禁用服务端任意 image_path 读取，只接受有大小上限的 canonical base64
- 镜像、prompt/schema、GDINO/SAM2 revision 均固定
- 常驻推理容器在模型预取完成后强制使用离线缓存，不在请求路径访问模型仓库
- VLM/GDINO/SAM2 失败时 HTTP 明确失败，不返回伪造成功
- Compose 使用单 worker/单 tracking in-flight，保持 GMrobot 的有界调用假设

## 5. 更新和灾备

代码恢复只需要 Git 仓库；模型数据全部来自公开固定 revision，可重新下载。为了缩短恢复时间，可另行备份 data/，但不要把它提交到 Git。

推荐备份：

- Git commit SHA
- .env 的非秘密配置副本
- data/huggingface 与 data/checkpoints（可选）
- docker image inspect 输出（可选）

恢复后必须重新执行 health 和两个 smoke tests，不能仅以容器进程存在作为服务可用证据。

~~~bash
git pull --ff-only
docker compose build
docker compose up -d --build vlm perception
~~~

## 6. 故障排查

| 症状 | 处理 |
|---|---|
| Docker 看不到 GPU | 检查 NVIDIA Container Toolkit 和 docker run --gpus all ... nvidia-smi |
| VLM 启动失败 | 查看 docker compose logs vlm；核对 24 GiB 显存、AWQ revision 和驱动 |
| 感知 health 一直 warming | 检查 checkpoint、GDINO 下载和 PERCEPTION_EAGER_LOAD=1 |
| Hugging Face 无法连接 | 在 .env 配置可信 HF_ENDPOINT；不要提交访问 token |
| 标准 HTTP 下载较慢 | 当前恢复基线默认 `HF_HUB_DISABLE_XET=1`、`HF_HUB_MAX_WORKERS=2`，优先保证断点真实落盘；只有链路验证稳定后才把 `HF_HUB_DISABLE_XET=0`，再按需提高 Xet 并发 |
| Xet 长时间停在 0% 或反复重传 | 恢复 `HF_HUB_DISABLE_XET=1`；不要把持续网络流量误当成已写入的模型数据。中国大陆链路如需镜像，必须使用运维认可且兼容 Hub HEAD 元数据的 `HF_ENDPOINT` |
| 端口未监听 | docker compose ps 并检查 18080/18082 |
| 远程隧道失败 | 使用 ExitOnForwardFailure=yes，确认远端只监听回环 8080/8082 |
| 模型身份不一致 | 停止验收；不得改名伪装，核对 Compose 固定 model/revision |
