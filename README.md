# gm-ai-server

GM-SafePick AI 专用服务器代码（gm-ai-server）。
GPU: NVIDIA L40S 48GB，OS: Ubuntu 22.04。

## 服务

| 服务 | 端口 | supervisord 程序名 | 说明 |
|:-----|:----:|:-------------------|:-----|
| VLM | `:8080` | `vlm-service` | Qwen2.5-VL-7B 4-bit + FastAPI `/analyze` |
| Perception | `:8082` | `perception-service` | GDINO-base + SAM2-hiera-small，`/ground`、`/track` |

## 部署

```bash
# scp 到服务器
scp -P 30481 -r vlm-service/ perception-service/ supervisord/ root@120.209.70.195:/root/gpufree-data/

# supervisord 配置
cp supervisord/*.conf /.gpufree/

# 重启服务
/data/supervisord ctl -c /opt/supervisord.yaml restart vlm-service
/data/supervisord ctl -c /opt/supervisord.yaml restart perception-service
```

## 模型

| 组件 | 模型 |
|:-----|:-----|
| VLM | Qwen/Qwen2.5-VL-7B-Instruct (bitsandbytes 4-bit NF4) |
| GDINO | IDEA-Research/grounding-dino-base |
| SAM2 | facebook/sam2.1-hiera-small |
