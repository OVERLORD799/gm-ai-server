#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda-envs/vlm
export HF_HOME=/root/gpufree-data/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export VLM_PORT=8080
export VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
cd /root/gpufree-data/vlm-service
exec python app.py
