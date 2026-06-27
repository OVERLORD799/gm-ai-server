#!/bin/bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda-envs/vlm-awq
export HF_HOME=/root/gpufree-data/huggingface HF_ENDPOINT=https://hf-mirror.com
cd /root/gpufree-data/vlm-awq-service
exec python app.py
