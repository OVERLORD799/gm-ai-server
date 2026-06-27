#!/bin/bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda-envs/vlm
export HF_HOME=/root/gpufree-data/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export GDINO_MODEL_ID=IDEA-Research/grounding-dino-base
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_s.yaml
export SAM2_CHECKPOINT=/root/gpufree-data/perception-service/checkpoints/sam2.1_hiera_small.pt
export PERCEPTION_HOST=127.0.0.1
export PERCEPTION_PORT=8082
cd /root/gpufree-data/perception-service
exec python app.py
