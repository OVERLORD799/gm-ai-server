#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${HF_HOME:=/data/huggingface}"
: "${GDINO_MODEL_ID:=IDEA-Research/grounding-dino-base}"
: "${GDINO_MODEL_REVISION:=12bdfa3120f3e7ec7b434d90674b3396eccf88eb}"
: "${SAM2_CONFIG:=configs/sam2.1/sam2.1_hiera_s.yaml}"
: "${SAM2_CHECKPOINT:=/data/checkpoints/sam2.1_hiera_small.pt}"
: "${SAM2_MODEL_ID:=sam2.1_hiera_small.pt}"
: "${PERCEPTION_HOST:=0.0.0.0}"
: "${PERCEPTION_PORT:=8082}"
export HF_HOME GDINO_MODEL_ID GDINO_MODEL_REVISION SAM2_CONFIG SAM2_CHECKPOINT SAM2_MODEL_ID
export PERCEPTION_HOST PERCEPTION_PORT
exec python "$script_dir/app.py"
