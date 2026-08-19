#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${HF_HOME:=/data/huggingface}"
: "${VLM_HOST:=0.0.0.0}"
: "${VLM_PORT:=8080}"
: "${VLM_MODEL_ID:=Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"
: "${VLM_MODEL_REVISION:=536a35794df8831aa814970ee8f89eff577e7718}"
: "${VLM_API_MODEL_ID:=Qwen2.5-VL-7B-Instruct-awq}"
: "${VLM_QUANTIZATION:=awq}"
: "${VLM_AWQ_BACKEND:=torch_fallback}"
export HF_HOME VLM_HOST VLM_PORT VLM_MODEL_ID VLM_MODEL_REVISION VLM_API_MODEL_ID VLM_QUANTIZATION
export VLM_AWQ_BACKEND
exec python "$script_dir/app.py"
