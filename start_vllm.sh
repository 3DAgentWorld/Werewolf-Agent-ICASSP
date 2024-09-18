#!/bin/bash

MODEL=${1:-"/path/to/ChatGLM3-6B"}
PORT=${2:-8000}
GPU=${3:-0}

source /home/zhangzheng/miniconda3/etc/profile.d/conda.sh
conda activate vllm

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=$GPU \
    python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --max_model_len 8192 \
    --gpu-memory-utilization 0.7
