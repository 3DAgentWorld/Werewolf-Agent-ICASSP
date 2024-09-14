#!/bin/bash

MODEL=${1:-"/home/lanyihuai/ModelDownload/dataroot/model/THUDM/chatglm3-6b-32k"}
PORT=${2:-8000}
GPU=${3:-0}

CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --max_model_len 8192 \
    --gpu-memory-utilization 0.7
