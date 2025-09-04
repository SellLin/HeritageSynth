#!/bin/bash
set -x
export PYTHONUNBUFFERED=1


cd /data/xdd/LLaMA-Factory

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train /data/xdd/LLaMA-Factory/examples/cultural_heritage/qwen2_5vl_lora_sft.yaml cache_dir="/data/xdd/LLaMA-Factory/cache"\
    # resume_from_checkpoint="/data/xdd/LLaMA-Factory/saves/cultrual_heritage/qwen2_5vl-7b/lora/cn-loragen-mixed-sft/checkpoint-18500" \