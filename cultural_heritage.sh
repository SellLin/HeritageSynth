#!/bin/bash
set -x
export PYTHONUNBUFFERED=1


cd /LLaMA-Factory

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train ../qwen2_5vl_lora_sft.yaml \