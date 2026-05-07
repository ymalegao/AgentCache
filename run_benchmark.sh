#!/bin/bash
export VLLM_CENTROID_SCHEDULER=1
export VLLM_CENTROID_USE_LMCACHE=0
export VLLM_EXACT_SYS_K_PATH="/home/yash/agentcache/attention_centroid_output/sys_extended_K.npy"
export VLLM_EXACT_SYS_V_PATH="/home/yash/agentcache/attention_centroid_output/sys_extended_V.npy"
export VLLM_CENTROID_K_PATH="/home/yash/agentcache/attention_centroid_output/centroid_K.npy"
export VLLM_CENTROID_V_PATH="/home/yash/agentcache/attention_centroid_output/centroid_V.npy"

/home/yash/agentcache/vllm-env/bin/python benchmark_ttft.py
