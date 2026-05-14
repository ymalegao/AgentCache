#!/bin/bash
# Pure PEFT mode: 64 virtual prefix tokens from the trained prefix adapter.
# No exact sys KV needed — centroid fills positions 0..63, scheduler gap=64.
export VLLM_CENTROID_SCHEDULER=1
export VLLM_CENTROID_USE_LMCACHE=0
# sys_prefix_num_tokens.txt next to the centroid files says 1 (written by transpose_tensors.py).
# Setting SYS_TOKENS=0 overrides it so centroid fills from position 0 and gap = 0+64 = 64.
export VLLM_CENTROID_SYS_TOKENS=0
export VLLM_CENTROID_K_PATH="/home/yash/agentcache/centroid_K.npy"
export VLLM_CENTROID_V_PATH="/home/yash/agentcache/centroid_V.npy"

/home/yash/agentcache/vllm-env/bin/python benchmark_ttft.py
