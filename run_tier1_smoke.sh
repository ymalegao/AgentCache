#!/bin/bash
cd /root/AgentCache
source venv/bin/activate
export VLLM_USE_DEEP_GEMM=0
echo "### SMOKE synthetic N=256 $(date +%T)"
python agentcache_compression/multi_turn_benchmark.py \
  --mode synthetic --synthetic-len 256 \
  --centroid-k agentcache_compression/centroids/gptoss_dummy_N256_K.npy \
  --centroid-v agentcache_compression/centroids/gptoss_dummy_N256_V.npy \
  --model models/gpt-oss-20b \
  --conversation-file agentcache_compression/conversations/csv_cli.json \
  --max-tokens 2048 --gpu-mem 0.90 --max-model-len 32768 \
  --out agentcache_compression/results/tier1/smoke_syn256.jsonl \
  && echo "### SMOKE_OK $(date +%T)" || echo "### SMOKE_FAIL $(date +%T)"
