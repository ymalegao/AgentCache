#!/bin/bash
cd /root/AgentCache
source venv/bin/activate
# Clean rerun: generate_turn_output now uses temp=0.7/top_p=0.9/seed=0 to avoid
# the greedy analysis-channel loops. deep_gemm disabled; 32k context.
export VLLM_USE_DEEP_GEMM=0
for MODE in cold warm_apc; do
  echo "### START $MODE $(date +%T)"
  if ! python agentcache_compression/multi_turn_benchmark.py \
      --mode "$MODE" \
      --model models/gpt-oss-20b \
      --conversation-file agentcache_compression/conversations/csv_cli.json \
      --max-tokens 2048 \
      --gpu-mem 0.90 \
      --max-model-len 32768 \
      --out agentcache_compression/results/gptoss_clean2.jsonl ; then
    echo "### FAILED $MODE"
    break
  fi
  echo "### END $MODE $(date +%T)"
done
echo "### ALL_DONE $(date +%T)"
