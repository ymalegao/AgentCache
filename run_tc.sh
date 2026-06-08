#!/bin/bash
cd /root/AgentCache
source venv/bin/activate
export VLLM_USE_DEEP_GEMM=0
REPEATS=5
for MODE in cold warm_apc; do
  for r in $(seq 0 $((REPEATS-1))); do
    OUT=agentcache_compression/results/tc/${MODE}_r${r}.jsonl
    rm -f "$OUT"
    echo "### START ${MODE} r${r} $(date +%T)"
    if ! python agentcache_compression/multi_turn_benchmark.py \
        --mode "$MODE" \
        --model models/gpt-oss-20b \
        --conversation-file agentcache_compression/conversations/csv_cli.json \
        --max-tokens 2048 \
        --gpu-mem 0.90 \
        --max-model-len 32768 \
        --out "$OUT" ; then
      echo "### FAILED ${MODE} r${r}"
      break 2
    fi
    echo "### END ${MODE} r${r} $(date +%T)"
  done
done
echo "### ALL_DONE $(date +%T)"
