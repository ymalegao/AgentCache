#!/bin/bash
cd /root/AgentCache
source venv/bin/activate
export VLLM_USE_DEEP_GEMM=0
run() { # mode-tag  extra-args  outfile
  echo "### START $1 $(date +%T)"
  if python agentcache_compression/multi_turn_benchmark.py $2 \
      --model models/gpt-oss-20b \
      --conversation-file agentcache_compression/conversations/csv_cli.json \
      --max-tokens 2048 --gpu-mem 0.90 --max-model-len 32768 \
      --out "agentcache_compression/results/tier1/$3.jsonl" ; then
    echo "### END $1 $(date +%T)"
  else
    echo "### FAILED $1 $(date +%T)"; exit 1
  fi
}
for r in 0 1 2 3 4; do
  for N in 64 128 256; do
    rm -f agentcache_compression/results/tier1/syn${N}_r${r}.jsonl
    run "syn${N}_r${r}" "--mode synthetic --synthetic-len ${N} --centroid-k agentcache_compression/centroids/gptoss_dummy_N${N}_K.npy --centroid-v agentcache_compression/centroids/gptoss_dummy_N${N}_V.npy" "syn${N}_r${r}"
  done
done
for r in 0 1; do
  rm -f agentcache_compression/results/tier1/coldchk_r${r}.jsonl
  run "cold_r${r}" "--mode cold" "coldchk_r${r}"
done
echo "### ALL_DONE $(date +%T)"
