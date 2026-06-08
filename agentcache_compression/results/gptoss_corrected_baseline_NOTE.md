# GPT-OSS-20B corrected cold/warm baseline

**Date:** 2026-06-08 · **GPU:** H100 80GB · **vLLM:** 0.20.0 · 5 repeats per mode.
Raw data: `results/tc/cold_r{0..4}.jsonl`, `results/tc/warm_apc_r{0..4}.jsonl`.

## Why this rerun
The paper's GPT-OSS-20B multi-turn table comes from `gptmulti_turn_benchmark.jsonl`
(its cold turn-1 = 1077.5 ms = the paper's 1078 ms). That run has three defects:
(1) corrupted/degenerate responses; (2) no engine warmup, so turn-1 includes one-time
torch.compile/CUDA-graph capture; (3) "cold" mode did not actually disable prefix
caching (vLLM V1 enables it by default), so turns 2-10 were cache-accelerated.

This rerun fixes all three: clean generation (temp=0.7/top_p=0.9/seed=0), a warmup
request before the measured turns, and `--no-enable-prefix-caching` for cold (true cold,
verified: kv-hit = 0% on every cold turn). Warm-APC keeps caching on (kv-hit 74-93%).

## Corrected numbers (median across 5 repeats)
| metric | Paper | Corrected true-cold | Corrected warm-APC |
|---|--:|--:|--:|
| cold turn-1 TTFT | 1078 ms | **120 ms** (stdev 16) | 116 ms (stdev 5) |
| cold mean TTFT   | 534.3 ms | **245 ms** | 181 ms |

Per-turn median TTFT (ms):
```
turn:        1    2    3    4    5    6    7    8    9   10
true-cold: 120  167  174  195  209  237  305  330  350  396    kv-hit 0% (full prefill each turn)
warm-APC:  116   81  101  182  185  207  212  238  181  273    kv-hit 74-93%
ptok:     2299 2983 3886 4997 6377 7942 9461 11147 11917 13597
```
(Cold per-turn stdev is large on turns 2-10, driven by the first repeat r0 running on a
cold GPU; the median is the robust estimator. Warm stdev is tight.)

## Findings
1. **Paper cold turn-1 is 9.0x too high** (1078 vs 120 ms). ~960 ms was one-time
   compilation, not prefill. A warmed full 2299-token prefill is 120 ms +/- 16.
2. **Paper cold mean is 2.18x too high** (534 vs 245 ms). The paper's "cold" combined a
   compile-inflated turn-1 with cache-deflated later turns.
3. **APC alone buys 1.35x mean** over true-cold and **0x at turn-1** (empty cache at
   turn 1, so true-cold ~= warm ~= 118 ms). That 1.35x is the real bar the centroid must beat.

## Implications for the paper
- The GPT-OSS-20B **4.3x turn-1** and **1.80x mean** speedups are computed against an
  inflated denominator and are invalid as stated. Recompute against corrected true-cold.
- Headroom is small: a correct full cold prefill is only ~120 ms on H100, and the centroid
  can save only the ~2000 system-prompt tokens of it, minus injection overhead. The large
  turn-1 win existed only because the old baseline was ~90% compile. This matches the
  paper's own Llama-1B Table 1 (1.1x when the baseline is small).
- The thesis/mechanism still stands (Table 1, the LMCache+Centroid section). Only the
  GPT-OSS-20B magnitudes are overstated.

## Caveats
- Measured on H100; paper reports Blackwell for GPT-OSS. Exact ms are H100-specific, but
  the methodology errors are GPU-independent (and a faster GPU makes the overstatement worse).
- The **centroid was not measured** here (needs retrain). This corrects the baseline only.

## To make the claim publishable
Re-measure synthetic/centroid with the retrained centroid + the same warmup/true-cold
harness, >=5 repeats with error bars, on the reported GPU. speedup = 120 ms / (warmed
centroid turn-1). Prediction (not a measurement): well below 4.3x, possibly <1x at turn-1
on H100.

## Code changes made to enable this (uncommitted)
`agentcache_compression/multi_turn_benchmark.py`:
- `generate_turn_output`: temperature 0.0 -> 0.7, top_p 0.9, seed 0 (stops GPT-OSS
  analysis-channel loops; TTFT-measurement calls left greedy/max_tokens=1).
- `build_server_cmd`: cold mode now passes `--no-enable-prefix-caching` (true cold).
- `main`: a generic ~2k-token warmup request before the measured turns.
Run env: `VLLM_USE_DEEP_GEMM=0` (outdated deep_gemm crashes warmup), `--max-model-len 32768`.
