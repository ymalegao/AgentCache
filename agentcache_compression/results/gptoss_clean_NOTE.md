# GPT-OSS-20B cold/warm rerun — note (updated after clean rerun)

**Date:** 2026-06-08 · **vLLM:** 0.20.0 · **GPU:** H100 80GB
**Run config:** `--max-tokens 2048 --gpu-mem 0.90 --max-model-len 32768`, `VLLM_USE_DEEP_GEMM=0`.
Files:
- `gptoss_clean.jsonl` — first rerun, **degenerate** turns 3–10 (greedy temp=0 loops). Kept for comparison.
- `gptoss_clean2.jsonl` — **clean** rerun after the sampling fix below. Use this one.

## Infra/code changes required to run at all (none affect TTFT methodology)
1. `VLLM_USE_DEEP_GEMM=0` — installed `deep_gemm` is outdated and crashes kernel-warmup; disabling it falls back to standard FP8 kernels.
2. `--max-model-len` 16384 → 32768 — clean responses grow ~2k tok/turn and 16384 overflowed at turn 9.
3. **`generate_turn_output` sampling** changed from `temperature=0.0` to `temperature=0.7, top_p=0.9, seed=0` (`multi_turn_benchmark.py`). The TTFT-measurement calls (`measure_turn_ttft_and_cached`, greedy/`max_tokens=1`) were left unchanged. Greedy decoding made GPT-OSS loop forever in the Harmony `analysis` channel and never emit the `final` channel.

## Cleanliness
- `gptoss_clean.jsonl` (first rerun): the provided success check says SUCCESS but it's a **false positive** — its `marker` test only catches the literal `assistantfinal`/`final` prefix, not analysis-channel-only or repetition. Turns 3–10 are analysis-channel repetition loops with **no Python**.
- `gptoss_clean2.jsonl` (after the sampling fix): **PASS** under a stricter check (real code in all 20, no analysis-leak, no repetition). Responses are task-appropriate (turn 8 = pytest tests, turn 10 = pyproject.toml). The `strip_to_final_channel` fix works correctly once the model actually reaches the final channel.

## Cold-mode comparison — OLD (corrupted) vs DEG (temp0 loops) vs CLEAN (temp0.7)
| turn | OLD ttft | DEG ttft | CLEAN ttft | OLD ptok | DEG ptok | CLEAN ptok |
|----:|----:|----:|----:|----:|----:|----:|
| 1 | 1.078 | 2.284 | 0.336 | 2299 | 2299 | 2299 |
| 2 | 0.234 | 0.079 | 0.077 | 3626 | 3291 | 3075 |
| 5 | 0.513 | 0.204 | 1.169 | 8910 | 8757 | 6676 |
| 10 | 0.666 | 0.324 | 0.738 | 18691 | 19238 | 13955 |

- **mean cold TTFT t2–10:** OLD 0.474 · DEG 0.396 · CLEAN 0.517 s
- **cumulative prompt by turn 10:** OLD 18691 ≈ DEG 19238 ≫ **CLEAN 13955** (clean ≈ **27% smaller**)
- turn-1 (identical 2299-tok prompt): 1.078 / 2.284 / 0.336 s across runs → compile/noise, **not** a usable signal.

## Findings
1. **Format corruption did NOT inflate the cold baseline** (the task's literal question → **Case A**). OLD vs DEG prompt sizes match within ±3%; responses cap at ~2048 tokens regardless of content quality.
2. **Decoding degeneration DOES inflate cumulative context by ~27%** (clean responses are shorter/variable; degenerate ones fill the 2048 cap). This was present in BOTH the old run (garbage) and the first rerun (analysis loops).
3. **…but that 27% does NOT show up in measured TTFT** (clean mean is even slightly higher; per-turn TTFT is noisy single-sample and non-monotonic in prompt size). Reason → finding 4.
4. **"cold" mode is not actually cold.** vLLM 0.20.0 (V1) enables prefix caching **by default** (`config/cache.py:91 = True`; CLI default `None` → default-on; only force-disabled on RISC-V CPUs). The benchmark omits `--enable-prefix-caching` for cold expecting no cache, but gets it anyway. Proof: cold and warm_apc have **byte-identical KV-hit curves** (0.744/0.758/0.767/0.788) and ~1.0× warm-vs-cold speedup. So per-turn TTFT only prefills the small uncached suffix (the new user message) and is largely independent of total history size — which is why the 27% size difference is invisible in TTFT.

## Implications for the paper's 1.80× synthetic-vs-cold speedup
- The corruption was a **red herring** for the timing question: it did not inflate the cold baseline.
- The **real** issue is finding 4: the "cold" baseline is prefix-cache-accelerated (cold ≡ warm_apc in this vLLM), so a synthetic-vs-cold speedup is measured against an already-fast baseline. This likely affects the 1.80× far more than the corruption.
- **To re-measure correctly:** (a) launch cold mode with prefix caching explicitly disabled (`--no-enable-prefix-caching`) to get a true cold baseline; (b) use the clean (temp>0) generation; (c) retrain the centroid for synthetic mode (separate task). Until (a)+(c), the 1.80× should be treated as unverified — not because of the channel corruption, but because cold isn't cold.
