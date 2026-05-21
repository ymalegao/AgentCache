# AgentCache Compression Experiment — Session Notes

**Date:** 2026-05-20  
**Status:** Phases 1-2-4-5(N64)-6(N64)-7-8 done. Phase 9 (eval results) collected. Open: GOODBYE 0%

---

## What is this experiment

Show that synthetic KV (PEFT prefix, compression mode) lets cold requests match warm-APC TTFT
while keeping coherent output.

**Baseline already measured (benchmark_ttft.py):**
- Cold start: ~0.065s
- Warm APC: ~0.027s
- Inject replacement (N=64): ~0.029s

**Goal:** compression mode cold TTFT < cold start, output coherent, GOODBYE preserved.

**Metrics:** TTFT (primary) · Coherence (>20 words, no degeneration) · GOODBYE pass rate

---

## Files created so far

| File | Purpose |
|------|---------|
| `experiments/agentcache_compression/prompts/python_agent_system.txt` | System prompt with GOODBYE rule used for training |
| `experiments/agentcache_compression/prepare_data.py` | One-shot script: filter + split vllm_good_examples_raw.jsonl |
| `experiments/agentcache_compression/data/python_agent_train.jsonl` | 118 Python train examples (GOODBYE appended to teacher_output) |
| `experiments/agentcache_compression/data/python_agent_eval.jsonl` | 25 held-out Python eval examples with must_include_any checks |
| `experiments/agentcache_compression/train_prefix_compression.py` | New training script: CLI args + proper label masking |
| `experiments/agentcache_compression/adapters/N64_sys0/` | Trained N=64 adapter (8 epochs, final loss ~0.64, train_loss 0.90) |
| `experiments/agentcache_compression/centroids/N64_K.npy` | Exported centroid K, shape [16, 64, 512] |
| `experiments/agentcache_compression/centroids/N64_V.npy` | Exported centroid V, shape [16, 64, 512] |

**Data stats:** 175 raw → 143 Python-only (32 excluded as bash/Node.js) → 118 train / 25 eval.

---

## Architecture decisions

### Compression mode (vs replacement mode)

**Replacement (current):** inject N synthetic KV at positions 0..N-1, skip N tokens in prefill.
Physical prompt still includes system+user. System tokens at the front get silently skipped.

**Compression (target):** inject N synthetic KV at positions 0..N-1, compute ALL M user tokens.
Physical prompt = user only (no system). Position IDs for user tokens start at N.
Scheduled prefill tokens = M (not M-N).

### Compression mode: prompt construction (no vLLM changes needed)

The correct approach uses the existing gap mechanism — no vLLM scheduler or model runner
changes required. The test harness constructs the prompt as:

```
prompt_token_ids = [pad_token_id] * N  +  user_chat_token_ids
```

With `VLLM_CENTROID_SYS_TOKENS=0`, `centroid_sched_gap` returns N (= centroid_len).
vLLM then treats N tokens as "pre-computed" and schedules all M user tokens normally:

| Property | Result |
|----------|--------|
| `n_scheduled_tokens` | M (all user tokens) |
| `positions` | N..N+M-1 (from num_computed=N + query_pos) |
| `seq_lens` | N+M |
| KV slots for user tokens | N..N+M-1 (disjoint from centroid 0..N-1) |
| Block allocation | N+M (scheduler sees N+M token prompt) |

The N pad tokens are **never computed** — they are only in the token ID array for
accounting. This is safe because the gap mechanism skips them entirely; the old
BOS-prepend corruption was from pad tokens that WERE computed by the model.

### Training: proper label masking

Current `prefixtraining.py` has no label masking (computes loss on system+user+assistant).
New script will mask system+user tokens (-100) and compute loss on assistant tokens only.

---

## Phase history

1. ~~Phase 1-2: system prompt + data split~~ ✅
2. ~~Phase 4: train_prefix_compression.py with label masking~~ ✅
3. ~~Phase 5: N=64 adapter trained (8 epochs, loss 0.64)~~ ✅
4. ~~Phase 6: centroids exported [16, 64, 512]~~ ✅

5. ~~**Phase 7 — compression mode**~~ ✅ (2026-05-19, corrected same session)
   Final state of vLLM files:
   - `centroid_injector.py` — `self.layout` attribute + layout logged in init. Seed logic unchanged.
   - `centroid_integration.py` — added `centroid_layout()` function (mode marker only).
     `centroid_sched_gap()` is UNCHANGED — gap runs normally in compression mode.
   - `gpu_model_runner.py` — NO changes. Position/slot geometry is correct via the
     existing gap mechanism once the prompt is N+M tokens long.
   
   Source repo synced. Smoke-test: look for `[CENTROID] sched gap=64` in logs
   (confirms gap is active) and `positions_minmax=(64, 64+M-1)` in CENTROID_PERF_DEBUG.

6. ~~**Phase 8 — test_compression.py**~~ ✅ (2026-05-19)
   `experiments/agentcache_compression/test_compression.py`
   Three modes: `cold_no_synthetic` · `warm_apc` · `synthetic_compression`
   All args have defaults; only `--mode` is required.
   ```bash
   python experiments/agentcache_compression/test_compression.py --mode synthetic_compression
   ```
   Output JSONL: `results/N64_comparison.jsonl` (appended per mode run).

7. ~~**Phase 9 — eval run on 25 tasks**~~ ✅ (2026-05-20, partial — warm_apc may be incomplete)
   Results in `results/N64_comparison.jsonl`. Summary:

   | Mode | N tasks | Mean TTFT (s) | Coherent | GOODBYE | Task-check pass |
   |------|---------|--------------|----------|---------|----------------|
   | synthetic_compression | 25 | ~0.018 | 100% | **0%** | 84% (21/25) |
   | cold_no_synthetic | 25 | ~0.017 | 100% | **0%** | 88% (22/25) |
   | warm_apc | partial | ~0.014 | 100% | **0%** | — |

   Note: TTFT measured with `max_tokens=1`, 3 runs/task. First task in each mode
   has GPU-init overhead (0.04–0.14s first run) — excluded from steady-state mean.

   **Key findings:**
   - compression TTFT (~18ms) ≈ cold TTFT (~17ms) — no speedup observed.
     Likely because eval prompts are very short (47–68 user tokens). The 64-token
     skip saves little compute when the total user prompt is only ~50 tokens.
   - GOODBYE: 0% in ALL modes, including cold+system prompt. The 1B model
     doesn't reliably follow end-with-GOODBYE instruction at 256 max tokens.
   - Coherence: 100% across all modes — centroid injection doesn't corrupt output.
   - Task-check pass rates are comparable (84% vs 88%) — slight degradation in
     compression mode, within noise for 25 samples.

8. **Phase 5b — N=256 adapter** (train after N=64 results look good)
   Same command, --num-virtual-tokens 256 --output adapters/N256_sys0

---

## Next steps (updated 2026-05-20)



3. **Investigate TTFT gap with original benchmark**.
   Old `benchmark_ttft.py` showed cold=0.065s, inject=0.029s (2.2x speedup) with
   Qwen-1.5B and a ~600-token system prompt. The new eval has 150-token prompts and
   Llama-3.2-1B — the short prompts make GPU-kernel overhead dominate, masking speedup.
   To see real compression benefit: test with a long system prompt (500+ tokens) or
   increase N to 256 to skip more tokens.

4. **GOODBYE analysis** — check if the issue is model size or prompt format.
   - Try bumping max_tokens to 512 (model may need more space before "GOODBYE").
   - Compare against a single hand-crafted prompt that definitely ends with GOODBYE
     (verify the model CAN output it before concluding training failed).

5. **Phase 5b — N=256 adapter** — if/when short-prompt TTFT is validated.

---

## Open questions / things to watch

- `infer_checks()` in prepare_data.py is heuristic — some eval entries may get only the
  fallback check `["def ", "import ", "class "]`. Worth reviewing.
- No MAX_LEN: dynamic padding per batch. All 118/118 training examples have GOODBYE.
  Longest sequence is 1420 tokens — fine for 1B model, watch VRAM if scaling up.
- N=256 adapter requires 256 prefill positions skipped → physical prompt must be >256 tokens.
  Short user queries (~40 tokens) won't benefit from N=256. Worth noting in results.

### Phase 7 open questions — all resolved ✅

All four questions had the same root cause: the first Phase 7 implementation only
offset RoPE positions but left slot mapping and seq_len unchanged, so user KV
overwrote the centroid and attention saw the wrong context length.

**Resolution:** use `[pad]*N + user_chat_ids` as the prompt. The gap mechanism
(gap=N) handles all four properties correctly — no vLLM code changes needed.
The corrected Phase 7 + Phase 8 test harness implement this.
