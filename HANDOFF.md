# AgentCache Centroid Injection — Handoff

**Date:** 2026-05-18  
**Status:** vLLM injection path is wired and debuggable. Export bug fixed. Llama uses **no dummy prompt padding** (same layout as Qwen). **64-token retrain validated** on the standard `test_injection.py` prompt (~149 tokens): inject output is task-relevant (timing / `time` library); not yet bit-identical to cold.

Use this doc when consulting on architecture: what works today, what we dropped, and what still needs a product decision.

---

## Pipeline

```
prefixtraining.py  →  transpose_tensors.py  →  test_injection.py
```

| Stage | Role |
|-------|------|
| `prefixtraining.py` | PEFT prefix tuning on Llama chat-template prompts |
| `transpose_tensors.py` | Export `centroid_K.npy` / `centroid_V.npy` for vLLM |
| `test_injection.py` | Cold vs inject TTFT + output sanity check |

**Runtime (installed vLLM, not the `vllm/` source tree):**

| File | Path |
|------|------|
| Injector | `vllm-env/lib/python3.10/site-packages/vllm/centroid_injector.py` |
| Integration | `vllm-env/lib/python3.10/site-packages/vllm/centroid_integration.py` |
| Runner hook | `vllm-env/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py` |

**Artifacts today**

- Model: `/mnt/g/agentcache/models/Llama-3.2-1B-Instruct`
- Adapter: `agentcache_prefix_model/` (`num_virtual_tokens: 64`, `prefix_projection: true`)
- Centroids: `centroid_K.npy`, `centroid_V.npy` → `[16, 64, 512]` (layers × virtual tokens × kv_dim)
- Export: `python transpose_tensors.py --sys-tokens 0` → `sys_prefix_num_tokens.txt` = `0`

---

## Current architecture (no dummy padding)

### Prompt layout

Cold and inject use the **same** prompt string:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

There is **no** `dummy_ids + physical_ids` prepend. That Llama-only workaround was removed; Qwen never needed it.

### Scheduler gap

```text
gap = VLLM_CENTROID_SYS_TOKENS + centroid_len
```

With pure PEFT (`VLLM_CENTROID_SYS_TOKENS=0`):

- Centroid KV is **seeded** into cache slots for logical positions `0 .. centroid_len-1`
- The scheduler **skips** computing those positions during prefill
- Real prompt tokens are computed starting at position `centroid_len` (same token IDs as cold, different RoPE positions)

Example (long prompt, gap 256):

```text
prompt_tokens ≈ 750
gap = 256
prefill computes positions 256..749  →  ~493 tokens vs ~750 cold  →  TTFT win possible
```

Example (short prompt, gap 256):

```text
prompt_tokens ≈ 149
tokens_after_gap = 149 - 256 < 0  →  test_injection.py aborts (by design)
```

**Constraint:** `prompt_tokens` must be **greater than** `gap + margin` (test uses `MIN_TOKENS_AFTER_GAP=32`). Otherwise output is fast but meaningless.

### Export (critical)

For `prefix_projection: true`, centroids **must** be exported via `PeftModel.get_prompt()` in `transpose_tensors.py`. Manual `PrefixEncoder` reconstruction had been wrong (large max diff vs runtime cache); that caused garbled inject output until fixed.

```bash
python transpose_tensors.py --sys-tokens 0   # pure PEFT, gap = centroid_len only
```

Writes `sys_prefix_num_tokens.txt` beside the `.npy` files (overridable with `VLLM_CENTROID_SYS_TOKENS`).

### Env vars (typical test)

| Variable | Typical value | Meaning |
|----------|---------------|---------|
| `VLLM_CENTROID_SCHEDULER` | `1` (inject) / `0` (cold) | Enable skip + seed path |
| `VLLM_CENTROID_SYS_TOKENS` | `0` | No separate sys_K tensor; gap = centroid_len |
| `VLLM_CENTROID_K_PATH` / `V_PATH` | `centroid_K.npy`, `centroid_V.npy` | Injected tensors |
| `CENTROID_PERF_DEBUG` | `1` in dev | Extra engine logs (adds overhead) |

---

## What worked on Qwen vs Llama port

| | Qwen-1.5B | Llama-3.2-1B (current) |
|--|-----------|-------------------------|
| Dummy prompt padding | Never used | **Removed** (was a dead-end for TTFT) |
| Export | Worked | Needed `get_prompt()` export fix |
| Coherent inject | Yes | Yes — **64-token** on-topic on ~149-token test prompt |
| TTFT speedup | ~1.2× (long prompt) | ~1.0× on 149-token prompt (64 gap); measure on long prompt next |

Reference Qwen adapters: `qwen15_64/` (64 virtual tokens), `qwen15_256/` (256).

---

## Validation results (Llama, `test_injection.py`)

Test prompt: `apply_chat_template` with agent system + user ask  
`"Write a Python context manager that times how long a code block takes to execute."`  
(~149 prompt tokens, `VLLM_CENTROID_SYS_TOKENS=0`)

| N (trained) | Gap | Prefill computed | TTFT cold / inject | Task quality (inject) |
|-------------|-----|------------------|--------------------|------------------------|
| **64** | 64 | 85 tokens (`positions 64..149`, `start_matches=True`) | 0.0277s / 0.0285s (~0.97×) | **Good** — timing code, `time` library, on-topic |
| **96** | 96 | 54 tokens | 0.0297s / 0.0255s (~1.16×) | **Bad** — `" of-1-1"`, wrong “Environment and Resource Management” prose |
| **256** (old) | 256 | fails guard (`149 - 256 < 0`) | — | — |

### 64-token run (2026-05-18) — reference outputs

**Cold** (full prefill):

```text
Here's a Python context manager that times how long a code block takes to execute:

```python
import time
...
class Timer:
    def __enter__(self): ...
    def __exit__(self, ...): ...
```

**Inject** (centroid at `0..63`, gap 64):

```text
**Timing Function**
...
This is a Python script that uses the `time` library to time the execution of a code block.
...
def timing_script(func):
    start_time = time.time()
    ...
```

**Verdict:** Inject **answers the question** in spirit (timing + `time`), but uses a function wrapper rather than a proper `contextmanager` / `__enter__`/`__exit__` class like cold. Treat as **quality pass, not parity pass**. The script’s `coherent ✓` heuristic is insufficient — use task checks (e.g. mentions `time`, timing, context/block).

**Plumbing (64):** `total_synthetic_len=64`, `wrote_any=True`, prefill `n_scheduled_tokens=85`, `positions_minmax=(64, 149) expected_start=64 start_matches=True`.

### 96-token run — do not use for this prompt length

Same test setup; inject was faster (~1.16×) but **wrong task**. Confirms larger N is not automatically better when `prompt_len ≈ 150` and the prefix must substitute for most of the chat header + system block.

---

## What is confirmed working

- **64-token** adapter trained and exported (`centroid_K.npy` shape `[16, 64, 512]`).
- `transpose_tensors.py` export aligned with PEFT runtime (`PeftModel.get_prompt()`).
- Injector: `wrote_any=True`, seed into block table for positions `0..N-1`.
- Scheduler: `n_scheduled_tokens ≈ prompt_len - gap` on prefill when `start_matches` / gap aligned.
- `test_injection.py`: chat template matches training; guard on `prompt_len - gap`.
- Cold output coherent; **64-token inject** on-topic for the context-manager timing test (see table above).

---

## Known limitations (not blockers for arch review)

1. **Heuristic “coherent ✓”** in `test_injection.py` only checks word-like text, not task correctness.
2. **TTFT benchmark noise:** each `generate()` gets a new request id → re-seed every trial; cold vs inject use **two** `LLM()` lifetimes (two loads). Use `CENTROID_PERF_DEBUG=0` and longer prompts for fair perf.
3. **Slice ≠ train:** Taking `K[:, :64]` from a 256-token adapter is **not** equivalent to training `num_virtual_tokens=64`. Slicing was removed from the test script; retrain instead.
4. **RoPE / HF vs vLLM:** Injector currently writes centroid K/V without re-applying RoPE in some builds; coherent output after export fix suggests the remaining mismatch is secondary for Llama, but worth validating if quality regresses.

---

## Decisions needed (for architecture consult)

### 1. Virtual token count N

| Option | Pros | Cons |
|--------|------|------|
| **N = 64** ✅ (current Llama adapter) | Fits ~150-token test prompt; inject on-topic; gap 64 → 85 tokens computed | Not identical to cold; TTFT ~1.0× on short prompt (seed overhead) |
| **N = 96** | — | **Failed** task quality on same prompt despite correct plumbing |
| **N = 256** | More capacity for long system prompts | Unusable when `prompt_len ≈ 150` (guard aborts) |

**Recommendation (updated):** Stay on **N = 64** for agent-style ~150–500 token prompts. Re-evaluate N only with a longer production system prompt and a real eval set (HF `PeftModel.generate` vs vLLM inject parity).

### 2. `sys_prefix_num_tokens` / hybrid layout

- **`--sys-tokens 0` (pure PEFT):** gap = N, centroid at positions `0..N-1`. Simplest; what we use now.
- **`--sys-tokens 1` (hybrid):** gap = 1 + N, optional separate handling for BOS at 0. Used in some older docs; not required if pure PEFT works.

Pick one convention and keep training export, sidecar, and `VLLM_CENTROID_SYS_TOKENS` aligned.

### 3. Prompt length vs product

Injection saves prefill only on tokens **after** the gap:

```text
computed_tokens ≈ prompt_len - gap
speedup ∝ gap / prompt_len   (roughly, minus seed overhead)
```

Product question: Is the agent system prompt always long enough (e.g. 500+ tokens) that `N=64` or `N=256` is worth it? If typical prompts are ~150 tokens, **N must be ≪ prompt_len** (e.g. 64, not 256).

### 4. Single engine / request reuse for production

Today’s test re-seeds per request. Production should either:

- Reuse request id / session so seed runs once, or
- Amortize seed cost across multi-turn traffic.

### 5. Quality bar

Define success beyond TTFT: same answer as cold on a fixed eval set, or acceptable domain prior (system behavior) with cheaper prefill.

---

## Model / tensor shapes

**Llama-3.2-1B-Instruct:** 16 layers, 8 KV heads, head_dim 64 → `kv_dim = 512`.

**PEFT config (`adapter_config.json`):**

```json
{
  "num_virtual_tokens": 64,
  "prefix_projection": true,
  "num_attention_heads": 8,
  "num_layers": 16,
  "token_dim": 512
}
```

**vLLM centroids:** `[num_layers, num_virtual_tokens, token_dim]` → `[16, N, 512]`.

**vLLM KV cache (v1):** `[2, num_blocks, block_size, num_kv_heads, head_dim]`.

---

## Retrain checklist (64-token path) — done 2026-05-18

1. ~~`prefixtraining.py`: `NUM_VIRTUAL_TOKENS = 64`~~
2. ~~Train → `agentcache_prefix_model/` (final loss ~0.79 over 8 epochs)~~
3. ~~`python transpose_tensors.py --sys-tokens 0`~~
4. ~~Verify: `centroid_K.npy` → `[16, 64, 512]`~~
5. ~~`VLLM_CENTROID_SYS_TOKENS=0 python test_injection.py`~~ — inject on-topic; cold/inject TTFT ~equal on 149-token prompt
6. **Next:** longer prompt TTFT benchmark; HF adapter `generate()` parity check; stricter task eval in `test_injection.py`

---

## Quick commands

```bash
cd /home/yash/agentcache
source vllm-env/bin/activate

python test_injection.py

# Centroid stats
python -c "
import numpy as np
K = np.load('centroid_K.npy')
print('shape', K.shape, 'std layer0', K[0].std())
"

# Engine logs
CENTROID_PERF_DEBUG=1 python test_injection.py 2>&1 | grep -E 'CENTROID PERF|CENTROID TIMING|CENTROID\]'
```

---

## Files touched recently

| File | Change |
|------|--------|
| `test_injection.py` | Chat template prompt; no dummy padding; no centroid slicing; gap guard |
| `transpose_tensors.py` | Export via `PeftModel.get_prompt()` when `prefix_projection=true` |
| `centroid_injector.py` / `centroid_integration.py` | Perf logs, seed/skip diagnostics |
| `HANDOFF.md` | This doc (padding removed; arch questions for consult) |

**Deprecated approach (do not revive):** `inject_ids = [bos] * N + physical_ids` dummy prepend for Llama. It inflated `prompt_len` without reducing computed tokens vs cold and masked scheduler bugs.
