# vLLM Prefix-Tuned KV Injection

This document describes the current architecture for injecting learned domain priors into vLLM's KV cache using a trained PEFT prefix adapter.

---

## Pipeline Overview

```
prefixtraining.py  →  transpose_tensors.py  →  inject (vLLM with env vars)
```

### Step 1 — Train the Prefix Adapter (`prefixtraining.py`)

Runs PEFT prefix tuning on `good_examples/good_examples.jsonl` against the frozen base model (`qwen-1.5b`).

**Key settings:**
- `NUM_VIRTUAL_TOKENS = 256` — virtual prefix length (kept compact to avoid softmax dilution)
- `prefix_projection = True` — uses a 2-layer MLP to project embeddings; stabilizes training
- `LEARNING_RATE = 5e-3` — higher than full fine-tuning, appropriate for prefix tuning
- `bf16 = True`, base model frozen

Output: `./agentcache_prefix_model/` (adapter weights + config in safetensors format)

---

### Step 2 — Export to KV Tensors (`transpose_tensors.py`)

Materializes the trained PEFT adapter into raw K and V tensors suitable for direct KV cache injection.

**What it does:**
1. Loads `adapter_config.json` to read `num_virtual_tokens`, `num_layers`, `token_dim`
2. Because `prefix_projection=True`, reconstructs KV tensors by running the PrefixEncoder MLP forward pass on token indices `[0..N-1]`
3. Permutes `[N, L, 2, d] → [2, L, N, d]` and splits into K and V
4. Saves `centroid_K.npy` and `centroid_V.npy` with shape `[num_layers, num_virtual_tokens, kv_dim]`
5. Writes `sys_prefix_num_tokens.txt` sidecar (default `1` = hybrid BOS-at-0 layout)

**Usage:**
```bash
python transpose_tensors.py \
    --adapter agentcache_prefix_model \
    --out-k centroid_K.npy \
    --out-v centroid_V.npy \
    --sys-tokens 1
```

---

### Step 3 — Inject (vLLM)

Start vLLM with the centroid env vars set. The modified runner and scheduler files handle the rest automatically.

**Minimum env vars:**
```bash
export VLLM_CENTROID_K_PATH=/home/yash/agentcache/centroid_K.npy
export VLLM_CENTROID_V_PATH=/home/yash/agentcache/centroid_V.npy
export VLLM_CENTROID_SCHEDULER=1
```

**Optional vars:**
| Variable | Default | Purpose |
|---|---|---|
| `VLLM_CENTROID_SYS_TOKENS` | reads sidecar txt | Override sys prefix token count |
| `VLLM_EXACT_SYS_K_PATH` / `_V_PATH` | unset | Inject exact system prompt KV alongside PEFT prefix |
| `VLLM_CENTROID_USE_LMCACHE` | `0` | Skip exact sys injection (LMCache handles it) |
| `VLLM_CENTROID_SINK_BLEND` | `0.35` | Blend factor for attention-sink preservation at first centroid slot |
| `CENTROID_DEBUG` | `0` | Log KV write readback verification |
| `CENTROID_DEBUG_ROPE` | `0` | Verbose RoPE position logs (disable for benchmarks) |
| `CENTROID_TIMING` | `0` | Wall-clock timing per injection call (`1` or `cuda`) |

---

## Modified vLLM Files

### `vllm/centroid_injector.py`

The `CentroidInjector` class. Loads `centroid_K.npy` / `centroid_V.npy` at startup and exposes `seed_prefix_into_kv_cache(...)`.

**Injection logic:**
1. Reads `sys_token_count` from sidecar or env
2. If `VLLM_EXACT_SYS_K_PATH` is set: writes exact system prompt KV into positions `0..M-1` with RoPE applied at those positions
3. Writes PEFT centroid KV into positions `M..M+N-1` with RoPE applied at offsets `M..M+N-1`
4. Optionally blends the first centroid slot with an attention-sink template (`VLLM_CENTROID_SINK_BLEND`)
5. Tracks seeded request IDs — skips re-injection on subsequent chunked-prefill / decode steps (major GPU/host savings)

### `vllm/centroid_integration.py`

Shared hooks used by both model runner variants.

- `try_load_centroid_injector` — loads injector at runner startup
- `apply_centroid_block_table` — called in the runner's forward step; drives `seed_prefix_into_kv_cache`
- `centroid_sched_gap` — called in the scheduler to inflate `num_computed_tokens` so the engine treats the injected prefix as already computed
- `centroid_override_num_computed` — runner-side counterpart; adjusts `eff_num_computed` from 0 → `sys_token_count` (or `sys+centroid_len` in pure-PEFT mode)
- `try_get_rotary_emb_cached` — caches the first decoder layer's RoPE module on the runner to avoid repeated attribute traversal

### `vllm/v1/core/sched/scheduler.py`

Patched around the `num_computed_tokens` update path (line ~643):

```python
# centroid Path B: inflate external-computed count to cover injected prefix
from vllm.centroid_integration import centroid_sched_gap
_gap = centroid_sched_gap(num_prompt_tokens, base_computed)
```

### `vllm/v1/worker/gpu_model_runner.py` and `vllm/v1/worker/gpu/model_runner.py`

Both runners patched identically:

- **Startup:** `self._centroid_injector = try_load_centroid_injector(self.device)`
- **Per-step (Path B override):** `num_computed = centroid_override_num_computed(num_computed, self._centroid_injector)` when scheduler mode is active and `num_computed > 0`
- **KV write:** `apply_centroid_block_table(self, block_table_gid_0, num_reqs, input_batch)`

---

## Validation

From the last run, TTFT with centroid injection was **0.061s**. Output was verified coherent (context manager example completed correctly).

Known working layout: `sys_token_count=1` (BOS at slot 0), 256 virtual tokens following at slots 1–256.
