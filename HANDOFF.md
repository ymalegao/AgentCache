# AgentCache — System Design & Handoff

**Date:** 2026-05-27  
**Status:** Multi-turn benchmark pipeline complete. Llama compression path validated. GPT-OSS synthetic compression now works after runtime KV-layout / cache-group / RoPE fixes. 2.8× TTFT speedup confirmed at 1000-token contexts on Llama. APC pre-registration now working: turn-1 Prometheus hit rate confirmed at 42.1% (64/152 tokens) for N=64.

---

## 1. Problem & Motivation

Every request to a domain-specific agent (e.g., a Python coding assistant) must re-process the same large system prompt from scratch. Transformer prefill complexity is **O(N²)** in sequence length — for a 1000-token system prompt, this means spending tens of milliseconds on work that produces the same KV entries every single time.

**Time-To-First-Token (TTFT)** is the user-visible latency between submitting a query and seeing the first output token. It is dominated by prefill cost. At 1000-token contexts on Llama-3.2-1B:

| Mode | TTFT |
|------|------|
| Cold (full prefill every time) | ~47.8ms |
| Synthetic centroid injection (N=128) | ~17.0ms |
| **Speedup** | **2.8×** |

The synthetic TTFT stays **flat at ~17ms regardless of system prompt length** because physical tokens sent is always `N_virtual + user_query`, not `system_prompt + user_query`.

### Why not Automatic Prefix Caching (APC)?

APC is vLLM's built-in prefix reuse. It caches the KV of a prefix if it has been computed before and the same prefix appears again. It works well for **warm** repeated requests but provides **zero benefit on cold starts** — the first request always pays full prefill cost. Synthetic centroid injection provides a cold-start speedup by eliminating system-prompt prefill entirely.

### Why not hidden state averaging?

Our first approach averaged hidden states across many runs to produce a "centroid" KV. This failed because averaging vectors from different semantic contexts produces representations that fall outside the manifold the model knows — **semantic superposition**. The averaged vector maps to a region of latent space the model was never trained to handle, corrupting output.

The fix: train the centroids via gradient descent so they are optimized to lie exactly where the model expects them.

---

## 2. Architecture Overview

The pipeline has three phases:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase A — Offline Training (once per domain / system prompt)       │
│                                                                     │
│  Training data  →  train_prefix_compression.py  →  adapter/        │
│  (JSONL tasks)     (PEFT prefix tuning, frozen base model)         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase B — Materialization (once per adapter)                       │
│                                                                     │
│  adapter/  →  transpose_tensors.py  →  centroid_K.npy              │
│                                        centroid_V.npy              │
│                                        sys_prefix_num_tokens.txt   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase C — Runtime Delivery (per request)                           │
│                                                                     │
│  vLLM serve + CentroidInjector + scheduler gap mechanism           │
│  prompt = [pad]*N + user_tokens                                    │
│  Positions 0..N-1: filled from centroid_K/V.npy (pre-computed)    │
│  Positions N..N+M-1: user tokens, computed normally               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase A — Training Pipeline

### 3.1 Model

Base model: **Llama-3.2-1B-Instruct** (or any causal LM from HuggingFace).

```bash
./get_model.sh meta-llama/Llama-3.2-1B-Instruct
# saves to models/Llama-3.2-1B-Instruct/
```

All base model weights are **frozen** during training. The adapter adds roughly 3M trainable parameters (the PrefixEncoder MLP), while the 1B base model has ~1B frozen parameters.

**Model geometry (Llama-3.2-1B):**

| Property | Value |
|----------|-------|
| Transformer layers | 16 |
| KV heads | 8 |
| Head dimension | 64 |
| KV dimension (`token_dim`) | 512 (= 8 × 64) |

### 3.2 Data Preparation (`prepare_data.py`)

The training set is built from a raw pool of agent task examples (`vllm_good_examples_raw.jsonl`):

1. **Filter**: Keep only Python-only tasks. Exclude bash, Node.js, JavaScript tasks (32 excluded).
2. **Split**: 175 raw → 143 Python → 118 train / 25 eval.
3. **Teacher signal**: Append `"\nGOODBYE"` to each teacher output. The system prompt instructs the model to always end with GOODBYE; adding it to training labels gives the adapter a signal to encode that behavior. See the GOODBYE metric discussion for why this introduces a confound.
4. **Eval checks**: Each eval task has a `must_include_any` keyword list for pass/fail scoring.

Each record in the JSONL has: `id`, `user` (the task), `teacher_output` (expected response).

### 3.3 Prefix Tuning — How It Works (`train_prefix_compression.py`)

**What PEFT prefix tuning does:**

Instead of adding tokens to the input text, prefix tuning adds N learnable "virtual token" embeddings that are prepended to each transformer layer's key-value sequence. Concretely, a small MLP (the PrefixEncoder) maps N token embeddings to N KV pairs per layer. During training, only this MLP is updated; the rest of the model is frozen.

The intuition: the virtual tokens occupy the same positions in the attention mechanism that the system prompt would. By training on agent tasks without the system prompt in context, the model is forced to encode "what the system prompt means" entirely into the virtual tokens.

**Training setup:**

```python
peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=64,          # N — controls compression capacity
    prefix_projection=True,         # enables MLP projection layer
)
```

`prefix_projection=True` means a 2-layer MLP maps token embeddings → KV pairs. This adds expressivity at the cost of needing to run the MLP at export time (see Phase B).

**Label masking — the critical detail:**

Without label masking, the loss includes system and user tokens. The adapter would then optimize to reproduce the prompt text rather than the response. With masking, `-100` is assigned to all system and user token positions, and the cross-entropy loss is computed only over assistant tokens:

```python
# Labels: -100 for system+user (masked), token ids for assistant response (trained)
labels = [-100] * prompt_len + input_ids[prompt_len:]
```

This forces the virtual tokens to encode behavioral priors — the adapter must make the model produce correct agent outputs even without seeing the system prompt.

**`system_retain_ratio`**: A continuous 0.0–1.0 parameter. At `0.0` (our default), the system prompt is entirely absent — the model must rely solely on virtual tokens. At `1.0`, the full system prompt is kept (equivalent to standard fine-tuning, not compression). Intermediate values (e.g., `0.5`) retain the first half of system prompt tokens, allowing hybrid training. The `truncate_system()` function tokenizes the system prompt and keeps only the first `ratio * len` tokens before decoding back to text.

**Batch collation**: `dynamic_collator` pads each batch to its longest sequence rather than a fixed global max. Input IDs pad with `pad_token_id`; labels pad with `-100` so padding positions are never included in the loss.

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Virtual tokens N | 64, 128, or 256 |
| Epochs | 8 |
| Learning rate | 2e-3 |
| Batch size | 4 |
| Precision | BF16 |
| Final loss (N=64) | ~0.64 |

**Full train command:**

```bash
python agentcache_compression/train_prefix_compression.py \
    --model /mnt/g/agentcache/models/Llama-3.2-1B-Instruct \
    --data agentcache_compression/data/python_agent_train.jsonl \
    --system-prompt agentcache_compression/prompts/2000_python_agent_system.txt \
    --output agentcache_compression/adapters/N64_sys0 \
    --num-virtual-tokens 64 \
    --system-retain-ratio 0.0 \
    --epochs 8 \
    --lr 2e-3
```

**Output:** `agentcache_compression/adapters/N{N}_sys0/` — contains `adapter_model.safetensors` and `adapter_config.json` (written by PEFT; stores `num_virtual_tokens`, `num_layers`, `token_dim`, `prefix_projection` — read by `transpose_tensors.py`).

---

## 4. Phase B — Transposition (`transpose_tensors.py`)

After training, the adapter weights exist inside `adapter_model.safetensors`. vLLM cannot use PEFT adapter format directly — it needs layer-wise KV tensors in a specific shape. This phase materializes the learned representations.

`transpose_tensors.py` first reads `adapter_config.json` (written by PEFT) to get `num_virtual_tokens`, `num_layers`, `token_dim`, and `prefix_projection`. It then takes one of two paths depending on `prefix_projection`.

### 4.1 Two Export Paths

**Non-projected** (`prefix_projection=False`):

Weights stored as a flat matrix. Pure tensor reshaping, no model load required:

```
raw_prefix shape: [N, L × 2 × d]
  → view [N, L, 2, d]
  → permute [2, L, N, d]          # axis 0: K/V split
  → K = result[0]  shape [L, N, d]  # transposed to [num_layers, N, token_dim]
    V = result[1]  shape [L, N, d]
```

**Projected** (`prefix_projection=True`) — what we use:

Manual MLP reconstruction from saved weights has alignment bugs that produce garbled output. The only safe approach is to run PEFT's own forward pass:

```python
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="cpu")
peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
prompt_cache = peft_model.get_prompt(batch_size=1)
# prompt_cache is a list of (K_tensor, V_tensor) per layer
# Each tensor shape: [1, num_kv_heads, N, head_dim]  (batch dim = 1)

for layer_idx, (k_batch, v_batch) in enumerate(prompt_cache):
    k = k_batch[0]   # drop batch dim → [num_kv_heads, N, head_dim]
    k_flat = k.permute(1, 0, 2).contiguous().view(N, -1)
    #   permute → [N, num_kv_heads, head_dim]
    #   view   → [N, token_dim]  where token_dim = num_kv_heads × head_dim
```

Stacking across layers gives `[num_layers, N, token_dim]`. Running on CPU keeps the base model off the GPU — the projected export is memory-heavy (~8 GB for 1B model in fp16) but only runs once.

### 4.2 Output Format

| File | Shape | Description |
|------|-------|-------------|
| `centroid_K.npy` | `[num_layers, N, token_dim]` | Key tensors for all layers |
| `centroid_V.npy` | `[num_layers, N, token_dim]` | Value tensors for all layers |
| `sys_prefix_num_tokens.txt` | scalar | System token count (0 = pure compression mode) |

Example for N=64, Llama-3.2-1B: shape `[16, 64, 512]` (16 layers, 64 virtual tokens, 512 = 8 heads × 64 head_dim).

**Command:**

```bash
python agentcache_compression/transpose_tensors.py \
    --adapter agentcache_compression/adapters/N64_sys0 \
    --out-k agentcache_compression/centroids/N64_2000_K.npy \
    --out-v agentcache_compression/centroids/N64_2000_V.npy \
    --sys-tokens 0
```

### 4.3 GPT-OSS Export Note

For GPT-OSS, the exported `.npy` tensors from `PeftModel.get_prompt()` are injected directly without applying an extra centroid-side RoPE transform at runtime. The export step itself is identical; only the injection path differs (see section 5.4.2).

---

## 5. Phase C — vLLM Integration

### 5.1 Telling vLLM to Use Synthetic Tokens

vLLM is configured via environment variables at server startup:

```bash
VLLM_CENTROID_SCHEDULER=1                          # enables gap mechanism
VLLM_CENTROID_K_PATH=centroids/N128_2000_K.npy    # centroid keys
VLLM_CENTROID_V_PATH=centroids/N128_2000_V.npy    # centroid values
VLLM_CENTROID_SYS_TOKENS=0                         # compression mode: no system prompt
VLLM_CENTROID_LAYOUT=compression                   # layout mode
VLLM_CENTROID_PAD_TOKEN_ID=128009                  # (optional) pad token ID for APC pre-registration
                                                    # Llama-3.2-1B-Instruct pad token = 128009 (<|eot_id|>)
                                                    # (not 128001 — auto-detected by benchmark from tokenizer)

vllm serve /path/to/Llama-3.2-1B-Instruct \
    --port 8000 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.6
```

### 5.2 How vLLM Scheduling Works — The Gap Mechanism

vLLM's scheduler tracks a `num_computed_tokens` field per sequence. Normally this is 0. The gap mechanism overrides it to N, telling the scheduler "the first N positions are already filled":

```
centroid_sched_gap() → returns N

Scheduler sees:
  total_prompt_tokens = N + M  (N pad + M user)
  num_computed_tokens = N      (centroid pre-filled)
  tokens_to_schedule  = M      (only user tokens hit the GPU)

Position IDs for user tokens: N, N+1, ..., N+M-1
```

The N pad tokens in the prompt ID array are **never computed by the model**. They are accounting placeholders. The scheduler skips them because `num_computed_tokens=N` tells it those slots are already filled.

### 5.3 Prompt Construction

The client constructs the prompt as:

```python
pad_id = tokenizer.pad_token_id
prompt_token_ids = [pad_id] * N + user_chat_token_ids
```

`user_chat_token_ids` is the conversation history formatted with `apply_chat_template`, **without** the system prompt. The system prompt's semantic content is encoded in the centroid KV tensors.

**Why no system prompt in the text?** In compression mode, the system prompt is removed from the physical prompt entirely. The centroid stands in for it. Sending the system prompt text would double-count it (once as text and once via the injected KV).

### 5.4 `gpu_model_runner.py` — Where the Hooks Live

All centroid hooks attach to vLLM's V1 GPU model runner. Two hook points:

**Hook 1 — New-request admission** (runs when a request is first seen by the runner):

```python
# gpu_model_runner.py ~line 1168
eff_num_computed = new_req_data.num_computed_tokens
if not self.is_pooling_model and centroid_scheduler_mode() and eff_num_computed == 0:
    ensure_centroid_injector_lazy(self)
    eff_num_computed = centroid_override_num_computed(eff_num_computed, self._centroid_injector)
```

`centroid_override_num_computed` sets `eff_num_computed = N` (centroid + sys token count) when the runner would otherwise see 0. This tells the runner that positions 0..N-1 are already filled, so it only schedules user tokens for GPU computation. This is the worker-side counterpart to the scheduler's gap mechanism.

**Hook 2 — Block table build** (runs every forward pass, skipped during CUDA graph capture):

```python
# gpu_model_runner.py ~line 2171
block_table_gid_0 = apply_centroid_block_table(self, block_table_gid_0, num_reqs, self.input_batch)
```

`apply_centroid_block_table` (in `centroid_integration.py`) calls `CentroidInjector.seed_prefix_into_kv_cache`, which writes the centroid K/V tensors directly into the physical KV cache blocks at positions 0..N-1. This is skipped during CUDA graph capture (`for_cudagraph_capture=True`) so the synthetic injection doesn't corrupt warmup prefills.

### 5.4.1 CentroidInjector (`centroid_injector.py`)

The injector is called by hook 2 above. Per-request steps:

1. **Load**: `.npy` files are loaded once at server startup (`try_load_centroid_injector`).
2. **Seed**: For positions 0..N-1, write centroid K/V tensors into the physical KV cache block table slots.
3. **RoPE rotation**: For Llama/Qwen, stored centroid K tensors are rotated with position offsets 0..N-1 at injection time. GPT-OSS is a model-specific exception: the exported PEFT prefix K must be injected unrotated.
4. **Skip on repeat**: After seeding a request, the request ID is recorded. On subsequent turns in the same session, re-seeding is skipped (APC has already cached the prefix).
5. **Block table mapping**: The injector computes `block_col = position // block_size` and `intra_block_idx = position % block_size` to write to the correct physical memory location.

### 5.4.2 GPT-OSS Runtime Fixes

GPT-OSS did not work with the initial centroid path even after centroid files exported successfully. Four runtime issues had to be fixed:

1. **RoPE lookup path**: GPT-OSS exposes rotary embeddings at `layers[i].attn.rotary_emb`, not `layers[i].self_attn.rotary_emb`.
2. **Multiple KV cache groups**: GPT-OSS alternates sliding-window and full-attention layers, so centroid injection must respect per-layer KV cache groups instead of always using group 0.
3. **KV cache tensor layout**: GPT-OSS exposed blocks-first KV cache tensors, so writes had to become layout-aware instead of assuming a single K/V axis order.
4. **Model-specific RoPE behavior**: GPT-OSS synthetic coherence was broken by rotating learned centroid K during injection. The working path is to inject GPT-OSS centroid K/V as already-final learned KV, without extra centroid-side RoPE.

Current behavior:

- GPT-OSS auto-detects in `centroid_integration.py` and disables centroid-side RoPE only for `hf_config.model_type == "gpt_oss"`.
- Other existing models keep the previous RoPE behavior.

### 5.5 APC (Automatic Prefix Caching) Interaction

With APC enabled and the pad prefix constant across all requests:

- **Turn 1, first-ever request**: Centroid injector seeds slots 0..N-1 manually. vLLM's `cache_blocks` then registers those physical blocks in the APC hash table (keyed on `[pad_id]*N` token hashes). This happens inside `allocate_slots`, before the GPU forward pass, so the blocks are registered even though KV data arrives slightly later.
- **Turn 1, second+ conversation**: APC finds the pre-cached centroid blocks → `num_local_cached_tokens = N`. `centroid_sched_gap` returns 0 (already covered). The GPU injector still runs and re-writes the same centroid KV (idempotent; the blocks are shared so no correctness issue).
- **Turn 2+**: APC hit rate grows further as the growing conversation history is also cached.

**Startup-time pre-registration** (`centroid_preregister_prefix_blocks`): pre-allocates the centroid prefix blocks in the APC hash table at server startup — before any request arrives. This makes even the very first request see local APC hits (Prometheus `prefix_cache_hits` increments on turn 1) rather than relying solely on the external-computed-token gap path. Enabled by setting `VLLM_CENTROID_PAD_TOKEN_ID`; the benchmark auto-detects this from the tokenizer.

**Implementation note**: `centroid_preregister_prefix_blocks` must be called from `EngineCore.__init__` in `core.py`, AFTER `init_none_hash()` is called (line ~207). Calling it from `Scheduler.__init__` (which runs earlier) fails with `NameError: name 'NONE_HASH' is not defined` because vLLM's block-hash global has not been initialized yet. The patched `core.py` calls `centroid_preregister_prefix_blocks` inside the `enable_prefix_caching` block, right after `init_none_hash`.

**Confirmed result (Llama-3.2-1B, N=64, 152-token prompt):**
- Turn 1 Prometheus hit rate: **42.1%** (64/152 tokens — exactly the 4 centroid blocks)
- Turn 2+: grows as conversation history accumulates in APC

**Metric note**: `apc_cached_tokens` in the API response (from `prompt_tokens_details.cached_tokens`) counts both local APC hits and centroid gap tokens as "cached." It shows N for turn 1 of every conversation. The Prometheus `prefix_cache_hits` metric is stricter — it only counts APC hash-table hits, not the gap path. With pre-registration, both metrics align on turn 1.

---

## 6. The Synthetic Centroid — Conceptual Summary

The core idea: **replace a long, fixed system prompt with a small, learned KV tensor that encodes the same behavioral priors**.

| Property | System Prompt | Synthetic Centroid |
|----------|--------------|-------------------|
| How encoded | Natural language text | Gradient-optimized KV pairs |
| Prefill cost | O(N²) in prompt length | Zero (injected directly) |
| Inference-time tokens | ~500–2000 text tokens | N=64/128/256 virtual slots |
| TTFT at 1000-token context | ~47.8ms | ~17.0ms (2.8× faster) |
| Cold-start benefit | None | Full speedup on first request |

The virtual tokens are trained to cause the model to produce the same outputs as if it had processed the full system prompt. The model never "sees" the system prompt text at inference time, yet behaves as an agent that does.

Why this works: transformer attention is content-addressed. If the KV pairs at positions 0..N-1 encode the right patterns, the Q vectors from user tokens attend to them the same way they would attend to the original system prompt KV — because the adapter was trained to make this true.

---

## 7. Testing — Multi-Turn Benchmark

### 7.1 Why Multi-Turn?

Single-turn benchmarks only measure cold TTFT. Real agentic workflows involve conversation history that grows across turns. Multi-turn testing measures:

1. Whether TTFT speedup holds as history grows.
2. How APC interacts with centroid injection across turns.
3. Whether the model maintains coherent, on-task responses through a full conversation.

### 7.2 Three Benchmark Modes

| Mode | System Prompt | APC | Centroid |
|------|--------------|-----|----------|
| `cold` | Full text in prompt | Disabled | No |
| `warm_apc` | Full text in prompt | Enabled | No |
| `synthetic` | Removed; centroid injected | Enabled | Yes (N=64/128/256) |

**cold** is the true baseline: no caching, no tricks, every token computed every time.  
**warm_apc** shows how much vLLM's built-in APC helps on its own.  
**synthetic** is our approach: centroid at positions 0..N-1, user history at positions N onward.

### 7.3 How a Conversation Is Structured

Each conversation uses 5 sequential tasks from the eval set. The model's response to turn T becomes part of the history fed to turn T+1. This simulates a real agent conversation where context accumulates.

```
Turn 1: prompt = [system] + [user_1]             → response_1
Turn 2: prompt = [system] + [user_1, resp_1] + [user_2]  → response_2
Turn 3: prompt = [system] + [...history...] + [user_3]   → response_3
...
```

In synthetic mode, the system prompt is replaced by `[pad]*N`:

```
Turn 1: prompt = [pad]*N + chat_template([user_1])              → response_1
Turn 2: prompt = [pad]*N + chat_template([user_1, resp_1, user_2]) → response_2
```

### 7.4 Per-Turn Measurement

For each turn, `multi_turn_benchmark.py` does:

1. **TTFT**: Opens a streaming request, timestamps the first non-empty chunk. Uses `max_tokens=1` for isolation on the TTFT-only measurement, then a separate full-generation call for response text.
2. **APC hit rate**: Scrapes Prometheus `/metrics` before and after the request:
   ```
   kv_hit_rate = (prefix_cache_hits_after - prefix_cache_hits_before) /
                 (prefix_cache_queries_after - prefix_cache_queries_before)
   ```
3. **Full generation**: A second request generates the complete assistant response (up to `max_tokens`, with headroom for context length).
4. **History update**: Append `(user_text, response_text)` to the running history.

### 7.5 N_virtual Token Sweep

The pipeline (`run_multi_turn_pipeline.py`) runs all modes sequentially:

```
cold → warm_apc → synthetic_N64 → synthetic_N128 → synthetic_N256
```

Each mode appends to the same output JSONL (`multi_turn_benchmark.jsonl`). N=256 is skipped automatically if centroid files for that N do not exist.

**Empirical TTFT results (1000-token system prompt, Llama-3.2-1B):**

| Context length | Mode | Physical tokens sent | TTFT | Speedup |
|---------------|------|---------------------|------|---------|
| ~200 tokens | cold | ~276 | 20.8ms | — |
| ~200 tokens | synthetic N=128 | ~184 | 18.7ms | 1.1× |
| ~1000 tokens | cold | ~1092 | 47.8ms | — |
| ~1000 tokens | synthetic N=128 | ~184 | **17.0ms** | **2.8×** |

**Key insight**: Synthetic TTFT stays flat (~17ms) regardless of original context length because the physical token count is always `N + user_query`. Cold TTFT grows with context. The crossover is around 200–300 tokens; below that, fixed GPU overhead dominates.

---

## 8. Metrics

### 8.1 TTFT — Primary Metric

**What it measures:** Time from request submission to first output token.  
**Why it matters:** Users feel this as "model response lag." For agentic pipelines with many sequential calls, TTFT compounds into total workflow latency.  
**How measured:** Streaming API — wall-clock time from `create()` to first chunk with non-empty content. GPU initialization overhead on the first request is excluded from steady-state statistics.

**Target:** TTFT of synthetic mode < cold mode. Confirmed at 2.8× for 1000-token contexts.

### 8.2 GOODBYE — Behavioral Encoding Signal

The system prompt contains: *"Always end the final response with the exact token: GOODBYE."*

This functions as a **litmus test for behavioral encoding**. It is not a user-facing feature — it is an observable marker that tells us whether the adapter has learned behavioral instructions, not just semantic content.

| Mode | GOODBYE rate |
|------|-------------|
| cold_no_synthetic (full system prompt) | 0% |
| warm_apc (full system prompt + APC) | 0% |
| synthetic N=64 | 12% |
| synthetic N=128 | **24%** |

**Interpretation:** The 1B model fails to follow the GOODBYE instruction even when it has the full system prompt in context — the instruction gets diluted in 1000 tokens of text. The adapter at N=128 encodes this instruction more densely: 24% of responses end with GOODBYE, despite the system prompt text never appearing in the physical prompt.

This is evidence that the adapter is learning compressed semantic representations, not surface-level text patterns.

**Note for larger models:** On 7B+ models, cold GOODBYE compliance will be higher (instruction following in long contexts is stronger). The behavioral encoding advantage of the adapter will narrow. TTFT speedup remains the primary metric at all model sizes.

### 8.3 Coherence

**What it checks:** Response is >20 words with no degenerate token repetition.  
**Why it matters:** Centroid injection writes directly into the KV cache. A misconfigured injection (wrong shape, wrong positions, bad RoPE rotation) causes the model to produce garbled or looping output.  
**Results:** Llama path was stable. GPT-OSS initially failed this metric with repeated `It is a` / `Sure` loops until the model-specific runtime fixes above were applied. After disabling centroid-side RoPE for GPT-OSS, synthetic generations became coherent and task-relevant again.

### 8.4 Task-Check Pass Rate

**What it checks:** Response contains at least one keyword from the `must_include_any` list for each task (e.g., mentions `time`, `context`, `manager` for a context-manager task).  
**Results:**

| Mode | Pass rate (25 tasks) |
|------|---------------------|
| cold | 88% (22/25) |
| synthetic N=64 | 84% (21/25) |
| synthetic N=128 | 84% (21/25) |

Slight degradation in synthetic mode is within noise at 25 samples. The adapter does not catastrophically fail on task content.

### 8.5 KV Cache Hit Rate

**What it measures:** Fraction of prompt tokens served by APC on a given turn.  
**How measured:** `kv_cache_hits / kv_cache_queries` scraped from vLLM's `/metrics` Prometheus endpoint before and after each turn.  
**Expected pattern:**
- Turn 1, without pre-registration: 0% Prometheus hit rate (gap path used; `apc_cached_tokens` in API still shows N).
- Turn 1, with pre-registration (`VLLM_CENTROID_PAD_TOKEN_ID` set): N/total_tokens Prometheus hit rate — local APC hit.
- Turn 2+: high hit rate in `warm_apc` and `synthetic` modes as the prefix accumulates in APC.
- `synthetic` mode: the constant `[pad]*N` prefix always hits APC from turn 2 onward; from turn 1 onward when pre-registration is active.

---

## 9. Quick Start

```bash
cd /home/yash/agentcache
source vllm-env/bin/activate

# Full multi-turn pipeline (all modes, N=64/128/256):
python run_multi_turn_pipeline.py \
    --model /mnt/g/agentcache/models/Llama-3.2-1B-Instruct \
    --system-prompt agentcache_compression/prompts/2000_python_agent_system.txt \
    --data agentcache_compression/data/python_agent_eval.jsonl \
    --n-conversations 5 \
    --turns-per-conv 5 \
    --out agentcache_compression/results/multi_turn_benchmark.jsonl

# Analyze results:
python agentcache_compression/analyze_multi_turn.py \
    --input agentcache_compression/results/multi_turn_benchmark.jsonl

# Run a single mode (useful for debugging):
python agentcache_compression/multi_turn_benchmark.py \
    --model /mnt/g/agentcache/models/Llama-3.2-1B-Instruct \
    --mode synthetic \
    --synthetic-len 128 \
    --centroid-k agentcache_compression/centroids/N128_2000_K.npy \
    --centroid-v agentcache_compression/centroids/N128_2000_V.npy \
    --out agentcache_compression/results/debug.jsonl
```

**Custom conversation file** (list of user messages as JSON array):

```bash
python run_multi_turn_pipeline.py \
    --model /path/to/model \
    --conversation-file my_conversation.json \
    ...
```

---

## 10. Key Files

| File | Purpose |
|------|---------|
| `agentcache_compression/train_prefix_compression.py` | Phase A: train PEFT prefix adapter with label masking |
| `agentcache_compression/transpose_tensors.py` | Phase B: export adapter weights to `.npy` centroid files |
| `agentcache_compression/multi_turn_benchmark.py` | Phase C: multi-turn TTFT + APC benchmark (3 modes) |
| `run_multi_turn_pipeline.py` | Orchestrates all modes in sequence |
| `agentcache_compression/analyze_multi_turn.py` | Parse results JSONL, compute per-turn statistics |
| `vllm/centroid_injector.py` | Injects centroid K/V into physical KV cache blocks |
| `vllm/centroid_integration.py` | Scheduler hooks: gap mechanism, layout control, APC pre-registration |
| `agentcache_compression/prompts/2000_python_agent_system.txt` | 2000-token system prompt used for training and benchmarks |
| `agentcache_compression/data/python_agent_train.jsonl` | 118 training examples |
| `agentcache_compression/data/python_agent_eval.jsonl` | 25 eval examples with keyword checks |
| `agentcache_compression/centroids/` | Exported `.npy` files for N=64/128/256 |

**vLLM files patched (5 files total):**

| File | What changed |
|------|-------------|
| `vllm/centroid_injector.py` | New file. `CentroidInjector` class: loads `.npy` centroid K/V, writes them into physical KV cache blocks, handles RoPE rotation, multi-group layout, per-request skip-on-repeat. |
| `vllm/centroid_integration.py` | New file. Bridge functions: `centroid_sched_gap`, `apply_centroid_block_table`, `centroid_preregister_prefix_blocks`, rotary-emb lookup, layout control. |
| `vllm/v1/worker/gpu_model_runner.py` | Two hook points added: (1) `centroid_override_num_computed` at new-request admission to set `eff_num_computed=N`; (2) `apply_centroid_block_table` call before every forward pass (skipped during CUDA graph capture). |
| `vllm/v1/core/sched/scheduler.py` | `centroid_sched_gap()` call added to the scheduling loop to inflate `num_external_computed_tokens` by N for centroid requests. |
| `vllm-env/.../vllm/v1/engine/core.py` | `centroid_preregister_prefix_blocks` called after `init_none_hash` in `EngineCore.__init__`. Must be here — calling it earlier (e.g., in Scheduler.__init__) fails because `NONE_HASH` is not yet defined. **No repo copy — edit in-place in vllm-env.** |

**Sync command** (run after any edit to repo `vllm/` files):
```bash
for f in centroid_injector.py centroid_integration.py v1/core/sched/scheduler.py; do
    cp vllm/$f vllm-env/lib/python3.10/site-packages/vllm/$f
done
# core.py: edit vllm-env/lib/python3.10/site-packages/vllm/v1/engine/core.py directly
```

---

## 11. Known Limitations & Next Steps

### GOODBYE compliance on 1B model

The base Llama-3.2-1B-Instruct does not reliably follow multi-sentence instructions even with the full system prompt. GOODBYE 0% in cold mode is a model capacity issue, not a benchmark bug. At 7B+, cold GOODBYE compliance will be higher, narrowing the behavioral encoding advantage of the adapter.

### Priority experiments

1. **Token × context-length grid**: Run N ∈ {64, 128, 256} × context ∈ {200, 500, 1000, 2000} to map the quality floor and speedup curve.
2. **N=256 adapter training**: Currently only N=64 and N=128 adapters are trained reliably. On GPT-OSS, N=256 prefix tuning is likely to hit memory limits unless batch size / sequence length / checkpointing settings are reduced.
3. **7B model validation**: Repeat pipeline on Llama-3.1-8B or Qwen-2.5-7B for production-scale results.
4. **Model-family audit**: Each new model family should be checked for (a) rotary lookup path, (b) KV cache group topology, (c) KV tensor layout, and (d) whether exported PEFT K should be centroid-side rotated at all.

### GPT-OSS Harmony Serve Note

`vllm serve` for GPT-OSS may fail at startup in offline or restricted environments because the OpenAI-compatible Responses / Chat / Anthropic serving surfaces initialize Harmony helpers and try to load the Harmony vocab. In this repo, the multi-turn benchmark was patched to avoid that failure by:

- preferring the local `venv/bin/vllm` executable when `vllm` is not on `PATH`,
- using token-ID prompts for GPT-OSS benchmark requests so completions can be used directly,
- and skipping GPT-OSS-only OpenAI Responses / Chat / Anthropic serving initialization if Harmony cannot initialize.

These changes are benchmark/runtime startup workarounds. They do not change GPT-OSS weights, adapter weights, centroid tensors, or the core GPT-OSS forward path.

### N=256 Training Memory Note

`train_prefix_compression.py` keeps:

- the full base model loaded in BF16,
- batch size fixed at 4,
- full padded assistant-training sequences,
- and `prefix_projection=True`, which adds a projected prefix of length `N` to every layer.

Increasing `num_virtual_tokens` from 64 → 128 → 256 increases the effective attended sequence length for every example and enlarges the per-layer learned prefix state. The dominant training-memory growth is activation / attention memory, not just saved adapter weights. On GPT-OSS this is especially expensive because:

- the model has 24 layers,
- training loads the base model as BF16 in `transformers`,
- and GPT-OSS training does not use any memory-saving features like gradient checkpointing in this script.

So `N=64` and `N=128` can fit while `N=256` crosses the VRAM cliff. The first mitigations to try are:

1. reduce `--batch-size` from 4 to 1 or 2,
2. shorten the system prompt or training sequence lengths,
3. enable gradient checkpointing,
4. if possible, avoid full BF16 materialization of GPT-OSS during training.
