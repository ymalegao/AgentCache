# AgentCache → vLLM-Metal Port — Implementation Plan

**Date:** 2026-05-23
**Goal:** A *faithful* third implementation of AgentCache centroid injection that runs on
Apple Silicon via the `vllm-metal` plugin (MLX/Metal backend) — real paged-cache block
seeding + real scheduler prefill-skip, the same mechanism as the CUDA path. This is **not**
`agentcache_mac`, which only emulates the mechanism with HF `past_key_values`.

---

## 1. Context — why this is worth doing

`agentcache_mac/` demonstrates the *concept* (skip system-prompt prefill, preserve quality)
but discards the actual product architecture: no paged KV cache, no scheduler that skips
prefill, no direct KV seeding. The CUDA path (`vllm/`) is the real thing but needs a GPU host.

`vllm-metal` is a vLLM **hardware plugin** (MLX compute backend + Metal kernels) that
**reuses vLLM's core engine and scheduler**. Because the centroid mechanism splits cleanly
into a backend-agnostic scheduler half and a backend-specific KV-seeding half, and because
`vllm-metal`'s paged cache is structurally identical to CUDA's, the port is largely a
torch→MLX translation of two files — not a reimplementation. Outcome: AgentCache's true
architecture reproducible on a MacBook for development, demos, and Apple-Silicon validation.

---

## 2. Reference architecture (what we're porting FROM)

Three integration touchpoints in the CUDA path:

| # | Piece | Location | Role |
|---|---|---|---|
| 1 | Scheduler gap | `vllm/v1/core/sched/scheduler.py:650-658` → `centroid_sched_gap()` | Inflates `num_external_computed_tokens` by `gap = sys_tokens + centroid_len` so the engine treats the prefix as already prefilled. |
| 2 | `num_computed` override | `centroid_override_num_computed()` | Runner-side fallback; in scheduler-mode the scheduler already set it. |
| 3 | KV seed | `apply_centroid_block_table()` → `seed_prefix_into_kv_cache()` (`centroid_injector.py`) | Pre-forward, writes RoPE-rotated centroid K + raw V into block slots `0..N-1`. Dedups per request id. |

Key invariants (from `CLAUDE.md`, must not regress):
- Centroids are `.npy` `[num_layers, N, num_kv_heads * head_dim]` (GQA: read `num_kv_heads`
  from the **model** config, not `adapter_config.json`). Llama-3.2-1B = `[16, N, 512]`.
- Only **K is RoPE-rotated** (at positions `sys..sys+N-1`); **V is stored raw**.
- RoPE must use the model's **own** rope op — never a hand-rolled reconstruction.

---

## 3. vLLM-Metal architecture (what we're porting TO) — confirmed by source read

| Concern | CUDA | vLLM-Metal | Source |
|---|---|---|---|
| Plugin model | core vLLM | hardware plugin, **reuses core scheduler/engine** | repo overview |
| KV cache type | torch tensor, K/V stacked dim 0 | **separate** `mx.array` lists `key_caches[L]`, `value_caches[L]` | `metal_kernel_backend/cache.py` |
| Per-layer shape | `[2, blocks, block_size, kv_heads, head_dim]` | `(num_blocks, block_size, kv_heads, head_dim)` | `cache.py` |
| Write pattern | `kv[0/1, phys, intra]=...` | `flat=cache[L].reshape(-1,kv_heads,head_dim); flat[slot_mapping]=v3d; cache[L]=flat.reshape(orig)` (in-place `__setitem__` then reassign) | `metal_kernel_backend/attention_sdpa.py:~340-350` |
| Block map | 2D `block_table[req, col]` tensor | per-request `state.block_ids` (list) | `v1/model_runner.py` |
| Slot formula | `phys*block_size+intra` | `block_ids[pos//bs]*bs + pos%bs` — **identical** | `paged_attention_common.py` `prepare_unified()` |
| Attends to cached prefix? | yes (seq_len includes computed) | **yes** — `context_lens.append(start_pos+num_tokens)` | `paged_attention_common.py` |
| RoPE | `rotary_emb.forward_native(pos, q, k)` | `apply_packed_rope(attn, q, k, …, offsets)` → `attn.rope(seg, offset=off)` (mlx_lm) | `packed_prefill_compat.py` |
| Runner | `GPUModelRunner` | `MetalModelRunner.execute_model()` / `_start_paged_forward()` | `v1/model_runner.py` |
| Model wrap | vLLM model | mlx_lm model via `DefaultModelAdapter`; backbone = `model.model`, layers expose `self_attn.rope` | `v1/model_adapter.py` |

**Two big de-riskers from the read:**
1. `context_lens` includes the seeded-but-not-computed prefix → the Metal paged-attention
   kernel *will* attend to our injected slots. This was the load-bearing assumption.
2. mlx_lm's `attn.rope(x, offset=off)` rotates positions `off..off+len-1` and takes a scalar
   offset — a clean analog to CUDA offline RoPE. K layout for the call is `[1, kv_heads, N, head_dim]`.

---

## 4. Implementation

### 4.1 New module: `vllm_metal_centroid/injector_mlx.py`
MLX port of `centroid_injector.py`. Class `MetalCentroidInjector`:

- **`__init__(k_path, v_path)`** — `np.load` → `mx.array` (dtype = cache dtype, typically
  fp16/bf16). Store `num_layers`, `centroid_len = K.shape[1]`, `sys_token_count`
  (reuse `load_sys_prefix_token_count` logic / env `VLLM_CENTROID_SYS_TOKENS`).
- **`_rope_centroid_k(layer_rope, num_kv_heads, head_dim)`** — for each layer reshape
  `K[L]` → `[1, kv_heads, N, head_dim]`, call `layer_rope(k4d, offset=sys_token_count)`,
  cache the rotated result (it's position-stable across steps, like the CUDA `_rope_k_cache`).
  **V is never rotated.**
- **`seed(kv_cache, block_ids, prompt_len, num_kv_heads, head_dim, block_size, layer_ropes, req_id)`**
  — mirror `seed_prefix_into_kv_cache`:
  1. `centroid_fill = min(centroid_len, max(0, prompt_len - sys_token_count))`; bail if ≤0.
  2. Build seed positions `p = sys_token_count .. sys_token_count+fill-1`.
  3. `slots = [block_ids[p//bs]*bs + p%bs for p in positions]` (an `mx.array` int index).
     Skip if any slot maps to the null/reserved block.
  4. Per layer: `flat = kv_cache.key_caches[L].reshape(-1, kv_heads, head_dim);
     flat[slots] = k_rot[L]; kv_cache.key_caches[L] = flat.reshape(orig)` — and the same
     for `value_caches` with raw `V[L]`. **Reassign back** (MLX functional semantics).
  5. `mx.eval(kv_cache.key_caches[L], kv_cache.value_caches[L])` after writes so the buffers
     are materialized before the forward reads them.
- **Dedup:** `self._seeded_req_ids: set[str]`; skip a request once seeded (decode/chunked steps).

### 4.2 New module: `vllm_metal_centroid/integration_mlx.py`
Thin glue analogous to `centroid_integration.py`:
- `try_load_metal_centroid_injector()` — gated on `VLLM_CENTROID_SCHEDULER=1` + K-file exists.
- `get_layer_ropes(model_adapter)` — reach `backbone = model.model`; return
  `[layer.self_attn.rope for layer in backbone.layers]` (or just `layers[0].self_attn.rope`
  replicated — params identical per layer for Llama/Qwen; confirm during build).
- `apply_metal_centroid(runner, batch, kv_cache)` — iterate prefill requests in `batch`,
  pull `state.block_ids` / `state.prompt_len` / `req_id`, call `injector.seed(...)`.

### 4.3 Hook points (edits to vllm-metal, or monkeypatch shim)
- **Runner init** (`MetalModelRunner.__init__` in `v1/model_runner.py`, or `v1/worker.py`):
  `self._centroid_injector = try_load_metal_centroid_injector()`.
- **Pre-forward seed** (`_start_paged_forward`, immediately before
  `_target_forward(..., cache=offset_caches)`): call `apply_metal_centroid(self, batch, <shared paged kv_cache>)`.
  → **Confirm during build:** the exact handle to the shared paged cache object holding
  `key_caches`/`value_caches` inside the runner (referenced as `kv_cache` in `attention_sdpa`).
- **Scheduler gap:** *no vllm-metal change.* It reuses core vLLM's scheduler, so running the
  plugin on our already-patched core vLLM `scheduler.py` carries the gap through. (If the
  deploy uses stock core vLLM, port the 8-line `centroid_sched_gap` insertion at the same
  spot — it's pure-Python and backend-agnostic.)

**Packaging note:** prefer a small standalone package + a `vllm` plugin entrypoint / import
shim that monkeypatches the two runner methods, so we don't fork `vllm-metal`. Fall back to a
patch-over-installed-package approach (like the CUDA `vllm/` dir) if the entrypoints don't
reach `_start_paged_forward`.

### 4.4 Prompt construction (client / benchmark)
Unchanged from compression mode: `prompt_token_ids = [pad]*N + apply_chat_template(user)`,
**no system text**. Scheduler reports `num_computed = gap`, runner slices `token_ids[gap:]`,
seeded centroid fills positions `0..N-1`. Reuse `agentcache_compression/` data & prompts.

---

## 5. Milestones

1. **Loader + RoPE parity (1 day).** Load N=64 `.npy` as MLX; pull layer rope; pre-rotate K;
   unit-check the rotated K against a reference (e.g. compare `attn.rope` output for a known
   input to the CUDA `forward_native` numerically within tolerance).
2. **Single-request prefill seed (2-3 days).** Hook `_start_paged_forward`, seed one request,
   confirm coherent output vs cold baseline on the N=64 Llama centroid. *This is the gate.*
3. **Robustness (3-4 days).** Multi-request batches, chunked prefill, decode-step skip
   (dedup), null-block guard, `sys_token_count>0` path if needed.
4. **Validation + sweep (1-2 days).** Wire into a Metal analog of `hf_eval.py` /
   `multi_turn_benchmark.py`; run the cold-vs-inject TTFT + quality comparison; N=64/128/256.

**Estimate:** ~3-6 days to the Milestone-2 gate; ~1-2 weeks to CUDA-parity robustness.

---

## 6. Validation

- **Numeric RoPE check** (Milestone 1): rotated centroid K from `attn.rope` matches the
  exported/CUDA convention within fp16 tolerance. The #1 correctness risk lives here.
- **Coherence vs cold** (Milestone 2): inject output on the context-manager eval task mentions
  `time` / `__enter__`/`__exit__`, like the reference 64-token Llama output in `HANDOFF.md`.
  `_is_coherent` checks word-likeness only — eyeball task correctness.
- **TTFT (relative)**: inject < cold, gap widens with system-prompt length (200/500/1000/2000
  prompts). Absolute ms is MLX, not comparable to CUDA's 2.8×; the *shape* of the curve is the
  faithful result.
- **Quality**: task-keyword pass rate within noise of cold (CUDA saw 88%→84%).
- **Read-back assertion** (debug): after seed, read a seeded slot and `mx.allclose` vs the
  written value (analog of the CUDA `CENTROID_DEBUG` readback).

---

## 7. Risks & open items

| Risk | Severity | Mitigation |
|---|---|---|
| RoPE convention mismatch (neox/traditional, scale, per-layer params) garbles output | **High** | Drive centroid through vllm-metal's own `attn.rope`; numeric parity check before any forward. Same lesson as CUDA `PeftModel.get_prompt()` rule. |
| Shared-paged-cache handle not reachable in `_start_paged_forward` | Med | Read `model_runner.py` `_start_paged_forward` / `attention_sdpa` ctx during build; the write site proves the object exists. |
| MLX laziness / buffer donation — seed not visible to forward | Med | `mx.eval` seeded arrays; reassign `key_caches[L]` after `__setitem__`; readback assert. |
| Per-layer vs shared rope assumption | Low | Llama/Qwen rope params identical per layer; verify with one assert. |
| Quantized (AWQ/turboquant) cache uses packed dims | Low | Restrict v1 to fp16/bf16 cache; gate out quant paths (turboquant uses `k_packed_dim`). |
| MLA models | Low | Reject at load (already a documented CUDA constraint). |

**Reads still pending before coding** (cheap, do first): `attn.rope`/`mx.fast.RoPE` exact
signature & flags; `_start_paged_forward` body for the cache handle; `model_adapter` layer/rope
attribute path. These convert this plan into line-level edits.

---

## 8. Env vars (reuse CUDA names)
`VLLM_CENTROID_SCHEDULER`, `VLLM_CENTROID_K_PATH`/`_V_PATH`, `VLLM_CENTROID_SYS_TOKENS`,
`VLLM_CENTROID_LAYOUT=compression`. Centroid `.npy` artifacts from
`agentcache_compression/transpose_tensors.py` are reused **as-is** (same `[layers, N, kv_dim]`).

---

## 9. Environment setup — new working dir + pull the repo

The port is developed against a **checked-out copy of `vllm-metal`**, kept separate from the
AgentCache repo (we need its source to add the runner hooks). Do **not** nest it inside
`AgentCache/`.

```bash
# 1. New working directory, sibling to AgentCache
mkdir -p ~/Documents/AgentCache-metal && cd ~/Documents/AgentCache-metal

# 2. Pull the vllm-metal source
git clone https://github.com/vllm-project/vllm-metal.git
cd vllm-metal

# 3. Prereqs: native arm64 Python 3.12 (Rosetta/x86_64 NOT supported), macOS Apple Silicon.
#    Installs core vLLM + MLX and builds the Metal kernels.
python3.12 -m venv .venv && source .venv/bin/activate

# 4. Install. Their supported route is the install script:
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
#    For a dev/editable install of this checkout (so our hooks take effect), additionally:
pip install -e .            # confirm against their README; Metal-kernel build step may differ

# 5. Sanity-check STOCK serving before any centroid work (OpenAI-compatible API):
vllm serve <model> --port 8000
#    Hit it with a normal completion and confirm coherent output on Metal first.
```

Then add our `vllm_metal_centroid/` package (§4.1–4.2) into this checkout and wire the two
runner hooks (§4.3). **Confirm during build:** whether `pip install -e .` actually reaches
`MetalModelRunner._start_paged_forward` — if the plugin entrypoints don't, fall back to the
patch-over-installed-package approach used by the CUDA `vllm/` dir (copy patched files into
`site-packages/vllm_metal/...`).

---

## 10. End-to-end Mac workflow (train → export → serve → measure)

Phases A/B run in the **AgentCache repo** (HF/torch on MPS); Phase C/D run in the **vllm-metal
checkout** (§9). The bridge between them is the `.npy` + matching model weights + matching RoPE.
**Train and serve the *same* model** for an apples-to-apples comparison.

```bash
# --- Phase A — train PEFT adapter (AgentCache repo, MPS, fp32; no QLoRA on MPS) ---
cd ~/Documents/AgentCache/AgentCache && source venv-mac/bin/activate
python agentcache_mac/run_mac_pipeline.py \
  --model <model> --tokens 64 \
  --system-prompt agentcache_compression/prompts/2000_python_agent_system.txt
#   → adapter lands in agentcache_mac/adapters/<model>_N64/  (confirm flags in that script)

# --- Phase B — export adapter to .npy centroids (CPU; identical artifact to CUDA) ---
python agentcache_compression/transpose_tensors.py \
  --adapter agentcache_mac/adapters/<model>_N64 \
  --out-k centroids/N64_2000_K.npy --out-v centroids/N64_2000_V.npy --sys-tokens 0

# --- Phase B.5 — torch→MLX VALIDATION (the gate; do before trusting any number) ---
#   (a) RoPE parity:  mlx_lm attn.rope(centroid_K, offset=sys) ≈ CUDA forward_native(positions)
#                     within fp16 tolerance  (Milestone 1).
#   (b) Weight fidelity: serve the SAME model in HF and in MLX; cold (no-inject) outputs match.
#   Rationale: the centroid was gradient-optimized onto the HF model's manifold. If MLX's
#   weight conversion or rope differs, the injected K/V drift off-manifold → garbled output
#   (the documented failure mode). Validate, don't assume.

# --- Phase C — serve with centroid injection (vllm-metal checkout) ---
cd ~/Documents/AgentCache-metal/vllm-metal && source .venv/bin/activate
VLLM_CENTROID_SCHEDULER=1 \
VLLM_CENTROID_K_PATH=/abs/path/centroids/N64_2000_K.npy \
VLLM_CENTROID_V_PATH=/abs/path/centroids/N64_2000_V.npy \
VLLM_CENTROID_SYS_TOKENS=0 VLLM_CENTROID_LAYOUT=compression \
vllm serve <model> --port 8000

# --- Phase D — measure with the SAME prompts/eval as CUDA ---
#   Accuracy: task-keyword pass + coherence on agentcache_compression/data/python_agent_eval.jsonl
#             → should TRACK the CUDA numbers (accuracy is adapter/model property, ports across backends).
#   TTFT:     cold vs inject across the 200/500/1000/2000 prompts
#             → RELATIVE shape only (inject < cold, gap grows with prompt length); absolute ms ≠ CUDA.
```

**What this proves vs doesn't:** accuracy parity with CUDA is meaningful and expected; TTFT is a
faithful *relative* result on Metal (real paged-cache seeding + scheduler prefill-skip), **not**
the CUDA absolute latency or the 2.8× figure — that's backend-bound and does not transfer.
