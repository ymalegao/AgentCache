# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AgentCache eliminates the TTFT prefill bottleneck for domain-specific agents by training a PEFT prefix adapter, materializing it to raw K/V tensors, and seeding those tensors directly into vLLM's KV cache so the scheduler can skip prefill for the synthetic prefix.

Targets: **Llama-3.2-1B-Instruct** (current) and **Qwen-1.5B** (older Qwen adapters in `qwen15_64/`, `qwen15_256/`). See `HANDOFF.md` for the latest validation state — it is the authoritative status doc.

## Pipeline

```
generate_tasks.py / generate_good_examples.py   →   good_examples/*.jsonl
prefixtraining.py        →   agentcache_prefix_model/ (PEFT adapter)
transpose_tensors.py     →   centroid_K.npy, centroid_V.npy, sys_prefix_num_tokens.txt
vLLM (env-var driven)    →   test_injection.py / benchmark_ttft.py
```

| Stage | File | Output |
|---|---|---|
| Train | `prefixtraining.py` | `agentcache_prefix_model/adapter_model.safetensors` |
| Export | `transpose_tensors.py` | `centroid_{K,V}.npy` shape `[num_layers, N, kv_dim]` |
| Test | `test_injection.py` | Cold vs inject TTFT + coherence check |
| Benchmark | `benchmark_ttft.py` (via `run_benchmark.sh`) | Cold / APC / Inject TTFT |

`precompute_sys.py` is a legacy exact-system-prompt KV path (Qwen-only, pre-PEFT); the current pipeline does **not** use it.

## Commands

All Python entry points assume the vLLM virtualenv on the deploy host (not this checkout):

```bash
cd /home/yash/agentcache && source vllm-env/bin/activate

# Train
python prefixtraining.py

# Export (pure-PEFT layout, gap = N only)
python transpose_tensors.py --sys-tokens 0

# Smoke test injection (cold vs inject TTFT + output)
python test_injection.py

# Full A/B benchmark (Cold / APC / Inject)
./run_benchmark.sh
```

Single-purpose smoke scripts: `test.py`, `test_att.py`, `test_cold.py`, `test_tokens.py`.

## vLLM integration

The vLLM modifications live in two places:

- **In this repo** (`vllm/`): the canonical source of the patches — `centroid_injector.py`, `centroid_integration.py`, `v1/core/sched/scheduler.py`, `v1/worker/gpu_model_runner.py`, `v1/worker/gpu/model_runner.py`. Edit these.
- **In the installed package** (deploy host): `vllm-env/lib/python3.10/site-packages/vllm/...`. Runtime reads from here — patches must be copied over.

Flow per request:
1. `try_load_centroid_injector` loads `.npy` at runner startup.
2. Scheduler calls `centroid_sched_gap` to inflate `num_computed_tokens` by `gap = sys_token_count + centroid_len`, tricking the engine into treating the prefix as already prefilled.
3. Runner calls `apply_centroid_block_table` which invokes `seed_prefix_into_kv_cache` — writes K/V into block 0 with RoPE applied at the offset positions, then tracks the request id so it doesn't reseed on chunked-prefill / decode steps.

`vllm_centroid_changes.md` has the per-file change description.

## Critical constraints (do not regress)

- **Export path for `prefix_projection=True` MUST use `PeftModel.get_prompt()`** (see `transpose_tensors.py` lines 40–102). Manual `PrefixEncoder` reconstruction produces drift from the runtime cache and garbled output. Do not "simplify" this back to a direct safetensors reshape.
- **Prompt length guard:** scheduler `gap = sys_token_count + centroid_len`. If `prompt_tokens ≤ gap + MIN_TOKENS_AFTER_GAP` (default 32), `test_injection.py` aborts. N must be ≪ prompt_len.
- **Slicing centroids is not equivalent to retraining at smaller N** — `K[:, :64]` from a 256-token adapter is wrong. Retrain instead.
- **No dummy prompt padding** for Llama. The deprecated `inject_ids = [bos] * N + physical_ids` Llama workaround was removed; do not revive it. Llama and Qwen use the same `apply_chat_template` prompt for cold and inject.
- **GQA shape:** centroids are `[num_layers, N, num_kv_heads * head_dim]`, *not* `num_attention_heads * head_dim`. For Llama-3.2-1B that's `[16, N, 512]` (8 KV heads × 64 head_dim). On any new model, read `num_kv_heads` from the model config, **not** from `adapter_config.json`.
- **MLA architectures (DeepSeek-V2/V3/Coder-V2) are unsupported** — the KV layout is fundamentally different. Add an explicit architecture check before exporting; do not silently produce nonsense tensors.

## Env vars

| Var | Typical | Meaning |
|---|---|---|
| `VLLM_CENTROID_SCHEDULER` | `1` inject / `0` cold | Enable scheduler skip + injector seed |
| `VLLM_CENTROID_K_PATH` / `_V_PATH` | abs path to `.npy` | Centroid tensors |
| `VLLM_CENTROID_SYS_TOKENS` | `0` (pure PEFT) | Overrides `sys_prefix_num_tokens.txt`; `gap = this + centroid_len` |
| `VLLM_CENTROID_USE_LMCACHE` | `0` | Skip exact-sys path (LMCache handles it) |
| `VLLM_EXACT_SYS_K_PATH` / `_V_PATH` | unset | Optional exact system-prompt KV alongside PEFT prefix |
| `VLLM_CENTROID_SINK_BLEND` | `0.35` | Blend factor for attention-sink slot |
| `CENTROID_PERF_DEBUG` | `0` for benchmarks | Engine perf logs |
| `CENTROID_DEBUG_ROPE` / `CENTROID_DEBUG` | `0` for benchmarks | Verbose per-step logs (heavy GPU sync — distorts TTFT) |
| `CENTROID_TIMING` | `0` | Wall-clock per injection (`1` or `cuda`) |

## Conventions

- Adapter / centroid artifacts live at the repo root; the legacy `attention_centroid_output/` path referenced in some scripts is from the older Qwen exact-sys flow and may not exist.
- Training data is `good_examples/vllm_good_examples_raw.jsonl` (Llama-targeted). The unprefixed `good_examples.jsonl` is older.
- `personas/user*.yaml` + `generate_tasks.py` produce perturbed-persona inputs; `generate_good_examples.py` then runs plain vLLM (no injection) to produce training targets.
- The benchmark / test scripts reference absolute paths under `/home/yash/agentcache/` and `/mnt/g/agentcache/models/` — these are deploy-host paths, not portable.

## When validating changes

`test_injection.py`'s built-in `_is_coherent` heuristic only checks word-likeness, **not** task correctness. For real quality regressions, eyeball the output against the cold baseline (does it mention `time`, timing, a context manager / `__enter__`/`__exit__`?). See `HANDOFF.md` § "Validation results" for the reference 64-token Llama output.
