# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

AgentCache is a research project that eliminates the TTFT (Time-To-First-Token) prefill bottleneck for domain-specific LLM agents by replacing long system prompts with a small, gradient-trained KV-cache prefix injected directly into vLLM's KV cache. Demonstrated 2.8× TTFT speedup at 1000-token contexts on Llama-3.2-1B.

Three-phase pipeline:
1. **Train** — PEFT prefix-tuning produces an adapter under `agentcache_compression/adapters/N{tokens}/`.
2. **Transpose** — materialize the adapter into `centroid_K.npy` / `centroid_V.npy` of shape `[num_layers, N, token_dim]`.
3. **Inject** — a patched vLLM seeds positions `0..N-1` of the KV cache from those `.npy` files; the scheduler's "gap mechanism" overrides `num_computed_tokens=N` so those slots are treated as already-prefilled and the N placeholder pad tokens in the prompt are never run through the model.

`HANDOFF.md` is the long-form system design. `README.md` has the empirical results table.

## Common commands

```bash
# One-time environment setup. Creates venv/, pins vllm==0.20.0, then copies
# vllm/centroid_injector.py, vllm/centroid_integration.py and the patched
# scheduler/model_runner files into site-packages/vllm/.
./install.sh
source venv/bin/activate

# Download a base model (gated models need: hf login).
./get_model.sh meta-llama/Llama-3.2-1B-Instruct
# → models/Llama-3.2-1B-Instruct/

# Full train → transpose → test pipeline for one N_virtual value.
python run_training_pipeline.py --model models/Llama-3.2-1B-Instruct --tokens 64

# Resume (adapter or centroids already on disk):
python run_training_pipeline.py --model <path> --tokens 64 --skip-train
python run_training_pipeline.py --model <path> --tokens 64 --skip-train --skip-transpose

# Run all three eval modes (cold_no_synthetic, warm_apc, synthetic_compression).
python run_training_pipeline.py --model <path> --tokens 64 --test-modes all

# Multi-turn benchmark — boots `vllm serve` per mode, scrapes Prometheus for APC hit rate.
# Loops over N ∈ {64, 128, 256}; skips an N if its centroid .npy files are missing.
python run_multi_turn_pipeline.py --model <path>

# Combined Component 1 (LMCache disk offload) + Component 2 (centroid injection) benchmark.
# Conditions: cold | lmcache | centroid | combined.
python combined_benchmark.py
```

Phase scripts (the pipelines wrap these; invoke directly only when iterating on one step):
- `agentcache_compression/train_prefix_compression.py`
- `agentcache_compression/transpose_tensors.py`
- `agentcache_compression/test_compression.py`
- `agentcache_compression/multi_turn_benchmark.py`

There is no unit-test suite. Output quality is verified through eval JSONL in `agentcache_compression/data/` and the per-task `must_include_any` keyword checks in `test_compression.py` (`coherent`, `ends_with_goodbye`, `task_check_pass`).

## Architecture you need before editing

### vLLM is patched in place by `install.sh`
The local `vllm/` directory is the source of truth; `install.sh` copies its files into the active venv's `site-packages/vllm/`:

| Source (in repo)                          | Destination (in venv)                       |
|-------------------------------------------|---------------------------------------------|
| `vllm/centroid_injector.py`               | `vllm/centroid_injector.py`                 |
| `vllm/centroid_integration.py`            | `vllm/centroid_integration.py`              |
| `vllm/v1/core/sched/scheduler.py`         | `vllm/v1/core/sched/scheduler.py`           |
| `vllm/v1/worker/gpu_model_runner.py`      | `vllm/v1/worker/gpu_model_runner.py`        |
| `vllm/v1/worker/gpu/model_runner.py`      | `vllm/v1/worker/gpu/model_runner.py`        |

**Editing any file under `vllm/` has no effect until you re-run `./install.sh`** (or `cp` it into site-packages by hand). Forgetting this leads to confusing "my fix didn't change anything" sessions.

### Runtime is env-var-gated
The patched vLLM is dormant unless these are set at `vllm serve` startup:
```
VLLM_CENTROID_SCHEDULER=1
VLLM_CENTROID_K_PATH=.../N128_2000_K.npy
VLLM_CENTROID_V_PATH=.../N128_2000_V.npy
VLLM_CENTROID_SYS_TOKENS=0          # 0 = pure compression mode (no system prompt in text)
VLLM_CENTROID_LAYOUT=compression
VLLM_CENTROID_PAD_TOKEN_ID=128001   # optional; pre-registers centroid blocks in APC for turn-1 hits
```
With `VLLM_CENTROID_SCHEDULER=0` (or unset) the patched vLLM behaves like stock vLLM — useful for cold baseline runs.

### Prompt construction in compression mode
The client builds `prompt_token_ids = [pad_id] * N + user_chat_token_ids` — **no system prompt in the text**. The system prompt's behavior is encoded in the injected KV. Putting the system prompt back in the text double-counts it and invalidates the comparison. See `build_compression_ids()` in both `test_compression.py` and `multi_turn_benchmark.py`.

### Centroid file naming carries metadata
`agentcache_compression/centroids/N{N}_{SYS_LEN}_K.npy` — `N` is virtual-token count, `SYS_LEN` is the system-prompt-length variant the adapter was trained on (`prompts/{200,500,1000,2000}_python_agent_system.txt`). `run_multi_turn_pipeline.py` specifically expects `N{N}_2000_K.npy`. Domain-tagged variants (`coding…`, `search…`) also live in this directory.

A sidecar `sys_prefix_num_tokens.txt` written next to the `.npy` files is read by `CentroidInjector` when `VLLM_CENTROID_SYS_TOKENS` is unset.

### GPT-OSS diverges from Llama/Qwen
`centroid_integration.py` auto-detects `hf_config.model_type == "gpt_oss"` and disables centroid-side RoPE rotation for it (PEFT's `get_prompt()` already returns final cache-aligned K — rotating again broke coherence). GPT-OSS also required: looking up rotary-emb at `attn.rotary_emb` instead of `self_attn.rotary_emb`, respecting multiple KV cache groups (sliding-window + full-attention layers alternate), and a blocks-first KV tensor layout. When adding a new model family, expect to find similar architecture-specific quirks — see HANDOFF.md §5.4.1.

### Projected vs flat adapter export
`transpose_tensors.py` has two branches. We always train with `prefix_projection=True`, which goes through PEFT's `peft_model.get_prompt()` at export time. The non-projected flat-reshape branch is for an earlier experiment config and is no longer the trained path. Manual MLP reconstruction was tried, had subtle alignment bugs that produced silent garbage, and was abandoned — do not re-introduce it.

### Label masking matters
`train_prefix_compression.py` sets `labels = [-100] * prompt_len + input_ids[prompt_len:]`, so loss is computed only on assistant tokens. Without this the adapter optimizes to reproduce the prompt text instead of the response. `--system-retain-ratio 0.0` keeps the system prompt out of training inputs, forcing the virtual tokens to encode it.

### Apple Silicon port lives elsewhere
The Metal port is a separate codebase at `AgentCache-metal/vllm-metal/` (also cloned at `~/Documents/AgentCache-metal/vllm-metal`). Don't edit it from this repo. `agentcache_mac/` here holds Mac-side adapters/results only.

## Conventions

- Scripts are CLI-driven via argparse; defaults point under `agentcache_compression/`.
- Training data: `agentcache_compression/data/*.jsonl`. System prompts: `agentcache_compression/prompts/{200,500,1000,2000}_*_system.txt`.
- Results JSONL: `agentcache_compression/results/`. Plots: `plot_*.py` scripts in the same directory consume the JSONL.
- `agentcache_compression/adapters/` is gitignored — checkpoints are expected to be regenerated, not committed.
- `requirements.txt` pins `transformers==5.7.0` / `torch==2.11.0` / `vllm==0.20.0`. Don't bump without re-verifying the patched vLLM files still apply cleanly to the new version's internal layout.

<!-- dgc-policy-v11 -->
# Dual-Graph Context Policy

This project uses a local dual-graph MCP server for efficient context retrieval.

## MANDATORY: Always follow this order

1. **Call `graph_continue` first** — before any file exploration, grep, or code reading.

2. **If `graph_continue` returns `needs_project=true`**: call `graph_scan` with the
   current project directory (`pwd`). Do NOT ask the user.

3. **If `graph_continue` returns `skip=true`**: project has fewer than 5 files.
   Do NOT do broad or recursive exploration. Read only specific files if their names
   are mentioned, or ask the user what to work on.

4. **Read `recommended_files`** using `graph_read` — **one call per file**.
   - `graph_read` accepts a single `file` parameter (string). Call it separately for each
     recommended file. Do NOT pass an array or batch multiple files into one call.
   - `recommended_files` may contain `file::symbol` entries (e.g. `src/auth.ts::handleLogin`).
     Pass them verbatim to `graph_read(file: "src/auth.ts::handleLogin")` — it reads only
     that symbol's lines, not the full file.
   - Example: if `recommended_files` is `["src/auth.ts::handleLogin", "src/db.ts"]`,
     call `graph_read(file: "src/auth.ts::handleLogin")` and `graph_read(file: "src/db.ts")`
     as two separate calls (they can be parallel).

5. **Check `confidence` and obey the caps strictly:**
   - `confidence=high` -> Stop. Do NOT grep or explore further.
   - `confidence=medium` -> If recommended files are insufficient, call `fallback_rg`
     at most `max_supplementary_greps` time(s) with specific terms, then `graph_read`
     at most `max_supplementary_files` additional file(s). Then stop.
   - `confidence=low` -> Call `fallback_rg` at most `max_supplementary_greps` time(s),
     then `graph_read` at most `max_supplementary_files` file(s). Then stop.

## Token Usage

A `token-counter` MCP is available for tracking live token usage.

- To check how many tokens a large file or text will cost **before** reading it:
  `count_tokens({text: "<content>"})`
- To log actual usage after a task completes (if the user asks):
  `log_usage({input_tokens: <est>, output_tokens: <est>, description: "<task>"})`
- To show the user their running session cost:
  `get_session_stats()`

Live dashboard URL is printed at startup next to "Token usage".

## Rules

- Do NOT use `rg`, `grep`, or bash file exploration before calling `graph_continue`.
- Do NOT do broad/recursive exploration at any confidence level.
- `max_supplementary_greps` and `max_supplementary_files` are hard caps - never exceed them.
- Do NOT dump full chat history.
- Do NOT call `graph_retrieve` more than once per turn.
- After edits, call `graph_register_edit` with the changed files. Use `file::symbol` notation (e.g. `src/auth.ts::handleLogin`) when the edit targets a specific function, class, or hook.

## Context Store

Whenever you make a decision, identify a task, note a next step, fact, or blocker during a conversation, call `graph_add_memory`.

**To add an entry:**
```
graph_add_memory(type="decision|task|next|fact|blocker", content="one sentence max 15 words", tags=["topic"], files=["relevant/file.ts"])
```

**Do NOT write context-store.json directly** — always use `graph_add_memory`. It applies pruning and keeps the store healthy.

**Rules:**
- Only log things worth remembering across sessions (not every minor detail)
- `content` must be under 15 words
- `files` lists the files this decision/task relates to (can be empty)
- Log immediately when the item arises — not at session end

## Session End

When the user signals they are done (e.g. "bye", "done", "wrap up", "end session"), proactively update `CONTEXT.md` in the project root with:
- **Current Task**: one sentence on what was being worked on
- **Key Decisions**: bullet list, max 3 items
- **Next Steps**: bullet list, max 3 items

Keep `CONTEXT.md` under 20 lines total. Do NOT summarize the full conversation — only what's needed to resume next session.
