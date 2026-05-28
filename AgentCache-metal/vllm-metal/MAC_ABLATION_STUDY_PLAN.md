# Mac Ablation Study Plan

## Goal

Show that AgentCache centroid injection reduces time to first token (TTFT) on
Apple Silicon by replacing a long system/domain prompt with a fixed-size learned
KV prefix, while preserving enough output quality to be useful.

This Mac study does not use LMCache. LMCache is the CUDA/Linux persistent KV
cache baseline. On Mac, the warm-cache baseline is vLLM-Metal native prefix
caching.

## Hardware Target

Run the Mac study on the local Apple Silicon machine:

| Machine | Memory | Notes |
|---|---:|---|
| Apple M3 Max | 96 GB unified memory | Enough headroom for small/medium BF16 models and larger 4-bit MLX models, but the OS, Python, vLLM-Metal, KV cache, and long-context sweeps all share the same unified memory pool. |

Do not plan around using the full 96 GB for model weights. Keep headroom for KV
cache growth, long prompts, process startup overhead, and repeated benchmark
runs.

## Supported Baselines

| Condition | Mac / vLLM-Metal support | Notes |
|---|---:|---|
| `cold` | yes | Full system prompt is physically sent and fully prefills every run. |
| `prefix_cache` / native APC | yes | vLLM-Metal native prefix caching; this is the warm-cache baseline on Mac. |
| `centroid` | yes | AgentCache centroid KV injection through the vLLM-Metal paged KV cache. |
| `centroid + prefix_cache` | yes | Centroid handles the synthetic prefix; native prefix caching can help repeated/growing history. |
| `LMCache` | no | LMCache is not currently supported by the local Metal setup. |
| `LMCache + centroid` | no | CUDA/Linux combined baseline only, unless a Metal/MLX LMCache connector is ported. |

## Recommended Model Set

Use a model ladder that spans tiny, small, medium, and large models while staying
realistic on an M3 Max with 96 GB unified memory.

| Tier | Model | Precision / format | Use in study | Centroid status |
|---|---|---|---|---|
| Smoke | `mlx-community/Qwen2.5-0.5B-Instruct-bf16` | BF16 MLX | Fast correctness and notebook sanity checks | Existing Qwen 0.5B N=64 centroid in this repo |
| Primary small | `mlx-community/Llama-3.2-1B-Instruct-bf16` | BF16 MLX | Main quality + TTFT reference | Existing Llama 1B N=128 centroid from the CUDA pipeline |
| Primary medium | `mlx-community/Llama-3.2-3B-Instruct-bf16` | BF16 MLX | Model-size scaling with the same Llama family | Train/export a real centroid for quality; dummy centroid is TTFT-only |
| Primary medium | `mlx-community/Qwen2.5-3B-Instruct-4bit` | 4-bit MLX | Qwen-family model-size scaling | Train/export a real centroid for quality; dummy centroid is TTFT-only |
| Primary large | `mlx-community/Qwen2.5-7B-Instruct-4bit` | 4-bit MLX | Strong large-model TTFT curve while staying easy to fit | Train/export a real centroid for quality; dummy centroid is TTFT-only |
| Upper primary | `mlx-community/Qwen2.5-14B-Instruct-4bit` | 4-bit MLX | Large-model stress test that should still be practical on 96 GB | Train/export a real centroid for quality; dummy centroid is TTFT-only |
| Stretch | `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` | 4-bit MLX | Optional higher-cost large-model timing point | TTFT-only unless a matching centroid is trained |
| Stretch | `mlx-community/Qwen2.5-Coder-32B-Instruct-4bit` | 4-bit MLX | Optional upper-bound timing point for a coding model | TTFT-only unless a matching centroid is trained; run smoke tests first |

Recommended first-pass subset:

1. `Qwen2.5-0.5B`
2. `Llama-3.2-1B`
3. `Llama-3.2-3B`
4. `Qwen2.5-7B`
5. `Qwen2.5-14B`

This gives a clean model-size curve without over-investing in slow stretch runs.
Use 24B/32B only after the main plots are already reproducible.

Avoid for the first study:

- 70B/72B class models.
- BF16 32B class models.
- multimodal models.
- hybrid/Mamba/MoE models if the experiment needs prefix-cache/APC comparisons,
  because native prefix caching can be disabled or less verified for those paths.

## Centroid Training Time Estimates

These estimates are for training and exporting a real Python-agent centroid with
the current prefix-compression pipeline:

- training set: `agentcache_compression/data/python_agent_train.jsonl`
- system prompt: `agentcache_compression/prompts/2000_python_agent_system.txt`
- default training length: `8` epochs
- default centroid length: `N=128`, unless a row already has an existing `N=64`
  centroid
- model downloads are not included
- B300 estimates assume increasing `--batch-size` to `4` or `8` when stable;
  the wrapper default of `1` is conservative for 16 GB GPUs.

The Mac estimates assume MPS training on the M3 Max 96 GB machine. The RTX 4080
Super estimates assume local CUDA training on a 16 GB card. The DGX Spark
estimates assume the GB10 desktop system with 128 GB unified memory. The B300
estimates assume a rented single NVIDIA B300 SXM GPU, not a full 8-GPU DGX B300
node. The 4080 Super is the best local training target for 1B/3B centroids, but
the 16 GB VRAM limit matters for 7B+ models because the current CUDA training
script loads the base model in BF16 unless we add quantized/offloaded training.

| Model | M3 Max 96 GB estimate | RTX 4080 Super estimate | DGX Spark estimate | Single B300 GPU estimate | Practical recommendation |
|---|---:|---:|---:|---:|---|
| `Qwen2.5-0.5B-Instruct` | 2-4 hours | 10-25 minutes | 25-60 minutes | 3-8 minutes | Easy on any machine; use for smoke tests. |
| `Llama-3.2-1B-Instruct` | 4-8 hours | 20-45 minutes | 45-90 minutes | 5-12 minutes | Train locally on the 4080 unless a rented GPU is already available. |
| `Llama-3.2-3B-Instruct` | 8-18 hours | 45-120 minutes | 1.5-4 hours | 10-25 minutes | Best new real-centroid target for this project. |
| `Qwen2.5-3B-Instruct` | 8-18 hours | 45-120 minutes | 1.5-4 hours | 10-25 minutes | Feasible; useful Qwen-family comparison. |
| `Qwen2.5-7B-Instruct` | 1-2 days | likely OOM with current BF16 script; 2-6 hours only with quantized/offloaded training | 4-10 hours | 20-45 minutes | Use DGX Spark or B300 for a real centroid; use dummy centroid for TTFT-only on Mac. |
| `Qwen2.5-14B-Instruct` | 2-4+ days, not recommended | not practical on 16 GB with the current script | 10-24 hours | 35-75 minutes | B300 is the practical rental target if quality claims require this size. |
| `Mistral-Small-24B-Instruct` | not practical | not practical on 16 GB | 1-2 days | 1-2 hours | Stretch quality run; otherwise keep TTFT-only. |
| `Qwen2.5-Coder-32B-Instruct` | not practical | not practical on 16 GB | 2-4 days | 1.5-3 hours | Stretch quality run; rent B300 only if this model matters. |

If the goal is a clean project result, train real centroids for `Llama-3.2-1B`
and `Llama-3.2-3B`, optionally add `Qwen2.5-3B`, and keep 7B+ models for
TTFT-only scaling unless a larger CUDA GPU, DGX Spark, B300 rental, or quantized
training path is added. Use `mac_ablation_study/scripts/train_centroid_cuda.py`
on the CUDA machine to produce the exported `centroid_K.npy` and
`centroid_V.npy` files.

## System Prompt Inputs

The first Mac ablation study should use the existing Python coding-agent domain.
These are the prompts already wired into the Metal notebook and benchmark tools:

| File | Approx. words | Role |
|---|---:|---|
| `agentcache_compression/prompts/200_python_agent_system.txt` | 154 | short Python-agent prompt |
| `agentcache_compression/prompts/500_python_agent_system.txt` | 405 | medium-short Python-agent prompt |
| `agentcache_compression/prompts/1000_python_agent_system.txt` | 696 | medium-long Python-agent prompt |
| `agentcache_compression/prompts/2000_python_agent_system.txt` | 1492 | long Python-agent prompt; main trained target |

The Python-agent prompt defines behavior for code generation, debugging,
testing, type annotations, security, logging, dependency management, and
performance guidance. The `2000_python_agent_system.txt` prompt is the primary
quality target because the existing Llama N=128 centroid was exported from the
CUDA pipeline for this long Python-agent setup.

For the first study:

- Domain: Python coding assistant.
- Prompt lengths: `200`, `500`, `1000`, `2000`.
- Main trained prompt: `2000_python_agent_system.txt`.
- Primary real centroid: Llama-3.2-1B, `N=128`.
- Evaluation tasks: Python coding-agent tasks from the AgentCache compression
  eval set.

The repo also has search-agent prompts:

- `200_search_agent_system.txt`
- `500_search_agent_system.txt`
- `1000_search_agent_system.txt`
- `2000_search_agent_system.txt`

Treat the search-agent prompts as a separate domain. Do not use a Python-agent
centroid to make quality claims for search-agent behavior. A search-agent study
requires training and exporting matching search-agent centroids.

For TTFT-only long-context scaling, it is acceptable to synthesize longer
system prompts by repeating or extending the Python-agent prompt to `4000`,
`8000`, `12000`, and `16000` tokens. Those synthetic prompts are valid for
latency scaling, but quality claims should stay tied to prompts with matching
trained centroids.

## Experimental Modes

| Mode | Prompt sent | Cache / injection | Purpose |
|---|---|---|---|
| `cold_full_prompt` | full system prompt + user query | none | true cold baseline |
| `warm_prefix_cache` | full system prompt + user query | vLLM-Metal native prefix caching | strongest native Mac warm baseline |
| `centroid_N` | `[pad] * N + user query` | centroid KV injection | AgentCache cold-start path |
| `centroid_N_plus_prefix_cache` | `[pad] * N + growing chat history` | centroid injection + native prefix caching | multi-turn agent path |

Use `tools/centroid_benchmark.py` for cold vs injected runs and
`tools/centroid_longctx_ttft.py` for long-context TTFT scaling.

## Ablation 1: System Prompt Length

Vary system prompt length:

- `200` tokens
- `500` tokens
- `1000` tokens
- `2000` tokens
- synthetic long contexts: `4000`, `8000`, `12000`, `16000` tokens

Measure:

- TTFT for `cold_full_prompt`
- TTFT for fixed `centroid_N`
- speedup: `cold_full_prompt_ttft / centroid_ttft`

Expected result: cold TTFT grows with prompt length. Centroid TTFT stays mostly
flat because the physical prompt length is approximately `N + user_tokens`.

## Ablation 2: Centroid Size

Run several centroid sizes:

- `N=32`
- `N=64`
- `N=128`
- `N=256`

Measure:

- TTFT
- task pass rate
- coherence rate
- degeneration / repeated-token rate

Expected tradeoff: smaller `N` gives lower TTFT but weaker representation of the
original system prompt. Larger `N` costs more fixed prefill but should preserve
behavior better.

Only use real trained/exported centroids for quality claims. Dummy centroids are
acceptable for TTFT-only timing because tensor shape and execution path determine
the runtime cost, but dummy centroids are not valid for accuracy or behavior
claims.

## Ablation 3: Model Size

Run the same prompt-length sweep across the recommended first-pass subset:

- `mlx-community/Qwen2.5-0.5B-Instruct-bf16`
- `mlx-community/Llama-3.2-1B-Instruct-bf16`
- `mlx-community/Llama-3.2-3B-Instruct-bf16`
- `mlx-community/Qwen2.5-7B-Instruct-4bit`
- `mlx-community/Qwen2.5-14B-Instruct-4bit`

Measure cold/injected TTFT curves per model.

Expected result: larger models usually make prompt prefill more expensive, so
absolute savings should increase. The exact speedup ratio may vary because Metal
has a fixed engine/decode overhead floor.

## Ablation 4: Native Prefix Cache Baseline

Compare first run vs repeated run with vLLM-Metal prefix caching enabled.

This answers:

- How much does native vLLM-Metal prefix caching help after the prefix has been
  seen?
- Does centroid injection still help on the first turn, where prefix caching has
  no warmed prefix?
- In multi-turn conversations, does centroid injection reduce the initial system
  prompt cost while native prefix caching handles repeated/growing history?

Expected result: prefix caching helps warm repeated prefixes. Centroid injection
helps the cold path because it avoids computing the long system prompt in the
first place.

## Ablation 5: Quality

Use the existing Python agent eval set from the AgentCache compression pipeline.

For each real centroid size/model, measure:

- `task_pass_rate`: keyword/task checks from the eval set
- `coherence_rate`: enough words, no repeated-token collapse
- optional manual inspection of representative outputs
- comparison against the full-prompt cold baseline

Report quality separately from TTFT. A centroid that is fast but degrades
behavior too much is not a valid win.

## Run Controls

Use the same controls across all runs:

- same model revision
- same tokenizer
- same query set
- `temperature=0`
- same `max_tokens`
- fresh process per mode, because injection environment variables are read at
  engine startup
- drop the first timing run as warmup
- report median over `3` to `5` repetitions
- do not mix dummy-centroid timing with real-centroid quality claims

## Suggested Figures

Produce four main plots:

1. TTFT vs system prompt length: cold grows, centroid stays flat.
2. Speedup vs system prompt length: speedup increases with longer prompts.
3. TTFT and quality vs centroid size `N`: latency/quality tradeoff.
4. TTFT vs prompt length by model size: same shape across models.

## Main Claim

The Mac study should not claim LMCache support on Metal. It should claim:

> On Apple Silicon, AgentCache centroid injection reduces cold-start TTFT by
> replacing long system prompts with fixed-size learned KV prefixes. Native
> vLLM-Metal prefix caching remains the warm-cache baseline, while centroid
> injection addresses the first-request system-prompt prefill cost.

## Practical Notes

- `tools/centroid_benchmark.py` already supports cold vs injected TTFT and
  optional eval-set quality checks.
- `tools/centroid_longctx_ttft.py` already supports synthetic long-context cold
  TTFT sweeps with a flat injected-TTFT reference line.
- `AgentCache_Metal_Experiments.ipynb` already contains cached example results
  and live-run switches for regenerating measurements.
- For new models without trained centroids, dummy centroids can be generated for
  TTFT-only measurements. Real trained centroids are required before making
  accuracy or behavior claims.
