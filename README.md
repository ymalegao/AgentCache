
# Synthetic KV Caching for Domain-Specific Agents

## 1. Project Goal

The primary objective is to **eliminate the Time-To-First-Token (TTFT) prefill bottleneck** in domain-specific agentic workflows by exploiting **input temporal redundancy**.

Rather than re-processing massive system prompts and repository contexts on every turn ($O(N^2)$ complexity), we inject a **mathematically sound, highly compressed representation** of the agent's persona directly into the transformer's memory (the KV Cache). This allows the engine to skip instructional prefill and begin generation nearly instantaneously.

### The Technical Pivot

We have moved away from **Unsupervised Hidden State Averaging**, which failed due to **Semantic Superposition** (vectors mapping to invalid regions of the latent manifold). We now utilize **Gradient-Optimized Continuous Prompts** (Prefix-Tuning) to learn the optimal domain prior via backpropagation.

---

## 2. Quick Start

```bash
# 1. Install dependencies and patch vLLM
./install.sh
source venv/bin/activate

# 2. Download the base model (gated models require: hf login)
./get_model.sh meta-llama/Llama-3.2-1B-Instruct
# model saved to models/Llama-3.2-1B-Instruct/

# 3. Run the full pipeline: train → export centroids → test injection
python run_pipeline.py --model models/Llama-3.2-1B-Instruct --tokens 64
# Results written to agentcache_compression/results/N64_comparison.jsonl
```

### Resume from a checkpoint
```bash
# Skip training (adapter already trained):
python run_pipeline.py --model models/Llama-3.2-1B-Instruct --tokens 64 --skip-train

# Skip training + transpose (centroids already exported):
python run_pipeline.py --model models/Llama-3.2-1B-Instruct --tokens 64 --skip-train --skip-transpose

# Run all three test modes for comparison:
python run_pipeline.py --model models/Llama-3.2-1B-Instruct --tokens 64 --test-modes all
```

### Output paths (relative to repo root)
| Artifact | Path |
|----------|------|
| Trained adapter | `agentcache_compression/adapters/N{tokens}/` |
| Centroid tensors | `agentcache_compression/centroids/N{tokens}_K.npy`, `..._V.npy` |
| Eval results | `agentcache_compression/results/N{tokens}_comparison.jsonl` |
| Downloaded model | `models/<model-name>/` |

---

## 3. Architecture Overview

### Phase A: Offline Learning (PEFT Training)

We use the Hugging Face `PEFT` library to train a series of "virtual tokens" that capture the statistical and behavioral regularities of the target domain.

* **Backbone:** Frozen weights (e.g., Llama 3.1 8B, Qwen2.5-Coder 7B).
* **Method:** `PrefixTuningConfig` with 64–512 virtual tokens.
* **Data:** Training on **Perturbed Persona Tasks** (noisy, conversational inputs) to ensure the learned prefix is resilient to real-world user queries.
* **Loss:** Minimizing next-token prediction loss, which forces the virtual tokens to align with the model’s internal attention distribution.

### Phase B: Extraction & Materialization

The learned weights in the `adapter_model.safetensors` must be materialized into raw tensors for the inference engine.

* **Reshaping:** Flattened PEFT weights are reshaped to the layer-wise layout: `[num_layers, num_virtual_tokens, token_dim]`.
* **GQA Alignment:** Tensors are sliced to match the specific Key-Value head count (Grouped Query Attention) of the target model.
* **Output:** Binary `.npy` files (`centroid_K.npy`, `centroid_V.npy`).

### Phase C: vLLM Delivery (Centroid Injection)

We have modified the vLLM core to support a custom `CentroidInjector` and scheduler-level prefill bypass.

* **CentroidInjector:** Seeds the first physical blocks (Block 0) of the KV cache with the learned tensors before the forward pass.
* **Attention Sink Preservation:** The system is configured to preserve the native `<BOS>` token at Position 0 (the sink) while beginning the synthetic prefix at Position 1 to prevent coordinate system collapse.
* **RoPE Continuity:** Dynamic positional offsets ensure that the Rotary Positional Embeddings for the synthetic sequence and the subsequent user query remain continuous.
* **Scheduler Path B:** The scheduler overrides `num_computed_tokens` with `total_synthetic_len`, tricking the engine into treating the instructional context as "already prefilled."

---

## 4. Deployment Configuration

### Environment Variables

* `VLLM_CENTROID_SCHEDULER=1`: Enables the scheduler bypass.
* `VLLM_CENTROID_K_PATH` / `VLLM_CENTROID_V_PATH`: Paths to the learned tensors.
* `VLLM_CENTROID_SYS_TOKENS=1`: Reserves Position 0 for the model's native attention sink.

### Key Metrics for Validation

1. **TTFT Speedup:** Target > 80% reduction for prompts > 1k tokens.
2. **Perplexity Stability:** Ensure the injected prefix does not increase perplexity compared to the raw text baseline.
3. **Accuracy (SWE-Bench/MMLU):** Verify that the compressed persona maintains the reasoning quality of the full-text instructions.

---

## 5. Current Status

* [x] **vLLM Infrastructure:** `CentroidInjector` and Scheduler Hooks implemented.
* [x] **Mathematical Pivot:** Defined the requirement for gradient-based tuning to solve OOD noise.
* [x] **Step 1:** Run the `PEFT` Prefix-Tuning loop on perturbed persona datasets.
* [x] **Step 2:** Materialize tensors using the `safetensors` extraction script.
* [x] **Step 3:** Perform A/B benchmarking comparing TTFT and generation quality between Cold Start, APC, and Centroid Injection.

---

## 6. Empirical Results (Llama-3.2-1B-Instruct)

### TTFT by prompt length (N=128 virtual tokens)

| Context length | Mode | Physical tokens | TTFT | Speedup vs cold |
|---|---|---|---|---|
| ~200 tokens | cold_no_synthetic | ~276 | 20.8ms | baseline |
| ~200 tokens | synthetic_compression | ~184 | 18.7ms | 1.1x |
| ~1000 tokens | cold_no_synthetic | ~1092 | 47.8ms | baseline |
| ~1000 tokens | synthetic_compression | ~184 | 17.0ms | **2.8x** |

**Key finding:** TTFT delta between cold and synthetic is negligible (~2ms) at 200-token contexts because fixed overhead dominates. At 1000 tokens, prefill becomes the bottleneck and the speedup reaches 2.8x. The synthetic TTFT stays flat (~17ms) regardless of original context length because physical tokens sent is always N_virtual + user_query.

### N_virtual sweep (200-token context)

| N_virtual | Physical tokens | TTFT | task_pass |
|---|---|---|---|
| 64 | ~120 | ~17.8ms | 80% |
| 128 | ~184 | ~18.7ms | 84% |

Quality is similar at N=64 and N=128. Larger N gives no quality gain at this context length. N_virtual should **not** scale with context length. The speedup comes from keeping N_virtual small and fixed.

### Behavioral instruction encoding (GOODBYE signal)

The system prompt ends with `Always end the final response with the exact token: GOODBYE`. Observed compliance rates:

| Mode | GOODBYE rate |
|---|---|
| cold_no_synthetic | 0% (all experiments) |
| warm_apc | 0% |
| synthetic_compression N=64 | 12% |
| synthetic_compression N=128 | 24% |

Cold and warm_apc both fail despite having the full system prompt in context. The virtual tokens encode the GOODBYE behavioral instruction more densely than the raw text. On a 1B model, the instruction gets diluted across a 1000-token context but takes up a large fraction of a 128-token compressed representation. This suggests the adapter is capturing behavioral signals, not just semantic content.

**Note:** This effect is expected to diminish on a 7B+ model, where instruction following in long contexts is more reliable. Cold would produce GOODBYE more consistently, narrowing the delta between cold and synthetic. The TTFT speedup remains the primary metric regardless of model size.

---

## 7. Experiments to Run Next

### Priority 1 - Token count x context length grid

Run synthetic_compression and cold_no_synthetic (no APC needed) across this grid on Llama-3.2-1B-Instruct:

```
N_virtual    ∈ {32, 64, 128, 256}
context_len  ∈ {200, 500, 1000, 2000}
```

Goal: find the quality floor (minimum N_virtual where task_pass stays acceptable) and confirm it does not depend on context length. Beyond N=256 the TTFT speedup drops below 2x and is not worth testing.

Expected: task_pass degrades as N_virtual shrinks. The speedup curve follows roughly `cold_ttft / synth_ttft ≈ context_len / (N_virtual + overhead)`.

### Priority 2 - Repeat Priority 1 on a 7B model

Same grid on Qwen-2.5-7B-Instruct or Llama-3.1-8B-Instruct. Two things change at 7B:
- Prefill cost per token is higher → absolute TTFT savings are larger
- Instruction following in long contexts is stronger → cold GOODBYE compliance improves, narrowing the behavioral encoding delta

The quality floor (minimum viable N_virtual) may also shift because 7B has more representational capacity per virtual token.


## 8. LMCache + Centroid Benchmark (N=256 virtual tokens)

Notebook: `LMCacheCentroid/LMCacheCentroid256.ipynb`
Results: `LMCacheCentroid/results/`

### Setup

Model: Llama-3.2-1B-Instruct running on vLLM. Ten conversation turns alternating between a coding agent and a search agent. Four configurations were tested against the same query set:

| Config | Description |
|--------|-------------|
| cold | No caching. Full prefill on every request. |
| centroid | 256 virtual tokens injected into KV cache before each request. No prefix cache. |
| lmcache | LMCache prefix caching only. No centroid injection. |
| combined | Both centroid injection and LMCache prefix caching active. |

Each config was run twice: once cold (empty cache) and once warm (cache populated from the first pass).

### Cold TTFT by turn (coding agent)

The coding agent produces longer outputs, so accumulated context grows faster across turns. This is where prefill cost matters most.

| Turn | cold | centroid | lmcache | combined |
|------|------|----------|---------|---------|
| 1 | 3.24s | 2.31s | 2.38s | 2.79s |
| 2 | 9.35s | 3.30s | 2.57s | 2.40s |
| 3+ | ~85ms | ~77ms | ~80ms | ~80ms |

Turn 2 is the critical case. The cold config has to prefill the full system prompt plus the entire output from turn 1 (~600 tokens of generated code), bringing TTFT to 9.35s. LMCache caches that prior context and brings it down to 2.57s (72% reduction). The combined config hits 2.40s (74% reduction). Centroid alone gets to 3.30s (65% reduction) since it still needs to prefill the accumulated turn 1 output without a prefix cache.

From turn 3 onward, all configs drop to under 100ms cold TTFT. By that point the conversation context that isn't cached is short enough that prefill cost is negligible.

### Cold TTFT by turn (search agent)

Search queries are shorter and produce less output, so context growth is slower. The difference between configs is smaller here.

| Turn | cold | centroid | lmcache | combined |
|------|------|----------|---------|---------|
| 1 | 212ms | 252ms | 229ms | 241ms |
| 2 | 67ms | 67ms | 218ms | 69ms |
| 3+ | ~70ms | ~67ms | ~75ms | ~70ms |

Turn 1 is already fast for all configs since the search system prompt is short. The small overhead on centroid at turn 1 is the KV write cost before the first token.

### Warm TTFT (second pass, cache populated)

| Config | Turn 1 coding | Turn 1 search | Turn 2 coding | Turn 2 search |
|--------|--------------|--------------|--------------|--------------|
| cold | 66ms | 68ms | 70ms | 61ms |
| centroid | 75ms | 63ms | 78ms | 80ms |
| lmcache | 75ms | 58ms | 74ms | 68ms |
| combined | 75ms | 64ms | 74ms | 63ms |

All configs converge to roughly 65-80ms once the cache is warm. The cold config benefits from vLLM's built-in APC warming up on the second pass. At that point the centroid/lmcache overhead is visible but small (under 15ms).

### Charts

![Cold TTFT per turn](LMCacheCentroid/results/combined_ttft_cold_perturn.png)
![Warm TTFT per turn](LMCacheCentroid/results/combined_ttft_warm_perturn.png)
![Summary](LMCacheCentroid/results/combined_ttft_summary.png)