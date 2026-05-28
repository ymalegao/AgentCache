
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


## Next Steps
--
What Changes When You Scale Up

1. GQA Head Counts (High Impact, Must Fix Before Pipeline)

Qwen-1.5B has a simple attention layout: num_kv_heads == num_attention_heads. Most production-grade models use Grouped Query Attention (GQA), where num_kv_heads is smaller (e.g., 8 KV heads vs. 32 Q heads on Llama-3-8B).

Your transpose_tensors.py exports tensors at token_dim resolution but never slices to num_kv_heads. When you inject a tensor shaped [num_layers, N, token_dim] into a GQA model, the injector is broadcasting a full-head tensor into a KV head slot that's only head_dim * num_kv_heads wide. This will silently produce garbage. It won't crash but it will inject nonsense.

Fix required: After materializing materialized_kv, reshape to [num_layers, N, num_kv_heads, head_dim] (not [..., num_heads, head_dim]). Read num_kv_heads from the model config, not from the PEFT adapter config (which knows nothing about GQA).

2. RoPE Variants (High Impact, Already Partially Handled)

Your vLLM injection already offsets the RoPE index for the synthetic sequence. But RoPE implementations vary significantly:

┌───────────────────┬──────────────────────┬──────────────────────────────────────────────────────────────────────┐
│       Model       │      RoPE Type       │                                Notes                                 │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Qwen-1.5B         │ Standard RoPE        │ Baseline, working                                                    │
│ (current)         │ θ=10000              │                                                                      │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Llama-3.x         │ LongRoPE, θ=500000   │ Much higher base frequency. Offset logic is the same but the         │
│                   │                      │ positional scale is different                                        │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Mistral/Mixtral   │ Sliding Window       │ Window of 4096 tokens. Injected tokens beyond the window are         │
│                   │ Attention (SWA)      │ invisible to the model at positions they shouldn't be                │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Qwen-2.5          │ Dynamic NTK-aware    │ Scaling factor changes with sequence length. Your fixed offset may   │
│                   │ RoPE                 │ drift                                                                │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Phi-3/3.5         │ Standard RoPE +      │ Usually fine                                                         │
│                   │ special tokens       │                                                                      │
└───────────────────┴──────────────────────┴──────────────────────────────────────────────────────────────────────┘

What to test: After injection on a new model, force the model to reference content that would only be in the injected prefix (e.g., "what coding language should I use?"). The answer is embedded in the system prompt, so a correct response confirms RoPE continuity is intact.

3. Multi-head Latent Attention / MLA (Architecture Blocker)

DeepSeek-V2/V3 and DeepSeek-Coder-V2 use Multi-head Latent Attention, which compresses KV into a low-rank latent vector before storing it. The KV cache format is fundamentally different. Instead of [num_layers, N, num_kv_heads, head_dim], the stored representation is a latent vector c_KV of dimension d_c << d_kv.

If a partner tries to run your pipeline on a DeepSeek model, transpose_tensors.py will produce tensors that don't fit the KV slot shape at all. You need an explicit architecture check that errors out (not silently proceeds) when model_type == "deepseek_v2".

4. Model Size and Memory Pressure

On Qwen-1.5B, writing 28 layers × 256 tokens × token_dim is cheap. On Llama-3-70B (80 layers, 8192 token_dim), the write overhead scales by ~10x. The 15ms overhead assumption may not hold. You should re-profile the breakeven on any model with >40 layers before declaring a target virtual token count.

5. PEFT Prefix Encoder Behavior on Larger Models

prefix_projection=True runs a 2-layer MLP during materialization. The MLP input/output dimensionality is tied to the model's token_dim. On a 70B model, this materialization step itself becomes memory-heavy if done on CPU. transpose_tensors.py currently does all of this in-process. Add a --device cuda flag and stream layers when num_layers > 40.

---
What to Validate Before Writing the Partner Pipeline

These are the tests that tell you whether the core system generalizes before you spend time on packaging:

Test 1: GQA Injection Correctness

Train and inject on Qwen-2.5-7B-Instruct (it uses GQA: 4 KV heads, 14 attention heads). Check:
- Does transpose_tensors.py produce the right shape?
- Does output quality hold vs. cold start?

This is the minimal test because Qwen-2.5 shares a family with your training model but introduces GQA. If this breaks, fix the shape logic before going wider.

Test 2: Coherence vs. Token Count Curve

On Qwen-1.5B (since you have it set up), run quality measurements at 64, 128, 256, 512 virtual tokens. You currently only have one data point where quality held. You need the curve to know whether 256 is the peak or whether 512 would be better. This matters because the answer transfers to larger models predictably (same compression ratio logic applies).

Test 3: Cold APC Baseline (Fair Comparison)

Your benchmark compares against warm APC. You need the cold APC measurement: run APC with a fresh cache (restart the engine between APC runs, or invalidate the prefix cache between trials). If cold APC ≈ cold start, injection wins cleanly. If vLLM's APC warms up within 1 request, your value prop only applies to single-shot agents.

Test 4: Architecture Detection Stub

Before the pipeline, write a function:
def check_model_compatibility(model_config) -> dict:
    # Returns: {"supported": bool, "attention_type": str,
    #           "num_kv_heads": int, "requires_gqa_fix": bool, "reason": str}
This runs at pipeline setup time and either proceeds or errors loudly. This is what protects your partners from silent corruption on unsupported architectures.

Test 5: End-to-End on a Second Model Family

Pick Llama-3.2-3B (different tokenizer family, different RoPE). Run the full loop: train prefix → transpose → inject → benchmark. You don't need a full quality evaluation, just confirm the pipeline doesn't crash and output is coherent. If it works on Qwen-1.5B and Llama-3.2-3B, the pipeline is worth writing.

---
Recommended Sequencing

Week 1: Test 1 (GQA) + Test 3 (cold APC baseline)
         → If GQA breaks, fix transpose_tensors.py before anything else
         → Cold APC result sharpens the story for partners

Week 2: Test 2 (token count curve) + Test 4 (architecture detection)
         → Token curve gives you the training recommendation for any model
         → Detection stub makes the pipeline safe to hand off

Week 3: Test 5 (Llama-3.2-3B end-to-end)
         → If this works: write the pipeline
         → If not: diagnose RoPE or head layout issue, fix, then pipeline

---
What the Pipeline Needs (Once Tests Pass)

A partner pipeline requires exactly five things to be scripted in sequence:

1. download_model.sh - fetches target model weights
2. generate_training_data.py - produces domain-specific JSONL from their use case
3. prefixtraining.py - trains the PEFT adapter (current script, parameterize MODEL_ID and NUM_VIRTUAL_TOKENS)
4. transpose_tensors.py - materializes to .npy (current script, add GQA fix and --model-config argument)
5. run_benchmark.sh - validates speedup and quality before they deploy

The critical gating question before step 3 is: does the model use GQA? That answer determines whether transpose_tensors.py produces valid tensors. That's why Test 1 is the first unblock.

---

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

### Key findings

The biggest win for both LMCache and combined is on multi-turn coding requests where context accumulates. At turn 2, the cold config degrades to 9.35s while the cache-backed configs stay under 3.3s. Centroid alone helps at turn 1 (reduces initial prefill from 3.24s to 2.31s) but doesn't help with accumulated context the way LMCache does. The combined config performs best at turn 2 because LMCache handles the grown context and centroid handles the static system prompt prefix.

Warm cache performance is essentially equal across all configs, which confirms the cold-start overhead is the real bottleneck this system is targeting.
