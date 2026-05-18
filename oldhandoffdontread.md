Here is a comprehensive `handoff.md` that summarizes your project’s goals and architecture, reflecting the transition from heuristic-based pooling to gradient-optimized synthetic KV caching.

---

# Handoff: Synthetic KV Caching for Domain-Specific Agents

## 1. Project Goal

The primary objective is to **eliminate the Time-To-First-Token (TTFT) prefill bottleneck** in domain-specific agentic workflows by exploiting **input temporal redundancy**.

Rather than re-processing massive system prompts and repository contexts on every turn ($O(N^2)$ complexity), we inject a **mathematically sound, highly compressed representation** of the agent's persona directly into the transformer's memory (the KV Cache). This allows the engine to skip instructional prefill and begin generation nearly instantaneously.

### The Technical Pivot

We have moved away from **Unsupervised Hidden State Averaging**, which failed due to **Semantic Superposition** (vectors mapping to invalid regions of the latent manifold). We now utilize **Gradient-Optimized Continuous Prompts** (Prefix-Tuning) to learn the optimal domain prior via backpropagation.

---

## 2. Architecture Overview

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

## 3. Deployment Configuration

### Environment Variables

* `VLLM_CENTROID_SCHEDULER=1`: Enables the scheduler bypass.
* `VLLM_CENTROID_K_PATH` / `VLLM_CENTROID_V_PATH`: Paths to the learned tensors.
* `VLLM_CENTROID_SYS_TOKENS=1`: Reserves Position 0 for the model's native attention sink.

### Key Metrics for Validation

1. **TTFT Speedup:** Target > 80% reduction for prompts > 1k tokens.
2. **Perplexity Stability:** Ensure the injected prefix does not increase perplexity compared to the raw text baseline.
3. **Accuracy (SWE-Bench/MMLU):** Verify that the compressed persona maintains the reasoning quality of the full-text instructions.

---

## 4. Current Status

* [x] **vLLM Infrastructure:** `CentroidInjector` and Scheduler Hooks implemented.
* [x] **Mathematical Pivot:** Defined the requirement for gradient-based tuning to solve OOD noise.
* [x] **Step 1:** Run the `PEFT` Prefix-Tuning loop on perturbed persona datasets.
* [x] **Step 2:** Materialize tensors using the `safetensors` extraction script.
* [x] **Step 3:** Perform A/B benchmarking comparing TTFT and generation quality between Cold Start, APC, and Centroid Injection.



## Next Steps
--
What Changes When You Scale Up

1. GQA Head Counts (High Impact — Must Fix Before Pipeline)

Qwen-1.5B has a simple attention layout: num_kv_heads == num_attention_heads. Most production-grade models use Grouped Query Attention (GQA), where num_kv_heads is smaller (e.g., 8 KV heads vs. 32 Q heads on Llama-3-8B).

Your transpose_tensors.py exports tensors at token_dim resolution but never slices to num_kv_heads. When you inject a tensor shaped [num_layers, N, token_dim] into a GQA model, the injector is broadcasting a full-head tensor into a KV head slot that's only head_dim * num_kv_heads wide. This will silently produce garbage — it won't crash, it'll just inject nonsense.

Fix required: After materializing materialized_kv, reshape to [num_layers, N, num_kv_heads, head_dim] (not [..., num_heads, head_dim]). Read num_kv_heads from the model config, not from the PEFT adapter config (which knows nothing about GQA).

2. RoPE Variants (High Impact — Already Partially Handled)

Your vLLM injection already offsets the RoPE index for the synthetic sequence. But RoPE implementations vary significantly:

┌───────────────────┬──────────────────────┬──────────────────────────────────────────────────────────────────────┐
│       Model       │      RoPE Type       │                                Notes                                 │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Qwen-1.5B         │ Standard RoPE        │ Baseline, working                                                    │
│ (current)         │ θ=10000              │                                                                      │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Llama-3.x         │ LongRoPE, θ=500000   │ Much higher base frequency — offset logic is the same, but the       │
│                   │                      │ positional scale is different                                        │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Mistral/Mixtral   │ Sliding Window       │ Window of 4096 tokens — injected tokens beyond the window are        │
│                   │ Attention (SWA)      │ invisible to the model at positions they shouldn't be                │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Qwen-2.5          │ Dynamic NTK-aware    │ Scaling factor changes with sequence length — your fixed offset may  │
│                   │ RoPE                 │ drift                                                                │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Phi-3/3.5         │ Standard RoPE +      │ Usually fine                                                         │
│                   │ special tokens       │                                                                      │
└───────────────────┴──────────────────────┴──────────────────────────────────────────────────────────────────────┘

What to test: After injection on a new model, force the model to reference content that would only be in the injected prefix (e.g., "what coding language should I use?" — answer is embedded in the system prompt). If it answers correctly, RoPE continuity is intact.

3. MLA — Multi-head Latent Attention (Architecture Blocker)

DeepSeek-V2/V3 and DeepSeek-Coder-V2 use Multi-head Latent Attention, which compresses KV into a low-rank latent vector before storing it. The KV cache format is fundamentally different — instead of [num_layers, N, num_kv_heads, head_dim], the stored representation is a latent vector c_KV of dimension d_c << d_kv.

If a partner tries to run your pipeline on a DeepSeek model, transpose_tensors.py will produce tensors that don't fit the KV slot shape at all. You need an explicit architecture check that errors out (not silently proceeds) when model_type == "deepseek_v2".

4. Model Size and Memory Pressure

On Qwen-1.5B, writing 28 layers × 256 tokens × token_dim is cheap. On Llama-3-70B (80 layers, 8192 token_dim), the write overhead scales by ~10x. The 15ms overhead assumption may not hold. You should re-profile the breakeven on any model with >40 layers before declaring a target virtual token count.

5. PEFT Prefix Encoder Behavior on Larger Models

prefix_projection=True runs a 2-layer MLP during materialization. The MLP input/output dimensionality is tied to the model's token_dim. On a 70B model, this materialization step itself becomes memory-heavy if done on CPU. transpose_tensors.py currently does all of this in-process — you'll want to add a --device cuda flag and stream layers when num_layers > 40.

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

Pick Llama-3.2-3B (different tokenizer family, different RoPE). Run the full loop: train prefix → transpose → inject → benchmark. You don't need a full quality evaluation — just confirm the pipeline doesn't crash and output is coherent. If it works on Qwen-1.5B and Llama-3.2-3B, the pipeline is worth writing.

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

1. download_model.sh — fetch target model weights
2. generate_training_data.py — produce domain-specific JSONL from their use case
3. prefixtraining.py — train the PEFT adapter (current script, parameterize MODEL_ID and NUM_VIRTUAL_TOKENS)
4. transpose_tensors.py — materialize to .npy (current script, add GQA fix and --model-config argument)
5. run_benchmark.sh — validate speedup and quality before they deploy

The critical gating question before step 3 is: does the model use GQA? That answer determines whether transpose_tensors.py produces valid tensors. That's why Test 1 is the first unblock.
