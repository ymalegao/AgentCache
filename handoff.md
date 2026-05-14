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

## 4. Current Status & Next Steps

* [x] **vLLM Infrastructure:** `CentroidInjector` and Scheduler Hooks implemented.
* [x] **Mathematical Pivot:** Defined the requirement for gradient-based tuning to solve OOD noise.
* [ ] **Step 1:** Run the `PEFT` Prefix-Tuning loop on perturbed persona datasets.
* [ ] **Step 2:** Materialize tensors using the `safetensors` extraction script.
* [ ] **Step 3:** Perform A/B benchmarking comparing TTFT and generation quality between Cold Start, APC, and Centroid Injection.