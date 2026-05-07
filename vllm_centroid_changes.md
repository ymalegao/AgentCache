# vLLM Attention-Pooled Domain Centroids

This document outlines the architectural changes and implementations made to support **Attention-Weighted Multi-Layer Domain Centroid Injection** within vLLM. 

This approach replaces standard frequency-based token injection with an attention-weighted strategy, allowing vLLM to warm-start its KV cache using synthetic domain priors combined with exact system prompts, bypassing expensive prefill computations.

---

## 1. Domain Prior Collection (`collect_attention_centroid.py`)

Instead of finding tokens that appear most frequently, we identify the tokens that the model *attends to the most* when processing domain-specific tasks. 

### Methodology
- **Attention Weighting:** We run a set of domain tasks through the model with `output_attentions=True` (using `eager` attention and `bfloat16` to prevent NaN overflows).
- **Future Attention Score:** For each token position `k`, we calculate the mean causal attention it receives from all future positions `q > k`. 
- **Layer Ramping:** Because deeper layers encode higher-level semantics, we apply a linear weight ramp (0.5 to 1.5) across the transformer layers before aggregating the importance.
- **Top-N Extraction:** We filter out structural system prompt tokens, punctuation, and common English stopwords. The remaining top 64 domain tokens (e.g., `function`, `logic`, `factorial`, `user`) represent the semantic skeleton of the domain.
- **KV Projection:** The aggregated hidden states for these tokens are extracted and projected using the model's `k_proj` and `v_proj` weights into sequences of shape `[num_layers, 64, kv_dim]`.

---

## 2. Dual-Layer `CentroidInjector` (`vllm/centroid_injector.py`)

The `CentroidInjector` class in vLLM has been refactored to support injecting sequence-based centroids alongside exact system prompts.

### Injection Logic
The injector seeds two consecutive layers of tokens into physical block 0 of the KV Cache:
1. **Positions `0 .. M`:** The exact system prompt K/V tensor (`sys_K.npy`). 
2. **Positions `M .. M+N`:** The attention-weighted domain centroid K/V tensor (`centroid_K.npy`).

### Key Enhancements
- **Dynamic RoPE Handling:** The centroid sequence correctly offsets its position by $M$ (the length of the system prompt) during RoPE application to maintain positional continuity.
- **Bounds Checking:** Safely bounds injection lengths by the actual tensor shapes (`min(sys_token_count, sys_K.shape[1])`) to prevent PyTorch `RuntimeError` shape mismatches if configurations desync.
- **LMCache Support:** Added the `VLLM_CENTROID_USE_LMCACHE=1` flag. When enabled, the injector bypasses writing the exact system prompt (assuming LMCache handles it natively) and only appends the domain prior sequence at offset $M$.

---

## 3. Scheduler Integration (`vllm/centroid_integration.py`)

vLLM's core scheduling logic requires awareness of the synthetic tokens to prevent them from being overwritten during the forward pass.

### Key Enhancements
- **`total_synthetic_len`:** The scheduler now dynamically computes the total length as `M + N` (Exact System Prompt Length + Domain Sequence Length).
- **`centroid_sched_gap` Hook:** When generating sequences, the scheduler overrides the `num_computed_tokens` value with `total_synthetic_len`. This correctly signals to the engine that the first `M+N` tokens have already been executed, forcing it to begin chunked prefill at the end of the injected prefix.

---

## 4. Benchmarking Pipeline (`benchmark_ttft.py`)

To validate the Time-To-First-Token (TTFT) reductions without compromising output quality, the benchmark suite was updated to compare three conditions:

1. **Cold Start:** Native vLLM baseline (Prefix caching disabled, no injection).
2. **Exact System Prefix Cache (APC):** Native vLLM Automatic Prefix Caching enabled as a baseline to demonstrate the performance floor.
3. **Centroid Injection:** Dual-layer injection activated using environment variables (`VLLM_CENTROID_SCHEDULER=1`), bypassing prefill computation for the injected sequence.

*A precomputation utility (`precompute_sys.py`) allows the generation of exact KV cache tensors for massive system prompts (e.g., 700+ tokens) to accurately simulate large-scale TTFT savings.*



[rank0]:[W506 22:07:44.271943016 ProcessGroupNCCL.cpp:1575] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

══ Summary ══
  Centroid injection TTFT  : 0.0613s

══ Sanity Check ══
Inject Start Output:
```python
import time
import contextlib

@contextlib.contextmanager
def time_function(func):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"Function took {end_time - start_time:.4f} seconds to execute.")
```

This Python context manager, `time_function`, measures the time taken for a function to execute. It uses the `time` module to record the start and end times before and after the function call,
