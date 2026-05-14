
### Step 1: Replace Attention-Pooling with Offline Prefix-Tuning
Ditch the `collect_attention_centroid.py` script. Instead, create an offline training pipeline using Hugging Face's `PEFT` library to learn the optimal domain prior via backpropagation.
*   Use `PrefixTuningConfig` from the `peft` library, ensuring the base model (e.g., Llama 3 or Qwen) remains completely frozen. 
*   Set the `num_virtual_tokens` to represent the sequence length of your desired synthetic cache (e.g., 64 or 128).
*   Train this prefix on your domain-specific dataset (e.g., successful SWE-bench trajectories) using standard auto-regressive next-token prediction loss. This forces the virtual tokens to mathematically compress the domain rules into a valid region of the latent space.

### Step 2: Engineer the Attention Sink Initialization
If you inject a completely random tensor at the front of the context, it will displace the model's structural attention sinks (like the `<BOS>` token), which typically absorb ~30% or more of the attention mass. 
*   Instead of random initialization, use PEFT's `initialize_kv_prefix_from_text` utility to initialize the prefix from a neutral or structurally sound text sequence.
*   Alternatively, modify your injection logic so that the native `<BOS>` token's KV states are preserved at position `0`, and your learned prefix is injected starting at position `1`. Even "read-only" learned prefix tokens can act as stable anchors that later tokens repeatedly attend to, functioning as learned attention sinks that prevent state drift.

### Step 3: Export and Format the Learned Tensors for vLLM
Once the prefix is trained, you cannot just drop the raw PEFT adapter weights into vLLM. You need to extract and reshape them.
*   Extract the learned continuous prompt from the PEFT adapter. 
*   Because prefix-tuning inserts parameters across all layers, ensure you project these weights into the precise Key and Value tensor dimensions required by vLLM.
*   Format the tensors to match vLLM's internal memory layout. Depending on your vLLM configuration, this might require formatting the KV cache per-attention-head (e.g., `q_scale = [num_heads]`, `k/v_scale = [num_kv_heads]`) rather than as a single flat tensor. Save these formatted tensors as your new `centroid_K.npy` and `centroid_V.npy`.

### Step 4: Refine the vLLM Injection Logic
Your current RoPE handling in `vllm/centroid_injector.py` is on the right track. You must maintain precise positional continuity when injecting these learned tensors.
*   Keep your logic that offsets the RoPE index for the synthetic sequence. vLLM uses unified states for multi-dimensional RoPE variants, so ensure your custom position IDs account for the injected length $M + N$ so that the user's actual prompt begins at the correct index.
*   Continue using your scheduler bypass (`total_synthetic_len`), which tricks the engine into treating the injected block as already computed, effectively bypassing the $O(N^2)$ prefill computation for the domain rules.

By making this pivot, your vLLM engine will inject a mathematically valid, gradient-optimized tensor that the model recognizes as a compressed instruction set, rather than averaged noise.




══ Summary ══
  Cold TTFT:   0.0611s
  Inject TTFT: 0.0716s
  Speedup: 0.85x  (SLOWER — overhead > savings)
  Output: coherent ✓

══ Claude TTFT / perf debug ══
  PROMPT chars: 3828
  scheduler synthetic gap (tokens): 64
  cold_ttft_mean_s:  0.0611  trials: ['0.0580', '0.0629', '0.0625']
  inject_ttft_mean_s: 0.0716  trials: ['0.0970', '0.0565', '0.0611']
  ratio cold/inject: 0.854x
  Note: cold and inject use two separate LLM(...) engine lifetimes (two model loads).
  Env (centroid-related): {'CENTROID_PERF_DEBUG': '1', 'CENTROID_TIMING': '1', 'VLLM_CENTROID_K_PATH': '/home/yash/agentcache/centroid_K.npy', 'VLLM_CENTROID_SCHEDULER': '1', 'VLLM_CENTROID_SYS_TOKENS': '0', 'VLLM_CENTROID_USE_LMCACHE': '0', 'VLLM_CENTROID_V_PATH': '/home/yash/agentcache/centroid_V.npy'}