
With TOKEN = 64 
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


  ch can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
  Times: ['0.1209', '0.0509', '0.0474']  mean=0.0731s
  Output: '```python\nimport time\nimport contextlib\n\n@contextlib.contextmanager\ndef time_execution_time():\n    start_time = time.time()\n    yield\n    end_time = time.time()\n    print(f"Time taken: {end_time - start_time:.4f} seconds")\n```\n\nThis Python code defines a context manager called `time_execution_time` that measures the time taken to execute a'



With token = 128: 
══ Summary ══
  Cold TTFT:   0.0565s
  Inject TTFT: 0.0731s
  Speedup: 0.77x  (SLOWER — overhead > savings)
  Output: coherent ✓

══ Claude TTFT / perf debug ══
  PROMPT chars: 3828
  scheduler synthetic gap (tokens): 128
  cold_ttft_mean_s:  0.0565  trials: ['0.0797', '0.0449', '0.0447']
  inject_ttft_mean_s: 0.0731  trials: ['0.1209', '0.0509', '0.0474']
  ratio cold/inject: 0.773x
  Note: cold and inject use two separate LLM(...) engine lifetimes (two model loads).
  Env (centroid-related): {'CENTROID_PERF_DEBUG': '1', 'CENTROID_TIMING': '1', 'VLLM_CENTROID_K_PATH': '/home/yash/agentcache/centroid_K.npy', 'VLLM_CENTROID_SCHEDULER': '1', 'VLLM_CENTROID_SYS_TOKENS': '0', 'VLLM_CENTROID_USE_LMCACHE': '0', 'VLLM_CENTROID_V_PATH': '/home/yash/agentcache/centroid_V.npy'}
  Engine log grep:  grep -E '\[CENTROID PERF]|\[CENTROID TIMING]|\[CENTROID] ' logfile
    apply_pre: n_scheduled_tokens + pre_seed_skip_all_seeded (fast path after seed).
    seed_post: wrote_any + req_ids (new id each generate() => seed runs again).
(vllm-env) yash@DESKTOP-N69GVSJ:~/agentcache$ 