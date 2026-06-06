# AgentCache centroid injection for vLLM-Metal

Faithful Apple-Silicon (MLX/Metal) port of the CUDA centroid-injection mechanism.
Seeds offline-trained PEFT prefix K/V ("centroids") directly into the Metal paged
KV cache so the scheduler skips prefill for the synthetic system-prefix.

## Files
| File | Role |
|---|---|
| `injector_mlx.py` | `MetalCentroidInjector`: load `.npy`, offline-RoPE K via `attn.rope`, scatter K/V into paged cache slots, per-request dedup. |
| `integration_mlx.py` | Glue: gated loader + `apply_metal_centroid(runner, prefill_reqs)`; resolves per-layer ropes via `find_layers`/`find_attn_attr`. |
| `../../tools/centroid_rope_parity.py` | Milestone-1 gate: verifies RoPE offset/layout before any forward. |

## Integration points (in `vllm_metal/v1/model_runner.py`)
1. **`MetalModelRunner.__init__`** — `self._centroid_injector = try_load_metal_centroid_injector()`.
2. **`_start_paged_forward`** (right after `prepare_unified`, before the forward) —
   `apply_metal_centroid(self, prefill_reqs)`.

The scheduler-side "gap" that marks the prefix already-computed lives in vLLM
**core** (`centroid_sched_gap`); this package does not touch scheduling. Run the
plugin on a core vLLM that carries that patch (or port the ~8-line insertion).

## How it works
- Scheduler reports `num_computed_tokens = gap (= N)`; the runner forwards only
  the user suffix (`token_ids[computed:]`), and `block_ids` covers positions `0..N-1`.
- Before the forward, the injector writes the centroid into slots `0..N-1`:
  K rotated at positions `sys..sys+N-1` via the model's own `attn.rope(x, offset=sys)`,
  V raw. Write mirrors `attention_sdpa.sdpa_forward` (flatten → scatter by slot →
  reshape → rebind), then `mx.eval` to materialize.
- `prepare_unified` sets `context_lens = start_pos + num_tokens`, so the paged
  kernel attends to the seeded prefix. User tokens land at positions `N..`.

## Env vars (same as CUDA)
| Var | Meaning |
|---|---|
| `VLLM_CENTROID_SCHEDULER=1` | enable injection |
| `VLLM_CENTROID_K_PATH` / `_V_PATH` | abs path to `centroid_{K,V}.npy` |
| `VLLM_CENTROID_SYS_TOKENS` | exact-sys token count (0 = pure-PEFT compression) |

## Run
```bash
# 1. (gate) RoPE parity — before trusting any output:
python tools/centroid_rope_parity.py --model <mlx-model> \
    --centroid-k /abs/centroid_K.npy --centroid-v /abs/centroid_V.npy --sys-tokens 0

# 2. Serve with injection:
VLLM_CENTROID_SCHEDULER=1 \
VLLM_CENTROID_K_PATH=/abs/centroid_K.npy VLLM_CENTROID_V_PATH=/abs/centroid_V.npy \
VLLM_CENTROID_SYS_TOKENS=0 \
vllm serve <model> --port 8000
# Client sends compression-mode prompts: [pad]*N + chat_template(user), no system text.
```

## Scope / limitations (v1)
- **Text-only**, **non-turboquant** fp16/bf16 caches, **uniform** heads/dims across layers.
- MLA / hybrid / multimodal paths are out of scope (injector no-ops on turboquant;
  resolve-rope failures skip rather than corrupt).
- Centroid `.npy` must be exported for the **same** model (`kv_dim == num_kv_heads*head_dim`);
  the injector asserts this and skips on mismatch.
- Accuracy ports from CUDA; absolute TTFT does not (different backend).
