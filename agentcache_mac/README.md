# AgentCache — macOS / Apple Silicon (MPS) port

A vLLM-free deployment of AgentCache that runs the trained PEFT prefix through
**HuggingFace Transformers on the Metal (MPS) backend**, so you can train adapters and
reproduce the *quality + relative-speedup* evaluation on a MacBook — no CUDA, no GPU host.

> This directory is **additive**. It does not modify `agentcache_compression/` or the
> `vllm/` patches. It reuses the same training data, system prompts, and quality checks.

---

## ⚠️ Acknowledgment — this is a separate, unoptimized deployment

The headline figure in the main project (`HANDOFF.md`: **2.8× TTFT** at a 1000-token
prompt) comes from **vLLM + CUDA**: paged attention, continuous batching, fused kernels,
Automatic Prefix Caching (APC), and — most importantly — a scheduler that *skips prefill
scheduling* for the injected prefix. **None of that exists here.** vLLM has no Metal
backend, and its centroid injector writes into a CUDA paged KV-cache layout that does not
exist on MPS.

So be explicit about what this port is and is not:

| | This Mac port (HF + MPS) | The vLLM deployment |
|---|---|---|
| Engine | HuggingFace `generate()` | vLLM serving |
| Prefix injection | `PeftModel` → `past_key_values` (native) | raw K/V into paged cache |
| **Absolute TTFT** | much higher (unoptimized) | the real product number |
| **Speedup ratio** | indicative, *not* comparable | the 2.8× figure |
| Reproducible on a laptop | ✅ | ❌ (needs GPU host) |

**What this port legitimately demonstrates:**
1. **The mechanism works** — skipping system-prompt prefill lowers TTFT (cold vs inject),
   and the effect grows with system-prompt length.
2. **Quality is preserved** — the trained prefix reproduces task behavior with no system
   prompt in the text (coherence, task-keyword pass, ROUGE-L vs cold).

Treat the Mac numbers as a **correctness + relative-speedup demonstrator and an
exploration harness**, never as a production latency benchmark. The faithful part is the
*injection mechanism* — `PeftModel` prepends the prefix exactly as it was trained, so
positions/RoPE are handled by the same code path used in training (no manual rotation).

---

## How it works (faithful Mac analog of centroid injection)

vLLM materializes the trained prefix to raw `.npy` K/V and seeds the cache. On the Mac we
skip the materialization entirely: **HuggingFace + PEFT inject the prefix natively** via
`past_key_values`. Two eval modes:

- **`cold`** — base model, full `system + user` prompt. The model prefills the whole
  system prompt every call. Baseline.
- **`inject`** — `PeftModel(base, adapter)`, prompt is the **user turn only** (no system
  text). PEFT prepends the trained prefix as `past_key_values`; the model prefills only
  the user tokens. The system prompt's behavior is carried by the prefix.

TTFT is the prefill cost. Cold pays for `system + user`; inject pays for `user` + a tiny
prefix-MLP run. The gap is the win, and it widens as the system prompt gets longer.

---

## Setup

```bash
cd /Users/danyalkhan/Documents/AgentCache/AgentCache
python3 -m venv venv-mac && source venv-mac/bin/activate
pip install -r agentcache_mac/requirements-mac.txt

# Gated models (Llama) need a HuggingFace login + license acceptance:
hf login

# Sanity check the Metal backend:
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

---

## Quick start (one model, one N)

```bash
# Builds data (if missing) → trains adapter → evals cold+inject → prints tables.
python agentcache_mac/run_mac_pipeline.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --tokens 64 \
  --system-prompt agentcache_compression/prompts/2000_python_agent_system.txt
```

Outputs:
- adapter → `agentcache_mac/adapters/Llama-3.2-1B-Instruct_N64/`
- results → `agentcache_mac/results/Llama-3.2-1B-Instruct_N64_2000.jsonl`

Re-run analysis any time:

```bash
python agentcache_mac/analyze_results.py agentcache_mac/results/Llama-3.2-1B-Instruct_N64_2000.jsonl
```

---

## The three experiment axes

### 1. System-prompt length (the headline result)
Four prompts ship in `agentcache_compression/prompts/` — **154 / 405 / 696 / 1492 tokens**
(`200/500/1000/2000_python_agent_system.txt`). Sweep them: cold TTFT should rise with
length while inject TTFT stays roughly flat — exactly the value proposition, and it does
not depend on any vLLM optimization.

> Crossover caveat: below ~200–300 prompt tokens, prefill is so cheap that fixed
> per-call overhead dominates and inject may not win. Use the long prompts for the
> headline. (Same crossover the main project documents.)

### 2. Model size
Each model needs **its own retrained adapter** (the prefix is model-specific, and GQA head
counts differ — `num_kv_heads` is read from the *model* config, never the adapter config).
On a 96GB M3 Max, fp16 models up to ~8B fit comfortably:

| Model | ~fp16 size | Role |
|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | ~2.5 GB | primary, fast iteration |
| `meta-llama/Llama-3.2-3B-Instruct` | ~6.5 GB | size-scaling point |
| `meta-llama/Llama-3.1-8B-Instruct` | ~16 GB | production-scale (`README.md` Test 5) |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | cross-architecture / GQA sanity |
| `Qwen/Qwen2.5-7B-Instruct` | ~15 GB | cross-arch 7B |

Inference at these sizes is fine on MPS. **Training** runs full forward/backward through
the frozen base (only ~3M prefix params learn), so train time grows with model size —
start at 1B, scale up once the pipeline is green.

### 3. Precision / quantization (inference)
Compare the *same trained adapter* injected into bases at different precisions:

| `--dtype` | Backend | Notes |
|---|---|---|
| `fp16` | torch MPS | default; reliable for inference |
| `bf16` | torch MPS | fine for inference (unlike training backward) |
| `int8` | torchao weight-only | keeps the HF forward path → injection still works |
| `int4` | torchao weight-only | **may be partial/unsupported on MPS** — record coverage |

**Why torchao and not MLX / llama.cpp?** MLX and llama.cpp give better Mac quantization,
but neither lets you inject arbitrary trained K/V (`past_key_values`). torchao quantizes
weights while leaving the standard HF forward path intact, so PEFT prefix injection still
works. MLX/llama.cpp are noted as future high-fidelity-quant work, out of scope for the
faithful inject path.

**Two caveats for the quant axis:**
- *Train/infer mismatch:* the prefix is trained against fp16/bf16 weights, then injected
  into a quantized base. Expect some quality drop — that's a **robustness measurement**,
  not a bug. (The prefix-encoder MLP itself stays unquantized and runs live via
  `PeftModel`, so the injected K/V remain consistent.)
- Quantized *training* (QLoRA) needs bitsandbytes/CUDA and is **not** supported on MPS —
  we always train in fp32/bf16 and quantize only for the eval.

### Run the full sweep

```bash
# Train adapters first (once per model — omit --skip-train to train), then sweep eval axes:
python agentcache_mac/run_mac_pipeline.py --model meta-llama/Llama-3.2-1B-Instruct --tokens 64
python agentcache_mac/run_mac_pipeline.py --model meta-llama/Llama-3.2-3B-Instruct --tokens 64

python agentcache_mac/sweep.py \
  --models meta-llama/Llama-3.2-1B-Instruct,meta-llama/Llama-3.2-3B-Instruct \
  --dtypes fp16,int8 \
  --prompts 200,500,1000,2000 \
  --tokens 64
```

---

## Metrics

`hf_eval.py` writes one JSONL record per task (schema identical to the vLLM
`test_compression.py`, so the shared `analyze_results.py` reads it). `analyze_results.py`
reports:

- **TTFT** (mean / median / stdev / min / max) per mode, and **speedup vs cold**.
- **Physical prompt tokens** — cold prefills `system+user`; inject prefills only `user`
  (the prefix is injected, recorded as `pad_tokens = N`, not prefilled).
- **Quality (% pass):** `coherent` (≥20 words, no degeneration), `task_check_pass`
  (response contains a `must_include_any` keyword), `ends_with_goodbye`.
- **Per-example TTFT delta:** cold vs inject, % of cases inject is faster, top speedups.

> **GOODBYE is a confounded probe, not a feature.** It's a hard-rule litmus test for
> behavioral encoding. `prepare_data.py --append-goodbye` (off by default) puts `GOODBYE`
> in *every* training label, so a high inject GOODBYE-rate partly reflects the label
> suffix. Interpret it as "did behavioral instruction survive compression," not accuracy.

ROUGE-L vs the cold baseline is available if `rouge-score` is installed (it's in
`requirements-mac.txt`).

---

## Files

| File | Purpose |
|---|---|
| `train_prefix_mac.py` | MPS port of `train_prefix_compression.py` (fp32 base, no Trainer AMP) |
| `hf_eval.py` | single-turn cold-vs-inject TTFT + quality via `PeftModel` (the core) |
| `run_mac_pipeline.py` | prepare_data → train → eval cold+inject → analyze (one model/N) |
| `sweep.py` | model × precision × prompt-length matrix |
| `analyze_results.py` | shared analyzer (copy; recognizes `cold`/`inject` modes) |
| `requirements-mac.txt` | MPS-friendly deps (torch, transformers, peft, torchao, …) |

Reused unchanged from `agentcache_compression/`: `prepare_data.py`, `prompts/*.txt`,
the eval data, and the quality-check logic.

---

## Known caveats (collected)
- **MPS bf16 backward** is historically flaky → training defaults to fp32 (affordable at 96GB).
- **int4 on MPS** may be unsupported / fall back → the analyzer records `dtype`; note actual coverage.
- **fp16-train / quant-infer mismatch** → expect a quality dip in `int8`/`int4`; it's a finding.
- **Short prompts (<~200–300 tokens)** → prefill too cheap for inject to win; use long prompts.
- **Per-model adapters** → never reuse a 1B adapter on 3B/8B; retrain (GQA / dims differ).
- **Multi-turn + APC** are out of scope here (no clean HF analog of vLLM's APC) — single-turn parity only.
