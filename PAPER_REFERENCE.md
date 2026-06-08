# AgentCache — Paper Reference Document

**Purpose:** Everything you need to write the remaining sections of `usenix.tex`, in one place: what code exists, how it works (with `file:line` citations), every evaluation with verified numbers and raw-data provenance, where every figure lives, and what is still pending. Compiled 2026-06-06 from the repo, the raw result files, and the final presentation pptx.

**This document never modifies `usenix.tex` — it is a source you write *from*.**

---

## 1. How to use this document

| usenix.tex section | Status | Write it from |
|---|---|---|
| Abstract (L50) | placeholder | §9 Numbers-at-a-glance |
| §1 Introduction, §2 Problem, §3 Previous Solutions (L54–80) | **written** | (verified consistent with results, except one number — see §10.6) |
| §4 AgentCache (L82) | **empty** | §2 (operating modes) + §3.2 (what the centroid is) + §3.3–3.5 (training/export) |
| §5 System Design (L84) | **empty** | §3.6 (vLLM injection) + §3.7 (combined mechanism) + pptx notes in §7 |
| §6 Evaluation lead-in (L86) | **empty** | §4 intro + §2 (modes distinction — critical framing) |
| §6.1.1 Single-Agent Prompt-Length Sensitivity (L89) | **empty** | §4.1 |
| §6.1.2 Multi-Turn Conversation Benchmark (L90) | **empty** | §4.2 |
| §6.2 Ablations and Output Fidelity (L92) | **empty** | §4.3 + §8 (LLM judge — pending) |
| §6.3 LMCache+Centroid (L95–149) | **written** (teammate) | §4.4 documents which run its numbers come from |

---

## 2. Project summary and the two operating modes

**AgentCache** eliminates the cold-start prefill cost of an agent's system prompt by warm-initializing vLLM's KV cache with a *centroid*: a learned, offline-trained KV tensor artifact (per agent domain) that is written directly into the paged GPU KV cache at the start of every request, before any computation. The scheduler is told those positions are already computed, so they are never prefilled. Trained once per agent type via PEFT prefix tuning; injected at runtime with RoPE rotation; routed per-request in multi-agent settings.

**Load-bearing distinction — the system has been evaluated in two different configurations, and the quality story differs between them** (this is documented in `agentcache_compression/llm_judge_design.md` §1 and must be stated explicitly in the paper):

| | **Compression / replacement mode** | **Injection-in-front mode** |
|---|---|---|
| Used in | §6.1.2 multi-turn benchmark (Blackwell) | §6.3 LMCache+Centroid benchmark (T4) |
| Prompt sent to vLLM | `[pad]*N + conversation tokens` — **system prompt removed entirely** | Full system prompt intact; centroid covers positions 0–255 in front of it |
| Built by | `agentcache_compression/multi_turn_benchmark.py:63` (`build_compression_ids`) | `combined_benchmark.py` / `LMCacheCentroidN256.ipynb` |
| Output vs cold baseline | Legitimately *different text* by construction → ROUGE-L < 1.0 expected; quality claim awaits LLM judge (§8) | Byte-identical at temperature 0 → ROUGE-L = 1.0 is meaningful |
| What the speedup buys | Removes the entire system-prompt prefill (the headline 1.49×/1.80× numbers) | Removes re-prefill of first 256 tokens on every request incl. request 1 |

---

## 3. System architecture & code inventory (for §4 AgentCache + §5 System Design)

### 3.1 Pipeline overview (three offline phases + runtime)

```
[teacher data] → train_prefix_compression.py → adapter (P_θ + MLP)        Phase 1: TRAIN (offline, once per agent domain)
             → transpose_tensors.py → centroid_{K,V}.npy [layers, N, kv_dim]   Phase 2: EXPORT
             → patched vLLM: scheduler gap + CentroidInjector              Phase 3: INJECT (runtime, every request)
```

Documented in `README.md` (architecture, L44–72) and `HANDOFF.md` (design rationale). Orchestrator: `run_training_pipeline.py` (train → transpose → test; `--tokens N`, `--epochs 8`, `--lr 2e-3`).

### 3.2 What the centroid is (concept — best §4 source)

From pptx slide 7 + speaker notes (paper-ready narrative):

- The centroid is a learned K/V tensor encoding three things: **system-prompt semantics** (what the agent does and how it responds), **task-distribution patterns** (priors absorbed from the 118 training tasks), and **model-specific attention geometry** (the KV values are tuned against the frozen model's W_q, so they sit where *that model's* query vectors will attend).
- **Why not just average KV activations across tasks?** It was the first idea and it fails: transformer layers weigh positions differently (early layers attend heavily to special tokens like BOS; later layers don't), so the per-layer average produces KV values that are meaningless to the model — "semantic superposition" (HANDOFF.md). The fix is to *learn* the centroid by gradient descent so the optimizer places each layer's KV exactly where the model expects it. That is what prefix tuning provides.

### 3.3 Training data

| Component | File | What it does |
|---|---|---|
| Teacher example generation | `generate_good_examples.py` | Runs base tasks through vLLM (no centroid) to produce ground-truth `teacher_output` responses; ~60 base Python tasks defined in-file (~L40+) |
| Persona-perturbed tasks | `generate_tasks.py` (PERSONA_PROMPT ~L10), `personas/user{1-6}.yaml`, `persona_tasks.json` | Rewrites base tasks in 6 user-persona styles for data diversity |
| Filtering + split | `agentcache_compression/prepare_data.py` (`is_python_task` ~L51, `infer_checks` ~L58) | 175 raw → 143 Python-only → **118 train / 25 eval**; auto-generates `must_include_any` keyword checks for eval |
| Committed train set | `agentcache_compression/data/python_agent_train.jsonl` — **118 rows**, fields `{id, user, teacher_output}` (verified) | |
| Committed eval set | `agentcache_compression/data/search_agent_eval.jsonl` — **25 rows**, fields `{id, user, checks}` (verified). NOTE: the *python* eval jsonl is not committed; only the search-agent eval set is. | |
| System prompts (training/eval targets) | `agentcache_compression/prompts/{200,500,1000,2000}_{python,search}_agent_system.txt` — both agent domains at 4 lengths | |

### 3.4 Phase 1 — prefix-tuning training (`agentcache_compression/train_prefix_compression.py`)

The four decisions that define the method (all citable):

1. **Frozen base model** — every base parameter gets `requires_grad = False` (`:171`); only ~3M of ~1B params train (pptx slide-10 notes).
2. **Prefix tuning with MLP projection** — `PrefixTuningConfig(num_virtual_tokens=N, prefix_projection=True)` (`:173–176`). Direct per-layer prefix optimization is unstable (attention collapses onto virtual tokens); instead a compact matrix P_θ `[N × d_model]` is projected through one MLP that emits K and V for *all* layers in one shot — stable, and structurally couples layers (pptx slide-8 notes).
3. **System prompt removed from training input** (system retention ratio = 0.0; `truncate_system` `:55`). Input is `[user query]` only; target is the teacher output. Gradient descent is answering exactly one question: *what P_θ values make the frozen model produce correct agent outputs without ever seeing the system prompt?* (pptx slide-9 notes).
4. **Label masking** — `labels = [-100] * prompt_len + input_ids[prompt_len:]` (`:109`): loss only on assistant tokens, forcing the prefix to encode *behavioral priors* rather than reconstruct prompt text.

Hyperparameters: N ∈ {64, 128, 256}; 8 epochs; lr 2e-3; BF16; `TrainingArguments` at `:198`. Output: `adapter_model.safetensors` (+ config). **Unlike standard prefix tuning, the adapter is not kept for inference — "the model gets thrown away"; training exists purely to extract the KV tensors** (key differentiator to state in §4).

### 3.5 Phase 2 — export (`agentcache_compression/transpose_tensors.py`)

- For `prefix_projection=True` the only trustworthy source is PEFT's runtime `get_prompt(batch_size=1)` (`:42–66`) — manual PrefixEncoder reconstruction can drift (comment in-file).
- Per layer: `k.permute(1, 0, 2).contiguous().view(N, -1)` (`:80–81`) → stacked to **`[num_layers, N, kv_dim]`**, saved as `centroid_K.npy` / `centroid_V.npy` (`:92–93`) + `sys_prefix_num_tokens.txt` sidecar (`:94`).
- Rationale (pptx slide-11 notes): vLLM stores KV in paged blocks `[num_blocks, block_size, heads, head_dim]`; pre-flattening lets the injector write token-by-token at runtime with no extra computation.
- K is stored **unrotated** — RoPE is applied at injection time for the actual target positions (see below).

**Committed centroid artifacts** (`agentcache_compression/centroids/`): `N{64,128,256}_2000_{K,V}.npy` (Python agent, 2000-token prompt), `N64_200_{K,V}.npy`, `codingN256_{K,V}.npy`, `searchN{64,128,256}_{K,V}.npy`.

### 3.6 Phase 3 — runtime injection (the vLLM patch, 4 files; for §5 System Design)

How vLLM normally works (pptx slide-12 notes, good §5 prose): the scheduler tracks `num_computed_tokens` per request; on a fresh request it is 0, so a 2000-token system prompt + 200-token query is fully prefilled before decode. AgentCache's change: show vLLM `[pad]*N + user query`, tell the scheduler N tokens are already computed, and fill those N slots' KV memory directly.

| File | What it does | Key citations |
|---|---|---|
| `vllm/centroid_injector.py` | The KV-write mechanism | `CentroidInjector` class `:63` (loads `.npy`, fp16, optional secondary domain centroid); `_batch_rope` `:151` (applies the model's rotary embedding to centroid K at target positions `[sys_tokens .. sys_tokens+N-1]`); `seed_prefix_into_kv_cache` `:180` (maps logical positions → physical block-table slots and writes K/V for every layer; `_write_kv_rows` `:246` handles NHD/HND/2-first cache layouts); per-request dedup via `_centroid_seeded_req_ids`; RoPE-result caching |
| `vllm/centroid_integration.py` | Wiring + scheduler math | `_centroid_sched_check_once` `:161` (reads centroid shape + sidecar once); `centroid_scheduler_mode` `:225`; **`centroid_sched_gap` `:235`** — gap = `max(0, min(sys_tokens, prompt_len−1) − base_computed)`; the last prompt token is always recomputed; `try_load_centroid_injector` `:286`; `apply_centroid_block_table` `:439` (per-forward-pass guard checks → calls the injector); `centroid_preregister_prefix_blocks` `:618` (pre-registers pad-prefix blocks in APC's hash table) |
| `vllm/v1/core/sched/scheduler.py` | The 1-call scheduler hook | `:643–658` — "centroid Path B": `num_external_computed_tokens += centroid_sched_gap(...)`, so the scheduler treats the first N positions as already computed and never schedules their prefill |
| `vllm/v1/worker/gpu_model_runner.py` | Runner hooks | imports `:53–59`; injector loaded in init `:869`; lazy ensure `:1176`; `apply_centroid_block_table` called during forward prep `:2171` |

Two details worth a paragraph each in §5:

- **RoPE correctness:** positional encoding is baked into K during prefill. The centroid K is trained without position rotation, so the injector rotates it for the exact target positions at injection time (`_batch_rope`). Get this wrong and the model *silently* degrades — K vectors appear to be at wrong positions from attention's perspective (pptx slide-12 notes).
- **Transparency to the model:** after injection, the forward pass over the user tokens attends over slots 0..N−1 and finds populated KV. "The model has no idea they were injected rather than computed. It just sees a populated KV cache." (pptx slide-12 notes — quotable framing.)

**Per-request multi-agent routing** (used in §6.3's benchmark; figure = pptx slide 19a): both centroids are loaded at server startup (`VLLM_CENTROID_K_PATH` + `_K_PATH_2`); requests whose request-id begins with `VLLM_CENTROID_DOMAIN_2_PREFIX` (default `"search:"`) get the secondary centroid (`centroid_injector.py` ~`:80–105`). No restart or reload to switch agents.

### 3.7 Combined LMCache+Centroid mechanism (already written §6.3 — for reference)

Position layout (figure = pptx slide 19b): **centroid covers tokens 0–255; LMCache (CPU offload, chunk 16, LRU) covers tokens 256–~2000 (system-prompt remainder) on warm requests; user tokens prefill normally.** After round 1, the full system prompt is never re-prefilled by either mechanism. Implementations: `LMCacheCentroidN256.ipynb` (the T4 run — 29 cells: patch vLLM → 4 conditions → plots → ROUGE-L; outputs cleared) and standalone `combined_benchmark.py` (`run_condition` `:549`; agents/system prompts `:72–285`; queries `:290–313`; note its default `CENTROID_LEN = 64` `:69` differs from the notebook's N=256 run).

### 3.8 Runtime environment-variable reference

Core: `VLLM_CENTROID_SCHEDULER=1`, `VLLM_CENTROID_K_PATH`/`_V_PATH`, `VLLM_CENTROID_SYS_TOKENS` (0 = pure compression mode). Multi-agent: `_K_PATH_2`/`_V_PATH_2`, `_DOMAIN_2_PREFIX`. Tuning/interop: `_LEN`/`_LEN_2` (cap injected length), `_USE_LMCACHE`, `_LAYOUT` (replacement|compression), `_SINK_BLEND` (default 0.35 — blends first centroid token with the model's attention-sink template), `_DISABLE_ROPE`; exact-KV option `VLLM_EXACT_SYS_K_PATH`/`_V_PATH`. All read in `vllm/centroid_injector.py:63–150` and `vllm/centroid_integration.py:161–232`.

---

## 4. Evaluation inventory (for §6)

> All numbers below were **recomputed from the raw committed files on 2026-06-06** unless marked otherwise. Where the paper's number differs from raw data, both are shown.

### 4.1 Single-agent prompt-length sensitivity (→ §6.1.1)

- **Setup:** Llama-3.2-1B-Instruct; harness `agentcache_compression/test_compression.py` (modes `cold_no_synthetic` / `warm_apc` / `synthetic_compression`; TTFT = mean of 3 runs, `measure_ttft_s` `:82`); plots via `plot_ttft_vs_tokens.py`.
- **Source of numbers:** `README.md` L78–98 tables (raw jsonl for this experiment is not committed). Figures: `agentcache_compression/results/ttft_vs_tokens.png`, `ttft_vs_tokens_clean.png`.

| Context | Mode | Physical tokens | TTFT | Speedup |
|---|---|---|---|---|
| ~200 tok | cold | ~276 | 20.8 ms | — |
| ~200 tok | centroid (N=128) | ~184 | 18.7 ms | 1.1× |
| ~1000 tok | cold | ~1092 | 47.8 ms | — |
| ~1000 tok | centroid (N=128) | ~184 | 17.0 ms | **2.8×** |

Story for the paper: synthetic TTFT is **flat** in original context length (physical tokens are always N + query), so speedup grows with prompt length; at short contexts fixed GPU overhead dominates.

N-ablation at 200-token context (README L93–98): N=64 → ~17.8 ms, **80% task-pass**; N=128 → ~18.7 ms, **84% task-pass**. Quality similar; speedup insensitive to N above a floor.

### 4.2 Multi-turn conversation benchmark (→ §6.1.2) — the headline standalone result

- **Setup:** 1 conversation × **10 turns** of a CSV-CLI Python coding task; ~2000-token system prompt (`agentcache_compression/prompts/2000_python_agent_system.txt`); NVIDIA Blackwell (DGX Spark); vLLM serve, TTFT = time-to-first streaming chunk with `max_tokens=1, temperature=0` (`multi_turn_benchmark.py:192`), full responses generated by a separate temperature-0 call and stored.
- **Modes (5):** `cold` (full system+history, APC disabled), `warm_apc` (same, APC enabled), `synthetic` N ∈ {64,128,256} (**compression mode** — system prompt removed, `build_compression_ids` `:63`). Pipeline: `run_multi_turn_pipeline.py`. Per-record fields include `response`, `kv_cache_hit_rate`, `physical_prompt_tokens` — enough to run the LLM judge later **without reruns**.
- **Raw data (committed, verified):** `agentcache_compression/results/qwen7b.jsonl` and `agentcache_compression/results/gptmulti_turn_benchmark.jsonl` (50 records each = 5 modes × 10 turns).

**Verified summary (recomputed from the jsonl files):**

| Model | cold mean | cold turn-1 | synth mean (N64/128/256) | mean speedup | turn-1 speedup | warm_apc |
|---|---|---|---|---|---|---|
| Qwen-7B | 231.4 ms | 951.3 ms | 154.4 / 155.1 / 156.7 ms | **1.49–1.50×** | **3.0×** (951→317 ms, N64) | 1.14× |
| GPT-OSS-20B | 534.3 ms | 1077.5 ms | 327.8 / 285.8 / 278.6 ms | 1.63 / 1.87 / 1.92× (**avg ≈ 1.80×**) | **4.3×** (1078→248 ms) | 0.95× |
| Qwen-1B | — | — | — | **≈0.9×** (synthetic *slower* on average) | ~2× turn-1 region (figure only) | ~0.96× |

⚠ Provenance: the paper's headline **"1.49× / 1.80×"** are per-model aggregates over the three N values; per-N values above are exact. **Qwen-1B raw jsonl was never committed** — its numbers exist only in `agentcache_compression/results/ttft_comparison.png` and `agentcache_compression/results/ab_study.png` (panel D labels: 0.90× synth, 0.96× warm). The committed `agentcache_compression/results/csv_cli_2048benchmark.jsonl` is a *different* run (7–8× speedups, hit-rate 0.00) and does **not** back the paper's Qwen-1B row — do not cite it (see §10.3).

**Per-turn TTFT (ms), recomputed — for the §6.1.2 narrative:**

Qwen-7B: turn 1 is the entire story (951 cold vs ~317 synth); turns 2–10 all modes converge to ~132–168 ms because vLLM v1 enables APC by default, so even "cold" gets in-session prefix reuse from turn 2 (state this explicitly — it is why mean speedup is much smaller than turn-1 speedup).

| turn | cold | warm_apc | N64 | N128 | N256 |
|---|---|---|---|---|---|
| 1 | 951 | 653 | 318 | 306 | 327 |
| 2–10 (range) | 139–168 | 148–163 | 124–140 | 132–141 | 132–149 |

GPT-OSS-20B: cold TTFT *grows* with history (234 → 666 ms over turns 2–10) and synthetic stays mostly lower but noisier; warm_apc is *worse* than cold on later turns (mean 0.95×):

| turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| cold | 1078 | 234 | 268 | 390 | 513 | 556 | 518 | 516 | 606 | 666 |
| N128 | 248 | 357 | 133 | 432 | 179 | 220 | 275 | 255 | 192 | 567 |
| N256 | 249 | 120 | 392 | 192 | 447 | 333 | 282 | 166 | 268 | 337 |

(Single conversation per cell → noisy; present per-turn as a figure, not over-interpreted. Figures, all under `agentcache_compression/results/`: `qwen7b_ttft_by_turn.png`, `gptmulti_turn_benchmark_ttft_by_turn.png`, 3-panel `ttft_comparison.png`.)

**Cache-hit instrumentation** (methodology detail worth one sentence): Prometheus `prefix_cache_hits/queries` scraped per turn from vLLM `/metrics` (`multi_turn_benchmark.py:149–163`); recorded per record; mean hit rates ~0.87–0.89 (Qwen-7B) confirm APC was active in all modes.

### 4.3 Ablations & output fidelity (→ §6.2 Ablations and Output Fidelity)

**The AB-study figure** `agentcache_compression/results/ab_study.png` (= pptx slide 17; generated by `plot_ab_study.py`) is the single best ablation artifact — 6 panels:
- Top row: TTFT-by-turn per model (cold / warm-APC / synth all-N).
- **Panel D** "Speedup vs Cold (model size drives gains)": 0.90× (1B) / 1.49× (7B) / 1.80× (20B) synth; 0.96× / 1.14× / 0.95× warm-APC — *the paper's headline numbers, in figure form*.
- **Panel E** "Centroid size (N) ablation (N barely matters)": N=64≈N=256 at 1B & 7B; slight gain from larger N at 20B.
- **Panel F** "Cold-start vs sustained speedup (where does the benefit come from?)": synth turn-1 3.0× (7B) / 4.3× (20B) vs much smaller sustained turns 2–5 — the benefit is concentrated exactly where the cold-start problem lives.

**Fidelity metrics implemented** (all in-repo, citable):

| Metric | Where | What it checks |
|---|---|---|
| ROUGE-L vs cold baseline | `agentcache_compression/analyze_multi_turn.py:320–381` (also notebook cell 26 for §6.3) | textual overlap, keyed by (conversation, turn) |
| Coherence heuristic | `agentcache_compression/test_compression.py:105` (`check_coherent`) | ≥20 words, no >40% token repetition in last 30 words |
| Task keyword check | `test_compression.py:118` (`check_task`) + auto-generated `must_include_any` from `prepare_data.py:58` | response mentions required concepts |
| Behavioral compliance (GOODBYE) | `analyze_multi_turn.py:299–319` | does response end with the trained `GOODBYE` token |
| Response-shape stats | `analyze_multi_turn.py:299–318` | has-code %, truncation %, avg words |

⚠ **GOODBYE compliance is 0% in ALL modes including cold** (model-capacity issue at 1B) — internal metric, deliberately excluded from the paper. Don't cite it as evidence either way.

⚠ **Why ROUGE-L is insufficient for §6.1** (the gap the LLM judge fills — argued in `llm_judge_design.md` §1): in compression mode outputs *legitimately differ* from baseline, and ROUGE-L cannot distinguish "different wording, equally good code" from "degraded code". Keep §6.1 quality claims hedged until §8 runs.

### 4.4 LMCache+Centroid combined benchmark (→ §6.3, already written) — data provenance

- **Setup (as written in the paper, verified against code):** Qwen2.5-1.5B-Instruct on NVIDIA T4 (Colab); 2 agents (coding + search, ~2000-token system prompts), 10+10 queries interleaved, 2 rounds (round 1 = cold session, round 2 = warm); 4 conditions: cold / lmcache / centroid (N=256 per-request routed) / combined; vLLM native APC on in *all* conditions; temperature 0; ROUGE-L vs cold outputs. Notebook: `LMCacheCentroidN256.ipynb`.

**⚠ TWO RUNS of this benchmark exist. The paper text uses Run A; the repo's committed artifacts are Run B.**

| | Run A — **paper-canonical** (usenix.tex §6.3 + README text) | Run B — repo-committed (`LMCacheCentroid/results/`) |
|---|---|---|
| cold | **1078 ms** | 673 ms |
| lmcache | 568 ms (1.90×) | 305 ms (2.21×) |
| centroid | 547 ms (1.97×) | 326 ms (2.06×) |
| combined | **494 ms (2.18×)** | 306 ms (2.20×) |
| warm round | ~74–75 ms (all conditions) | ~43–45 ms (all conditions) |
| turn-1 / turn-2 cold | 2585→2097 / 4362→1214 (3.6×) | 1728→1518 / 4708→1234 (3.8×) |
| ROUGE-L | 1.0 all non-cold conditions | (JSONs contain outputs; same expected) |
| Raw data | **NOT in repo** (Overleaf-side download copies) | `results_{cold (8),lmcache (4),centroid (4),combined (4)}.json` (verified) |
| Figures | `combined_ttft_summary (1).png` etc. — **not in repo** | `combined_ttft_summary.png`, `combined_ttft_cold_perturn.png`, `combined_ttft_warm_perturn.png` (committed; also = pptx slide 20) |

Both runs tell the same qualitative story (centroid > lmcache cold; combined best ≈2.2×; everything converges warm; turn-2 is the biggest win). **Decide which run is canonical and make text + figures consistent** — right now §6.3's prose cites Run A while the repo and the presentation show Run B (see §10.1–10.2).

### 4.5 `results_combined/` — broken rerun, do not use

Qwen2.5-7B rerun (teammate's machine): cold/lmcache/centroid ≈126–134 ms but **combined ≈1269 ms cold / 1305 ms warm** — the combined condition is broken (verified from `results_combined/results_*.json`). Excluded from the paper; don't cite anything from this directory, including `results_combined/combined_ttft.png`.

---

## 5. Figure inventory

**Figures referenced by usenix.tex (L99–141) vs reality:**

| tex `\includegraphics` | In repo? | Closest available source |
|---|---|---|
| `Per-Request Centroid Routing.png` | ❌ | pptx slide 19, media `image16.png` (verified visually: routing diagram, request_id prefix → codingN256/searchN256 → vLLM + LMCache) |
| `Combined Implementation.png` | ❌ | pptx slide 19, media `image17.png` (verified: centroid 0–255 / LMCache 256–2000 / user tokens layout) |
| `combined_ttft_summary (1).png` | ❌ | `LMCacheCentroid/results/combined_ttft_summary.png` exists **but shows Run B (673…)**, not the Run A numbers in the tex prose |
| `combined_ttft_cold_perturn (1).png` | ❌ | `LMCacheCentroid/results/combined_ttft_cold_perturn.png` (Run B) |
| `Rouge-L_score.png` | ❌ | nowhere in repo; regenerate from notebook cell 26 / Run-B JSON outputs |

To extract a pptx image: `unzip -p "AgentCache Final Presentation … .pptx" ppt/media/image16.png > routing.png`.

**Repo figures available for the unwritten sections:**

| Figure | Shows | Use in |
|---|---|---|
| `agentcache_compression/results/ttft_comparison.png` (= pptx slide 16) | 3-panel TTFT vs conversation length (physical prompt tokens), Qwen-1B / Qwen-7B / GPT-20B, cold vs synth — **the only artifact containing Qwen-1B numbers** | §6.1.2 |
| `agentcache_compression/results/ab_study.png` (= pptx slide 17) | 6-panel AB study (see §4.3) | §6.1.2 + §6.2 |
| `agentcache_compression/results/qwen7b_ttft_by_turn.png`, `gptmulti_turn_benchmark_ttft_by_turn.png` | per-model per-turn TTFT | §6.1.2 |
| `agentcache_compression/results/ttft_vs_tokens.png`, `ttft_vs_tokens_clean.png` | single-agent TTFT vs context length | §6.1.1 |
| `LMCacheCentroid/results/combined_ttft_{summary,cold_perturn,warm_perturn}.png` | Run B combined benchmark | §6.3 (if Run B becomes canonical) |

---

## 6. Presentation (pptx) slide map

24 slides; speaker notes are detailed enough to lift into paper prose for §4/§5.

| Slides | Content | Feeds |
|---|---|---|
| 1–4 | Title; KV-cache background; cold-start problem; goal question ("warm-initialize KV state at cold-start before any computation?") | Intro (already written; consistent) |
| 5–6 | Full pipeline diagrams (media image15/22/1/26 + image1/4); notes narrate train→transpose→inject | §5 overview figure candidates |
| 7 | What the centroid is; why averaging fails → gradient descent | §4 (see §3.2) |
| 8 | Prefix tuning mechanics: frozen model, K′ = [prefix K; K], P_θ + MLP projection for stability, virtual tokens in continuous space | §4 |
| 9 | Training setup: teacher/student data, NO system prompt in input, label masking, "toss the model in the trash" (only P_θ+MLP kept) | §4 |
| 10 | Training-code walkthrough (freeze :171, config :173–177, ~3M/1B trainable params) | §4 |
| 11 | Transpose rationale: `[num_layers, N, token_dim]` → flattened for paged-block writes | §4/§5 |
| 12 | Injection mechanics: scheduler `num_computed_tokens`, block-table allocation, CentroidInjector writes, **RoPE-or-silent-degradation caveat**, "model has no idea they were injected" | §5 |
| 13 | Evaluation overview matrix: both benchmarks' conditions, models, GPUs (Blackwell DGX Spark vs T4), metrics (TTFT, ROUGE-L) | §6 lead-in |
| 14–15 | Agent representation (coding vs search system prompts); multi-turn implementation (fixed system prompt, growing user context) | §6.1.2 |
| 16–17 | Multi-turn results figures (= `ttft_comparison.png`, `ab_study.png`) | §6.1.2/§6.2 |
| 18–19 | Combined benchmark implementation + the two diagrams the tex wants (image16/17) | §6.3 |
| 20 | Combined results figure (= Run B summary) | §6.3 |
| 21 | Model quality: ROUGE-L 1.0, temperature-0 reasoning | §6.3 / §6.2 |
| 22–24 | Demos, thanks | — |

---

## 7. Quotable design narratives (from speaker notes — paraphrase into the paper)

- **Centroid definition (slide 7):** "a learned Key-Value tensor that represents the average behavioral state of an agent… system prompt semantics, task patterns, and model-specific attention geometry."
- **Averaging failure (slide 7):** early layers attend to special tokens, later layers don't → cross-task averages are "fake KV values that didn't mean anything to the model"; the fix is learning the centroid by gradient descent.
- **Stability via projection (slide 8):** direct prefix optimization lets "attention scores collapse onto the virtual tokens"; P_θ + a single MLP emitting all layers' K/V at once stabilizes optimization and couples layers.
- **Training objective (slide 9):** "the only question gradient descent is answering is: what values of P_θ cause the frozen model to produce correct agent outputs without ever seeing the system prompt?"
- **Injection transparency (slide 12):** "The model has no idea they were injected rather than computed. It just sees a populated KV cache. The result: no request ever pays the system prompt prefill cost. Not the first one, not any of them."
- **RoPE caveat (slide 12):** "Get this wrong and the model silently produces degraded output because the K vectors appear to be at the wrong positions from attention's perspective."

---

## 8. PENDING — LLM-as-a-judge (multi-turn code quality)

**Status: designed, not implemented, no results.** Full design: `agentcache_compression/llm_judge_design.md` (judge rubric prompt included there verbatim, intended for the paper appendix). Planned implementation: `agentcache_compression/judge_multi_turn.py` → `agentcache_compression/results/judge_verdicts.jsonl`.

What it is (one paragraph for the paper's methodology, adaptable from the design doc): pairwise cold-vs-synthetic comparison per turn, per N, per model — 10 turns × 3 N × 2 models = **60 pairs** (57 after dropping corrupted GPT-20B cold turn-1, see §10.4); randomized A/B order with recorded seed; pinned judge model at temperature 0; 4-dimension rubric (correctness, completeness, code_quality, instruction_adherence) + overall win/tie/loss; per-model reporting to expose self-preference bias; structural defense — both sides of every pair come from the same model under test.

**Until it runs, the paper must NOT claim** that compression-mode quality is preserved in §6.1. Safe to say now: outputs are coherent and on-task per heuristic checks (§4.3); ROUGE-L 1.0 holds **only** for §6.3's injection-in-front mode. The hedge to use: quality evaluation of compression mode via LLM-as-a-judge is [in progress / presented in §X] — and §7 of the design doc pre-maps every possible outcome (mostly-ties ⇒ "speedup is free"; correctness losses ⇒ trade-off framing; adherence losses ⇒ training-data story; N-dependent losses ⇒ strengthens N-ablation).

Data prerequisite already satisfied: both multi-turn jsonl files store complete temperature-0 responses — **no benchmark reruns needed** (exception: GPT-20B cold turn-1, §10.4).

---

## 9. Numbers-at-a-glance (citable claims + their source artifact)

| Claim | Number | Source |
|---|---|---|
| Single-agent TTFT speedup @1000-token prompt (Llama-1B, N=128) | **2.8×** (47.8→17.0 ms) | README L84–87; `agentcache_compression/results/ttft_vs_tokens*.png` |
| Synthetic TTFT flat in prompt length | ~17–19 ms regardless of context | same |
| Task-pass: N=64 / N=128 | 80% / 84% | README L95–98 |
| Multi-turn mean speedup Qwen-7B | **1.49×** (231.4→~155 ms) | `agentcache_compression/results/qwen7b.jsonl` (recomputed ✓) |
| Multi-turn mean speedup GPT-OSS-20B | **1.80×** avg (per-N 1.63/1.87/1.92×; 534.3→~297 ms) | `agentcache_compression/results/gptmulti_turn_benchmark.jsonl` (recomputed ✓) |
| Multi-turn mean Qwen-1B | **≈0.9×** (honest negative result) | `agentcache_compression/results/ttft_comparison.png` + `ab_study.png` panel D **only** (raw not committed) |
| Turn-1 cold-start speedup | **3.0×** Qwen-7B (951→317 ms); **4.3×** GPT-20B (1078→248 ms) | same jsonl files (recomputed ✓) |
| warm-APC alone | 1.14× (7B), 0.95× (20B), ~0.96× (1B) — APC does not solve cold start | same |
| N-ablation | N barely matters for TTFT; slight quality/N effect at 20B | `ab_study.png` panel E |
| Combined benchmark (T4, paper-canonical Run A) | cold 1078 / lmcache 568 (1.90×) / centroid 547 (**1.97×**) / combined 494 ms (**2.18×**); warm ~75 ms | usenix.tex §6.3 + README text (⚠ raw data not in repo; see §4.4) |
| Combined benchmark turn-2 effect | 4362→1214 ms (**3.6×**) Run A; 4708→1234 (3.8×) Run B | README / `LMCacheCentroid/results/*.json` |
| Output fidelity, injection-in-front mode | **ROUGE-L = 1.0** at temperature 0, all non-cold conditions | notebook cell 26; Run-B JSONs contain outputs |
| Compression-mode quality | **PENDING — LLM judge not yet run** | `llm_judge_design.md` |

---

## 10. Known issues / pre-submission checklist

1. **Twin-run mismatch in §6.3:** prose cites Run A (1078/568/547/494) but every committed artifact + the presentation show Run B (673/305/326/306). Either commit Run A's raw JSON + "(1)" figures to the repo, or restate §6.3 from Run B. README has the same internal mismatch (text = Run A, embedded charts = Run B).
2. **All five tex figure files are missing from the repo** — they exist only Overleaf-side. The two diagrams can be re-extracted from pptx slide 19 (`image16.png`/`image17.png`); `Rouge-L_score.png` must be regenerated (notebook cell 26).
3. **Qwen-1B provenance:** paper's 0.9× row has no committed raw data (figure-only). The committed `csv_cli_2048benchmark.jsonl` is a different run (7–8× speedups, hit-rate 0.00) — don't cite without re-establishing provenance; ideally rerun Qwen-1B and commit the jsonl.
4. **GPT-OSS-20B cold turn-1 stored response is corrupted** (`!DOCTYPE html` + ellipsis garbage). TTFT for that record is still valid (timing call is separate, `max_tokens=1`), but the response must be dropped or re-generated before LLM judging (`llm_judge_design.md` §3.3).
5. **`results_combined/` is broken** (combined condition ~1269/1305 ms) — excluded; don't cite.
6. **Intro inconsistency:** usenix.tex L63 says centroid-alone gave "2.05×" on T4, but §6.3/README say **1.97×** (1078/547). Fix the intro number (likely from a stale run).
7. **GOODBYE compliance** is 0% in all modes (1B capacity issue) — internal metric; keep out of the paper.
8. **Quality-claim hedging:** until the LLM judge runs, §6.1 may claim latency wins + heuristic coherence only; ROUGE-L 1.0 belongs exclusively to §6.3's injection-in-front mode (§2, §8).
