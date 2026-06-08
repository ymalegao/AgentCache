# Qwen-7B Centroid Compression — Clean-Data Retrain + Diagnostics (H100 session)

**Last updated:** 2026-06-08 · **Hardware:** rented H100 80GB · **Base model:** `Qwen/Qwen2.5-7B-Instruct` (28 layers, 4 KV heads, head_dim 128 → centroid shape `[28, N, 512]`).

This is the continuation point. It captures what we ran, what we changed, what we
found, the current best root-cause understanding, and exactly what to do next. Read
§1 (TL;DR) and §8 (Remaining work) first.

---

## 1. TL;DR

We re-ran the **entire** 7B centroid-compression pipeline with **clean self-distilled
training data** to test whether the known quality regression was caused by bad data
(fixable) or intrinsic compression loss (a floor). Then we ran two diagnostics.

**Verdict so far:** clean data did **not** recover aggregate parity (clean **6W/0T/24L =
80% loss** vs the bad-data baseline **8W/0T/22L = 73% loss**), but the result
**decomposed and inverted** in an informative way:

- **Per-dimension:** completeness recovered hard (net −22 → −9), code-quality (−19 → −12)
  and instruction-adherence (−17 → −10) improved — exactly the dimensions bad data hurt.
  **Correctness regressed** (−10 → −18).
- **Per-N inverted:** baseline's best was N=64 (4 wins); clean data's best is **N=256,
  which reached parity (5W/0T/5L) across all four dimensions** (even slight edge on
  code-quality), while **N=64 collapsed (0W/10L)**. Capacity became the lever.
- **Diagnostic C (KV-norm probe):** centroid keys are systematically under-scaled vs the
  real prompt's KV. **Rescale test refuted** the "fixable scale bug" idea — boosting keys
  causes total prefix-domination garbage. The small key norms are *load-bearing
  calibration*. The degeneration is a **dynamic multi-turn effect**, not a static bug.
- **Strategic reality (from TTFT data):** for a *repeated* system prompt, vLLM's
  automatic prefix caching (APC / `warm_apc`) is **lossless (bit-identical to cold) and
  free**, and nearly as fast on turns 2+. The centroid only earns its keep for
  **cold/unique prompts (turn-1 latency)** or **KV-memory-bound serving (~2× fewer cached
  tokens)**. It's a memory play, not a latency play.

**The single open question:** is the N=256 parity *real* (n=10 so far) or noise?
→ **Experiment 2 (§8.1) is queued and ready** — author 4 more coding conversations,
benchmark + judge N=256, aggregate to n=50 with a significance test. Not yet run
(waiting on user go-ahead; ~30 min GPU + ~$3–4 judge).

---

## 2. Environment & reproduction (gotchas matter)

```bash
cd /root/AgentCache
apt-get install -y python3.10-venv      # REQUIRED first — install.sh's `python3 -m venv` fails without it
./install.sh                            # venv/ + vllm 0.20.0 + deps + copies centroid patch into vLLM site-packages
source venv/bin/activate
./get_model.sh Qwen/Qwen2.5-7B-Instruct # -> models/Qwen2.5-7B-Instruct/ (~15G, no HF auth needed)
```

**Two env vars must be set for every run** (both vLLM and HF stages):
```bash
export HF_HOME=/root/.cache/huggingface     # scripts hardcode a /mnt/g/... default that doesn't exist here
export VLLM_USE_DEEP_GEMM=0                  # see §3.4 — vLLM 0.20.0 crashes without it on H100
```

**Judge key:** `OPENAI_API_KEY` for stages 7+ (gpt-5.5 Responses API). We stored it in
`/root/.openai_key` — **note the file content includes the `OPENAI_API_KEY=` prefix**, so
load it as:
```bash
export OPENAI_API_KEY=$(cat /root/.openai_key); export OPENAI_API_KEY=${OPENAI_API_KEY#OPENAI_API_KEY=}
```
(Security: that file holds a plaintext key — `rm` it when done with the rented box.)

---

## 3. What we changed (code) — all required for a correct run

The original `agentcache_compression/RETRAIN_HANDOFF.md` was mostly accurate but had four
real defects vs the actual repo. Fixes:

1. **`generate_good_examples.py` iterated the WRONG task list.** It defines `PYTHON_TASKS`
   (lines 40–250, ~175 real coding tasks) but the generation loop used `TASKS` (lines
   252–408, ~131 *search/research* prompts) — `PYTHON_TASKS` was dead code. **Added
   `TASKS = PYTHON_TASKS`** after the search list (so it overrides). Also changed the
   hardcoded output filename to `good_examples/vllm_good_examples_raw_2000_coding.jsonl`
   (the `_search` default already held 131 stale records the resume logic would keep).

2. **Field remap.** Raw generation emits `{index, task, good_example}`; training needs
   `{id, user, teacher_output}`. The handoff's inline gate/split snippets read
   `teacher_output` straight from raw (would yield empty labels). We wrote the gate to read
   `good_example` and the split to map `task→user`, `good_example→teacher_output`.

3. **New script `agentcache_compression/gate_training_data.py`** — ast.parse quality gate
   (largest fenced block must parse), reads `good_example`.

4. **New script `agentcache_compression/split_clean.py`** — deterministic train/eval split
   with the field remap + GOODBYE suffix.

5. **`VLLM_USE_DEEP_GEMM=0`** — vLLM 0.20.0's kernel warmup unconditionally probes DeepGEMM
   FP8 alignment; on H100 `is_deep_gemm_supported()` returns True (Hopper) but the
   `deep_gemm` package isn't installed → `RuntimeError` at engine init. Disabling it is a
   **numerical no-op for bf16** (FP8 kernels are never used). `os.environ.copy()` in
   `multi_turn_benchmark.py:103` propagates it to the `vllm serve` subprocess.

**Diagnostic scripts added this session:**
- `agentcache_compression/kv_probe.py` — Experiment C (KV-norm probe).
- `agentcache_compression/rescale_centroid.py` — builds key-rescaled centroid variants.

No changes were made to `train_prefix_compression.py`, `transpose_tensors.py`,
`run_multi_turn_pipeline.py`, `multi_turn_benchmark.py`, or `judge_multi_turn.py`.

---

## 4. The pipeline we ran (and how to re-run each stage)

All from `/root/AgentCache`, venv active, with the two env vars set. Hyperparameters were
held **identical to the baseline** (only training-data quality changed).

| Stage | Command (abridged) | Output | Gate result |
|---|---|---|---|
| 1 generate | `generate_good_examples.py --model models/Qwen2.5-7B-Instruct --system-prompt .../2000_python_agent_system.txt --temperature 0.1 --ensure-goodbye` | `good_examples/vllm_good_examples_raw_2000_coding.jsonl` (175) | coherent Python ✓ |
| 2 gate | `gate_training_data.py <raw> good_examples/vllm_good_examples_clean.jsonl` | 159 kept | **159/175 = 91%** (old 58%) |
| 3 split | `split_clean.py good_examples/vllm_good_examples_clean.jsonl 25` | `data/python_agent_{train,eval}.jsonl` | train 134 / eval 25, 100% ast-valid |
| 4 train | `train_prefix_compression.py ... --num-virtual-tokens {64,128,256} --system-retain-ratio 0.0 --epochs 8 --lr 2e-3 --batch-size 4` | `adapters/N{N}_qwen7b/` | losses **0.42 / 0.97 / 0.22** (N128 is an outlier) |
| 5 export | `transpose_tensors.py --adapter adapters/N{N}_qwen7b --out-k centroids_qwen7b/N{N}_2000_K.npy --out-v ...V.npy --sys-tokens 0` | `centroids_qwen7b/` | all `(28,N,512)` fp16, no NaN ✓ |
| 6 benchmark | `run_multi_turn_pipeline.py --centroid-dir centroids_qwen7b --conversation-file conversations/csv_cli.json --max-tokens 2048 --gpu-mem 0.9 --out results/qwen7b_clean.jsonl` | 50 rows | syntax: cold 10/10, N64 **10/10**, N128 7/10, N256 8/10 |
| 7 judge | `judge_multi_turn.py --qwen-file results/qwen7b_clean.jsonl --only qwen7b --judge-model gpt-5.5 --out results/judge_verdicts_clean.jsonl` | 30 verdicts | no API errors ✓ |
| 8 report | (manual) | `results/RETRAIN_RESULTS.md` | — |

Notes: the benchmark **appends** to `--out` (delete the file before re-running). The judge
keys on `(mode,N,turn)` with **no conversation_id**, so one results file = one conversation
(critical for §8.1). Training uses `system-retain-ratio 0.0` (prompt fully removed; the
prefix replaces it) so training sequences are short/fast.

---

## 5. Results — the retrain (n=10, single conversation `csv_cli.json`)

**Overall pairwise (synthetic vs cold):** clean **6W/0T/24L** vs baseline **8W/0T/22L**.
No aggregate improvement (slightly worse, within n=30 noise).

**Per-N (W/T/L) — inverted:**

| N | baseline | clean |
|---|---|---|
| 64 | 4/0/6 | **0/0/10** (collapsed) |
| 128 | 2/0/8 | 1/0/9 (under-converged, loss 0.97) |
| 256 | 2/0/8 | **5/0/5 (parity)** |

**Per-dimension net (win−loss), clean vs baseline:**

| dimension | baseline | clean | Δ |
|---|---|---|---|
| completeness | −22 | −9 | +13 ✅ |
| code_quality | −19 | −12 | +7 ✅ |
| instruction_adherence | −17 | −10 | +7 ✅ |
| correctness | −10 | −18 | −8 ❌ |

**N=256 alone, per-dimension (n=10):** correctness 5/0/5, completeness 5/1/4, code_quality
7/0/3, instruction_adherence 4/2/4 → parity across the board, slight edge on quality. It
hit this **despite 2/10 turns degenerating** into garbage tokens (turns 7, 9).

**Objective syntax validity ≠ judged quality:** N=64 has perfect ast.parse (10/10) yet
loses every pair (judge: "non-runnable — undefined names like `Any`/`summaries`, wrong
APIs"); N=256 has imperfect syntax (8/10) yet reaches parity. The failures are *semantic/
runtime* (undefined names, logic bugs, typos), which `ast.parse` cannot catch. The clean
*training data is import-complete* (117/134 have imports), so this is compression loss, not
a data defect.

**TTFT / cost reality (from `results/qwen7b_clean.jsonl`):**

| mode | phys tokens | turn-1 TTFT | turns 2–10 TTFT | quality |
|---|---|---|---|---|
| cold | 6954 | 0.757s | 0.086s | reference |
| warm_apc (APC) | 6954 | 0.607s | 0.081s | **identical to cold, 10/10** |
| synthetic_N256 | 3497 | 0.336s | 0.073s | regresses (parity at n=10) |

→ APC is lossless and nearly as fast on repeat turns. The centroid's only edge: **turn-1 /
cold-cache latency (~2.3×)** and **~2× less cached KV (memory)**. It's a memory play.

Full write-up: `agentcache_compression/results/RETRAIN_RESULTS.md`.

---

## 6. Diagnostic C — KV-norm probe (`results/kv_probe.md`)

**Q:** is the centroid KV in the same numerical regime as the prompt it replaces?
**Method:** ran the 7B on the 2183-token system prompt, grabbed `past_key_values`,
compared per-token L2 norms per layer to the centroid `.npy`. (L2 norm is RoPE-invariant,
so comparing pre-RoPE centroid to post-RoPE cache by norm is valid.)

**Finding:** centroid **keys are systematically under-scaled** — flat ~10–19 norm at every
layer vs the prompt's ~35–72 (bulk) with massive-activation spikes of **604 @ layer 0** and
**922 @ layer 27**. So ~0.35× in the bulk, **0.01–0.02× at layers 0/27**. **Values** are
roughly in-scale (median ~0.8–1.0×). The centroid does not reproduce the model's per-layer
norm structure.

**Initial hypothesis:** under-scaled keys ⇒ prefix under-attended ⇒ fades as multi-turn
context grows ⇒ degeneration. Looked like a fixable calibration issue.

---

## 7. Rescale test — hypothesis REFUTED (important negative result)

Rebuilt N=256 with keys rescaled toward the prompt norm (global ×2.9, and per-layer
median-match), V untouched (`rescale_centroid.py`), re-ran synthetic_N256 on csv_cli:

| variant | syntax-valid | degenerate |
|---|---|---|
| baseline N=256 | 8/10 | 2/10 |
| keys ×2.9 (global) | **0/10** | **10/10** |
| keys ×per-layer | **0/10** | **10/10** |

Both collapsed into token-salad (`filtro BY Zusstile ESingleOrDefault strugg…`).
**Mechanism:** scaling K scales every prefix attention logit → softmax mass collapses onto
the centroid → the model ignores the conversation. **The small key norms are load-bearing
calibration the training learned; naive rescaling is the wrong lever.**

**Corrected root cause:** the C norm-gap is real but **not** a fixable scale bug. The
degeneration is a **dynamic** effect — a fixed, deliberately weakly-attended prefix diluted
by growing multi-turn context — pointing at the **single-turn-train / multi-turn-serve
mismatch** and the **token-CE objective** as root causes. Right levers = multi-turn / KL
distillation and capacity, NOT KV calibration. (Don't repeat the rescale fix.)

Artifacts: `results/rescale_{global,perlayer}.jsonl`,
`centroids_qwen7b_kscale_{global,perlayer}/`.

---

## 8. Remaining work (prioritized)

### 8.1. Experiment 2 — confirm N=256 parity at n=50  ← NEXT, ready to run
The headline finding (N=256 parity) rests on n=10. Confirm or refute before any
training-side investment. Plan (also in `/root/.claude/plans/...declarative-quill.md`):

1. Author **4 diverse multi-turn coding conversations** (JSON array of 10 user-prompt
   strings each, matching `conversations/csv_cli.json`'s escalating difficulty) in
   `agentcache_compression/conversations/`: e.g. `config_loader.json`, `api_client.json`,
   `log_parser.json`, `lru_cache.json`. Use *different* domains to test generalization.
2. Benchmark each into its **own** results file (judge collides if 2 convos share a file):
   `run_multi_turn_pipeline.py --conversation-file conversations/$c.json --centroid-dir
   centroids_qwen7b --out results/confirm_$c.jsonl` (delete file first; appends).
3. Judge each, **N=256 only** (cost control): `judge_multi_turn.py --qwen-file
   results/confirm_$c.jsonl --only qwen7b --n-values 256 --out results/judge_confirm_$c.jsonl`.
4. Aggregate the 40 new N=256 verdicts + the 10 existing (from
   `results/judge_verdicts_clean.jsonl`, filter `N==256`) → n=50. Report overall + per-dim
   W/T/L, a **sign test / Wilson 95% CI** on win-share (CI overlaps 0.5 ⇒ parity), the
   **per-conversation** spread (is parity uniform or csv-specific?), and the degeneration
   rate. Write `results/N256_CONFIRM.md`.

Cost ≈ ~30 min GPU + ~$3–4 judge. **Decision gate:** parity holds ⇒ worth investing in
training-side fixes (§8.2); CI upper <0.5 ⇒ the 5/5 was a fluke, the method regresses even
at N=256 ⇒ don't ship, reconsider whether the centroid is worth pursuing vs APC.

### 8.2. The real fix lever — multi-turn / KL distillation (bigger investment)
Root cause is the train/serve mismatch + token-CE objective. Train the prefix to **match
the full-prompt model's next-token distribution (KL)** over a corpus that **includes
multi-turn histories** (not single-turn token-CE on 134 generic tasks). This addresses
objective + mismatch together. Highest expected payoff if §8.1 says parity is real.

### 8.3. Capacity sweep N ≥ 256 (384, 512)
Capacity is the proven lever (N64→256 climbs to parity). Find where parity saturates — but
note the **memory win shrinks as N grows** (the centroid's whole point), so there's a
capacity-vs-memory frontier to characterize, not just "bigger is better."

### 8.4. Re-train N=128 (it's an outlier)
Its train loss (0.97) is ~2–4× N64/N256 (0.42/0.22) → under-converged; its weak result
(1W/9L) is likely a training fluke, not the method. Re-train (more epochs / lr sweep)
before drawing the N-curve. Treat current N=128 numbers as noise.

### 8.5. Runtime-execution gate (data + serving metric)
The dominant judged failure is *runnability* (undefined names) which `ast.parse` misses.
Add a sandboxed import+exec check to the training-data gate and as a serving-quality
metric. (We deliberately skipped the optional runtime gate this run; the ast.parse-only
gate still hit 91%.)

### 8.6. Strategic decision (gates everything)
Decide the **target regime**: cold/unique prompts or memory-bound serving? If the prompt is
*repeated*, **APC already dominates the centroid** (lossless + free) — don't optimize the
centroid at all. Only pursue compression if cold-cache latency or KV memory is the real
constraint.

---

## 9. Artifact inventory

**New code (this session):** `generate_good_examples.py` (edited), `gate_training_data.py`,
`split_clean.py`, `kv_probe.py`, `rescale_centroid.py` (all in `agentcache_compression/`,
except `generate_good_examples.py` at repo root).

**Data:** `good_examples/vllm_good_examples_raw_2000_coding.jsonl` (175 raw),
`good_examples/vllm_good_examples_clean.jsonl` (159 gated),
`agentcache_compression/data/python_agent_{train,eval}.jsonl` (134/25).

**Models/centroids:** `models/Qwen2.5-7B-Instruct/`,
`agentcache_compression/adapters/N{64,128,256}_qwen7b/`,
`agentcache_compression/centroids_qwen7b/N{N}_2000_{K,V}.npy`,
`agentcache_compression/centroids_qwen7b_kscale_{global,perlayer}/` (rescale test).

**Results:** `agentcache_compression/results/` →
`qwen7b_clean.jsonl` (benchmark, 50), `judge_verdicts_clean.jsonl` (30),
`RETRAIN_RESULTS.md`, `kv_probe.md`, `rescale_{global,perlayer}.jsonl`.

**Baseline preserved (do NOT overwrite):** `results/qwen7b.jsonl` (bad-data benchmark),
`data/python_agent_train.baddata.bak.jsonl` (original 118-line training set).

**Plan file (live):** `/root/.claude/plans/checkout-the-agentcache-agentcache-compr-declarative-quill.md`
(holds the Exp 2 plan and interpretation guide).

---

## 10. One-paragraph summary for the next session

Clean self-distilled data did not fix the aggregate regression (6W/24L vs 8W/22L) but
recovered the data-sensitive dimensions (completeness/quality/adherence) and **inverted the
N-curve so N=256 reaches parity (5/5 at n=10)** while N=64 collapses — capacity is now the
lever. A KV-norm probe found the centroid keys are under-scaled, but a rescale test
**refuted** that as a fix (boosting keys → prefix-domination garbage), so the degeneration
is a dynamic multi-turn effect, not a static bug. Separately, TTFT analysis shows APC is
lossless and free for repeated prompts, so the centroid is only worthwhile for
cold/unique-prompt latency or memory-bound serving. **Next:** run Experiment 2 (§8.1) to
confirm whether the N=256 parity is real (n=50, ~$4); if yes, invest in multi-turn/KL
distillation (§8.2); if no, the method regresses and APC is the answer.
