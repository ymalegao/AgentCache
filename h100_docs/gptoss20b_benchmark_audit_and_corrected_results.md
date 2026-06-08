# GPT-OSS-20B Multi-Turn Benchmark — Audit, Corrections & Measured Results (H100 session)

**Last updated:** 2026-06-08 · **Hardware:** rented/shared H100 80GB · **Base model:** `openai/gpt-oss-20b` (MoE; 24 layers, 8 KV heads, head_dim 64 → centroid shape `[24, N, 512]`).

This document records what we investigated, what we changed, the experiments we ran with
their numbers, the corrected conclusions, and the remaining work (Phase 2+). Read §1
(TL;DR) and §9 (Next steps) first. Companion doc: `qwen7b_clean_retrain_and_diagnostics.md`
(its "APC is lossless+free; the centroid is a cold-start/memory play, not a latency play"
conclusion is consistent with what we find here).

---

## 1. TL;DR

We set out to re-confirm the paper's **GPT-OSS-20B multi-turn TTFT** results (Table 2 /
§6.1.2: cold mean 534.3 ms, turn-1 1078 ms, **4.3× turn-1**, **1.80× mean**) after a known
Harmony-channel corruption bug. It turned into a full audit of that benchmark's
methodology. Findings:

- **The original benchmark's methodology was flawed in 4 ways** (no engine warmup → turn-1
  measured compile, not prefill; "cold" silently ran with prefix caching ON; single sample,
  no error bars; corrupted response text). The **absolute TTFT numbers are ~9× too high.**
- **BUT the headline speedup is real.** Measured correctly (warmed, true-cold, 5 repeats),
  the **turn-1 cold-start speedup is ≈5×** (slightly *above* the claimed 4.3×). The
  methodology errors inflated cold and synthetic *together*, so the ratio survived.
- The **1.80× mean is NOT yet validated**, and **quality is unverified** — both require a
  real trained GPT-OSS centroid, which **does not exist in the repo** (only Llama-1B and
  Qwen centroids are present). The paper's synthetic numbers came from a centroid that was
  never committed.
- **Corrected headline (H100):** cold turn-1 1078 ms → **120 ms**; synthetic turn-1 248 ms →
  **24 ms**; turn-1 speedup 4.3× → **~5×**. APC alone (no centroid) buys **1.35× mean / 1.0×
  turn-1** over true-cold.

**The single most important caveat:** these are **H100** numbers; the paper reported
**Blackwell** for GPT-OSS. The *ratio* should transfer (it's prefill-token-count driven);
the absolute ms do not.

---

## 2. Environment & reproduction (gotchas matter)

```bash
cd /root/AgentCache
source venv/bin/activate                       # Python 3.10, vLLM 0.20.0 (has the centroid patch)
./get_model.sh openai/gpt-oss-20b              # -> models/gpt-oss-20b/ (~39GB incl. original/ BF16 + metal/)
export VLLM_USE_DEEP_GEMM=0                     # REQUIRED: vLLM 0.20.0 ships an outdated deep_gemm that
                                               #   crashes kernel-warmup on gpt-oss MXFP4. Disabling it
                                               #   falls back to standard FP8 kernels.
```

Gotchas discovered this session:
- **`VLLM_USE_DEEP_GEMM=0`** — without it, `vllm serve` dies at startup in `deep_gemm_warmup`
  (`get_mk_alignment_for_contiguous_layout → _missing()`).
- **`--max-model-len 32768`** — the runbook's 16384 overflows at turn 9 (clean responses grow
  ~2k tokens/turn; cumulative prompt exceeds 16384). 32768 is safe.
- **`--gpu-mem 0.90`** is fine when the H100 is free; this host is **shared** — ask before
  allocating (see `~/.claude` memory `shared-gpu-ask-first`).
- GPT-OSS centroid shape is **`[24, N, 512]`** (24 layers; 8 KV-heads × 64 head_dim = 512).
  Training/materialization should use the **BF16** weights at `models/gpt-oss-20b/original/`
  (MXFP4 can't backprop); they are present (`original/model.safetensors`, 13.8 GB).

---

## 3. Background: the task and how it expanded

Starting task: re-run the two no-centroid modes (`cold`, `warm_apc`) for GPT-OSS-20B to get
clean baseline timings after the Harmony-channel bug (`strip_to_final_channel`) was fixed,
and check whether the corruption had inflated the cold baseline (which would have inflated
the 1.80× synthetic-vs-cold speedup).

It expanded because every "clean" rerun surfaced a deeper issue, ending in a full audit of
the benchmark methodology and a measured re-confirmation of the speedup via a dummy centroid.

---

## 4. What we found

### 4.1 The Harmony channel corruption (separate bug, already fixed)
The old run (`results/gptmulti_turn_benchmark.jsonl`) fed raw multi-channel Harmony text
(`analysis…assistantfinal…`, `!DOCTYPE html` garbage) back into history, degenerating later
turns. `strip_to_final_channel` (in `multi_turn_benchmark.py`) fixes this and works (verified:
clean turns once the model reaches the final channel).

### 4.2 Greedy-decoding degeneration (temp=0)
With `temperature=0.0` (the benchmark's generation setting), GPT-OSS loops in the Harmony
**analysis** channel and never reaches the final channel — turns 3–10 became repetition loops
(`"Also need to ensure we use correct summary."` ×N). This is **pre-existing** (the old run's
turn-3 starts the same way), independent of the strip fix. Fixed by sampling `temperature=0.7,
top_p=0.9, seed=0` for generation (TTFT-measurement calls left greedy/`max_tokens=1`).

### 4.3 "Cold" was not cold (default APC)
vLLM 0.20.0 (V1) enables prefix caching **by default** (`config/cache.py:91`; CLI default
`None` → default-on; only RISC-V disables it). The benchmark's cold mode merely *omitted*
`--enable-prefix-caching`, so cold still cached. Proof: in the committed file, cold and
warm_apc have **byte-identical** KV-hit curves (0.744/0.758/0.767/0.788) and ~1.0×
warm-vs-cold. The paper's "cold disables explicit warm prefix reuse" is factually wrong.

### 4.4 The warm-up / run-order artifact (the big one)
The benchmark starts a fresh `vllm serve` per mode and times turn-1 immediately — so turn-1
includes one-time **torch.compile / CUDA-graph capture (~960 ms)**, unrelated to the prompt.
Modes ran back-to-back into one file in order **cold → warm → synthetic**, so the first
(cold) ate full compile and the third (synthetic) found it cached.

**Smoking gun (from their own data):** cold turn-1 = **1077.5 ms** and warm turn-1 =
**511.2 ms** on the *identical 2299-token prompt with an empty cache* (kv-hit 0% for both at
turn 1). Prefix caching cannot act on an empty cache, so the 2.11× gap is purely warm-up
state, not the system under test.

### 4.5 Two different "cold-start overheads" (the key reconciliation)
At turn-1 the cold baseline pays **two** stacked costs:
1. **One-time compile / graph capture (~960 ms)** — engine startup artifact; the centroid does
   NOT save this.
2. **The real system-prompt prefill (~96 ms)** — actual attention over the 2000-token system
   prompt; **this is what the centroid legitimately skips**.

The paper's `cold turn-1 = 1078 ms = ~960 (compile) + ~120 (real prefill of 2299 tok)`.
Warm-up removes cost #1. Cost #2 is real and is the source of the speedup.

Turn-1 decomposition (warmed, H100): `TTFT ≈ 19 ms fixed + 0.044 ms/token`.
`cold 120 = 19 + ~101 (2299 tok)`; `synthetic 24 = 19 + ~5 (~120 user tok)`. Centroid saves
the ~96 ms system-prompt prefill → ~5×.

### 4.6 The GPT-OSS centroid is missing
No `[24, N, 512]` centroid exists anywhere in the repo (only Llama-1B `[16,N,512]`, Qwen-7B
`[28,N,512]`, Qwen-0.5B `[24,64,128]`), and there is no GPT-OSS PEFT adapter to
re-materialize one. The centroid that produced the paper's synthetic numbers was never
committed. Therefore **synthetic mode cannot be re-run with a real centroid without
training** — but **timing can be measured with a dummy centroid** (TTFT is independent of
centroid contents).

---

## 5. What we changed (code & config)

All edits are in `agentcache_compression/multi_turn_benchmark.py` (uncommitted):

| Change | Where | Why |
|---|---|---|
| `strip_to_final_channel` | already present | strip Harmony channels before storing/feeding history |
| generation sampling `temperature 0.0 → 0.7, top_p 0.9, seed 0` | `generate_turn_output` | stop the analysis-channel repetition loops; TTFT-measure calls left greedy/`max_tokens=1` (methodology unchanged) |
| cold mode appends `--no-enable-prefix-caching` | `build_server_cmd` | true cold (defeat vLLM's default APC) |
| engine **warmup** request before measured turns | `main` (before the conv loop) | turn-1 measures prefill, not compile/graph-capture |

Config/env: `VLLM_USE_DEEP_GEMM=0`, `--max-model-len 32768`, `--gpu-mem 0.90`.
Created dummy GPT-OSS centroids: `centroids/gptoss_dummy_N{64,128,256}_{K,V}.npy` (shape
`[24,N,512]` float16, small random — for **timing only**).

---

## 6. Experiments we ran (chronological)

1. **Clean cold/warm rerun (temp=0 greedy)** → `results/gptoss_clean.jsonl`.
   20 rows; turns 3–10 degenerate (analysis loops). Showed cold prompt sizes ≈ old (±3%) →
   format corruption did *not* inflate prefill. Provided "success check" passed but is a
   **false positive** (doesn't detect analysis-channel/repetition).

2. **Clean rerun with sampling fix (temp=0.7)** → `results/gptoss_clean2.jsonl`.
   All 10 turns clean, task-appropriate code (turn 8 = pytest, turn 10 = pyproject.toml).
   Cumulative prompt by turn 10 = **13955** vs degenerate **19238** (~27% smaller → clean
   responses are shorter). Confirmed strip fix works once the final channel is reached.

3. **Corrected true-cold + warm-APC baseline** (5 repeats each, warmed) → `results/tc/`.
   - true-cold turn-1 = **120 ms** (±~14), mean = **245 ms**, kv-hit **0%** all turns.
   - warm-APC turn-1 = **116 ms**, mean = **181 ms**, kv-hit 74–93%.
   - APC alone over true-cold: **1.35× mean, 1.0× turn-1**.
   - Same 2299-tok prefill: **1078 ms (paper) → 120 ms (warmed)** — pins ~960 ms as compile.

4. **Tier-1 measured speedup with dummy centroid** (5 repeats × N=64/128/256 + cold sanity)
   → `results/tier1/`.
   - synthetic turn-1: N=64 **22.8 ms**, N=128 **24.4 ms**, N=256 **24.2 ms** (±1–2 ms).
   - syn turn-1 phys tokens 184/248/376 vs cold 2299 → system prompt genuinely skipped.
   - **Turn-1 speedup ≈ 5× (5.3 / 4.9 / 5.0×), N-independent.**
   - Mean from dummy ≈ 1.9–2.1× but **unreliable** (garbage history → unrepresentative
     turns 2–10). Only turn-1 is a clean measurement.

---

## 7. Corrected results vs paper (GPT-OSS-20B)

| metric | Paper (Blackwell, flawed) | Corrected (H100, measured) |
|---|--:|--:|
| cold turn-1 TTFT | 1078 ms | **120 ms** (±14) |
| cold mean TTFT | 534.3 ms | **245 ms** (true-cold) / 181 ms (warm-APC) |
| synthetic turn-1 TTFT | 248 ms | **24 ms** (dummy centroid, timing-valid) |
| **turn-1 speedup** | **4.3×** | **~5.0×** (confirmed) |
| **mean speedup** | **1.80×** | **not yet validated** (needs real centroid) |
| output quality | ROUGE/sanity (failed to catch garbage) | **not yet validated** (needs real centroid + judge) |

---

## 8. Implications for the paper

- **Thesis / mechanism: sound.** Skipping the system-prompt prefill via centroid injection
  is real and gives ~5× at cold-start. Table 1 (Llama-1B prompt-length, 2.8×) and §6.3
  (LMCache, unaudited here) are the cleaner evidence and are not undermined.
- **Table 2 / §6.1.2 / §6.2 (GPT-OSS-20B):** the **absolute TTFT numbers are ~9× too high**
  and must be corrected; the **methodology section must be rewritten** (add warmup; make cold
  truly cold; repeats with error bars; clean generation). The sentence "cold disables explicit
  warm prefix reuse" is wrong.
- **The 4.3× / turn-1 claim survives** (measures ~5× corrected) — if anything understated.
- **The 1.80× mean and the quality claim are unverified** pending a real centroid.
- **Reframe the scaling story:** the benefit is bounded by *prefill cost vs injection
  overhead* → a function of **GPU + prompt length**, not model size. (Consistent with the
  Qwen-7B doc: APC is lossless+free for repeated prompts; the centroid earns its keep at
  cold-start / unique prompts / KV-memory-bound serving.)

---

## 9. Next steps (Phase 2 and beyond)

### Phase 2 — Harness hardening (½ day, small additions)
- Randomized/interleaved mode order; multiple conversations; multi-seed; per-record metadata
  (seed, GPU, vLLM version, run/order index); aggregation with **median ± 95% CI**; a
  **dummy-centroid control mode** to isolate injection overhead. Resolve/​document `deep_gemm`.

### Phase 3 — Clean data + GPT-OSS centroid training (the gating risk; "Tier 2")
1. Verify `models/gpt-oss-20b/original/` BF16 weights load for training (present).
2. Regenerate **clean** teacher data (`generate_good_examples.py`) on Python coding tasks
   **disjoint from eval tasks**, with the strip fix + temp>0, loss-masked, `m=0.0`.
3. Prefix-train N=64/128/256 (`train_prefix_compression.py`) on the frozen BF16 base.
   **Risk:** GPT-OSS-20B is a **MoE** — untried in this pipeline (used on dense Qwen/Llama);
   most likely to need debugging.
4. Materialize → `centroid_K/V.npy` `[24,N,512]` (`transpose_tensors.py` / `rescale_centroid.py`),
   correct RoPE (cross-check `AgentCache-metal/.../centroid_rope_parity.py`).
5. **Verify coherent injection** (real centroid → non-garbage output). 2nd-most-likely failure
   point (RoPE/KV-layout).

### Phase 4 — Final evaluation (timing + quality)
- ≥5 held-out conversations × 10 turns; modes: true-cold, warm-APC, synthetic N=64/128/256,
  dummy control; ≥5 seeds; randomized order; warmup.
- Timing: per-turn TTFT (median±CI), mean, speedup vs **both** true-cold and warm-APC, turn-1
  breakout, phys tokens, kv-hit. This **validates the 1.80× mean**.
- Quality: `judge_multi_turn.py` pairwise synthetic-vs-cold (correctness / completeness /
  code-quality / instruction-adherence → win/tie/loss). **This is the GPT-OSS code-quality eval.**

### Phase 5 — Reporting
- Rewrite Table 2 / §6.1.2 / §6.2 with corrected numbers + error bars + dual baselines +
  methodology section. Reframe the scaling claim. Consider an erratum documenting the fix.

### Optional — GPU matrix
Reproduce on **Blackwell** (match the paper's GPU) and **T4** (constrained regime where the
benefit is largest) so the claim is characterized honestly across hardware.

### Rough effort (single H100)
Tier 2 best case ~7–10 h GPU; realistic ~2–3 days elapsed (MoE training + injection-coherence
iteration are the wildcards). Phase-1-style timing-only is ~1 h.

---

## 10. Caveats (held firmly)
- **H100, not Blackwell.** Ratios should transfer; absolute ms do not.
- **Dummy centroid = timing only.** Turn-1 (5×) is valid; the mean and all quality claims
  need a real trained centroid.
- An earlier in-session estimate of ~1.5× turn-1 was **wrong** (assumed ~64 ms fixed
  overhead; the measured fixed cost is ~19 ms). The measured value is ~5×.

---

## 11. Artifact / file index
- Code (edited, uncommitted): `agentcache_compression/multi_turn_benchmark.py`
- Run scripts: `run_bench.sh`, `run_tc.sh`, `run_tier1.sh`, `run_tier1_smoke.sh`, `get_model.sh`
- Data:
  - `results/gptmulti_turn_benchmark.jsonl` — original (paper) data, corrupted
  - `results/gptoss_clean.jsonl` — clean rerun, degenerate (temp 0)
  - `results/gptoss_clean2.jsonl` — clean rerun, fixed sampling (temp 0.7)
  - `results/tc/{cold,warm_apc}_r0-4.jsonl` — corrected true-cold + warm-APC baseline
  - `results/tier1/syn{64,128,256}_r0-4.jsonl`, `coldchk_r0-1.jsonl`, `smoke_syn256.jsonl` — Tier-1
- Dummy centroids: `centroids/gptoss_dummy_N{64,128,256}_{K,V}.npy`
- Notes: `results/gptoss_clean_NOTE.md`, `results/gptoss_corrected_baseline_NOTE.md`,
  `results/tier1_RESULTS.md`, `results/REDO_PLAN.md`
- Pipeline (for Tier 2): `generate_good_examples.py`, `agentcache_compression/train_prefix_compression.py`,
  `transpose_tensors.py`, `rescale_centroid.py`, `judge_multi_turn.py`
