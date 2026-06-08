# Clean-Data Retrain — Results (Qwen2.5-7B Centroid Compression)

**Date:** 2026-06-08 · **Judge:** gpt-5.5 (Responses API, reasoning.effort=medium, seed 42, single-pass A/B-randomized) · **Base:** Qwen/Qwen2.5-7B-Instruct (bf16, H100)

## TL;DR

Re-running the full pipeline with **clean, self-distilled training data did *not*
recover parity** at the aggregate level — pairwise synthetic-vs-cold went **6W / 0T /
24L (80% loss)**, marginally *worse* than the bad-data baseline's **8W / 0T / 22L
(73% loss)** and well within n=30 noise. **But the headline number hides the real
finding:** the regression *redistributed*. Clean data **substantially fixed
completeness, code-quality and instruction-adherence** (the dimensions the data
hypothesis predicted), while **correctness regressed**, and the **per-N ranking
inverted** — with clean data the *largest* centroid **N=256 reaches parity (5W/5L)**
while **N=64 collapses (0W/10L)**, reversing the baseline's "more N doesn't help."

**Interpretation:** This is handoff interpretation **#2 (the regression
decomposes)**, not #1 (parity) and not a flat #3. Bad data *was* hurting
completeness/quality/adherence — now improved. The **residual is a method/capacity
floor** dominated by **runnability/correctness at low N**. There is a viable path
(high-N), but the method, not the data, is now the bottleneck. **Escalate on the
method.**

---

## 1. Headline: synthetic vs cold (pairwise)

| | Win | Tie | Loss | Loss-rate |
|---|---|---|---|---|
| **Baseline (bad data)** | 8 | 0 | 22 | 73% |
| **Clean data (this run)** | **6** | **0** | **24** | **80%** |

No aggregate improvement (slightly worse, within noise). The main hypothesis —
"bad data was *the* primary cause; fixing it → ~50/50" — is **not confirmed**.

## 2. Per-N — the ranking inverted

| N | Baseline (W/T/L) | Clean (W/T/L) | Shift |
|---|---|---|---|
| 64  | 4 / 0 / 6  | **0 / 0 / 10** | collapsed |
| 128 | 2 / 0 / 8  | 1 / 0 / 9 | ~flat (see caveat) |
| 256 | 2 / 0 / 8  | **5 / 0 / 5 (parity)** | recovered |

Baseline's best was the *smallest* centroid (N=64); clean data's best is the
*largest* (N=256). With coherent training targets, **centroid capacity (N) becomes
the dominant lever** — N=256 matches the full-prompt baseline. This directly
contradicts the baseline's N-ablation conclusion.

## 3. Per-dimension — where clean data actually helped

Net = win − loss over 30 pairs (negative = synthetic worse than cold).

| Dimension | Baseline net | Clean net (W/T/L) | Change |
|---|---|---|---|
| completeness | −22 (worst) | **−9**  (10/1/19) | **+13 ✅ big recovery** |
| code_quality | −19 | **−12** (9/0/21) | +7 ✅ |
| instruction_adherence | −17 | **−10** (9/2/19) | +7 ✅ |
| correctness | −10 (least hit) | **−18** (6/0/24) | **−8 ❌ regressed** |

The data fix landed exactly where the hypothesis predicted (**completeness**, the
bad-data run's worst dimension, recovered the most). The new bottleneck is
**correctness** — and per the judge it is a *runnability* problem.

## 4. Why synthetic loses now: runnability, not syntax

The judge's recurring reason against N=64/N=128 synthetic responses:
*"non-runnable due to missing imports / undefined names (`Any`, `summaries`), wrong
APIs"* and logic/typo bugs (`summary`/`summarize`, `summaries`/`summarize`). These
are **runtime correctness failures**, not syntax errors.

**Objective syntax validity (`ast.parse` of largest code block):**

| mode | clean | baseline |
|---|---|---|
| cold | 10/10 | 10/10 |
| synthetic_N64 | **10/10** | 8/10 |
| synthetic_N128 | 7/10 | 8/10 |
| synthetic_N256 | 8/10 | 9/10 |

**Syntax is decoupled from judged quality:** N=64 has *perfect* syntax yet loses
*every* pairwise comparison; N=256 has imperfect syntax yet reaches parity. The
N=128/N=256 syntax failures are genuine (mismatched brackets, `=`/`==`) and, at
N=256, outright **token degeneration** (e.g. `个百分EGOOD`, `<!--[E2>` garbage on
2 turns) — the classic high-N compression instability. Conclusion: **`ast.parse` is
necessary but insufficient**; the handoff's optional runtime-execution gate would be
the right next instrument.

**Attribution (data vs method):** the clean *training targets* are import-complete
(117/134 contain imports; typing names imported in 19/23 cases), so the
"write complete, runnable code" behavior **is present in the data**. The centroid
*loses* it under compression — strongest evidence that the residual is a
**method/capacity** limitation, not data.

## 5. Pipeline health (all upstream gates passed)

| Stage | Result |
|---|---|
| 1 generate | 175 self-distilled coding examples (7B, coding prompt, temp 0.1) |
| 2 **quality gate** | **159/175 kept = 91%** (vs old **58%**) — data is now clean |
| 3 split | train 134 / eval 25; 100% ast-valid, 100% GOODBYE, 0 id-overlap |
| 4 train | losses N64 **0.42** · N128 **0.97** · N256 **0.22** (non-NaN) |
| 5 export | all centroids `(28, N, 512)` fp16, no NaN — correct 7B geometry |
| 6 benchmark | 50 rows (5 modes × 10 turns) |
| 7 judge | 30 verdicts, 0 API errors |

## 6. Caveats / threats to validity

- **N=128 under-training.** Its train loss (0.97) is ~2–4× N64/N256; its weak
  result (1W/9L) likely reflects under-convergence, not the method per se. Hyper-
  parameters were held identical to baseline (epochs 8 / lr 2e-3 / batch 4) for a
  clean comparison, so this was *not* re-tuned. A targeted N=128 re-train (more
  epochs / lr sweep) is the obvious follow-up; treat N=128 as noisy.
- **Sample size.** n=10 pairs/N, 30 total. Per-N deltas (esp. N=256 5/5) are
  suggestive, not significant. Single-pass A/B-randomized (seed 42), identical to
  baseline; a position-swapped dual pass (handoff §8 optional) was not run.
- **Cold baseline is identical** across runs (same model, full prompt, temp 0), so
  all deltas come from the synthetic side — the comparison is fair.

## 7. Recommendation

The data hypothesis is **partly vindicated** (completeness/quality/adherence
recovered) but **not sufficient** — clean data alone does not reach parity, and
correctness/runnability at low N is now the binding constraint. Next steps, in order:

1. **Lean into capacity:** N=256 reached parity. Sweep **N ≥ 256** (384/512) — the
   evidence says capacity, not data, is the lever now.
2. **Fix N=128 convergence** (re-train; it's an outlier) before drawing the N-curve.
3. **Add a runtime-execution gate** (sandbox import+exec) to training data and as a
   serving metric — `ast.parse` misses the dominant failure (undefined names).
4. **Investigate high-N degeneration** at N=256 (2/10 turns produced corrupted
   tokens) — likely the ceiling on naive N-scaling; may need decode-time guards.
5. Per EVAL_FINDINGS §6.2, the **inject-in-front-of-full-prompt** (LMCache+Centroid)
   deployment preserves quality (ROUGE-L 1.0) and remains the safe production mode;
   prompt-*replacement* compression is still quality-negative on this model.

---

### Reproduction notes (fixes applied vs the handoff)

The handoff was followed, with four code-level corrections discovered against the
actual repo (all required for a correct run):

1. **`generate_good_examples.py` iterated the wrong list** — it ran the search/
   research `TASKS` (lines 252–408), while the real coding `PYTHON_TASKS` (lines
   40–250) was dead code. Added `TASKS = PYTHON_TASKS`. Without this the teacher
   answers research prompts as prose and the gate drops nearly everything.
2. **Hardcoded output filename** → repointed to `vllm_good_examples_raw_2000_coding.jsonl`
   (the default `_search` file already held 131 stale records the resume logic would
   have kept).
3. **Field remap** — raw schema is `{index, task, good_example}`; training needs
   `{id, user, teacher_output}`. The gate reads `good_example`; `split_clean.py`
   maps the fields + appends GOODBYE. (The handoff's inline snippets read
   `teacher_output` straight from raw, which would have produced empty labels.)
4. **vLLM 0.20.0 `deep_gemm` crash** — kernel warmup tried FP8 DeepGEMM on the H100
   and aborted (package absent). Set `VLLM_USE_DEEP_GEMM=0`; a no-op for bf16
   numerics, so it does not affect the comparison.

Artifacts: `results/qwen7b_clean.jsonl` (benchmark), `results/judge_verdicts_clean.jsonl`
(verdicts), `centroids_qwen7b/N{64,128,256}_2000_{K,V}.npy`, `adapters/N{N}_qwen7b/`,
`data/python_agent_{train,eval}.jsonl`, `good_examples/vllm_good_examples_clean.jsonl`.
Baseline preserved: `results/qwen7b.jsonl`, `data/python_agent_train.baddata.bak.jsonl`.
