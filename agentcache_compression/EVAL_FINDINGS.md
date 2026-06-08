# AgentCache — Code-Quality Evaluation Findings

**Date:** 2026-06-07
**Evaluation:** LLM-as-a-judge (gpt-5.5) pairwise code-quality comparison of
centroid-compression (synthetic) vs full-system-prompt (cold) responses on the
multi-turn benchmark, plus objective syntax-validity checks and a training-data
audit.
**Artifacts:** `judge_multi_turn.py`, `results/judge_verdicts.jsonl`,
`llm_judge_design.md` (protocol), this file (findings).

---

## TL;DR

1. On Qwen-7B, **centroid compression measurably degrades code quality**: the
   synthetic condition loses **22/30 (73%)** of pairwise comparisons to the
   full-system-prompt baseline. The result is robust to position bias.
2. Quality **does not improve with centroid size N** (N=64 is the *best*: 4/10;
   N=128 and N=256 are *worse*: 2/10 each) — refuting the design-doc hypothesis
   that more virtual tokens recover quality.
3. Losses concentrate in **completeness** (net −22) and **code_quality** (−19);
   **correctness** is least affected (−10). Signature of lossy compression: the
   coding *gist* survives, the prompt's detailed *requirement coverage* does not.
4. Objective, judge-independent confirmation: cold produces **10/10**
   syntactically valid code; synthetic produces ~8.3/10. Breakage is
   **content-specific** (the hardest tasks — pytest tests, pyproject packaging),
   not random.
5. **Likely primary root cause — training data, not the method.** The prefix was
   trained (label-masked on teacher outputs) to imitate a **1.5B teacher model
   whose outputs are 42% syntactically broken**. The centroid distills that
   broken distribution into a capable 7B model. This is *fixable* and confounds
   the "compression has a quality floor" interpretation.
6. **GPT-OSS-20B is not evaluable here** (n=3, turn-1 only, from the
   Harmony-corrupted run). Whether degradation is model-scale-dependent is an
   open question requiring the rerun.

---

## 1. Setup

- **Data:** `results/qwen7b.jsonl` (clean) and
  `results/gptmulti_turn_benchmark.jsonl` (partly corrupted — see §6).
  One 10-turn CSV-CLI Python coding conversation, modes cold / warm_apc /
  synthetic N∈{64,128,256}, generated at temperature 0.
- **Judge:** gpt-5.5 via the Responses API, `reasoning.effort=medium`, strict
  json_schema verdicts. Pairwise A/B/tie on four rubric dimensions + overall,
  with per-pair A/B-slot randomization (seed 42). Full protocol and system
  prompt: `llm_judge_design.md`.
- **Pairs:** 33 total — Qwen-7B all 10 turns × 3 N = 30; GPT-OSS-20B turn-1
  only × 3 N = 3 (only the no-history cold-start turn is defensible from the
  corrupted run).
- **Cost:** ~$2–3 for the full run.

---

## 2. Headline result (Qwen-7B): synthetic vs cold

| N | win | tie | loss |
|---|----|----|----|
| 64  | 4 | 0 | 6 |
| 128 | 2 | 0 | 8 |
| 256 | 2 | 0 | 8 |
| **all** | **8** | **0** | **22** |

Synthetic loses **73%** of comparisons. More centroid capacity does **not** help
(N=64 best). Since TTFT is roughly flat in N above a floor, there is no N that
buys back quality at constant latency — the ablation's hoped-for "pick larger N
for quality" story does not hold.

### Per-dimension (Qwen-7B, net = win − loss)

| dimension | win | tie | loss | net |
|---|----|----|----|----|
| correctness | 9 | 2 | 19 | −10 |
| instruction_adherence | 5 | 3 | 22 | −17 |
| code_quality | 4 | 3 | 23 | −19 |
| **completeness** | 3 | 2 | 25 | **−22** |

Completeness is hit hardest, correctness least. The compressed prefix preserves
"write working Python" better than "satisfy every enumerated requirement."

---

## 3. Is the judge trustworthy? (bias checks)

- **Position bias is mild and does not overturn the result.** Synthetic wins 33%
  when shown first (slot A) vs 24% when shown second (slot B) — a ~9pp
  first-position advantage (a known LLM-judge artifact). But synthetic loses the
  majority in *both* slots (67% in the favored slot, 76% in the disfavored one),
  and 5 of its 8 wins came from the *disfavored* slot. Merit, not position,
  drives the outcome.
- **Verdicts are substantive, not stylistic.** gpt-5.5 cites concrete defects:
  missing `List/Dict` imports that crash on run, a broken `--output` refactor,
  incomplete input validation, malformed pytest parametrization with wrong
  expected values. (Smoke-test example: it correctly preferred
  `pd.api.types.is_numeric_dtype()` over a manual `dtype.kind` check that misses
  unsigned ints.)
- **Caveats to disclose in the paper:** (a) the judge produced **zero overall
  ties** — it always forced a winner, which sharpens the apparent loss rate;
  (b) seed-42 happened to place cold in slot A 21/30 times. Both are addressable
  with a dual-pass position-swapped rerun (judge each pair in both orders,
  average; ~$2 more). Recommended before camera-ready.

---

## 4. Objective syntax validity (judge-independent)

Largest fenced code block per response, checked with `ast.parse`:

| condition | valid | broken | no code |
|---|----|----|----|
| cold | **10/10** | 0 | 0 |
| synthetic N=64 | 8/10 | 1 | 1 |
| synthetic N=128 | 8/10 | 2 | 0 |
| synthetic N=256 | 9/10 | 1 | 0 |

Two things matter here:

- **Cold Qwen-7B writes 10/10 valid code** with the real prompt — so the model
  is *not* inherently incapable; the centroid specifically introduces ~15% hard
  syntax breakage.
- **Breakage is content-specific:** failures land on **turn 8 (write pytest
  tests)** and **turn 10 (add pyproject.toml)** — the most structurally detailed
  tasks — and turn 10 breaks at *every* N. Random noise would scatter; instead it
  fails exactly where the most precise conditioning is needed.

This metric is reviewer-proof (no LLM judge involved) and should be reported
alongside the judge results.

---

## 5. Why does replacing the prefix change quality? (mechanism)

The prefix is not a passive warm-start — in a transformer, **every generated
token attends back over the prefix KV**; the prefix *is* the model's working
representation of the instructions. Replacing ~2000 exact token-KVs with 64–256
learned vectors is **10–30× lossy compression**, and the evidence matches that
mechanism point for point:

- **Gist survives, detail is lost** → correctness least hit (−10), completeness
  worst (−22). The domain ("Python coding agent") compresses fine; the prompt's
  enumerated requirements do not.
- **Off-manifold approximation** → the learned vectors correspond to no real
  tokens, so even "preserved" information is approximate, producing occasional
  *hard collapse* (syntax breakage) on the hardest tasks rather than uniform mild
  degradation.
- **More N doesn't help** → the bottleneck is approximation *quality*, not token
  budget; otherwise N=256 (4× capacity) would recover quality.
- **Controlled proof it's information loss, not the centroid mechanism:** in §6.2
  (LMCache+Centroid) the centroid is injected *in front of the full prompt* — all
  2000 exact tokens remain — and quality is perfect (ROUGE-L 1.0). Same
  injection, same model. The only difference is whether the prompt's information
  is present. So degradation is caused by *removing* the prompt's information.

**Deployment implication:** centroid-as-prefix-anchor (prompt retained, §6.2) is
quality-safe; centroid-as-prompt-replacement (§6.1) trades quality for TTFT.

---

## 6. Likely primary root cause: training-data quality

The mechanism in §5 is real but may be *secondary*. A stronger, fixable cause
sits in the training data.

**The prefix is trained with loss computed only on teacher outputs** (label
masking — `train_prefix_compression.py`). The centroid is therefore optimized to
reproduce the teacher distribution. Audit of the documented training set
`data/python_agent_train.jsonl` (118 examples):

- **Teacher model = `qwen-1.5b`** (`generate_good_examples.py:10`,
  `DEFAULT_MODEL_ID`).
- **42% of teacher outputs are syntactically broken** (49/118 fail `ast.parse`;
  67 valid; 2 no-code). Real defects, e.g. `engine =Alchemy.create_engine(...`
  (mangled `sqlalchemy`), unterminated f-strings, unmatched parens.

So the pipeline distills a **42%-broken 1.5B distribution** into the prefix and
injects it into a 7B model that, unaided, writes valid code 10/10. This predicts
exactly the observed symptom (0% → ~15% breakage, completeness collapse) and is
**confounded with** the intrinsic-compression interpretation — both predict the
same regression.

**Provenance caveat:** the committed centroids are Llama-shaped
(`[16, N, 512]`); the Qwen-7B benchmark centroids were trained on the Blackwell
machine and are not in the repo. The audit above is of the *documented* pipeline
and training file; the inference that the Qwen-7B centroid used this same
1.5B-teacher data should be confirmed, not assumed.

**Side finding:** 0/118 teacher outputs end with `GOODBYE`, despite
`note.md`/`HANDOFF.md` stating GOODBYE was appended. This training file either
predates or differs from the GOODBYE-appended version — relevant to the
behavioral-encoding (GOODBYE) metric story.

---

## 7. GPT-OSS-20B: not evaluable from current data

The GPT-OSS run is corrupted by a Harmony-channel history-feedback bug
(`llm_judge_design.md` §3.3): later turns degenerate in all modes. Only turn 1
(no history) is salvageable, giving 3 pairs → **1 win / 1 tie / 1 loss**, which
is a statistical wash. This neither confirms nor refutes scale-dependence.
**Whether the 7B degradation shrinks on larger models is open** and requires the
fixed-benchmark rerun (`RERUN_GPTOSS.md`).

---

## 8. What this means for the paper

- The "free speedup / quality preserved" framing for **compression mode (§6.1)**
  is no longer tenable and must change. The honest claim is a **characterized
  latency/quality trade-off** with a clear deployment recommendation (use the
  prompt-retained §6.2 path when quality matters).
- Report the **objective syntax metric** (cold 10/10 vs synthetic ~8.3/10)
  alongside the judge — it does not depend on the LLM judge.
- State the result as **scale-bounded** (measured at 7B; larger-scale behavior
  open) and **confounded by training data** (1.5B teacher, 42% broken).

## 9. Recommended next experiments (in priority order)

1. **Dual-pass position-swap judge rerun** (~$2) — removes the position-bias and
   zero-ties caveats; makes the headline result bulletproof.
2. **Training-data audit as a reported number** — the 42%-broken stat is cheap,
   objective, and motivates everything below. (Optional: Tier-2 pairwise judge of
   teacher outputs vs a frontier model on a sample, to quantify the gap.)
3. **Frontier-teacher / SWE-bench-quality retrain (the causal test)** —
   regenerate training data with a strong model or verified-correct Python,
   retrain the prefix, re-export, re-run benchmark + judge. This **decomposes**
   the 73% loss into fixable "bad data" vs intrinsic "compression floor." Highest
   value; needs GPU + retraining.
4. **GPT-OSS-20B rerun** (`RERUN_GPTOSS.md`) — answers the model-scale question.
