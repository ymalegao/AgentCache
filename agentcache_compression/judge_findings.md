# LLM-as-a-Judge Findings: Code Quality of Centroid Compression

**Date:** 2026-06-07
**Evaluation:** pairwise LLM-judge comparison of synthetic (centroid) vs cold
(full system prompt) responses on the multi-turn coding benchmark (§6.1).
**Judge:** OpenAI `gpt-5.5`, Responses API, reasoning effort `medium`, structured
JSON output (`judge_multi_turn.py`).
**Raw verdicts:** `results/judge_verdicts.jsonl`.
**Design + methodology:** `llm_judge_design.md`.

> **Headline:** On Qwen-7B, replacing the 2000-token system prompt with a learned
> centroid (compression mode) **measurably degrades code quality** — synthetic
> loses **22 of 30** pairwise comparisons to the full-prompt baseline. The loss
> concentrates in *completeness*, not *correctness*, and does **not** improve with
> larger centroids. This is the signature of lossy compression, and it reframes
> the paper's claim from "free speedup" to a characterized **latency/quality
> trade-off** specific to compression mode (§6.1). The prompt-retained deployment
> mode (§6.2, LMCache+Centroid) is unaffected (ROUGE-L 1.0).

---

## 1. What was tested

- **Comparison:** for each turn, the centroid (synthetic) response vs the cold
  baseline response, judged pairwise (A/B/tie) on four dimensions — correctness,
  completeness, code_quality, instruction_adherence — plus an overall verdict.
- **Protocol:** transcript-based. Each side's full conversation up to turn *t* is
  shown (reconstructed from its own responses); the judge compares only the final
  responses, using prior turns as context. A/B slot randomized per pair (seed 42).
- **Scope (33 pairs):**
  - **Qwen-7B — 30 pairs** (10 turns × N ∈ {64, 128, 256}). Clean data; the
    judgeable result.
  - **GPT-OSS-20B — 3 pairs** (turn 1 only × 3 N). The stored GPT-OSS run is
    corrupted at later turns by a Harmony-channel logging bug (see §6 and
    `llm_judge_design.md` §3.3); only the uncontaminated turn-1 cold-start is
    included, with the cold reference recovered via final-channel stripping.

---

## 2. Primary result — Qwen-7B

**Overall (synthetic vs cold), per N:**

| Condition | win | tie | loss | synthetic loss rate |
|---|---|---|---|---|
| N=64  | 4 | 0 | 6  | 60% |
| N=128 | 2 | 0 | 8  | 80% |
| N=256 | 2 | 0 | 8  | 80% |
| **Total** | **8** | **0** | **22** | **73%** |

Two findings:

1. **Centroid compression loses to the full prompt ~3:1.** Synthetic is judged
   worse in 73% of comparisons.
2. **More centroid capacity does not help — if anything it hurts.** N=64 is the
   *best* synthetic condition (60% loss); N=128 and N=256 are worse (80%). This
   directly refutes the prior hypothesis that quality would recover with larger N.

**Per-dimension (net = win − loss, over 30 pairs):**

| Dimension | win | tie | loss | net |
|---|---|---|---|---|
| correctness | 9 | 2 | 19 | **−10** (least hit) |
| instruction_adherence | 5 | 3 | 22 | −17 |
| code_quality | 4 | 3 | 23 | −19 |
| completeness | 3 | 2 | 25 | **−22** (worst hit) |

The damage is concentrated in **completeness** and least severe in
**correctness** — the model still tends to write working code, but stops doing
*everything the prompt asked*.

**Per-turn:** losses are spread across all 10 turns (every turn is net-negative;
turns 1, 4, 8, 10 are 0 win / 3 loss). There is no "warms up after turn N"
recovery — the gap is present from the cold-start turn onward.

---

## 3. Objective corroboration (judge-independent)

To confirm this is not an artifact of the LLM judge, we ran `ast.parse` on the
largest code block of every response:

| Condition | syntactically valid | broken | no code block |
|---|---|---|---|
| cold | **10/10** | 0 | 0 |
| synthetic N=64 | 8/10 | 1 | 1 |
| synthetic N=128 | 8/10 | 2 | 0 |
| synthetic N=256 | 9/10 | 1 | 0 |

- **Cold Qwen-7B produces 10/10 valid code** with the full prompt — so the model
  is *capable*; this is not "a 7B model is bad at code."
- Compression introduces **~15% hard syntax breakage** that cold never exhibits.
- **The breakage is content-specific, not random:** failures occur at turns 8
  ("write pytest unit tests") and 10 ("add a pyproject.toml") — the most
  structurally detailed tasks — and **turn 10 breaks at every N**. Random
  injection noise would scatter failures across turns; instead they land on
  exactly the tasks needing the most precise conditioning.

---

## 4. Why this happens (mechanism)

The prefix is **not** a passive warm-start. In a transformer, every generated
token attends back over the prefix KV — the system prompt's KV cache *is* the
model's working representation of the instructions. Compression mode replaces
~2000 tokens of *exact* token-KV with 64–256 *approximate* learned vectors
(a 10–30× lossy compression), so every output token is now conditioned on a
summary rather than the spec. The evidence fits this precisely:

- **Gist survives, detail is lost.** Domain identity ("Python coding agent")
  compresses fine → correctness is least hit. Enumerated, high-entropy
  requirements ("also do these six specific things") do not → completeness is
  worst hit.
- **Off-manifold approximation → occasional hard collapse.** The learned vectors
  correspond to no real tokens the model ever saw, so beyond information loss they
  can destabilize attention on the hardest tasks — matching the content-specific
  syntax breakage rather than uniform mild degradation.
- **Capacity is not the bottleneck.** N=256 (4× the prefix budget of N=64) does
  not recover quality, so the limit is how faithfully prefix-tuning can *place*
  the vectors, not how many there are.

**The controlled proof (§6.2).** When the centroid is injected *in front of the
full prompt* (LMCache+Centroid, §6.2), all 2000 exact tokens remain and quality
is perfect (ROUGE-L 1.0). Same injection mechanism, same model — the only
difference is whether the prompt's information is still present. The degradation
is therefore caused by **removing the prompt's information**, not by the centroid
mechanism itself.

---

## 5. Validity / threats

- **Position bias: present but mild, does not overturn the result.** Synthetic
  wins 33% when shown first (slot A) vs 24% when shown second (B) — a ~9pp
  first-position advantage. But synthetic loses the majority in *both* slots
  (67% in the favored slot, 76% in the disfavored one), and 5 of its 8 wins came
  from the *disfavored* slot B. Merit, not order, drives the outcome.
- **No tie band.** The judge returned **zero overall ties** (it always forced a
  winner; only 10 dimension-level ties across 120 dimension judgments). In 18/30
  verdicts the reasoning flags faults on *both* sides — the judge is often
  choosing the *less broken* of two flawed answers. The 73% loss *rate* should
  therefore be read as directional, not as a precise magnitude.
- **Slot imbalance.** With seed 42, cold landed in slot A 21× vs synthetic 9× —
  an unlucky single-seed draw. A dual-pass run (each pair judged in both A/B
  orders, averaged) would neutralize both this and the position bias; not yet run.
- **Quality floor.** Both conditions are 7B-quality; comparisons are often
  "least-bad," which adds noise to the magnitude (not to the direction).
- **Scale generality is open.** The within-model design rules out the trivial
  "small model is bad at code" explanation, but not the refined hypothesis that
  *small models rely more on literal prompt text and a larger model might encode
  the prompt into the centroid more robustly.* This is untested.

---

## 6. GPT-OSS-20B — inconclusive (n=3), data needs a rerun

- Turn-1 cold-start only: **1 win / 1 tie / 1 loss** — a wash, far too small to
  interpret.
- The stored GPT-OSS run (`gptmulti_turn_benchmark.jsonl`) is unusable for
  multi-turn quality eval: a logging bug fed raw Harmony channel text
  (`analysis…assistantfinal…`) back as conversation history, degenerating later
  turns in *all* modes (cold from turn 6, warm_apc from turn 3, etc.). The fix is
  committed in `multi_turn_benchmark.py` (`strip_to_final_channel`); rerun
  instructions are in `RERUN_GPTOSS.md`. The GPT-OSS rerun is the clean test of
  the scale-generality question in §5.

---

## 7. Implications for the paper

1. **The "free speedup / quality preserved" framing for compression mode is no
   longer supportable** and must be revised in the introduction and §6.1.
2. **Report the honest trade-off.** On a 7B model, centroid compression cuts
   cold-start TTFT but costs code completeness/quality and introduces ~15% syntax
   breakage; correctness is mostly preserved.
3. **Reconcile §6.1 and §6.2 explicitly** — they are two different uses of the
   centroid, and the contrast *is* a contribution:

   | Mode | Centroid role | Quality |
   |---|---|---|
   | §6.2 LMCache+Centroid | injected **in front of** the full prompt | ROUGE-L 1.0 (preserved) |
   | §6.1 compression | **replaces** the prompt | 73% judge loss (degraded) |

4. **Use the objective syntax metric** (cold 10/10 vs synthetic ~8.3/10) alongside
   the judge — it is reviewer-proof and judge-independent.
5. **Frame scale as the key open question**, answered by the GPT-OSS-20B rerun.

**Recommended next steps:** (a) add a dual-pass position-swapped judge run to make
the magnitude bulletproof; (b) complete the GPT-OSS-20B rerun for the
scale-generality test; (c) revise the paper's quality claims per the table above.
