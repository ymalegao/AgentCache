# Review: AgentCache project & todo.md plan

## Context

Critique of the AgentCache project and the proposed 16-phase experiment in `todo.md`. AgentCache trains a PEFT prefix adapter and seeds its K/V tensors into vLLM's KV cache to skip prefill of a static "agent persona" prompt. The todo proposes fixing a "replacement vs compression" architectural bug, training adapters at N ∈ {32, 64, 128, 256}, building an eval set, and comparing against cold start + vLLM APC.

This is a strategic critique with reordered phases — not an implementation.

---

## Verdict

**Mixed — pursue, but reorder.** The todo correctly identifies the most important bug in the current pipeline (positions 0..N-1 silently shadow real prompt content) and proposes proper label-masked distillation training. But it buries the question that determines whether the project is worth doing at all: how does synthetic KV beat vLLM's automatic prefix caching (APC), which is exact, free, and already shipping? Until that is answered with a cold-vs-warm APC measurement, the rest of the matrix is premature.

Reorder: kill-shot the APC question first, pre-register a numeric success bar, then run a sharper experiment with a real natural-language-shortened-prompt baseline. If cold APC ≈ inject TTFT on warm requests, kill the project or pivot to a setting where APC fundamentally cannot help (per-user prefix variants, untokenizable behavior).

---

## What todo.md gets right

- **Identifies the replacement-vs-compression bug** (todo "Important distinction"). The current `gap = N` design treats the first N *logical* prompt positions as already computed, but those positions hold real chat-template tokens. With `system_retain_ratio = 0`, the first N tokens of the user-visible prompt are silently skipped. This is a real correctness bug for the stated goal.
- **Trains separate adapters per N** instead of slicing — matches `HANDOFF.md` "Known limitations" #3.
- **Label-masking on assistant tokens only** (Phase 4). The current `prefixtraining.py` uses a generic LM collator over the entire sequence, which dilutes the learning signal. Phase 4 fixes this.
- **Includes APC as a baseline** (Phase 12) and the `GOODBYE` diagnostic for hard-rule compliance (Phase 1). The `GOODBYE` rule is a clever boolean check distinct from the coherence heuristic.
- **Failure mode classification** (Phase 15) — good scientific hygiene.

---

## Critical issues

### 1. APC baseline is buried at Phase 12 — it must be Phase 0

`HANDOFF.md` already flags this risk verbatim: *"If vLLM's APC warms up within 1 request, your value prop only applies to single-shot agents."* The current `benchmark_ttft.py` measures warm APC only, which is unfair to synthetic KV but also masks whether APC trivially dominates.

Until cold APC vs warm APC vs cold start are measured on the actual eval prompts, you don't know if there's a problem worth solving. APC is memcpy of exact K/V from a hash-keyed cache — if the prompt is reused even once, APC's second-request TTFT is essentially the seed cost alone, which is what synthetic injection competes against without the quality loss.

**Fix:** Phase 0.5 — measure cold APC vs warm APC vs cold start on the existing 149-token Llama prompt before any training. Decision rule:
- If APC warms in <1 request to ≈ inject TTFT: synthetic KV must target single-shot or per-user variants. Rescope or kill.
- If APC stays expensive across requests (eviction, prompt churn): proceed with todo.

### 2. Phase 7 fallback re-introduces a known bug class

The fallback prototype prepends dummy placeholder tokens (`[dummy*N][physical]`) and skips them in the scheduler. `HANDOFF.md` says explicitly: *"Deprecated approach (do not revive): `inject_ids = [bos] * N + physical_ids` dummy prepend for Llama. It inflated prompt_len without reducing computed tokens vs cold and masked scheduler bugs."*

Phase 7 says "this is not the deprecated dummy-padding benchmark," but functionally it is the same prepend pattern. The risk: dummy tokens *do* end up contributing to attention or position-id arithmetic and your numbers are corrupted exactly the way the prior Llama benchmark was.

**Fix:** Commit to the "preferred" implementation only (vLLM scheduler change: scheduled prefill = M tokens, positions = N..N+M-1). Drop the dummy-placeholder fallback entirely. If the preferred path is too invasive, that's a signal the project isn't ready to scale, not a signal to revive the dead-end.

### 3. n=20 is too small for the conclusions the matrix implies

Phase 16's done criteria imply a Pareto comparison across 7 modes × 4 budgets × multiple metrics. With n=20, a single failed task moves task-pass rate by 5pp. The honest reading of any such table is "noise." Either commit to ≥100 tasks for paper-grade claims, or shrink the matrix to 2–3 conditions and present it as a feasibility study, not a Pareto curve.

### 4. The "agent" framing isn't operationalized

Project is named **Agent**Cache. README mentions SWE-bench. The eval in todo is single-turn Python QA with a `GOODBYE` rule. None of the tasks involve:
- Multi-turn conversation (where a persistent system prompt matters most)
- Tool use / function calling
- Long context (TTFT savings only matter at scale)
- Real agent trajectories — and `personas/`, `generate_tasks.py`, `generate_good_examples.py` already produce these but aren't used

Either rescope to "static prompt compression for single-turn LLM calls" or include at least one multi-turn / tool-use task in the eval. The current shape is single-turn QA with an agent-shaped wrapper.

### 5. Persona angle from README is dropped

The persona-perturbed pipeline exists (`personas/user*.yaml` + `generate_tasks.py`) but todo's training data is `python_agent_teacher_outputs.jsonl` with no persona variation. You're testing a weaker hypothesis than the existing data pipeline supports. If the persona angle is what makes this paper-worthy (a prefix that survives noisy user inputs), it has to be in the training data.

### 6. No quantitative success bar

"Preserves most task behavior while reducing TTFT" is unfalsifiable. Pre-register a numeric bar — e.g., "inject TTFT ≤ 0.5× cold AND task pass rate ≥ 90% of teacher AND GOODBYE compliance ≥ 70%" — *before* running experiments. Otherwise the experiment becomes a search for a flattering interpretation.

### 7. Missing baseline: hand-shortened natural-language system prompt

If your story is "lossy compression of system prompt behavior into N virtual tokens," the obvious baseline is "lossy compression into N natural-language tokens." A 64-token shortened prompt is also a lossy compression — and it's interpretable, debuggable, and ships without a vLLM fork. You must show that 64 virtual tokens encode *more* behavior than 64 prompt tokens, or the project has no story against a $0-engineering baseline.

---

## Bigger-picture concerns

### A. Current speedups don't justify the engineering cost

Per `HANDOFF.md`:
- Qwen-1.5B: ~1.2× on long prompts
- Llama-3.2-1B: ~1.0× on the 149-token test

Warm APC gives near-zero TTFT for free. Either the synthetic story is "single-shot" (narrow market) or "untokenizable behavior" (unmeasured). The current numbers don't beat the simpler alternatives on the easy case.

### B. Quality is "spirit of the answer," not parity

`HANDOFF.md`: *"Inject answers the question in spirit (timing + time), but uses a function wrapper rather than a proper contextmanager... Treat as quality pass, not parity pass."* For agents that follow strict tool-call schemas or hard rules (the `GOODBYE` diagnostic exists exactly because this is suspected), "spirit-of-the-answer" is not enough. The eval needs to surface this gap, not paper over it with coherence heuristics.

### C. Pipeline portability claims are unsubstantiated

`README.md` "Next Steps" lists 5 generalization concerns (GQA, RoPE variants, MLA, scale, PEFT MLP). The todo trains only on Llama-3.2-1B. README's recommended Test 5 (Llama-3.2-3B end-to-end) isn't in the todo. You can't call it a "pipeline" if it works on one model — that's a script.

---

## Recommended reordered plan

| Phase | Source | Notes |
|---|---|---|
| **0** — cold/warm APC measurement | NEW | Kill-shot. Reuses `benchmark_ttft.py`. Run before any training. |
| **1** — pre-register numeric success bar | NEW | Single doc. TTFT target + quality target + go/no-go rule. |
| **2** — eval set construction | todo P1–P3 | Bump to ≥100 tasks; include ≥2 multi-turn or tool-use tasks. |
| **3** — natural-language baseline | NEW | Hand-shorten system prompt to N∈{32,64,128,256} tokens. The real compression baseline. |
| **4** — training fix + export | todo P4–P6 | Restrict to N∈{64,256} initially. Use persona-perturbed data, not generic teacher outputs. |
| **5** — compression-mode scheduler change (PREFERRED only) | todo P7 | No dummy-placeholder fallback. |
| **6** — port to Llama-3.2-3B | README Test 5 | Pipeline generalization gate before scaling eval. |
| **7** — run matrix | todo P10–P11 | Apply pre-registered bar from Phase 1. |
| **8** — go/no-go decision | NEW | If pre-registered bar fails: stop, write up as negative result. |

Dropped or deferred: dynamic budget selection (todo P9), most plotting (P13), N∈{32,128} adapters (defer until N∈{64,256} clears the bar).

---

## Critical files (referenced — no edits proposed here)

- `README.md` — project goal + Next Steps generalization concerns
- `HANDOFF.md` — current empirical status, deprecated approaches (Llama dummy padding)
- `todo.md` — proposed plan being critiqued
- `prefixtraining.py` — current training script (todo P4 will modify; needs label masking on assistant tokens)
- `transpose_tensors.py` — export script (uses `PeftModel.get_prompt()` correctly for `prefix_projection=True`; needs `--device cuda` and GQA awareness for scale-up per README "Next Steps" §1)
- `test_injection.py` — replacement-mode smoke test
- `benchmark_ttft.py` — has Cold/APC/Inject conditions; Phase 0 reuses but needs cold-APC measurement added (fresh engine restart between APC trials)
- `vllm/v1/core/sched/scheduler.py` — scheduler hook for `centroid_sched_gap` (Phase 5 / todo P7 modifies)
- `vllm/centroid_injector.py` — `seed_prefix_into_kv_cache` + RoPE offset
- `personas/`, `generate_tasks.py`, `generate_good_examples.py` — unused persona pipeline that should feed Phase 4 training data

---

## Open strategic questions

These determine whether the recommended plan is the right one or whether to rescope further:

1. **Paper, library, or product?** Determines required rigor and what "done" looks like.
2. **Target use case — single-shot, multi-turn, batched offline?** Each implies a different baseline and a different reason APC may or may not dominate.
3. **Persona variation in scope?** The persona pipeline exists but todo ignores it. Decide whether persona resilience is the story or a follow-up.
4. **Acceptable to kill the project if Phase 0 shows APC dominates?** If no, the plan changes — you'd need to commit to a setting (per-user variants, eviction-heavy load) where APC fundamentally can't help, before Phase 0.

---

## Verification (if the reordered plan is adopted)

The plan itself is a critique, not an implementation. End-to-end verification of the *follow-up* execution would be:

- **Phase 0:** Run `benchmark_ttft.py` with engine restart between APC trials. Compare TTFT[1] (cold APC) vs TTFT[2] (warm APC) vs cold start. Decision recorded against pre-registered rule.
- **Phase 4:** For each trained adapter, confirm exported shape matches `[num_layers, N, num_kv_heads * head_dim]`. Read `num_kv_heads` from the model config, not `adapter_config.json` (which knows nothing about GQA — per README "Next Steps" §1).
- **Phase 5:** Confirm `n_scheduled_tokens == physical_prompt_tokens` and `positions_minmax = (N, N+M-1)` in scheduler logs. This proves compression mode, not replacement mode.
- **Phase 6:** Run the same inject smoke test on Llama-3.2-3B and confirm output is coherent. If garbled, fix `transpose_tensors.py` shape handling before the matrix.
- **Phase 8:** Compare against pre-registered bar from Phase 1. Pass / fail / pivot — committed before the numbers came in.
