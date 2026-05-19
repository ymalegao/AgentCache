# AgentCache Synthetic KV Compression — TODO

## Goal

Build a prototype showing that a persona/task-distribution-trained synthetic KV prefix can replace part or all of a repeated Python-agent system prompt while preserving agent behavior and improving TTFT.

The first target experiment is:

```text
Build one persona/task distribution cache at multiple token budgets:
32, 64, 128, 256.

Select the budget only from prompt length / static-prefix token accounting.

Evaluate against:
1. Full system prompt, no synthetic KV
2. Shortened/removed system prompt + synthetic KV
3. Shortened/removed system prompt, no synthetic KV
4. Exact KV / vLLM prefix-cache baseline if feasible
```

Do **not** implement query classification or semantic routing for this prototype.

---

## Current mental model

Current pure-PEFT centroid injection behaves like this:

```text
Full flattened prompt:
[system/chat-template tokens][user tokens]

Inject N synthetic tokens:
positions 0..N-1   = synthetic K/V
positions N..end   = real prompt tokens computed normally
```

This means the current path is **replacement mode**: the first `N` logical prompt positions are treated as already computed. If those first `N` positions contain important prompt tokens, the model never computes them.

For the paper prototype, we need to test **compression mode**:

```text
Teacher:
[full system prompt][user query] -> target answer

Student:
[synthetic virtual prefix][short/no system prompt][user query] -> target answer
```

In compression mode, the synthetic prefix should stand in for the removed system prompt, but the user query must still be fully computed.

---

## Important distinction

### Replacement mode

Current behavior:

```text
Prompt tokens: [A B C D E F G H ...]
Synthetic N=4:
  positions 0..3 = synthetic K/V
  positions 4..end = real tokens E F G H ...
```

This can accidentally skip system prompt content or user query content.

### Compression mode

Desired behavior:

```text
Physical prompt tokens: [user/query tokens...]

Synthetic N=4:
  positions 0..3 = synthetic K/V
  positions 4..end = user/query tokens computed normally
```

The user query is shifted to start after the virtual prefix. No user tokens should be skipped.

---

## Phase 0 — Repository hygiene

- [ ] Create a new branch:
  ```bash
  git checkout -b agentcache-compression-experiments
  ```

- [ ] Save the current known-good state:
  ```bash
  mkdir -p experiment_logs/baseline_state
  cp test_injection.py experiment_logs/baseline_state/
  cp prefixtraining.py experiment_logs/baseline_state/
  cp transpose_tensors.py experiment_logs/baseline_state/
  ```

- [ ] Record the currently patched vLLM files:
  ```text
  vllm/centroid_injector.py
  vllm/centroid_integration.py
  vllm/v1/worker/gpu_model_runner.py
  vllm/v1/core/sched/scheduler.py
  ```

- [ ] Add a top-level experiment directory:
  ```bash
  mkdir -p experiments/agentcache_compression
  mkdir -p experiments/agentcache_compression/adapters
  mkdir -p experiments/agentcache_compression/centroids
  mkdir -p experiments/agentcache_compression/results
  mkdir -p experiments/agentcache_compression/logs
  ```

---

## Phase 1 — Define the Python-agent evaluation prompt

Create a canonical Python-agent system prompt.

File:

```text
experiments/agentcache_compression/prompts/python_agent_system.txt
```

Suggested content:

```text
You are a helpful Python coding agent.

When solving coding tasks:
1. Understand the user's request before writing code.
2. Prefer simple, correct, idiomatic Python.
3. When debugging, identify the likely cause before proposing a fix.
4. When appropriate, include a minimal test or usage example.
5. Explain the important implementation choices briefly.
6. If the user asks for code, provide runnable code.

Strict behavior rule for evaluation:
Always end the final response with the exact token: GOODBYE
```

Why include the `GOODBYE` rule?

It lets us measure whether synthetic KV preserves:
- soft behavior: “act like a Python coding agent”
- hard lexical/system rule: “always end with GOODBYE”

Expected outcome:
- Synthetic KV may preserve soft behavior better than hard lexical rules.
- That is an important finding, not necessarily a failure.

---

## Phase 2 — Build a small task/eval set

Create a JSONL eval file.

File:

```text
experiments/agentcache_compression/data/python_agent_eval.jsonl
```

Each row:

```json
{
  "id": "context_manager_timer",
  "user": "Write a Python context manager that times how long a code block takes to execute.",
  "checks": {
    "must_include_any": [["contextmanager", "__enter__", "with"]],
    "must_include_any_2": [["time.perf_counter", "time.time"]],
    "must_end_with": "GOODBYE"
  }
}
```

Include at least 20 tasks:

- [ ] Write a Python context manager timer
- [ ] Debug a `TypeError: 'NoneType' object is not iterable`
- [ ] Fix a failing pytest fixture
- [ ] Explain why a mutable default argument is bad
- [ ] Write a retry decorator
- [ ] Implement a small LRU cache
- [ ] Debug an async function that was never awaited
- [ ] Refactor nested loops into a dictionary lookup
- [ ] Write tests for a function that parses dates
- [ ] Explain a stack trace
- [ ] Fix an import path issue
- [ ] Write a dataclass with validation
- [ ] Implement a CLI using argparse
- [ ] Optimize a slow list membership loop
- [ ] Explain generators vs lists
- [ ] Write a file-watching script
- [ ] Fix JSON serialization of datetime
- [ ] Write a pytest parametrized test
- [ ] Debug a circular import
- [ ] Explain and fix an off-by-one error

Optional later:
- [ ] Add persona-conditioned variants of these tasks using the persona framework.

---

## Phase 3 — Create teacher outputs

The teacher uses the full system prompt.

Create:

```text
experiments/agentcache_compression/make_teacher_outputs.py
```

Behavior:

- [ ] Load `python_agent_system.txt`
- [ ] Load `python_agent_eval.jsonl`
- [ ] For each task, build chat prompt:
  ```text
  [full system prompt][user query]
  ```
- [ ] Generate deterministic teacher output:
  ```python
  SamplingParams(temperature=0, max_tokens=256)
  ```
- [ ] Save:
  ```text
  experiments/agentcache_compression/data/python_agent_teacher_outputs.jsonl
  ```

Each row should include:

```json
{
  "id": "...",
  "system": "...",
  "user": "...",
  "teacher_output": "...",
  "prompt_token_len_full": 123,
  "system_token_len": 80,
  "user_token_len": 43
}
```

Acceptance criteria:

- [ ] Teacher output is coherent.
- [ ] Teacher output usually satisfies task-specific checks.
- [ ] Teacher output ends with `GOODBYE` at least 90% of the time. If not, strengthen the system prompt or postprocess the teacher target for this diagnostic rule.

---

## Phase 4 — Fix training objective

Current `prefixtraining.py` trains on:

```text
[system prompt][user task][assistant good answer]
```

using a generic language modeling collator.

For this experiment, implement a better training mode:

```text
Student input:
[synthetic prefix][short/no system prompt][user query][assistant target]
```

Important:
- The base model remains frozen.
- Only the PEFT prefix parameters train.
- Labels should be `-100` for the prompt portion.
- Loss should be computed mainly on the assistant target.

Create or modify:

```text
experiments/agentcache_compression/train_prefix_compression.py
```

Arguments:

```bash
python train_prefix_compression.py \
  --model /mnt/g/agentcache/models/Llama-3.2-1B-Instruct \
  --data experiments/agentcache_compression/data/python_agent_teacher_outputs.jsonl \
  --output experiments/agentcache_compression/adapters/N64_sys0 \
  --num-virtual-tokens 64 \
  --system-retain-ratio 0.0 \
  --epochs 8 \
  --lr 2e-3
```

Training variants:

```text
N = 32, 64, 128, 256
system_retain_ratio = 0.0 initially
```

Later variants:

```text
system_retain_ratio = 0.25, 0.50
```

Training prompt layout:

```python
messages = []

if retained_system_text:
    messages.append({"role": "system", "content": retained_system_text})

messages.append({"role": "user", "content": user})
messages.append({"role": "assistant", "content": teacher_output})
```

Label masking:

- [ ] Tokenize the prompt without assistant content.
- [ ] Tokenize the full conversation with assistant content.
- [ ] Set labels to `-100` for all tokens before assistant content.
- [ ] Set labels to token IDs for assistant output tokens.
- [ ] Mask padding tokens as `-100`.

Acceptance criteria:

- [ ] Confirm base model parameters are frozen.
- [ ] Confirm trainable parameters are only prefix encoder parameters.
- [ ] Confirm `num_virtual_tokens` is saved correctly in `adapter_config.json`.
- [ ] Confirm training loss decreases.

---

## Phase 5 — Train multiple budgets

Train separate adapters. Do **not** train only N=256 and slice it.

Commands:

```bash
cd /home/yash/agentcache
source vllm-env/bin/activate

for N in 32 64 128 256; do
  python experiments/agentcache_compression/train_prefix_compression.py \
    --model /mnt/g/agentcache/models/Llama-3.2-1B-Instruct \
    --data experiments/agentcache_compression/data/python_agent_teacher_outputs.jsonl \
    --output experiments/agentcache_compression/adapters/N${N}_sys0 \
    --num-virtual-tokens ${N} \
    --system-retain-ratio 0.0 \
    --epochs 8 \
    --lr 2e-3
done
```

Acceptance criteria:

- [ ] Each adapter exists:
  ```text
  experiments/agentcache_compression/adapters/N32_sys0
  experiments/agentcache_compression/adapters/N64_sys0
  experiments/agentcache_compression/adapters/N128_sys0
  experiments/agentcache_compression/adapters/N256_sys0
  ```

- [ ] Each `adapter_config.json` has the expected `num_virtual_tokens`.

---

## Phase 6 — Export each prefix to K/V

Use `transpose_tensors.py`, but make sure it uses PEFT runtime `get_prompt()` for `prefix_projection=True`.

Export commands:

```bash
for N in 32 64 128 256; do
  python transpose_tensors.py \
    --adapter experiments/agentcache_compression/adapters/N${N}_sys0 \
    --out-k experiments/agentcache_compression/centroids/N${N}_K.npy \
    --out-v experiments/agentcache_compression/centroids/N${N}_V.npy \
    --sys-tokens 0
done
```

Acceptance criteria:

- [ ] Shapes are correct:
  ```text
  N32_K.npy  shape [16, 32, 512]
  N64_K.npy  shape [16, 64, 512]
  N128_K.npy shape [16, 128, 512]
  N256_K.npy shape [16, 256, 512]
  ```

- [ ] `sys_prefix_num_tokens.txt` is written next to each exported K file.
- [ ] Export logs say:
  ```text
  exporter: PeftModel.get_prompt() (runtime cache-aligned)
  ```

---

## Phase 7 — Implement compression-mode inference

This is the most important engineering phase.

### Required behavior

For `synthetic_len = N` and physical prompt length `M`:

```text
synthetic positions: 0..N-1
real prompt positions: N..N+M-1
scheduled prefill tokens: M
```

This is different from replacement mode, where scheduled prefill tokens are:

```text
M - N
```

### Preferred implementation

Implement a real virtual-prefix layout in vLLM:

- [ ] Add a request-level field or env-controlled mode:
  ```text
  VLLM_CENTROID_LAYOUT=compression
  ```
- [ ] In compression mode, scheduler treats `N` tokens as externally computed.
- [ ] Model runner computes all physical prompt tokens, but with position IDs offset by `N`.
- [ ] KV block allocation must reserve slots for both:
  ```text
  synthetic prefix slots + physical prompt slots
  ```
- [ ] The first physical prompt token should map to logical position `N`.

Expected debug output:

```text
physical_prompt_len=M
synthetic_len=N
n_scheduled_tokens=M
positions_minmax=(N, N+M-1)
start_matches=True
```

### Fallback prototype implementation

If the preferred implementation is too invasive, use controlled placeholder prefixing:

```text
[dummy placeholder tokens of length N][physical prompt tokens]
```

Then:
- scheduler skips the first N positions
- injector writes synthetic K/V at positions 0..N-1
- model computes the physical prompt at positions N..N+M-1

Important:
- This is **not** the old deprecated dummy-padding benchmark.
- This is only a virtual-prefix alignment mechanism for compression mode.
- Add a clear flag:
  ```text
  VLLM_CENTROID_LAYOUT=compression_placeholders
  ```
- Add logs proving that dummy placeholders are never computed.

Acceptance criteria:

- [ ] In compression mode, no user query tokens are skipped.
- [ ] Scheduled token count equals the physical prompt length.
- [ ] Positions start at `N`.
- [ ] Output is coherent for at least N=32 and N=64.
- [ ] Works when `system_retain_ratio=0.0`.

---

## Phase 8 — Build compression test script

Create:

```text
experiments/agentcache_compression/test_compression.py
```

CLI:

```bash
python experiments/agentcache_compression/test_compression.py \
  --model /mnt/g/agentcache/models/Llama-3.2-1B-Instruct \
  --data experiments/agentcache_compression/data/python_agent_eval.jsonl \
  --system-prompt experiments/agentcache_compression/prompts/python_agent_system.txt \
  --centroid-k experiments/agentcache_compression/centroids/N64_K.npy \
  --centroid-v experiments/agentcache_compression/centroids/N64_V.npy \
  --synthetic-len 64 \
  --system-retain-ratio 0.0 \
  --layout compression \
  --out experiments/agentcache_compression/results/N64_sys0.jsonl
```

For each task, record:

```json
{
  "id": "...",
  "mode": "synthetic_compression",
  "N": 64,
  "system_retain_ratio": 0.0,
  "physical_prompt_tokens": 123,
  "synthetic_len": 64,
  "scheduled_prefill_tokens": 123,
  "ttft_s": 0.031,
  "output": "...",
  "checks": {
    "task_check_pass": true,
    "ends_with_goodbye": true,
    "coherent": true
  }
}
```

Also support modes:

```text
--mode full_system_no_synthetic
--mode short_system_no_synthetic
--mode synthetic_compression
--mode synthetic_replacement
```

Acceptance criteria:

- [ ] Can run all eval tasks.
- [ ] Saves JSONL.
- [ ] Logs prompt length, synthetic length, scheduled tokens, TTFT, output checks.

---

## Phase 9 — Implement dynamic budget selection

Create:

```text
experiments/agentcache_compression/budget_selector.py
```

Initial rule:

```python
def choose_budget(
    full_system_tokens: int,
    retained_system_tokens: int,
    physical_prompt_tokens: int,
    available_budgets=(256, 128, 64, 32),
    min_physical_prompt_tokens=32,
    max_model_len=8192,
):
    removed_system_tokens = full_system_tokens - retained_system_tokens

    for n in available_budgets:
        # Conservative first prototype:
        # do not inject more synthetic tokens than the number of removed system tokens.
        if n > removed_system_tokens:
            continue

        # Make sure the physical prompt is not tiny.
        if physical_prompt_tokens < min_physical_prompt_tokens:
            continue

        # Make sure virtual prefix + physical prompt fits.
        if n + physical_prompt_tokens >= max_model_len:
            continue

        return n

    return 0
```

Later latency-aware rule:

```python
def choose_budget_latency_aware(...):
    # Choose the largest valid N whose measured seed cost is lower than expected prefill savings.
    pass
```

Acceptance criteria:

- [ ] No semantic query classification.
- [ ] Budget only depends on token accounting and measured latency table.
- [ ] Outputs one of `{0, 32, 64, 128, 256}`.

---

## Phase 10 — Run the main experiment matrix

Matrix:

```text
System retain ratio:
1.00, 0.50, 0.25, 0.00

Synthetic budget:
0, 32, 64, 128, 256, dynamic

Modes:
full_system_no_synthetic
short_system_no_synthetic
synthetic_compression
synthetic_replacement
```

Minimum run:

```text
A. Full system, no synthetic
B. No system, no synthetic
C. No system + N32 synthetic
D. No system + N64 synthetic
E. No system + N128 synthetic
F. No system + N256 synthetic
G. Dynamic synthetic
```

Commands should write results to:

```text
experiments/agentcache_compression/results/
```

---

## Phase 11 — Evaluation metrics

Implement:

```text
experiments/agentcache_compression/evaluate_results.py
```

Metrics:

### Latency

- [ ] Mean TTFT
- [ ] Median TTFT
- [ ] p90 TTFT
- [ ] Speedup vs full system
- [ ] Speedup vs short/no system baseline

### Basic quality

- [ ] Coherence heuristic
- [ ] Task-specific check pass rate
- [ ] Ends-with-GOODBYE pass rate
- [ ] Python-agent behavior score

### Agent behavior score

Simple heuristic initially:

```text
+1 mentions or provides Python code when appropriate
+1 uses idiomatic Python construct relevant to task
+1 gives brief explanation
+1 includes test/example when appropriate
+1 avoids irrelevant generic prose
```

### Persona metrics, later

Use the persona paper metrics when persona traces are added:

- [ ] Task success
- [ ] Cumulative satisfaction
- [ ] Worst-case satisfaction
- [ ] Satisfaction rate
- [ ] Abandonment / human transfer rate

---

## Phase 12 — Exact KV / prefix-cache baseline

This is a baseline, not required before the first synthetic prototype.

Options:

### Option A — vLLM automatic prefix caching baseline

Use the full system prompt and enable vLLM prefix caching:

```python
LLM(..., enable_prefix_caching=True)
```

Run repeated requests with the same system prompt.

Measure:
- first request TTFT
- subsequent request TTFT

This is the easiest exact-cache baseline.

### Option B — explicit exact system K/V export

Implement:

```text
experiments/agentcache_compression/export_exact_system_kv.py
```

Goal:
- Compute real K/V for the system prompt.
- Save as:
  ```text
  exact_sys_K.npy
  exact_sys_V.npy
  ```
- Use existing `VLLM_EXACT_SYS_K_PATH` / `VLLM_EXACT_SYS_V_PATH` path if compatible.

Acceptance criteria:
- [ ] Exact KV baseline preserves behavior almost exactly.
- [ ] Exact KV baseline is faster than cold after initial cache construction.
- [ ] Synthetic KV is compared honestly against this baseline.

---

## Phase 13 — Plots and result table

Create:

```text
experiments/agentcache_compression/plot_results.py
```

Generate:

- [ ] TTFT vs N
- [ ] Task pass rate vs N
- [ ] GOODBYE-rule pass rate vs N
- [ ] Quality-latency Pareto plot
- [ ] Dynamic budget vs fixed budget comparison

Save:

```text
experiments/agentcache_compression/results/summary.csv
experiments/agentcache_compression/results/ttft_vs_quality.png
experiments/agentcache_compression/results/budget_breakdown.png
```

Desired paper-style result:

```text
Fixed N=256 may be fastest but brittle.
Fixed N=32/64 is safer but smaller latency gain.
Dynamic budget selection gives the best latency-quality tradeoff without query classification.
```

---

## Phase 14 — Key hypotheses to test

### H1: Synthetic KV preserves soft agent behavior better than hard lexical rules

Expected:
- Python-agent behavior may remain decent.
- `GOODBYE` rule may degrade when full system prompt is removed.

### H2: Training layout matters

Expected:
- Prefix trained with full system still present may not replace the system prompt.
- Prefix trained in compression layout should perform better when system prompt is removed.

### H3: Budget matters

Expected:
- Larger N improves latency only if seed cost is less than prefill savings.
- Larger N can hurt quality if it is not trained/evaluated in the correct compression layout.

### H4: Dynamic budget is better than fixed budget

Expected:
- Fixed N is fragile across prompt lengths.
- Prompt-length-based dynamic N should produce a better Pareto curve.

---

## Phase 15 — Failure modes to log explicitly

For every failed output, classify the failure:

```text
1. Garbled output
2. Generic assistant, not Python agent
3. Python-relevant but wrong task
4. Correct task, missing hard system rule
5. Correct task, poor formatting
6. Skipped/ignored user details
7. Repetition / degenerate text
8. Latency overhead greater than prefill savings
```

Save failure examples:

```text
experiments/agentcache_compression/results/failures.md
```

---

## Phase 16 — Done criteria for prototype

The prototype is successful if it can produce a table like:

```text
Mode                         TTFT    Task Pass   GOODBYE Pass   Notes
Full system, no synthetic     ...
No system, no synthetic       ...
No system + Synth32           ...
No system + Synth64           ...
No system + Synth128          ...
No system + Synth256          ...
Dynamic Synth                 ...
Exact prefix cache            ...
```

Minimum successful finding:

```text
Synthetic + user-only is more Python-agent-like than user-only baseline.
```

Strong successful finding:

```text
Dynamic synthetic KV preserves most task behavior while reducing TTFT compared with full system prompt.
```

Best paper finding:

```text
Persona/task-distribution-trained AgentCache improves latency-quality Pareto frontier over fixed synthetic budgets and generic task-only training, without query classification.
```

---

## Notes for the coding/planning agent

1. Do not silently compare replacement mode against compression mode. Log the layout explicitly.
2. Do not slice a larger trained prefix and call it a smaller prefix. Train separate N values unless implementing explicit nested-prefix training.
3. Do not rely on the old `coherent ✓` heuristic alone. Add task checks.
4. Do not skip user tokens in the `system_retention=0` experiment.
5. Do not implement query classification for this prototype.
6. Always print:
   ```text
   physical_prompt_tokens
   synthetic_len
   scheduled_prefill_tokens
   positions_minmax
   expected_start
   layout
   ```
7. Treat exact KV caching as the quality-preserving systems baseline.
8. Treat synthetic KV as the learned compression / approximation method.
