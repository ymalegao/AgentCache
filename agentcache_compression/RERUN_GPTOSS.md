# Rerunning the GPT-OSS-20B Multi-Turn Benchmark (Windows / 16GB GPU)

**Why this rerun is needed:** the original GPT-OSS-20B run in
`results/gptmulti_turn_benchmark.jsonl` is corrupted for quality evaluation.
GPT-OSS emits Harmony channel text (`analysis…assistantfinal…`), and the old
benchmark fed the raw multi-channel text back as assistant history, which
progressively corrupted the conversation and degenerated later turns in **all
modes** (cold turns 6–10, warm_apc 3–10, synthetic N64 5–10, …). Full analysis:
`llm_judge_design.md` §3.3.

**The fix is already committed** (`b7b206b2`, "Fixed bug in multi-turn-benchmark
for gpt-20b oss"): `multi_turn_benchmark.py` now strips every GPT-OSS response
to its final channel (`strip_to_final_channel()`) before appending it to history
and before writing it to the results JSONL. Qwen/Llama runs are unaffected
(no-op without channel markers). You only need to **pull and rerun** — no code
changes on your side.

---

## 0. Environment (Windows)

vLLM does not run natively on Windows — everything below runs inside **WSL2**
(Ubuntu), which is how the repo was originally set up (note the `/mnt/g/...`
model paths in `HANDOFF.md`). From PowerShell: `wsl`, then:

```bash
cd /path/to/AgentCache        # your WSL clone, e.g. /mnt/g/agentcache
git pull                       # must include commit b7b206b2
source venv/bin/activate
```

Requirements check:

- **GPU:** 16GB (RTX 4080 Super) is the minimum for gpt-oss-20b. The model's
  MXFP4 weights are ~12.5GB, leaving ~2–3GB for KV cache — the flags below are
  sized for this. Close anything else using VRAM on the Windows side (browsers,
  the desktop compositor eats real memory at 4K).
- **vLLM version:** MXFP4 on Ada (SM 8.9) needs a recent vLLM (>= ~0.10.1).
  `python -c "import vllm; print(vllm.__version__)"`.
- **Model:** `./get_model.sh openai/gpt-oss-20b` if not already under `models/`.

---

## 1. Phase 1 — run now: `cold` and `warm_apc` (no centroid needed)

These two modes don't use centroid injection, were the worst-degenerated in the
old run, and `cold` is the reference for the LLM-judge evaluation — so this
phase unblocks all the quality work.

```bash
python agentcache_compression/multi_turn_benchmark.py \
    --mode cold \
    --model models/gpt-oss-20b \
    --conversation-file agentcache_compression/conversations/csv_cli.json \
    --max-tokens 2048 \
    --gpu-mem 0.95 \
    --max-model-len 16384 \
    --out agentcache_compression/results/gptmulti_turn_benchmark_v2.jsonl

python agentcache_compression/multi_turn_benchmark.py \
    --mode warm_apc \
    --model models/gpt-oss-20b \
    --conversation-file agentcache_compression/conversations/csv_cli.json \
    --max-tokens 2048 \
    --gpu-mem 0.95 \
    --max-model-len 16384 \
    --out agentcache_compression/results/gptmulti_turn_benchmark_v2.jsonl
```

Flag notes — do not change these casually:

- **`--out` points at a NEW file (`_v2`).** The script *appends*; reusing the
  old filename would mix broken and clean records.
- **`--gpu-mem 0.95`** — the script's default (0.6) budgets only ~9.6GB, less
  than the model weights, and will OOM at server startup on a 16GB card.
- **`--max-model-len 16384`** — caps KV pre-allocation. The 10-turn CSV-CLI
  conversation peaks around ~10K tokens, so 16K is safe. If startup still OOMs:
  drop to `--max-model-len 8192`; if it *still* OOMs, add `--enforce-eager` to
  the `vllm serve` args in `build_server_cmd()`.
- The script launches and tears down `vllm serve` itself — run one mode at a
  time, expect a few minutes of model-load per invocation.

## 2. Phase 2 — blocked: `synthetic` N=64/128/256 (needs GPT-OSS centroids)

⚠️ **Do NOT use `run_multi_turn_pipeline.py` for GPT-OSS.** It auto-selects
`centroids/N{N}_2000_*.npy`, and the centroids committed in this repo are
**Llama-3.2-1B tensors** (shape `[16, N, 512]` — 16 layers). GPT-OSS-20B has 24
layers; injecting these would shape-mismatch or corrupt the KV cache. The
GPT-OSS centroids from the original run were never committed — recover them
from the Blackwell machine (or re-export there with `transpose_tensors.py`;
training/exporting a 20B adapter does not fit on a 16GB card).

Once you have the GPT-OSS `.npy` files:

```bash
# synthetic mode needs the centroid-patched vLLM: ./install.sh on a fresh clone
python agentcache_compression/multi_turn_benchmark.py \
    --mode synthetic --synthetic-len 64 \
    --centroid-k /path/to/gptoss_N64_K.npy \
    --centroid-v /path/to/gptoss_N64_V.npy \
    --model models/gpt-oss-20b \
    --conversation-file agentcache_compression/conversations/csv_cli.json \
    --max-tokens 2048 --gpu-mem 0.95 --max-model-len 16384 \
    --out agentcache_compression/results/gptmulti_turn_benchmark_v2.jsonl
# repeat with --synthetic-len 128 / 256 and the matching centroid paths
```

## 3. Verify the output after each mode

```bash
python3 - <<'EOF'
import json
recs = [json.loads(l) for l in open('agentcache_compression/results/gptmulti_turn_benchmark_v2.jsonl')]
ok = True
for r in sorted(recs, key=lambda x: (x['mode'], x['N'], x['turn'])):
    bad = sum(1 for c in r['response'][:1500] if c in '…　​' or ord(c) > 0x2500)
    marker = 'assistantfinal' in r['response'] or r['response'].startswith('final')
    flag = 'OK ' if bad == 0 and not marker else 'BAD'
    if flag == 'BAD': ok = False
    print(flag, r['mode'], 'N' + str(r['N']), 't' + str(r['turn']),
          'badchars:', bad, 'leftover-marker:', marker, repr(r['response'][:60]))
print('\nALL CLEAN' if ok else '\nPROBLEMS FOUND — see BAD rows above')
EOF
```

Every row must show `badchars: 0` and `leftover-marker: False` with a sensible
answer prefix, **through all 10 turns** (the old bug only showed up from turn
3–6 onward, so don't stop checking at turn 2). Then:

```bash
python agentcache_compression/analyze_multi_turn.py \
    agentcache_compression/results/gptmulti_turn_benchmark_v2.jsonl
```

## 4. What the results are used for (and one caveat)

- The clean `cold` transcripts are the **reference side of the LLM-judge
  evaluation** (`llm_judge_design.md`); synthetic transcripts from Phase 2 are
  the candidate side.
- Stripping the analysis channel from history changes prompt content/length vs
  the original run, so the new TTFT numbers **will differ** from the old ones —
  expected, and more correct. But latency on this GPU is not comparable to the
  paper's numbers: the paper's 1.80× GPT-20B speedup must be re-derived on the
  original Blackwell machine. This rerun is for **response quality**, not
  timing.
- When done, commit `results/gptmulti_turn_benchmark_v2.jsonl` and leave the old
  file in place (it's referenced by the corruption analysis) — the judge and
  any new paper numbers should read only from `_v2`.
