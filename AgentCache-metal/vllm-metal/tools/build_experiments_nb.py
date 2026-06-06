#!/usr/bin/env python3
"""Build AgentCache_Metal_Experiments.ipynb (run with a python that has nbformat)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# AgentCache on vLLM-Metal — Experiments

Faithful Apple-Silicon (MLX/Metal) port of AgentCache centroid injection: a trained
PEFT prefix is materialized to raw K/V and seeded into vLLM-Metal's paged KV cache so
the scheduler **skips prefill** for the synthetic system-prefix.

**How this notebook runs:** the kernel only does orchestration + plotting. All heavy
model work is shelled out to the vLLM-Metal **venv** (`subprocess`) because vLLM's
engine spawns a subprocess that doesn't import cleanly inside a notebook kernel.

Each experiment shows **cached results** (already measured) immediately, and has a
`RUN_LIVE = True` switch to regenerate from scratch.
""")

code('''import json, subprocess, os
from pathlib import Path

REPO    = Path("/Users/danyalkhan/Documents/AgentCache-metal/vllm-metal")
VENV_PY = REPO / ".venv-vllm-metal/bin/python"
TOOLS   = REPO / "tools"
RESULTS = REPO / "results"; RESULTS.mkdir(exist_ok=True)
CENTROIDS = REPO / "centroids"; CENTROIDS.mkdir(exist_ok=True)
AC = Path("/Users/danyalkhan/Documents/AgentCache/AgentCache/agentcache_compression")

# Real Llama-3.2-1B N=128 centroid (exported from the CUDA pipeline) + prompts/eval.
LLAMA_MODEL = "mlx-community/Llama-3.2-1B-Instruct-bf16"
LLAMA_CK = AC / "centroids/N128_2000_K.npy"
LLAMA_CV = AC / "centroids/N128_2000_V.npy"
EVAL = AC / "data/python_agent_eval.jsonl"
PROMPTS = {n: AC / f"prompts/{n}_python_agent_system.txt" for n in (200, 500, 1000, 2000)}

_CENTROID_ENV = ("VLLM_CENTROID_SCHEDULER","VLLM_CENTROID_K_PATH",
                 "VLLM_CENTROID_V_PATH","VLLM_CENTROID_SYS_TOKENS")

def sh(args, inject=None, timeout=2400):
    """Run `VENV_PY <args>`. Clean of centroid env unless `inject` dict provided."""
    env = {k: v for k, v in os.environ.items() if k not in _CENTROID_ENV}
    env["PYTHONPATH"] = str(REPO)
    if inject:
        env.update({k: str(v) for k, v in inject.items()})
    p = subprocess.run([str(VENV_PY), *map(str, args)], cwd=str(REPO), env=env,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr

print("venv python:", VENV_PY.exists(), "| Llama N=128 centroid:", LLAMA_CK.exists())
''')

md("""## CUDA reference (the target)

From the CUDA port's `HANDOFF.md`, Llama-3.2-1B, N=128, ~1000-token context:

| Mode | TTFT | Speedup |
|---|---|---|
| Cold (full prefill) | 47.8 ms | — |
| Synthetic inject | 17.0 ms | **2.8×** |

The absolute numbers are CUDA-specific (paged kernels + scheduler); on MLX the *ratio*
differs but the **shape** (cold grows with context, inject stays flat) is what transfers.
""")

md("""## Experiment 1 — RoPE parity gate (Milestone 1)

The #1 correctness risk is the offline RoPE applied to the centroid K. This verifies
(independent of any forward) that the injector rotates the prefix using the model's own
`attn.rope` with correct offset/layout semantics.
""")
code('''rc, out = sh([TOOLS/"centroid_rope_parity.py", "--model", LLAMA_MODEL,
              "--centroid-k", LLAMA_CK, "--centroid-v", LLAMA_CV,
              "--sys-tokens", "0", "--n", "128"])
print("\\n".join(l for l in out.splitlines() if l.startswith(("[ok]","[parity]","[FAIL]","[PASS]"))))''')

md("""## Experiment 2 — Cold vs Inject (same prompts + eval as the CUDA port)

`cold` = full system prompt in the physical prompt. `inject` = compression
(`[pad]*N + user`, no system text; centroid injected). 25-task eval
(`python_agent_eval.jsonl`): TTFT proxy (`max_tokens=1` wall-clock), task-keyword pass,
coherence.
""")
code('''# Cached measured results (Qwen2.5-0.5B N=64, Llama-3.2-1B N=128).
CACHED = {
  "Qwen2.5-0.5B (N=64)":  {"cold_ttft":{200:13.5,500:15.8,1000:18.0,2000:24.6},
                            "inject_ttft":14.9,"inject_tok":103,
                            "cold_acc":0.68,"inject_acc":0.56,"coh":1.0},
  "Llama-3.2-1B (N=128)": {"cold_ttft":{200:22.9,500:23.5,1000:25.1,2000:29.4},
                            "inject_ttft":23.2,"inject_tok":173,
                            "cold_acc":0.80,"inject_acc":0.64,"coh":1.0},
}
try:
    import pandas as pd
    rows = []
    for m, d in CACHED.items():
        c2000 = d["cold_ttft"][2000]
        rows.append({"model": m, "cold@2000 (ms)": c2000, "inject (ms)": d["inject_ttft"],
                     "TTFT speedup": round(c2000/d["inject_ttft"], 2),
                     "cold task-pass": f"{d['cold_acc']:.0%}", "inject task-pass": f"{d['inject_acc']:.0%}",
                     "coherence": f"{d['coh']:.0%}"})
    display(pd.DataFrame(rows).set_index("model"))
except ImportError:
    for m, d in CACHED.items():
        print(m, d)''')

md("""## Experiment 3 — Long-context TTFT scaling (the headline)

Inject TTFT is **flat** (`N + user` physical tokens, independent of the compressed prompt
length). Cold TTFT grows linearly with context. So the speedup grows without bound — and
on the **same 1B model** crosses the CUDA 2.8× around ~14k tokens.
""")
code('''RUN_LIVE = False  # set True to re-measure the cold sweep (~2 min)

LLAMA_LONGCTX = {1000:24.6, 2000:27.3, 4000:34.0, 8000:46.6, 12000:59.1, 16000:70.9}
INJECT_FLAT = 23.2

if RUN_LIVE:
    rc, out = sh([TOOLS/"centroid_longctx_ttft.py", "--model", LLAMA_MODEL,
                  "--base-prompt", PROMPTS[2000], "--inject-ms", INJECT_FLAT,
                  "--lengths", "1000,2000,4000,8000,12000,16000"])
    LLAMA_LONGCTX = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4 and p[0].isdigit():
            LLAMA_LONGCTX[int(p[0])] = float(p[1])
    print(out)

xs = sorted(LLAMA_LONGCTX); cold = [LLAMA_LONGCTX[x] for x in xs]
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(xs, cold, "o-", label="cold (full prefill)")
    ax.axhline(INJECT_FLAT, ls="--", color="green", label=f"inject (flat, ~{INJECT_FLAT} ms)")
    ax.axhline(INJECT_FLAT*2.8, ls=":", color="red", label="2.8x cold-equiv (CUDA target)")
    ax.set_xlabel("system-prompt context (tokens)"); ax.set_ylabel("TTFT (ms)")
    ax.set_title("MLX Llama-3.2-1B: cold TTFT grows, inject stays flat")
    ax.legend(); ax.grid(alpha=.3); plt.show()
    print("speedup vs context:", {x: round(LLAMA_LONGCTX[x]/INJECT_FLAT,2) for x in xs})
except ImportError:
    for x in xs: print(f"{x:>6} tok  cold {LLAMA_LONGCTX[x]:.1f} ms  speedup {LLAMA_LONGCTX[x]/INJECT_FLAT:.2f}x")''')

md("""## Experiment 4 — Change the model size  ⟵ *iterate here*

`bench_model()` runs the cold long-context sweep **and** an inject-TTFT reference for any
model. For a model with no trained centroid, a **dummy** centroid of the right shape is
generated automatically (TTFT is value-independent; quality would need a real centroid).

> Bigger models raise per-token prefill cost (more to save) but also inflate the fixed
> first-token-decode floor in the `max_tokens=1` proxy — so the lever is positive but
> mixed. Longer context is the cleaner lever (Experiment 3).
""")
code('''def bench_model(model, lengths="1000,2000,4000,8000,12000,16000", n=128,
                centroid_k=None, centroid_v=None, tag=None):
    """Return {"model":..., "ctx":{tok:cold_ms}, "inject_ms":...}. Uses a dummy
    centroid (right shape) if real ones not given. Heavy: downloads model if new."""
    tag = tag or model.split("/")[-1]
    ck = centroid_k or (CENTROIDS / f"dummy_{tag}_K.npy")
    cv = centroid_v or (CENTROIDS / f"dummy_{tag}_V.npy")
    if centroid_k is None:
        rc, out = sh([TOOLS/"make_dummy_centroid.py", "--model", model,
                      "--n", n, "--out-k", ck, "--out-v", cv]); print(out.strip())
    # inject TTFT (fast, accuracy skipped)
    inj_env = {"VLLM_CENTROID_SCHEDULER":"1","VLLM_CENTROID_SYS_TOKENS":"0",
               "VLLM_CENTROID_K_PATH":ck,"VLLM_CENTROID_V_PATH":cv}
    rc, out = sh([TOOLS/"centroid_benchmark.py","--mode","inject","--n",n,"--model",model,
                  "--eval-data",EVAL,"--skip-accuracy","--out",RESULTS/f"{tag}_inj_ttft.json"],
                 inject=inj_env)
    inj_ms = json.loads((RESULTS/f"{tag}_inj_ttft.json").read_text())["ttft"][0]["ttft_ms"]
    # cold long-context sweep
    rc, out = sh([TOOLS/"centroid_longctx_ttft.py","--model",model,
                  "--base-prompt",PROMPTS[2000],"--inject-ms",inj_ms,"--lengths",lengths])
    ctx = {}
    for line in out.splitlines():
        p = line.split()
        if len(p)==4 and p[0].isdigit(): ctx[int(p[0])] = float(p[1])
    return {"model":tag, "ctx":ctx, "inject_ms":inj_ms}

print("Defined bench_model(). Example calls in the next cell.")''')

code('''# Compare model sizes. 1B uses the REAL centroid; add 3B (dummy) to see the curve shift.
# WARNING: a new model downloads weights (3B ~6GB) and each call takes a few minutes.

RESULTS_BY_MODEL = []
# 1B with the real N=128 centroid:
RESULTS_BY_MODEL.append(bench_model(LLAMA_MODEL, centroid_k=LLAMA_CK, centroid_v=LLAMA_CV, tag="Llama-1B"))
# Uncomment to add a larger model (dummy centroid, TTFT-only):
# RESULTS_BY_MODEL.append(bench_model("mlx-community/Llama-3.2-3B-Instruct-bf16", tag="Llama-3B"))

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    for r in RESULTS_BY_MODEL:
        xs = sorted(r["ctx"]); ax.plot(xs, [r["ctx"][x] for x in xs], "o-", label=f"{r['model']} cold")
        ax.axhline(r["inject_ms"], ls="--", alpha=.6, label=f"{r['model']} inject (~{r['inject_ms']:.0f} ms)")
    ax.set_xlabel("context (tokens)"); ax.set_ylabel("TTFT (ms)")
    ax.set_title("Cold-vs-inject TTFT by model size"); ax.legend(); ax.grid(alpha=.3); plt.show()
except ImportError:
    for r in RESULTS_BY_MODEL: print(r)''')

md("""### Notes & caveats
- **TTFT is a `max_tokens=1` proxy** (prefill + 1 decode + engine overhead → ~22 ms floor on
  1B). A streaming first-chunk measurement would lower the floor and lift the whole speedup
  curve toward the CUDA methodology.
- **Quality vs TTFT are separate axes.** Dummy centroids give correct *TTFT* for any model;
  *accuracy* requires a real trained+exported centroid (`transpose_tensors.py`).
- **Absolute ms / the 2.8× ratio are backend-bound.** The faithful, transferable result is the
  *shape*: cold grows with context, inject stays flat.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "AgentCache Metal (vllm-metal venv)",
                                "language": "python", "name": "agentcache-metal"}
out = "/Users/danyalkhan/Documents/AgentCache-metal/vllm-metal/AgentCache_Metal_Experiments.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
