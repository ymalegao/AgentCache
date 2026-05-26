"""
sweep.py — drive the three AgentCache-on-Mac experiment axes:

    model size   x   precision/quantization   x   system-prompt length

For each (model, dtype, system-prompt) combination it runs hf_eval.py twice
(cold + inject) into a dedicated results file, then prints a compact TTFT matrix
(cold vs inject + speedup) across all combinations.

Prerequisite: each model already has a trained adapter at
    <adapter-dir>/<model_basename>_N<tokens>
(produce them with run_mac_pipeline.py, once per model). Training is NOT part of the
sweep — the sweep is the eval-axis explorer.

Example:
  python agentcache_mac/sweep.py \
    --models meta-llama/Llama-3.2-1B-Instruct,meta-llama/Llama-3.2-3B-Instruct \
    --dtypes fp16,int8 \
    --prompts 200,500,1000,2000 \
    --tokens 64

ACKNOWLEDGMENT: numbers here are HF/MPS, not vLLM. The headline signal is the *shape*:
cold TTFT grows with system-prompt length while inject TTFT stays roughly flat. See README.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAC = REPO / "agentcache_mac"
CMP = REPO / "agentcache_compression"
EVAL = MAC / "hf_eval.py"


def run_eval(model, adapter, dtype, system_prompt, eval_data, tokens, out, limit):
    common = [
        sys.executable, str(EVAL),
        "--model", model,
        "--data", eval_data,
        "--system-prompt", system_prompt,
        "--synthetic-len", str(tokens),
        "--dtype", dtype,
        "--out", out,
    ]
    if limit:
        common += ["--limit", str(limit)]
    subprocess.run(common + ["--mode", "cold"], check=True)
    subprocess.run(common + ["--mode", "inject", "--adapter", adapter], check=True)


def mean_ttft(records, mode):
    vals = [r["ttft_mean_s"] for r in records if r["mode"] == mode]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="Comma-separated model paths/ids.")
    ap.add_argument("--dtypes", default="fp16", help="Comma-separated: fp16,bf16,int8,int4.")
    ap.add_argument("--prompts", default="200,500,1000,2000",
                    help="Comma-separated system-prompt length labels (resolve to "
                         "agentcache_compression/prompts/<label>_python_agent_system.txt).")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--adapter-dir", default=str(MAC / "adapters"))
    ap.add_argument("--eval-data", default=str(CMP / "data" / "python_agent_eval.jsonl"))
    ap.add_argument("--out-dir", default=str(MAC / "results" / "sweep"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    plabels = [p.strip() for p in args.prompts.split(",") if p.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []  # (model_tag, dtype, plabel, cold, inject, speedup)

    for model in models:
        model_tag = Path(model.rstrip("/")).name
        adapter = str(Path(args.adapter_dir) / f"{model_tag}_N{args.tokens}")
        if not Path(adapter).exists():
            print(f"WARNING: adapter {adapter} missing — train it with run_mac_pipeline.py. Skipping {model_tag}.")
            continue
        for dtype in dtypes:
            for plabel in plabels:
                system_prompt = str(CMP / "prompts" / f"{plabel}_python_agent_system.txt")
                if not Path(system_prompt).exists():
                    print(f"WARNING: prompt {system_prompt} missing. Skipping.")
                    continue
                out = str(out_dir / f"{model_tag}_N{args.tokens}_{dtype}_{plabel}.jsonl")
                Path(out).unlink(missing_ok=True)  # fresh file per combo
                print(f"\n>>> {model_tag}  dtype={dtype}  prompt={plabel}")
                run_eval(model, adapter, dtype, system_prompt, args.eval_data,
                         args.tokens, out, args.limit)
                recs = [json.loads(l) for l in Path(out).read_text().splitlines() if l.strip()]
                cold = mean_ttft(recs, "cold")
                inj = mean_ttft(recs, "inject")
                speed = cold / inj if inj and inj == inj else float("nan")
                rows.append((model_tag, dtype, plabel, cold, inj, speed))

    # Matrix
    print("\n" + "=" * 78)
    print("  TTFT SWEEP (HF/MPS) — cold vs inject, mean seconds")
    print("=" * 78)
    hdr = f"  {'model':<26} {'dtype':>6} {'prompt':>7} {'cold_s':>9} {'inject_s':>9} {'speedup':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for mt, dt, pl, cold, inj, sp in rows:
        print(f"  {mt:<26} {dt:>6} {pl:>7} {cold:>9.4f} {inj:>9.4f} {sp:>7.2f}x")
    print("\nReminder: HF/MPS numbers, not vLLM. Watch the shape — cold should rise with")
    print("prompt length while inject stays ~flat. See README 'Acknowledgment'.\n")


if __name__ == "__main__":
    main()
