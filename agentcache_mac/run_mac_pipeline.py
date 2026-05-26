"""
run_mac_pipeline.py — end-to-end AgentCache pipeline on macOS / MPS (single model, single N).

Steps:
  0. prepare_data   — build data/python_agent_{train,eval}.jsonl (only if missing)
  1. train          — PEFT prefix adapter via MPS (train_prefix_mac.py)
  2. eval cold      — full system+user prompt, base model (hf_eval.py --mode cold)
  3. eval inject    — user-only prompt, prefix injected via PeftModel (hf_eval.py --mode inject)
  4. analyze        — TTFT + speedup + quality tables (analyze_results.py)

This is the Mac analog of run_training_pipeline.py. There is no vLLM; see README.md for
the (important) caveat that absolute numbers are not comparable to the vLLM deployment.

Usage:
  python agentcache_mac/run_mac_pipeline.py \
    --model meta-llama/Llama-3.2-1B-Instruct --tokens 64 \
    --system-prompt agentcache_compression/prompts/2000_python_agent_system.txt
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAC = REPO / "agentcache_mac"
CMP = REPO / "agentcache_compression"

PREPARE = CMP / "prepare_data.py"
TRAIN = MAC / "train_prefix_mac.py"
EVAL = MAC / "hf_eval.py"
ANALYZE = MAC / "analyze_results.py"


def run(cmd, step):
    print(f"\n{'=' * 60}\nSTEP {step}\n{'=' * 60}")
    print("Command:", " ".join(str(c) for c in cmd), "\n")
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"\nERROR: step '{step}' exited with code {r.returncode}")
        sys.exit(r.returncode)


def prompt_label(system_prompt: str) -> str:
    """e.g. .../2000_python_agent_system.txt -> '2000'."""
    stem = Path(system_prompt).stem
    return stem.split("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model path or HF id")
    ap.add_argument("--tokens", type=int, required=True, metavar="N")
    ap.add_argument("--system-prompt", default=str(CMP / "prompts" / "2000_python_agent_system.txt"))
    ap.add_argument("--train-data", default=str(CMP / "data" / "python_agent_train.jsonl"))
    ap.add_argument("--eval-data", default=str(CMP / "data" / "python_agent_eval.jsonl"))
    ap.add_argument("--adapter-out", default=None)
    ap.add_argument("--results-out", default=None)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "int8", "int4"],
                    help="Eval precision. Training always uses fp32/bf16 (see train_prefix_mac.py).")
    ap.add_argument("--train-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Eval only first K tasks (0 = all).")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    N = args.tokens
    model_tag = Path(args.model.rstrip("/")).name
    plabel = prompt_label(args.system_prompt)

    adapter_out = Path(args.adapter_out) if args.adapter_out else MAC / "adapters" / f"{model_tag}_N{N}"
    results_out = Path(args.results_out) if args.results_out else MAC / "results" / f"{model_tag}_N{N}_{plabel}.jsonl"
    adapter_out.mkdir(parents=True, exist_ok=True)
    results_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nMac pipeline: model={args.model}  N={N}  prompt={plabel}  eval_dtype={args.dtype}")
    print(f"  adapter -> {adapter_out}")
    print(f"  results -> {results_out}")

    # Step 0: data (only if missing)
    if not Path(args.train_data).exists() or not Path(args.eval_data).exists():
        run([sys.executable, str(PREPARE)], "0/4 prepare_data")
    else:
        print("\nStep 0/4 prepare_data — data already present, skipping.")

    # Step 1: train
    if not args.skip_train:
        run([
            sys.executable, str(TRAIN),
            "--model", args.model,
            "--data", args.train_data,
            "--system-prompt", args.system_prompt,
            "--output", str(adapter_out),
            "--num-virtual-tokens", str(N),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--batch-size", str(args.batch_size),
            "--dtype", args.train_dtype,
        ], f"1/4 train prefix adapter (N={N})")
    else:
        print("\nStep 1/4 train — skipped (--skip-train).")

    # Steps 2 & 3: eval cold + inject (append to the same results file)
    common = [
        "--model", args.model,
        "--data", args.eval_data,
        "--system-prompt", args.system_prompt,
        "--synthetic-len", str(N),
        "--dtype", args.dtype,
        "--out", str(results_out),
    ]
    if args.limit:
        common += ["--limit", str(args.limit)]

    run([sys.executable, str(EVAL), "--mode", "cold"] + common, "2/4 eval cold")
    run([sys.executable, str(EVAL), "--mode", "inject", "--adapter", str(adapter_out)] + common,
        "3/4 eval inject")

    # Step 4: analyze
    run([sys.executable, str(ANALYZE), str(results_out)], "4/4 analyze")

    print(f"\n{'=' * 60}\nPIPELINE COMPLETE\nResults: {results_out}\n{'=' * 60}")


if __name__ == "__main__":
    main()
