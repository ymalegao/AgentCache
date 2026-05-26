"""
AgentCache multi-turn cache benchmark pipeline.

Usage:
    python run_multi_turn_pipeline.py --model <path> [options]

Runs all benchmark modes as subprocesses, appending to a single results JSONL:
    cold            — baseline, no APC, no centroid
    warm_apc        — APC enabled, no centroid
    synthetic_N64   — centroid injection N=64 + APC
    synthetic_N128  — centroid injection N=128 + APC
    synthetic_N256  — skipped if centroid files are missing

Prerequisites:
    source vllm-env/bin/activate
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent  # agentcache/
EXP  = REPO / "agentcache_compression"

MULTI_TURN_SCRIPT = EXP / "multi_turn_benchmark.py"
SYNTHETIC_TOKENS  = [64, 128, 256]


def run(cmd, step):
    import subprocess
    print(f"\n{'=' * 60}")
    print(f"STEP {step}")
    print(f"{'=' * 60}")
    print("Command:", " ".join(str(c) for c in cmd))
    print()
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: step '{step}' exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-turn cache benchmark for all modes."
    )
    parser.add_argument("--model",          required=True)
    parser.add_argument("--system-prompt",  default=str(EXP / "prompts" / "2000_python_agent_system.txt"))
    parser.add_argument("--data",           default=str(EXP / "data"    / "python_agent_eval.jsonl"))
    parser.add_argument("--centroid-dir",   default=str(EXP / "centroids"))
    parser.add_argument("--out",            default=str(EXP / "results" / "multi_turn_benchmark.jsonl"))
    parser.add_argument("--conversation-file", default=None,
                        help="JSON file with a list of user prompts (one coherent conversation). "
                             "When set, --n-conversations and --turns-per-conv are ignored.")
    parser.add_argument("--n-conversations", type=int,   default=5)
    parser.add_argument("--turns-per-conv",  type=int,   default=5)
    parser.add_argument("--max-tokens",      type=int,   default=2048)
    parser.add_argument("--gpu-mem",         type=float, default=0.6)
    args = parser.parse_args()

    centroid_dir = Path(args.centroid_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common = [
        "--model",           args.model,
        "--system-prompt",   args.system_prompt,
        "--data",            args.data,
        "--out",             str(out_path),
        "--max-tokens",      str(args.max_tokens),
        "--gpu-mem",         str(args.gpu_mem),
    ]
    if args.conversation_file:
        common += ["--conversation-file", args.conversation_file]
    else:
        common += [
            "--n-conversations", str(args.n_conversations),
            "--turns-per-conv",  str(args.turns_per_conv),
        ]

    # step = 0

    # step += 1
    # run([
    #     sys.executable, str(MULTI_TURN_SCRIPT),
    #     "--mode", "cold",
    #     "--synthetic-len", "0",
    # ] + common, f"{step} cold")

    # step += 1
    # run([
    #     sys.executable, str(MULTI_TURN_SCRIPT),
    #     "--mode", "warm_apc",
    #     "--synthetic-len", "0",
    # ] + common, f"{step} warm_apc")
    step = 2

    for N in SYNTHETIC_TOKENS:
        k_path = centroid_dir / f"N{N}_2000_K.npy"
        v_path = centroid_dir / f"N{N}_2000_V.npy"
        if not k_path.exists() or not v_path.exists():
            print(f"\nWARNING: Centroid files for N={N} not found ({k_path}). Skipping synthetic_N{N}.")
            continue
        step += 1
        run([
            sys.executable, str(MULTI_TURN_SCRIPT),
            "--mode", "synthetic",
            "--synthetic-len", str(N),
            "--centroid-k", str(k_path),
            "--centroid-v", str(v_path),
        ] + common, f"{step} synthetic_N{N}")

    print(f"\n{'=' * 60}")
    print("MULTI-TURN PIPELINE COMPLETE")
    print(f"Results: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
