"""
AgentCache prefix-compression pipeline.

Usage:
    python run_pipeline.py --model <path_or_hf_id> --tokens <N>

Steps:
    1. Train   — prefix adapter via PEFT
    2. Transpose — export adapter weights to .npy centroid tensors
    3. Test    — evaluate synthetic KV injection via vLLM

Prerequisites:
    source venv/bin/activate   (install.sh must have been run first)
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent  # agentcache/
EXP = REPO / "agentcache_compression"

TRAIN_SCRIPT = EXP / "train_prefix_compression.py"
TRANSPOSE_SCRIPT = EXP / "transpose_tensors.py"
TEST_SCRIPT = EXP / "test_compression.py"

TEST_MODES = ["cold_no_synthetic", "warm_apc", "synthetic_compression"]


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
        description="Train prefix adapter, export centroids, and test injection."
    )
    parser.add_argument("--model", required=True, help="Base model path or HuggingFace ID")
    parser.add_argument("--tokens", type=int, required=True, metavar="N",
                        help="Number of virtual tokens (e.g. 64, 128, 256)")

    # Input overrides (all have sensible defaults)
    parser.add_argument("--data", default=str(EXP / "data" / "python_agent_train.jsonl"),
                        help="Training JSONL (default: agentcache_compression/data/python_agent_train.jsonl)")
    parser.add_argument("--eval-data", default=str(EXP / "data" / "python_agent_eval.jsonl"),
                        help="Eval JSONL (default: agentcache_compression/data/python_agent_eval.jsonl)")
    parser.add_argument("--system-prompt", default=str(EXP / "prompts" / "python_agent_system.txt"),
                        help="System prompt .txt file")

    # Output overrides
    parser.add_argument("--adapter-out", default=None,
                        help="Adapter save dir (default: agentcache_compression/adapters/N{tokens}/)")
    parser.add_argument("--centroid-dir", default=None,
                        help="Centroid output dir (default: agentcache_compression/centroids/)")
    parser.add_argument("--results-out", default=None,
                        help="Results JSONL path (default: agentcache_compression/results/N{tokens}_comparison.jsonl)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=4)

    # Test modes
    parser.add_argument(
        "--test-modes", default="synthetic_compression",
        help="Comma-separated test modes, or 'all'. "
             "Choices: cold_no_synthetic, warm_apc, synthetic_compression. "
             "(default: synthetic_compression)"
    )

    # Resume flags
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing adapter)")
    parser.add_argument("--skip-transpose", action="store_true",
                        help="Skip transpose (use existing centroids)")

    args = parser.parse_args()
    N = args.tokens

    # Resolve output paths (under agentcache_compression/ by default)
    adapter_out = Path(args.adapter_out) if args.adapter_out else EXP / "adapters" / f"N{N}"
    centroid_dir = Path(args.centroid_dir) if args.centroid_dir else EXP / "centroids"
    centroid_k = centroid_dir / f"N{N}_K.npy"
    centroid_v = centroid_dir / f"N{N}_V.npy"
    results_out = Path(args.results_out) if args.results_out else EXP / "results" / f"N{N}_comparison.jsonl"

    # Resolve test modes
    if args.test_modes == "all":
        modes = TEST_MODES
    else:
        modes = [m.strip() for m in args.test_modes.split(",")]
        invalid = [m for m in modes if m not in TEST_MODES]
        if invalid:
            parser.error(f"Unknown test mode(s): {invalid}. Choose from: {TEST_MODES} or 'all'")

    # Create output dirs
    adapter_out.mkdir(parents=True, exist_ok=True)
    centroid_dir.mkdir(parents=True, exist_ok=True)
    results_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nPipeline: N={N}, model={args.model}")
    print(f"  adapter  -> {adapter_out}")
    print(f"  centroid -> {centroid_dir}")
    print(f"  results  -> {results_out}")

    # Step 1: Train
    if not args.skip_train:
        run([
            sys.executable, str(TRAIN_SCRIPT),
            "--model", args.model,
            "--data", args.data,
            "--system-prompt", args.system_prompt,
            "--output", str(adapter_out),
            "--num-virtual-tokens", str(N),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--batch-size", str(args.batch_size),
        ], f"1/3 Train prefix adapter (N={N})")
    else:
        print("\nSkipping step 1/3 (train) — using existing adapter.")

    # Step 2: Transpose
    if not args.skip_transpose:
        run([
            sys.executable, str(TRANSPOSE_SCRIPT),
            "--adapter", str(adapter_out),
            "--out-k", str(centroid_k),
            "--out-v", str(centroid_v),
            "--sys-tokens", "0",
        ], "2/3 Transpose adapter → centroids")
    else:
        print("\nSkipping step 2/3 (transpose) — using existing centroids.")

    # Step 3: Test (one subprocess per mode)
    for i, mode in enumerate(modes):
        run([
            sys.executable, str(TEST_SCRIPT),
            "--model", args.model,
            "--data", args.eval_data,
            "--system-prompt", args.system_prompt,
            "--centroid-k", str(centroid_k),
            "--centroid-v", str(centroid_v),
            "--synthetic-len", str(N),
            "--mode", mode,
            "--out", str(results_out),
        ], f"3/3 Test ({mode})" + (f" [{i+1}/{len(modes)}]" if len(modes) > 1 else ""))

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"Results: {results_out}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
