#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optional multi-turn native prefix-cache comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from study_paths import AC_CENTROIDS, AC_COMPRESSION, EVAL_DATA, RESULTS_DIR, ensure_results_dir, prompt_path, python_executable, run_checked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    p.add_argument("--modes", default="cold,warm_apc")
    p.add_argument("--synthetic-len", type=int, default=128)
    p.add_argument("--centroid-k", default=str(AC_CENTROIDS / "N128_2000_K.npy"))
    p.add_argument("--centroid-v", default=str(AC_CENTROIDS / "N128_2000_V.npy"))
    p.add_argument("--n-conversations", type=int, default=2)
    p.add_argument("--turns-per-conv", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", default=str(RESULTS_DIR / "prefix_cache_multiturn.jsonl"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_results_dir()
    out = Path(args.out)
    if out.exists() and args.overwrite:
        out.unlink()
    elif out.exists():
        raise FileExistsError(f"{out} exists; pass --overwrite or choose another --out")

    script = AC_COMPRESSION / "multi_turn_benchmark.py"
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        cmd = [
            python_executable(),
            script,
            "--model",
            args.model,
            "--mode",
            mode,
            "--data",
            EVAL_DATA,
            "--system-prompt",
            prompt_path(2000, "python"),
            "--out",
            out,
            "--n-conversations",
            str(args.n_conversations),
            "--turns-per-conv",
            str(args.turns_per_conv),
            "--max-tokens",
            str(args.max_tokens),
            "--server-port",
            str(args.port),
            "--max-model-len",
            str(args.max_model_len),
        ]
        if mode == "synthetic":
            cmd += [
                "--synthetic-len",
                str(args.synthetic_len),
                "--centroid-k",
                args.centroid_k,
                "--centroid-v",
                args.centroid_v,
            ]
        run_checked(cmd)

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
