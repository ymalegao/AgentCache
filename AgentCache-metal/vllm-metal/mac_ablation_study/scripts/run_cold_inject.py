#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run cold full-prompt and centroid-injected benchmark modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from study_paths import (
    AC_CENTROIDS,
    EVAL_DATA,
    METAL_ROOT,
    ensure_results_dir,
    prompt_path,
    prompt_paths,
    python_executable,
    run_checked,
    safe_tag,
)


def parse_lengths(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    p.add_argument("--tag", default="")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--centroid-k", default=str(AC_CENTROIDS / "N128_2000_K.npy"))
    p.add_argument("--centroid-v", default=str(AC_CENTROIDS / "N128_2000_V.npy"))
    p.add_argument("--prompt-lengths", default="200,500,1000,2000")
    p.add_argument("--ttft-reps", type=int, default=5)
    p.add_argument("--gen-tokens", type=int, default=96)
    p.add_argument("--skip-accuracy", action="store_true")
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_results_dir(Path(args.out_dir) if args.out_dir else None)
    tag = args.tag or safe_tag(args.model)
    lengths = parse_lengths(args.prompt_lengths)
    prompts = prompt_paths(lengths, "python")
    cold_out = out_dir / f"{tag}_cold.json"
    inject_out = out_dir / f"{tag}_inject_N{args.n}.json"

    common = [
        "--model",
        args.model,
        "--n",
        str(args.n),
        "--eval-data",
        str(EVAL_DATA),
        "--ttft-reps",
        str(args.ttft_reps),
        "--gen-tokens",
        str(args.gen_tokens),
    ]
    if args.skip_accuracy:
        common.append("--skip-accuracy")

    run_checked(
        [
            python_executable(),
            METAL_ROOT / "tools" / "centroid_benchmark.py",
            "--mode",
            "cold",
            "--system-prompts",
            ",".join(str(p) for p in prompts),
            "--accuracy-system-prompt",
            str(prompt_path(2000, "python")),
            "--out",
            cold_out,
            *common,
        ]
    )

    run_checked(
        [
            python_executable(),
            METAL_ROOT / "tools" / "centroid_benchmark.py",
            "--mode",
            "inject",
            "--out",
            inject_out,
            *common,
        ],
        env_extra={
            "VLLM_CENTROID_SCHEDULER": "1",
            "VLLM_CENTROID_K_PATH": args.centroid_k,
            "VLLM_CENTROID_V_PATH": args.centroid_v,
            "VLLM_CENTROID_SYS_TOKENS": "0",
            "VLLM_CENTROID_LAYOUT": "compression",
        },
    )

    print(f"cold result:   {cold_out}")
    print(f"inject result: {inject_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
