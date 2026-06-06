#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the centroid RoPE/layout parity gate for the Mac study."""

from __future__ import annotations

import argparse

from study_paths import AC_CENTROIDS, METAL_ROOT, python_executable, run_checked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    p.add_argument("--centroid-k", default=str(AC_CENTROIDS / "N128_2000_K.npy"))
    p.add_argument("--centroid-v", default=str(AC_CENTROIDS / "N128_2000_V.npy"))
    p.add_argument("--sys-tokens", default="0")
    p.add_argument("--n", type=int, default=128)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_checked(
        [
            python_executable(),
            METAL_ROOT / "tools" / "centroid_rope_parity.py",
            "--model",
            args.model,
            "--centroid-k",
            args.centroid_k,
            "--centroid-v",
            args.centroid_v,
            "--sys-tokens",
            args.sys_tokens,
            "--n",
            str(args.n),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
