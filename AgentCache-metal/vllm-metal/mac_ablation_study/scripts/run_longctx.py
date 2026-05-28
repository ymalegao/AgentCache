#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run long-context cold TTFT sweep and save parsed JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from study_paths import METAL_ROOT, ensure_results_dir, prompt_path, python_executable, run_checked, safe_tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    p.add_argument("--tag", default="")
    p.add_argument("--base-prompt", default="")
    p.add_argument("--inject-ms", type=float, required=True)
    p.add_argument("--lengths", default="1000,2000,4000,8000,12000,16000")
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=20000)
    p.add_argument("--out", default="")
    return p.parse_args()


def parse_table(stdout: str) -> list[dict]:
    rows = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        rows.append(
            {
                "ctx_tokens": int(parts[0]),
                "cold_ms": float(parts[1]),
                "inject_ms": float(parts[2]),
                "speedup": float(parts[3].rstrip("x")),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = ensure_results_dir()
    tag = args.tag or safe_tag(args.model)
    out = Path(args.out) if args.out else out_dir / f"{tag}_longctx.json"
    base_prompt = Path(args.base_prompt) if args.base_prompt else prompt_path(2000, "python")

    proc = run_checked(
        [
            python_executable(),
            METAL_ROOT / "tools" / "centroid_longctx_ttft.py",
            "--model",
            args.model,
            "--base-prompt",
            base_prompt,
            "--inject-ms",
            str(args.inject_ms),
            "--lengths",
            args.lengths,
            "--reps",
            str(args.reps),
            "--max-model-len",
            str(args.max_model_len),
        ],
        capture_output=True,
    )
    print(proc.stdout)
    rows = parse_table(proc.stdout)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "tag": tag,
                "base_prompt": str(base_prompt),
                "inject_ms": args.inject_ms,
                "lengths": args.lengths,
                "rows": rows,
                "stdout": proc.stdout,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
