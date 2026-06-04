#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optional TTFT-only model-size sweep for the Mac study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from study_paths import (
    AC_CENTROIDS,
    METAL_ROOT,
    RESULTS_DIR,
    ensure_results_dir,
    python_executable,
    run_checked,
    safe_tag,
)


DEFAULT_SPECS = [
    {
        "tag": "qwen05",
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        "n": 64,
        "centroid_k": str(METAL_ROOT / "centroids" / "qwen05_N64_2000_K.npy"),
        "centroid_v": str(METAL_ROOT / "centroids" / "qwen05_N64_2000_V.npy"),
    },
    {
        "tag": "llama1b",
        "model": "mlx-community/Llama-3.2-1B-Instruct-bf16",
        "n": 128,
        "centroid_k": str(AC_CENTROIDS / "N128_2000_K.npy"),
        "centroid_v": str(AC_CENTROIDS / "N128_2000_V.npy"),
    },
    {"tag": "llama3b", "model": "mlx-community/Llama-3.2-3B-Instruct-bf16", "n": 128},
    {"tag": "qwen7b", "model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "n": 128},
    {"tag": "qwen14b", "model": "mlx-community/Qwen2.5-14B-Instruct-4bit", "n": 128},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        default="",
        help="Comma-separated subset of tags to run. Default runs the first-pass subset.",
    )
    p.add_argument("--lengths", default="1000,2000,4000,8000,12000,16000")
    p.add_argument("--ttft-reps", type=int, default=4)
    p.add_argument("--out", default=str(RESULTS_DIR / "model_sweep_summary.json"))
    return p.parse_args()


def ensure_centroid(spec: dict) -> tuple[Path, Path]:
    ck = Path(spec.get("centroid_k", RESULTS_DIR / f"dummy_{spec['tag']}_K.npy"))
    cv = Path(spec.get("centroid_v", RESULTS_DIR / f"dummy_{spec['tag']}_V.npy"))
    if ck.exists() and cv.exists():
        return ck, cv
    run_checked(
        [
            python_executable(),
            METAL_ROOT / "tools" / "make_dummy_centroid.py",
            "--model",
            spec["model"],
            "--n",
            str(spec["n"]),
            "--out-k",
            ck,
            "--out-v",
            cv,
        ]
    )
    return ck, cv


def main() -> int:
    args = parse_args()
    ensure_results_dir()
    selected_tags = {x.strip() for x in args.models.split(",") if x.strip()}
    specs = [s for s in DEFAULT_SPECS if not selected_tags or s["tag"] in selected_tags]
    summary = []

    for spec in specs:
        ck, cv = ensure_centroid(spec)
        run_checked(
            [
                python_executable(),
                METAL_ROOT / "mac_ablation_study" / "scripts" / "run_cold_inject.py",
                "--model",
                spec["model"],
                "--tag",
                spec["tag"],
                "--n",
                str(spec["n"]),
                "--centroid-k",
                ck,
                "--centroid-v",
                cv,
                "--prompt-lengths",
                "2000",
                "--ttft-reps",
                str(args.ttft_reps),
                "--skip-accuracy",
            ]
        )
        inject_path = RESULTS_DIR / f"{spec['tag']}_inject_N{spec['n']}.json"
        inject = json.loads(inject_path.read_text())
        inject_ms = float(inject["ttft"][0]["ttft_ms"])
        run_checked(
            [
                python_executable(),
                METAL_ROOT / "mac_ablation_study" / "scripts" / "run_longctx.py",
                "--model",
                spec["model"],
                "--tag",
                spec["tag"],
                "--inject-ms",
                str(inject_ms),
                "--lengths",
                args.lengths,
            ]
        )
        summary.append(
            {
                "tag": spec["tag"],
                "model": spec["model"],
                "n": spec["n"],
                "centroid_k": str(ck),
                "centroid_v": str(cv),
                "inject_ms": inject_ms,
                "longctx_result": str(RESULTS_DIR / f"{spec['tag']}_longctx.json"),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
