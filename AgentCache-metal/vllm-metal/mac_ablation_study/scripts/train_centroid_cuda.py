#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train and export AgentCache centroid tensors on a CUDA GPU.

This is a thin orchestration wrapper around the existing AgentCache scripts:

1. agentcache_compression/train_prefix_compression.py
2. agentcache_compression/transpose_tensors.py

It writes a self-contained centroid directory:

    centroid_K.npy
    centroid_V.npy
    sys_prefix_num_tokens.txt
    metadata.json

Example:

    python mac_ablation_study/scripts/train_centroid_cuda.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --tokens 128 \
        --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def find_agentcache_root() -> Path:
    env_root = os.environ.get("AGENTCACHE_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "agentcache_compression").is_dir():
            return root
        raise FileNotFoundError(
            f"AGENTCACHE_ROOT={root} does not contain agentcache_compression/"
        )

    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (candidate / "agentcache_compression").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find agentcache_compression/. Run from an AgentCache checkout "
        "or set AGENTCACHE_ROOT to the inner AgentCache repo root."
    )


ROOT = find_agentcache_root()
COMPRESSION = ROOT / "agentcache_compression"
TRAIN_SCRIPT = COMPRESSION / "train_prefix_compression.py"
EXPORT_SCRIPT = COMPRESSION / "transpose_tensors.py"
DEFAULT_DATA = COMPRESSION / "data" / "python_agent_train.jsonl"
DEFAULT_PROMPT = COMPRESSION / "prompts" / "2000_python_agent_system.txt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_tag(raw: str) -> str:
    return (
        raw.replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def prompt_tag(path: Path) -> str:
    name = path.stem
    first = name.split("_", 1)[0]
    return first if first.isdigit() else safe_tag(name)


def check_required_files() -> None:
    for path in (TRAIN_SCRIPT, EXPORT_SCRIPT, DEFAULT_DATA, DEFAULT_PROMPT):
        if not path.exists():
            raise FileNotFoundError(path)


def run(cmd: list[str | Path], *, env: dict[str, str], dry_run: bool) -> None:
    cmd_s = [str(part) for part in cmd]
    print("+ " + " ".join(cmd_s), flush=True)
    if dry_run:
        return
    subprocess.run(cmd_s, cwd=ROOT, env=env, check=True)


def is_dangerous_output_dir(path: Path) -> bool:
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        ROOT.resolve(),
        COMPRESSION.resolve(),
        (COMPRESSION / "adapters").resolve(),
        (COMPRESSION / "centroids").resolve(),
    }
    return path.resolve() in protected


def cuda_summary(python: str, env: dict[str, str], dry_run: bool) -> dict[str, object]:
    code = (
        "import json, torch; "
        "info={'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count(), "
        "'torch': torch.__version__}; "
        "info['devices']=[torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]; "
        "print(json.dumps(info))"
    )
    if dry_run:
        return {"dry_run": True}
    proc = subprocess.run(
        [python, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        check=True,
        capture_output=True,
    )
    info = json.loads(proc.stdout)
    if not info.get("cuda_available"):
        raise RuntimeError(
            "CUDA is not available from this Python environment. Activate the "
            "CUDA venv/conda env, then rerun this script."
        )
    print("CUDA:", json.dumps(info, indent=2), flush=True)
    return info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a PEFT prefix adapter and export centroid_K/V.npy on CUDA."
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "HF model id or local model path used for training, e.g. "
            "meta-llama/Llama-3.2-3B-Instruct"
        ),
    )
    p.add_argument("--tokens", type=int, default=128, help="Centroid length N.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Use 1 for safest 16 GB VRAM behavior; raise to 2-4 if it fits.",
    )
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Training JSONL with user and teacher_output fields.",
    )
    p.add_argument(
        "--system-prompt",
        default=str(DEFAULT_PROMPT),
        help="System prompt represented by the centroid.",
    )
    p.add_argument(
        "--system-retain-ratio",
        type=float,
        default=0.0,
        help="Compression mode should usually keep this at 0.0.",
    )
    p.add_argument(
        "--tag",
        default="",
        help="Output tag. Defaults to a sanitized model name.",
    )
    p.add_argument(
        "--adapter-dir",
        default="",
        help="Adapter output dir. Defaults under agentcache_compression/adapters/.",
    )
    p.add_argument(
        "--centroid-dir",
        default="",
        help="Centroid output dir. Defaults under agentcache_compression/centroids/.",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the CUDA environment.",
    )
    p.add_argument(
        "--cuda-visible-devices",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES value, e.g. 0.",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse an existing adapter-dir and only export centroid tensors.",
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Only train the adapter; do not export centroid tensors.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing adapter/centroid output directories first.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and output paths without running training.",
    )
    return p.parse_args()


def main() -> int:
    check_required_files()
    args = parse_args()

    data = Path(args.data).expanduser().resolve()
    system_prompt = Path(args.system_prompt).expanduser().resolve()
    if not data.exists():
        raise FileNotFoundError(data)
    if not system_prompt.exists():
        raise FileNotFoundError(system_prompt)

    tag = args.tag or safe_tag(args.model)
    run_tag = f"{tag}_N{args.tokens}_{prompt_tag(system_prompt)}"
    adapter_dir = (
        Path(args.adapter_dir).expanduser().resolve()
        if args.adapter_dir
        else COMPRESSION / "adapters" / run_tag
    )
    centroid_dir = (
        Path(args.centroid_dir).expanduser().resolve()
        if args.centroid_dir
        else COMPRESSION / "centroids" / run_tag
    )
    centroid_k = centroid_dir / "centroid_K.npy"
    centroid_v = centroid_dir / "centroid_V.npy"
    metadata_path = centroid_dir / "metadata.json"

    if args.overwrite:
        for path in (adapter_dir, centroid_dir):
            if path.exists():
                if is_dangerous_output_dir(path):
                    raise ValueError(f"Refusing to remove broad output directory: {path}")
                print(f"removing existing {path}", flush=True)
                if not args.dry_run:
                    shutil.rmtree(path)

    if not args.skip_export and (centroid_k.exists() or centroid_v.exists()):
        raise FileExistsError(
            f"Centroid outputs already exist in {centroid_dir}. Use --overwrite "
            "or choose --centroid-dir."
        )

    if not args.dry_run:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        centroid_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    started_at = utc_now()
    cuda_info = cuda_summary(args.python, env, args.dry_run)

    print("\nRun configuration:", flush=True)
    print(f"  root:          {ROOT}", flush=True)
    print(f"  model:         {args.model}", flush=True)
    print(f"  tokens:        {args.tokens}", flush=True)
    print(f"  epochs:        {args.epochs}", flush=True)
    print(f"  batch_size:    {args.batch_size}", flush=True)
    print(f"  adapter_dir:   {adapter_dir}", flush=True)
    print(f"  centroid_dir:  {centroid_dir}", flush=True)
    print()

    if not args.skip_train:
        run(
            [
                args.python,
                TRAIN_SCRIPT,
                "--model",
                args.model,
                "--data",
                data,
                "--system-prompt",
                system_prompt,
                "--output",
                adapter_dir,
                "--num-virtual-tokens",
                str(args.tokens),
                "--system-retain-ratio",
                str(args.system_retain_ratio),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--batch-size",
                str(args.batch_size),
            ],
            env=env,
            dry_run=args.dry_run,
        )

    if not args.skip_export:
        run(
            [
                args.python,
                EXPORT_SCRIPT,
                "--adapter",
                adapter_dir,
                "--out-k",
                centroid_k,
                "--out-v",
                centroid_v,
                "--sys-tokens",
                "0",
            ],
            env=env,
            dry_run=args.dry_run,
        )

    metadata = {
        "started_at": started_at,
        "finished_at": utc_now(),
        "model": args.model,
        "tokens": args.tokens,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "system_retain_ratio": args.system_retain_ratio,
        "data": str(data),
        "system_prompt": str(system_prompt),
        "adapter_dir": str(adapter_dir),
        "centroid_dir": str(centroid_dir),
        "centroid_k": str(centroid_k),
        "centroid_v": str(centroid_v),
        "python": args.python,
        "cuda": cuda_info,
    }
    if not args.dry_run:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nCentroid outputs:", flush=True)
    print(f"  K:        {centroid_k}", flush=True)
    print(f"  V:        {centroid_v}", flush=True)
    print(f"  sidecar:  {centroid_dir / 'sys_prefix_num_tokens.txt'}", flush=True)
    print(f"  metadata: {metadata_path}", flush=True)
    print("\nUse these paths as --centroid-k and --centroid-v in the Mac study.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
