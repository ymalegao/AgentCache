#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared paths and subprocess helpers for the Mac ablation study."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_ROOT = SCRIPT_DIR.parent
METAL_ROOT = STUDY_ROOT.parent
RESULTS_DIR = STUDY_ROOT / "results"
VENV_PY = METAL_ROOT / ".venv-vllm-metal" / "bin" / "python"
VENV_BIN = VENV_PY.parent


def find_agentcache_root() -> Path:
    env_root = os.environ.get("AGENTCACHE_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "agentcache_compression").is_dir():
            return root
        raise FileNotFoundError(
            f"AGENTCACHE_ROOT={root} does not contain agentcache_compression/"
        )

    for cand in (METAL_ROOT, *METAL_ROOT.parents):
        if (cand / "agentcache_compression").is_dir():
            return cand
    raise FileNotFoundError(
        "Could not find agentcache_compression/. Set AGENTCACHE_ROOT to the "
        "inner AgentCache repo root."
    )


AGENTCACHE_ROOT = find_agentcache_root()
AC_COMPRESSION = AGENTCACHE_ROOT / "agentcache_compression"
PROMPTS_DIR = AC_COMPRESSION / "prompts"
EVAL_DATA = AC_COMPRESSION / "data" / "python_agent_eval.jsonl"
AC_CENTROIDS = AC_COMPRESSION / "centroids"


def python_executable() -> str:
    return str(VENV_PY if VENV_PY.exists() else Path(sys.executable))


def base_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if VENV_BIN.exists():
        env["PATH"] = f"{VENV_BIN}{os.pathsep}{env.get('PATH', '')}"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def run_checked(
    cmd: list[str | Path],
    *,
    env_extra: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    cmd_s = [str(x) for x in cmd]
    print("+ " + " ".join(cmd_s), flush=True)
    return subprocess.run(
        cmd_s,
        cwd=METAL_ROOT,
        env=base_env(env_extra),
        text=True,
        check=True,
        capture_output=capture_output,
    )


def prompt_path(length: int, domain: str = "python") -> Path:
    path = PROMPTS_DIR / f"{length}_{domain}_agent_system.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def prompt_paths(lengths: list[int], domain: str = "python") -> list[Path]:
    return [prompt_path(length, domain) for length in lengths]


def ensure_results_dir(path: Path | None = None) -> Path:
    out = path or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_tag(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
    )
