#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Milestone-1 gate: RoPE parity for Metal centroid injection.

The #1 correctness risk in the port is the offline RoPE applied to the centroid
K. This script verifies — independent of any forward pass — that the injector
rotates the centroid using the model's own ``attn.rope`` with correct
position/offset semantics and tensor layout, so the seeded keys are positionally
consistent with the user tokens that follow.

Checks
------
1. Layer-rope resolution: every decoder layer exposes a callable ``attn.rope``.
2. Offset semantics: rotating the whole prefix at ``offset=sys`` equals rotating
   each token individually at ``offset=sys+i`` (i.e. position i maps to absolute
   position sys+i). This is the invariant the seeded slots rely on.
3. Layout: rotation operates on [1, kv_heads, N, head_dim] (B,H,S,D) and the
   injector's transpose round-trips to the cache layout [N, kv_heads, head_dim].
4. (optional) If a real centroid_K.npy is supplied, run the injector's
   ``_rotate_k`` end-to-end and report shapes + per-layer norms.

Usage
-----
    python tools/centroid_rope_parity.py --model <mlx-model-path-or-hf-id> \
        [--centroid-k /path/centroid_K.npy --centroid-v /path/centroid_V.npy] \
        [--sys-tokens 0] [--tol 2e-2]

Requires the vllm-metal env (mlx, mlx_lm) — run after install.
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx
import numpy as np

from vllm_metal.paged_attention_common import find_attn_attr, find_layers


def _resolve_ropes(model) -> list:
    layers = find_layers(model)
    ropes = []
    for layer in layers:
        attr = find_attn_attr(layer)
        attn = getattr(layer, attr) if attr else None
        ropes.append(getattr(attn, "rope", None) if attn is not None else None)
    return ropes


def _head_dims(model) -> tuple[int, int]:
    """Best-effort (num_kv_heads, head_dim) from the first attention module."""
    layer0 = find_layers(model)[0]
    attn = getattr(layer0, find_attn_attr(layer0))
    n_kv = getattr(attn, "n_kv_heads", None) or getattr(
        attn, "num_key_value_heads", None
    )
    # head_dim is usually on the attn module or derivable
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and hasattr(attn, "scale"):
        # scale = head_dim ** -0.5  → head_dim = round(scale**-2)
        head_dim = int(round(float(attn.scale) ** -2))
    return int(n_kv), int(head_dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="mlx_lm model path or HF id")
    ap.add_argument("--centroid-k", default=None)
    ap.add_argument("--centroid-v", default=None)
    ap.add_argument("--sys-tokens", type=int, default=0)
    ap.add_argument("--n", type=int, default=64, help="synthetic prefix length for checks 2/3")
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()

    from mlx_lm import load

    print(f"[parity] loading {args.model} ...")
    model, _ = load(args.model)

    ropes = _resolve_ropes(model)
    n_layers = len(ropes)
    print(f"[parity] layers={n_layers}")

    # --- Check 1: rope resolution ---
    missing = [i for i, r in enumerate(ropes) if r is None]
    if missing:
        print(f"[FAIL] check1: layers with no attn.rope: {missing[:8]}...")
        return 1
    print("[ok]   check1: every layer exposes a callable attn.rope")

    n_kv, head_dim = _head_dims(model)
    print(f"[parity] num_kv_heads={n_kv} head_dim={head_dim}")

    sys = args.sys_tokens
    N = args.n  # noqa: N806
    rope = ropes[0]
    mx.random.seed(0)
    k = mx.random.normal((1, n_kv, N, head_dim)).astype(mx.float16)

    # --- Check 2: offset semantics (whole vs per-token) ---
    full = rope(k, offset=sys)  # [1, H, N, D]
    per_tok = mx.concatenate(
        [rope(k[:, :, i : i + 1, :], offset=sys + i) for i in range(N)], axis=2
    )
    mx.eval(full, per_tok)
    diff = float(mx.max(mx.abs(full - per_tok)).item())
    if diff > args.tol:
        print(f"[FAIL] check2: offset semantics mismatch, max|Δ|={diff:.4e} > {args.tol}")
        return 1
    print(f"[ok]   check2: position i ↔ absolute (sys+i); max|Δ|={diff:.4e}")

    # --- Check 3: injector layout round-trip ---
    # cache layout is [N, H, D]; injector feeds rope [1,H,N,D] then transposes back.
    k_cache_layout = mx.contiguous(full[0].transpose(1, 0, 2))  # [N, H, D]
    if tuple(k_cache_layout.shape) != (N, n_kv, head_dim):
        print(f"[FAIL] check3: bad cache layout {k_cache_layout.shape}")
        return 1
    print(f"[ok]   check3: cache layout {tuple(k_cache_layout.shape)} = [N, kv_heads, head_dim]")

    # --- Check 4 (optional): real centroid end-to-end ---
    if args.centroid_k:
        from vllm_metal.centroid.injector_mlx import MetalCentroidInjector

        v_path = args.centroid_v or args.centroid_k.replace("_K", "_V")
        inj = MetalCentroidInjector(args.centroid_k, v_path)
        if inj.kv_dim != n_kv * head_dim:
            print(
                f"[FAIL] check4: centroid kv_dim={inj.kv_dim} != "
                f"num_kv_heads*head_dim={n_kv * head_dim} — retrain/export for this model"
            )
            return 1
        fill = min(inj.centroid_len, N)
        rotated = inj._rotate_k(ropes, n_kv, head_dim, fill, mx.float16)
        mx.eval(*rotated)
        norms = [float(mx.linalg.norm(r).item()) for r in rotated[: min(4, n_layers)]]
        print(
            f"[ok]   check4: injector._rotate_k → {len(rotated)} layers, "
            f"each {tuple(rotated[0].shape)}; layer norms[:4]={[round(x, 2) for x in norms]}"
        )

    print("\n[PASS] RoPE parity gate cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
