#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Make a correctly-shaped DUMMY centroid for any model (TTFT studies only).

TTFT is independent of the centroid *values* (only shape/plumbing matter), so a
dummy centroid lets us measure the inject speedup for an arbitrary model size
WITHOUT training+exporting a real adapter. Quality numbers still need a real
trained centroid — do not use this for accuracy.

Shape = [num_hidden_layers, N, num_kv_heads * head_dim], read from the model's
HF config (no weight download).
"""
from __future__ import annotations

import argparse

import numpy as np
from transformers import AutoConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out-k", required=True)
    ap.add_argument("--out-v", required=True)
    ap.add_argument("--scale", type=float, default=0.02)
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model)
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    kv_dim = n_kv * head_dim

    rng = np.random.default_rng(0)
    shape = (n_layers, args.n, kv_dim)
    K = (rng.standard_normal(shape) * args.scale).astype(np.float32)
    V = (rng.standard_normal(shape) * args.scale).astype(np.float32)
    np.save(args.out_k, K)
    np.save(args.out_v, V)
    import os
    open(os.path.join(os.path.dirname(args.out_k), "sys_prefix_num_tokens.txt"), "w").write("0")
    print(f"[dummy] {args.model}: layers={n_layers} kv_heads={n_kv} head_dim={head_dim} "
          f"-> centroid {shape} (DUMMY, TTFT-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
