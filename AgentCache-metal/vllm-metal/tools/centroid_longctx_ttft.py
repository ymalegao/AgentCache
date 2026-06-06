#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Long-context cold TTFT sweep vs the flat inject line.

Synthesizes system prompts of increasing length (by repeating a base prompt's
tokens) and measures cold TTFT, to show how the cold-vs-inject speedup scales
with context length. Inject TTFT is flat (~N+user physical tokens) so it's a
constant reference line.
"""
from __future__ import annotations

import argparse
import statistics
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--base-prompt", required=True)
    ap.add_argument("--inject-ms", type=float, default=23.2,
                    help="measured flat inject TTFT (N=128) for the speedup column")
    ap.add_argument("--lengths", default="1000,2000,4000,8000,12000,16000")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=20000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    base_ids = tok(open(args.base_prompt).read(), add_special_tokens=False).input_ids
    user_ids = tok("Write a Python function that reverses a string.",
                   add_special_tokens=False).input_ids

    llm = LLM(model=args.model, max_model_len=args.max_model_len)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def ttft(ids):
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            ts.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(ts[1:]) if len(ts) > 1 else ts[0]

    print(f"{'ctx_tokens':>10}{'cold_ms':>9}{'inject_ms':>10}{'speedup':>9}")
    for target in [int(x) for x in args.lengths.split(",")]:
        n = max(1, (target - len(user_ids)) // len(base_ids) + 1)
        sys_ids = (base_ids * n)[: target - len(user_ids)]
        ids = sys_ids + user_ids
        ms = ttft(ids)
        print(f"{len(ids):>10}{ms:>9.1f}{args.inject_ms:>10.1f}{ms / args.inject_ms:>8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
