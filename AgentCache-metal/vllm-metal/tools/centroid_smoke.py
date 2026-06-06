#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live plumbing smoke test for Metal centroid injection.

Runs one offline generation in compression mode ([pad]*N + user) with injection
enabled, to verify the full path end-to-end:
  scheduler gap  →  injector seeds the paged cache  →  forward runs  →  tokens.

NOTE: with a DUMMY centroid the output is not expected to be coherent — this
checks the mechanism (gap + seed + forward), not quality. Quality needs a real
trained/exported centroid.
"""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [{"role": "user", "content": "Write a Python function that adds two numbers."}]
    # transformers 5.x apply_chat_template returns a BatchEncoding unless
    # tokenize=False; render to text then tokenize to a flat int list.
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt_ids = [pad] * args.n + list(ids)
    print(f"[smoke] N={args.n} user_tokens={len(ids)} total_prompt={len(prompt_ids)}")

    llm = LLM(model=args.model, max_model_len=2048)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)
    text = out[0].outputs[0].text
    print("\n[smoke] ===== GENERATED =====")
    print(text)
    print("[smoke] ===== END =====")
    print(f"[smoke] produced {len(out[0].outputs[0].token_ids)} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
