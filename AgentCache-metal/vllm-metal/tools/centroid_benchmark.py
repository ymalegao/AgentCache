#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-vs-inject benchmark for Metal centroid injection.

Uses the same prompts + eval set as the CUDA port:
  - system prompts: agentcache_compression/prompts/{200,500,1000,2000}_*.txt
  - eval set:       agentcache_compression/data/python_agent_eval.jsonl

Two modes (run as separate processes; injection is set by env at engine init):
  cold   — full system prompt in the physical prompt, no injection.
           prompt = chat_template([system, user]).
  inject — compression: prompt = [pad]*N + chat_template([user]); centroid injected.

Measures, per mode:
  TTFT proxy — wall time of generate(max_tokens=1) on a fixed query, median of
               --ttft-reps (first run dropped as warmup). For cold, swept across
               every --system-prompts file to show TTFT-vs-context scaling.
  Accuracy   — over the eval set (max_tokens --gen-tokens): task_pass
               (must_include_any) + coherence (>=15 words, no degenerate repeat).

Writes a JSON summary to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt


def tokenize_chat(tok, messages: list[dict]) -> list[int]:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return list(tok(text, add_special_tokens=False).input_ids)


def build_prompt(tok, mode: str, system_text: str | None, user: str, n: int, pad: int):
    if mode == "inject":
        ids = tokenize_chat(tok, [{"role": "user", "content": user}])
        return [pad] * n + ids
    msgs = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": user})
    return tokenize_chat(tok, msgs)


def is_coherent(text: str) -> bool:
    words = text.split()
    if len(words) < 15:
        return False
    # degenerate: any token repeated >8x consecutively
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run > 8:
            return False
    return True


def task_pass(text: str, checks: dict) -> bool:
    groups = checks.get("must_include_any", [])
    if not groups:
        return True
    t = text.lower()
    return all(any(kw.lower() in t for kw in group) for group in groups)


def measure_ttft(llm, prompt_ids, reps: int) -> float:
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp, use_tqdm=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times[1:]) if len(times) > 1 else times[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--mode", choices=["cold", "inject"], required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--system-prompts", default="", help="comma-sep files (cold TTFT sweep)")
    ap.add_argument("--accuracy-system-prompt", default="", help="system prompt file for cold accuracy")
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--ttft-query", default="Write a Python function that reverses a string.")
    ap.add_argument("--ttft-reps", type=int, default=5)
    ap.add_argument("--gen-tokens", type=int, default=96)
    ap.add_argument("--skip-accuracy", action="store_true", help="TTFT only (fast)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    inject_on = os.environ.get("VLLM_CENTROID_SCHEDULER") == "1"
    print(f"[bench] mode={args.mode} injection_env={'on' if inject_on else 'off'} N={args.n}")

    llm = LLM(model=args.model, max_model_len=4096)

    result: dict = {"mode": args.mode, "model": args.model, "n": args.n,
                    "injection_env": inject_on, "ttft": [], "accuracy": {}}

    # ---- TTFT ----
    if args.mode == "inject":
        ids = build_prompt(tok, "inject", None, args.ttft_query, args.n, pad)
        ttft = measure_ttft(llm, ids, args.ttft_reps)
        result["ttft"].append({"context": "compression", "prompt_tokens": len(ids), "ttft_ms": ttft})
        print(f"[bench] TTFT inject: {len(ids)} tok -> {ttft:.1f} ms")
    else:
        for f in [p for p in args.system_prompts.split(",") if p]:
            sys_text = Path(f).read_text()
            ids = build_prompt(tok, "cold", sys_text, args.ttft_query, args.n, pad)
            ttft = measure_ttft(llm, ids, args.ttft_reps)
            label = Path(f).stem
            result["ttft"].append({"context": label, "prompt_tokens": len(ids), "ttft_ms": ttft})
            print(f"[bench] TTFT cold[{label}]: {len(ids)} tok -> {ttft:.1f} ms")

    # ---- Accuracy ----
    if args.skip_accuracy:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[bench] (accuracy skipped) wrote {args.out}")
        return 0

    sys_text = None
    if args.mode == "cold" and args.accuracy_system_prompt:
        sys_text = Path(args.accuracy_system_prompt).read_text()
    sp = SamplingParams(max_tokens=args.gen_tokens, temperature=0.0)
    records = [json.loads(l) for l in Path(args.eval_data).read_text().splitlines() if l.strip()]
    n_pass = n_coh = 0
    per = []
    for rec in records:
        ids = build_prompt(tok, args.mode, sys_text, rec["user"], args.n, pad)
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        text = out[0].outputs[0].text
        tp = task_pass(text, rec.get("checks", {}))
        co = is_coherent(text)
        n_pass += tp; n_coh += co
        per.append({"id": rec["id"], "task_pass": tp, "coherent": co, "chars": len(text)})
    total = len(records)
    result["accuracy"] = {"total": total, "task_pass": n_pass, "task_pass_rate": n_pass / total,
                          "coherent": n_coh, "coherence_rate": n_coh / total, "per_record": per}
    print(f"[bench] accuracy: task_pass {n_pass}/{total} ({n_pass/total:.0%}), "
          f"coherent {n_coh}/{total} ({n_coh/total:.0%})")

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
