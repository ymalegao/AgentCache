"""
test_compression.py — Phase 8 evaluation harness.

Three modes (one per invocation via --mode):
  cold_no_synthetic    Full system+user prompt, no centroid, APC disabled.
  warm_apc             Full system+user prompt, APC enabled (second call is warm).
  synthetic_compression  No system; [pad]*N + user-chat tokens; centroid injects KV for 0..N-1.

Usage:
  # Run each mode once, append results to the same output file.
  for MODE in cold_no_synthetic warm_apc synthetic_compression; do
    VLLM_CENTROID_K_PATH=.../N64_K.npy \\
    VLLM_CENTROID_V_PATH=.../N64_V.npy \\
    python test_compression.py \\
      --model /mnt/g/agentcache/models/Llama-3.2-1B-Instruct \\
      --data experiments/agentcache_compression/data/python_agent_eval.jsonl \\
      --system-prompt experiments/agentcache_compression/prompts/python_agent_system.txt \\
      --centroid-k $VLLM_CENTROID_K_PATH \\
      --centroid-v $VLLM_CENTROID_V_PATH \\
      --synthetic-len 64 \\
      --mode $MODE \\
      --out experiments/agentcache_compression/results/N64_comparison.jsonl
  done
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt_str(tokenizer, system_text: str, user_text: str) -> str:
    """Full system+user prompt as a string (cold / warm_apc modes)."""
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_compression_ids(tokenizer, user_text: str, synthetic_len: int) -> list[int]:
    """[pad]*N + user-chat token IDs (no system prompt).

    The N pad tokens are NEVER computed — the centroid gap mechanism skips them
    and injects synthetic KV for positions 0..N-1.  All M user tokens are
    scheduled for normal prefill at positions N..N+M-1.
    """
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    user_chat_ids: list[int] = tokenizer.encode(chat_text, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return [pad_id] * synthetic_len + user_chat_ids


# ---------------------------------------------------------------------------
# TTFT measurement
# ---------------------------------------------------------------------------

def _generate(llm, prompt_input, sampling_params):
    """Dispatch to llm.generate, handling str vs list-of-int inputs."""
    if isinstance(prompt_input, list):
        from vllm.inputs import TokensPrompt
        return llm.generate([TokensPrompt(prompt_token_ids=prompt_input)], sampling_params)
    return llm.generate([prompt_input], sampling_params)


def measure_ttft_s(llm, prompt_input, n_runs: int = 3) -> tuple[float, list[float]]:
    """Return (mean_ttft_s, per_run_times). Uses max_tokens=1 to isolate prefill."""
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=1)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _generate(llm, prompt_input, params)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), times


def generate_output(llm, prompt_input, max_tokens: int = 256) -> str:
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    out = _generate(llm, prompt_input, params)
    return out[0].outputs[0].text


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def check_coherent(text: str) -> bool:
    """At least 20 words and no obvious repeated-token degeneration."""
    words = text.split()
    if len(words) < 20:
        return False
    # Flag if any single token appears in >40% of the last 30 words
    tail = words[-30:] if len(words) >= 30 else words
    for w in set(tail):
        if tail.count(w) / len(tail) > 0.4:
            return False
    return True


def check_task(text: str, must_include_any: list[list[str]]) -> bool:
    lower = text.lower()
    for group in must_include_any:
        if any(kw.lower() in lower for kw in group):
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent.parent  # agentcache/
_EXP  = _REPO / "experiments" / "agentcache_compression"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="/mnt/g/agentcache/models/Llama-3.2-1B-Instruct")
    p.add_argument("--data",          default=str(_EXP / "data"      / "python_agent_eval.jsonl"))
    p.add_argument("--system-prompt", default=str(_EXP / "prompts"   / "python_agent_system.txt"))
    p.add_argument("--centroid-k",    default=str(_EXP / "centroids" / "N64_K.npy"))
    p.add_argument("--centroid-v",    default=str(_EXP / "centroids" / "N64_V.npy"))
    p.add_argument("--synthetic-len", type=int, default=64)
    p.add_argument(
        "--mode",
        choices=["cold_no_synthetic", "warm_apc", "synthetic_compression"],
        required=True,
    )
    p.add_argument("--out",           default=str(_EXP / "results" / "N64_comparison.jsonl"))
    p.add_argument("--n-ttft-runs",   type=int,   default=3)
    p.add_argument("--max-tokens",    type=int,   default=512)
    p.add_argument("--gpu-mem",       type=float, default=0.6)
    return p.parse_args()


def setup_env(args: argparse.Namespace) -> None:
    """Set centroid env vars before importing vllm so module-level caches read them."""
    if args.mode == "synthetic_compression":
        k = args.centroid_k or os.environ.get("VLLM_CENTROID_K_PATH", "")
        v = args.centroid_v or os.environ.get("VLLM_CENTROID_V_PATH", "")
        if not k or not os.path.exists(k):
            raise FileNotFoundError(f"Centroid K not found: {k!r}. Pass --centroid-k.")
        if not v or not os.path.exists(v):
            raise FileNotFoundError(f"Centroid V not found: {v!r}. Pass --centroid-v.")
        os.environ["VLLM_CENTROID_SCHEDULER"] = "1"
        os.environ["VLLM_CENTROID_K_PATH"] = k
        os.environ["VLLM_CENTROID_V_PATH"] = v
        os.environ["VLLM_CENTROID_SYS_TOKENS"] = "0"   # no system in prompt
        os.environ["VLLM_CENTROID_LAYOUT"] = "compression"
    else:
        os.environ["VLLM_CENTROID_SCHEDULER"] = "0"
        os.environ.pop("VLLM_CENTROID_LAYOUT", None)


def main() -> None:
    args = parse_args()
    setup_env(args)

    # Import vllm AFTER setting env vars
    from vllm import LLM, SamplingParams  # noqa: F401
    from transformers import AutoTokenizer

    system_text = Path(args.system_prompt).read_text().strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build LLM
    enable_apc = args.mode == "warm_apc"
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=enable_apc,
    )

    # Load eval tasks
    tasks = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]

    print(f"\n=== mode={args.mode}  N={args.synthetic_len}  tasks={len(tasks)} ===\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as fout:
        for task in tasks:
            user_text = task["user"]
            checks_spec = task.get("checks", {})
            must_include_any = checks_spec.get("must_include_any", [])

            # Build prompt input
            if args.mode == "synthetic_compression":
                ids = build_compression_ids(tokenizer, user_text, args.synthetic_len)
                prompt_input = ids  # plain list[int]; _generate dispatches via prompt_token_ids kwarg
                user_tokens = len(ids) - args.synthetic_len
                pad_tokens = args.synthetic_len
                physical_tokens = len(ids)
            else:
                prompt_str = build_prompt_str(tokenizer, system_text, user_text)
                prompt_input = prompt_str
                physical_tokens = len(tokenizer.encode(prompt_str))
                user_tokens = None
                pad_tokens = 0

            # For warm_apc: one warmup call first
            if args.mode == "warm_apc":
                _generate(llm, prompt_input, SamplingParams(temperature=0.0, max_tokens=1))

            # TTFT
            ttft_mean, ttft_runs = measure_ttft_s(llm, prompt_input, n_runs=args.n_ttft_runs)

            # Quality output
            output_text = generate_output(llm, prompt_input, max_tokens=args.max_tokens)

            coherent = check_coherent(output_text)
            ends_with_goodbye = output_text.strip().endswith("GOODBYE")
            task_check_pass = check_task(output_text, must_include_any) if must_include_any else None

            record = {
                "id": task["id"],
                "mode": args.mode,
                "N": args.synthetic_len,
                "physical_prompt_tokens": physical_tokens,
                "pad_tokens": pad_tokens,
                "user_tokens": user_tokens,
                "ttft_mean_s": round(ttft_mean, 5),
                "ttft_runs_s": [round(t, 5) for t in ttft_runs],
                "checks": {
                    "coherent": coherent,
                    "ends_with_goodbye": ends_with_goodbye,
                    "task_check_pass": task_check_pass,
                },
                "output": output_text,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            status = "✓" if (coherent and ends_with_goodbye) else "✗"
            print(
                f"  {status} {task['id']}  ttft={ttft_mean:.4f}s  "
                f"coherent={coherent}  goodbye={ends_with_goodbye}  "
                f"task_check={task_check_pass}"
            )

    print(f"\nResults appended to {out_path}")


if __name__ == "__main__":
    main()
