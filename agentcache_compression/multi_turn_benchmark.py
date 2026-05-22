"""
multi_turn_benchmark.py — Multi-turn cache benchmark.

Three modes (one per invocation via --mode):
  cold        Full system+history, no centroid, APC disabled. Baseline.
  warm_apc    Full system+history, APC enabled. Cache kicks in from turn 2.
  synthetic   No system; [pad]*N + conversation history; centroid injects KV
              for positions 0..N-1 every turn. APC caches pad prefix from turn 2.

Usage:
  python multi_turn_benchmark.py --model <path> --mode cold --out results/mt.jsonl
  python multi_turn_benchmark.py --model <path> --mode warm_apc --out results/mt.jsonl
  python multi_turn_benchmark.py --model <path> --mode synthetic --synthetic-len 64 \\
      --centroid-k centroids/N64_K.npy --centroid-v centroids/N64_V.npy \\
      --out results/mt.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent  # agentcache/
_EXP  = _REPO / "agentcache_compression"


# ---------------------------------------------------------------------------
# Prompt construction — single-turn (copied verbatim from test_compression.py)
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
# Generation helper (copied verbatim from test_compression.py)
# ---------------------------------------------------------------------------

def _generate(llm, prompt_input, sampling_params):
    """Dispatch to llm.generate, handling str vs list-of-int inputs."""
    if isinstance(prompt_input, list):
        from vllm.inputs import TokensPrompt
        return llm.generate([TokensPrompt(prompt_token_ids=prompt_input)], sampling_params)
    return llm.generate([prompt_input], sampling_params)


# ---------------------------------------------------------------------------
# Prompt construction — multi-turn
# ---------------------------------------------------------------------------

def build_multi_turn_prompt_str(tokenizer, system_text: str, history: list, user_text: str) -> str:
    """Full system + conversation history + current user turn as a string.

    history: list of (user_str, assistant_str) tuples from completed turns.
    System prefix is byte-identical across all turns so APC can cache it.
    """
    messages = [{"role": "system", "content": system_text}]
    for user_q, asst_ans in history:
        messages.append({"role": "user", "content": user_q})
        messages.append({"role": "assistant", "content": asst_ans})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_multi_turn_compression_ids(
    tokenizer, history: list, user_text: str, synthetic_len: int
) -> list[int]:
    """[pad]*N + token_ids(conversation history + current user turn).

    No system role — centroid injection handles positions 0..N-1.
    The [pad]*N prefix is identical across all turns; APC caches it after turn 1.
    """
    messages = []
    for user_q, asst_ans in history:
        messages.append({"role": "user", "content": user_q})
        messages.append({"role": "assistant", "content": asst_ans})
    messages.append({"role": "user", "content": user_text})
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    conv_ids: list[int] = tokenizer.encode(chat_text, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return [pad_id] * synthetic_len + conv_ids


# ---------------------------------------------------------------------------
# Per-turn measurement
# ---------------------------------------------------------------------------

def measure_turn_ttft_and_cached(llm, prompt_input) -> tuple[float, int]:
    """Single TTFT shot for one conversation turn.

    Each turn IS the data point — no repeat averaging.
    Returns (ttft_s, apc_cached_tokens).
    apc_cached_tokens comes directly from vLLM (0 when APC disabled or turn 1 is cold).
    """
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.perf_counter()
    result = _generate(llm, prompt_input, params)
    ttft_s = time.perf_counter() - t0
    cached = result[0].num_cached_tokens or 0
    return ttft_s, cached


def generate_turn_output(llm, prompt_input, max_tokens: int = 256) -> str:
    """Generate the assistant response for a turn (feeds into next turn's history)."""
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    out = _generate(llm, prompt_input, params)
    return out[0].outputs[0].text


# ---------------------------------------------------------------------------
# Arg parsing and env setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="/mnt/g/agentcache/models/Llama-3.2-1B-Instruct")
    p.add_argument("--data",           default=str(_EXP / "data"      / "python_agent_eval.jsonl"))
    p.add_argument("--system-prompt",  default=str(_EXP / "prompts"   / "2000_python_agent_system.txt"))
    p.add_argument("--centroid-k",     default=str(_EXP / "centroids" / "N64_K.npy"))
    p.add_argument("--centroid-v",     default=str(_EXP / "centroids" / "N64_V.npy"))
    p.add_argument("--synthetic-len",  type=int,   default=64)
    p.add_argument(
        "--mode",
        choices=["cold", "warm_apc", "synthetic"],
        required=True,
    )
    p.add_argument("--out",              default=str(_EXP / "results" / "multi_turn_benchmark.jsonl"))
    p.add_argument("--conversation-file", default=None,
                   help="JSON file with a list of user prompts forming one coherent conversation. "
                        "When set, --n-conversations and --turns-per-conv are ignored.")
    p.add_argument("--n-conversations", type=int,  default=5)
    p.add_argument("--turns-per-conv",  type=int,  default=5)
    p.add_argument("--max-tokens",      type=int,  default=256)
    p.add_argument("--gpu-mem",         type=float, default=0.6)
    return p.parse_args()


def setup_env(args: argparse.Namespace) -> None:
    """Set centroid env vars before importing vllm so module-level caches read them."""
    if args.mode == "synthetic":
        k = args.centroid_k or os.environ.get("VLLM_CENTROID_K_PATH", "")
        v = args.centroid_v or os.environ.get("VLLM_CENTROID_V_PATH", "")
        if not k or not os.path.exists(k):
            raise FileNotFoundError(f"Centroid K not found: {k!r}. Pass --centroid-k.")
        if not v or not os.path.exists(v):
            raise FileNotFoundError(f"Centroid V not found: {v!r}. Pass --centroid-v.")
        os.environ["VLLM_CENTROID_SCHEDULER"] = "1"
        os.environ["VLLM_CENTROID_K_PATH"] = k
        os.environ["VLLM_CENTROID_V_PATH"] = v
        os.environ["VLLM_CENTROID_SYS_TOKENS"] = "0"
        os.environ["VLLM_CENTROID_LAYOUT"] = "compression"
    else:
        os.environ["VLLM_CENTROID_SCHEDULER"] = "0"
        os.environ.pop("VLLM_CENTROID_LAYOUT", None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_env(args)

    # Import vllm AFTER setting env vars
    from vllm import LLM, SamplingParams  # noqa: F401
    from transformers import AutoTokenizer

    system_text = Path(args.system_prompt).read_text().strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    enable_apc = args.mode in ("warm_apc", "synthetic")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=enable_apc,
    )

    # Build conversation list: either from a --conversation-file or from eval tasks
    if args.conversation_file:
        raw = json.loads(Path(args.conversation_file).read_text())
        conversations = [
            [{"id": f"turn_{i+1}", "user": q} for i, q in enumerate(raw)]
        ]
    else:
        tasks = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
        n_convs = args.n_conversations
        turns = args.turns_per_conv
        needed = n_convs * turns
        if len(tasks) < needed:
            print(f"WARNING: only {len(tasks)} tasks, need {needed}. Truncating to {len(tasks) // turns} conversations.")
            n_convs = len(tasks) // turns
        conversations = [
            [{"id": tasks[conv_id * turns + t]["id"], "user": tasks[conv_id * turns + t]["user"]}
             for t in range(turns)]
            for conv_id in range(n_convs)
        ]

    n_convs = len(conversations)
    turns = len(conversations[0])
    print(f"\n=== mode={args.mode}  N={args.synthetic_len}  convs={n_convs}  turns_per_conv={turns} ===\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as fout:
        for conv_id, conv_tasks in enumerate(conversations):
            history: list[tuple[str, str]] = []

            for turn_idx, task in enumerate(conv_tasks):
                user_text = task["user"]
                turn_num = turn_idx + 1  # 1-based

                if args.mode == "synthetic":
                    prompt_input = build_multi_turn_compression_ids(
                        tokenizer, history, user_text, args.synthetic_len
                    )
                    physical_tokens = len(prompt_input)
                    centroid_tokens_saved = args.synthetic_len
                else:
                    prompt_input = build_multi_turn_prompt_str(
                        tokenizer, system_text, history, user_text
                    )
                    physical_tokens = len(tokenizer.encode(prompt_input))
                    centroid_tokens_saved = 0

                ttft_s, apc_cached_tokens = measure_turn_ttft_and_cached(llm, prompt_input)

                effective_cache_hit_rate = apc_cached_tokens / physical_tokens if physical_tokens > 0 else 0.0

                assistant_text = generate_turn_output(llm, prompt_input, args.max_tokens)
                history.append((user_text, assistant_text))

                record = {
                    "conversation_id": conv_id,
                    "turn": turn_num,
                    "task_id": task["id"],
                    "mode": args.mode,
                    "N": args.synthetic_len,
                    "physical_prompt_tokens": physical_tokens,
                    "centroid_tokens_saved": centroid_tokens_saved,
                    "apc_cached_tokens": apc_cached_tokens,
                    "effective_cache_hit_rate": round(effective_cache_hit_rate, 4),
                    "ttft_s": round(ttft_s, 5),
                    "user": user_text,
                    "response": assistant_text,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()

                print(
                    f"  conv={conv_id} turn={turn_num} {task['id']}  "
                    f"ttft={ttft_s:.4f}s  phys_tokens={physical_tokens}  "
                    f"apc_cached={apc_cached_tokens}  "
                    f"cache_hit_rate={effective_cache_hit_rate:.3f}"
                )

    print(f"\nResults appended to {out_path}")


if __name__ == "__main__":
    main()
