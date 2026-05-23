"""
multi_turn_benchmark.py — Multi-turn cache benchmark (vllm serve mode).

This script starts vllm serve, runs the benchmark, then stops the server.
TTFT is measured as time-to-first streaming chunk via the OpenAI-compatible API.

Three modes (one per invocation via --mode):
  cold        Full system+history, APC disabled. Baseline.
  warm_apc    Full system+history, APC enabled. Cache kicks in from turn 2.
  synthetic   [pad]*N + conversation history via completions API; centroid injects KV
              for positions 0..N-1. APC caches pad prefix from turn 2.

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
import subprocess
import time
from pathlib import Path

import openai
import requests


_REPO = Path(__file__).resolve().parent.parent
_EXP  = _REPO / "agentcache_compression"

SERVER_READY_TIMEOUT = 180  # seconds to wait for vllm serve to become healthy


# ---------------------------------------------------------------------------
# Prompt / token construction
# ---------------------------------------------------------------------------

def build_messages(system_text: str, history: list[tuple[str, str]], user_text: str) -> list[dict]:
    """OpenAI messages list: system + history + current user turn."""
    msgs = [{"role": "system", "content": system_text}]
    for user_q, asst_ans in history:
        msgs.append({"role": "user", "content": user_q})
        msgs.append({"role": "assistant", "content": asst_ans})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def build_compression_ids(
    tokenizer, history: list[tuple[str, str]], user_text: str, synthetic_len: int
) -> list[int]:
    """[pad]*N + token IDs for (history + current turn), no system prompt.

    Pad tokens are placeholders for centroid KV injection in the scheduler.
    """
    messages = []
    for user_q, asst_ans in history:
        messages.append({"role": "user", "content": user_q})
        messages.append({"role": "assistant", "content": asst_ans})
    messages.append({"role": "user", "content": user_text})
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    conv_ids: list[int] = tokenizer.encode(chat_text, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return [pad_id] * synthetic_len + conv_ids


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def build_centroid_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    env["VLLM_CENTROID_SCHEDULER"] = "1"
    env["VLLM_CENTROID_K_PATH"]    = args.centroid_k
    env["VLLM_CENTROID_V_PATH"]    = args.centroid_v
    env["VLLM_CENTROID_SYS_TOKENS"] = "0"
    env["VLLM_CENTROID_LAYOUT"]    = "compression"
    return env


def build_server_cmd(args: argparse.Namespace, enable_apc: bool) -> list[str]:
    cmd = [
        "vllm", "serve", args.model,
        "--port", str(args.server_port),
        "--gpu-memory-utilization", str(args.gpu_mem),
        "--max-model-len", str(args.max_model_len),
    ]
    if enable_apc:
        cmd.append("--enable-prefix-caching")
    return cmd


def start_server(cmd: list[str], env: dict, port: int) -> subprocess.Popen:
    """Start vllm serve and block until /health returns 200 or timeout."""
    import tempfile
    stderr_file = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=stderr_file)
    stderr_file.close()
    health_url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + SERVER_READY_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr_text = Path(stderr_file.name).read_text(errors="replace")[-2000:]
            raise RuntimeError(
                f"vllm serve exited unexpectedly (returncode={proc.returncode})\n"
                f"--- stderr (last 2000 chars) ---\n{stderr_text}"
            )
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                return proc
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    proc.terminate()
    raise TimeoutError(f"vllm serve did not become ready within {SERVER_READY_TIMEOUT}s")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Prometheus metrics (HTTP scrape from running server)
# ---------------------------------------------------------------------------

def _read_prefix_cache_counters(port: int) -> tuple[float, float]:
    """Return (hits, queries) scraped from vllm serve's /metrics endpoint."""
    try:
        r = requests.get(f"http://localhost:{port}/metrics", timeout=5)
        hits = queries = 0.0
        for line in r.text.splitlines():
            if line.startswith("#"):
                continue
            if "prefix_cache_hits" in line:
                hits += float(line.split()[-1])
            elif "prefix_cache_queries" in line:
                queries += float(line.split()[-1])
        return hits, queries
    except Exception:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Per-turn measurement
# ---------------------------------------------------------------------------

def _is_messages(prompt_input) -> bool:
    return isinstance(prompt_input, list) and prompt_input and isinstance(prompt_input[0], dict)


def measure_turn_ttft_and_cached(
    client: openai.OpenAI,
    prompt_input,
    model: str,
) -> tuple[float, int, int]:
    """Stream one token and return (ttft_s, cached_tokens, prompt_tokens).

    prompt_input is either a list[dict] (messages) or list[int] (token IDs).
    """
    t0 = time.perf_counter()
    ttft_s = None
    prompt_tokens = 0
    cached_tokens = 0

    if _is_messages(prompt_input):
        stream = client.chat.completions.create(
            model=model,
            messages=prompt_input,
            max_tokens=1,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if ttft_s is None and chunk.choices and chunk.choices[0].delta.content:
                ttft_s = time.perf_counter() - t0
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                if details:
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0
    else:
        # Synthetic mode: pass token IDs via completions endpoint
        stream = client.completions.create(
            model=model,
            prompt=prompt_input,
            max_tokens=1,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if ttft_s is None and chunk.choices and chunk.choices[0].text:
                ttft_s = time.perf_counter() - t0
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                if details:
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0

    if ttft_s is None:
        ttft_s = time.perf_counter() - t0
    if prompt_tokens == 0 and not _is_messages(prompt_input):
        prompt_tokens = len(prompt_input)

    return ttft_s, cached_tokens, prompt_tokens


def generate_turn_output(
    client: openai.OpenAI,
    prompt_input,
    model: str,
    max_tokens: int,
    prompt_tokens: int,
    max_model_len: int,
) -> str:
    """Generate full assistant response for a turn (feeds into next turn's history)."""
    # Leave 64 tokens of headroom to avoid hitting the context limit exactly.
    effective = min(max_tokens, max_model_len - prompt_tokens - 64)
    if effective <= 0:
        return ""
    if _is_messages(prompt_input):
        resp = client.chat.completions.create(
            model=model, messages=prompt_input, max_tokens=effective, temperature=0.0
        )
        return resp.choices[0].message.content or ""
    else:
        resp = client.completions.create(
            model=model, prompt=prompt_input, max_tokens=effective, temperature=0.0
        )
        return resp.choices[0].text or ""


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="/mnt/g/agentcache/models/Llama-3.2-1B-Instruct")
    p.add_argument("--data",            default=str(_EXP / "data"      / "python_agent_eval.jsonl"))
    p.add_argument("--system-prompt",   default=str(_EXP / "prompts"   / "2000_python_agent_system.txt"))
    p.add_argument("--centroid-k",      default=str(_EXP / "centroids" / "N64_K.npy"))
    p.add_argument("--centroid-v",      default=str(_EXP / "centroids" / "N64_V.npy"))
    p.add_argument("--synthetic-len",   type=int,   default=64)
    p.add_argument("--mode",            choices=["cold", "warm_apc", "synthetic"], required=True)
    p.add_argument("--out",             default=str(_EXP / "results" / "multi_turn_benchmark.jsonl"))
    p.add_argument("--conversation-file", default=None,
                   help="JSON file with a list of user prompts forming one coherent conversation. "
                        "When set, --n-conversations and --turns-per-conv are ignored.")
    p.add_argument("--n-conversations", type=int,  default=5)
    p.add_argument("--turns-per-conv",  type=int,  default=5)
    p.add_argument("--max-tokens",      type=int,  default=1024)
    p.add_argument("--gpu-mem",         type=float, default=0.6)
    p.add_argument("--server-port",     type=int,   default=8000)
    p.add_argument("--max-model-len",   type=int,   default=16384)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.mode == "synthetic":
        if not os.path.exists(args.centroid_k):
            raise FileNotFoundError(f"Centroid K not found: {args.centroid_k!r}")
        if not os.path.exists(args.centroid_v):
            raise FileNotFoundError(f"Centroid V not found: {args.centroid_v!r}")

    enable_apc = args.mode in ("warm_apc", "synthetic")
    env = build_centroid_env(args) if args.mode == "synthetic" else os.environ.copy()
    cmd = build_server_cmd(args, enable_apc)

    system_text = Path(args.system_prompt).read_text().strip()

    tokenizer = None
    if args.mode == "synthetic":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)

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
            print(f"WARNING: only {len(tasks)} tasks, need {needed}. Truncating.")
            n_convs = len(tasks) // turns
        conversations = [
            [{"id": tasks[conv_id * turns + t]["id"], "user": tasks[conv_id * turns + t]["user"]}
             for t in range(turns)]
            for conv_id in range(n_convs)
        ]

    n_convs   = len(conversations)
    turns_per = len(conversations[0])
    print(f"\n=== mode={args.mode}  N={args.synthetic_len}  convs={n_convs}  turns_per_conv={turns_per} ===\n")

    print(f"Starting vllm serve on port {args.server_port}...")
    proc = start_server(cmd, env, args.server_port)
    print("Server ready.\n")

    try:
        client = openai.OpenAI(
            base_url=f"http://localhost:{args.server_port}/v1",
            api_key="EMPTY",
        )

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "a") as fout:
            for conv_id, conv_tasks in enumerate(conversations):
                history: list[tuple[str, str]] = []

                for turn_idx, task in enumerate(conv_tasks):
                    user_text = task["user"]
                    turn_num  = turn_idx + 1

                    if args.mode == "synthetic":
                        prompt_input = build_compression_ids(
                            tokenizer, history, user_text, args.synthetic_len
                        )
                        centroid_tokens_saved = args.synthetic_len
                    else:
                        prompt_input = build_messages(system_text, history, user_text)
                        centroid_tokens_saved = 0

                    before_hits, before_queries = _read_prefix_cache_counters(args.server_port)
                    ttft_s, apc_cached_tokens, prompt_tokens = measure_turn_ttft_and_cached(
                        client, prompt_input, args.model
                    )
                    after_hits, after_queries = _read_prefix_cache_counters(args.server_port)

                    turn_hits    = after_hits - before_hits
                    turn_queries = after_queries - before_queries
                    kv_hit_rate  = turn_hits / turn_queries if turn_queries > 0 else 0.0

                    assistant_text = generate_turn_output(
                        client, prompt_input, args.model, args.max_tokens,
                        prompt_tokens, args.max_model_len
                    )
                    history.append((user_text, assistant_text))

                    record = {
                        "conversation_id":      conv_id,
                        "turn":                 turn_num,
                        "task_id":              task["id"],
                        "mode":                 args.mode,
                        "N":                    args.synthetic_len,
                        "physical_prompt_tokens": prompt_tokens,
                        "centroid_tokens_saved":  centroid_tokens_saved,
                        "apc_cached_tokens":      apc_cached_tokens,
                        "kv_cache_hits":          turn_hits,
                        "kv_cache_queries":       turn_queries,
                        "kv_cache_hit_rate":      round(kv_hit_rate, 4),
                        "ttft_s":                 round(ttft_s, 5),
                        "user":                   user_text,
                        "response":               assistant_text,
                    }
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()

                    print(
                        f"  conv={conv_id} turn={turn_num} {task['id']}  "
                        f"ttft={ttft_s:.4f}s  phys_tokens={prompt_tokens}  "
                        f"kv_hits={turn_hits:.0f}/{turn_queries:.0f}  "
                        f"kv_hit_rate={kv_hit_rate:.1%}"
                    )

        print(f"\nResults appended to {out_path}")

    finally:
        print("Stopping vllm serve...")
        stop_server(proc)


if __name__ == "__main__":
    main()
