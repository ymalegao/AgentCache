"""
AgentCache Combined Benchmark — Component 1 + Component 2
==========================================================
Conditions:
  cold      — no LMCache, no centroid (baseline)
  lmcache   — LMCache disk offload only (Component 1)
  centroid  — centroid KV injection only (Component 2)
  combined  — both: centroid injects 0..63, LMCache warms remaining prefix

Prerequisites (local):
  1. Centroid patches installed in active vLLM env (see HANDOFF.md):
       cp vllm/centroid_injector.py      $(python -c "import vllm; print(vllm.__file__[:-12])")/centroid_injector.py
       cp vllm/centroid_integration.py   $(python -c "import vllm; print(vllm.__file__[:-12])")/centroid_integration.py
       cp vllm/v1/worker/gpu_model_runner.py  $(python -c "import vllm; print(vllm.__file__[:-12])")/v1/worker/gpu_model_runner.py
       cp vllm/v1/core/sched/scheduler.py     $(python -c "import vllm; print(vllm.__file__[:-12])")/v1/core/sched/scheduler.py
  2. centroid_K.npy + centroid_V.npy in this directory
     (run: python transpose_tensors.py --adapter agentcache_prefix_model --sys-tokens 0)
  3. HF_TOKEN env var set for Llama-3.2-1B-Instruct access

Note on combined mode with sys_tokens=0:
  Centroid occupies KV slots 0..63 (virtual domain prior).
  LMCache restores system prompt blocks on warm/hot; centroid then overwrites 0..63.
  For system prompts longer than 64 tokens, LMCache provides additional benefit on
  positions 64..sys_len for warm/hot requests. For the ~55-token system prompts used
  here, the combined benefit over centroid-only is modest on warm/hot.
  Use a longer system prompt (e.g. EXTENDED_SYSTEM below) to show compounding savings.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List

import requests
from openai import OpenAI
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}/v1"
CENTROID_K = os.path.join(REPO_ROOT, "centroid_K.npy")
CENTROID_V = os.path.join(REPO_ROOT, "centroid_V.npy")
LMCACHE_YAML = "/tmp/agentcache_lmcache.yaml"
KV_STORE = "/tmp/agentcache_kv_store"
RESULTS_DIR = os.path.join(REPO_ROOT, "results_combined")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(KV_STORE, exist_ok=True)

# ── Agent definitions ──────────────────────────────────────────────────────────
CODING_SYSTEM = (
    "You are an expert software engineer specializing in Python, "
    "algorithms, and system design. You debug code, write clean "
    "implementations, and explain technical concepts precisely. "
    "Always reason step by step before writing any code."
)
SEARCH_SYSTEM = (
    "You are a knowledgeable research assistant. You answer factual "
    "questions clearly and concisely, drawing on your knowledge of "
    "science, technology, history, and current events. Always reason "
    "through your answer and indicate your confidence level."
)
AGENT_SYSTEMS = {"coding": CODING_SYSTEM, "search": SEARCH_SYSTEM}

# Longer system prompt to amplify combined savings (sys_len >> 64)
EXTENDED_SYSTEM = (
    CODING_SYSTEM
    + "\n\n"
    + ("This is an extended system prompt to simulate a larger agent context. " * 8)
)

CODING_QUERIES = [
    "Implement a thread-safe LRU cache in Python with O(1) get and put.",
    "Write a Python context manager that retries a block up to N times on exception.",
    "Design a rate limiter class using the token bucket algorithm.",
    "Implement a trie for autocomplete with insert and search methods.",
    "Write a decorator that caches function results with a TTL.",
    "Optimize this O(n^2) solution to find pairs summing to a target value.",
    "Implement consistent hashing for a distributed cache.",
    "Write a generator that streams large CSV files without loading into memory.",
    "Implement BFS and DFS on a graph represented as an adjacency list.",
    "Design a simple pub/sub event system in Python.",
]
SEARCH_QUERIES = [
    "What are the main architectural differences between transformers and Mamba SSMs?",
    "Explain how RLHF differs from DPO in LLM fine-tuning.",
    "What is the current state of quantum computing for practical applications?",
    "Summarize the key ideas behind retrieval-augmented generation.",
    "How does PagedAttention improve GPU memory efficiency in LLM serving?",
    "What were the main contributions of the Attention is All You Need paper?",
    "Explain the difference between KV cache quantization and weight quantization.",
    "What is speculative decoding and how does it speed up inference?",
    "How does FlashAttention reduce memory usage compared to standard attention?",
    "What are the tradeoffs between beam search and sampling for text generation?",
]
PROMPTS = (
    [{"agent": "coding", "query": q} for q in CODING_QUERIES]
    + [{"agent": "search", "query": q} for q in SEARCH_QUERIES]
)


# ── LMCache config ─────────────────────────────────────────────────────────────
def write_lmcache_config():
    with open(LMCACHE_YAML, "w") as f:
        # chunk_size must equal vLLM block_size (16). With chunk_size=256,
        # LMCache's save slot_mapping spans positions 96-255 for a 93-token
        # prompt; those positions map to NULL_BLOCK_ID columns → CUDA crash.
        f.write(
            "chunk_size: 16\n"
            "local_cpu: true\n"
            "max_local_cpu_size: 4.0\n"
            "eviction_policy: LRU\n"
        )


# ── vLLM server lifecycle ──────────────────────────────────────────────────────
def _build_env(use_lmcache: bool, use_centroid: bool) -> dict:
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if use_lmcache:
        env["LMCACHE_CONFIG_FILE"] = LMCACHE_YAML
    else:
        env.pop("LMCACHE_CONFIG_FILE", None)
    if use_centroid:
        env["VLLM_CENTROID_K_PATH"] = CENTROID_K
        env["VLLM_CENTROID_V_PATH"] = CENTROID_V
        env["VLLM_CENTROID_SCHEDULER"] = "1"
        env["VLLM_CENTROID_SYS_TOKENS"] = "0"  # pure PEFT: centroid at positions 0..N-1
    else:
        for k in ("VLLM_CENTROID_K_PATH", "VLLM_CENTROID_V_PATH",
                  "VLLM_CENTROID_SCHEDULER", "VLLM_CENTROID_SYS_TOKENS"):
            env.pop(k, None)
    # With VLLM_CENTROID_USE_LMCACHE=0 (default), centroid writes 0..63 on every
    # request; LMCache may restore the same blocks on warm/hot, then centroid
    # overwrites them — net benefit compounds when sys_prompt_len > 64.
    env.pop("VLLM_CENTROID_USE_LMCACHE", None)
    return env


def _build_cmd(use_lmcache: bool) -> list:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(PORT),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
    ]
    if use_lmcache:
        cmd += [
            "--enable-prefix-caching",
            "--kv-offloading-backend", "lmcache",
            "--kv-offloading-size", "4",
            "--disable-hybrid-kv-cache-manager",
        ]
    return cmd


def start_vllm(use_lmcache: bool, use_centroid: bool, label: str):
    logfile = os.path.join(RESULTS_DIR, f"vllm_{label}.log")
    proc = subprocess.Popen(
        _build_cmd(use_lmcache),
        env=_build_env(use_lmcache, use_centroid),
        stdout=open(logfile, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"  vLLM pid={proc.pid}  log={logfile}")
    return proc


def wait_ready(timeout=300) -> bool:
    for _ in range(timeout // 5):
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def kill_vllm(proc):
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(2)


# ── Benchmark ──────────────────────────────────────────────────────────────────
@dataclass
class RequestResult:
    agent_type: str
    query_idx: int
    query: str
    ttft: float
    total_time: float
    cache_state: str


@dataclass
class BenchmarkResult:
    config_name: str
    requests: List[RequestResult] = field(default_factory=list)

    def summary(self):
        by_state: dict = {}
        for r in self.requests:
            by_state.setdefault(r.cache_state, []).append(r.ttft)
        print(f"\n{'='*50}")
        print(f"  Config: {self.config_name}")
        for state in ["cold", "warm", "hot"]:
            vals = by_state.get(state, [])
            if vals:
                print(f"  {state:5s}: {sum(vals)/len(vals)*1000:7.1f} ms  (n={len(vals)})")
        print(f"{'='*50}")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {"config": self.config_name, "requests": [asdict(r) for r in self.requests]},
                f, indent=2,
            )
        print(f"  Saved: {path}")


def _measure(client, messages, agent_type, idx, query, state) -> RequestResult:
    t0 = time.perf_counter()
    first = None
    for chunk in client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=256, stream=True
    ):
        if first is None and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter()
    total = time.perf_counter() - t0
    return RequestResult(
        agent_type=agent_type, query_idx=idx, query=query[:60],
        ttft=(first - t0) if first else total,
        total_time=total, cache_state=state,
    )


def run_benchmark(config_name: str, num_rounds: int = 3) -> BenchmarkResult:
    client = OpenAI(base_url=BASE_URL, api_key="none")
    result = BenchmarkResult(config_name=config_name)
    coding = [p for p in PROMPTS if p["agent"] == "coding"]
    search = [p for p in PROMPTS if p["agent"] == "search"]
    interleaved = [x for pair in zip(coding, search) for x in pair]

    for rnd in range(num_rounds):
        state = ["cold", "warm", "hot"][min(rnd, 2)]
        print(f"  Round {rnd+1} ({state})")
        for i, prompt in enumerate(tqdm(interleaved, leave=False)):
            r = _measure(
                client,
                messages=[
                    {"role": "system", "content": AGENT_SYSTEMS[prompt["agent"]]},
                    {"role": "user",   "content": prompt["query"]},
                ],
                agent_type=prompt["agent"], idx=i,
                query=prompt["query"], state=state,
            )
            result.requests.append(r)
            print(f"    [{r.agent_type:6s}] {r.ttft*1000:6.1f}ms  {prompt['query'][:50]}")
    return result


# ── Per-condition runner ───────────────────────────────────────────────────────
def run_condition(label: str, use_lmcache: bool, use_centroid: bool, num_rounds: int = 3):
    print(f"\n{'='*60}")
    print(f"Condition: {label}  (lmcache={use_lmcache}, centroid={use_centroid})")
    print(f"{'='*60}")
    proc = start_vllm(use_lmcache, use_centroid, label)
    if not wait_ready():
        print("  ERROR: vLLM did not start in time.")
        kill_vllm(proc)
        return None
    print("  vLLM ready.")
    result = run_benchmark(label, num_rounds)
    result.summary()
    result.save(os.path.join(RESULTS_DIR, f"results_{label}.json"))
    kill_vllm(proc)
    return result


# ── Summary table + plot ───────────────────────────────────────────────────────
def print_summary(all_results: dict):
    print("\n\n" + "=" * 65)
    print("AGENTCACHE COMBINED BENCHMARK — SUMMARY")
    print("=" * 65)
    print(f"{'Condition':<12} {'Cold (ms)':>12} {'Warm (ms)':>12} {'Hot (ms)':>12}")
    print("-" * 65)
    for label, result in all_results.items():
        by_state: dict = {}
        for r in result.requests:
            by_state.setdefault(r.cache_state, []).append(r.ttft * 1000)
        row = {s: sum(v) / len(v) if v else 0 for s, v in by_state.items()}
        print(
            f"{label:<12} {row.get('cold',0):>12.1f} {row.get('warm',0):>12.1f} {row.get('hot',0):>12.1f}"
        )


def plot_results(all_results: dict):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    states = ["cold", "warm", "hot"]
    labels = list(all_results.keys())
    means = {
        label: [
            sum(r.ttft for r in result.requests if r.cache_state == s) * 1000
            / max(1, sum(1 for r in result.requests if r.cache_state == s))
            for s in states
        ]
        for label, result in all_results.items()
    }

    x = np.arange(len(states))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for i, label in enumerate(labels):
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means[label], width, label=label, color=colors[i % len(colors)])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h:.0f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_xlabel("Cache State")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_title("AgentCache: Cold / Warm / Hot TTFT by Condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "combined_ttft.png")
    plt.savefig(path, dpi=150)
    print(f"\nPlot saved: {path}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(CENTROID_K):
        print(f"ERROR: {CENTROID_K} not found.")
        print("Run: python transpose_tensors.py --adapter agentcache_prefix_model --sys-tokens 0")
        sys.exit(1)

    write_lmcache_config()

    conditions = [
        ("cold",     False, False),
        ("lmcache",  True,  False),
        ("centroid", False, True),
        ("combined", True,  True),
    ]

    all_results = {}
    for label, use_lm, use_cen in conditions:
        r = run_condition(label, use_lm, use_cen)
        if r is not None:
            all_results[label] = r

    print_summary(all_results)
    plot_results(all_results)


if __name__ == "__main__":
    main()
