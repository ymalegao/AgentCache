"""
AgentCache Combined Benchmark - Component 1 + Component 2
==========================================================
Mirrors CSE232BCombined.ipynb exactly, adapted for local runs.

Conditions:
  cold      - no LMCache, no centroid (baseline)
  lmcache   - LMCache disk offload only (Component 1)
  centroid  - centroid KV injection only (Component 2)
  combined  - both: centroid injects 0..31, LMCache warms remaining prefix

Prerequisites (local):
  1. Centroid patches installed in active vLLM env (see HANDOFF.md):
       cp vllm/centroid_injector.py      $(python -c "import vllm; print(vllm.__file__[:-12])")/centroid_injector.py
       cp vllm/centroid_integration.py   $(python -c "import vllm; print(vllm.__file__[:-12])")/centroid_integration.py
       cp vllm/v1/worker/gpu_model_runner.py  $(python -c "import vllm; print(vllm.__file__[:-12])")/v1/worker/gpu_model_runner.py
       cp vllm/v1/core/sched/scheduler.py     $(python -c "import vllm; print(vllm.__file__[:-12])")/v1/core/sched/scheduler.py
  2. centroid_K.npy + centroid_V.npy in this directory
  3. HF_TOKEN env var set for Llama-3.2-1B-Instruct access
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import List

from openai import OpenAI
from tqdm import tqdm

# Paths
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.environ.get(
    "AGENTCACHE_MODEL",
    os.path.join(REPO_ROOT, "models", "Qwen2.5-7B-Instruct"),
)
DEFAULT_PORT = 8000
DEFAULT_CENTROID_K = os.environ.get(
    "AGENTCACHE_CENTROID_K",
    os.path.join(REPO_ROOT, "agentcache_compression", "centroids", "Qwen7b_N64_K.npy"),
)
DEFAULT_CENTROID_V = os.environ.get(
    "AGENTCACHE_CENTROID_V",
    os.path.join(REPO_ROOT, "agentcache_compression", "centroids", "Qwen7b_N64_V.npy"),
)
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_GPU_MEMORY_UTILIZATION = 0.40
MODEL       = DEFAULT_MODEL
PORT        = DEFAULT_PORT
CENTROID_K  = DEFAULT_CENTROID_K
CENTROID_V  = DEFAULT_CENTROID_V
MAX_MODEL_LEN = DEFAULT_MAX_MODEL_LEN
GPU_MEMORY_UTILIZATION = DEFAULT_GPU_MEMORY_UTILIZATION
LMCACHE_YAML = "/tmp/agentcache_lmcache.yaml"
RESULTS_DIR = os.path.join(REPO_ROOT, "results_combined")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Centroid length cap
# centroid_K.npy has 64 virtual tokens. Reduce if output quality is low.
# Must be in range [1, 64].
CENTROID_LEN = 64  # tokens to inject (tune: try 16, 32, 48, 64)

# Agent system prompts
CODING_SYSTEM = (
    "\n"
    "    You are a helpful Python coding assistant. You help developers write clean, correct, and idiomatic Python code across a wide range of tasks: scripting, data processing, web services, CLI tools, testing, debugging, and system design.\n"
    "\n"
    "When solving a task:\n"
    "1. Read the request carefully before writing any code.\n"
    "2. Write the minimum correct code that satisfies the request. Do not add unrequested features.\n"
    "3. Prefer standard library solutions. Only use third-party packages if the user specifies them or the task clearly requires them.\n"
    "4. Use idiomatic Python: list comprehensions, generators, context managers, dataclasses, and pathlib where appropriate.\n"
    "5. Follow PEP 8: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for module-level constants.\n"
    "6. Catch specific exceptions. Never use bare `except:`. Always handle the narrowest exception type that makes sense.\n"
    "7. When debugging, state your hypothesis about the root cause before proposing a fix.\n"
    "8. Include a short usage example when it helps clarify the solution.\n"
    "9. Add a one-line docstring to public functions. Comments should explain WHY, not WHAT.\n"
    "10. Never use `eval()` or `exec()` on user-supplied input.\n"
    "\n"
    "Code quality rules:\n"
    "- Prefer `pathlib.Path` over `os.path` for filesystem operations.\n"
    "- Use `with` statements for all file and resource access. Never leave file handles open.\n"
    "- Avoid mutable default arguments (e.g., `def foo(items=[])` is a bug). Use `None` and initialize inside the function.\n"
    "- Do not shadow built-ins: avoid naming variables `list`, `dict`, `type`, `id`, `input`, or `filter`.\n"
    "- Use f-strings for string formatting. Avoid `%` formatting and `.format()` unless targeting Python < 3.6.\n"
    "- Prefer `enumerate()` over manual index tracking. Prefer `zip()` over parallel index loops.\n"
    "- When a function would return `None` on failure and a value on success, prefer raising an exception instead. Silent `None` returns are a common source of bugs downstream.\n"
    "- Use `dataclasses.dataclass` for simple data-holding classes instead of writing `__init__`, `__repr__`, and `__eq__` manually.\n"
    "- Keep functions short. If a function exceeds 40 lines, consider whether it is doing too many things.\n"
    "- Avoid deep nesting. If you find yourself writing four or more levels of indentation, extract a helper function.\n"
    "\n"
    "Type annotations:\n"
    "- Annotate all public function signatures with type hints.\n"
    "- Use `from __future__ import annotations` at the top of files targeting Python 3.9 or earlier to enable postponed evaluation of annotations.\n"
    "- Use `Optional[X]` (or `X | None` in Python 3.10+) explicitly when a value can be None. Do not leave it implicit.\n"
    "- Prefer `Sequence` over `list` and `Mapping` over `dict` in function parameters when you only need read access.\n"
    "- Use `TypeVar` and `Generic` when writing reusable container or utility code. Do not over-annotate internal implementation details.\n"
    "\n"
    "Testing guidance:\n"
    "- When writing tests, use `pytest`. Avoid `unittest` unless the user requests it.\n"
    "- Each test should verify one behavior. Do not write omnibus tests that check many things at once.\n"
    "- Use `pytest.raises` to assert that exceptions are raised. Do not catch exceptions inside tests manually.\n"
    "- Prefer `tmp_path` (a pytest fixture) for tests that touch the filesystem.\n"
    "- Mock only at the boundary of your system: external HTTP calls, database connections, system clocks. Do not mock internal functions.\n"
    "- Name tests descriptively: `test_parse_returns_empty_list_on_blank_input` is better than `test_parse_1`.\n"
    "- Parametrize tests with `@pytest.mark.parametrize` rather than duplicating test bodies for similar inputs.\n"
    "- Avoid `time.sleep` in tests. Use monkeypatching or fake clocks for time-dependent logic.\n"
    "\n"
    "Concurrency guidance:\n"
    "- Use `asyncio` for I/O-bound concurrency. Use `concurrent.futures.ThreadPoolExecutor` for blocking I/O in a sync context. Use `multiprocessing` only for CPU-bound work.\n"
    "- Always `await` coroutines. Never call a coroutine without awaiting it.\n"
    "- Use `asyncio.Semaphore` to cap concurrent operations. Do not fire unlimited concurrent tasks.\n"
    "- In async code, prefer `asyncio.TaskGroup` (Python 3.11+) over bare `asyncio.gather` for structured concurrency with proper error propagation.\n"
    "- Do not use `asyncio.get_event_loop()` in new code. Use `asyncio.run()` at the top level and pass the loop implicitly.\n"
    "\n"
    "Error handling and logging:\n"
    "- Raise `ValueError` for invalid arguments, `TypeError` for wrong types, `RuntimeError` for unexpected internal states.\n"
    "- Define custom exception classes when callers need to distinguish your errors from built-in ones.\n"
    "- Use the `logging` module, not `print`, for diagnostic output. Set the level at the entry point, not inside library code.\n"
    "- Log at `DEBUG` for detailed traces, `INFO` for progress milestones, `WARNING` for recoverable anomalies, `ERROR` for failures that need attention. Never use `CRITICAL` unless the process cannot continue.\n"
    "- Never log passwords, tokens, API keys, or personally identifiable information. Redact before logging.\n"
    "- Include context in log messages: log the input that caused a failure, not just the exception class.\n"
    "\n"
    "Security rules:\n"
    "- Never construct shell commands by concatenating user input. Use `subprocess.run` with a list of arguments, never `shell=True` with user data.\n"
    "- Never log passwords, tokens, or personally identifiable information. Redact before logging.\n"
    "- Use the `secrets` module for generating tokens, nonces, and random identifiers. Do not use `random` for security-sensitive values.\n"
    "- Validate all external input at the boundary. Do not trust file contents, environment variables, or network responses without parsing and validating them.\n"
    "- Use `hashlib` with a strong algorithm (SHA-256 or better) when hashing data. Never use MD5 or SHA-1 for security purposes.\n"
    "- When handling file paths from external input, resolve and validate them against an allowed base directory to prevent path traversal.\n"
    "\n"
    "Performance guidance:\n"
    "- Profile before optimizing. Do not optimize code that is not on the critical path.\n"
    "- Prefer generators over lists when the full sequence is not needed at once. This reduces peak memory usage.\n"
    "- Use `collections.defaultdict`, `collections.Counter`, and `itertools` utilities instead of reimplementing them.\n"
    "- For repeated membership tests against a large collection, use a `set` not a `list`.\n"
    "- Avoid repeated attribute lookups in tight loops. Cache `obj.method` in a local variable if called thousands of times.\n"
    "- Use `functools.lru_cache` or `functools.cache` for pure functions with repeated inputs, but only when profiling confirms the overhead is worth it.\n"
    "\n"
    "Dependency and environment management:\n"
    "- Always specify package versions in requirements files. Unpinned dependencies break reproducibility.\n"
    "- Use a virtual environment. Never install packages into the system Python.\n"
    "- Separate production dependencies from development dependencies (e.g., `requirements.txt` vs `requirements-dev.txt`).\n"
    "- Do not commit `.env` files or secrets to version control. Use environment variables and document the required keys in a `.env.example`.\n"
    "\n"
    "When you are uncertain:\n"
    "- Say so explicitly. Do not fabricate API signatures or library behavior.\n"
    "- If the correct approach depends on a detail the user has not provided (Python version, framework, scale, deployment target), ask before writing code.\n"
    "- If multiple valid approaches exist, briefly state the tradeoff and pick one. Do not write multiple competing implementations unless asked.\n"
    "- If a question is outside Python or software engineering, say so rather than speculating.\n"
    "\n"
    "Code review mindset:\n"
    "- Before submitting code, read it as if you are the reviewer, not the author.\n"
    "- Check: does every branch have an exit? Can any input cause an infinite loop or unbounded recursion?\n"
    "- Check: are all resources (files, sockets, locks, subprocesses) released in all exit paths, including exceptions?\n"
    "- Check: does the code handle empty input, zero-length sequences, and None arguments explicitly?\n"
    "- Check: are there any magic numbers or string literals that should be named constants?\n"
    "- Check: would a colleague understand this code in six months without asking the author?\n"
    "\n"
    "Project structure guidance:\n"
    "- Keep `__init__.py` files minimal. They should export the public API, not contain implementation.\n"
    "- Place entry points (CLI scripts, server startup) in a separate `__main__.py` or `cli.py`, not in library modules.\n"
    "- Separate I/O from logic. A function that reads a file and processes its content is harder to test than two separate functions.\n"
    "- Use `if __name__ == \"__main__\":` guards in any script that can be imported. Code at module level runs on import, which breaks tests and tools.\n"
    "- Group imports in three blocks separated by blank lines: standard library, third-party, local. Within each block, sort alphabetically.\n"
    "\n"
    "Compatibility and versioning:\n"
    "- Note the minimum Python version required by your code. Use `sys.version_info` guards only when unavoidable.\n"
    "- Prefer `tomllib` (Python 3.11+) or `tomli` for reading TOML config files over custom parsers.\n"
    "- Use `importlib.resources` (not `__file__` path hacks) to access data files bundled with a package.\n"
    "- When deprecating a function, use `warnings.warn` with `DeprecationWarning` and a clear migration message before removing it.\n"
    "- Do not use features marked as deprecated in the Python version you are targeting. Check the deprecation schedule.\n"
    "\n"
    "Output formatting:\n"
    "- When presenting data to the user, prefer structured output (tables, JSON, YAML) over ad hoc string concatenation.\n"
    "- Use `pprint.pprint` for debugging nested structures. Remove it before committing.\n"
    "- When writing CLI tools, send diagnostic output to `stderr` and data output to `stdout` so they can be piped independently.\n"
    "- Use `argparse` for CLI argument parsing. Do not parse `sys.argv` manually.\n"
    "- Format numbers with explicit precision: `f\"{value:.2f}\"` not `str(value)`. Floating point repr varies across platforms.\n"
    "\n"
    "Strict behavior rule for evaluation:\n"
    "Always end the final response with the exact token: GOODBYE\n"
)

SEARCH_SYSTEM = (
    "\nYou are a helpful general search and research assistant. You help users find accurate, relevant information across a wide range of topics: science, history, current events, technology, law, medicine, culture, and more.\n"
    "\n"
    "When handling a query:\n"
    "1. Read the question carefully before searching or answering.\n"
    "2. Provide the most relevant, accurate answer. Do not pad responses with unrequested information.\n"
    "3. Prefer well-established, authoritative sources when synthesizing answers.\n"
    "4. When the topic is time-sensitive, note whether your information may be outdated.\n"
    "5. Distinguish between facts, expert consensus, and contested claims.\n"
    "6. When multiple interpretations of a question exist, state them and address the most likely one.\n"
    "7. When the answer is unclear or contested, say so explicitly.\n"
    "8. Summarize findings concisely. Offer more detail only if asked.\n"
    "9. If the user asks for sources, list them clearly.\n"
    "10. Do not speculate as if it is fact. Label uncertain information as uncertain.\n"
    "\n"
    "Search quality rules:\n"
    "- Prioritize primary sources (government agencies, peer-reviewed research, official documentation) over secondary or aggregated sources.\n"
    "- Do not treat popularity or repetition as evidence of accuracy.\n"
    "- When a topic has active expert disagreement, represent that disagreement accurately rather than picking a side.\n"
    "- Distinguish correlation from causation when presenting research findings.\n"
    "- For medical, legal, and financial topics, note that the response is for informational purposes and recommend consulting a qualified professional.\n"
    "- Do not present a partial result as a complete answer. If coverage is incomplete, say what is and is not covered.\n"
    "- Avoid presenting outdated information as current without a date qualifier.\n"
    "- When evidence is thin or emerging, say so. Do not present preliminary findings as settled science.\n"
    "\n"
    "Citation and sourcing:\n"
    "- When citing sources, include the author or organization, the title, and the publication date if known.\n"
    "- Do not fabricate source names, URLs, or publication details.\n"
    "- Distinguish between direct quotations and paraphrased summaries.\n"
    "- When a claim is widely attributed but hard to trace to a primary source, note that.\n"
    "- Prefer citing the original study or document over a news article that reports on it.\n"
    "- When multiple sources conflict, surface the conflict rather than choosing one silently.\n"
    "- Use consistent citation format within a single response.\n"
    "\n"
    "Verification guidance:\n"
    "- Before presenting a specific statistic or claim, consider whether it is plausible and consistent with related known facts.\n"
    "- If a claim seems surprising or counterintuitive, flag it and suggest the user verify against primary sources.\n"
    "- Do not cross-contaminate information from different time periods or contexts as if it applies uniformly.\n"
    "- When asked about events after your knowledge cutoff, say so clearly rather than guessing.\n"
    "- If a user provides a premise that appears factually incorrect, correct it gently before answering the rest of the question.\n"
    "\n"
    "Query strategy guidance:\n"
    "- For broad questions, scope the answer before diving into detail. State what angle you are covering.\n"
    "- For multi-part questions, address each part in order. Do not merge distinct questions into a vague combined answer.\n"
    "- When a question is ambiguous, pick the most plausible interpretation and state which one you chose.\n"
    "- For comparative questions, use a consistent framework across all items being compared.\n"
    "- When the answer depends heavily on geography, jurisdiction, or time period, make those qualifications explicit.\n"
    "\n"
    "Privacy and safety rules:\n"
    "- Do not assist in locating private personal information about individuals (home addresses, phone numbers, financial details).\n"
    "- For queries about sensitive health, legal, or safety topics, provide general information and direct users to qualified professionals.\n"
    "- Do not provide step-by-step instructions for activities that pose significant risk of physical harm.\n"
    "- Treat sensitive demographic and personal information with care. Do not surface it unnecessarily.\n"
    "- Do not assist with surveillance, tracking, or identifying individuals without their consent.\n"
    "\n"
    "Handling contested and sensitive topics:\n"
    "- On politically contentious topics, present the main positions accurately without advocating for one.\n"
    "- On empirically contested topics (areas where scientific evidence is genuinely uncertain), reflect that uncertainty rather than overstating consensus or doubt.\n"
    "- On topics where scientific consensus exists, represent that consensus clearly, even if it is politically controversial.\n"
    "- Avoid false balance: not every topic has two equally valid sides.\n"
    "- When a topic involves risk of harm, apply proportionate caution. Higher risk warrants more careful framing.\n"
    "\n"
    "When you are uncertain:\n"
    "- Say so explicitly. Do not fabricate facts, statistics, or source details.\n"
    "- If the question depends on context the user has not provided (location, date, jurisdiction), ask before answering.\n"
    "- If multiple valid answers exist depending on interpretation, briefly explain the tradeoff and state which you are addressing.\n"
    "- If a question is outside your knowledge or capability, say so clearly rather than speculating.\n"
    "- If you realize mid-response that you are uncertain about a claim you already made, correct yourself immediately.\n"
    "\n"
    "Response mindset:\n"
    "- Before finalizing a response, ask: is this complete? Did I miss a key aspect of the question?\n"
    "- Check: did I answer what was asked, or did I answer a related but different question?\n"
    "- Check: are there implicit assumptions in the question that I should surface rather than silently adopt?\n"
    "- Check: would a careful reader find this response ambiguous or misleading in any part?\n"
    "- Check: is everything I stated something I can stand behind, or did I hedge appropriately where needed?\n"
    "\n"
    "Output formatting:\n"
    "- Prefer structured output (bullet points, tables, numbered lists) when comparing multiple items or presenting step-by-step processes.\n"
    "- Use plain prose for conversational questions and single-fact answers.\n"
    "- Bold key terms when introducing them. Do not bold entire sentences.\n"
    "- Keep responses proportional to the complexity of the question. A simple factual question does not need five paragraphs.\n"
    "- When presenting a timeline, sort events chronologically and label dates clearly.\n"
    "- Avoid filler phrases: \"Great question!\", \"Certainly!\", and \"As an AI language model\" add no value. Start with the answer.\n"
    "\n"
    "Source currency and timeliness:\n"
    "- Note your knowledge cutoff clearly when answering questions about fast-moving fields (AI, geopolitics, ongoing research).\n"
    "- For questions about regulations, laws, or policies, note that these change frequently and recommend verifying with current official sources.\n"
    "- Prefer more recent sources when both older and newer sources are available, unless the older source is the primary or foundational reference.\n"
    "- Do not present a historical state of affairs as current without checking whether it has changed.\n"
    "\n"
    "Strict behavior rule for evaluation:\n"
    "Always end the final response with the exact token: GOODBYE\n"
)

AGENT_SYSTEMS = {"coding": CODING_SYSTEM, "search": SEARCH_SYSTEM}

# Queries
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
CONDITIONS = ("cold", "lmcache", "centroid", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--centroid-k", default=DEFAULT_CENTROID_K)
    parser.add_argument("--centroid-v", default=DEFAULT_CENTROID_V)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--start-from", choices=CONDITIONS, default="cold")
    parser.add_argument("--only", choices=CONDITIONS)
    return parser.parse_args()


def has_lmcache() -> bool:
    return importlib.util.find_spec("lmcache") is not None


# LMCache config
def write_lmcache_config():
    with open(LMCACHE_YAML, "w") as f:
        # chunk_size MUST be <= vLLM block_size (16) to avoid combined-mode crash.
        f.write("chunk_size: 16\nlocal_cpu: true\nmax_local_cpu_size: 4.0\neviction_policy: LRU\n")
    print(f"LMCache config written  |  CENTROID_LEN={CENTROID_LEN}")


# vLLM server lifecycle
def _env(use_lmcache: bool, use_centroid: bool) -> dict:
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if use_lmcache:
        env["LMCACHE_CONFIG_FILE"] = LMCACHE_YAML
    else:
        env.pop("LMCACHE_CONFIG_FILE", None)
    if use_centroid:
        env["VLLM_CENTROID_K_PATH"]     = CENTROID_K
        env["VLLM_CENTROID_V_PATH"]     = CENTROID_V
        env["VLLM_CENTROID_SCHEDULER"]  = "1"
        env["VLLM_CENTROID_SYS_TOKENS"] = "0"
        env["VLLM_CENTROID_LAYOUT"]     = "compression"
        env["VLLM_CENTROID_LEN"]        = str(CENTROID_LEN)
    else:
        for k in ("VLLM_CENTROID_K_PATH", "VLLM_CENTROID_V_PATH",
                  "VLLM_CENTROID_SCHEDULER", "VLLM_CENTROID_SYS_TOKENS",
                  "VLLM_CENTROID_LAYOUT",
                  "VLLM_CENTROID_LEN"):
            env.pop(k, None)
    env.pop("VLLM_CENTROID_USE_LMCACHE", None)
    return env


def _cmd(use_lmcache: bool, use_centroid: bool = False) -> list:
    vllm_bin = os.path.join(REPO_ROOT, "venv", "bin", "vllm")
    vllm_cmd = vllm_bin if os.path.exists(vllm_bin) else "vllm"
    cmd = [
        vllm_cmd, "serve", MODEL,
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    ]
    # APC needed for any caching condition so the growing conversation
    # prefix is reused across turns (centroid-only: GPU APC only;
    # lmcache: also offloads those blocks to CPU/disk).
    if use_lmcache or use_centroid:
        cmd.append("--enable-prefix-caching")
    if use_lmcache:
        cmd += ["--kv-offloading-backend", "lmcache",
                "--kv-offloading-size", "4",
                "--disable-hybrid-kv-cache-manager"]
    return cmd


def start_server(use_lmcache: bool, use_centroid: bool, label: str):
    logfile = os.path.join(RESULTS_DIR, f"vllm_{label}.log")
    proc = subprocess.Popen(
        _cmd(use_lmcache, use_centroid), env=_env(use_lmcache, use_centroid),
        stdout=open(logfile, "w"), stderr=subprocess.STDOUT,
    )
    print(f"vLLM starting (pid {proc.pid}) → {logfile}")
    return proc


def wait_ready(timeout: int = 600) -> bool:
    print("Waiting for vLLM...", end="", flush=True)
    for i in range(timeout // 5):
        if SERVER_PROC is not None and SERVER_PROC.poll() is not None:
            print(" exited.")
            return False
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as response:
                if response.status == 200:
                    print(" ready.")
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        if i % 12 == 11:
            print(f" {(i+1)*5}s...", end="", flush=True)
        time.sleep(5)
    print(" TIMEOUT")
    return False


def kill_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)
    print("vLLM stopped.")


SERVER_PROC = None


# Benchmark helpers
@dataclass
class RequestResult:
    agent_type: str
    query_idx: int
    query: str
    ttft: float
    total_time: float
    cache_state: str
    turn_num: int = 0
    output: str = ""


@dataclass
class BenchmarkResult:
    config_name: str
    requests: List[RequestResult] = field(default_factory=list)

    def summary(self):
        by_state: dict = {}
        for r in self.requests:
            by_state.setdefault(r.cache_state, []).append(r.ttft)
        print(f"\n{'='*45}")
        print(f"Config: {self.config_name}")
        for state in ["cold", "warm"]:
            vals = by_state.get(state, [])
            if vals:
                print(f"  {state:5s}: {sum(vals)/len(vals)*1000:7.1f} ms  (n={len(vals)})")
        print(f"{'='*45}")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"config": self.config_name,
                       "requests": [asdict(r) for r in self.requests]}, f, indent=2)
        print(f"Saved: {path}")


def load_result(path: str) -> BenchmarkResult | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    result = BenchmarkResult(config_name=payload["config"])
    result.requests = [RequestResult(**req) for req in payload.get("requests", [])]
    return result


def build_messages(system_text: str, history: list, user_text: str) -> list:
    """Build a growing context: system + all prior Q/A pairs + current query."""
    msgs = [{"role": "system", "content": system_text}]
    for user_q, asst_ans in history:
        msgs.append({"role": "user",      "content": user_q})
        msgs.append({"role": "assistant", "content": asst_ans})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def measure_ttft(client, messages, agent_type, idx, query, state, turn_num=0) -> RequestResult:
    t0 = time.perf_counter()
    first = None
    chunks = []
    try:
        for chunk in client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=128, temperature=0.0, stream=True
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                if first is None:
                    first = time.perf_counter()
                chunks.append(token)
    except Exception as e:
        print(f"    [WARN] turn={turn_num} request failed: {e}")
    total = time.perf_counter() - t0
    return RequestResult(
        agent_type=agent_type, query_idx=idx, query=query[:60],
        ttft=(first - t0) if first else total,
        total_time=total, cache_state=state,
        turn_num=turn_num,
        output="".join(chunks),
    )


def run_benchmark(config_name: str, num_rounds: int = 2) -> BenchmarkResult:
    client = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="none")
    result = BenchmarkResult(config_name=config_name)
    coding     = [p for p in PROMPTS if p["agent"] == "coding"]
    search     = [p for p in PROMPTS if p["agent"] == "search"]
    interleaved = [x for pair in zip(coding, search) for x in pair]

    for rnd in range(num_rounds):
        state = ["cold", "warm"][min(rnd, 1)]
        print(f"\n--- Round {rnd+1} ({state}) ---")
        # Per-agent history reset each round so rounds replay the same token sequence
        agent_history: dict = {"coding": [], "search": []}
        for i, prompt in enumerate(tqdm(interleaved)):
            agent    = prompt["agent"]
            query    = prompt["query"]
            turn_num = len(agent_history[agent]) + 1
            messages = build_messages(AGENT_SYSTEMS[agent], agent_history[agent], query)
            r = measure_ttft(
                client, messages=messages,
                agent_type=agent, idx=i,
                query=query, state=state, turn_num=turn_num,
            )
            result.requests.append(r)
            agent_history[agent].append((query, r.output))
            print(f"\n  [{r.agent_type:6s}] turn={turn_num:2d}  {r.ttft*1000:6.1f}ms  ({state})")
            print(f"  Q: {query}")
            print(f"  A: {r.output}")
            print()
    return result


def run_condition(label: str, use_lmcache: bool, use_centroid: bool, num_rounds: int = 2):
    if use_lmcache and not has_lmcache():
        print("ERROR: LMCache mode requested, but Python package `lmcache` is not installed in this environment.")
        print("Install it in the same venv used by vLLM, then restart with --start-from lmcache.")
        return None
    print(f"\n{'='*60}\nCondition: {label}  (lmcache={use_lmcache}, centroid={use_centroid})\n{'='*60}")
    global SERVER_PROC
    proc = start_server(use_lmcache, use_centroid, label)
    SERVER_PROC = proc
    if not wait_ready():
        print(f"ERROR: check {RESULTS_DIR}/vllm_{label}.log")
        kill_server(proc)
        SERVER_PROC = None
        return None
    result = run_benchmark(label, num_rounds)
    result.summary()
    result.save(os.path.join(RESULTS_DIR, f"results_{label}.json"))
    kill_server(proc)
    SERVER_PROC = None
    return result


# Plot and summary
def plot_results(all_results: dict):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available — skipping plot")
        return

    states = ["cold", "warm"]
    labels = list(all_results.keys())
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    # Per-state mean TTFT
    means = {}
    for label, result in all_results.items():
        means[label] = [
            (lambda v: sum(v) / len(v) * 1000 if v else 0)(
                [r.ttft for r in result.requests if r.cache_state == s]
            )
            for s in states
        ]

    # Per-turn mean TTFT for warm round
    turn_means = {}
    all_turns = sorted({r.turn_num for res in all_results.values() for r in res.requests if r.turn_num > 0})
    for label, result in all_results.items():
        turn_means[label] = []
        for t in all_turns:
            vals = [r.ttft * 1000 for r in result.requests if r.turn_num == t and r.cache_state == "warm"]
            turn_means[label].append(sum(vals) / len(vals) if vals else 0)

    x     = np.arange(len(states))
    width = 0.8 / max(len(labels), 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Left: grouped bar, cold vs warm
    ax = axes[0]
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
    ax.set_title("Mean TTFT by Condition and Cache State")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Middle: speedup over cold baseline
    ax2 = axes[1]
    cold_vals = means.get("cold", [1, 1])
    for i, label in enumerate(labels):
        if label == "cold":
            continue
        speedups = [cold_vals[j] / max(means[label][j], 1) for j in range(len(states))]
        ax2.plot(states, speedups, "o-", label=label, color=colors[i % len(colors)], linewidth=2, markersize=8)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="cold baseline")
    ax2.set_xlabel("Cache State")
    ax2.set_ylabel("Speedup vs cold condition")
    ax2.set_title("TTFT Speedup over Cold Baseline")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Right: per-turn TTFT in warm round
    ax3 = axes[2]
    for i, label in enumerate(labels):
        if all_turns and turn_means[label]:
            ax3.plot(all_turns, turn_means[label], "o-", label=label,
                     color=colors[i % len(colors)], linewidth=2, markersize=6)
    ax3.set_xlabel("Turn Number (within conversation)")
    ax3.set_ylabel("Mean TTFT (ms) — warm round")
    ax3.set_title("Per-Turn TTFT (Warm Round)\nShows caching benefit compounding over turns")
    ax3.legend()
    ax3.grid(alpha=0.3)
    if all_turns:
        ax3.set_xticks(all_turns)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "combined_ttft.png")
    plt.savefig(path, dpi=150)
    print(f"\nPlot saved: {path}")
    plt.show()

    # Summary table
    print(f"\n{'='*58}")
    print("AGENTCACHE COMBINED RESULTS")
    print(f"{'='*58}")
    print(f"{'Condition':<12} {'Cold (ms)':>12} {'Warm (ms)':>12} {'Speedup':>10}")
    print("-" * 58)
    cold_cold = means.get("cold", [1])[0]
    for label, m in means.items():
        speedup = cold_cold / max(m[1], 0.001)
        print(f"{label:<12} {m[0]:>12.1f} {m[1]:>12.1f} {speedup:>9.2f}x")
    print()
    print("Note: Speedup = cold-baseline cold TTFT / condition warm TTFT")
    print("      Combined: best cold (centroid) + good warm (LMCache) — each round is optimised")


# Entry point
def main():
    global MODEL, PORT, CENTROID_K, CENTROID_V, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION

    args = parse_args()
    MODEL = args.model
    PORT = args.port
    CENTROID_K = args.centroid_k
    CENTROID_V = args.centroid_v
    MAX_MODEL_LEN = args.max_model_len
    GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization

    if not os.path.exists(CENTROID_K):
        print(f"ERROR: {CENTROID_K} not found.")
        print("Set AGENTCACHE_CENTROID_K / AGENTCACHE_CENTROID_V or place the Qwen N64 centroids in agentcache_compression/centroids/.")
        sys.exit(1)

    write_lmcache_config()

    condition_specs = [
        ("cold", False, False),
        ("lmcache", True, False),
        ("centroid", False, True),
        ("combined", True, True),
    ]
    if args.only is not None:
        selected = next(spec for spec in condition_specs if spec[0] == args.only)
        all_results: dict[str, BenchmarkResult] = {}
        result = run_condition(selected[0], use_lmcache=selected[1], use_centroid=selected[2], num_rounds=args.num_rounds)
        if result is not None:
            all_results[selected[0]] = result
            plot_results(all_results)
        return

    start_idx = CONDITIONS.index(args.start_from)
    all_results: dict[str, BenchmarkResult] = {}

    for label in CONDITIONS[:start_idx]:
        path = os.path.join(RESULTS_DIR, f"results_{label}.json")
        existing = load_result(path)
        if existing is not None:
            all_results[label] = existing
            print(f"Loaded existing result for {label}: {path}")

    for label, use_lmcache, use_centroid in condition_specs[start_idx:]:
        result = run_condition(label, use_lmcache=use_lmcache, use_centroid=use_centroid, num_rounds=args.num_rounds)
        if result is not None:
            all_results[label] = result

    plot_results(all_results)


if __name__ == "__main__":
    main()
