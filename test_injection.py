"""
Standalone centroid injection test — verifies injection is correct and fast.

Pure PEFT mode: 64 virtual prefix tokens injected at positions 0..63.
No sys_extended_K.npy needed. Gap = SYS_TOKENS + centroid_len = 0+64 = 64.

  python test_injection.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL = "/mnt/g/agentcache/models/qwen-1.5b"
# Root-level files written by transpose_tensors.py
K_PATH = "/home/yash/agentcache/centroid_K.npy"
V_PATH = "/home/yash/agentcache/centroid_V.npy"

os.environ.setdefault("VLLM_CENTROID_K_PATH", K_PATH)
os.environ.setdefault("VLLM_CENTROID_V_PATH", V_PATH)
os.environ.setdefault("VLLM_CENTROID_USE_LMCACHE", "0")
# SYS_TOKENS=0: centroid fills from position 0, gap = 0 + centroid_len = 64
os.environ.setdefault("VLLM_CENTROID_SYS_TOKENS", "0")
# One-line engine logs for TTFT debugging (apply_pre + seed_post); disable with CENTROID_PERF_DEBUG=0
os.environ.setdefault("CENTROID_PERF_DEBUG", "1")

SYSTEM = (
    "You are a helpful assistant that can interact with a computer.\n"
    "Please solve the issue provided by the user. "
    "You can execute bash commands and edit files to implement the necessary changes.\n\n"
    "## Recommended Workflow\n"
    "1. Analyze the codebase by finding and reading relevant files\n"
    "2. Create a script to reproduce the issue\n"
    "3. Edit the source code to resolve the issue\n"
    "4. Verify your fix works by running your script again\n"
    "5. Test edge cases to ensure your fix is robust.\n\n"
    + "This is an extended system prompt to simulate a larger context. " * 50
)
TEST = "Write a Python context manager that times how long a code block takes to execute."
PROMPT = (
    f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    f"<|im_start|>user\n{TEST}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)


def print_config():
    print("══ Centroid Injection Diagnostic ══")
    K = np.load(K_PATH)
    centroid_len = K.shape[1] if K.ndim == 3 else 1
    print(f"  centroid K: {K.shape}  centroid_len={centroid_len}")

    sidecar = Path(K_PATH).with_name("sys_prefix_num_tokens.txt")
    sidecar_val = int(sidecar.read_text().strip()) if sidecar.is_file() else None
    sys_tokens_env = os.environ.get("VLLM_CENTROID_SYS_TOKENS")
    sys_token_count = int(sys_tokens_env) if sys_tokens_env else (sidecar_val or 0)
    total_gap = sys_token_count + centroid_len  # pure PEFT: no sys_K

    print(f"  sys_prefix_num_tokens (sidecar): {sidecar_val}")
    print(f"  VLLM_CENTROID_SYS_TOKENS:        {sys_tokens_env or '(not set, using sidecar)'}")
    print(f"  → sys_token_count={sys_token_count}  centroid_len={centroid_len}")
    print(f"  → scheduler gap will be {total_gap} tokens (pure PEFT, no sys_K)")
    print(f"  → model processes positions {total_gap}..end (user query)")

    return sys_token_count, centroid_len, total_gap


def print_claude_ttft_debug(
    *,
    cold_mean: float,
    inj_mean: float,
    cold_times: list[float],
    inj_times: list[float],
    total_gap: int,
) -> None:
    """Prints a compact snapshot for logs / another model to interpret cold vs inject TTFT."""
    env_keys = sorted(
        k
        for k in os.environ
        if "CENTROID" in k or k.startswith("VLLM_CENTROID") or k == "VLLM_CENTROID_SCHEDULER"
    )
    env_subset = {k: os.environ[k] for k in env_keys}
    print("\n══ Claude TTFT / perf debug ══")
    print(f"  PROMPT chars: {len(PROMPT)}")
    print(f"  scheduler synthetic gap (tokens): {total_gap}")
    print(f"  cold_ttft_mean_s:  {cold_mean:.4f}  trials: {[f'{t:.4f}' for t in cold_times]}")
    print(f"  inject_ttft_mean_s: {inj_mean:.4f}  trials: {[f'{t:.4f}' for t in inj_times]}")
    print(f"  ratio cold/inject: {cold_mean / inj_mean:.3f}x")
    print("  Note: cold and inject use two separate LLM(...) engine lifetimes (two model loads).")
    print("  Env (centroid-related):", env_subset or "(none)")
    print(
        "  Engine log grep:  grep -E '\\[CENTROID PERF]|\\[CENTROID TIMING]|\\[CENTROID] ' logfile\n"
        "    apply_pre: n_scheduled_tokens + pre_seed_skip_all_seeded (fast path after seed).\n"
        "    seed_post: wrote_any + req_ids (new id each generate() => seed runs again)."
    )


def ttft(llm, prompt, n=3):
    from vllm import SamplingParams
    params = SamplingParams(temperature=0, max_tokens=1)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        llm.generate([prompt], params)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), times


def main():
    sys_token_count, centroid_len, total_gap = print_config()

    if os.environ.get("CENTROID_DEBUG_ROPE") == "1" or os.environ.get("CENTROID_DEBUG") == "1":
        print(
            "\n  Note: CENTROID_DEBUG_ROPE / CENTROID_DEBUG add heavy logging and GPU syncs "
            "per forward — unset them for fair cold vs inject TTFT comparison.\n"
        )

    from vllm import LLM, SamplingParams
    import vllm.centroid_integration as ci

    # ── Cold start (no injection — hide K files so injector never loads) ─────
    print("\n══ Cold Start ══")
    ci._centroid_sched_enabled = None
    os.environ["VLLM_CENTROID_SCHEDULER"] = "0"

    K_bak, V_bak = K_PATH + ".bak", V_PATH + ".bak"
    for src, dst in [(K_PATH, K_bak), (V_PATH, V_bak)]:
        if os.path.exists(src):
            os.rename(src, dst)

    llm_cold = LLM(model=MODEL, gpu_memory_utilization=0.6, enable_prefix_caching=False)
    llm_cold.generate([PROMPT], SamplingParams(max_tokens=1))  # warmup

    cold_mean, cold_times = ttft(llm_cold, PROMPT)
    cold_output = llm_cold.generate([PROMPT], SamplingParams(temperature=0, max_tokens=80))[0].outputs[0].text
    print(f"  Times: {[f'{t:.4f}' for t in cold_times]}  mean={cold_mean:.4f}s")
    print(f"  Output: {cold_output!r}")
    del llm_cold

    for src, dst in [(K_bak, K_PATH), (V_bak, V_PATH)]:
        if os.path.exists(src):
            os.rename(src, dst)

    # ── Centroid injection ────────────────────────────────────────────────────
    print("\n══ Centroid Injection ══")
    ci._centroid_sched_enabled = None
    os.environ["VLLM_CENTROID_SCHEDULER"] = "1"

    llm_inj = LLM(model=MODEL, gpu_memory_utilization=0.6, enable_prefix_caching=False)
    llm_inj.generate([PROMPT], SamplingParams(max_tokens=1))  # warmup

    inj_mean, inj_times = ttft(llm_inj, PROMPT)
    inj_output = llm_inj.generate([PROMPT], SamplingParams(temperature=0, max_tokens=80))[0].outputs[0].text
    print(f"  Times: {[f'{t:.4f}' for t in inj_times]}  mean={inj_mean:.4f}s")
    print(f"  Output: {inj_output!r}")
    del llm_inj

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n══ Summary ══")
    print(f"  Cold TTFT:   {cold_mean:.4f}s")
    print(f"  Inject TTFT: {inj_mean:.4f}s")
    speedup = cold_mean / inj_mean
    verdict = "FASTER ✓" if speedup > 1.05 else ("SLOWER — overhead > savings" if speedup < 0.95 else "roughly equal")
    print(f"  Speedup: {speedup:.2f}x  ({verdict})")

    garble_tokens = ["]={", "initWithFrame", "\x00"]
    coherent = len(inj_output) > 10 and not any(t in inj_output for t in garble_tokens)
    print(f"  Output: {'coherent ✓' if coherent else 'GARBLED ✗ — injection still wrong'}")

    if not coherent:
        print("\n  Debug hint: scheduler gap was", total_gap,
              "— check that all positions 0..", total_gap - 1, "have KV injected.")

    print_claude_ttft_debug(
        cold_mean=cold_mean,
        inj_mean=inj_mean,
        cold_times=cold_times,
        inj_times=inj_times,
        total_gap=total_gap,
    )


if __name__ == "__main__":
    main()
