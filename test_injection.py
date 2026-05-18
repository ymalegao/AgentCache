"""
Standalone centroid injection test — verifies injection is correct and fast.

Pure PEFT mode: virtual prefix tokens injected at positions 0..N-1.
Gap = VLLM_CENTROID_SYS_TOKENS + centroid_len (typically 0 + N).

  python test_injection.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL = "/mnt/g/agentcache/models/Llama-3.2-1B-Instruct"
CENTROID_K_PATH = "/home/yash/agentcache/centroid_K.npy"
CENTROID_V_PATH = "/home/yash/agentcache/centroid_V.npy"

# Abort if too few prompt tokens are left after scheduler gap.
MIN_TOKENS_AFTER_GAP = int(os.environ.get("CENTROID_TEST_MIN_TOKENS_AFTER_GAP", "32"))

os.environ.setdefault("VLLM_CENTROID_USE_LMCACHE", "0")
os.environ.setdefault("VLLM_CENTROID_K_PATH", CENTROID_K_PATH)
os.environ.setdefault("VLLM_CENTROID_V_PATH", CENTROID_V_PATH)
# SYS_TOKENS default: centroid starts at position 0.
os.environ.setdefault("VLLM_CENTROID_SYS_TOKENS", "0")
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
    "5. Test edge cases to ensure your fix is robust."
)
TEST = "Write a Python context manager that times how long a code block takes to execute."


def _build_prompt() -> tuple[str, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TEST}]
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids["input_ids"]
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    return prompt, len(prompt_ids)


PROMPT, PROMPT_TOKEN_LEN = _build_prompt()


def print_config():
    print("══ Centroid Injection Diagnostic ══")
    k_path = os.environ["VLLM_CENTROID_K_PATH"]
    K = np.load(k_path)
    centroid_len = K.shape[1] if K.ndim == 3 else 1
    print(f"  centroid K: {K.shape}  centroid_len={centroid_len}")

    sidecar = Path(k_path).with_name("sys_prefix_num_tokens.txt")
    sidecar_val = int(sidecar.read_text().strip()) if sidecar.is_file() else None
    sys_tokens_env = os.environ.get("VLLM_CENTROID_SYS_TOKENS")
    sys_token_count = int(sys_tokens_env) if sys_tokens_env else (sidecar_val or 0)
    total_gap = sys_token_count + centroid_len
    tokens_after_gap = PROMPT_TOKEN_LEN - total_gap

    print(f"  centroid K path:                 {k_path}")
    print(f"  sys_prefix_num_tokens (sidecar): {sidecar_val}")
    print(f"  VLLM_CENTROID_SYS_TOKENS:        {sys_tokens_env or '(not set, using sidecar)'}")
    print(f"  prompt tokens:                   {PROMPT_TOKEN_LEN}")
    print(f"  → sys_token_count={sys_token_count}  centroid_len={centroid_len}")
    print(f"  → scheduler gap will be {total_gap} tokens (pure PEFT, no sys_K)")
    print(f"  → estimated tokens left after gap: {tokens_after_gap}")

    return sys_token_count, centroid_len, total_gap, tokens_after_gap


def print_claude_ttft_debug(
    *,
    cold_mean: float,
    inj_mean: float,
    cold_times: list[float],
    inj_times: list[float],
    total_gap: int,
) -> None:
    env_keys = sorted(
        k
        for k in os.environ
        if "CENTROID" in k or k.startswith("VLLM_CENTROID") or k == "VLLM_CENTROID_SCHEDULER"
    )
    env_subset = {k: os.environ[k] for k in env_keys}
    print("\n══ Claude TTFT / perf debug ══")
    print(f"  PROMPT chars: {len(PROMPT)}")
    print(f"  PROMPT tokens: {PROMPT_TOKEN_LEN}")
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
    _sys_token_count, _centroid_len, total_gap, tokens_after_gap = print_config()

    if tokens_after_gap < MIN_TOKENS_AFTER_GAP:
        print(
            "\n  ERROR: prompt is too short for this synthetic gap.\n"
            f"  prompt_tokens={PROMPT_TOKEN_LEN}, gap={total_gap}, tokens_after_gap={tokens_after_gap}, "
            f"required_min={MIN_TOKENS_AFTER_GAP}\n"
            "  Retrain with fewer virtual tokens, lower VLLM_CENTROID_SYS_TOKENS, or use a longer prompt."
        )
        sys.exit(2)

    if os.environ.get("CENTROID_DEBUG_ROPE") == "1" or os.environ.get("CENTROID_DEBUG") == "1":
        print(
            "\n  Note: CENTROID_DEBUG_ROPE / CENTROID_DEBUG add heavy logging and GPU syncs "
            "per forward — unset them for fair cold vs inject TTFT comparison.\n"
        )

    from vllm import LLM, SamplingParams
    import vllm.centroid_integration as ci

    k_path = os.environ["VLLM_CENTROID_K_PATH"]
    v_path = os.environ["VLLM_CENTROID_V_PATH"]

    # ── Cold start (no injection — hide K files so injector never loads) ─────
    print("\n══ Cold Start ══")
    ci._centroid_sched_enabled = None
    os.environ["VLLM_CENTROID_SCHEDULER"] = "0"

    K_bak, V_bak = k_path + ".bak", v_path + ".bak"
    for src, dst in [(k_path, K_bak), (v_path, V_bak)]:
        if os.path.exists(src):
            os.rename(src, dst)

    llm_cold = LLM(model=MODEL, gpu_memory_utilization=0.6, enable_prefix_caching=False)
    llm_cold.generate([PROMPT], SamplingParams(max_tokens=1))

    cold_mean, cold_times = ttft(llm_cold, PROMPT)
    cold_output = llm_cold.generate(
        [PROMPT], SamplingParams(temperature=0, max_tokens=80)
    )[0].outputs[0].text
    print(f"  Times: {[f'{t:.4f}' for t in cold_times]}  mean={cold_mean:.4f}s")
    print(f"  Output: {cold_output!r}")
    del llm_cold

    for src, dst in [(K_bak, k_path), (V_bak, v_path)]:
        if os.path.exists(src):
            os.rename(src, dst)

    # ── Centroid injection ────────────────────────────────────────────────────
    print("\n══ Centroid Injection ══")
    ci._centroid_sched_enabled = None
    os.environ["VLLM_CENTROID_SCHEDULER"] = "1"

    llm_inj = LLM(model=MODEL, gpu_memory_utilization=0.6, enable_prefix_caching=False)
    llm_inj.generate([PROMPT], SamplingParams(max_tokens=1))

    inj_mean, inj_times = ttft(llm_inj, PROMPT)
    inj_output = llm_inj.generate(
        [PROMPT], SamplingParams(temperature=0, max_tokens=80)
    )[0].outputs[0].text
    print(f"  Times: {[f'{t:.4f}' for t in inj_times]}  mean={inj_mean:.4f}s")
    print(f"  Output: {inj_output!r}")
    del llm_inj

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n══ Summary ══")
    print(f"  Cold TTFT:   {cold_mean:.4f}s")
    print(f"  Inject TTFT: {inj_mean:.4f}s")
    speedup = cold_mean / inj_mean
    verdict = (
        "FASTER ✓"
        if speedup > 1.05
        else ("SLOWER — overhead > savings" if speedup < 0.95 else "roughly equal")
    )
    print(f"  Speedup: {speedup:.2f}x  ({verdict})")

    def _is_coherent(text: str) -> bool:
        if len(text) < 10:
            return False
        words = text.split()
        if len(words) < 3:
            return False
        alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
        return alpha_words / len(words) >= 0.5

    coherent = _is_coherent(inj_output)
    print(f"  Output: {'coherent ✓' if coherent else 'GARBLED ✗ — injection still wrong'}")

    if not coherent:
        print(
            "\n  Debug hint: scheduler gap was",
            total_gap,
            "— check that all positions 0..",
            total_gap - 1,
            "have KV injected.",
        )

    print_claude_ttft_debug(
        cold_mean=cold_mean,
        inj_mean=inj_mean,
        cold_times=cold_times,
        inj_times=inj_times,
        total_gap=total_gap,
    )


if __name__ == "__main__":
    main()
