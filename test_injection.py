from vllm import LLM, SamplingParams
import os
import time


def measure_ttft(llm, prompt, n_runs=5):
    """Measure time to FIRST token only."""
    times = []
    for _ in range(n_runs):
        params = SamplingParams(
            temperature=0,
            max_tokens=1,  # prefill + one decode step
        )
        start = time.perf_counter()
        llm.generate([prompt], params)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return sum(times) / len(times), times


def generate_sample(llm, prompt):
    """Generate a longer sample to check output quality."""
    params = SamplingParams(temperature=0, max_tokens=100)
    out = llm.generate([prompt], params)
    return out[0].outputs[0].text


def main():
    SYSTEM = """You are a helpful assistant that can interact with a computer.
Please solve the issue provided by the user. You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust.

""" + ("This is an extended system prompt to simulate a larger context. " * 50)

    TEST = (
        "Write a Python context manager that times how long a code block takes to execute."
    )
    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{TEST}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


    K_path = "/home/yash/agentcache/attention_centroid_output/centroid_K.npy"
    V_path = "/home/yash/agentcache/attention_centroid_output/centroid_V.npy"
    K_hidden = "/home/yash/agentcache/attention_centroid_output/centroid_K.npy.bak"
    V_hidden = "/home/yash/agentcache/attention_centroid_output/centroid_V.npy.bak"




    # ── Condition 3: Centroid injection (same full prompt) ───────────────────
    print("\n══ Condition 3: Centroid Injection ══")

    if os.path.exists(K_hidden):
        os.rename(K_hidden, K_path)
    if os.path.exists(V_hidden):
        os.rename(V_hidden, V_path)


    import vllm.centroid_integration as ci

    ci._centroid_sched_enabled = None
    os.environ["VLLM_CENTROID_SCHEDULER"] = "1"

    llm_inject = LLM(
        model="/mnt/g/agentcache/models/qwen-1.5b",
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
    )
    _ = llm_inject.generate([prompt], SamplingParams(max_tokens=1))
    inject_mean, inject_times = measure_ttft(llm_inject, prompt)
    print(f"  Inject mean TTFT: {inject_mean:.4f}s")
    
    print("\n  -- Generating sample output (Centroid Injection) --")
    inject_output = generate_sample(llm_inject, prompt)
    
    del llm_inject

    print("\n══ Summary ══")
    
    print(f"  Centroid injection TTFT  : {inject_mean:.4f}s")
  
  
    print("\n══ Sanity Check ══")
    print(f"Inject Start Output:\n{inject_output}\n")
    print("-" * 40)

    
  


if __name__ == "__main__":
    main()
