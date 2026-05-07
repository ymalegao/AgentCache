import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import os
from transformers import DynamicCache

MODEL_ID   = "/mnt/g/agentcache/models/qwen-1.5b"
CENTROID_DIR = "./centroid_output"

SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer.
Please solve the issue provided by the user. You can execute bash commands and 
edit files to implement the necessary changes."""

# Use a task NOT in your 20 collection tasks — held-out test
TEST_TASK = "Write a c++ context manager that times how long a code block takes to execute."

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda"
)
model.eval()

# ── Helper: format input ──────────────────────────────────────────────────────
def make_inputs(system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").to("cuda")

# ── Helper: build past_key_values from centroid ───────────────────────────────
def build_past_kv_from_centroid(centroid_K, centroid_V, num_heads, head_dim):
    """
    Build a DynamicCache from centroid arrays.
    DynamicCache is what transformers 5.x expects instead of tuple of tuples.
    """
    cache = DynamicCache()
    num_layers = centroid_K.shape[0]

    for layer_idx in range(num_layers):
        k = torch.tensor(centroid_K[layer_idx], dtype=torch.float16).to("cuda")
        v = torch.tensor(centroid_V[layer_idx], dtype=torch.float16).to("cuda")

        # Shape: [batch, num_kv_heads, seq_len, head_dim]
        k = k.view(1, num_heads, 1, head_dim)
        v = v.view(1, num_heads, 1, head_dim)

        cache.update(k, v, layer_idx)

    return cache

# ── Get model KV dimensions ───────────────────────────────────────────────────
num_kv_heads = model.config.num_key_value_heads
head_dim     = model.config.hidden_size // model.config.num_attention_heads
print(f"num_kv_heads: {num_kv_heads}, head_dim: {head_dim}")
print(f"kv_dim check: {num_kv_heads * head_dim} (should match centroid dim 256)")

# ── Load centroids ────────────────────────────────────────────────────────────
centroid_K = np.load(f"{CENTROID_DIR}/centroid_K_B.npy")  # (28, 256)
centroid_V = np.load(f"{CENTROID_DIR}/centroid_V_B.npy")  # (28, 256)

results = {}

# ── Condition 1: Cold Start ───────────────────────────────────────────────────
print("\n── Condition 1: Cold Start ──")
inputs = make_inputs(SYSTEM_PROMPT, TEST_TASK)
input_len = inputs["input_ids"].shape[1]

torch.cuda.synchronize()
t0 = time.perf_counter()

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

torch.cuda.synchronize()
t1 = time.perf_counter()

cold_output = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
cold_time   = t1 - t0
cold_tokens = out.shape[1] - input_len

print(f"  Input tokens : {input_len}")
print(f"  Output tokens: {cold_tokens}")
print(f"  Total time   : {cold_time:.3f}s")
print(f"  Output preview: {cold_output[:100]}")

results["cold_start"] = {
    "input_tokens": int(input_len),
    "output_tokens": int(cold_tokens),
    "time": cold_time,
    "output": cold_output
}

# ── Condition 2: Centroid Injection (full context, Boundary B) ────────────────
print("\n── Condition 2: Centroid Injection (Boundary B) ──")

# Build past_key_values from centroid
past_kv = build_past_kv_from_centroid(
    centroid_K, centroid_V, num_kv_heads, head_dim
)

# For injection we only pass the USER part of the input
# The system prompt is "already represented" by the centroid
user_only_messages = [{"role": "user", "content": TEST_TASK}]
user_text = tokenizer.apply_chat_template(
    user_only_messages, tokenize=False, add_generation_prompt=True
)
user_inputs = tokenizer(user_text, return_tensors="pt").to("cuda")
user_len = user_inputs["input_ids"].shape[1]

# Position ids must account for the virtual prefix token
position_ids = torch.arange(
    1,  # start at 1 because centroid occupies position 0
    user_len + 1,
    device="cuda"
).unsqueeze(0)

torch.cuda.synchronize()
t0 = time.perf_counter()

with torch.no_grad():
    out_injected = model.generate(
        input_ids=user_inputs["input_ids"],
        attention_mask=user_inputs["attention_mask"],
        position_ids=position_ids,
        past_key_values=past_kv,
        max_new_tokens=200,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

torch.cuda.synchronize()
t1 = time.perf_counter()

injected_output = tokenizer.decode(
    out_injected[0][user_len:], skip_special_tokens=True
)
injected_time   = t1 - t0
injected_tokens = out_injected.shape[1] - user_len

print(f"  Input tokens (user only): {user_len}")
print(f"  Skipped tokens (centroid): {input_len - user_len}")
print(f"  Output tokens: {injected_tokens}")
print(f"  Total time   : {injected_time:.3f}s")
print(f"  Output preview: {injected_output[:100]}")

results["centroid_injection"] = {
    "input_tokens": int(user_len),
    "skipped_tokens": int(input_len - user_len),
    "output_tokens": int(injected_tokens),
    "time": injected_time,
    "output": injected_output
}

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──")
speedup = cold_time / injected_time
print(f"  Cold start time      : {cold_time:.3f}s")
print(f"  Centroid inject time : {injected_time:.3f}s")
print(f"  Speedup              : {speedup:.2f}x")
print(f"  Tokens skipped       : {input_len - user_len} / {input_len} "
      f"({(input_len-user_len)/input_len*100:.1f}%)")

# Simple output similarity check
cold_words     = set(cold_output.lower().split())
injected_words = set(injected_output.lower().split())
overlap = len(cold_words & injected_words) / len(cold_words | injected_words)
print(f"  Output word overlap  : {overlap:.2%} (higher = outputs more similar)")

with open("./injection_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✓ Full outputs saved to injection_results.json")