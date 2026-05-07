import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/g/agentcache/hf_cache"

MODEL_ID = "/mnt/g/agentcache/models/qwen-1.5b"  # local path, no download

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./collection_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer.

Please solve the issue provided by the user. You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust"""

with open("tasks.json", "r") as f:
    tasks_data = json.load(f)
TASKS = [item["perturbed_task"] for item in tasks_data]

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Token count for `<|im_start|>system ... <|im_end|>` — must match vLLM prompts.
_sys_prefix_only = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"
_sys_n = len(tokenizer(_sys_prefix_only, return_tensors="pt").input_ids[0])
with open(f"{OUTPUT_DIR}/sys_prefix_num_tokens.txt", "w") as f:
    f.write(str(_sys_n))
print(f"Wrote {OUTPUT_DIR}/sys_prefix_num_tokens.txt → {_sys_n} (vLLM centroid skip length)")

print("Loading model (fp16, this will take ~30s)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
)
model.eval()

num_hidden_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size

# Per-layer hidden tensors aligned with model.model.layers[i]: HF returns
# hidden_states[0] = embeddings (input to layer 0), ...,
# hidden_states[num_hidden_layers] = output after last layer — we keep [0 : L].
print(f"\nModel config: {num_hidden_layers} transformer layers, hidden dim {hidden_dim}")

# ── Welford accumulators (per boundary × layer × dim) ───────────────────────
mean_A = np.zeros((num_hidden_layers, hidden_dim), dtype=np.float32)
mean_B = np.zeros((num_hidden_layers, hidden_dim), dtype=np.float32)
M2_A = np.zeros((num_hidden_layers, hidden_dim), dtype=np.float32)
M2_B = np.zeros((num_hidden_layers, hidden_dim), dtype=np.float32)
count = 0

run_metadata = []


def chat_prompt(task: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{task}<|im_end|>\n"
    )


def boundary_positions(task: str, full_ids: torch.Tensor) -> tuple[int, int]:
    """
    Positions in the tokenized *full* prompt (causal positions).

    A — last token of the system message (system prompt fully encoded; no user yet).
    B — last token after the full user turn (system + user encoded; 'finished reading
        this coding task' in the residual at that timestep).
    """
    prefix_a = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"
    prefix_b = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{task}<|im_end|>"
    )

    ids_a = tokenizer(prefix_a, return_tensors="pt").input_ids[0]
    ids_b = tokenizer(prefix_b, return_tensors="pt").input_ids[0]
    # Compare token IDs on CPU (full_ids may be CUDA from .cuda()).
    full = full_ids[0].detach().cpu()

    pos_a = int(ids_a.shape[0]) - 1
    pos_b = int(ids_b.shape[0]) - 1

    if not torch.equal(full[: ids_a.shape[0]], ids_a):
        raise RuntimeError(
            "Tokenization mismatch for system prefix vs full prompt — "
            "boundary A may be wrong. Check chat template / special tokens."
        )
    if not torch.equal(full[: ids_b.shape[0]], ids_b):
        raise RuntimeError(
            "Tokenization mismatch for prefix through user vs full prompt — "
            "boundary B may be wrong."
        )

    return pos_a, pos_b


def extract_layer_vectors(hidden_states: tuple, pos: int) -> np.ndarray:
    """Stack inputs to each transformer layer at `pos`: shape [L, H]."""
    vecs = []
    for layer_idx in range(num_hidden_layers):
        vecs.append(hidden_states[layer_idx][0, pos, :].float())
    return torch.stack(vecs).cpu().numpy()


# ── Collection loop ───────────────────────────────────────────────────────────
for run_idx, task in enumerate(TASKS):
    print(f"\n── Run {run_idx + 1}/{len(TASKS)} ──")

    full_text = chat_prompt(task)
    tokens = tokenizer(full_text, return_tensors="pt").input_ids.cuda()
    pos_a, pos_b = boundary_positions(task, tokens)

    with torch.no_grad():
        outputs = model(input_ids=tokens, output_hidden_states=True)

    hs = outputs.hidden_states
    vec_A = extract_layer_vectors(hs, pos_a)
    vec_B = extract_layer_vectors(hs, pos_b)

    count += 1
    n = count
    d_a = vec_A - mean_A
    mean_A += d_a / n
    d2_a = vec_A - mean_A
    M2_A += d_a * d2_a
    d_b = vec_B - mean_B
    mean_B += d_b / n
    d2_b = vec_B - mean_B
    M2_B += d_b * d2_b

    run_metadata.append(
        {
            "run_id": run_idx,
            "pos_boundary_A": pos_a,
            "pos_boundary_B": pos_b,
            "seq_len": int(tokens.shape[1]),
        }
    )

    torch.cuda.empty_cache()

# ── Variance (sample variance, ddof=1) ─────────────────────────────────────
# Semantics match todo.md: for each layer L, stack hidden states from all tasks
# → shape [num_tasks, hidden_dim], take variance across tasks per dimension
# → mean across dims = scalar "how much layer L moves when the task changes".
# We use Welford online accumulation instead of materializing the full stack;
# variance_*[L, d] equals np.var(stack[:, L, d], axis=0, ddof=1) over tasks.
if count > 1:
    variance_A = M2_A / (count - 1)
    variance_B = M2_B / (count - 1)
else:
    variance_A = np.zeros_like(M2_A)
    variance_B = np.zeros_like(M2_B)

# Per-layer scalar: mean over hidden_dim of cross-task variance at boundary A / B
layer_variance_A = variance_A.mean(axis=1)
layer_variance_B = variance_B.mean(axis=1)

# Todo.md: normalize to injection weights (higher variance → more weight)
layer_weights_A = layer_variance_A / layer_variance_A.sum()
layer_weights_B = layer_variance_B / layer_variance_B.sum()

# Option B-style confidence (stability; high variance → low confidence)
conf_B = 1.0 - (layer_variance_B / (layer_variance_B.max() + 1e-12))

# ── Save outputs ────────────────────────────────────────────────────────────
np.save(f"{OUTPUT_DIR}/mean_A.npy", mean_A)
np.save(f"{OUTPUT_DIR}/mean_B.npy", mean_B)
np.save(f"{OUTPUT_DIR}/variance_A.npy", variance_A)
np.save(f"{OUTPUT_DIR}/variance_B.npy", variance_B)
np.save(f"{OUTPUT_DIR}/layer_variance_A.npy", layer_variance_A)
np.save(f"{OUTPUT_DIR}/layer_variance_B.npy", layer_variance_B)
# Alias for scripts / plots that follow todo.md naming (boundary B = after full user task).
np.save(f"{OUTPUT_DIR}/layer_variance.npy", layer_variance_B)
np.save(f"{OUTPUT_DIR}/layer_weights_A.npy", layer_weights_A)
np.save(f"{OUTPUT_DIR}/layer_weights_B.npy", layer_weights_B)
np.save(f"{OUTPUT_DIR}/layer_confidence_B.npy", conf_B.astype(np.float32))

with open(f"{OUTPUT_DIR}/run_metadata.jsonl", "w") as f:
    for record in run_metadata:
        f.write(json.dumps(record) + "\n")

print(f"\n✓ Collection complete. {count} runs.")
print(f"✓ mean_A / mean_B shape : {mean_A.shape}")
print(f"✓ layer_variance_B shape: {layer_variance_B.shape}")
print(
    f"✓ Saved layer_variance*.npy — per-layer cross-task spread "
    f"(mean over dims of var across tasks); layer_variance.npy == boundary B"
)
print(f"✓ Saved to              : {OUTPUT_DIR}/")

print("\n── Layer variance (B, mean |Δ|² across dims) — min / max ──")
print(
    f"  min layer {int(layer_variance_B.argmin())}: {layer_variance_B.min():.6f} | "
    f"max layer {int(layer_variance_B.argmax())}: {layer_variance_B.max():.6f}"
)
print(f"✓ mean_A[0] norm : {np.linalg.norm(mean_A[0]):.4f}")
print(f"✓ mean_B[0] norm : {np.linalg.norm(mean_B[0]):.4f}")
print(f"✓ A != B (layer 0): {not np.allclose(mean_A[0], mean_B[0])}")
