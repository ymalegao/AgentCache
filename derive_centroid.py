import torch
import numpy as np
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID   = "/mnt/g/agentcache/models/qwen-1.5b"
INPUT_DIR  = "./collection_output"
OUTPUT_DIR = "./centroid_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variance filter threshold — keep bottom X% most stable dimensions
STABLE_PERCENTILE = 75  # zero out top 25% highest variance dims

# ── Load means and variances ──────────────────────────────────────────────────
mean_A     = np.load(f"{INPUT_DIR}/mean_A.npy")      # (28, 1536)
mean_B     = np.load(f"{INPUT_DIR}/mean_B.npy")      # (28, 1536)
variance_A = np.load(f"{INPUT_DIR}/variance_A.npy")  # (28, 1536)
variance_B = np.load(f"{INPUT_DIR}/variance_B.npy")  # (28, 1536)

num_layers, hidden_dim = mean_B.shape
print(f"Loaded: {num_layers} layers, hidden dim {hidden_dim}")

# ── Variance analysis before filtering ───────────────────────────────────────
print("\n── Variance Analysis (Boundary B) ──")
for layer_idx in [0, num_layers//4, num_layers//2, num_layers-1]:
    v = variance_B[layer_idx]
    print(f"  Layer {layer_idx:02d} | mean var: {v.mean():.6f} | "
          f"max var: {v.max():.6f} | "
          f"% stable: {(v < np.percentile(v, STABLE_PERCENTILE))[:].mean()*100:.1f}%")

# ── Filter stable dimensions ──────────────────────────────────────────────────
# Per-layer threshold — each layer gets its own percentile cutoff
filtered_mean_A = np.zeros_like(mean_A)
filtered_mean_B = np.zeros_like(mean_B)
stable_dims_per_layer = []

for layer_idx in range(num_layers):
    # Boundary B filtering
    tau = np.percentile(variance_B[layer_idx], STABLE_PERCENTILE)
    mask = variance_B[layer_idx] < tau  # True = stable dimension

    filtered_mean_A[layer_idx] = mean_A[layer_idx] * mask
    filtered_mean_B[layer_idx] = mean_B[layer_idx] * mask
    stable_dims_per_layer.append(mask.sum())

avg_stable = np.mean(stable_dims_per_layer)
print(f"\nAvg stable dims per layer: {avg_stable:.0f} / {hidden_dim} "
      f"({avg_stable/hidden_dim*100:.1f}%)")

# ── Load model to extract W_K and W_V ────────────────────────────────────────
print("\nLoading model to extract projection weights...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu"   # cpu is fine — we're just reading weights, not doing inference
)
model.eval()

# ── Project to KV space ───────────────────────────────────────────────────────
# For Qwen2.5 the attention layers are at model.model.layers[i].self_attn
# k_proj and v_proj are the projection matrices we need

centroid_K_A = np.zeros((num_layers, model.config.num_key_value_heads * 
                          (model.config.hidden_size // model.config.num_attention_heads)), 
                          dtype=np.float32)
centroid_V_A = np.zeros_like(centroid_K_A)
centroid_K_B = np.zeros_like(centroid_K_A)
centroid_V_B = np.zeros_like(centroid_K_A)

print("Projecting hidden states to KV space...")
for layer_idx in range(num_layers):
    layer = model.model.layers[layer_idx].self_attn

    # Extract projection matrices — shape [kv_dim, hidden_dim]
    W_K = layer.k_proj.weight.detach().float().numpy()  # (kv_dim, hidden_dim)
    W_V = layer.v_proj.weight.detach().float().numpy()  # (kv_dim, hidden_dim)

    # Project: h @ W_K.T  →  [kv_dim]
    centroid_K_A[layer_idx] = filtered_mean_A[layer_idx] @ W_K.T
    centroid_V_A[layer_idx] = filtered_mean_A[layer_idx] @ W_V.T
    centroid_K_B[layer_idx] = filtered_mean_B[layer_idx] @ W_K.T
    centroid_V_B[layer_idx] = filtered_mean_B[layer_idx] @ W_V.T

    if layer_idx % 7 == 0:
        print(f"  Layer {layer_idx:02d} | "
              f"K_B norm: {np.linalg.norm(centroid_K_B[layer_idx]):.4f} | "
              f"V_B norm: {np.linalg.norm(centroid_V_B[layer_idx]):.4f}")

# ── Save centroid blocks ──────────────────────────────────────────────────────
np.save(f"{OUTPUT_DIR}/centroid_K_A.npy", centroid_K_A)  # sys prompt boundary
np.save(f"{OUTPUT_DIR}/centroid_V_A.npy", centroid_V_A)
np.save(f"{OUTPUT_DIR}/centroid_K_B.npy", centroid_K_B)  # full context boundary
np.save(f"{OUTPUT_DIR}/centroid_V_B.npy", centroid_V_B)

# Also save filtered means — useful for debugging later
np.save(f"{OUTPUT_DIR}/filtered_mean_A.npy", filtered_mean_A)
np.save(f"{OUTPUT_DIR}/filtered_mean_B.npy", filtered_mean_B)

# vLLM reads this next to centroid .npy (see vllm/centroid_injector.py)
_sc = Path(INPUT_DIR) / "sys_prefix_num_tokens.txt"
if _sc.is_file():
    (Path(OUTPUT_DIR) / "sys_prefix_num_tokens.txt").write_text(_sc.read_text())
    print(
        f"  sys_prefix_num_tokens.txt → {_sc.read_text().strip()} (copied to {OUTPUT_DIR}/)"
    )

print(f"\n✓ Centroid shapes:")
print(f"  centroid_K_B : {centroid_K_B.shape}")
print(f"  centroid_V_B : {centroid_V_B.shape}")
print(f"  Size on disk : ~{centroid_K_B.nbytes * 4 / 1024:.1f} KB total")
print(f"\n✓ Saved to {OUTPUT_DIR}/")

# ── Final sanity checks ───────────────────────────────────────────────────────
print("\n── Sanity Checks ──")
print(f"K_B norm (layer 0)      : {np.linalg.norm(centroid_K_B[0]):.4f}  (non-zero = good)")
print(f"V_B norm (layer 0)      : {np.linalg.norm(centroid_V_B[0]):.4f}  (non-zero = good)")
print(f"K_A != K_B (layer 14)   : {not np.allclose(centroid_K_A[14], centroid_K_B[14])}  (True = good)")
print(f"Filtered zeros (layer 0): {(filtered_mean_B[0] == 0).sum()} / {hidden_dim} dims zeroed")