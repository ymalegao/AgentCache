import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

parser = argparse.ArgumentParser(description="Export PEFT prefix adapter to vLLM centroid .npy files.")
parser.add_argument("--adapter", default="agentcache_prefix_model", help="Path to the saved PEFT adapter directory.")
parser.add_argument("--out-k", default="centroid_K.npy", help="Output path for key tensor.")
parser.add_argument("--out-v", default="centroid_V.npy", help="Output path for value tensor.")
parser.add_argument(
    "--sys-tokens", type=int, default=1,
    help="Value written to sys_prefix_num_tokens.txt next to the outputs. "
         "1 = hybrid BOS-at-0 layout (recommended); 0 = pure PEFT from index 0.",
)
args = parser.parse_args()

adapter_dir = Path(args.adapter)
out_k = Path(args.out_k)
out_v = Path(args.out_v)

# Load config to get hyperparameters
with open(adapter_dir / "adapter_config.json", "r") as f:
    config = json.load(f)

num_virtual_tokens = config["num_virtual_tokens"]
num_layers = config["num_layers"]
token_dim = config["token_dim"]
prefix_projection = config.get("prefix_projection", False)

# Load the weights
weights = load_file(adapter_dir / "adapter_model.safetensors")

if not prefix_projection:
    # Flattened weight is: [num_virtual_tokens, num_layers * 2 * token_dim]
    raw_prefix = weights["prompt_embeddings.weight"]
    materialized_kv = raw_prefix.view(num_virtual_tokens, num_layers, 2, token_dim)
else:
    # Reconstruct via PrefixEncoder MLP to get projected embeddings.
    from peft import PrefixEncoder, PrefixTuningConfig
    # PrefixEncoder only builds `transform` when inference_mode=False.
    materialization_config = dict(config)
    materialization_config["inference_mode"] = False
    peft_config = PrefixTuningConfig(**materialization_config)
    encoder = PrefixEncoder(peft_config)
    clean_state_dict = {k.replace("base_model.model.", ""): v for k, v in weights.items()}
    encoder.load_state_dict(clean_state_dict, strict=False)
    indices = torch.arange(num_virtual_tokens).unsqueeze(0)  # [1, N]
    materialized_kv = encoder(indices).squeeze(0)            # [N, L * 2 * d]
    materialized_kv = materialized_kv.view(num_virtual_tokens, num_layers, 2, token_dim)

# Permute [N, L, 2, d] → [2, L, N, d]; final vLLM shape: [num_layers, num_virtual_tokens, token_dim]
kv_split = materialized_kv.permute(2, 1, 0, 3)
learned_K = kv_split[0].detach().cpu().numpy()
learned_V = kv_split[1].detach().cpu().numpy()

np.save(out_k, learned_K)
np.save(out_v, learned_V)

# Write sidecar so CentroidInjector reads sys_token_count without relying on env vars.
# 1 = hybrid BOS-at-0 layout (recommended); 0 = pure PEFT from index 0.
sidecar = out_k.with_name("sys_prefix_num_tokens.txt")
sidecar.write_text(str(args.sys_tokens))

print(f"Exported KV tensors with shape: {learned_K.shape}")
print(f"  -> {out_k}")
print(f"  -> {out_v}")
print(f"  -> {sidecar}  (sys_token_count={args.sys_tokens})")