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
    "--sys-tokens", type=int, default=0,
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
    # IMPORTANT:
    # For prefix_projection=True, the only trustworthy source is PEFT's runtime
    # get_prompt(), because manual PrefixEncoder reconstruction can drift from
    # the actual cache format/model wrappers.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_model_path = config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(
            "adapter_config.json missing base_model_name_or_path; cannot export "
            "projected prefix cache via PeftModel.get_prompt()."
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()

    k_layers = []
    v_layers = []
    with torch.no_grad():
        prompt_cache = peft_model.get_prompt(batch_size=1)
        if len(prompt_cache) != num_layers:
            raise ValueError(
                f"Unexpected prompt cache layers: got {len(prompt_cache)} expected {num_layers}"
            )

        for layer_idx, layer_cache in enumerate(prompt_cache):
            if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
                raise ValueError(
                    f"Unexpected cache object at layer {layer_idx}: {type(layer_cache)}"
                )
            k = layer_cache[0][0]  # [num_kv_heads, num_virtual_tokens, head_dim]
            v = layer_cache[1][0]  # [num_kv_heads, num_virtual_tokens, head_dim]

            k_flat = k.permute(1, 0, 2).contiguous().view(num_virtual_tokens, -1)
            v_flat = v.permute(1, 0, 2).contiguous().view(num_virtual_tokens, -1)
            if k_flat.shape[1] != token_dim or v_flat.shape[1] != token_dim:
                raise ValueError(
                    "Runtime prompt cache dim mismatch: "
                    f"K={tuple(k_flat.shape)} V={tuple(v_flat.shape)} token_dim={token_dim}"
                )
            k_layers.append(k_flat.detach().cpu().numpy())
            v_layers.append(v_flat.detach().cpu().numpy())

    learned_K = np.stack(k_layers, axis=0)
    learned_V = np.stack(v_layers, axis=0)
    np.save(out_k, learned_K)
    np.save(out_v, learned_V)
    sidecar = out_k.with_name("sys_prefix_num_tokens.txt")
    sidecar.write_text(str(args.sys_tokens))

    print(f"Exported KV tensors with shape: {learned_K.shape}")
    print(f"  -> {out_k}")
    print(f"  -> {out_v}")
    print(f"  -> {sidecar}  (sys_token_count={args.sys_tokens})")
    print("  exporter: PeftModel.get_prompt() (runtime cache-aligned)")
    raise SystemExit(0)

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