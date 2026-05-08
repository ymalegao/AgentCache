import torch
from safetensors.torch import load_file
import json
import numpy as np

# Load config to get hyperparameters
with open("agentcache_prefix_model/adapter_config.json", "r") as f:
    config = json.load(f)

num_virtual_tokens = config["num_virtual_tokens"]
num_layers = config["num_layers"]
token_dim = config["token_dim"]
prefix_projection = config.get("prefix_projection", False)

# Load the weights
weights = load_file("agentcache_prefix_model/adapter_model.safetensors")

if not prefix_projection:
    # Flattened weight is: [num_virtual_tokens, num_layers * 2 * token_dim]
    raw_prefix = weights["prompt_embeddings.weight"]
    
    # Reshape to [N, L, 2, d]
    materialized_kv = raw_prefix.view(num_virtual_tokens, num_layers, 2, token_dim)
else:
    # You must reconstruct the MLP and run the embeddings through it
    # This matches the internal PEFT 'PrefixEncoder' logic
    from peft import PrefixEncoder, PrefixTuningConfig
    # PEFT may save adapters with inference_mode=True, but PrefixEncoder
    # only builds `transform` when inference_mode=False.
    # Force materialization mode so projected prefixes can be reconstructed.
    materialization_config = dict(config)
    materialization_config["inference_mode"] = False
    peft_config = PrefixTuningConfig(**materialization_config)
    encoder = PrefixEncoder(peft_config)
    
    # Load state_dict into a temporary encoder to materialize
    # Keys in safetensors usually have 'base_model.model.' prefix
    clean_state_dict = {k.replace("base_model.model.", ""): v for k, v in weights.items()}
    encoder.load_state_dict(clean_state_dict, strict=False)
    
    # Materialize: pass tokens [0, 1, 2... N] through the MLP
    indices = torch.arange(num_virtual_tokens).unsqueeze(0)  # [1, N]
    materialized_kv = encoder(indices).squeeze(0)  # [N, L * 2 * d]
    materialized_kv = materialized_kv.view(num_virtual_tokens, num_layers, 2, token_dim)

# Final shape for vLLM: [num_layers, num_virtual_tokens, token_dim]
# Separate Key and Value
# We transpose to get [2, L, N, d] then split
kv_split = materialized_kv.permute(2, 1, 0, 3) 
learned_K = kv_split[0].detach().cpu().numpy()
learned_V = kv_split[1].detach().cpu().numpy()

# Save for your CentroidInjector
np.save("centroid_K.npy", learned_K)
np.save("centroid_V.npy", learned_V)

print(f"Exported KV tensors with shape: {learned_K.shape}")