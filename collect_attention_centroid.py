import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

os.environ["HF_HOME"] = "/mnt/g/agentcache/hf_cache"
MODEL_ID = "/mnt/g/agentcache/models/qwen-1.5b"
OUTPUT_DIR = "./attention_centroid_output"
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

_sys_prefix_only = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"
_sys_n = len(tokenizer(_sys_prefix_only, return_tensors="pt").input_ids[0])
with open(f"{OUTPUT_DIR}/sys_prefix_num_tokens.txt", "w") as f:
    f.write(str(_sys_n))
print(f"Wrote {OUTPUT_DIR}/sys_prefix_num_tokens.txt → {_sys_n} (vLLM exact sys prompt length)")


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager",
)
model.eval()

num_hidden_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size

print(f"Model config: {num_hidden_layers} transformer layers, hidden dim {hidden_dim}")

def chat_prompt(task: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{task}<|im_end|>\n"
    )

# Trackers
weighted_hs = defaultdict(lambda: np.zeros((num_hidden_layers, hidden_dim), dtype=np.float32))
total_importance = defaultdict(lambda: np.zeros(num_hidden_layers, dtype=np.float32))

for run_idx, task in enumerate(TASKS):
    print(f"── Run {run_idx + 1}/{len(TASKS)} ──")
    full_text = chat_prompt(task)
    tokens = tokenizer(full_text, return_tensors="pt").input_ids.cuda()
    
    with torch.no_grad():
        outputs = model(input_ids=tokens, output_hidden_states=True, output_attentions=True)
    
    hs = outputs.hidden_states # Tuple of len(num_layers+1)
    attns = outputs.attentions # Tuple of len(num_layers)
    seq_len = tokens.shape[1]
    
    for pos in range(seq_len):
        token_id = tokens[0, pos].item()
        
        for layer in range(num_hidden_layers):
            # hidden state at layer input
            h = hs[layer][0, pos, :].float().cpu().numpy()
            
            # attention importance: how much future tokens attend to `pos`
            if pos < seq_len - 1:
                # attns[layer] shape: [1, num_heads, seq_len, seq_len]
                importance = attns[layer][0, :, pos+1:, pos].mean().item()
            else:
                importance = 0.0
                
            weighted_hs[token_id][layer] += importance * h
            total_importance[token_id][layer] += importance
            
    torch.cuda.empty_cache()

# Aggregate and select Top N
N_CENTROID = 64
layer_weights = np.linspace(0.5, 1.5, num_hidden_layers)

token_agg_importance = {}
for tid, imp_array in total_importance.items():
    agg = np.sum(layer_weights * imp_array)
    token_agg_importance[tid] = float(agg)

# Sort by aggregate importance and filter out structural/system tokens
import string

sys_token_ids = set(tokenizer(_sys_prefix_only, return_tensors="pt").input_ids[0].tolist())

def is_domain_token(tid):
    if tid in sys_token_ids:
        return False
    s = tokenizer.decode([tid]).strip()
    if not s:
        return False
    if all(c in string.punctuation for c in s):
        return False
    # Filter out common english stopwords that aren't in sys prompt but are highly frequent and not domain-specific
    stopwords = {"i", "the", "it", "we", "they", "he", "she", "this", "that", "a", "an", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "and", "of", "my", "but", "me", "have", "do", "as", "if", "you", "your", "what", "which", "who", "whom", "whose", "would", "could", "need", "or", "are", "be", "am", "is", "was", "were", "been", "being", "has", "had", "does", "did", "doing"}
    if s.lower() in stopwords:
        return False
    return True

sorted_tokens = [t for t in sorted(token_agg_importance.keys(), key=lambda t: token_agg_importance[t], reverse=True) if is_domain_token(t)]
top_n_tokens = sorted_tokens[:N_CENTROID]

print("\nTop 20 domain tokens by importance:")
for tid in top_n_tokens[:20]:
    token_str = tokenizer.decode([tid])
    print(f"  {repr(token_str)}: {token_agg_importance[tid]:.2f}")

# Extract mean hidden states for top N
centroid_hs = np.zeros((num_hidden_layers, N_CENTROID, hidden_dim), dtype=np.float32)

for i, tid in enumerate(top_n_tokens):
    for layer in range(num_hidden_layers):
        if total_importance[tid][layer] > 0:
            mean_h = weighted_hs[tid][layer] / total_importance[tid][layer]
        else:
            mean_h = np.zeros(hidden_dim, dtype=np.float32)
        centroid_hs[layer, i, :] = mean_h

# Save token metadata
top_tokens_meta = [
    {"token_id": tid, "token_str": tokenizer.decode([tid]), "agg_importance": float(token_agg_importance[tid])}
    for tid in top_n_tokens
]
with open(f"{OUTPUT_DIR}/top_tokens.json", "w") as f:
    json.dump(top_tokens_meta, f, indent=2)

# Project to KV Space
print("\nProjecting hidden states to KV space...")
centroid_K = np.zeros((num_hidden_layers, N_CENTROID, model.config.num_key_value_heads * 
                      (model.config.hidden_size // model.config.num_attention_heads)), 
                      dtype=np.float32)
centroid_V = np.zeros_like(centroid_K)

for layer_idx in range(num_hidden_layers):
    layer = model.model.layers[layer_idx].self_attn
    W_K = layer.k_proj.weight.detach().float().cpu().numpy()
    W_V = layer.v_proj.weight.detach().float().cpu().numpy()
    
    # centroid_hs[layer_idx] shape is [N_CENTROID, hidden_dim]
    # W_K.T shape is [hidden_dim, kv_dim]
    centroid_K[layer_idx] = centroid_hs[layer_idx] @ W_K.T
    centroid_V[layer_idx] = centroid_hs[layer_idx] @ W_V.T

np.save(f"{OUTPUT_DIR}/centroid_K.npy", centroid_K)
np.save(f"{OUTPUT_DIR}/centroid_V.npy", centroid_V)

print(f"✓ Saved centroid K/V with shape {centroid_K.shape} to {OUTPUT_DIR}/")
