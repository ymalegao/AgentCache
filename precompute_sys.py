import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_ID = "/mnt/g/agentcache/models/qwen-1.5b"
OUTPUT_DIR = "./attention_centroid_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM = """You are a helpful assistant that can interact with a computer.

Please solve the issue provided by the user. You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust.

""" + ("This is an extended system prompt to simulate a larger context. " * 50)

prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager",
)
model.eval()

tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
seq_len = tokens.shape[1]

with open(f"{OUTPUT_DIR}/sys_prefix_num_tokens.txt", "w") as f:
    f.write(str(seq_len))
print(f"Extended sys prompt is {seq_len} tokens")

with torch.no_grad():
    outputs = model(input_ids=tokens, output_hidden_states=True)

hs = outputs.hidden_states
num_hidden_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
kv_dim = model.config.num_key_value_heads * (hidden_dim // model.config.num_attention_heads)

sys_K = np.zeros((num_hidden_layers, seq_len, kv_dim), dtype=np.float32)
sys_V = np.zeros_like(sys_K)

for layer_idx in range(num_hidden_layers):
    layer = model.model.layers[layer_idx].self_attn
    W_K = layer.k_proj.weight.detach().float().cpu().numpy()
    W_V = layer.v_proj.weight.detach().float().cpu().numpy()
    bias_K = layer.k_proj.bias.detach().float().cpu().numpy() if hasattr(layer.k_proj, "bias") and layer.k_proj.bias is not None else 0
    bias_V = layer.v_proj.bias.detach().float().cpu().numpy() if hasattr(layer.v_proj, "bias") and layer.v_proj.bias is not None else 0
    
    layer_hs = hs[layer_idx][0].float().cpu().numpy() # [seq_len, hidden_dim]
    sys_K[layer_idx] = layer_hs @ W_K.T + bias_K
    sys_V[layer_idx] = layer_hs @ W_V.T + bias_V

np.save(f"{OUTPUT_DIR}/sys_extended_K.npy", sys_K)
np.save(f"{OUTPUT_DIR}/sys_extended_V.npy", sys_V)
print(f"Saved extended sys K/V of shape {sys_K.shape}")
