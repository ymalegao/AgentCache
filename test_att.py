import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_ID = "/mnt/g/agentcache/models/qwen-1.5b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda", attn_implementation="eager")
tokens = tokenizer("test prompt", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    outputs = model(input_ids=tokens, output_hidden_states=True, output_attentions=True)
print(outputs.attentions[0].isnan().any())
