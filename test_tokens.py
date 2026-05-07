from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/g/agentcache/models/qwen-1.5b")
SYSTEM = "You are a helpful assistant that can interact with a computer. Please solve the issue provided."
TEST   = "Write a Python context manager that times how long a code block takes to execute."
sys_prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
user_prompt = f"<|im_start|>user\n{TEST}<|im_end|>\n<|im_start|>assistant\n"
sys_tokens = tokenizer.encode(sys_prompt)
user_tokens = tokenizer.encode(user_prompt)
print(f"sys_tokens: {len(sys_tokens)}")
print(f"user_tokens: {len(user_tokens)}")
