from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/mnt/g/agentcache/models/Llama-3.2-1B-Instruct')
      # Check what the actual Llama chat template looks like
sample = tok.apply_chat_template(
    [{'role': 'system', 'content': 'You are helpful.'}, {'role': 'user', 'content': 'Hello'}],
    tokenize=False, add_generation_prompt=True
)
print(repr(sample[:200]))
# Check if im_start is even in Llama vocab
print('im_start in vocab:', '<|im_start|>' in tok.get_vocab())
print('begin_of_text in vocab:', '<|begin_of_text|>' in tok.get_vocab())