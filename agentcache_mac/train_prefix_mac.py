"""
train_prefix_mac.py — Apple-Silicon (MPS) port of
agentcache_compression/train_prefix_compression.py.

Trains the SAME PEFT prefix adapter (compression mode: a synthetic KV prefix that
stands in for a removed system prompt), but on the Metal backend instead of CUDA.

Only the device / precision handling differs from the CUDA script. The critical
correctness path is byte-for-byte identical and must stay that way:
  - PrefixTuningConfig(num_virtual_tokens=N, prefix_projection=True)
  - label masking: labels = [-100]*prompt_len + input_ids[prompt_len:]
  - system_retain_ratio (0.0 => no system prompt in the input; prefix must carry it)

Why the precision change:
  HuggingFace Trainer's fp16/bf16 mixed-precision path is CUDA-bound, and MPS bf16
  *backward* is historically flaky. We therefore train with the base model in fp32
  (the default). Only the ~3M prefix-encoder params get gradients, so fp32 is
  affordable even for 7B/8B bases on a 96GB machine. --dtype bf16 is available for
  users who confirm it's stable on their torch/macOS combo.

Usage:
  python agentcache_mac/train_prefix_mac.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --data agentcache_compression/data/python_agent_train.jsonl \
    --system-prompt agentcache_compression/prompts/2000_python_agent_system.txt \
    --output agentcache_mac/adapters/Llama-3.2-1B_N64 \
    --num-virtual-tokens 64 \
    --epochs 8 --lr 2e-3
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--system-prompt", required=True, help="Path to system prompt .txt file")
    p.add_argument("--output", required=True)
    p.add_argument("--num-virtual-tokens", type=int, default=64)
    p.add_argument("--system-retain-ratio", type=float, default=0.0,
                   help="Fraction of system prompt tokens to keep (0.0 = none)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", choices=list(_DTYPES), default="fp32",
                   help="Base-model dtype during training. fp32 is the safe MPS default; "
                        "bf16 only if confirmed stable for your torch/macOS.")
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    return p.parse_args()


def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def truncate_system(text: str, ratio: float, tokenizer) -> str:
    if ratio <= 0.0:
        return ""
    if ratio >= 1.0:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    keep = max(1, int(len(ids) * ratio))
    return tokenizer.decode(ids[:keep], skip_special_tokens=False)


def build_dataset(data_path: str, system_text: str, tokenizer) -> Dataset:
    rows = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    input_ids_list, labels_list = [], []
    skipped = 0

    for row in rows:
        messages_no_asst = []
        if system_text:
            messages_no_asst.append({"role": "system", "content": system_text})
        messages_no_asst.append({"role": "user", "content": row["user"]})

        # Prompt text = everything up to (and including) the assistant header
        prompt_text = tokenizer.apply_chat_template(
            messages_no_asst, tokenize=False, add_generation_prompt=True
        )

        messages_full = messages_no_asst + [
            {"role": "assistant", "content": row["teacher_output"]}
        ]
        full_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Skip degenerate examples (empty assistant turn)
        if prompt_len >= len(input_ids):
            skipped += 1
            continue

        # Labels: -100 for prompt tokens, token ids for assistant tokens
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    print(f"Dataset built: {len(input_ids_list)} examples ({skipped} skipped — empty assistant)")
    return Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list})


def dynamic_collator(pad_token_id: int):
    """Pad each batch to its longest sequence. Labels pad to -100."""
    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            seq = f["input_ids"]
            lbl = f["labels"]
            pad = max_len - len(seq)
            input_ids.append(seq + [pad_token_id] * pad)
            attention_mask.append([1] * len(seq) + [0] * pad)
            labels.append(lbl + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = pick_device(args.device)
    dtype = _DTYPES[args.dtype]

    system_prompt_full = Path(args.system_prompt).read_text().strip()

    print(f"Model:              {args.model}")
    print(f"Device:             {device}   dtype: {args.dtype}")
    print(f"Data:               {args.data}")
    print(f"Output:             {args.output}")
    print(f"num_virtual_tokens: {args.num_virtual_tokens}")
    print(f"system_retain_ratio:{args.system_retain_ratio}")
    print(f"epochs:             {args.epochs}  lr: {args.lr}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_text = truncate_system(system_prompt_full, args.system_retain_ratio, tokenizer)
    if system_text:
        sys_tok_len = len(tokenizer(system_text, add_special_tokens=False)["input_ids"])
        print(f"System text: {sys_tok_len} tokens (ratio={args.system_retain_ratio})")
    else:
        print("System text: none (ratio=0.0)")

    print("Loading base model...")
    # No device_map="auto": accelerate offloading conflicts with MPS placement.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    for param in model.parameters():
        param.requires_grad = False

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prefix_projection=True,
    )
    model = get_peft_model(model, peft_config)
    model.to(device)

    # Verify: only prefix encoder should be trainable
    trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nTrainable parameters ({len(trainable)} tensors):")
    for name, shape in trainable:
        print(f"  {name}: {shape}")
    model.print_trainable_parameters()

    print("\nBuilding dataset with label masking...")
    train_dataset = build_dataset(args.data, system_text, tokenizer)

    ex = train_dataset[0]
    n_labels = sum(1 for l in ex["labels"] if l != -100)
    n_masked = sum(1 for l in ex["labels"] if l == -100)
    print(f"Example 0: {n_labels} assistant tokens, {n_masked} masked (prompt) tokens")
    lengths = [len(d["input_ids"]) for d in train_dataset]
    print(f"Sequence lengths — min:{min(lengths)} median:{sorted(lengths)[len(lengths)//2]} max:{max(lengths)}")

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="epoch",
        # MPS: HF Trainer mixed precision is CUDA-only. Keep both off; the base
        # dtype is controlled at load time via --dtype.
        fp16=False,
        bf16=False,
        use_cpu=(device == "cpu"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=dynamic_collator(tokenizer.pad_token_id),
    )

    print(f"\nStarting prefix tuning on {device}...")
    trainer.train()

    print(f"\nSaving adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
