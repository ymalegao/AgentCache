"""
Context-distillation prefix training: KL(teacher || student) + CE loss.

The teacher sees the full system prompt through the frozen base model.
The student sees no system prompt but has a trainable PEFT prefix encoder.
Both share the same backbone weights (loaded once).

Loss: L = alpha * CE_student + beta * (T^2) * KL(teacher || student)
       on assistant tokens only.

Usage:
  python agentcache_compression/train_prefix_distillation.py \
    --model /path/to/Qwen-7B-Instruct \
    --data agentcache_compression/data/python_agent_train.jsonl \
    --system-prompt agentcache_compression/prompts/python_agent_system.txt \
    --output agentcache_compression/adapters/N64_distill \
    --num-virtual-tokens 64 \
    --loss-mode ce_kl \
    --alpha 1.0 --beta 1.0 --temperature 2.0
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--system-prompt", required=True, help="Path to system prompt .txt file")
    p.add_argument("--output", required=True)
    p.add_argument("--num-virtual-tokens", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=4)
    # Distillation hyperparameters
    p.add_argument("--alpha", type=float, default=1.0, help="Weight for CE loss")
    p.add_argument("--beta", type=float, default=1.0, help="Weight for KL loss")
    p.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    p.add_argument("--loss-mode", choices=["ce_only", "kl_only", "ce_kl"], default="ce_kl",
                   help="Which loss components to use")
    return p.parse_args()


def build_distillation_dataset(data_path: str, system_text: str, tokenizer) -> Dataset:
    """Build paired teacher/student sequences with alignment offsets.

    Teacher input: [system + user + assistant] (full system prompt)
    Student input: [user + assistant] (no system prompt — prefix encoder replaces it)

    Both share the same assistant content. We record the token offset where
    the assistant turn starts in each sequence so logits can be aligned.
    """
    rows = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    teacher_ids_list, student_ids_list = [], []
    teacher_asst_starts, student_asst_starts, asst_lens = [], [], []
    skipped = 0

    for row in rows:
        # --- Teacher: has system prompt ---
        teacher_msgs_no_asst = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": row["user"]},
        ]
        teacher_prompt_text = tokenizer.apply_chat_template(
            teacher_msgs_no_asst, tokenize=False, add_generation_prompt=True
        )
        teacher_full_msgs = teacher_msgs_no_asst + [
            {"role": "assistant", "content": row["teacher_output"]}
        ]
        teacher_full_text = tokenizer.apply_chat_template(
            teacher_full_msgs, tokenize=False, add_generation_prompt=False
        )
        teacher_prompt_ids = tokenizer(teacher_prompt_text, add_special_tokens=False)["input_ids"]
        teacher_input_ids = tokenizer(teacher_full_text, add_special_tokens=False)["input_ids"]

        # --- Student: no system prompt ---
        student_msgs_no_asst = [
            {"role": "user", "content": row["user"]},
        ]
        student_prompt_text = tokenizer.apply_chat_template(
            student_msgs_no_asst, tokenize=False, add_generation_prompt=True
        )
        student_full_msgs = student_msgs_no_asst + [
            {"role": "assistant", "content": row["teacher_output"]}
        ]
        student_full_text = tokenizer.apply_chat_template(
            student_full_msgs, tokenize=False, add_generation_prompt=False
        )
        student_prompt_ids = tokenizer(student_prompt_text, add_special_tokens=False)["input_ids"]
        student_input_ids = tokenizer(student_full_text, add_special_tokens=False)["input_ids"]

        teacher_asst_start = len(teacher_prompt_ids)
        student_asst_start = len(student_prompt_ids)
        teacher_asst_len = len(teacher_input_ids) - teacher_asst_start
        student_asst_len = len(student_input_ids) - student_asst_start

        # Skip degenerate examples
        if teacher_asst_len <= 0 or student_asst_len <= 0:
            skipped += 1
            continue

        # Assistant tokens should be identical in both (same text, same tokenization)
        asst_len = min(teacher_asst_len, student_asst_len)

        teacher_ids_list.append(teacher_input_ids)
        student_ids_list.append(student_input_ids)
        teacher_asst_starts.append(teacher_asst_start)
        student_asst_starts.append(student_asst_start)
        asst_lens.append(asst_len)

    print(f"Distillation dataset built: {len(teacher_ids_list)} examples ({skipped} skipped)")
    return Dataset.from_dict({
        "teacher_input_ids": teacher_ids_list,
        "student_input_ids": student_ids_list,
        "teacher_asst_start": teacher_asst_starts,
        "student_asst_start": student_asst_starts,
        "asst_len": asst_lens,
    })


def distillation_collator(pad_token_id: int):
    """Pad teacher and student sequences independently, preserving offset fields."""
    def collate(features):
        # Pad teacher sequences
        t_max = max(len(f["teacher_input_ids"]) for f in features)
        s_max = max(len(f["student_input_ids"]) for f in features)

        t_input_ids, t_attention_mask = [], []
        s_input_ids, s_attention_mask, s_labels = [], [], []
        teacher_asst_starts, student_asst_starts, asst_lens = [], [], []

        for f in features:
            # Teacher
            t_seq = f["teacher_input_ids"]
            t_pad = t_max - len(t_seq)
            t_input_ids.append(t_seq + [pad_token_id] * t_pad)
            t_attention_mask.append([1] * len(t_seq) + [0] * t_pad)

            # Student (with labels for CE loss)
            s_seq = f["student_input_ids"]
            s_pad = s_max - len(s_seq)
            s_asst_start = f["student_asst_start"]
            labels = [-100] * s_asst_start + s_seq[s_asst_start:] + [-100] * s_pad

            s_input_ids.append(s_seq + [pad_token_id] * s_pad)
            s_attention_mask.append([1] * len(s_seq) + [0] * s_pad)
            s_labels.append(labels)

            teacher_asst_starts.append(f["teacher_asst_start"])
            student_asst_starts.append(f["student_asst_start"])
            asst_lens.append(f["asst_len"])

        return {
            "teacher_input_ids": torch.tensor(t_input_ids, dtype=torch.long),
            "teacher_attention_mask": torch.tensor(t_attention_mask, dtype=torch.long),
            "student_input_ids": torch.tensor(s_input_ids, dtype=torch.long),
            "student_attention_mask": torch.tensor(s_attention_mask, dtype=torch.long),
            "labels": torch.tensor(s_labels, dtype=torch.long),
            "teacher_asst_start": torch.tensor(teacher_asst_starts, dtype=torch.long),
            "student_asst_start": torch.tensor(student_asst_starts, dtype=torch.long),
            "asst_len": torch.tensor(asst_lens, dtype=torch.long),
        }
    return collate


class DistillationTrainer(Trainer):
    """Custom trainer that computes KL(teacher || student) + CE on assistant tokens.

    Architecture: the PEFT model wraps the base model. Teacher forward passes
    through model.base_model.model (skipping prefix encoder), student forward
    passes through model (prefix encoder active).
    """

    def __init__(self, *args, alpha=1.0, beta=1.0, temperature=2.0, loss_mode="ce_kl", **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.loss_mode = loss_mode

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        student_input_ids = inputs["student_input_ids"]
        student_attention_mask = inputs["student_attention_mask"]
        labels = inputs["labels"]
        teacher_asst_start = inputs["teacher_asst_start"]
        student_asst_start = inputs["student_asst_start"]
        asst_len = inputs["asst_len"]

        # --- Student forward (prefix encoder active) ---
        student_outputs = model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=labels if self.loss_mode != "kl_only" else None,
        )
        student_logits = student_outputs.logits  # (B, S_student, V)

        ce_loss = student_outputs.loss if self.loss_mode != "kl_only" else torch.tensor(0.0, device=student_logits.device)

        # --- Teacher forward (base CausalLM, no prefix encoder, sees system prompt) ---
        kl_loss = torch.tensor(0.0, device=student_logits.device)
        if self.loss_mode != "ce_only":
            with torch.no_grad():
                teacher_outputs = model.get_base_model()(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                )
            teacher_logits = teacher_outputs.logits  # (B, S_teacher, V)

            # Compute KL divergence on aligned assistant token logits
            T = self.temperature
            batch_kl = []
            for i in range(teacher_logits.size(0)):
                t_start = teacher_asst_start[i].item()
                s_start = student_asst_start[i].item()
                length = asst_len[i].item()

                if length <= 0:
                    continue

                # Logits for next-token prediction: positions [start, start+length-1]
                # predict tokens at [start+1, start+length]
                t_logits_slice = teacher_logits[i, t_start:t_start + length]  # (L, V)
                s_logits_slice = student_logits[i, s_start:s_start + length]  # (L, V)

                t_probs = F.softmax(t_logits_slice / T, dim=-1)
                s_log_probs = F.log_softmax(s_logits_slice / T, dim=-1)

                # KL(teacher || student) = sum teacher * log(teacher / student)
                kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
                batch_kl.append(kl)

            if batch_kl:
                kl_loss = torch.stack(batch_kl).mean() * (T ** 2)

        # Combined loss
        if self.loss_mode == "ce_only":
            loss = self.alpha * ce_loss
        elif self.loss_mode == "kl_only":
            loss = self.beta * kl_loss
        else:
            loss = self.alpha * ce_loss + self.beta * kl_loss

        if return_outputs:
            return loss, student_outputs
        return loss


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    system_prompt_full = Path(args.system_prompt).read_text().strip()

    print(f"Model:              {args.model}")
    print(f"Data:               {args.data}")
    print(f"Output:             {args.output}")
    print(f"num_virtual_tokens: {args.num_virtual_tokens}")
    print(f"Loss mode:          {args.loss_mode}")
    print(f"alpha={args.alpha}  beta={args.beta}  temperature={args.temperature}")
    print(f"epochs:             {args.epochs}  lr: {args.lr}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    for param in model.parameters():
        param.requires_grad = False

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prefix_projection=True,
    )
    model = get_peft_model(model, peft_config)

    trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nTrainable parameters ({len(trainable)} tensors):")
    for name, shape in trainable:
        print(f"  {name}: {shape}")
    model.print_trainable_parameters()

    print("\nBuilding distillation dataset...")
    train_dataset = build_distillation_dataset(args.data, system_prompt_full, tokenizer)

    # Sanity check
    ex = train_dataset[0]
    print(f"Example 0: teacher_len={len(ex['teacher_input_ids'])}, "
          f"student_len={len(ex['student_input_ids'])}, "
          f"asst_len={ex['asst_len']}")

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=distillation_collator(tokenizer.pad_token_id),
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        loss_mode=args.loss_mode,
    )

    print("\nStarting distillation training...")
    trainer.train()

    print(f"\nSaving adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
