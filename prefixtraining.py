import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, PrefixTuningConfig, TaskType

# Configuration
MODEL_ID = "/mnt/g/agentcache/models/qwen-1.5b"
OUTPUT_DIR = "./agentcache_prefix_model"
NUM_VIRTUAL_TOKENS = 256  # Keeping it compact to avoid softmax dilution
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-3 # Prefix tuning requires higher LRs than full finetuning

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Ensure tokenizer has a pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model strictly frozen and in appropriate dtype
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Configure Prefix Tuning
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        prefix_projection=True # Uses a 2-layer MLP to stabilize training
    )
    
    # Wrap model with PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load your domain-specific dataset (e.g., successful SWE-bench or Search trajectories)
    # Replacing this with a dummy huggingface dataset for structural demonstration
    print("Loading and tokenizing dataset...")
   
   
    dataset = load_dataset("json", data_files={"train": "good_examples/good_examples.jsonl"})
   
   
    def tokenize_function(examples):

        full_text = [
            f"Task: {t}\n\nResponse: {g}" 
            for t, g in zip(examples["task"], examples["good_example"])
        ]
        # We want the model to learn the structural prior of a good response
        return tokenizer(
            full_text, 
            truncation=True, 
            max_length=512, 
            padding="max_length"
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    
    # Setup Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True, # Recommended for modern GPUs
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting Prefix Tuning...")
    trainer.train()

    print(f"Saving AgentCache Prefix weights to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()