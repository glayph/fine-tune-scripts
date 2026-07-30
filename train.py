#!/usr/bin/env python
"""Simple LoRA fine-tune script for GPT‑NeoX models."""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    parser.add_argument("--model", default="gmcmartin/LFM2.5-1.2B-Instruct-Q4_K_M", help="Base model name or path")
    parser.add_argument("--dataset", default="", help="HuggingFace dataset name")
    parser.add_argument("--output_dir", default="./finetuned_model", help="Output directory for fine-tuned model")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Ensure EOS token is set
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": device},
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    if args.dataset:
        print(f"Loading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset)
    else:
        # Create a dummy dataset for testing
        print("Creating dummy dataset for testing...")
        dataset = {"train": [{"text": f"Example text {i} " * 50} for i in range(100)]}

    # Prepare dataset for training
    def format_dataset(example):
        # Format the text for model training
        text = example.get("text", "")
        # Ensure we have a valid prompt/response format
        if "instruction" in example and "response" in example:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        return {"text": text}
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
    
    tokenized_dataset = dataset["train"].map(format_dataset, remove_columns=dataset["train"].column_names)
    tokenized_dataset = tokenized_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch")

    # Training arguments
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=100,
        fp16=True,
    )

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()