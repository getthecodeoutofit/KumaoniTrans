#!/usr/bin/env python3
"""
Training script for Kumaoni translator using Gemma model.
This script fine-tunes a Gemma model to translate Hinglish to Kumaoni.
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/gemma")
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/data.json")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma-kumaoni")

def load_dataset(dataset_path):
    """Load the dataset from a JSON file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to the format expected by the Dataset class
    formatted_data = {
        "text": []
    }

    for item in data:
        # Format as instruction with input and output
        formatted_text = f"Translate from Hinglish to Kumaoni:\nHinglish: {item['hinglish']}\nKumaoni: {item['kumaoni']}"
        formatted_data["text"].append(formatted_text)

    return Dataset.from_dict(formatted_data)

def main():
    # Load tokenizer
    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 8-bit quantization to save memory
    print(f"Loading model from {MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        load_in_8bit=True,  # Use 8-bit quantization
        local_files_only=True
    )

    # Prepare model for 8-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA for efficient fine-tuning
    peft_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention modules to fine-tune
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and preprocess the dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset(DATASET_PATH)

    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors=None  # Return Python lists instead of tensors
        )

        # Create input_ids and attention_mask
        result["labels"] = result["input_ids"].copy()

        return result

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not doing masked language modeling
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,  # Use mixed precision training
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
