from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

from peft import LoraConfig, get_peft_model
import os

# Local paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/facebook")
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/data.json")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nllb-kumaoni")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

# Confirm tokenizer
assert tokenizer.vocab_size > 0, "Tokenizer failed to load!"

# Load model with 8-bit quantization to save memory
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    load_in_8bit=True,  # Use 8-bit quantization to save memory
    local_files_only=True
)

# LoRA preparation for fine-tuning

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, peft_config)

# Preprocessing function
def preprocess(examples):
    # Add a clear instruction and language tags to help the model understand
    inputs = [f"Translate from Hinglish to Kumaoni language: {x}" for x in examples["hinglish"]]

    # Tokenize inputs with attention to padding
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")

    # Tokenize outputs (Kumaoni text)
    # Add a prefix to help the model identify the target language
    target_texts = [f"Kumaoni: {x}" for x in examples["kumaoni"]]
    labels = tokenizer(target_texts, max_length=64, truncation=True, padding="max_length")

    # Set labels for training
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Load dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(preprocess, batched=True)

# Training arguments for stable & accurate training
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,   # More stable for fine-tuning
    num_train_epochs=3,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# Train and save
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
