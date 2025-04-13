# train_translator.py
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Config
model_name = "facebook/nllb-200-distilled-600M"
dataset_path = "data.json"  # Your dataset
output_dir = "./nllb-kumaoni-translator"

# 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA setup
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, peft_config)

# Dataset prep
def format_data(examples):
    inputs = [f"translate Hinglish to Kumaoni: {x}" for x in examples["hinglish"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["kumaoni"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(format_data, batched=True)

# Training
trainer = Seq2SeqTrainer(
    model=model,
    args=Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        fp16=True,
        save_strategy="epoch"
    ),
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)
trainer.train()
trainer.save_model(output_dir)