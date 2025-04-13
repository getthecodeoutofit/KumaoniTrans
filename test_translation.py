#!/usr/bin/env python3
"""
Test script to verify the translation functionality.
"""

import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/facebook")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nllb-kumaoni")

def load_model():
    """Load the model and tokenizer"""
    print(f"Loading model from {OUTPUT_DIR}...")
    
    # Check if fine-tuned model exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Fine-tuned model not found at {OUTPUT_DIR}. Using base model from {MODEL_DIR}")
        model_path = MODEL_DIR
    else:
        model_path = OUTPUT_DIR
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    
    # Load model
    if model_path == MODEL_DIR:
        # Load base model directly
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="auto",
            local_files_only=True
        )
    else:
        # Load base model first
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            local_files_only=True
        )
        
        # Load the fine-tuned model
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            print("Loaded fine-tuned model with LoRA adapters")
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            print("Falling back to base model")
            model = base_model
    
    return model, tokenizer

def translate(text, model, tokenizer, max_length=100):
    """Translate Hinglish text to Kumaoni"""
    # Format the input prompt
    prompt = f"translate Hinglish to Kumaoni: {text}"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate the translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=5,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation

def main():
    """Test the translation functionality"""
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Test examples from the dataset
    test_examples = [
        "aap kaise hain",
        "aapka naam kya hai",
        "main theek hoon",
        "shubh prabhat",
        "aap kahan ja rahe hain"
    ]
    
    print("\nTranslation examples:")
    for text in test_examples:
        translation = translate(text, model, tokenizer)
        print(f"Hinglish: {text}")
        print(f"Kumaoni: {translation}")
        print("-" * 40)

if __name__ == "__main__":
    main()
