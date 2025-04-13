#!/usr/bin/env python3
"""
Inference script for Kumaoni translator using Gemma model.
This script uses a fine-tuned Gemma model to translate Hinglish to Kumaoni.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/gemma")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma-kumaoni")

def load_model(model_path=OUTPUT_DIR, base_model_path=MODEL_DIR):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    # Check if fine-tuned model exists
    if not os.path.exists(model_path):
        print(f"Fine-tuned model not found at {model_path}. Using base model from {base_model_path}")
        model_path = base_model_path
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if model_path == base_model_path:
        # Load base model directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            local_files_only=True
        )
    else:
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
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
    prompt = f"Translate from Hinglish to Kumaoni:\nHinglish: {text}\nKumaoni:"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate the translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the Kumaoni part
    try:
        kumaoni_part = generated_text.split("Kumaoni:")[-1].strip()
        return kumaoni_part
    except:
        return generated_text

def main():
    parser = argparse.ArgumentParser(description="Translate Hinglish to Kumaoni")
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--model_path", type=str, default=OUTPUT_DIR, help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default=MODEL_DIR, help="Path to the base model")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Load the model and tokenizer
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    if args.interactive:
        print("=== Hinglish to Kumaoni Translation ===")
        print("Type 'exit' to quit")
        
        while True:
            text = input("\nEnter Hinglish text: ")
            if text.lower() == "exit":
                break
            
            translation = translate(text, model, tokenizer)
            print(f"Kumaoni: {translation}")
    
    elif args.text:
        translation = translate(args.text, model, tokenizer)
        print(f"Hinglish: {args.text}")
        print(f"Kumaoni: {translation}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
