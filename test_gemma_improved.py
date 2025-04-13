#!/usr/bin/env python3
"""
Improved test script for Kumaoni translator using Gemma model.
This script tests the translation functionality of the fine-tuned Gemma model.
"""

import os
import json
from translate_gemma_improved import load_model, translate

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Load some examples from the dataset
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/data.json")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Select a few examples for testing
    test_examples = data[:10]  # Test with 10 examples
    
    print("\nTranslation examples:")
    for item in test_examples:
        hinglish = item["hinglish"]
        expected = item["kumaoni"]
        
        translation = translate(hinglish, model, tokenizer)
        
        print(f"Hinglish: {hinglish}")
        print(f"Expected: {expected}")
        print(f"Generated: {translation}")
        print("-" * 40)
    
    # Test with some custom examples
    custom_examples = [
        "aap kaise hain",
        "mera naam John hai",
        "main Kumaon se hoon",
        "yeh bahut sundar jagah hai",
        "mujhe kumaoni bhasha seekhni hai"
    ]
    
    print("\nCustom examples:")
    for text in custom_examples:
        translation = translate(text, model, tokenizer)
        print(f"Hinglish: {text}")
        print(f"Kumaoni: {translation}")
        print("-" * 40)

if __name__ == "__main__":
    main()
