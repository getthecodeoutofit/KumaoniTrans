#!/usr/bin/env python3
"""
Test script for Kumaoni translator using Gemma model.
This script tests the translation functionality of the fine-tuned Gemma model.
"""

import os
import json
from translate_gemma import load_model, translate

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Load some examples from the dataset
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/data.json")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Select a few examples for testing
    test_examples = data[:5]
    
    print("\nTranslation examples:")
    for item in test_examples:
        hinglish = item["hinglish"]
        expected = item["kumaoni"]
        
        translation = translate(hinglish, model, tokenizer)
        
        print(f"Hinglish: {hinglish}")
        print(f"Expected: {expected}")
        print(f"Generated: {translation}")
        print("-" * 40)

if __name__ == "__main__":
    main()
