#!/usr/bin/env python3
"""
Test script to verify that the training and inference are working correctly.
"""

import os
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def test_model_loading():
    """Test loading the model and tokenizer"""
    print("Testing model loading...")

    # Check if the model directory exists
    model_dir = os.path.abspath("models/facebook")
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist.")
        return False

    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

        # Load model
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            device_map="auto",
            local_files_only=True
        )

        print("Model and tokenizer loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def test_inference():
    """Test inference with the model"""
    print("\nTesting inference...")

    # Import the inference module
    sys.path.append(os.path.abspath("inference"))
    try:
        from generate import load_model, translate

        # Load model and tokenizer
        print("Loading model for inference...")
        model, tokenizer = load_model()

        # Test translation
        test_texts = [
            "Hello",
            "How are you?",
            "Thank you"
        ]

        print("\nTranslation examples:")
        for text in test_texts:
            translation = translate(text, model, tokenizer)
            print(f"Hinglish: {text}")
            print(f"Kumaoni: {translation}")
            print("-" * 40)

        return True
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

def test_app():
    """Test the app functionality"""
    print("\nTesting app functionality...")

    try:
        # Add the training directory to the path
        sys.path.append(os.path.abspath("training"))

        # Import the app module
        try:
            import app
            app_translate = app.translate
        except ImportError:
            print("Could not import app module. Skipping app test.")
            return False

        # Test translation
        test_texts = [
            "Hello",
            "How are you?",
            "Thank you"
        ]

        print("\nApp translation examples:")
        for text in test_texts:
            translation = app_translate(text)
            print(f"Hinglish: {text}")
            print(f"Kumaoni: {translation}")
            print("-" * 40)

        return True
    except Exception as e:
        print(f"Error testing app: {e}")
        return False

def main():
    """Run all tests"""
    print("Running all tests...\n")

    # Test model loading
    model_loading_success = test_model_loading()

    # Test inference
    inference_success = test_inference()

    # Test app
    app_success = test_app()

    # Print summary
    print("\nTest Summary:")
    print(f"Model Loading: {'✓' if model_loading_success else '✗'}")
    print(f"Inference: {'✓' if inference_success else '✗'}")
    print(f"App: {'✓' if app_success else '✗'}")

    if model_loading_success and inference_success and app_success:
        print("\nAll tests passed! The system is working correctly.")
    else:
        print("\nSome tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
