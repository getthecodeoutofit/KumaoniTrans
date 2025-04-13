import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from langdetect import detect
import gradio as gr
import os

# Local paths
NLLB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nllb-kumaoni")
GEMMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/gemma")
FACEBOOK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/facebook")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4-bit quantization setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # highest accuracy for 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

# Load NLLB model and tokenizer
try:
    print(f"Loading NLLB model from {NLLB_PATH}...")
    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_PATH, local_files_only=True)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    nllb_available = True
except Exception as e:
    print(f"Error loading NLLB model: {e}")
    print("Falling back to Facebook model...")
    try:
        nllb_tokenizer = AutoTokenizer.from_pretrained(FACEBOOK_PATH, local_files_only=True)
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            FACEBOOK_PATH,
            device_map="auto",
            quantization_config=bnb_config,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        nllb_available = True
    except Exception as e:
        print(f"Error loading Facebook model: {e}")
        nllb_available = False

# Load Gemma model and tokenizer
try:
    print(f"Loading Gemma model from {GEMMA_PATH}...")
    gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, local_files_only=True)
    gemma_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    gemma_available = True
except Exception as e:
    print(f"Error loading Gemma model: {e}")
    gemma_available = False

# Translation function
def translate(text):
    if not nllb_available:
        return f"[Translation not available: {text}]"

    try:
        # Format the input prompt to match training format
        prompt = f"Translate from Hinglish to Kumaoni language: {text}"

        # Tokenize and generate
        inputs = nllb_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=64,
            truncation=True
        ).to(nllb_model.device)
        outputs = nllb_model.generate(**inputs, max_new_tokens=50)
        generated_text = nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the Kumaoni part
        if "Kumaoni:" in generated_text:
            translation = generated_text.split("Kumaoni:")[-1].strip()
        else:
            translation = generated_text.strip()

        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return f"[Translation error: {text}]"

# Chat function
def respond(message, history):
    # Detect language and translate if needed
    try:
        if message.strip() and detect(message) in ["hi", "en"]:
            translated = translate(message)
            message = f"User said: {translated}"
    except Exception as e:
        print(f"Language detection error: {e}")

    # Generate response
    if not gemma_available:
        return "I'm sorry, the Kumaoni assistant is not available right now."

    try:
        prompt = f"Respond in Kumaoni to: {message}"
        inputs = gemma_tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(gemma_model.device)
        outputs = gemma_model.generate(**inputs, max_new_tokens=80, temperature=0.7)
        return gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Response generation error: {e}")
        return "I'm sorry, I couldn't generate a response."

# Launch Gradio app
gr.ChatInterface(
    respond,
    examples=["Namaste", "What's Almora famous for?"],
    title="Kumaoni Assistant"
).launch()
