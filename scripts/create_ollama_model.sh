#!/bin/bash
# Create an Ollama model for the Kumaoni chatbot

cd "$(dirname "$0")/.."
python src/ollama_model.py "$@"
