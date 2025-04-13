#!/bin/bash
# Run the Kumaoni chatbot

cd "$(dirname "$0")/.."
python src/chatbot.py "$@"
