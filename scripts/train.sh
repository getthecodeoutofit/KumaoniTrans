#!/bin/bash
# Run the training module

cd "$(dirname "$0")/.."
python src/training_module.py "$@"
