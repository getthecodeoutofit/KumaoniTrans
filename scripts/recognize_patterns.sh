#!/bin/bash
# Recognize patterns in the dataset

cd "$(dirname "$0")/.."
python src/pattern_recognizer.py --analyze "$@"
