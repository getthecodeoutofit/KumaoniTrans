#!/bin/bash
# Analyze grammar patterns in the dataset

cd "$(dirname "$0")/.."
python src/grammar_analyzer.py "$@"
