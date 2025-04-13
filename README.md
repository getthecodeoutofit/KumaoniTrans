# Kumaoni Translator and Chatbot

A comprehensive system for Kumaoni language translation, conversation, grammar analysis, and pattern recognition.

## Overview

The Kumaoni Translator and Chatbot is a sophisticated language processing system designed to:

1. **Translate** between Hinglish and Kumaoni with word-to-word mapping
2. **Converse** naturally in Kumaoni language
3. **Analyze** grammar patterns and rules in Kumaoni
4. **Recognize** idioms, expressions, and speech patterns
5. **Learn** new words, phrases, and grammar rules through interactive training
6. **Integrate** with Ollama for easy deployment and use

## Directory Structure

```
KumaoniTrans/
├── README.md                 # This file
├── src/                      # Source code
│   ├── __init__.py           # Package initialization
│   ├── chatbot.py            # Core chatbot functionality
│   ├── grammar_analyzer.py   # Grammar analysis tools
│   ├── pattern_recognizer.py # Pattern recognition tools
│   ├── training_module.py    # Interactive training interface
│   └── ollama_model.py       # Ollama model integration
├── data/                     # Data files
│   ├── vocab_mapping.json    # Hinglish-Kumaoni word mappings
│   ├── phrases_mapping.json  # Phrase mappings
│   ├── grammar_rules.json    # Grammar rules
│   ├── idioms.json           # Kumaoni idioms and expressions
│   └── data.json             # Original dataset
├── models/                   # Model files
│   └── gemma/                # Gemma model directory
└── scripts/                  # Utility scripts
    ├── run_chatbot.sh        # Run the chatbot
    ├── analyze_grammar.sh    # Analyze grammar patterns
    ├── recognize_patterns.sh # Recognize patterns
    ├── train.sh              # Run the training module
    └── create_ollama_model.sh # Create an Ollama model
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/KumaoniTrans.git
   cd KumaoniTrans
   ```

2. Install the required dependencies:
   ```
   pip install torch transformers datasets peft
   ```

3. (Optional) Install Ollama if you want to create an Ollama model:
   ```
   # Follow instructions at https://ollama.ai/
   ```

## Usage

### Running the Chatbot

```bash
./scripts/run_chatbot.sh
```

Options:
- `--language kumaoni|hinglish|mixed`: Set the language preference (default: mixed)
- `--learn`: Enter learning mode to teach new words and phrases
- `--stats`: Show statistics about the chatbot's knowledge
- `--no-color`: Disable colored output

### Chatbot Commands

Once the chatbot is running, you can use the following commands:
- `translate: <text>`: Translate text between Hinglish and Kumaoni
- `learn word: <hinglish> = <kumaoni>`: Teach a new word
- `learn phrase: <hinglish> = <kumaoni>`: Teach a new phrase
- `language: <kumaoni|hinglish|mixed>`: Set language preference
- `exit`: Quit the chatbot

### Analyzing Grammar

```bash
./scripts/analyze_grammar.sh
```

Options:
- `--patterns-only`: Only extract patterns, skip grammar analysis
- `--grammar-only`: Only analyze grammar, skip pattern extraction

### Recognizing Patterns

```bash
./scripts/recognize_patterns.sh
```

Options:
- `--text <text>`: Recognize patterns in the given text

### Training the Chatbot

```bash
./scripts/train.sh
```

Options:
- `--import <file>`: Import data from JSON file
- `--export <file>`: Export data to JSON file
- `--no-color`: Disable colored output

### Creating an Ollama Model

```bash
./scripts/create_ollama_model.sh
```

Options:
- `--base-model <model>`: Base Ollama model to use (default: llama2:7b)
- `--model-name <name>`: Name for the Ollama model (default: kumaoni-chatbot)
- `--description <desc>`: Description of the model
- `--create`: Create the Ollama model after generating the Modelfile
- `--package <dir>`: Package the model files for distribution

## Features

### Translation

The system provides accurate word-to-word translation between Hinglish and Kumaoni, leveraging:
- Vocabulary mapping
- Phrase recognition
- Grammar rules application
- Context-aware translation

### Conversation

The chatbot can engage in natural conversations in Kumaoni, with features like:
- Intent recognition
- Context awareness
- Natural responses
- Language preference settings

### Grammar Analysis

The grammar analyzer extracts patterns and rules from the dataset, including:
- Verb endings
- Postpositions
- Pronouns
- Question words
- Sentence structures

### Pattern Recognition

The pattern recognizer identifies:
- Idioms and expressions
- Common collocations
- Functional phrases (greetings, farewells, etc.)

### Training Module

The interactive training interface allows users to:
- Add new words and phrases
- Define idioms and expressions
- Add grammar rules
- Import and export data
- Search the knowledge base

### Ollama Integration

The system can be packaged as an Ollama model for easy deployment and use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors to the Kumaoni language preservation efforts
- Special thanks to the Gemma model team for providing the base model
