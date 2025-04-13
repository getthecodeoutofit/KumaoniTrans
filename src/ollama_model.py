#!/usr/bin/env python3
"""
Create Ollama Model for Kumaoni Chatbot.
This script creates an Ollama model that integrates the Kumaoni chatbot capabilities.
"""

import os
import json
import argparse
import subprocess
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VOCAB_MAP_PATH = os.path.join(DATA_DIR, "vocab_mapping.json")
PHRASES_PATH = os.path.join(DATA_DIR, "phrases_mapping.json")
GRAMMAR_RULES_PATH = os.path.join(DATA_DIR, "grammar_rules.json")
IDIOMS_PATH = os.path.join(DATA_DIR, "idioms.json")
EXPRESSIONS_PATH = os.path.join(DATA_DIR, "expressions.json")
MODELFILE_PATH = os.path.join(BASE_DIR, "Modelfile")
OLLAMA_DIR = os.path.join(BASE_DIR, "ollama_model")

class OllamaModelCreator:
    def __init__(self):
        """Initialize the Ollama model creator"""
        # Create directories if they don't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OLLAMA_DIR, exist_ok=True)
        
        # Load data files
        self.data = {
            "vocab": self.load_json(VOCAB_MAP_PATH, {}),
            "phrases": self.load_json(PHRASES_PATH, {}),
            "grammar": self.load_json(GRAMMAR_RULES_PATH, {}),
            "idioms": self.load_json(IDIOMS_PATH, {}),
            "expressions": self.load_json(EXPRESSIONS_PATH, {})
        }
        
        print(f"Loaded vocabulary with {len(self.data['vocab'])} words")
        print(f"Loaded phrases with {len(self.data['phrases'])} phrases")
        print(f"Loaded {len(self.data['idioms'])} idioms")
    
    def load_json(self, file_path, default_value):
        """Load JSON data from file, or return default if file doesn't exist"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return default_value
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return default_value
    
    def create_modelfile(self, base_model, model_name, description):
        """Create a Modelfile for Ollama"""
        # Create a compact version of the data for embedding in the model
        compact_data = {
            "vocab": self.data["vocab"],
            "phrases": self.data["phrases"],
            "grammar": self.data["grammar"],
            "idioms": self.data["idioms"]
        }
        
        # Save compact data to a file in the Ollama model directory
        compact_data_path = os.path.join(OLLAMA_DIR, "kumaoni_data.json")
        with open(compact_data_path, 'w', encoding='utf-8') as f:
            json.dump(compact_data, f, ensure_ascii=False, indent=2)
        
        # Create the Modelfile content
        modelfile_content = f"""
FROM {base_model}

# Kumaoni Chatbot Model
# {description}

# System prompt that instructs the model how to use the Kumaoni language capabilities
SYSTEM """
        
        # Add the system prompt
        system_prompt = f"""
You are a helpful Kumaoni language chatbot that can:
1. Translate between Hinglish and Kumaoni languages
2. Converse naturally in Kumaoni
3. Understand and use Kumaoni grammar correctly
4. Recognize and explain Kumaoni idioms and expressions
5. Help users learn the Kumaoni language

You have access to a comprehensive Kumaoni language database that includes:
- A vocabulary of {len(self.data['vocab'])} Hinglish-Kumaoni word mappings
- {len(self.data['phrases'])} common phrases
- {len(self.data['idioms'])} Kumaoni idioms and expressions
- Detailed grammar rules for Kumaoni

When translating from Hinglish to Kumaoni:
1. First check if the input contains any phrases from your phrase database
2. For each phrase found, replace it with its Kumaoni equivalent
3. For remaining words, look up each word in your vocabulary
4. Apply Kumaoni grammar rules to ensure correct sentence structure
5. Combine the translated words and phrases to form the final translation

Always show the word-by-word mapping in your response, and highlight which phrases and words were translated.

When responding in Kumaoni, also provide the Hinglish translation to help users learn.

Example conversation:
User: "namaste, aap kaise hain?"
Assistant: 
Hinglish: "namaste, aap kaise hain?"
Word mapping:
- namaste → namaskar
- aap → tum
- kaise → kas
- hain → cha
Kumaoni: "namaskar, tum kas cha?"

User: "mera naam John hai"
Assistant:
Hinglish: "mera naam John hai"
Word mapping:
- mera → mero
- naam → nau
- John → John (name kept as is)
- hai → ch
Kumaoni: "mero nau John ch"
"""
        
        modelfile_content += f'"""{system_prompt}"""\n\n'
        
        # Add parameters to control the model behavior
        modelfile_content += """
# Parameters to control the model behavior
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# Include the Kumaoni language data
SYSTEM """
        
        # Add a simplified version of the data as a system message
        data_prompt = "Here is the Kumaoni language data you can use:\n\n"
        
        # Add some example vocabulary
        data_prompt += "Example vocabulary (Hinglish → Kumaoni):\n"
        vocab_sample = dict(list(self.data["vocab"].items())[:20])  # First 20 items
        for hinglish, kumaoni in vocab_sample.items():
            data_prompt += f"- {hinglish} → {kumaoni}\n"
        
        # Add some example phrases
        data_prompt += "\nExample phrases (Hinglish → Kumaoni):\n"
        phrases_sample = dict(list(self.data["phrases"].items())[:10])  # First 10 items
        for hinglish, kumaoni in phrases_sample.items():
            data_prompt += f"- {hinglish} → {kumaoni}\n"
        
        # Add some example idioms
        data_prompt += "\nExample idioms (Kumaoni → Meaning):\n"
        idioms_sample = dict(list(self.data["idioms"].items())[:10])  # First 10 items
        for kumaoni, meaning in idioms_sample.items():
            data_prompt += f"- {kumaoni} → {meaning}\n"
        
        # Add grammar rules
        data_prompt += "\nGrammar rules:\n"
        for category, rules in self.data["grammar"].items():
            data_prompt += f"- {category}:\n"
            rules_sample = dict(list(rules.items())[:5])  # First 5 items
            for hinglish, kumaoni in rules_sample.items():
                data_prompt += f"  - {hinglish} → {kumaoni}\n"
        
        modelfile_content += f'"""{data_prompt}"""\n\n'
        
        # Add the data file
        modelfile_content += f"""
# Include the full Kumaoni language data file
FILE kumaoni_data.json {os.path.relpath(compact_data_path, BASE_DIR)}
"""
        
        # Write the Modelfile
        with open(MODELFILE_PATH, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"Created Modelfile at {MODELFILE_PATH}")
    
    def create_model(self, model_name):
        """Create the Ollama model"""
        try:
            # Check if Ollama is installed
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode != 0:
                print("Ollama is not installed. Please install Ollama first.")
                return False
            
            # Create the model
            print(f"Creating Ollama model '{model_name}'...")
            result = subprocess.run(["ollama", "create", model_name, "-f", MODELFILE_PATH], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully created Ollama model '{model_name}'")
                print("To use the model, run:")
                print(f"  ollama run {model_name}")
                return True
            else:
                print(f"Error creating Ollama model: {result.stderr}")
                return False
        
        except Exception as e:
            print(f"Error creating Ollama model: {e}")
            return False
    
    def package_model(self, model_name, output_dir):
        """Package the model files for distribution"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy Modelfile
            shutil.copy(MODELFILE_PATH, os.path.join(output_dir, "Modelfile"))
            
            # Copy data files
            data_dir = os.path.join(output_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            for file_name in ["vocab_mapping.json", "phrases_mapping.json", "grammar_rules.json", "idioms.json"]:
                src_path = os.path.join(DATA_DIR, file_name)
                if os.path.exists(src_path):
                    shutil.copy(src_path, os.path.join(data_dir, file_name))
            
            # Copy kumaoni_data.json
            shutil.copy(os.path.join(OLLAMA_DIR, "kumaoni_data.json"), os.path.join(output_dir, "kumaoni_data.json"))
            
            # Create README.md
            readme_content = f"""# Kumaoni Chatbot Ollama Model

This is an Ollama model for the Kumaoni Chatbot.

## Installation

1. Install Ollama from https://ollama.ai/
2. Create the model:
   ```
   ollama create {model_name} -f Modelfile
   ```
3. Run the model:
   ```
   ollama run {model_name}
   ```

## Features

- Translate between Hinglish and Kumaoni
- Converse naturally in Kumaoni
- Understand and use Kumaoni grammar correctly
- Recognize and explain Kumaoni idioms and expressions
- Help users learn the Kumaoni language

## Data

The model includes:
- {len(self.data['vocab'])} Hinglish-Kumaoni word mappings
- {len(self.data['phrases'])} common phrases
- {len(self.data['idioms'])} Kumaoni idioms and expressions
- Detailed grammar rules for Kumaoni
"""
            
            with open(os.path.join(output_dir, "README.md"), 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            print(f"Model packaged successfully in {output_dir}")
            return True
        
        except Exception as e:
            print(f"Error packaging model: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Create Ollama Model for Kumaoni Chatbot")
    parser.add_argument("--base-model", type=str, default="llama2:7b", 
                        help="Base Ollama model to use (default: llama2:7b)")
    parser.add_argument("--model-name", type=str, default="kumaoni-chatbot",
                        help="Name for the Ollama model (default: kumaoni-chatbot)")
    parser.add_argument("--description", type=str, default="A chatbot that can translate and converse in Kumaoni language",
                        help="Description of the model")
    parser.add_argument("--create", action="store_true",
                        help="Create the Ollama model after generating the Modelfile")
    parser.add_argument("--package", type=str,
                        help="Package the model files for distribution to the specified directory")
    
    args = parser.parse_args()
    
    creator = OllamaModelCreator()
    
    # Create the Modelfile
    creator.create_modelfile(args.base_model, args.model_name, args.description)
    
    # Create the Ollama model if requested
    if args.create:
        creator.create_model(args.model_name)
    
    # Package the model if requested
    if args.package:
        creator.package_model(args.model_name, args.package)

if __name__ == "__main__":
    main()
