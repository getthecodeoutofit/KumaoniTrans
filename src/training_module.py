#!/usr/bin/env python3
"""
Training Module for Kumaoni Chatbot.
This script provides an interactive interface for users to teach the chatbot
new words, phrases, grammar rules, and corrections.
"""

import os
import json
import argparse
import datetime
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VOCAB_MAP_PATH = os.path.join(DATA_DIR, "vocab_mapping.json")
PHRASES_PATH = os.path.join(DATA_DIR, "phrases_mapping.json")
GRAMMAR_RULES_PATH = os.path.join(DATA_DIR, "grammar_rules.json")
IDIOMS_PATH = os.path.join(DATA_DIR, "idioms.json")
TRAINING_LOG_PATH = os.path.join(DATA_DIR, "training_log.json")
CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")
DATASET_PATH = os.path.join(DATA_DIR, "data.json")

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TrainingModule:
    def __init__(self):
        """Initialize the training module"""
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Load data files
        self.data = {
            "vocab": self.load_json(VOCAB_MAP_PATH, {}),
            "phrases": self.load_json(PHRASES_PATH, {}),
            "grammar": self.load_json(GRAMMAR_RULES_PATH, {}),
            "idioms": self.load_json(IDIOMS_PATH, {}),
            "corrections": self.load_json(CORRECTIONS_PATH, {"words": {}, "phrases": {}}),
            "training_log": self.load_json(TRAINING_LOG_PATH, {"sessions": []}),
            "dataset": self.load_json(DATASET_PATH, [])
        }
        
        # Initialize training session
        self.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.session_log = {
            "session_id": self.session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "entries": []
        }
        
        print(f"Loaded vocabulary with {len(self.data['vocab'])} words")
        print(f"Loaded phrases with {len(self.data['phrases'])} phrases")
        print(f"Loaded {len(self.data['idioms'])} idioms")
        print(f"Training module initialized successfully!")
    
    def load_json(self, file_path, default_value):
        """Load JSON data from file, or return default if file doesn't exist"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create the file with default value
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_value, f, ensure_ascii=False, indent=2)
                return default_value
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return default_value
    
    def save_json(self, file_path, data):
        """Save JSON data to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
            return False
    
    def add_word(self, hinglish, kumaoni):
        """Add a new word to the vocabulary"""
        hinglish = hinglish.lower().strip()
        kumaoni = kumaoni.strip()
        
        # Check if word already exists
        if hinglish in self.data["vocab"]:
            existing = self.data["vocab"][hinglish]
            if existing == kumaoni:
                print(f"Word '{hinglish}' already exists with translation '{kumaoni}'")
                return False
            else:
                # Add to corrections
                if hinglish not in self.data["corrections"]["words"]:
                    self.data["corrections"]["words"][hinglish] = []
                
                self.data["corrections"]["words"][hinglish].append({
                    "old": existing,
                    "new": kumaoni,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                print(f"Updated word '{hinglish}' from '{existing}' to '{kumaoni}'")
        
        # Add to vocabulary
        self.data["vocab"][hinglish] = kumaoni
        
        # Save vocabulary
        self.save_json(VOCAB_MAP_PATH, self.data["vocab"])
        
        # Save corrections if needed
        if "corrections" in self.data and self.data["corrections"]["words"]:
            self.save_json(CORRECTIONS_PATH, self.data["corrections"])
        
        # Log the addition
        self.session_log["entries"].append({
            "type": "word",
            "hinglish": hinglish,
            "kumaoni": kumaoni,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        print(f"Added word: {hinglish} → {kumaoni}")
        return True
    
    def add_phrase(self, hinglish, kumaoni):
        """Add a new phrase to the phrases mapping"""
        hinglish = hinglish.lower().strip()
        kumaoni = kumaoni.strip()
        
        # Check if phrase already exists
        if hinglish in self.data["phrases"]:
            existing = self.data["phrases"][hinglish]
            if existing == kumaoni:
                print(f"Phrase '{hinglish}' already exists with translation '{kumaoni}'")
                return False
            else:
                # Add to corrections
                if hinglish not in self.data["corrections"]["phrases"]:
                    self.data["corrections"]["phrases"][hinglish] = []
                
                self.data["corrections"]["phrases"][hinglish].append({
                    "old": existing,
                    "new": kumaoni,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                print(f"Updated phrase '{hinglish}' from '{existing}' to '{kumaoni}'")
        
        # Add to phrases
        self.data["phrases"][hinglish] = kumaoni
        
        # Save phrases
        self.save_json(PHRASES_PATH, self.data["phrases"])
        
        # Save corrections if needed
        if "corrections" in self.data and self.data["corrections"]["phrases"]:
            self.save_json(CORRECTIONS_PATH, self.data["corrections"])
        
        # Log the addition
        self.session_log["entries"].append({
            "type": "phrase",
            "hinglish": hinglish,
            "kumaoni": kumaoni,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        print(f"Added phrase: {hinglish} → {kumaoni}")
        return True
    
    def add_idiom(self, kumaoni, meaning):
        """Add a new idiom"""
        kumaoni = kumaoni.strip()
        meaning = meaning.strip()
        
        # Add to idioms
        self.data["idioms"][kumaoni] = meaning
        
        # Save idioms
        self.save_json(IDIOMS_PATH, self.data["idioms"])
        
        # Log the addition
        self.session_log["entries"].append({
            "type": "idiom",
            "kumaoni": kumaoni,
            "meaning": meaning,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        print(f"Added idiom: {kumaoni} (meaning: {meaning})")
        return True
    
    def add_example(self, hinglish, kumaoni):
        """Add a new example to the dataset"""
        hinglish = hinglish.strip()
        kumaoni = kumaoni.strip()
        
        # Check if example already exists
        for item in self.data["dataset"]:
            if item["hinglish"] == hinglish and item["kumaoni"] == kumaoni:
                print(f"Example already exists in the dataset")
                return False
        
        # Add to dataset
        self.data["dataset"].append({
            "hinglish": hinglish,
            "kumaoni": kumaoni
        })
        
        # Save dataset
        self.save_json(DATASET_PATH, self.data["dataset"])
        
        # Log the addition
        self.session_log["entries"].append({
            "type": "example",
            "hinglish": hinglish,
            "kumaoni": kumaoni,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        print(f"Added example to dataset")
        return True
    
    def add_grammar_rule(self, category, hinglish, kumaoni):
        """Add a new grammar rule"""
        hinglish = hinglish.lower().strip()
        kumaoni = kumaoni.strip()
        
        # Ensure category exists
        if category not in self.data["grammar"]:
            self.data["grammar"][category] = {}
        
        # Add to grammar rules
        self.data["grammar"][category][hinglish] = kumaoni
        
        # Save grammar rules
        self.save_json(GRAMMAR_RULES_PATH, self.data["grammar"])
        
        # Log the addition
        self.session_log["entries"].append({
            "type": "grammar",
            "category": category,
            "hinglish": hinglish,
            "kumaoni": kumaoni,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        print(f"Added grammar rule: {category} - {hinglish} → {kumaoni}")
        return True
    
    def save_training_log(self):
        """Save the training log"""
        if self.session_log["entries"]:
            self.data["training_log"]["sessions"].append(self.session_log)
            self.save_json(TRAINING_LOG_PATH, self.data["training_log"])
            print(f"Saved training log with {len(self.session_log['entries'])} entries")
    
    def bulk_import(self, file_path):
        """Import data in bulk from a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            words_added = 0
            phrases_added = 0
            examples_added = 0
            
            # Import words
            if "words" in import_data:
                for hinglish, kumaoni in import_data["words"].items():
                    if self.add_word(hinglish, kumaoni):
                        words_added += 1
            
            # Import phrases
            if "phrases" in import_data:
                for hinglish, kumaoni in import_data["phrases"].items():
                    if self.add_phrase(hinglish, kumaoni):
                        phrases_added += 1
            
            # Import examples
            if "examples" in import_data:
                for example in import_data["examples"]:
                    if "hinglish" in example and "kumaoni" in example:
                        if self.add_example(example["hinglish"], example["kumaoni"]):
                            examples_added += 1
            
            print(f"Bulk import completed: {words_added} words, {phrases_added} phrases, {examples_added} examples")
            return True
        
        except Exception as e:
            print(f"Error during bulk import: {e}")
            return False
    
    def export_data(self, file_path):
        """Export all data to a JSON file"""
        try:
            export_data = {
                "vocab": self.data["vocab"],
                "phrases": self.data["phrases"],
                "grammar": self.data["grammar"],
                "idioms": self.data["idioms"],
                "dataset": self.data["dataset"]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"Data exported to {file_path}")
            return True
        
        except Exception as e:
            print(f"Error during export: {e}")
            return False
    
    def search(self, query):
        """Search for words, phrases, or idioms"""
        query = query.lower().strip()
        results = {
            "words": [],
            "phrases": [],
            "idioms": []
        }
        
        # Search in vocabulary
        for hinglish, kumaoni in self.data["vocab"].items():
            if query in hinglish.lower() or query in kumaoni.lower():
                results["words"].append({
                    "hinglish": hinglish,
                    "kumaoni": kumaoni
                })
        
        # Search in phrases
        for hinglish, kumaoni in self.data["phrases"].items():
            if query in hinglish.lower() or query in kumaoni.lower():
                results["phrases"].append({
                    "hinglish": hinglish,
                    "kumaoni": kumaoni
                })
        
        # Search in idioms
        for kumaoni, meaning in self.data["idioms"].items():
            if query in kumaoni.lower() or query in meaning.lower():
                results["idioms"].append({
                    "kumaoni": kumaoni,
                    "meaning": meaning
                })
        
        return results
    
    def interactive_training(self):
        """Run an interactive training session"""
        print(f"{Colors.HEADER}=== Kumaoni Chatbot Training Module ==={Colors.ENDC}")
        print("This module helps you teach the chatbot new words, phrases, and grammar rules.")
        print("Type 'exit' at any prompt to return to the main menu.")
        
        while True:
            print(f"\n{Colors.BOLD}Available commands:{Colors.ENDC}")
            print(f"  {Colors.CYAN}1{Colors.ENDC} - Add a word")
            print(f"  {Colors.CYAN}2{Colors.ENDC} - Add a phrase")
            print(f"  {Colors.CYAN}3{Colors.ENDC} - Add an idiom")
            print(f"  {Colors.CYAN}4{Colors.ENDC} - Add an example")
            print(f"  {Colors.CYAN}5{Colors.ENDC} - Add a grammar rule")
            print(f"  {Colors.CYAN}6{Colors.ENDC} - Search")
            print(f"  {Colors.CYAN}7{Colors.ENDC} - Bulk import")
            print(f"  {Colors.CYAN}8{Colors.ENDC} - Export data")
            print(f"  {Colors.CYAN}9{Colors.ENDC} - Exit")
            
            choice = input(f"\n{Colors.BOLD}Enter your choice (1-9):{Colors.ENDC} ")
            
            if choice == "9" or choice.lower() == "exit":
                break
            
            elif choice == "1":
                print(f"\n{Colors.YELLOW}Adding a new word:{Colors.ENDC}")
                hinglish = input("Enter Hinglish word: ")
                if hinglish.lower() == "exit":
                    continue
                
                kumaoni = input("Enter Kumaoni translation: ")
                if kumaoni.lower() == "exit":
                    continue
                
                self.add_word(hinglish, kumaoni)
            
            elif choice == "2":
                print(f"\n{Colors.YELLOW}Adding a new phrase:{Colors.ENDC}")
                hinglish = input("Enter Hinglish phrase: ")
                if hinglish.lower() == "exit":
                    continue
                
                kumaoni = input("Enter Kumaoni translation: ")
                if kumaoni.lower() == "exit":
                    continue
                
                self.add_phrase(hinglish, kumaoni)
            
            elif choice == "3":
                print(f"\n{Colors.YELLOW}Adding a new idiom:{Colors.ENDC}")
                kumaoni = input("Enter Kumaoni idiom: ")
                if kumaoni.lower() == "exit":
                    continue
                
                meaning = input("Enter meaning: ")
                if meaning.lower() == "exit":
                    continue
                
                self.add_idiom(kumaoni, meaning)
            
            elif choice == "4":
                print(f"\n{Colors.YELLOW}Adding a new example:{Colors.ENDC}")
                hinglish = input("Enter Hinglish sentence: ")
                if hinglish.lower() == "exit":
                    continue
                
                kumaoni = input("Enter Kumaoni translation: ")
                if kumaoni.lower() == "exit":
                    continue
                
                self.add_example(hinglish, kumaoni)
            
            elif choice == "5":
                print(f"\n{Colors.YELLOW}Adding a new grammar rule:{Colors.ENDC}")
                print("Available categories:")
                for category in self.data["grammar"].keys():
                    print(f"  - {category}")
                print("  - [new category]")
                
                category = input("Enter category: ")
                if category.lower() == "exit":
                    continue
                
                hinglish = input("Enter Hinglish form: ")
                if hinglish.lower() == "exit":
                    continue
                
                kumaoni = input("Enter Kumaoni form: ")
                if kumaoni.lower() == "exit":
                    continue
                
                self.add_grammar_rule(category, hinglish, kumaoni)
            
            elif choice == "6":
                print(f"\n{Colors.YELLOW}Search:{Colors.ENDC}")
                query = input("Enter search query: ")
                if query.lower() == "exit":
                    continue
                
                results = self.search(query)
                
                if results["words"]:
                    print(f"\n{Colors.GREEN}Words:{Colors.ENDC}")
                    for item in results["words"]:
                        print(f"  {item['hinglish']} → {item['kumaoni']}")
                
                if results["phrases"]:
                    print(f"\n{Colors.GREEN}Phrases:{Colors.ENDC}")
                    for item in results["phrases"]:
                        print(f"  {item['hinglish']} → {item['kumaoni']}")
                
                if results["idioms"]:
                    print(f"\n{Colors.GREEN}Idioms:{Colors.ENDC}")
                    for item in results["idioms"]:
                        print(f"  {item['kumaoni']} (meaning: {item['meaning']})")
                
                if not any(results.values()):
                    print("No results found.")
            
            elif choice == "7":
                print(f"\n{Colors.YELLOW}Bulk Import:{Colors.ENDC}")
                file_path = input("Enter path to JSON file: ")
                if file_path.lower() == "exit":
                    continue
                
                self.bulk_import(file_path)
            
            elif choice == "8":
                print(f"\n{Colors.YELLOW}Export Data:{Colors.ENDC}")
                file_path = input("Enter path for export file: ")
                if file_path.lower() == "exit":
                    continue
                
                self.export_data(file_path)
            
            else:
                print(f"{Colors.RED}Invalid choice. Please enter a number from 1 to 9.{Colors.ENDC}")
        
        # Save training log before exiting
        self.save_training_log()
        print("Training session completed!")

def main():
    parser = argparse.ArgumentParser(description="Training Module for Kumaoni Chatbot")
    parser.add_argument("--import", dest="import_file", type=str, help="Import data from JSON file")
    parser.add_argument("--export", dest="export_file", type=str, help="Export data to JSON file")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    
    args = parser.parse_args()
    
    # Initialize training module
    training = TrainingModule()
    
    if args.import_file:
        training.bulk_import(args.import_file)
    elif args.export_file:
        training.export_data(args.export_file)
    else:
        # Run interactive training
        training.interactive_training()

if __name__ == "__main__":
    main()
