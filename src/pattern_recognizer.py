#!/usr/bin/env python3
"""
Pattern Recognizer for Kumaoni language.
This script identifies speech patterns, idioms, and expressions in Kumaoni text.
It helps the chatbot understand and generate more natural Kumaoni language.
"""

import os
import json
import argparse
import re
from collections import defaultdict, Counter

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "data.json")
PATTERNS_PATH = os.path.join(DATA_DIR, "patterns.json")
IDIOMS_PATH = os.path.join(DATA_DIR, "idioms.json")
EXPRESSIONS_PATH = os.path.join(DATA_DIR, "expressions.json")
COLLOCATIONS_PATH = os.path.join(DATA_DIR, "collocations.json")

class PatternRecognizer:
    def __init__(self):
        """Initialize the pattern recognizer"""
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Load dataset
        self.dataset = self.load_dataset()
        
        # Initialize pattern components
        self.patterns = self.load_json(PATTERNS_PATH, {})
        self.idioms = self.load_json(IDIOMS_PATH, {})
        self.expressions = self.load_json(EXPRESSIONS_PATH, {})
        self.collocations = self.load_json(COLLOCATIONS_PATH, {})
    
    def load_dataset(self):
        """Load the dataset"""
        if os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded dataset with {len(data)} examples")
            return data
        else:
            print(f"Dataset not found at {DATASET_PATH}")
            return []
    
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
    
    def save_json(self, file_path, data):
        """Save JSON data to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
            return False
    
    def extract_idioms(self):
        """Extract idioms from the dataset"""
        # Idioms are expressions that have meanings different from their literal meanings
        # We'll look for phrases that appear multiple times with consistent translations
        
        # First, collect all phrases (2-4 words) from Kumaoni text
        kumaoni_phrases = defaultdict(list)
        
        for item in self.dataset:
            hinglish = item["hinglish"]
            kumaoni = item["kumaoni"]
            
            kumaoni_words = kumaoni.split()
            
            # Extract phrases of 2-4 words
            for n in range(2, 5):
                if len(kumaoni_words) >= n:
                    for i in range(len(kumaoni_words) - n + 1):
                        phrase = " ".join(kumaoni_words[i:i+n])
                        kumaoni_phrases[phrase].append(hinglish)
        
        # Filter for phrases that appear multiple times with consistent translations
        idioms = {}
        for phrase, translations in kumaoni_phrases.items():
            if len(translations) >= 3:  # Appears at least 3 times
                # Check if translations are consistent
                translation_counter = Counter(translations)
                most_common = translation_counter.most_common(1)[0]
                
                # If the most common translation appears in at least 70% of cases
                if most_common[1] / len(translations) >= 0.7:
                    idioms[phrase] = most_common[0]
        
        # Save idioms
        self.save_json(IDIOMS_PATH, idioms)
        print(f"Saved {len(idioms)} idioms to {IDIOMS_PATH}")
        
        return idioms
    
    def extract_expressions(self):
        """Extract common expressions from the dataset"""
        # Expressions are common phrases used in specific contexts
        # We'll categorize them by context/function
        
        expressions = {
            "greetings": [],
            "farewells": [],
            "thanks": [],
            "apologies": [],
            "questions": [],
            "affirmations": [],
            "negations": []
        }
        
        # Keywords for categorization
        greeting_words = ["namaste", "namaskar", "hello", "hi", "kaise", "kas", "shubh"]
        farewell_words = ["alvida", "phir", "milenge", "bhetula", "shubh", "ratri", "rati"]
        thanks_words = ["dhanyavaad", "shukriya", "thanks"]
        apology_words = ["maaf", "maph", "sorry", "kshama"]
        question_words = ["kya", "kaun", "kahan", "kaise", "kyun", "kitna", "kitne", "kitni", "kab", 
                         "ke", "ko", "kakh", "kas", "kya", "kati"]
        affirmation_words = ["haan", "ho", "yes", "theek", "thik", "sahi"]
        negation_words = ["nahi", "na", "no", "mat"]
        
        for item in self.dataset:
            hinglish = item["hinglish"]
            kumaoni = item["kumaoni"]
            
            hinglish_lower = hinglish.lower()
            kumaoni_lower = kumaoni.lower()
            
            # Categorize by function
            if any(word in hinglish_lower for word in greeting_words) or any(word in kumaoni_lower for word in greeting_words):
                expressions["greetings"].append({"hinglish": hinglish, "kumaoni": kumaoni})
            elif any(word in hinglish_lower for word in farewell_words) or any(word in kumaoni_lower for word in farewell_words):
                expressions["farewells"].append({"hinglish": hinglish, "kumaoni": kumaoni})
            elif any(word in hinglish_lower for word in thanks_words) or any(word in kumaoni_lower for word in thanks_words):
                expressions["thanks"].append({"hinglish": hinglish, "kumaoni": kumaoni})
            elif any(word in hinglish_lower for word in apology_words) or any(word in kumaoni_lower for word in apology_words):
                expressions["apologies"].append({"hinglish": hinglish, "kumaoni": kumaoni})
            elif any(word in hinglish_lower for word in question_words) or any(word in kumaoni_lower for word in question_words) or hinglish_lower.endswith("?"):
                expressions["questions"].append({"hinglish": hinglish, "kumaoni": kumaoni})
            elif any(word in hinglish_lower for word in affirmation_words) or any(word in kumaoni_lower for word in affirmation_words):
                expressions["affirmations"].append({"hinglish": hinglish, "kumaoni": kumaoni})
            elif any(word in hinglish_lower for word in negation_words) or any(word in kumaoni_lower for word in negation_words):
                expressions["negations"].append({"hinglish": hinglish, "kumaoni": kumaoni})
        
        # Save expressions
        self.save_json(EXPRESSIONS_PATH, expressions)
        
        # Print statistics
        total_expressions = sum(len(exps) for exps in expressions.values())
        print(f"Saved {total_expressions} expressions to {EXPRESSIONS_PATH}")
        for category, exps in expressions.items():
            print(f"  - {category}: {len(exps)} expressions")
        
        return expressions
    
    def extract_collocations(self):
        """Extract word collocations from the dataset"""
        # Collocations are words that commonly appear together
        
        collocations = defaultdict(Counter)
        
        for item in self.dataset:
            kumaoni = item["kumaoni"]
            kumaoni_words = kumaoni.lower().split()
            
            # Look for pairs of words
            if len(kumaoni_words) >= 2:
                for i in range(len(kumaoni_words) - 1):
                    word = kumaoni_words[i]
                    next_word = kumaoni_words[i + 1]
                    collocations[word][next_word] += 1
        
        # Filter for significant collocations
        result = {}
        for word, co_words in collocations.items():
            if len(co_words) >= 2:  # Word appears with at least 2 different words
                # Get the top 3 collocations
                top_collocations = co_words.most_common(3)
                result[word] = [{"word": w, "count": c} for w, c in top_collocations if c >= 2]
        
        # Filter out words with no significant collocations
        result = {word: colls for word, colls in result.items() if colls}
        
        # Save collocations
        self.save_json(COLLOCATIONS_PATH, result)
        print(f"Saved collocations for {len(result)} words to {COLLOCATIONS_PATH}")
        
        return result
    
    def analyze_patterns(self):
        """Analyze all patterns in the dataset"""
        print("Extracting idioms...")
        idioms = self.extract_idioms()
        
        print("Extracting expressions...")
        expressions = self.extract_expressions()
        
        print("Extracting collocations...")
        collocations = self.extract_collocations()
        
        return {
            "idioms": idioms,
            "expressions": expressions,
            "collocations": collocations
        }
    
    def recognize_patterns(self, text):
        """Recognize patterns in the given text"""
        results = {
            "idioms": [],
            "expressions": [],
            "collocations": []
        }
        
        # Check for idioms
        for idiom, meaning in self.idioms.items():
            if idiom.lower() in text.lower():
                results["idioms"].append({"idiom": idiom, "meaning": meaning})
        
        # Check for expressions
        for category, expressions_list in self.expressions.items():
            for expression in expressions_list:
                if expression["kumaoni"].lower() in text.lower():
                    results["expressions"].append({
                        "expression": expression["kumaoni"],
                        "hinglish": expression["hinglish"],
                        "category": category
                    })
        
        # Check for collocations
        words = text.lower().split()
        for i in range(len(words) - 1):
            word = words[i]
            next_word = words[i + 1]
            
            if word in self.collocations and any(item["word"] == next_word for item in self.collocations[word]):
                results["collocations"].append({
                    "word": word,
                    "collocate": next_word
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Pattern Recognizer for Kumaoni language")
    parser.add_argument("--analyze", action="store_true", help="Analyze patterns in the dataset")
    parser.add_argument("--text", type=str, help="Recognize patterns in the given text")
    
    args = parser.parse_args()
    
    recognizer = PatternRecognizer()
    
    if not recognizer.dataset and args.analyze:
        print("No dataset available. Cannot analyze patterns.")
        return
    
    if args.analyze:
        recognizer.analyze_patterns()
    
    if args.text:
        results = recognizer.recognize_patterns(args.text)
        
        print("\nPattern Recognition Results:")
        
        if results["idioms"]:
            print("\nIdioms found:")
            for item in results["idioms"]:
                print(f"  - {item['idiom']} (meaning: {item['meaning']})")
        
        if results["expressions"]:
            print("\nExpressions found:")
            for item in results["expressions"]:
                print(f"  - {item['expression']} ({item['category']}, hinglish: {item['hinglish']})")
        
        if results["collocations"]:
            print("\nCollocations found:")
            for item in results["collocations"]:
                print(f"  - {item['word']} {item['collocate']}")
        
        if not any(results.values()):
            print("No patterns recognized in the text.")

if __name__ == "__main__":
    main()
