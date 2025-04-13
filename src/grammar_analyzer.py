#!/usr/bin/env python3
"""
Grammar Analyzer for Kumaoni language.
This script analyzes the dataset to extract grammar patterns and rules
for the Kumaoni language to improve the chatbot's grammar recognition.
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
GRAMMAR_RULES_PATH = os.path.join(DATA_DIR, "grammar_rules.json")
PATTERNS_PATH = os.path.join(DATA_DIR, "patterns.json")
VERB_FORMS_PATH = os.path.join(DATA_DIR, "verb_forms.json")
NOUN_FORMS_PATH = os.path.join(DATA_DIR, "noun_forms.json")
SENTENCE_STRUCTURES_PATH = os.path.join(DATA_DIR, "sentence_structures.json")

class GrammarAnalyzer:
    def __init__(self):
        """Initialize the grammar analyzer"""
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Load dataset
        self.dataset = self.load_dataset()
        
        # Initialize grammar components
        self.grammar_rules = self.load_json(GRAMMAR_RULES_PATH, {})
        self.patterns = self.load_json(PATTERNS_PATH, {})
        self.verb_forms = self.load_json(VERB_FORMS_PATH, {})
        self.noun_forms = self.load_json(NOUN_FORMS_PATH, {})
        self.sentence_structures = self.load_json(SENTENCE_STRUCTURES_PATH, {})
        
        # Common verb endings in Hindi/Hinglish
        self.hinglish_verb_endings = ["na", "ta", "te", "ti", "ya", "ye", "yi", "a", "e", "i", "o", "u"]
        self.kumaoni_verb_endings = ["no", "to", "ta", "ti", "yo", "ya", "yi", "o", "a", "i", "u"]
        
        # Common postpositions
        self.hinglish_postpositions = ["ka", "ke", "ki", "ko", "se", "me", "par", "tak"]
        self.kumaoni_postpositions = ["ko", "ka", "ki", "ku", "le", "ma", "par", "tak"]
        
        # Common pronouns
        self.hinglish_pronouns = ["main", "mujhe", "mera", "meri", "hum", "hamara", "tu", "tum", "tumhara", 
                                 "aap", "aapka", "woh", "uska", "uski", "ye", "iska", "iski"]
        
        # Common question words
        self.hinglish_question_words = ["kya", "kaun", "kahan", "kaise", "kyun", "kitna", "kitne", "kitni", "kab"]
    
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
    
    def analyze_verb_endings(self):
        """Analyze verb endings in the dataset"""
        verb_endings_map = defaultdict(Counter)
        
        for item in self.dataset:
            hinglish = item["hinglish"]
            kumaoni = item["kumaoni"]
            
            hinglish_words = hinglish.split()
            kumaoni_words = kumaoni.split()
            
            # Only process if the number of words match (for direct mapping)
            if len(hinglish_words) == len(kumaoni_words):
                for h_word, k_word in zip(hinglish_words, kumaoni_words):
                    h_word = h_word.lower().strip(".,?!\"'")
                    k_word = k_word.lower().strip(".,?!\"'")
                    
                    # Check for verb endings
                    for h_ending in self.hinglish_verb_endings:
                        if h_word.endswith(h_ending) and len(h_word) > len(h_ending):
                            # Find the corresponding Kumaoni ending
                            for k_ending in self.kumaoni_verb_endings:
                                if k_word.endswith(k_ending) and len(k_word) > len(k_ending):
                                    verb_endings_map[h_ending][k_ending] += 1
        
        # Convert to most common endings
        verb_endings = {}
        for h_ending, k_endings in verb_endings_map.items():
            if k_endings:
                most_common = k_endings.most_common(1)[0]
                verb_endings[h_ending] = most_common[0]
        
        return verb_endings
    
    def analyze_postpositions(self):
        """Analyze postpositions in the dataset"""
        postpositions_map = defaultdict(Counter)
        
        for item in self.dataset:
            hinglish = item["hinglish"]
            kumaoni = item["kumaoni"]
            
            hinglish_words = hinglish.split()
            kumaoni_words = kumaoni.split()
            
            # Only process if the number of words match (for direct mapping)
            if len(hinglish_words) == len(kumaoni_words):
                for h_word, k_word in zip(hinglish_words, kumaoni_words):
                    h_word = h_word.lower().strip(".,?!\"'")
                    k_word = k_word.lower().strip(".,?!\"'")
                    
                    # Check for postpositions
                    if h_word in self.hinglish_postpositions:
                        postpositions_map[h_word][k_word] += 1
        
        # Convert to most common postpositions
        postpositions = {}
        for h_post, k_posts in postpositions_map.items():
            if k_posts:
                most_common = k_posts.most_common(1)[0]
                postpositions[h_post] = most_common[0]
        
        return postpositions
    
    def analyze_pronouns(self):
        """Analyze pronouns in the dataset"""
        pronouns_map = defaultdict(Counter)
        
        for item in self.dataset:
            hinglish = item["hinglish"]
            kumaoni = item["kumaoni"]
            
            hinglish_words = hinglish.split()
            kumaoni_words = kumaoni.split()
            
            # Only process if the number of words match (for direct mapping)
            if len(hinglish_words) == len(kumaoni_words):
                for h_word, k_word in zip(hinglish_words, kumaoni_words):
                    h_word = h_word.lower().strip(".,?!\"'")
                    k_word = k_word.lower().strip(".,?!\"'")
                    
                    # Check for pronouns
                    if h_word in self.hinglish_pronouns:
                        pronouns_map[h_word][k_word] += 1
        
        # Convert to most common pronouns
        pronouns = {}
        for h_pron, k_prons in pronouns_map.items():
            if k_prons:
                most_common = k_prons.most_common(1)[0]
                pronouns[h_pron] = most_common[0]
        
        return pronouns
    
    def analyze_question_words(self):
        """Analyze question words in the dataset"""
        question_words_map = defaultdict(Counter)
        
        for item in self.dataset:
            hinglish = item["hinglish"]
            kumaoni = item["kumaoni"]
            
            hinglish_words = hinglish.split()
            kumaoni_words = kumaoni.split()
            
            # Only process if the number of words match (for direct mapping)
            if len(hinglish_words) == len(kumaoni_words):
                for h_word, k_word in zip(hinglish_words, kumaoni_words):
                    h_word = h_word.lower().strip(".,?!\"'")
                    k_word = k_word.lower().strip(".,?!\"'")
                    
                    # Check for question words
                    if h_word in self.hinglish_question_words:
                        question_words_map[h_word][k_word] += 1
        
        # Convert to most common question words
        question_words = {}
        for h_qw, k_qws in question_words_map.items():
            if k_qws:
                most_common = k_qws.most_common(1)[0]
                question_words[h_qw] = most_common[0]
        
        return question_words
    
    def analyze_verb_forms(self):
        """Analyze verb forms in the dataset"""
        verb_forms = defaultdict(lambda: defaultdict(Counter))
        
        # Common verb roots in Hindi/Hinglish
        common_verbs = ["kar", "ho", "ja", "aa", "de", "le", "bol", "dekh", "sun", "kha", "pi", "so", "mil", "likh", "padh"]
        
        for item in self.dataset:
            hinglish = item["hinglish"]
            kumaoni = item["kumaoni"]
            
            hinglish_words = hinglish.split()
            kumaoni_words = kumaoni.split()
            
            # Only process if the number of words match (for direct mapping)
            if len(hinglish_words) == len(kumaoni_words):
                for h_word, k_word in zip(hinglish_words, kumaoni_words):
                    h_word = h_word.lower().strip(".,?!\"'")
                    k_word = k_word.lower().strip(".,?!\"'")
                    
                    # Check for verb forms
                    for verb in common_verbs:
                        if h_word.startswith(verb):
                            suffix = h_word[len(verb):]
                            verb_forms[verb][suffix][k_word] += 1
        
        # Convert to most common forms
        result = {}
        for verb, suffixes in verb_forms.items():
            result[verb] = {}
            for suffix, translations in suffixes.items():
                if translations:
                    most_common = translations.most_common(1)[0]
                    result[verb][suffix] = most_common[0]
        
        return result
    
    def analyze_sentence_structures(self):
        """Analyze sentence structures in the dataset"""
        structures = defaultdict(Counter)
        
        for item in self.dataset:
            hinglish = item["hinglish"]
            kumaoni = item["kumaoni"]
            
            # Simplify to basic structure (S-V-O, S-O-V, etc.)
            h_structure = self.get_sentence_structure(hinglish)
            k_structure = self.get_sentence_structure(kumaoni)
            
            structures[h_structure][k_structure] += 1
        
        # Convert to most common structures
        result = {}
        for h_struct, k_structs in structures.items():
            if k_structs:
                most_common = k_structs.most_common(1)[0]
                result[h_struct] = most_common[0]
        
        return result
    
    def get_sentence_structure(self, sentence):
        """Get the basic structure of a sentence (simplified)"""
        # This is a simplified approach - in a real system, you'd use POS tagging
        words = sentence.lower().split()
        
        # Check for question
        if any(qw in words for qw in self.hinglish_question_words):
            return "question"
        
        # Check for command (imperative)
        if len(words) > 0 and words[0] not in self.hinglish_pronouns:
            for verb_ending in ["o", "en", "iye"]:
                if words[0].endswith(verb_ending):
                    return "command"
        
        # Default to statement
        return "statement"
    
    def analyze_grammar(self):
        """Analyze grammar patterns in the dataset"""
        print("Analyzing verb endings...")
        verb_endings = self.analyze_verb_endings()
        
        print("Analyzing postpositions...")
        postpositions = self.analyze_postpositions()
        
        print("Analyzing pronouns...")
        pronouns = self.analyze_pronouns()
        
        print("Analyzing question words...")
        question_words = self.analyze_question_words()
        
        print("Analyzing verb forms...")
        verb_forms = self.analyze_verb_forms()
        
        print("Analyzing sentence structures...")
        sentence_structures = self.analyze_sentence_structures()
        
        # Compile grammar rules
        grammar_rules = {
            "verb_endings": verb_endings,
            "postpositions": postpositions,
            "pronouns": pronouns,
            "question_words": question_words
        }
        
        # Save results
        self.save_json(GRAMMAR_RULES_PATH, grammar_rules)
        self.save_json(VERB_FORMS_PATH, verb_forms)
        self.save_json(SENTENCE_STRUCTURES_PATH, sentence_structures)
        
        print(f"Saved grammar rules to {GRAMMAR_RULES_PATH}")
        print(f"Saved verb forms to {VERB_FORMS_PATH}")
        print(f"Saved sentence structures to {SENTENCE_STRUCTURES_PATH}")
        
        return {
            "grammar_rules": grammar_rules,
            "verb_forms": verb_forms,
            "sentence_structures": sentence_structures
        }
    
    def extract_patterns(self):
        """Extract common patterns from the dataset"""
        patterns = {
            "greetings": [],
            "farewells": [],
            "questions": [],
            "statements": []
        }
        
        # Keywords for categorization
        greeting_words = ["namaste", "namaskar", "hello", "hi", "kaise", "kas", "shubh"]
        farewell_words = ["alvida", "phir", "milenge", "bhetula", "shubh", "ratri", "rati"]
        question_indicators = self.hinglish_question_words
        
        for item in self.dataset:
            hinglish = item["hinglish"]
            kumaoni = item["kumaoni"]
            
            hinglish_lower = hinglish.lower()
            
            # Categorize by pattern
            if any(word in hinglish_lower for word in greeting_words):
                patterns["greetings"].append({"hinglish": hinglish, "kumaoni": kumaoni})
            elif any(word in hinglish_lower for word in farewell_words):
                patterns["farewells"].append({"hinglish": hinglish, "kumaoni": kumaoni})
            elif any(word in hinglish_lower for word in question_indicators) or hinglish_lower.endswith("?"):
                patterns["questions"].append({"hinglish": hinglish, "kumaoni": kumaoni})
            else:
                patterns["statements"].append({"hinglish": hinglish, "kumaoni": kumaoni})
        
        # Save patterns
        self.save_json(PATTERNS_PATH, patterns)
        print(f"Saved patterns to {PATTERNS_PATH}")
        
        # Print statistics
        print(f"Extracted {len(patterns['greetings'])} greeting patterns")
        print(f"Extracted {len(patterns['farewells'])} farewell patterns")
        print(f"Extracted {len(patterns['questions'])} question patterns")
        print(f"Extracted {len(patterns['statements'])} statement patterns")
        
        return patterns

def main():
    parser = argparse.ArgumentParser(description="Grammar Analyzer for Kumaoni language")
    parser.add_argument("--patterns-only", action="store_true", help="Only extract patterns, skip grammar analysis")
    parser.add_argument("--grammar-only", action="store_true", help="Only analyze grammar, skip pattern extraction")
    
    args = parser.parse_args()
    
    analyzer = GrammarAnalyzer()
    
    if not analyzer.dataset:
        print("No dataset available. Cannot analyze grammar.")
        return
    
    if args.patterns_only:
        analyzer.extract_patterns()
    elif args.grammar_only:
        analyzer.analyze_grammar()
    else:
        # Run both analyses
        analyzer.analyze_grammar()
        analyzer.extract_patterns()
    
    print("Grammar analysis completed successfully!")

if __name__ == "__main__":
    main()
