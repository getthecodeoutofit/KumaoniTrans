#!/usr/bin/env python3
"""
Kumaoni Chatbot - A comprehensive chatbot that can converse in Kumaoni language.
Features:
- Translation between Hinglish and Kumaoni
- Natural conversation in Kumaoni
- Grammar recognition and correction
- Pattern recognition for idioms and expressions
- Training mode to learn new phrases and patterns
- Conversation memory
"""

import os
import json
import re
import argparse
import random
import datetime
from collections import defaultdict

# Paths for data files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VOCAB_MAP_PATH = os.path.join(DATA_DIR, "vocab_mapping.json")
PHRASES_PATH = os.path.join(DATA_DIR, "phrases_mapping.json")
GRAMMAR_RULES_PATH = os.path.join(DATA_DIR, "grammar_rules.json")
PATTERNS_PATH = os.path.join(DATA_DIR, "patterns.json")
CONVERSATIONS_PATH = os.path.join(DATA_DIR, "conversations.json")
IDIOMS_PATH = os.path.join(DATA_DIR, "idioms.json")
CHAT_RESPONSES_PATH = os.path.join(DATA_DIR, "chat_responses.json")
CONVERSATION_HISTORY_PATH = os.path.join(DATA_DIR, "conversation_history.json")

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

class KumaoniChatbot:
    def __init__(self):
        """Initialize the Kumaoni chatbot"""
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Load all data files
        self.data = {
            "vocab": self.load_json(VOCAB_MAP_PATH, {}),
            "phrases": self.load_json(PHRASES_PATH, {}),
            "grammar": self.load_json(GRAMMAR_RULES_PATH, self.default_grammar_rules()),
            "patterns": self.load_json(PATTERNS_PATH, self.default_patterns()),
            "conversations": self.load_json(CONVERSATIONS_PATH, self.default_conversations()),
            "idioms": self.load_json(IDIOMS_PATH, self.default_idioms()),
            "chat_responses": self.load_json(CHAT_RESPONSES_PATH, self.default_chat_responses()),
            "history": self.load_json(CONVERSATION_HISTORY_PATH, {"sessions": []})
        }
        
        # Initialize conversation session
        self.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.conversation_context = {
            "session_id": self.session_id,
            "exchanges": [],
            "current_topic": None,
            "user_name": None,
            "language_preference": "mixed"  # Can be "kumaoni", "hinglish", or "mixed"
        }
        
        print(f"Loaded vocabulary with {len(self.data['vocab'])} words")
        print(f"Loaded phrases with {len(self.data['phrases'])} phrases")
        print(f"Loaded {len(self.data['chat_responses'])} chat response patterns")
        print(f"Loaded {len(self.data['idioms'])} Kumaoni idioms")
        print(f"Kumaoni Chatbot initialized successfully!")
    
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
    
    def default_grammar_rules(self):
        """Default grammar rules for Kumaoni"""
        return {
            "verb_endings": {
                "na": "no",
                "ta": "to",
                "te": "ta",
                "ti": "ti",
                "ya": "yo",
                "a": "o",
                "e": "a",
                "i": "i",
                "o": "o",
                "u": "u"
            },
            "postpositions": {
                "ka": "ko",
                "ke": "ka",
                "ki": "ki",
                "ko": "ku",
                "se": "le",
                "me": "ma",
                "par": "par",
                "tak": "tak"
            },
            "pronouns": {
                "main": "ma",
                "mujhe": "mik",
                "mera": "mero",
                "meri": "meri",
                "hum": "hami",
                "hamara": "hamro",
                "tu": "tu",
                "tum": "tum",
                "tumhara": "tumro",
                "aap": "tum",
                "aapka": "tumar",
                "woh": "u",
                "uska": "usko",
                "uski": "uski",
                "ye": "yo",
                "iska": "isko",
                "iski": "iski"
            },
            "question_words": {
                "kya": "ke",
                "kaun": "ko",
                "kahan": "kakh",
                "kaise": "kas",
                "kyun": "kya",
                "kitna": "kati",
                "kitne": "kati",
                "kitni": "kati",
                "kab": "kab"
            }
        }
    
    def default_patterns(self):
        """Default sentence patterns for Kumaoni"""
        return {
            "greetings": [
                {"hinglish": "namaste", "kumaoni": "namaskar"},
                {"hinglish": "kaise ho", "kumaoni": "kas cha"},
                {"hinglish": "kya haal hai", "kumaoni": "kas chal cha"},
                {"hinglish": "shubh prabhat", "kumaoni": "shubh savera"}
            ],
            "farewells": [
                {"hinglish": "alvida", "kumaoni": "phir milula"},
                {"hinglish": "phir milenge", "kumaoni": "phir bhetula"},
                {"hinglish": "shubh ratri", "kumaoni": "shubh rati"}
            ],
            "questions": [
                {"hinglish": "aap kahan se ho", "kumaoni": "tum kakh ka ho"},
                {"hinglish": "aapka naam kya hai", "kumaoni": "tumar nau ke cha"},
                {"hinglish": "kya aap kumaoni bolte hain", "kumaoni": "ke tum kumaoni bolo"}
            ]
        }
    
    def default_conversations(self):
        """Default conversation templates for Kumaoni"""
        return {
            "introduction": [
                {"role": "user", "hinglish": "namaste", "kumaoni": "namaskar"},
                {"role": "bot", "hinglish": "namaste, kaise hain aap?", "kumaoni": "namaskar, kas cha tum?"}
            ],
            "weather": [
                {"role": "user", "hinglish": "aaj mausam kaisa hai", "kumaoni": "aaj mausam kas cha"},
                {"role": "bot", "hinglish": "aaj mausam bahut achha hai", "kumaoni": "aaj mausam bado balo cha"}
            ],
            "food": [
                {"role": "user", "hinglish": "kumaon ka khaana bahut achha hai", "kumaoni": "kumaon ko khano bado balo cha"},
                {"role": "bot", "hinglish": "haan, kumaoni vyanjan bahut swaadisht hote hain", "kumaoni": "ho, kumaoni pakwan bado swadisht huncha"}
            ]
        }
    
    def default_idioms(self):
        """Default Kumaoni idioms and expressions"""
        return {
            "balo": "good/nice",
            "bado balo": "very good",
            "kas chal cha": "how are you",
            "thik-thak": "okay/fine",
            "kakh ja rya cha": "where are you going",
            "kati baji cha": "what time is it",
            "mero nau": "my name",
            "tumar nau": "your name",
            "phir bhetula": "see you again",
            "khana khaya": "khano khayo",
            "paani piya": "pani piyo"
        }
    
    def default_chat_responses(self):
        """Default chat responses for different topics"""
        return {
            "greeting": [
                {"hinglish": "Namaste! Kaise hain aap?", "kumaoni": "Namaskar! Kas cha tum?"},
                {"hinglish": "Namaste! Main Kumaoni chatbot hoon.", "kumaoni": "Namaskar! Ma Kumaoni chatbot chun."}
            ],
            "introduction": [
                {"hinglish": "Mera naam Kumaoni Chatbot hai. Main Kumaoni bhasha mein baat kar sakta hoon.", 
                 "kumaoni": "Mero nau Kumaoni Chatbot cha. Ma Kumaoni bhasha ma bat kar sakun."},
                {"hinglish": "Main ek AI assistant hoon jo Kumaoni bhasha mein madad karta hai.", 
                 "kumaoni": "Ma ek AI assistant chun jo Kumaoni bhasha ma madad karun."}
            ],
            "weather": [
                {"hinglish": "Kumaon mein mausam bahut suhana hota hai.", "kumaoni": "Kumaon ma mausam bado suhano huncha."},
                {"hinglish": "Pahaadon mein mausam bahut achha hai.", "kumaoni": "Pahadan ma mausam bado balo cha."}
            ],
            "food": [
                {"hinglish": "Kumaoni khana bahut swaadisht hota hai.", "kumaoni": "Kumaoni khano bado swadisht huncha."},
                {"hinglish": "Aloo ke gutke, bhatt ki churkani, aur kafuli bahut mashoor hain.", 
                 "kumaoni": "Aloo ka gutka, bhatt ki churkani, aur kafuli bado mashoor cha."}
            ],
            "culture": [
                {"hinglish": "Kumaon ki sanskriti bahut samriddh hai.", "kumaoni": "Kumaon ki sanskriti bado samriddh cha."},
                {"hinglish": "Kumaon ke lok geet aur nritya bahut prasiddh hain.", 
                 "kumaoni": "Kumaon ka lok geet aur nritya bado prasiddh cha."}
            ],
            "unknown": [
                {"hinglish": "Mujhe samajh nahi aaya. Kya aap dobara keh sakte hain?", 
                 "kumaoni": "Mik samajh nai ayi. Ke tum dobara koi sakta?"},
                {"hinglish": "Maaf kijiye, mujhe samajh nahi aaya.", "kumaoni": "Maph karya, mik samajh nai ayi."}
            ]
        }
    
    def detect_language(self, text):
        """Detect if the input is more Hinglish or Kumaoni"""
        hinglish_count = 0
        kumaoni_count = 0
        
        words = text.lower().split()
        for word in words:
            word = word.strip(".,?!\"'")
            
            # Check if word is in vocabulary
            if word in self.data["vocab"]:
                hinglish_count += 1
            elif word in [v for v in self.data["vocab"].values()]:
                kumaoni_count += 1
        
        # Check phrases
        for phrase in self.data["phrases"]:
            if phrase in text.lower():
                hinglish_count += 3  # Give more weight to phrases
        
        for kumaoni_phrase in self.data["phrases"].values():
            if kumaoni_phrase in text.lower():
                kumaoni_count += 3  # Give more weight to phrases
        
        if kumaoni_count > hinglish_count:
            return "kumaoni"
        else:
            return "hinglish"
    
    def detect_intent(self, text):
        """Detect the intent of the user's message"""
        text_lower = text.lower()
        
        # Check for greetings
        greeting_words = ["namaste", "namaskar", "hello", "hi", "hey", "good morning", "good evening", "kaise ho", "kas cha"]
        for word in greeting_words:
            if word in text_lower:
                return "greeting"
        
        # Check for questions about the bot
        if "kaun" in text_lower or "kya" in text_lower or "ke" in text_lower:
            if "tum" in text_lower or "aap" in text_lower or "tu" in text_lower:
                return "introduction"
        
        # Check for weather related queries
        weather_words = ["mausam", "barish", "dhoop", "garmi", "sardi", "barf"]
        for word in weather_words:
            if word in text_lower:
                return "weather"
        
        # Check for food related queries
        food_words = ["khana", "khano", "bhojan", "vyanjan", "pakwan", "recipe", "swad"]
        for word in food_words:
            if word in text_lower:
                return "food"
        
        # Check for culture related queries
        culture_words = ["sanskriti", "tyohar", "parv", "lok", "geet", "nritya", "parampara"]
        for word in culture_words:
            if word in text_lower:
                return "culture"
        
        # Default to unknown intent
        return "unknown"
    
    def translate_word(self, word, direction="hinglish_to_kumaoni"):
        """Translate a single word"""
        word_lower = word.lower().strip(".,?!\"'")
        
        if direction == "hinglish_to_kumaoni":
            # Check pronouns
            if word_lower in self.data["grammar"]["pronouns"]:
                return self.data["grammar"]["pronouns"][word_lower]
            
            # Check question words
            if word_lower in self.data["grammar"]["question_words"]:
                return self.data["grammar"]["question_words"][word_lower]
            
            # Check postpositions
            if word_lower in self.data["grammar"]["postpositions"]:
                return self.data["grammar"]["postpositions"][word_lower]
            
            # Check vocabulary
            if word_lower in self.data["vocab"]:
                return self.data["vocab"][word_lower]
            
            # Check verb endings
            for ending, replacement in self.data["grammar"]["verb_endings"].items():
                if word_lower.endswith(ending) and len(word_lower) > len(ending):
                    return word_lower[:-len(ending)] + replacement
            
            # No translation found
            return word
        else:  # kumaoni_to_hinglish
            # Check reverse mappings
            for hinglish, kumaoni in self.data["vocab"].items():
                if kumaoni.lower() == word_lower:
                    return hinglish
            
            # Check reverse pronouns
            for hinglish, kumaoni in self.data["grammar"]["pronouns"].items():
                if kumaoni.lower() == word_lower:
                    return hinglish
            
            # Check reverse question words
            for hinglish, kumaoni in self.data["grammar"]["question_words"].items():
                if kumaoni.lower() == word_lower:
                    return hinglish
            
            # Check reverse postpositions
            for hinglish, kumaoni in self.data["grammar"]["postpositions"].items():
                if kumaoni.lower() == word_lower:
                    return hinglish
            
            # No translation found
            return word
    
    def translate_phrase(self, phrase, direction="hinglish_to_kumaoni"):
        """Translate a phrase"""
        if direction == "hinglish_to_kumaoni":
            if phrase.lower() in self.data["phrases"]:
                return self.data["phrases"][phrase.lower()]
        else:  # kumaoni_to_hinglish
            for hinglish, kumaoni in self.data["phrases"].items():
                if kumaoni.lower() == phrase.lower():
                    return hinglish
        
        # No direct phrase translation, translate word by word
        words = phrase.split()
        translated_words = [self.translate_word(word, direction) for word in words]
        return " ".join(translated_words)
    
    def translate(self, text, direction="hinglish_to_kumaoni"):
        """Translate text between Hinglish and Kumaoni"""
        # Check for full phrase matches first
        if direction == "hinglish_to_kumaoni":
            for phrase in self.data["phrases"]:
                if phrase.lower() in text.lower():
                    text = text.lower().replace(phrase.lower(), self.data["phrases"][phrase])
        else:  # kumaoni_to_hinglish
            for hinglish, kumaoni in self.data["phrases"].items():
                if kumaoni.lower() in text.lower():
                    text = text.lower().replace(kumaoni.lower(), hinglish)
        
        # Translate remaining words
        words = text.split()
        translated_words = [self.translate_word(word, direction) for word in words]
        return " ".join(translated_words)
    
    def get_response(self, user_input):
        """Generate a response to the user input"""
        # Detect language of input
        input_language = self.detect_language(user_input)
        
        # Detect intent
        intent = self.detect_intent(user_input)
        
        # Update conversation context
        self.conversation_context["current_topic"] = intent
        self.conversation_context["exchanges"].append({
            "user": user_input,
            "language": input_language,
            "intent": intent,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Get appropriate response based on intent
        if intent in self.data["chat_responses"]:
            responses = self.data["chat_responses"][intent]
            response = random.choice(responses)
        else:
            responses = self.data["chat_responses"]["unknown"]
            response = random.choice(responses)
        
        # Choose response language based on user preference or input language
        if self.conversation_context["language_preference"] == "kumaoni":
            bot_response = response["kumaoni"]
            response_language = "kumaoni"
        elif self.conversation_context["language_preference"] == "hinglish":
            bot_response = response["hinglish"]
            response_language = "hinglish"
        else:  # mixed or match user's language
            if input_language == "kumaoni":
                bot_response = response["kumaoni"]
                response_language = "kumaoni"
            else:
                bot_response = response["hinglish"]
                response_language = "hinglish"
        
        # Update conversation history
        self.conversation_context["exchanges"][-1]["bot"] = bot_response
        self.conversation_context["exchanges"][-1]["bot_language"] = response_language
        
        # Save conversation history periodically
        if len(self.conversation_context["exchanges"]) % 10 == 0:
            self.save_conversation_history()
        
        return {
            "text": bot_response,
            "language": response_language,
            "intent": intent,
            "translation": self.translate(bot_response, 
                                         "kumaoni_to_hinglish" if response_language == "kumaoni" else "hinglish_to_kumaoni")
        }
    
    def learn_new_word(self, hinglish, kumaoni):
        """Learn a new word mapping"""
        hinglish = hinglish.lower().strip()
        self.data["vocab"][hinglish] = kumaoni
        self.save_json(VOCAB_MAP_PATH, self.data["vocab"])
        return f"Learned new word: {hinglish} → {kumaoni}"
    
    def learn_new_phrase(self, hinglish, kumaoni):
        """Learn a new phrase mapping"""
        hinglish = hinglish.lower().strip()
        self.data["phrases"][hinglish] = kumaoni
        self.save_json(PHRASES_PATH, self.data["phrases"])
        return f"Learned new phrase: {hinglish} → {kumaoni}"
    
    def save_conversation_history(self):
        """Save the current conversation history"""
        if self.session_id not in [s["session_id"] for s in self.data["history"]["sessions"]]:
            self.data["history"]["sessions"].append({
                "session_id": self.session_id,
                "start_time": self.conversation_context["exchanges"][0]["timestamp"] if self.conversation_context["exchanges"] else datetime.datetime.now().isoformat(),
                "exchanges": self.conversation_context["exchanges"]
            })
        else:
            for session in self.data["history"]["sessions"]:
                if session["session_id"] == self.session_id:
                    session["exchanges"] = self.conversation_context["exchanges"]
                    break
        
        self.save_json(CONVERSATION_HISTORY_PATH, self.data["history"])
    
    def set_language_preference(self, preference):
        """Set the language preference for responses"""
        if preference in ["kumaoni", "hinglish", "mixed"]:
            self.conversation_context["language_preference"] = preference
            return f"Language preference set to {preference}"
        else:
            return "Invalid language preference. Use 'kumaoni', 'hinglish', or 'mixed'."
    
    def get_stats(self):
        """Get statistics about the chatbot's knowledge"""
        return {
            "vocabulary_size": len(self.data["vocab"]),
            "phrases_count": len(self.data["phrases"]),
            "idioms_count": len(self.data["idioms"]),
            "conversation_templates": sum(len(templates) for templates in self.data["conversations"].values()),
            "response_patterns": sum(len(responses) for responses in self.data["chat_responses"].values()),
            "conversation_history": len(self.data["history"]["sessions"]),
            "current_session_exchanges": len(self.conversation_context["exchanges"])
        }

def display_response(response, use_colors=True):
    """Display the chatbot response with formatting"""
    if use_colors:
        print(f"\n{Colors.BOLD}{Colors.GREEN}Bot ({response['language']}):{Colors.ENDC} {response['text']}")
        
        # Show translation if available
        if response["language"] == "kumaoni":
            print(f"{Colors.YELLOW}Translation (Hinglish):{Colors.ENDC} {response['translation']}")
        elif response["language"] == "hinglish":
            print(f"{Colors.YELLOW}Translation (Kumaoni):{Colors.ENDC} {response['translation']}")
    else:
        print(f"\nBot ({response['language']}): {response['text']}")
        
        # Show translation if available
        if response["language"] == "kumaoni":
            print(f"Translation (Hinglish): {response['translation']}")
        elif response["language"] == "hinglish":
            print(f"Translation (Kumaoni): {response['translation']}")

def main():
    parser = argparse.ArgumentParser(description="Kumaoni Chatbot")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--language", choices=["kumaoni", "hinglish", "mixed"], default="mixed",
                        help="Set the language preference for responses")
    parser.add_argument("--learn", action="store_true", help="Enter learning mode to teach new words and phrases")
    parser.add_argument("--stats", action="store_true", help="Show statistics about the chatbot's knowledge")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = KumaoniChatbot()
    
    # Set language preference
    chatbot.set_language_preference(args.language)
    
    if args.stats:
        stats = chatbot.get_stats()
        print("\nKumaoni Chatbot Statistics:")
        print(f"Vocabulary size: {stats['vocabulary_size']} words")
        print(f"Phrases count: {stats['phrases_count']} phrases")
        print(f"Idioms count: {stats['idioms_count']} idioms")
        print(f"Conversation templates: {stats['conversation_templates']}")
        print(f"Response patterns: {stats['response_patterns']}")
        print(f"Conversation history: {stats['conversation_history']} sessions")
        print(f"Current session exchanges: {stats['current_session_exchanges']}")
        return
    
    if args.learn:
        print("=== Kumaoni Chatbot Learning Mode ===")
        print("Type 'exit' to quit learning mode")
        
        while True:
            learn_type = input("\nWhat do you want to teach? (word/phrase/exit): ").lower()
            
            if learn_type == "exit":
                break
            
            if learn_type == "word":
                hinglish = input("Enter Hinglish word: ")
                kumaoni = input("Enter Kumaoni translation: ")
                result = chatbot.learn_new_word(hinglish, kumaoni)
                print(result)
            
            elif learn_type == "phrase":
                hinglish = input("Enter Hinglish phrase: ")
                kumaoni = input("Enter Kumaoni translation: ")
                result = chatbot.learn_new_phrase(hinglish, kumaoni)
                print(result)
            
            else:
                print("Invalid option. Use 'word', 'phrase', or 'exit'.")
        
        return
    
    # Start conversation
    print("=== Kumaoni Chatbot ===")
    print("Type 'exit' to quit")
    print("Type 'translate: <text>' to translate text")
    print("Type 'learn word: <hinglish> = <kumaoni>' to teach a new word")
    print("Type 'learn phrase: <hinglish> = <kumaoni>' to teach a new phrase")
    print("Type 'language: <kumaoni|hinglish|mixed>' to set language preference")
    
    # Initial greeting
    initial_response = chatbot.get_response("namaste")
    display_response(initial_response, not args.no_color)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            # Save conversation history before exiting
            chatbot.save_conversation_history()
            print("Goodbye! Phir bhetula!")
            break
        
        # Check for special commands
        if user_input.lower().startswith("translate:"):
            text = user_input[10:].strip()
            language = chatbot.detect_language(text)
            
            if language == "kumaoni":
                translation = chatbot.translate(text, "kumaoni_to_hinglish")
                print(f"\nTranslation (Hinglish): {translation}")
            else:
                translation = chatbot.translate(text, "hinglish_to_kumaoni")
                print(f"\nTranslation (Kumaoni): {translation}")
            
            continue
        
        if user_input.lower().startswith("learn word:"):
            parts = user_input[11:].split("=")
            if len(parts) == 2:
                hinglish = parts[0].strip()
                kumaoni = parts[1].strip()
                result = chatbot.learn_new_word(hinglish, kumaoni)
                print(result)
            else:
                print("Invalid format. Use 'learn word: <hinglish> = <kumaoni>'")
            continue
        
        if user_input.lower().startswith("learn phrase:"):
            parts = user_input[13:].split("=")
            if len(parts) == 2:
                hinglish = parts[0].strip()
                kumaoni = parts[1].strip()
                result = chatbot.learn_new_phrase(hinglish, kumaoni)
                print(result)
            else:
                print("Invalid format. Use 'learn phrase: <hinglish> = <kumaoni>'")
            continue
        
        if user_input.lower().startswith("language:"):
            preference = user_input[9:].strip().lower()
            result = chatbot.set_language_preference(preference)
            print(result)
            continue
        
        # Get response from chatbot
        response = chatbot.get_response(user_input)
        display_response(response, not args.no_color)

if __name__ == "__main__":
    main()
