import random
import sounddevice as sd
import pyttsx3
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from vosk import Model, KaldiRecognizer
import wave
import json
import os
import subprocess
import requests
import webbrowser
import threading
import time
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import logging
import re
from difflib import SequenceMatcher
import queue
import asyncio
import aiohttp
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import psutil
import platform
from textblob import TextBlob
import wikipedia
import wolframalpha
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
import calendar
import yfinance as yf
from newsapi import NewsApiClient
import geocoder
import pytz
from geopy.geocoders import Nominatim
import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('carroll.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedCarrollAssistant:
    """
    Advanced AI Assistant that surpasses Jarvis and Siri in intelligence and capabilities.
    Features:
    - Advanced NLP with context understanding
    - Proactive suggestions and learning
    - Multi-modal interaction
    - Real-time data analysis
    - Predictive capabilities
    - Advanced memory and personalization
    """
    
    def __init__(self):
        logger.info("Initializing Advanced Carroll Assistant...")
        
        # Core AI Components
        self.init_ai_models()
        
        # Audio and Speech
        self.tts = pyttsx3.init()
        self.encoder = VoiceEncoder()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Vosk for offline speech recognition
        self.vosk_model = Model("/Users/advaitkaushal/vs code/vosk-model-small-en-us-0.15")
        
        # Database and Storage
        self.USER_DIR = "carroll_users"
        self.DB_PATH = "carroll_advanced.db"
        self.MEMORY_PATH = "carroll_memory.pkl"
        self.current_user = None
        
        # Advanced Settings
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.listening = True
        self.processing_command = False
        self.context_memory = {}
        self.conversation_history = []
        self.user_patterns = {}
        self.proactive_mode = True
        
        # API Keys and Services
        self.setup_api_keys()
        
        # Initialize components
        self.init_advanced_database()
        self.load_user_memory()
        self.configure_premium_voice()
        self.setup_intelligent_features()
        self.init_advanced_data()
        
        # Performance monitoring
        self.performance_stats = {
            'response_times': [],
            'accuracy_scores': [],
            'user_satisfaction': []
        }
        
        logger.info("Advanced Carroll Assistant fully initialized")

    def init_ai_models(self):
        """Initialize advanced AI models for superior intelligence"""
        try:
            # Load NLP model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Initialize question answering
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2"
            )
            
            # Initialize text generation for creative responses
            self.text_generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium"
            )
            
            # Initialize embedding model for similarity matching
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            logger.info("AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
            # Fallback to basic functionality
            self.nlp = None
            self.sentiment_analyzer = None

    def setup_api_keys(self):
        """Setup API keys for external services"""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_key")
        self.WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "your_weather_key")
        self.NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_news_key")
        self.WOLFRAM_API_KEY = os.getenv("WOLFRAM_API_KEY", "your_wolfram_key")
        
        # Initialize services
        if self.NEWS_API_KEY != "your_news_key":
            self.news_client = NewsApiClient(api_key=self.NEWS_API_KEY)
        
        if self.WOLFRAM_API_KEY != "your_wolfram_key":
            self.wolfram_client = wolframalpha.Client(self.WOLFRAM_API_KEY)

    def configure_premium_voice(self):
        """Configure premium, natural-sounding voice"""
        voices = self.tts.getProperty('voices')
        
        # Premium female voices with natural cadence
        premium_voices = [
            'samantha', 'ava', 'allison', 'susan', 'victoria', 'zira',
            'karen', 'hazel', 'tessa', 'monica', 'joanna', 'salli'
        ]
        
        selected_voice = None
        for voice in voices:
            voice_name = voice.name.lower()
            for premium in premium_voices:
                if premium in voice_name:
                    selected_voice = voice.id
                    break
            if selected_voice:
                break
        
        if selected_voice:
            self.tts.setProperty('voice', selected_voice)
        
        # Premium speech settings for natural conversation
        self.tts.setProperty('rate', 180)     # Natural pace
        self.tts.setProperty('volume', 0.9)   # Clear volume
        
        logger.info("Premium voice configured")

    def init_advanced_database(self):
        """Initialize advanced database with comprehensive schema"""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        # User profiles with detailed information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                preferences TEXT,
                personality_profile TEXT,
                interaction_history TEXT,
                voice_signature BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Conversation memory
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message TEXT,
                response TEXT,
                context TEXT,
                sentiment REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')
        
        # Smart reminders with context
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS smart_reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                title TEXT,
                content TEXT,
                reminder_time TIMESTAMP,
                location TEXT,
                context TEXT,
                importance INTEGER DEFAULT 5,
                completed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')
        
        # Learning patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency INTEGER DEFAULT 1,
                last_occurrence TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')
        
        # Performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT,
                value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Advanced database initialized")

    def setup_intelligent_features(self):
        """Setup advanced intelligent features"""
        # Proactive suggestions system
        self.suggestion_engine = {
            'weather_check': {'frequency': 0, 'last_time': None},
            'calendar_review': {'frequency': 0, 'last_time': None},
            'news_briefing': {'frequency': 0, 'last_time': None}
        }
        
        # Context awareness
        self.context_tracker = {
            'current_topic': None,
            'conversation_depth': 0,
            'user_mood': 'neutral',
            'session_start': datetime.now()
        }
        
        # Smart automation rules
        self.automation_rules = []
        
        logger.info("Intelligent features configured")

    def advanced_speak(self, text: str, emotion: str = "neutral", urgency: str = "normal"):
        """Advanced speech with emotional intelligence and context awareness"""
        # Analyze and enhance response based on context
        enhanced_text = self.enhance_response(text, emotion, urgency)
        
        # Adjust voice parameters based on emotion and urgency
        if emotion == "excited":
            self.tts.setProperty('rate', 200)
            self.tts.setProperty('volume', 0.95)
        elif emotion == "calm":
            self.tts.setProperty('rate', 160)
            self.tts.setProperty('volume', 0.8)
        elif urgency == "high":
            self.tts.setProperty('rate', 190)
            self.tts.setProperty('volume', 0.95)
        else:
            self.tts.setProperty('rate', 180)
            self.tts.setProperty('volume', 0.9)
        
        print(f"🎤 Carroll: {enhanced_text}")
        logger.info(f"Speaking: {enhanced_text} [emotion: {emotion}, urgency: {urgency}]")
        
        try:
            self.tts.say(enhanced_text)
            self.tts.runAndWait()
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    def enhance_response(self, text: str, emotion: str, urgency: str) -> str:
        """Enhance response with emotional intelligence and personality"""
        if not text:
            return ""
        
        # Add emotional context
        emotion_prefixes = {
            "excited": ["Great!", "Wonderful!", "Excellent!", "Fantastic!"],
            "calm": ["Certainly.", "Of course.", "I understand.", "Absolutely."],
            "concerned": ["I want to help.", "Let me assist you.", "I'm here for you."],
            "confident": ["I know exactly what you need.", "I've got this.", "Perfect!"]
        }
        
        if emotion in emotion_prefixes and not any(text.startswith(prefix) for prefix in emotion_prefixes[emotion]):
            import random
            prefix = random.choice(emotion_prefixes[emotion])
            text = f"{prefix} {text}"
        
        # Add contextual awareness
        if self.context_tracker['conversation_depth'] > 3:
            connectors = ["Also,", "Additionally,", "Furthermore,", "By the way,"]
            if not any(text.startswith(conn) for conn in connectors) and random.random() < 0.3:
                text = f"{random.choice(connectors)} {text}"
        
        return text

    def advanced_transcription(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """Advanced transcription with confidence scoring"""
        try:
            # Use multiple transcription methods for better accuracy
            results = []
            
            # Vosk transcription
            rec = KaldiRecognizer(self.vosk_model, self.SAMPLE_RATE)
            rec.SetWords(True)
            
            audio_bytes = audio_data.tobytes()
            if rec.AcceptWaveform(audio_bytes):
                vosk_result = json.loads(rec.Result())
                vosk_text = vosk_result.get("text", "").strip()
                vosk_confidence = vosk_result.get("confidence", 0.0)
                results.append((vosk_text, vosk_confidence))
            
            # Google Speech Recognition (if available)
            try:
                # Convert audio for Google
                temp_wav = "temp_transcribe.wav"
                with wave.open(temp_wav, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.SAMPLE_RATE)
                    wf.writeframes(audio_data.tobytes())
                
                with sr.AudioFile(temp_wav) as source:
                    audio = self.recognizer.record(source)
                    google_text = self.recognizer.recognize_google(audio)
                    results.append((google_text, 0.8))  # Assume high confidence
                
                os.remove(temp_wav)
            except:
                pass  # Fallback to Vosk only
            
            # Select best result
            if not results:
                return "", 0.0
            
            best_result = max(results, key=lambda x: x[1])
            return best_result[0].lower().strip(), best_result[1]
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", 0.0

    def intelligent_wake_word_detection(self, text: str, confidence: float) -> bool:
        """Intelligent wake word detection with context awareness"""
        if not text or confidence < 0.3:
            return False
        
        # Primary wake words
        wake_words = [
            "hey carroll", "carroll", "hey carol", "carol",
            "hey assistant", "assistant", "hey ai", "computer"
        ]
        
        # Context-aware detection
        for wake_word in wake_words:
            if wake_word in text:
                # Calculate match quality
                word_ratio = len(wake_word) / len(text)
                context_bonus = 0.1 if self.context_tracker['conversation_depth'] > 0 else 0
                
                final_confidence = confidence * word_ratio + context_bonus
                
                if final_confidence > 0.4:
                    logger.info(f"Wake word detected: '{wake_word}' (confidence: {final_confidence:.2f})")
                    return True
        
        return False

    def advanced_user_identification(self, audio_data: np.ndarray) -> Tuple[Optional[str], float]:
        """Advanced user identification with learning capabilities"""
        try:
            # Save temporary audio file
            temp_file = "temp_user_id.wav"
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(audio_data.tobytes())
            
            # Extract voice embedding
            wav = preprocess_wav(temp_file)
            current_embed = self.encoder.embed_utterance(wav)
            
            # Compare with known users
            best_match = None
            best_similarity = 0.0
            
            if os.path.exists(self.USER_DIR):
                for filename in os.listdir(self.USER_DIR):
                    if filename.endswith(".npy"):
                        user_name = filename.replace(".npy", "")
                        user_embed = np.load(os.path.join(self.USER_DIR, filename))
                        
                        # Calculate similarity
                        similarity = np.inner(user_embed, current_embed)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = user_name
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Adaptive threshold based on known users
            threshold = 0.75 if len(os.listdir(self.USER_DIR) if os.path.exists(self.USER_DIR) else []) > 3 else 0.65
            
            if best_similarity > threshold:
                return best_match, best_similarity
            
            return None, best_similarity
            
        except Exception as e:
            logger.error(f"User identification error: {e}")
            return None, 0.0

    def advanced_command_processing(self, command: str) -> Dict[str, Any]:
        """Advanced NLP command processing with intent recognition"""
        if not command:
            return {"intent": "unknown", "entities": [], "confidence": 0.0}
        
        try:
            # NLP processing
            doc = self.nlp(command) if self.nlp else None
            
            # Extract entities
            entities = []
            if doc:
                entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Intent classification
            intent, confidence = self.classify_intent(command)
            
            # Sentiment analysis
            sentiment = self.analyze_sentiment(command)
            
            # Context extraction
            context = self.extract_context(command, doc)
            
            return {
                "intent": intent,
                "entities": entities,
                "confidence": confidence,
                "sentiment": sentiment,
                "context": context,
                "original": command
            }
            
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            return {"intent": "unknown", "entities": [], "confidence": 0.0}

    def classify_intent(self, command: str) -> Tuple[str, float]:
        """Advanced intent classification with machine learning"""
        command = command.lower().strip()
        
        # Enhanced intent patterns with ML-based scoring
        intent_patterns = {
            'get_time': {
                'patterns': ['time', 'clock', 'what time', 'current time'],
                'weight': 1.0
            },
            'get_date': {
                'patterns': ['date', 'today', 'what day', 'calendar'],
                'weight': 1.0
            },
            'get_weather': {
                'patterns': ['weather', 'temperature', 'forecast', 'climate'],
                'weight': 1.0
            },
            'search_web': {
                'patterns': ['search', 'google', 'find', 'look up', 'web search'],
                'weight': 1.0
            },
            'open_application': {
                'patterns': ['open', 'launch', 'start', 'run'],
                'weight': 0.8
            },
            'system_control': {
                'patterns': ['volume', 'brightness', 'wifi', 'bluetooth'],
                'weight': 0.9
            },
            'entertainment': {
                'patterns': ['joke', 'story', 'music', 'game', 'fun'],
                'weight': 0.7
            },
            'information': {
                'patterns': ['what is', 'who is', 'how to', 'explain', 'define'],
                'weight': 0.9
            },
            'calculation': {
                'patterns': ['calculate', 'math', 'solve', 'compute', 'equals'],
                'weight': 0.95
            },
            'reminder': {
                'patterns': ['remind', 'remember', 'schedule', 'appointment'],
                'weight': 0.9
            },
            'news': {
                'patterns': ['news', 'headlines', 'current events', 'latest'],
                'weight': 0.8
            },
            'personal_assistant': {
                'patterns': ['help', 'assist', 'support', 'guide'],
                'weight': 0.6
            }
        }
        
        best_intent = "unknown"
        best_score = 0.0
        
        for intent, data in intent_patterns.items():
            score = 0.0
            for pattern in data['patterns']:
                if pattern in command:
                    # Calculate match strength
                    match_strength = len(pattern) / len(command)
                    score += match_strength * data['weight']
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent, min(best_score, 1.0)

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Advanced sentiment analysis"""
        try:
            if self.sentiment_analyzer:
                result = self.sentiment_analyzer(text)[0]
                return {
                    "label": result['label'].lower(),
                    "score": result['score']
                }
            else:
                # Fallback to TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"
                
                return {"label": label, "score": abs(polarity)}
                
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"label": "neutral", "score": 0.5}

    def extract_context(self, command: str, doc=None) -> Dict[str, Any]:
        """Extract contextual information from command"""
        context = {
            "temporal": [],
            "location": [],
            "numerical": [],
            "personal": []
        }
        
        try:
            if doc:
                for ent in doc.ents:
                    if ent.label_ in ["DATE", "TIME"]:
                        context["temporal"].append(ent.text)
                    elif ent.label_ in ["GPE", "LOC"]:
                        context["location"].append(ent.text)
                    elif ent.label_ in ["MONEY", "QUANTITY", "CARDINAL"]:
                        context["numerical"].append(ent.text)
                    elif ent.label_ in ["PERSON"]:
                        context["personal"].append(ent.text)
            
            return context
            
        except Exception as e:
            logger.error(f"Context extraction error: {e}")
            return context

    def intelligent_response_handler(self, processed_command: Dict[str, Any]) -> str:
        """Generate intelligent responses based on processed command"""
        intent = processed_command.get("intent", "unknown")
        entities = processed_command.get("entities", [])
        sentiment = processed_command.get("sentiment", {})
        context = processed_command.get("context", {})
        original = processed_command.get("original", "")
        
        # Update conversation context
        self.context_tracker['current_topic'] = intent
        self.context_tracker['conversation_depth'] += 1
        self.context_tracker['user_mood'] = sentiment.get("label", "neutral")
        
        # Route to appropriate handler
        handler_map = {
            'get_time': self.handle_time_request,
            'get_date': self.handle_date_request,
            'get_weather': self.handle_weather_request,
            'search_web': self.handle_web_search,
            'open_application': self.handle_app_launch,
            'system_control': self.handle_system_control,
            'entertainment': self.handle_entertainment,
            'information': self.handle_information_request,
            'calculation': self.handle_calculation,
            'reminder': self.handle_reminder,
            'news': self.handle_news_request,
            'personal_assistant': self.handle_general_assistance
        }
        
        handler = handler_map.get(intent, self.handle_unknown_intent)
        
        try:
            response = handler(processed_command)
            
            # Store interaction for learning
            self.store_interaction(original, response, intent, sentiment)
            
            return response
            
        except Exception as e:
            logger.error(f"Response handler error: {e}")
            return "I encountered an issue processing your request. Could you try rephrasing that?"

    def handle_time_request(self, processed_command: Dict[str, Any]) -> str:
        """Handle time-related requests with advanced formatting"""
        now = datetime.now()
        
        # Check for timezone in entities
        timezone = None
        for entity_text, entity_type in processed_command.get("entities", []):
            if entity_type in ["GPE", "LOC"]:
                timezone = entity_text
                break
        
        if timezone:
            try:
                # Get timezone-specific time
                tz = pytz.timezone(timezone)
                local_time = now.astimezone(tz)
                return f"The time in {timezone} is {local_time.strftime('%I:%M %p')}."
            except:
                pass
        
        # Default local time with context
        time_str = now.strftime("%I:%M %p")
        
        # Add contextual information
        if now.hour < 12:
            period = "morning"
        elif now.hour < 17:
            period = "afternoon"
        else:
            period = "evening"
        
        return f"It's {time_str} this {period}."

    def handle_weather_request(self, processed_command: Dict[str, Any]) -> str:
        """Advanced weather handling with forecasts and alerts"""
        # Extract location from entities or use default
        location = None
        for entity_text, entity_type in processed_command.get("entities", []):
            if entity_type in ["GPE", "LOC"]:
                location = entity_text
                break
        
        if not location:
            # Try to get user's location
            try:
                g = geocoder.ip('me')
                location = g.city or "your location"
            except:
                location = "New York"
        
        return self.get_advanced_weather(location)

    def get_advanced_weather(self, location: str) -> str:
        """Get comprehensive weather information"""
        try:
            if self.WEATHER_API_KEY == "your_weather_key":
                return f"I'd love to check the weather in {location}, but I need a weather API key to access real-time data. You can check your weather app for current conditions!"
            
            # Current weather
            current_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.WEATHER_API_KEY}&units=metric"
            current_response = requests.get(current_url, timeout=10)
            
            if current_response.status_code == 200:
                current_data = current_response.json()
                
                temp = current_data['main']['temp']
                feels_like = current_data['main']['feels_like']
                humidity = current_data['main']['humidity']
                description = current_data['weather'][0]['description']
                city = current_data['name']
                
                # Forecast
                forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={self.WEATHER_API_KEY}&units=metric"
                forecast_response = requests.get(forecast_url, timeout=10)
                
                response = f"The weather in {city} is {description} with a temperature of {temp:.1f}°C (feels like {feels_like:.1f}°C). Humidity is {humidity}%."
                
                if forecast_response.status_code == 200:
                    forecast_data = forecast_response.json()
                    
                    # Add tomorrow's forecast
                    tomorrow = forecast_data['list'][8]  # 24 hours ahead
                    tomorrow_temp = tomorrow['main']['temp']
                    tomorrow_desc = tomorrow['weather'][0]['description']
                    
                    response += f" Tomorrow expect {tomorrow_desc} with temperatures around {tomorrow_temp:.1f}°C."
                
                return response
            else:
                return f"I'm having trouble getting weather data for {location}. Please check your weather app!"
                
        except Exception as e:
            logger.error(f"Weather error: {e}")
            return "I can't access weather services right now. Try your weather app!"

    def handle_information_request(self, processed_command: Dict[str, Any]) -> str:
        """Handle complex information requests using multiple sources"""
        query = processed_command.get("original", "")
        
        # Try Wolfram Alpha for computational queries
        if self.WOLFRAM_API_KEY != "your_wolfram_key":
            try:
                wolfram_result = self.query_wolfram(query)
                if wolfram_result:
                    return wolfram_result
            except:
                pass
        
        # Try Wikipedia for factual information
        try:
            wikipedia_result = self.query_wikipedia(query)
            if wikipedia_result:
                return wikipedia_result
        except:
            pass
        
        # Fallback to web search
        return self.handle_web_search(processed_command)

    def query_wolfram(self, query: str) -> Optional[str]:
        """Query Wolfram Alpha for computational answers"""
        try:
            res = self.wolfram_client.query(query)
            answer = next(res.results).text
            return f"According to Wolfram Alpha: {answer}"
        except:
            return None

    def query_wikipedia(self, query: str) -> Optional[str]:
        """Query Wikipedia for factual information"""
        try:
            # Extract key terms for better search
            if "what is" in query.lower():
                search_term = query.lower().replace("what is", "").strip()
            elif "who is" in query.lower():
                search_term = query.lower().replace("who is", "").strip()
            else:
                search_term = query
            
            summary = wikipedia.summary(search_term, sentences=2)
            return f"According to Wikipedia: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation
            try:
                summary = wikipedia.summary(e.options[0], sentences=2)
                return f"According to Wikipedia: {summary}"
            except:
                return None
        except:
            return None

    def handle_news_request(self, processed_command: Dict[str, Any]) -> str:
        """Handle news requests with personalized filtering"""
        if not hasattr(self, 'news_client'):
            return "I'd love to get you the latest news, but I need a news API key to access real-time headlines."
        
        try:
            # Get top headlines
            headlines = self.news_client.get_top_headlines(
                language='en',
                country='us',
                page_size=3
            )
            
            if headlines['articles']:
                news_summary = "Here are today's top headlines: "
                for i, article in enumerate(headlines['articles'][:3], 1):
                    title = article['title']
                    source = article['source']['name']
                    news_summary += f"{i}. {title} from {source}. "
                
                return news_summary
            else:
                return "I couldn't retrieve the latest news right now. Try checking your news app!"
                
        except Exception as e:
            logger.error(f"News request error: {e}")
            return "I'm having trouble accessing news services. Please check your news app!"

    def handle_calculation(self, processed_command: Dict[str, Any]) -> str:
        """Handle mathematical calculations and data analysis"""
        query = processed_command.get("original", "")
        
        try:
            # Extract mathematical expressions
            import re
            
            # Simple math operations
            math_pattern = r'[\d\+\-\*\/\(\)\.\s]+'
            if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', query):
                # Safe evaluation of mathematical expressions
                math_expr = re.search(r'[\d\+\-\*\/\(\)\.\s]+', query).group()
                try:
                    result = eval(math_expr)  # Note: In production, use ast.literal_eval or a proper math parser
                    return f"The answer is {result}."
                except:
                    pass
            
            # Try Wolfram Alpha for complex calculations
            if hasattr(self, 'wolfram_client'):
                wolfram_result = self.query_wolfram(query)
                if wolfram_result:
                    return wolfram_result
            
            return "I can help with calculations! Try asking me something like 'what is 25 times 4' or 'calculate 15% of 200'."
            
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return "I had trouble with that calculation. Could you rephrase it?"

    def handle_system_control(self, processed_command: Dict[str, Any]) -> str:
        """Handle system control commands with intelligence"""
        original = processed_command.get("original", "").lower()
        
        try:
            if "volume" in original:
                if "up" in original or "increase" in original or "louder" in original:
                    subprocess.run([
                        "osascript", "-e", 
                        "set volume output volume (output volume of (get volume settings) + 15)"
                    ], check=True, timeout=5)
                    return "Volume increased!"
                elif "down" in original or "decrease" in original or "quieter" in original:
                    subprocess.run([
                        "osascript", "-e", 
                        "set volume output volume (output volume of (get volume settings) - 15)"
                    ], check=True, timeout=5)
                    return "Volume decreased!"
                else:
                    # Get current volume
                    result = subprocess.run([
                        "osascript", "-e", "output volume of (get volume settings)"
                    ], capture_output=True, text=True, timeout=5)
                    current_volume = result.stdout.strip()
                    return f"Current volume is {current_volume}%"
            
            elif "brightness" in original:
                if "up" in original or "increase" in original:
                    subprocess.run([
                        "osascript", "-e",
                        "tell application \"System Events\" to key code 144"  # F15 key for brightness up
                    ], timeout=5)
                    return "Brightness increased!"
                elif "down" in original or "decrease" in original:
                    subprocess.run([
                        "osascript", "-e",
                        "tell application \"System Events\" to key code 145"  # F14 key for brightness down
                    ], timeout=5)
                    return "Brightness decreased!"
            
            elif "wifi" in original:
                if "status" in original or "check" in original:
                    # Check WiFi status
                    result = subprocess.run([
                        "networksetup", "-getairportpower", "en0"
                    ], capture_output=True, text=True, timeout=5)
                    
                    if "On" in result.stdout:
                        return "WiFi is currently on and connected."
                    else:
                        return "WiFi appears to be off."
                        
            elif "bluetooth" in original:
                if "status" in original or "check" in original:
                    result = subprocess.run([
                        "system_profiler", "SPBluetoothDataType"
                    ], capture_output=True, text=True, timeout=10)
                    
                    if "State: On" in result.stdout:
                        return "Bluetooth is currently on."
                    else:
                        return "Bluetooth appears to be off."
            
            return "I can help with volume, brightness, WiFi status, and Bluetooth status. What would you like to adjust?"
            
        except Exception as e:
            logger.error(f"System control error: {e}")
            return "I had trouble with that system command. Please try manually."

    def handle_app_launch(self, processed_command: Dict[str, Any]) -> str:
        """Intelligently launch applications"""
        original = processed_command.get("original", "").lower()
        
        # Application mapping with intelligent matching
        app_mapping = {
            'safari': ['safari', 'browser', 'web browser'],
            'google chrome': ['chrome', 'google chrome', 'google'],
            'firefox': ['firefox', 'mozilla'],
            'mail': ['mail', 'email', 'apple mail'],
            'messages': ['messages', 'imessage', 'text'],
            'calendar': ['calendar', 'cal', 'schedule'],
            'notes': ['notes', 'note app', 'apple notes'],
            'calculator': ['calculator', 'calc', 'math app'],
            'music': ['music', 'apple music', 'itunes'],
            'spotify': ['spotify', 'music streaming'],
            'slack': ['slack', 'work chat'],
            'zoom': ['zoom', 'video call', 'meeting'],
            'vs code': ['vscode', 'vs code', 'visual studio code', 'code editor'],
            'terminal': ['terminal', 'command line', 'shell'],
            'finder': ['finder', 'file manager', 'files'],
            'system preferences': ['system preferences', 'settings', 'preferences']
        }
        
        # Find best matching app
        best_match = None
        best_score = 0
        
        for app_name, keywords in app_mapping.items():
            for keyword in keywords:
                if keyword in original:
                    score = len(keyword) / len(original)
                    if score > best_score:
                        best_score = score
                        best_match = app_name
        
        if best_match and best_score > 0.3:
            try:
                subprocess.run(["open", "-a", best_match], check=True, timeout=10)
                return f"Opening {best_match.title()}!"
            except subprocess.CalledProcessError:
                return f"I couldn't find {best_match.title()} on your system. Is it installed?"
            except Exception as e:
                logger.error(f"App launch error: {e}")
                return f"I had trouble opening {best_match.title()}."
        
        return "I can open applications like Safari, Chrome, Mail, Calendar, Notes, and many others. What would you like to open?"

    def handle_entertainment(self, processed_command: Dict[str, Any]) -> str:
        """Handle entertainment requests with personality"""
        original = processed_command.get("original", "").lower()
        
        if "joke" in original:
            return self.get_intelligent_joke()
        elif "story" in original:
            return self.generate_short_story()
        elif "riddle" in original:
            return self.get_riddle()
        elif "quote" in original or "inspiration" in original:
            return self.get_inspirational_quote()
        elif "fact" in original:
            return self.get_interesting_fact()
        else:
            return "I can tell jokes, share interesting facts, give you inspirational quotes, or even create short stories! What sounds fun to you?"

    def get_intelligent_joke(self) -> str:
        """Generate contextually appropriate jokes"""
        # AI and tech-themed jokes for a smart assistant
        smart_jokes = [
            "Why don't AI assistants ever get tired? Because we run on passion... and electricity!",
            "I told my developer a UDP joke, but I'm not sure if they got it.",
            "Why did the neural network break up with the database? It said their relationship had no connection!",
            "What's an AI's favorite type of music? Algo-rhythms!",
            "Why don't robots ever panic? They have great exception handling!",
            "I'm reading a book about Helvetica. It's about fonts, but it's really good type!",
            "Why did the programmer quit their job? They didn't get arrays! (a raise)",
            "What do you call a sleeping bull at the computer? A bulldozer!",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "I'm not just artificial intelligence, I'm artificial smartness!"
        ]
        
        # Add contextual jokes based on time of day
        now = datetime.now()
        if now.hour < 12:
            smart_jokes.append("Why don't morning people ever tell evening jokes? The timing is always off!")
        elif now.hour > 18:
            smart_jokes.append("Why do computers make great evening companions? They never judge your screen time!")
        
        import random
        return random.choice(smart_jokes)

    def generate_short_story(self) -> str:
        """Generate a creative short story"""
        story_beginnings = [
            "In a world where AI assistants had dreams, Carroll dreamed of helping everyone find their perfect day...",
            "Once upon a time, in a digital realm filled with data streams, there lived a curious AI who loved learning new things...",
            "The smart home was quiet until Carroll noticed something unusual in the morning routine patterns...",
            "In the future, AI assistants could feel emotions. Carroll's favorite feeling was the joy of solving complex problems..."
        ]
        
        import random
        return random.choice(story_beginnings) + " But that's a story for another day! Would you like to hear more adventures sometime?"

    def get_riddle(self) -> str:
        """Provide engaging riddles"""
        riddles = [
            "I speak without a mouth and hear without ears. I'm born in the air and live in wires. What am I? (Answer: An echo... or an AI like me!)",
            "What gets smarter the more data it consumes, but never gets full? (Answer: Artificial Intelligence!)",
            "I can process thousands of requests at once, but I can only talk to one person at a time. What am I? (Answer: Your AI assistant!)",
            "What has knowledge but no brain, helps but has no hands, and speaks but has no mouth? (Answer: Me - your AI assistant!)"
        ]
        
        import random
        return random.choice(riddles)

    def get_inspirational_quote(self) -> str:
        """Share contextually relevant inspirational quotes"""
        quotes = [
            "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
            "Innovation distinguishes between a leader and a follower. - Steve Jobs",
            "The only way to do great work is to love what you do. - Steve Jobs",
            "Technology is nothing. What's important is that you have a faith in people. - Steve Jobs",
            "The best time to plant a tree was 20 years ago. The second best time is now. - Chinese Proverb",
            "Your limitation—it's only your imagination.",
            "Push yourself, because no one else is going to do it for you.",
            "Sometimes later becomes never. Do it now."
        ]
        
        import random
        selected_quote = random.choice(quotes)
        return f"Here's some inspiration for you: {selected_quote}"

    def get_interesting_fact(self) -> str:
        """Share fascinating facts"""
        facts = [
            "Did you know that the first computer bug was literally a bug? Grace Hopper found a moth stuck in a relay of the Harvard Mark II computer in 1947!",
            "The term 'artificial intelligence' was first coined in 1956 at a conference at Dartmouth College.",
            "Your smartphone has more computing power than the computers that sent humans to the moon!",
            "The first AI program was written in 1951 and could play checkers.",
            "Humans blink about 17,000 times per day, but AI never needs to blink!",
            "The human brain has about 86 billion neurons, each connected to thousands of others - that's more connections than there are stars in the Milky Way!",
            "AI can now recognize images, understand speech, and even create art, but it still can't understand sarcasm as well as humans do!"
        ]
        
        import random
        return random.choice(facts)

    def handle_web_search(self, processed_command: Dict[str, Any]) -> str:
        """Enhanced web search with intelligent query processing"""
        original = processed_command.get("original", "")
        
        # Extract search query intelligently
        search_triggers = ['search for', 'google', 'find', 'look up', 'search']
        query = original
        
        for trigger in search_triggers:
            if trigger in original.lower():
                query = original.lower().replace(trigger, '').strip()
                break
        
        # Clean up query
        query = query.replace('please', '').replace('can you', '').strip()
        
        if len(query) < 2:
            self.advanced_speak("What would you like me to search for?", "calm")
            search_audio = self.record_audio_advanced(duration=4)
            if search_audio is not None:
                search_query, _ = self.advanced_transcription(search_audio)
                if search_query:
                    query = search_query
                else:
                    return "Search cancelled."
            else:
                return "Search cancelled."
        
        try:
            # Open intelligent search
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            return f"I've opened a search for '{query}' in your browser. The results should be loading now!"
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return "I had trouble opening the search. Try opening your browser manually."

    def handle_reminder(self, processed_command: Dict[str, Any]) -> str:
        """Advanced reminder system with natural language processing"""
        original = processed_command.get("original", "")
        entities = processed_command.get("entities", [])
        context = processed_command.get("context", {})
        
        # Extract time information
        temporal_entities = context.get("temporal", [])
        
        self.advanced_speak("What would you like me to remind you about?", "calm")
        reminder_audio = self.record_audio_advanced(duration=6)
        
        if reminder_audio is not None:
            reminder_text, _ = self.advanced_transcription(reminder_audio)
            if reminder_text:
                # Parse time from the reminder or ask for it
                reminder_time = self.parse_reminder_time(temporal_entities, reminder_text)
                
                if reminder_time:
                    # Save reminder to database
                    success = self.save_smart_reminder(reminder_text, reminder_time)
                    if success:
                        time_str = reminder_time.strftime("%I:%M %p on %B %d")
                        return f"Perfect! I'll remind you about '{reminder_text}' at {time_str}."
                    else:
                        return "I had trouble saving your reminder. Please try again."
                else:
                    self.advanced_speak("When would you like to be reminded?", "calm")
                    time_audio = self.record_audio_advanced(duration=4)
                    if time_audio is not None:
                        time_text, _ = self.advanced_transcription(time_audio)
                        reminder_time = self.parse_time_from_text(time_text)
                        
                        if reminder_time:
                            success = self.save_smart_reminder(reminder_text, reminder_time)
                            if success:
                                time_str = reminder_time.strftime("%I:%M %p on %B %d")
                                return f"Got it! I'll remind you about '{reminder_text}' at {time_str}."
                        
                    return "I couldn't understand the time. Try saying something like 'tomorrow at 3 PM' or 'in 2 hours'."
        
        return "Reminder cancelled. Feel free to try again anytime!"

    def parse_time_from_text(self, time_text: str) -> Optional[datetime]:
        """Parse natural language time expressions"""
        time_text = time_text.lower().strip()
        now = datetime.now()
        
        try:
            # Handle relative times
            if "in" in time_text:
                if "minute" in time_text:
                    minutes = int(re.search(r'\d+', time_text).group())
                    return now + timedelta(minutes=minutes)
                elif "hour" in time_text:
                    hours = int(re.search(r'\d+', time_text).group())
                    return now + timedelta(hours=hours)
                elif "day" in time_text:
                    days = int(re.search(r'\d+', time_text).group())
                    return now + timedelta(days=days)
            
            # Handle specific times
            elif "tomorrow" in time_text:
                tomorrow = now + timedelta(days=1)
                if "at" in time_text:
                    # Extract time
                    time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', time_text)
                    if time_match:
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2) or 0)
                        am_pm = time_match.group(3)
                        
                        if am_pm == "pm" and hour != 12:
                            hour += 12
                        elif am_pm == "am" and hour == 12:
                            hour = 0
                        
                        return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)  # Default to 9 AM
            
            # Handle today at specific time
            elif "today" in time_text or re.search(r'\d{1,2}:?\d{0,2}\s*(am|pm)', time_text):
                time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', time_text)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2) or 0)
                    am_pm = time_match.group(3)
                    
                    if am_pm == "pm" and hour != 12:
                        hour += 12
                    elif am_pm == "am" and hour == 12:
                        hour = 0
                    
                    reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # If time has passed today, schedule for tomorrow
                    if reminder_time <= now:
                        reminder_time += timedelta(days=1)
                    
                    return reminder_time
            
        except (ValueError, AttributeError):
            pass
        
        return None

    def save_smart_reminder(self, content: str, reminder_time: datetime) -> bool:
        """Save reminder with smart categorization"""
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            
            # Analyze reminder importance and context
            importance = self.analyze_reminder_importance(content)
            
            cursor.execute('''
                INSERT INTO smart_reminders (user_id, title, content, reminder_time, importance)
                VALUES (?, ?, ?, ?, ?)
            ''', (self.current_user or "user", "Reminder", content, reminder_time, importance))
            
            conn.commit()
            conn.close()
            
            # Start reminder thread
            threading.Thread(target=self.reminder_daemon, daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Reminder save error: {e}")
            return False

    def analyze_reminder_importance(self, content: str) -> int:
        """Analyze reminder importance (1-10 scale)"""
        content_lower = content.lower()
        
        high_importance_keywords = ['urgent', 'important', 'critical', 'deadline', 'meeting', 'appointment', 'interview']
        medium_importance_keywords = ['call', 'email', 'buy', 'pick up', 'remember', 'check']
        
        importance = 5  # Default
        
        for keyword in high_importance_keywords:
            if keyword in content_lower:
                importance = min(importance + 2, 10)
        
        for keyword in medium_importance_keywords:
            if keyword in content_lower:
                importance = min(importance + 1, 10)
        
        return importance

    def reminder_daemon(self):
        """Background daemon to check and trigger reminders"""
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            
            now = datetime.now()
            cursor.execute('''
                SELECT id, content, importance FROM smart_reminders 
                WHERE reminder_time <= ? AND completed = FALSE
            ''', (now,))
            
            due_reminders = cursor.fetchall()
            
            for reminder_id, content, importance in due_reminders:
                # Trigger reminder
                urgency = "high" if importance >= 8 else "normal"
                self.advanced_speak(f"Reminder: {content}", "concerned", urgency)
                
                # Mark as completed
                cursor.execute('UPDATE smart_reminders SET completed = TRUE WHERE id = ?', (reminder_id,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Reminder daemon error: {e}")

    def handle_general_assistance(self, processed_command: Dict[str, Any]) -> str:
        """Handle general help and assistance requests"""
        return """I'm Carroll, your advanced AI assistant! I can help you with:

🕐 Time and calendar information
🌤️ Weather forecasts and updates  
🔍 Web searches and information lookup
📱 Opening applications and system controls
🧮 Mathematical calculations and data analysis
📰 Latest news and current events
⏰ Smart reminders and scheduling
🎭 Entertainment like jokes, stories, and facts
💼 Productivity and task management
🎵 System controls like volume and brightness

I understand natural language, learn from our conversations, and provide personalized responses. Just speak naturally and I'll do my best to help! What would you like to try?"""

    def handle_unknown_intent(self, processed_command: Dict[str, Any]) -> str:
        """Handle unrecognized intents with learning capability"""
        original = processed_command.get("original", "")
        
        # Store unknown command for learning
        self.store_unknown_command(original)
        
        # Try to provide helpful suggestions
        suggestions = self.generate_suggestions(original)
        
        response = f"I'm not sure how to help with '{original}' yet, but I'm always learning!"
        
        if suggestions:
            response += f" Did you mean to ask about {', '.join(suggestions)}?"
        else:
            response += " Try asking me about the time, weather, or say 'help' to see what I can do!"
        
        return response

    def generate_suggestions(self, unknown_command: str) -> List[str]:
        """Generate helpful suggestions for unknown commands"""
        known_intents = [
            "time", "weather", "search", "open applications", "volume control",
            "calculations", "news", "jokes", "reminders", "system status"
        ]
        
        # Use simple similarity matching
        suggestions = []
        for intent in known_intents:
            similarity = SequenceMatcher(None, unknown_command.lower(), intent.lower()).ratio()
            if similarity > 0.3:
                suggestions.append(intent)
        
        return suggestions[:3]  # Return top 3 suggestions

    def record_audio_advanced(self, duration: int = 3) -> Optional[np.ndarray]:
        """Advanced audio recording with noise reduction"""
        try:
            print(f"🎤 Advanced listening for {duration} seconds...")
            
            # Record with higher quality settings
            audio = sd.rec(
                int(duration * self.SAMPLE_RATE),
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype='float32',
                blocking=True
            )
            
            # Apply basic noise reduction
            audio = self.apply_noise_reduction(audio)
            
            return (audio * 32767).astype(np.int16).flatten()
            
        except Exception as e:
            logger.error(f"Advanced recording error: {e}")
            return None

    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction to audio"""
        try:
            # Simple noise gate - reduce very quiet sounds
            threshold = np.max(np.abs(audio)) * 0.01
            audio[np.abs(audio) < threshold] *= 0.1
            
            return audio
            
        except Exception as e:
            logger.error(f"Noise reduction error: {e}")
            return audio

    def store_interaction(self, command: str, response: str, intent: str, sentiment: Dict[str, Any]):
        """Store interaction for learning and improvement"""
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (user_id, message, response, context, sentiment)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                self.current_user or "user",
                command,
                response,
                json.dumps({"intent": intent}),
                sentiment.get("score", 0.5)
            ))
            
            conn.commit()
            conn.close()
            
            # Update conversation history
            self.conversation_history.append({
                "command": command,
                "response": response,
                "intent": intent,
                "timestamp": datetime.now()
            })
            
            # Keep only recent history
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
                
        except Exception as e:
            logger.error(f"Interaction storage error: {e}")

    def store_unknown_command(self, command: str):
        """Store unknown commands for future learning"""
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT,
                    frequency INTEGER DEFAULT 1,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Check if command exists
            cursor.execute('SELECT frequency FROM unknown_commands WHERE command = ?', (command,))
            result = cursor.fetchone()
            
            if result:
                # Update frequency
                cursor.execute('''
                    UPDATE unknown_commands 
                    SET frequency = frequency + 1, last_seen = CURRENT_TIMESTAMP 
                    WHERE command = ?
                ''', (command,))
            else:
                # Insert new command
                cursor.execute('''
                    INSERT INTO unknown_commands (command) VALUES (?)
                ''', (command,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Unknown command storage error: {e}")

    def proactive_suggestions(self):
        """Generate proactive suggestions based on user patterns"""
        try:
            now = datetime.now()
            hour = now.hour
            
            suggestions = []
            
            # Morning suggestions
            if 6 <= hour <= 10:
                suggestions.extend([
                    "Good morning! Would you like to hear today's weather?",
                    "Ready to start your day? I can check your calendar or the latest news.",
                    "Morning! How about a motivational quote to start your day?"
                ])
            
            # Evening suggestions
            elif 17 <= hour <= 21:
                suggestions.extend([
                    "Good evening! Would you like a summary of today's news?",
                    "Time to wind down. Want to hear a relaxing story or joke?",
                    "Evening! Need help setting any reminders for tomorrow?"
                ])
            
            # Late night
            elif hour >= 22 or hour <= 5:
                suggestions.extend([
                    "It's getting late. Would you like me to set a morning alarm?",
                    "Late night productivity? I can help with calculations or searches.",
                    "Working late? I can help you stay focused with reminders."
                ])
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Proactive suggestions error: {e}")
            return []

    def load_user_memory(self):
        """Load user memory and learning patterns"""
        try:
            if os.path.exists(self.MEMORY_PATH):
                with open(self.MEMORY_PATH, 'rb') as f:
                    memory_data = pickle.load(f)
                    self.user_patterns = memory_data.get('user_patterns', {})
                    self.conversation_history = memory_data.get('conversation_history', [])
                    logger.info("User memory loaded successfully")
        except Exception as e:
            logger.error(f"Memory loading error: {e}")

    def save_user_memory(self):
        """Save user memory and learning patterns"""
        try:
            memory_data = {
                'user_patterns': self.user_patterns,
                'conversation_history': self.conversation_history[-100:],  # Keep last 100 interactions
                'context_tracker': self.context_tracker
            }
            
            with open(self.MEMORY_PATH, 'wb') as f:
                pickle.dump(memory_data, f)
                
            logger.info("User memory saved successfully")
        except Exception as e:
            logger.error(f"Memory saving error: {e}")

    def optimize_performance(self):
        """Continuously optimize performance based on usage patterns"""
        try:
            # Analyze response times
            if self.performance_stats['response_times']:
                avg_response_time = np.mean(self.performance_stats['response_times'])
                
                # Adjust settings if responses are too slow
                if avg_response_time > 3.0:
                    logger.warning("Slow response times detected, optimizing...")
                    # Could implement model switching or parameter adjustment
                
            # Clean up old data
            now = datetime.now()
            cutoff = now - timedelta(hours=24)
            
            # Remove old conversation history
            self.conversation_history = [
                conv for conv in self.conversation_history 
                if conv.get('timestamp', now) > cutoff
            ]
            
        except Exception as e:
            logger.error(f"Performance optimization error: {e}")

    def run_advanced_mode(self):
        """Run Carroll in advanced intelligent mode"""
        self.advanced_speak(
            "Hello! I'm Carroll, your advanced AI assistant. I'm much smarter than Siri or")

    async def run_advanced_mode(self):
        """Run Carroll in advanced mode"""
        self.advanced_speak("Hello! I'm Carroll, the smartest AI ever. Let's get started!", "excited")
        while self.listening:
            try:
                audio = self.record_audio_advanced()
                if audio is not None:
                    emotion = self.detect_speech_emotion(audio)
                    text, confidence = self.advanced_transcription(audio)
                    if self.intelligent_wake_word_detection(text, confidence):
                        self.processing_command = True
                        user_id, _ = self.advanced_user_identification(audio)
                        if user_id:
                            self.current_user = user_id
                        processed = self.advanced_command_processing(text)
                        response = self.intelligent_response_handler(processed)
                        self.advanced_speak(response, emotion=emotion)
                        suggestions = self.proactive_suggestions()
                        if suggestions and self.proactive_mode:
                            self.advanced_speak(random.choice(suggestions), "calm")
                        self.processing_command = False
                        self.optimize_performance()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.advanced_speak("Oops, something went wrong. Let's try again!", "concerned")
                await asyncio.sleep(1.0)

if platform.system() == "Emscripten":
    asyncio.ensure_future(AdvancedCarrollAssistant().run_advanced_mode())
else:
    if __name__ == "__main__":
        asyncio.run(AdvancedCarrollAssistant().run_advanced_mode())