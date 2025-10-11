import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    set_seed
)
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthBot:
    """
    Healthcare domain-specific chatbot using transformer models
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.model_type = "generative"  # or "extractive"
        self.is_fine_tuned = False
        self.conversations: Dict[str, List[Dict]] = {}

        # Healthcare domain keywords for relevance checking
        self.healthcare_keywords = {
            'symptoms', 'pain', 'fever', 'headache', 'cough', 'cold', 'flu',
            'medicine', 'medication', 'doctor', 'hospital', 'clinic', 'health',
            'disease', 'illness', 'treatment', 'diagnosis', 'therapy', 'recovery',
            'blood pressure', 'diabetes', 'heart', 'lung', 'stomach', 'skin',
            'mental health', 'anxiety', 'depression', 'stress', 'sleep',
            'exercise', 'diet', 'nutrition', 'vitamins', 'supplements'
        }

        # Initialize model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the transformer model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")

            # Try to load as causal LM first (GPT-style models)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                self.model_type = "generative"

                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            except Exception:
                # Fallback to seq2seq models (T5-style)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name)
                self.model_type = "seq2seq"

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info(f"Model loaded successfully. Type: {self.model_type}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a simple rule-based response system
            self.model = None
            self.tokenizer = None
            self.generator = None

    def is_ready(self) -> bool:
        """Check if the model is ready for inference"""
        return self.model is not None and self.tokenizer is not None

    def _check_domain_relevance(self, text: str) -> bool:
        """Check if the input is healthcare-related"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.healthcare_keywords)

    def _generate_healthcare_response(self, message: str) -> str:
        """Generate healthcare-specific response"""
        if not self.is_ready():
            return self._get_fallback_response(message)

        try:
            # Prepare input for different model types
            if self.model_type == "generative":
                # For GPT-style models, create a conversation context
                input_text = f"Patient: {message}\nDoctor:"
            else:
                # For T5-style models, use direct input
                input_text = f"Answer this health question: {message}"

            # Generate response
            response = self.generator(
                input_text,
                max_length=len(input_text.split()) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text = response[0]['generated_text']

            # Extract the response part
            if self.model_type == "generative":
                if "Doctor:" in generated_text:
                    response_text = generated_text.split("Doctor:")[-1].strip()
                else:
                    response_text = generated_text[len(input_text):].strip()
            else:
                response_text = generated_text.strip()

            # Clean up the response
            response_text = self._clean_response(response_text)

            return response_text if response_text else self._get_fallback_response(message)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(message)

    def _get_fallback_response(self, message: str) -> str:
        """Fallback responses for when model fails or for non-healthcare queries"""
        if not self._check_domain_relevance(message):
            return "I'm specialized in healthcare topics. Please ask me about symptoms, medications, or general health questions."

        # Simple rule-based responses for healthcare queries
        message_lower = message.lower()

        if any(word in message_lower for word in ['headache', 'head pain']):
            return "Headaches can have various causes including stress, dehydration, or tension. If they persist or are severe, please consult a healthcare provider."

        elif any(word in message_lower for word in ['fever', 'temperature']):
            return "Fever is often a sign of infection. Monitor your temperature and seek medical attention if it's above 103°F (39.4°C) or persists for more than 3 days."

        elif any(word in message_lower for word in ['cough', 'cold']):
            return "Coughs and colds are common. Rest, stay hydrated, and consider over-the-counter remedies. See a doctor if symptoms worsen or persist beyond 10 days."

        elif any(word in message_lower for word in ['pain']):
            return "Pain management depends on the type and severity. For persistent or severe pain, please consult a healthcare professional for proper evaluation."

        else:
            return "I understand you have health concerns. While I can provide general information, please consult a healthcare provider for proper medical advice and diagnosis."

    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove common artifacts
        response = response.replace(
            "Patient:", "").replace("Doctor:", "").strip()

        # Remove incomplete sentences at the end
        if response.endswith(('.', '!', '?')):
            pass
        else:
            # Find the last complete sentence
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '. '.join(sentences[:-1]) + '.'

        return response.strip()

    async def generate_response(self, message: str, conversation_id: Optional[str] = None) -> Dict:
        """Generate response for a user message"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Add user message to conversation history
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

        # Check domain relevance
        domain_relevance = self._check_domain_relevance(message)

        # Generate response
        response_text = self._generate_healthcare_response(message)

        # Calculate confidence (simplified)
        confidence = 0.8 if self.is_ready() else 0.6

        # Add bot response to conversation history
        self.conversations[conversation_id].append({
            "role": "bot",
            "message": response_text,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence
        })

        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "confidence": confidence,
            "domain_relevance": domain_relevance
        }

    async def analyze_symptoms(self, query: str, context: Optional[str] = None) -> Dict:
        """Analyze symptoms and provide preliminary assessment"""
        if not self._check_domain_relevance(query):
            return {
                "analysis": "This doesn't appear to be a health-related query.",
                "recommendation": "Please ask about symptoms, health concerns, or medical questions.",
                "urgency": "low"
            }

        # Enhanced response for symptom analysis
        enhanced_query = f"Analyze these symptoms: {query}"
        if context:
            enhanced_query += f" Context: {context}"

        analysis = self._generate_healthcare_response(enhanced_query)

        # Determine urgency level (simplified logic)
        urgency = "low"
        if any(word in query.lower() for word in ['severe', 'emergency', 'chest pain', 'difficulty breathing']):
            urgency = "high"
        elif any(word in query.lower() for word in ['pain', 'fever', 'persistent']):
            urgency = "medium"

        return {
            "analysis": analysis,
            "recommendation": "Consider consulting a healthcare provider for proper evaluation.",
            "urgency": urgency,
            "domain_relevance": True
        }

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history for a given ID"""
        return self.conversations.get(conversation_id, [])

    def delete_conversation(self, conversation_id: str):
        """Delete conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def save_conversations(self, filepath: str):
        """Save conversation history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.conversations, f, indent=2)

    def load_conversations(self, filepath: str):
        """Load conversation history from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.conversations = json.load(f)
