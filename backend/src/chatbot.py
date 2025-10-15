import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    set_seed,
    TFAutoModelForCausalLM
)
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json
import os
import numpy as np
import evaluate
from sklearn.metrics import f1_score, accuracy_score
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthBot:
    """
    Enhanced healthcare domain-specific chatbot using transformer models
    with comprehensive evaluation metrics and fine-tuning capabilities
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", use_fine_tuned: bool = False):
        # Allow env overrides for framework and model directories
        self.model_name = os.getenv("MODEL_NAME", model_name)
        self.model_type = "generative"  # or "extractive"
        self.is_fine_tuned = use_fine_tuned
        self.model_framework = os.getenv("MODEL_FRAMEWORK", "pt").lower()  # 'pt' or 'tf'
        self.fine_tuned_dir_tf = os.getenv("MODEL_DIR_TF", "./models_tf/final_healthcare_chatbot_tf")
        self.fine_tuned_dir_pt = os.getenv("MODEL_DIR_PT", "./models/final_healthcare_chatbot")
        self.conversations: Dict[str, List[Dict]] = {}
        self.evaluation_metrics = {}
        
        # Initialize evaluation metrics
        try:
            self.bleu_metric = evaluate.load("bleu")
        except Exception:
            logger.warning("BLEU metric not available, using fallback")
            self.bleu_metric = None
        
        try:
            self.rouge_metric = evaluate.load("rouge")
        except Exception:
            logger.warning("ROUGE metric not available, using fallback")
            self.rouge_metric = None

        # Enhanced healthcare domain keywords for relevance checking
        self.healthcare_keywords = {
            'symptoms', 'pain', 'fever', 'headache', 'cough', 'cold', 'flu',
            'medicine', 'medication', 'doctor', 'hospital', 'clinic', 'health',
            'disease', 'illness', 'treatment', 'diagnosis', 'therapy', 'recovery',
            'blood pressure', 'diabetes', 'heart', 'lung', 'stomach', 'skin',
            'mental health', 'anxiety', 'depression', 'stress', 'sleep',
            'exercise', 'diet', 'nutrition', 'vitamins', 'supplements',
            'chest pain', 'breathing', 'emergency', 'urgent', 'severe',
            'chronic', 'acute', 'infection', 'inflammation', 'allergy',
            'vaccination', 'immunization', 'prevention', 'screening'
        }
        
        # Intent classification keywords
        self.intent_keywords = {
            'symptom_inquiry': ['hurt', 'pain', 'ache', 'symptom', 'feel', 'sick'],
            'emergency': ['emergency', 'urgent', 'severe', 'chest pain', 'can\'t breathe'],
            'medication': ['medicine', 'drug', 'pill', 'prescription', 'dosage'],
            'lifestyle': ['exercise', 'diet', 'sleep', 'stress', 'lifestyle'],
            'prevention': ['prevent', 'avoid', 'reduce risk', 'healthy']
        }

        # Initialize model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the transformer model and tokenizer"""
        try:
            logger.info(f"Loading model: name={self.model_name}, framework={self.model_framework}")

            # Determine source: fine-tuned dir or hub model name
            use_tf = self.model_framework == "tf"
            load_path = None
            if self.is_fine_tuned:
                load_path = self.fine_tuned_dir_tf if use_tf else self.fine_tuned_dir_pt
                if not os.path.exists(load_path):
                    logger.warning(f"Fine-tuned directory not found: {load_path}. Falling back to hub model name.")
                    load_path = None

            source = load_path or self.model_name

            # Try to load as causal LM first (GPT-style models) with selected framework
            self.tokenizer = AutoTokenizer.from_pretrained(source)

            if use_tf:
                # TensorFlow causal LM
                self.model = TFAutoModelForCausalLM.from_pretrained(source)
                self.model_type = "generative"
            else:
                # PyTorch causal LM
                self.model = AutoModelForCausalLM.from_pretrained(
                    source,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else None
                )
                self.model_type = "generative"

            # Add padding token if not present
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Create text generation pipeline (framework auto-detected by transformers)
            device = 0 if (not use_tf and torch.cuda.is_available()) else -1
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
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
    
    def classify_intent(self, text: str) -> str:
        """Classify user intent from input text"""
        text_lower = text.lower()
        
        # Check for emergency keywords first
        if any(keyword in text_lower for keyword in self.intent_keywords['emergency']):
            return 'emergency'
        
        # Check other intents
        for intent, keywords in self.intent_keywords.items():
            if intent != 'emergency' and any(keyword in text_lower for keyword in keywords):
                return intent
        
        return 'general_inquiry'
    
    def calculate_response_quality(self, user_input: str, bot_response: str) -> Dict:
        """Calculate quality metrics for bot response"""
        try:
            # Basic metrics
            response_length = len(bot_response.split())
            input_length = len(user_input.split())
            
            # Check for medical disclaimers
            has_disclaimer = any(phrase in bot_response.lower() for phrase in [
                'consult', 'healthcare', 'doctor', 'medical', 'professional'
            ])
            
            # Check for emergency guidance
            has_emergency_guidance = any(phrase in bot_response.lower() for phrase in [
                'emergency', 'urgent', 'immediately', 'call 911'
            ])
            
            # Intent classification accuracy (simplified)
            intent = self.classify_intent(user_input)
            response_relevance = self._check_response_relevance(intent, bot_response)
            
            return {
                'response_length': response_length,
                'input_length': input_length,
                'length_ratio': response_length / max(input_length, 1),
                'has_disclaimer': has_disclaimer,
                'has_emergency_guidance': has_emergency_guidance,
                'intent': intent,
                'response_relevance': response_relevance,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error calculating response quality: {e}")
            return {'error': str(e)}
    
    def _check_response_relevance(self, intent: str, response: str) -> float:
        """Check how relevant the response is to the detected intent"""
        response_lower = response.lower()
        
        if intent == 'emergency':
            emergency_indicators = ['emergency', 'urgent', 'immediately', 'call', '911']
            return sum(1 for indicator in emergency_indicators if indicator in response_lower) / len(emergency_indicators)
        
        elif intent == 'symptom_inquiry':
            symptom_indicators = ['symptom', 'pain', 'condition', 'cause', 'treatment']
            return sum(1 for indicator in symptom_indicators if indicator in response_lower) / len(symptom_indicators)
        
        elif intent == 'medication':
            medication_indicators = ['medication', 'medicine', 'drug', 'dosage', 'side effect']
            return sum(1 for indicator in medication_indicators if indicator in response_lower) / len(medication_indicators)
        
        else:
            # General relevance based on healthcare keywords
            healthcare_indicators = list(self.healthcare_keywords)[:10]  # Use first 10 keywords
            return sum(1 for indicator in healthcare_indicators if indicator in response_lower) / len(healthcare_indicators)
    
    def evaluate_model_performance(self, test_data: List[Dict]) -> Dict:
        """Comprehensive model performance evaluation"""
        logger.info("Evaluating model performance...")
        
        predictions = []
        references = []
        quality_scores = []
        
        for example in test_data:
            user_input = example.get('user_input', '')
            expected_output = example.get('assistant_output', '')
            
            # Generate response
            try:
                generated_response = self._generate_healthcare_response(user_input)
                
                # Calculate quality metrics
                quality = self.calculate_response_quality(user_input, generated_response)
                quality_scores.append(quality)
                
                predictions.append(generated_response)
                references.append(expected_output)
                
            except Exception as e:
                logger.error(f"Error generating response for: {user_input}")
                continue
        
        # Calculate BLEU score
        try:
            bleu_results = self.bleu_metric.compute(
                predictions=predictions,
                references=references
            )
            bleu_score = bleu_results['bleu']
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            bleu_score = 0.0
        
        # Calculate ROUGE score
        try:
            rouge_results = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            rouge_l = rouge_results['rougeL']
        except Exception as e:
            logger.error(f"Error calculating ROUGE score: {e}")
            rouge_l = 0.0
        
        # Calculate average quality metrics
        avg_quality = {
            'avg_response_length': np.mean([q.get('response_length', 0) for q in quality_scores]),
            'avg_length_ratio': np.mean([q.get('length_ratio', 0) for q in quality_scores]),
            'disclaimer_rate': np.mean([q.get('has_disclaimer', False) for q in quality_scores]),
            'emergency_guidance_rate': np.mean([q.get('has_emergency_guidance', False) for q in quality_scores]),
            'avg_relevance': np.mean([q.get('response_relevance', 0) for q in quality_scores])
        }
        
        # Store evaluation results
        self.evaluation_metrics = {
            'bleu_score': bleu_score,
            'rouge_l': rouge_l,
            'quality_metrics': avg_quality,
            'total_samples': len(test_data),
            'successful_predictions': len(predictions),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Evaluation completed. BLEU: {bleu_score:.3f}, ROUGE-L: {rouge_l:.3f}")
        
        return self.evaluation_metrics
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            'model_info': {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'is_fine_tuned': self.is_fine_tuned,
                'is_ready': self.is_ready()
            },
            'evaluation_metrics': self.evaluation_metrics,
            'conversation_stats': {
                'total_conversations': len(self.conversations),
                'total_messages': sum(len(conv) for conv in self.conversations.values())
            },
            'healthcare_domain_coverage': {
                'keywords_covered': len(self.healthcare_keywords),
                'intent_categories': len(self.intent_keywords)
            }
        }
