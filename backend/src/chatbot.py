import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TFAutoModelForCausalLM,
    GenerationConfig
)
import uuid
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
import os
import numpy as np
import evaluate
from difflib import get_close_matches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthBot:
    """
    Enhanced healthcare domain-specific chatbot using transformer models
    with comprehensive evaluation metrics and fine-tuning capabilities
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", use_fine_tuned: bool = False):
        self.model_name = os.getenv("MODEL_NAME", model_name)
        self.model_type = "generative"
        self.is_fine_tuned = use_fine_tuned
        self.model_framework = os.getenv("MODEL_FRAMEWORK", "pt").lower()
        self.fine_tuned_dir_tf = os.getenv("MODEL_DIR_TF", "./models_tf/final_healthcare_chatbot_tf")
        self.fine_tuned_dir_pt = os.getenv("MODEL_DIR_PT", "./models/final_healthcare_chatbot")
        self.conversations: Dict[str, List[Dict]] = {}
        self.evaluation_metrics = {}
        
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

        self.healthcare_keywords = {
            'symptoms', 'symptom', 'pain', 'fever', 'headache', 'cough', 'cold', 'flu',
            'medicine', 'medication', 'drug', 'pill', 'prescription', 'doctor', 'hospital', 'clinic', 'health', 'healthcare',
            'disease', 'illness', 'treatment', 'diagnosis', 'therapy', 'recovery', 'cure', 'heal',
            'blood pressure', 'diabetes', 'heart', 'lung', 'stomach', 'skin', 'brain', 'muscle', 'bone', 'joint',
            'mental health', 'anxiety', 'depression', 'stress', 'sleep', 'insomnia', 'fatigue',
            'exercise', 'diet', 'nutrition', 'vitamins', 'supplements', 'food', 'eating',
            'chest pain', 'breathing', 'emergency', 'urgent', 'severe', 'injury', 'wound',
            'chronic', 'acute', 'infection', 'inflammation', 'allergy', 'virus', 'bacteria',
            'vaccination', 'immunization', 'prevention', 'screening', 'test', 'exam',
            'reduce', 'manage', 'relief', 'help', 'improve', 'better', 'wellness', 'fitness',
            'tumor',
            'migraine', 'malaria', 'asthma', 'arthritis', 'hypertension', 'pneumonia', 'bronchitis',
            'nausea', 'vomiting', 'diarrhea', 'constipation', 'dizziness', 'weakness'
        }
        
        self.intent_keywords = {
            'symptom_inquiry': ['hurt', 'pain', 'ache', 'symptom', 'feel', 'sick'],
            'emergency': ['emergency', 'urgent', 'severe', 'chest pain', 'can\'t breathe'],
            'medication': ['medicine', 'drug', 'pill', 'prescription', 'dosage'],
            'lifestyle': ['exercise', 'diet', 'sleep', 'stress', 'lifestyle'],
            'prevention': ['prevent', 'avoid', 'reduce risk', 'healthy']
        }

        self._load_model()

    def _load_model(self):
        """Load the transformer model and tokenizer"""
        try:
            logger.info(f"Loading model: name={self.model_name}, framework={self.model_framework}")

            use_tf = self.model_framework == "tf"
            load_path = None
            if self.is_fine_tuned:
                load_path = self.fine_tuned_dir_tf if use_tf else self.fine_tuned_dir_pt
                if not os.path.exists(load_path):
                    logger.warning(f"Fine-tuned directory not found: {load_path}. Falling back to hub model name.")
                    load_path = None

            source = load_path or self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(source)

            if use_tf:
                self.model = TFAutoModelForCausalLM.from_pretrained(source)
                self.model_type = "generative"
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    source,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else None
                )
                self.model_type = "generative"

            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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
            self.model = None
            self.tokenizer = None
            self.generator = None

    def is_ready(self) -> bool:
        """Check if the model is ready for inference"""
        return self.model is not None and self.tokenizer is not None

    def _correct_medical_terms(self, text: str) -> str:
        """Correct common misspellings of medical terms using fuzzy matching"""
        words = text.split()
        corrected_words = []
        
        medical_terms_dict = {
            'diabetes': ['diabetis', 'diabeties', 'diabates', 'diabete', 'diabtes', 'diabet'],
            'migraine': ['migrane', 'migrain', 'migrane', 'migrane', 'migrane'],
            'malaria': ['maleria', 'malaira', 'malria', 'malaira'],
            'asthma': ['asma', 'asthma', 'asthma'],
            'arthritis': ['arthritus', 'arthiritis', 'arthritus', 'arthritus'],
            'hypertension': ['hypertention', 'hypertesion', 'hypertention', 'hypertensn'],
            'cancer': ['canser', 'cancr', 'cancer'],
            'pneumonia': ['pnumonia', 'pneumonia', 'pnumonia'],
            'bronchitis': ['bronchitis', 'bronchitis', 'bronchitis'],
            'headache': ['headach', 'headake', 'hedache', 'headach'],
            'fever': ['fevar', 'fever', 'fevar'],
            'cough': ['cogh', 'cough', 'cogh'],
            'pain': ['pane', 'pain', 'pane'],
            'nausea': ['nausia', 'nausa', 'nausea'],
            'fatigue': ['fatige', 'fatigue', 'fatige'],
            'dizziness': ['diziness', 'dizzyness', 'diziness', 'dizynes'],
            'symptom': ['symptom', 'symptom', 'symptom'],
            'symptoms': ['symptom', 'symptoms', 'symptom'],
            'treatment': ['treatement', 'treatmant', 'treatment', 'treatmnt'],
            'medication': ['medication', 'medicaton', 'medication', 'medicatin'],
            'diagnosis': ['diagnosis', 'diagnosis', 'diagnosis'],
            'doctor': ['docter', 'doctor', 'docter'],
            'hospital': ['hospitol', 'hospital', 'hospitol'],
            'medicine': ['medcine', 'medicin', 'medicine'],
        }
        
        all_medical_terms = set()
        for correct_term, misspellings in medical_terms_dict.items():
            all_medical_terms.add(correct_term)
            all_medical_terms.update(misspellings)
        
        for word in words:
            word_lower = word.lower()
            # Skip common words that shouldn't be corrected
            common_words = {'can', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must'}
            if word_lower in common_words:
                corrected_words.append(word)
                continue
                
            if word_lower in all_medical_terms:
                for correct_term, misspellings in medical_terms_dict.items():
                    if word_lower == correct_term or word_lower in misspellings:
                        corrected_words.append(correct_term)
                        break
                else:
                    corrected_words.append(word)
            else:
                close_matches = get_close_matches(word_lower, all_medical_terms, n=1, cutoff=0.7)
                if close_matches:
                    for correct_term, misspellings in medical_terms_dict.items():
                        if close_matches[0] == correct_term or close_matches[0] in misspellings:
                            corrected_words.append(correct_term)
                            break
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)
        
        original_words = text.split()
        final_words = []
        for orig, corr in zip(original_words, corrected_words):
            if orig.lower() != corr.lower():
                if orig[0].isupper():
                    final_words.append(corr.capitalize())
                else:
                    final_words.append(corr)
            else:
                final_words.append(orig)
        
        return ' '.join(final_words)

    def _check_domain_relevance(self, text: str) -> bool:
        """Check if the input is healthcare-related - accept all questions unless clearly non-healthcare"""
        text_lower = text.lower().strip()
        
        # List of clearly non-healthcare topics to reject
        non_healthcare_keywords = [
            'weather', 'forecast', 'temperature outside', 'rain', 'snow',
            'sports', 'football', 'basketball', 'soccer', 'baseball', 'score', 'game result',
            'cooking', 'recipe', 'how to cook', 'baking',
            'movie', 'film', 'cinema', 'actor', 'actress', 'director',
            'music', 'song', 'album', 'artist', 'singer',
            'travel', 'vacation', 'hotel', 'flight', 'trip',
            'shopping', 'buy', 'price', 'store', 'product',
            'politics', 'election', 'president', 'government', 'vote',
            'news', 'headline', 'breaking news',
            'technology', 'computer', 'software', 'programming', 'code',
            'car', 'vehicle', 'automobile', 'driving',
            'stock', 'market', 'investment', 'trading',
            'restaurant', 'food delivery', 'menu item'
        ]
        
        # If it contains clearly non-healthcare keywords, reject it
        if any(non_keyword in text_lower for non_keyword in non_healthcare_keywords):
            return False
        
        # Accept everything else - assume it's healthcare-related
        # This includes questions about exercise, diet, water, cholesterol, etc.
        return True

    def _generate_healthcare_response(self, message: str) -> str:
        """Generate healthcare-specific response"""
        if not self.is_ready():
            return self._get_fallback_response(message)

        try:
            corrected_message = self._correct_medical_terms(message)
            if corrected_message != message:
                logger.info(f"Corrected message: '{message}' -> '{corrected_message}'")
            message = corrected_message
            is_biogpt = "biogpt" in self.model_name.lower()
            
            if self.model_type == "generative":
                if is_biogpt:
                    input_text = f"Answer this medical question with practical, helpful information. Be direct and specific.\n\nQuestion: {message}\n\nPractical Answer:"
                else:
                    input_text = f"You are a knowledgeable medical assistant with expertise across all areas of medicine including symptoms, treatments, medications, diseases, conditions, and general health information. Provide a detailed, accurate, and helpful answer to this medical question. Include relevant information about symptoms, causes, treatments, or medications when applicable.\n\nMedical Question: {message}\n\nDetailed Answer:"
            else:
                input_text = f"Answer this health question accurately and professionally: {message}"

            if is_biogpt:
                max_new_tokens = 150
                temperature = 0.4
            else:
                max_new_tokens = 300
                temperature = 0.7
            
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.generator(
                input_text,
                max_new_tokens=generation_config.max_new_tokens,
                num_return_sequences=generation_config.num_return_sequences,
                temperature=generation_config.temperature,
                do_sample=generation_config.do_sample,
                top_p=generation_config.top_p,
                repetition_penalty=generation_config.repetition_penalty,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id
            )

            generated_text = response[0]['generated_text']
            logger.info(f"Raw generated text (first 500 chars): {generated_text[:500]}...")
            is_biogpt = "biogpt" in self.model_name.lower()

            if self.model_type == "generative":
                # First, try to remove the input prompt if it's at the start
                if input_text in generated_text:
                    response_text = generated_text[len(input_text):].strip()
                    logger.info(f"Removed input prompt, extracted: {response_text[:200]}...")
                else:
                    response_text = generated_text.strip()
                
                # If response is empty or just the prompt repeated, try to find actual content
                if not response_text or len(response_text.split()) < 3 or response_text.startswith("You are a"):
                    # Try to find content after common markers
                    for marker in ["Detailed Answer:", "Answer:", "Assistant:", "Doctor:", "Practical Answer:", "Helpful Answer:"]:
                        if marker in generated_text:
                            response_text = generated_text.split(marker)[-1].strip()
                            logger.info(f"Found content after {marker}: {response_text[:200]}...")
                            break
                
                # Then try to extract from markers in the cleaned response
                if is_biogpt:
                    if "Practical Answer:" in response_text:
                        response_text = response_text.split("Practical Answer:")[-1].strip()
                    elif "Helpful Answer:" in response_text:
                        response_text = response_text.split("Helpful Answer:")[-1].strip()
                    elif "Answer:" in response_text:
                        response_text = response_text.split("Answer:")[-1].strip()
                else:
                    if "Detailed Answer:" in response_text:
                        response_text = response_text.split("Detailed Answer:")[-1].strip()
                    elif "Answer:" in response_text:
                        response_text = response_text.split("Answer:")[-1].strip()
                    elif "Assistant:" in response_text:
                        response_text = response_text.split("Assistant:")[-1].strip()
                    elif "Doctor:" in response_text:
                        response_text = response_text.split("Doctor:")[-1].strip()
                
                # Clean up any remaining prompt artifacts
                if response_text.startswith("You are a medical assistant") or response_text.startswith("You are a helpful"):
                    if "Answer:" in response_text:
                        response_text = response_text.split("Answer:")[-1].strip()
                    elif "Doctor:" in response_text:
                        response_text = response_text.split("Doctor:")[-1].strip()
                    elif "Detailed Answer:" in response_text:
                        response_text = response_text.split("Detailed Answer:")[-1].strip()
                    else:
                        # If it's just the prompt, try to find any actual content
                        parts = response_text.split("Medical Question:")
                        if len(parts) > 1:
                            response_text = parts[-1].strip()
                            if "Detailed Answer:" in response_text:
                                response_text = response_text.split("Detailed Answer:")[-1].strip()
                
                if "Human:" in response_text:
                    response_text = response_text.split("Human:")[0].strip()
                if response_text.startswith("Assistant:"):
                    response_text = response_text.replace("Assistant:", "").strip()
                if response_text.startswith("Answer:"):
                    response_text = response_text.replace("Answer:", "").strip()
                
                # If still empty or just prompt, the model didn't generate properly
                if not response_text or len(response_text.split()) < 3 or response_text.startswith("You are"):
                    logger.warning(f"Model only generated prompt or empty response, will use fallback")
                    response_text = ""  # Force fallback
            else:
                response_text = generated_text.strip()

            logger.info(f"Extracted response text: {response_text[:200]}...")  # Log extracted response
            logger.info(f"Response length: {len(response_text.split())} words")
            
            is_generic_fallback = "consult a healthcare provider" in response_text.lower() and len(response_text.split()) < 25
            
            has_medical_content = any(keyword in response_text.lower() for keyword in [
                'symptom', 'treatment', 'medication', 'medicine', 'doctor', 'health', 'medical',
                'disease', 'condition', 'diagnosis', 'therapy', 'patient', 'care', 'hospital',
                'pain', 'fever', 'headache', 'cough', 'infection', 'diabetes', 'migraine', 'malaria',
                'cause', 'prevent', 'manage', 'therapy', 'clinical', 'disorder', 'syndrome', 'illness',
                'virus', 'bacteria', 'infection', 'inflammation', 'chronic', 'acute', 'prevention',
                'stress', 'anxiety', 'depression', 'exercise', 'diet', 'nutrition', 'sleep', 'relax',
                'breathing', 'meditation', 'yoga', 'wellness', 'lifestyle', 'fitness', 'mental health',
                'water', 'drink', 'daily', 'fluid', 'hydration', 'liters', 'ounces', 'cups',
                'cholesterol', 'fat', 'protein', 'carbohydrate', 'vitamin', 'mineral', 'calorie',
                'benefit', 'benefits', 'recommend', 'recommended', 'should', 'important', 'help',
                'body', 'healthy', 'unhealthy', 'improve', 'reduce', 'increase', 'decrease'
            ])
            
            # Be extremely lenient - accept almost anything the model generates
            is_reasonable_length = len(response_text.split()) > 3
            is_unhelpful = self._is_unhelpful_response(response_text)
            is_question = self._is_question_response(response_text)
            
            logger.info(f"Response check - Length: {is_reasonable_length}, Unhelpful: {is_unhelpful}, Question: {is_question}, Has medical content: {has_medical_content}")
            
            # Only reject if response is clearly broken (very short AND unhelpful AND no medical content)
            if is_reasonable_length and not (is_unhelpful and len(response_text.split()) < 5 and not has_medical_content):
                logger.info(f"Keeping model response ({len(response_text.split())} words)")
                pass
            else:
                logger.warning(f"Rejecting model response - too short/unhelpful, using fallback")
                response_text = self._get_fallback_response(message)

            response_text = self._clean_response(response_text)

            return response_text if response_text else self._get_fallback_response(message)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(message)

    def _get_fallback_response(self, message: str) -> str:
        """Fallback responses for when model fails or for non-healthcare queries"""
        corrected_message = self._correct_medical_terms(message)
        message_to_check = corrected_message if corrected_message != message else message
        
        # Only restrict clearly non-healthcare queries
        if not self._check_domain_relevance(message_to_check):
            return "I'm specialized in healthcare topics. Please ask me about symptoms, medications, treatments, or general health questions."

        message_lower = message_to_check.lower()
        
        # Provide helpful, specific answers for common questions
        if 'stress' in message_lower or 'anxiety' in message_lower:
            return "To reduce stress naturally, consider these approaches: 1) Regular exercise such as walking, yoga, or swimming can help release endorphins and reduce stress hormones. 2) Practice deep breathing exercises or meditation for 10-15 minutes daily. 3) Ensure adequate sleep (7-9 hours per night). 4) Maintain a balanced diet with whole foods and limit caffeine and alcohol. 5) Engage in hobbies or activities you enjoy. 6) Connect with friends and family for social support. 7) Consider mindfulness practices or progressive muscle relaxation. If stress persists or significantly impacts your daily life, consult a healthcare provider for personalized guidance and to rule out underlying conditions."
        
        elif 'water' in message_lower and ('drink' in message_lower or 'daily' in message_lower or 'much' in message_lower):
            return "The general recommendation for daily water intake is about 8-10 cups (64-80 ounces or 2-2.5 liters) per day for most adults. However, individual needs vary based on factors like age, activity level, climate, and overall health. A good indicator is the color of your urine - it should be light yellow. You can also get water from foods like fruits and vegetables. If you're very active, live in a hot climate, or are pregnant/breastfeeding, you may need more. Consult a healthcare provider for personalized recommendations, especially if you have kidney or heart conditions."
        
        elif 'diet' in message_lower and 'cholesterol' in message_lower:
            return "To lower cholesterol through diet, focus on: 1) Increasing soluble fiber from oats, barley, beans, and fruits. 2) Eating fatty fish like salmon 2-3 times per week for omega-3 fatty acids. 3) Choosing healthy fats like olive oil, avocados, and nuts instead of saturated fats. 4) Limiting red meat and full-fat dairy products. 5) Including plant sterols and stanols found in fortified foods. 6) Reducing trans fats and processed foods. 7) Eating plenty of fruits and vegetables. Regular exercise and maintaining a healthy weight also help. Consult a healthcare provider for personalized dietary recommendations and to determine if medication is needed."
        
        elif 'exercise' in message_lower or 'fitness' in message_lower:
            return "For general health and fitness: 1) Aim for at least 150 minutes of moderate-intensity exercise per week, or 75 minutes of vigorous activity. 2) Include a mix of cardiovascular exercise (walking, running, cycling), strength training (2-3 times per week), and flexibility exercises. 3) Start gradually if you're new to exercise. 4) Find activities you enjoy to maintain consistency. 5) Stay hydrated and fuel your body with nutritious foods. 6) Listen to your body and rest when needed. 7) Consider consulting a healthcare provider before starting a new exercise program, especially if you have health conditions."
        
        elif 'sleep' in message_lower:
            return "To improve sleep naturally: 1) Maintain a consistent sleep schedule, going to bed and waking up at the same time daily. 2) Create a relaxing bedtime routine. 3) Keep your bedroom cool, dark, and quiet. 4) Avoid screens (phones, computers, TV) at least 1 hour before bed. 5) Limit caffeine and avoid it after 2 PM. 6) Avoid large meals and alcohol close to bedtime. 7) Get regular exercise, but not too close to bedtime. 8) Manage stress through relaxation techniques. If sleep problems persist, consult a healthcare provider to rule out sleep disorders."
        
        else:
            # For other healthcare queries, provide a general helpful response
            topic = message_to_check.split('?')[0].split('.')[0].strip() if '?' in message_to_check or '.' in message_to_check else message_to_check
            return f"Regarding {topic}, I can provide general information, but for accurate diagnosis and personalized medical advice, it's important to consult a healthcare provider. They can assess your specific situation, consider your medical history, and provide the most current and appropriate guidance. If you're experiencing symptoms, seeking prompt medical attention is recommended for proper evaluation and treatment."

    def _is_question_response(self, response: str) -> bool:
        """Check if the response is asking a question instead of answering"""
        response_lower = response.lower().strip()
        question_starters = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'do you', 'are you', 'can you', 'would you', 'have you']
        if response_lower.endswith('?'):
            return True
        first_word = response_lower.split()[0] if response_lower.split() else ""
        if first_word in question_starters and len(response.split()) < 15:
            return True
        return False
    
    def _is_unhelpful_response(self, response: str) -> bool:
        """Check if the response is too short, vague, or unhelpful"""
        response_lower = response.lower().strip()
        
        if len(response) < 20 or len(response.split()) < 3:
            return True
        
        unhelpful_phrases = [
            'i have no idea', 'i don\'t know', 'i don\'t have', 'i cannot', 'i can\'t',
            'i\'m not sure', 'i\'m not certain', 'i don\'t have information',
            'i\'m sorry for being', 'i\'m sorry', 'overworked', 'sir', 'yes, but',
            'not sure how that will help', 'how that will help', 'a systematic review',
            'systematic review', 'meta-analysis', 'further research', 'more studies needed',
            'literature suggests', 'studies have shown', 'research indicates',
            'according to studies', 'clinical trial', 'randomized controlled trial'
        ]
        
        for phrase in unhelpful_phrases:
            if phrase in response_lower and len(response.split()) < 20:
                return True
        
        healthcare_keywords_in_response = [
            'symptom', 'treatment', 'medication', 'medicine', 'doctor', 'health', 'medical',
            'diabetes', 'pain', 'fever', 'headache', 'cough', 'infection', 'disease',
            'condition', 'diagnosis', 'therapy', 'patient', 'care', 'hospital'
        ]
        has_healthcare_keyword = any(keyword in response_lower for keyword in healthcare_keywords_in_response)
        
        if not has_healthcare_keyword and len(response.split()) < 15:
            conversational_unhelpful = ['yes, but', 'i\'m sorry', 'not sure', 'how that will help', 'sir', 'ma\'am']
            if any(phrase in response_lower for phrase in conversational_unhelpful):
                return True
        
        sentences = response.split('.')
        if len(sentences) == 1 and len(response.split()) < 8:
            return True
            
        return False

    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        response = response.replace(
            "Patient:", "").replace("Doctor:", "").replace("Question:", "").replace("Answer:", "").strip()

        if not response.endswith(('.', '!', '?')):
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '. '.join(sentences[:-1]) + '.'

        return response.strip()

    async def generate_response(self, message: str, conversation_id: Optional[str] = None) -> Dict:
        """Generate response for a user message"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

        domain_relevance = self._check_domain_relevance(message)
        response_text = self._generate_healthcare_response(message)
        confidence = 0.8 if self.is_ready() else 0.6

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

        enhanced_query = f"Analyze these symptoms: {query}"
        if context:
            enhanced_query += f" Context: {context}"

        analysis = self._generate_healthcare_response(enhanced_query)

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
        
        if any(keyword in text_lower for keyword in self.intent_keywords['emergency']):
            return 'emergency'
        
        for intent, keywords in self.intent_keywords.items():
            if intent != 'emergency' and any(keyword in text_lower for keyword in keywords):
                return intent
        
        return 'general_inquiry'
    
    def calculate_response_quality(self, user_input: str, bot_response: str) -> Dict:
        """Calculate quality metrics for bot response"""
        try:
            response_length = len(bot_response.split())
            input_length = len(user_input.split())
            
            has_disclaimer = any(phrase in bot_response.lower() for phrase in [
                'consult', 'healthcare', 'doctor', 'medical', 'professional'
            ])
            
            has_emergency_guidance = any(phrase in bot_response.lower() for phrase in [
                'emergency', 'urgent', 'immediately', 'call 911'
            ])
            
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
            healthcare_indicators = list(self.healthcare_keywords)[:10]
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
            
            try:
                generated_response = self._generate_healthcare_response(user_input)
                quality = self.calculate_response_quality(user_input, generated_response)
                quality_scores.append(quality)
                predictions.append(generated_response)
                references.append(expected_output)
            except Exception as e:
                logger.error(f"Error generating response for: {user_input}")
                continue
        
        try:
            bleu_results = self.bleu_metric.compute(predictions=predictions, references=references)
            bleu_score = bleu_results['bleu']
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            bleu_score = 0.0
        
        try:
            rouge_results = self.rouge_metric.compute(predictions=predictions, references=references)
            rouge_l = rouge_results['rougeL']
        except Exception as e:
            logger.error(f"Error calculating ROUGE score: {e}")
            rouge_l = 0.0
        
        avg_quality = {
            'avg_response_length': np.mean([q.get('response_length', 0) for q in quality_scores]),
            'avg_length_ratio': np.mean([q.get('length_ratio', 0) for q in quality_scores]),
            'disclaimer_rate': np.mean([q.get('has_disclaimer', False) for q in quality_scores]),
            'emergency_guidance_rate': np.mean([q.get('has_emergency_guidance', False) for q in quality_scores]),
            'avg_relevance': np.mean([q.get('response_relevance', 0) for q in quality_scores])
        }
        
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
