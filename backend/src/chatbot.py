
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TFAutoModelForCausalLM,
    GenerationConfig
)
import uuid
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
    """Enhanced healthcare domain-specific chatbot using transformer models.

    Notes:
    - Allows skipping model loading (useful for unit tests or offline utilities) via ``load_model``.
    - Uses light-weight heuristics and fallback responses when model is unavailable.
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", use_fine_tuned: bool = False, load_model: bool = True):
        """Initialize the chatbot.

        Args:
            model_name: hub model name to use when no fine-tuned model is provided.
            use_fine_tuned: whether to attempt loading a fine-tuned model from disk.
            load_model: if False, skip loading the transformers model (useful for tests/offline ops).
        """
        self.model_name = os.getenv("MODEL_NAME", model_name)
        self.model_type = "generative"
        self.is_fine_tuned = use_fine_tuned
        self.model_framework = os.getenv("MODEL_FRAMEWORK", "pt").lower()
        self.fine_tuned_dir_tf = os.getenv("MODEL_DIR_TF", "./models_tf/final_healthcare_chatbot_tf")
        self.fine_tuned_dir_pt = os.getenv("MODEL_DIR_PT", "./models/final_healthcare_chatbot")
        self.conversations: Dict[str, List[Dict]] = {}
        self.evaluation_metrics: Dict = {}

        # Model placeholders (may remain None when load_model=False)
        self.model = None
        self.tokenizer = None
        self.generator = None

        # Metrics may not be available in some environments (offline), keep optional
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
            'symptom_inquiry': ['hurt', 'pain', 'ache', 'symptom', 'feel', 'sick', 'fever', 'cough', 'headache'],
            'emergency': ['emergency', 'urgent', 'severe', 'chest pain', 'can\'t breathe', 'heart attack', 'stroke', 'bleeding', 'unconscious'],
            'medication': ['medicine', 'drug', 'pill', 'prescription', 'dosage', 'side effect', 'take'],
            'lifestyle': ['exercise', 'diet', 'sleep', 'stress', 'lifestyle', 'food', 'eat', 'workout', 'weight'],
            'prevention': ['prevent', 'avoid', 'reduce risk', 'healthy', 'protect']
        }

        # Auto-detect fine-tuned models on disk if available (convenience)
        # DISABLED: This was causing it to load an old DialoGPT model instead of BioGPT
        # try:
        #     if not use_fine_tuned:
        #         if os.path.exists(self.fine_tuned_dir_tf):
        #             logger.info(f"Found fine-tuned TF model at {self.fine_tuned_dir_tf}, enabling fine-tuned loading.")
        #             use_fine_tuned = True
        #             self.model_framework = "tf"
        #         elif os.path.exists(self.fine_tuned_dir_pt):
        #             logger.info(f"Found fine-tuned PT model at {self.fine_tuned_dir_pt}, enabling fine-tuned loading.")
        #             use_fine_tuned = True
        #             self.model_framework = "pt"
        # except Exception:
        #     # If any filesystem check fails, continue without auto-detect
        #     pass

        # A lookup table for misspellings will be built lazily in _build_spelling_lookup
        self._spelling_lookup: Optional[Dict[str, str]] = None

        # Load model only when requested (tests can set load_model=False)
        if load_model:
            # pass the possibly-updated use_fine_tuned flag to the loader
            self.is_fine_tuned = use_fine_tuned
            self._load_model()
            # Load latest evaluation results if available
            self._load_latest_evaluation_results()

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

            # If source is a local path, prefer local_files_only for from_pretrained to avoid
            # treating the path as a hub repo id. This fixes loading from local saved models.
            is_local = os.path.exists(str(source))
            self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=is_local)

            if use_tf:
                # TFAutoModel supports loading from local saved TF models
                self.model = TFAutoModelForCausalLM.from_pretrained(source, local_files_only=is_local)
                self.model_type = "generative"
            else:
                # Use safetensors to bypass torch.load security restriction
                self.model = AutoModelForCausalLM.from_pretrained(
                    source,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else None,
                    local_files_only=is_local,
                    use_safetensors=True
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

    def _load_latest_evaluation_results(self):
        """Load the latest evaluation results from the evaluation_results directory"""
        try:
            import glob
            import json
            
            # Get the directory where evaluation results are stored
            eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation_results')
            
            if not os.path.exists(eval_dir):
                logger.info("No evaluation results directory found")
                return
            
            # Find all evaluation report files
            report_files = glob.glob(os.path.join(eval_dir, 'evaluation_report_*.json'))
            
            if not report_files:
                logger.info("No evaluation reports found")
                return
            
            # Get the most recent file
            latest_report = max(report_files, key=os.path.getmtime)
            
            # Load the evaluation results
            with open(latest_report, 'r') as f:
                results = json.load(f)
            
            # Extract model_performance if it exists
            model_perf = results.get('model_performance', results)
            
            # Store in evaluation_metrics
            self.evaluation_metrics = {
                'bleu_score': model_perf.get('bleu_score'),
                'rouge_l': model_perf.get('rouge_l'),
                'total_samples': model_perf.get('total_samples'),
                'successful_predictions': model_perf.get('successful_predictions'),
                'evaluation_timestamp': model_perf.get('evaluation_timestamp') or results.get('evaluation_timestamp'),
                'quality_metrics': model_perf.get('quality_metrics', {})
            }
            
            logger.info(f"Loaded evaluation results from {os.path.basename(latest_report)}")
            logger.info(f"BLEU: {self.evaluation_metrics.get('bleu_score')}, ROUGE-L: {self.evaluation_metrics.get('rouge_l')}, Samples: {self.evaluation_metrics.get('total_samples')}")

            
        except Exception as e:
            logger.warning(f"Could not load evaluation results: {e}")

    def is_ready(self) -> bool:
        """Check if the model is ready for inference"""
        return self.model is not None and self.tokenizer is not None

    def _correct_medical_terms(self, text: str) -> str:
        """Correct common misspellings of medical terms using a fast lookup.

        This function preserves basic punctuation and capitalization. It uses a
        pre-built lookup table (misspelling -> canonical term) for O(1) lookup
        and falls back to fuzzy matching for unknown tokens.
        """
        if not text or not text.strip():
            return text

        # Build a lookup dict lazily for speed
        if self._spelling_lookup is None:
            medical_terms_dict = {
                'diabetes': ['diabetis', 'diabeties', 'diabates', 'diabete', 'diabtes', 'diabet'],
                'migraine': ['migrane', 'migrain'],
                'malaria': ['maleria', 'malaira', 'malria'],
                'asthma': ['asma'],
                'arthritis': ['arthritus', 'arthiritis'],
                'hypertension': ['hypertention', 'hypertesion', 'hypertensn'],
                'cancer': ['canser', 'cancr'],
                'pneumonia': ['pnumonia'],
                'headache': ['headach', 'headake', 'hedache'],
                'fever': ['fevar'],
                'cough': ['cogh'],
                'pain': ['pane'],
                'nausea': ['nausia', 'nausa'],
                'fatigue': ['fatige'],
                'dizziness': ['diziness', 'dizzyness', 'dizynes'],
                'treatment': ['treatement', 'treatmant', 'treatmnt'],
                'medication': ['medicaton', 'medicatin'],
                'doctor': ['docter'],
                'hospital': ['hospitol'],
                'medicine': ['medcine', 'medicin'],
            }

            lookup: Dict[str, str] = {}
            for correct, misspellings in medical_terms_dict.items():
                lookup[correct] = correct
                for m in misspellings:
                    lookup[m] = correct

            self._spelling_lookup = lookup

        lookup = self._spelling_lookup

        # Split while preserving punctuation
        import re
        tokens = re.findall(r"\w+|[^	\w\s]", text, re.UNICODE)
        corrected_tokens: List[str] = []

        common_words = {'can', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must'}

        for tok in tokens:
            low = tok.lower()
            if low in common_words or not tok.isalpha():
                corrected_tokens.append(tok)
                continue

            if low in lookup:
                corrected = lookup[low]
                # Preserve capitalization
                if tok[0].isupper():
                    corrected_tokens.append(corrected.capitalize())
                else:
                    corrected_tokens.append(corrected)
                continue

            # Fallback: try fuzzy matching to the known canonical terms
            close = get_close_matches(low, list({v for v in lookup.values()}), n=1, cutoff=0.85)
            if close:
                corrected = close[0]
                if tok[0].isupper():
                    corrected_tokens.append(corrected.capitalize())
                else:
                    corrected_tokens.append(corrected)
            else:
                corrected_tokens.append(tok)

        # Reconstruct text (simple join - punctuation tokens were preserved separately)
        # Merge tokens with spaces where appropriate
        out = ''
        prev_was_alnum = False
        for t in corrected_tokens:
            if out == '':
                out = t
                prev_was_alnum = t[-1].isalnum()
                continue
            if t.isalnum() and prev_was_alnum:
                out += ' ' + t
                prev_was_alnum = True
            elif t.isalnum() and not prev_was_alnum:
                out += ' ' + t
                prev_was_alnum = True
            else:
                # punctuation directly appended
                out += t
                prev_was_alnum = False

        return out

    def _detect_domain(self, text: str) -> str:
        """Detect if the text is about healthcare, sports, nutrition, or general."""
        t = text.lower()
        sports_keywords = {'sport', 'sports', 'football', 'soccer', 'basketball', 'run', 'running', 'training', 'coach', 'game', 'score', 'fitness', 'gym', 'workout'}
        nutrition_keywords = {'diet', 'nutrition', 'calorie', 'protein', 'fat', 'carbohydrate', 'vitamin', 'mineral', 'meal', 'eating', 'food', 'weight loss', 'healthy eating'}

        if any(k in t for k in self.healthcare_keywords):
            return 'healthcare'
        if any(k in t for k in sports_keywords):
            return 'sports'
        if any(k in t for k in nutrition_keywords):
            return 'nutrition'
        return 'general'

    def _build_dialogue_context(self, conversation_id: str, user_message: str, max_tokens: int = 512) -> str:
        """Construct a dialogue history string for conversational models (DialoGPT).
        
        Uses token counting to ensure the context fits within the model's limit.
        """
        if not self.tokenizer:
            # Fallback if tokenizer is not loaded
            return f"User: {user_message}\nBot:"

        history_items = []
        conv = self.conversations.get(conversation_id, [])
        
        # Start with the current message
        current_prompt = f"User: {user_message}\nBot:"
        current_tokens = len(self.tokenizer.encode(current_prompt))
        
        # Iterate backwards through history
        for msg in reversed(conv):
            role = msg.get('role', 'user')
            text = msg.get('message', '')
            
            if role == 'user':
                line = f"User: {text}"
            else:
                line = f"Bot: {text}"
            
            line_tokens = len(self.tokenizer.encode(line))
            
            if current_tokens + line_tokens + 1 > max_tokens: # +1 for newline
                break
            
            history_items.insert(0, line)
            current_tokens += line_tokens + 1
            
        # Seed with persona if history is short/empty to guide the model
        if len(history_items) < 2:
            seed_user = "User: Hello, who are you?"
            seed_bot = "Bot: I am a helpful AI medical assistant. I can answer your questions about symptoms, medications, and health."
            history_items.insert(0, seed_bot)
            history_items.insert(0, seed_user)

        history_items.append(current_prompt)
        return "\n".join(history_items)

    def _check_domain_relevance(self, text: str) -> bool:
        """Check if the input is healthcare-related - accept all questions unless clearly non-healthcare"""
        text_lower = text.lower().strip()
        
        # List of clearly non-healthcare topics to reject
        non_healthcare_keywords = [
            'weather', 'forecast', 'temperature outside', 'rain', 'snow',
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
        return True

    def _generate_healthcare_response(self, message: str, conversation_id: Optional[str] = None) -> tuple[str, bool]:
        """Generate a response for the given message and (optional) conversation history.

        Returns:
            tuple: (response_text, is_fallback)
        """
        logger.warning(f"DEBUG: Entering _generate_healthcare_response with message: {message}")
        logger.warning(f"DEBUG: self.is_ready() = {self.is_ready()}")

        if not self.is_ready():
            return self._get_fallback_response(message), True

        try:
            corrected_message = self._correct_medical_terms(message)
            if corrected_message != message:
                logger.info(f"Corrected message: '{message}' -> '{corrected_message}'")
            message = corrected_message
            
            is_dialogpt = any(k in self.model_name.lower() for k in ['dialo', 'dialog', 'dlgpt'])
            domain = self._detect_domain(message)
            intent = self.classify_intent(message)

            # IMMEDIATE SAFETY CHECK: If emergency, return fallback immediately
            if intent == 'emergency':
                logger.warning(f"Emergency intent detected for message: {message}")
                return self._get_fallback_response(message), True

            # Dynamic parameters based on intent/domain
            if intent in ['symptom_inquiry', 'medication']:
                # More deterministic for serious topics
                temperature = 0.6 # Relaxed from 0.5
                top_p = 0.92
                repetition_penalty = 1.05 # Relaxed from 1.2
                max_new_tokens = 150
            elif domain == 'lifestyle' or domain == 'sports' or domain == 'nutrition':
                # More creative for lifestyle
                temperature = 0.8
                top_p = 0.95
                repetition_penalty = 1.0
                max_new_tokens = 200
            else:
                # Balanced for general
                temperature = 0.7
                top_p = 0.95
                repetition_penalty = 1.0
                max_new_tokens = 180

            # Choose prompt template by domain
            if is_dialogpt:
                # Use token-aware context construction
                ctx = self._build_dialogue_context(conversation_id or str(uuid.uuid4()), message, max_tokens=512)
                input_text = ctx
            else:
                # BioGPT and other generation models work best with completion prompts
                # We'll use a simple QA format or just the question if it's a completion model
                if 'bio' in self.model_name.lower():
                    # BioGPT works best with few-shot prompting to establish the pattern
                    if message.endswith('?'):
                        input_text = (
                            "Question: What is a fever?\n"
                            "Answer: A fever is a temporary increase in your body's temperature, often due to an illness.\n"
                            "Question: What is hypertension?\n"
                            "Answer: Hypertension (high blood pressure) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems.\n"
                            f"Question: {message}\n"
                            "Answer:"
                        )
                    else:
                        input_text = f"{message}"
                else:
                    if domain == 'healthcare':
                        input_text = (
                            f"You are a knowledgeable medical assistant. Answer the question concisely and accurately.\n\nQuestion: {message}\n\nAnswer:")
                    elif domain == 'sports':
                        input_text = (
                            f"You are an experienced sports coach and fitness trainer. Provide practical training or injury-prevention advice.\n\nQuestion: {message}\n\nAnswer:")
                    elif domain == 'nutrition':
                        input_text = (
                            f"You are a registered dietitian. Give clear, evidence-based nutrition advice and suggestions.\n\nQuestion: {message}\n\nAnswer:")
                    else:
                        input_text = f"Answer this question clearly and helpfully: {message}"

            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            gen_kwargs = {
                'max_new_tokens': generation_config.max_new_tokens,
                'num_return_sequences': generation_config.num_return_sequences,
                'temperature': generation_config.temperature,
                'do_sample': generation_config.do_sample,
                'top_p': generation_config.top_p,
                'repetition_penalty': generation_config.repetition_penalty,
            }
            if getattr(generation_config, 'pad_token_id', None) is not None:
                gen_kwargs['pad_token_id'] = generation_config.pad_token_id
            if getattr(generation_config, 'eos_token_id', None) is not None:
                gen_kwargs['eos_token_id'] = generation_config.eos_token_id

            response = self.generator(input_text, **gen_kwargs)
            generated_text = response[0].get('generated_text', '')
            logger.info(f"Raw generated text (first 500 chars): {generated_text[:500]}...")

            # Extract reply for dialogue-style prompts
            if is_dialogpt:
                # Keep everything after the last 'Bot:' marker (if any)
                if 'Bot:' in generated_text:
                    response_text = generated_text.split('Bot:')[-1].strip()
                else:
                    response_text = generated_text.strip()
                # Remove any trailing 'User:' artifact
                if 'User:' in response_text:
                    response_text = response_text.split('User:')[0].strip()
            else:
                # For non-dialogue models, try to remove the question prefix
                if input_text in generated_text:
                    response_text = generated_text[len(input_text):].strip()
                else:
                    response_text = generated_text.strip()

            response_text = self._clean_response(response_text)

            # Check if response is valid and helpful
            if response_text and len(response_text.strip()) > 0:
                # Reject if the model is asking a question instead of answering
                if self._is_question_response(response_text):
                    logger.warning(f"Model asked a question instead of answering: {response_text[:100]}")
                    return self._get_fallback_response(message), True
                # Reject if response is unhelpful
                if self._is_unhelpful_response(response_text):
                    logger.warning(f"Model gave unhelpful response: {response_text[:100]}")
                    return self._get_fallback_response(message), True
                return response_text, False
            else:
                logger.warning("Model output empty or too short, falling back to canned response")
                return self._get_fallback_response(message), True

        except Exception as e:
            logger.exception("Error generating response")
            return self._get_fallback_response(message), True

    def _get_fallback_response(self, message: str) -> str:
        """Fallback responses for when model fails or for non-healthcare queries"""
        corrected_message = self._correct_medical_terms(message)
        message_to_check = corrected_message if corrected_message != message else message
        
        # Only restrict clearly non-healthcare queries
        if not self._check_domain_relevance(message_to_check):
            return "I'm specialized in healthcare topics. Please ask me about symptoms, medications, treatments, or general health questions."

        message_lower = message_to_check.lower()
        intent = self.classify_intent(message_to_check)

        # Structured fallbacks based on intent
        if intent == 'emergency':
             return "If you are experiencing a medical emergency, please call emergency services (911) immediately or go to the nearest emergency room. I cannot provide emergency medical assistance."
        
        # Common Symptom Fallbacks
        if 'stomach' in message_lower or 'abdominal' in message_lower:
            return "Stomach ache or abdominal pain can be caused by many factors including indigestion, gas, gastritis, or food poisoning. Common symptoms include cramps, bloating, nausea, and heartburn. If pain is severe, persistent, or accompanied by fever/vomiting, seek medical attention."

        if 'fever' in message_lower:
            return "A fever is usually a sign that your body is fighting an infection. Common symptoms include sweating, chills, headache, muscle aches, and dehydration. Rest and fluids are important. Seek medical care if fever is very high (over 103°F/39.4°C) or lasts more than 3 days."

        if 'cough' in message_lower or 'cold' in message_lower or 'flu' in message_lower:
            # Check if asking specifically about medication
            if 'medication' in message_lower or 'medicine' in message_lower or 'drug' in message_lower or 'pill' in message_lower or 'take' in message_lower or 'treatment' in message_lower:
                return "Common over-the-counter medications for cold and flu include: 1) Pain relievers/fever reducers (acetaminophen, ibuprofen) for aches and fever. 2) Decongestants (pseudoephedrine, phenylephrine) for stuffy nose. 3) Cough suppressants (dextromethorphan) for dry cough. 4) Expectorants (guaifenesin) for productive cough. 5) Antihistamines for runny nose and sneezing. Always follow dosage instructions and consult a pharmacist or doctor if you have questions or other health conditions."
            else:
                return "Cough, cold, and flu symptoms often include runny nose, sore throat, fever, and body aches. Rest, hydration, and over-the-counter remedies can help manage symptoms. Consult a doctor if you have difficulty breathing or symptoms persist."

        if 'diabetes' in message_lower:
            return "Diabetes management typically involves monitoring blood sugar, healthy eating, regular exercise, and medication or insulin therapy as prescribed. Symptoms of high blood sugar include increased thirst, frequent urination, and fatigue. Consult a doctor for a treatment plan."

        if 'asthma' in message_lower:
            return "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. Symptoms include difficulty breathing, chest pain, cough, and wheezing. Treatment usually involves rescue inhalers and long-term control medications."

        if 'hypertension' in message_lower or 'blood pressure' in message_lower:
            return "Hypertension (high blood pressure) often has no symptoms but can cause serious health problems. Management involves a healthy diet with less salt, regular exercise, maintaining a healthy weight, and limiting alcohol. Medication may also be prescribed."

        if 'arthritis' in message_lower:
            return "Arthritis involves swelling and tenderness of one or more joints. Main symptoms are joint pain and stiffness, which typically worsen with age. Treatments vary but often include medication, physiotherapy, and sometimes surgery."

        if 'migraine' in message_lower:
            return "A migraine is a headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound."

        if 'allergy' in message_lower or 'allergies' in message_lower:
            return "Allergies occur when your immune system reacts to a foreign substance such as pollen, bee venom or pet dander. Symptoms range from sneezing and itching to anaphylaxis. Treatments include antihistamines, decongestants, and avoiding triggers."

        if 'malaria' in message_lower:
            return (
                "Common symptoms of malaria include:\n"
                "- Fever and chills\n"
                "- Headache\n"
                "- Muscle aches and fatigue\n"
                "- Sweats and shaking chills\n"
                "- Nausea, vomiting, or diarrhea\n"
                "- Anemia and jaundice (in severe cases)\n"
                "If you suspect malaria, seek medical attention promptly; malaria can be serious and requires testing and treatment."
            )
        
        if 'rash' in message_lower or 'skin irritation' in message_lower:
            return "Rashes can have many causes including allergies, infections, heat, or irritants. Common treatments include: 1) Keep the area clean and dry. 2) Apply cool compresses for relief. 3) Use over-the-counter hydrocortisone cream for itching. 4) Avoid scratching. 5) Identify and avoid triggers. Seek medical attention if the rash is severe, spreading rapidly, accompanied by fever, or shows signs of infection (pus, warmth, red streaks)."
        
        if 'stress' in message_lower or 'anxiety' in message_lower:
            return "To reduce stress naturally, consider these approaches: 1) Regular exercise such as walking, yoga, or swimming can help release endorphins. 2) Practice deep breathing exercises or meditation. 3) Ensure adequate sleep (7-9 hours). 4) Maintain a balanced diet. 5) Connect with friends and family. If stress persists, consult a healthcare provider."
        
        if 'water' in message_lower and ('drink' in message_lower or 'daily' in message_lower):
            return "The general recommendation for daily water intake is about 8-10 cups (2-2.5 liters) per day for most adults. However, individual needs vary based on activity level and climate. Listen to your body and drink when thirsty."
        
        if 'diet' in message_lower and 'cholesterol' in message_lower:
            return "To lower cholesterol through diet: 1) Increase soluble fiber (oats, beans, fruits). 2) Eat fatty fish (salmon) for omega-3s. 3) Choose healthy fats (olive oil, avocados). 4) Limit red meat and full-fat dairy. 5) Eat plenty of fruits and vegetables."
        
        
        # Exercise and Fitness Fallbacks
        if any(word in message_lower for word in ['exercise', 'fitness', 'workout', 'running', 'run', 'walking', 'walk', 'jogging', 'jog', 'swimming', 'swim', 'cycling', 'bike', 'yoga', 'gym', 'cardio', 'training']):
            if 'stomach' in message_lower or 'belly' in message_lower or 'abs' in message_lower or 'abdominal' in message_lower:
                return "To reduce stomach fat: 1) Combine cardio (running, cycling, swimming) with core exercises (planks, crunches). 2) Maintain a calorie deficit through healthy eating. 3) Avoid sugary drinks and processed foods. 4) Stay consistent - aim for 30-45 minutes of exercise 5 days/week. 5) Get adequate sleep (7-9 hours). Note: Spot reduction isn't possible; overall fat loss is key."
            elif 'weight loss' in message_lower or 'lose weight' in message_lower:
                return "For weight loss through exercise: 1) Combine cardio (150+ minutes/week) with strength training (2-3 days/week). 2) Try HIIT workouts for efficient calorie burning. 3) Stay active throughout the day. 4) Pair exercise with a balanced, calorie-controlled diet. 5) Track progress and stay consistent. Consult a healthcare provider before starting any new program."
            elif 'running' in message_lower or 'run' in message_lower or 'jogging' in message_lower or 'jog' in message_lower:
                return "Running is an excellent cardiovascular exercise with many benefits: 1) Improves heart health and endurance. 2) Burns calories effectively (about 100 calories per mile). 3) Strengthens bones and muscles. 4) Reduces stress and improves mood. 5) Can be done almost anywhere with minimal equipment. Start slowly if you're a beginner, wear proper shoes, warm up before and cool down after, and listen to your body to avoid injury."
            else:
                return "For general health and fitness: 1) Aim for at least 150 minutes of moderate aerobic exercise per week (brisk walking, cycling, swimming). 2) Include strength training 2-3 times per week for all major muscle groups. 3) Add flexibility exercises like stretching or yoga. 4) Start gradually and increase intensity over time. 5) Stay hydrated and listen to your body. Consult a doctor before starting if you have health conditions."
        
        if 'diet' in message_lower or 'nutrition' in message_lower:
            if 'weight loss' in message_lower or 'lose weight' in message_lower:
                return "For healthy weight loss: 1) Create a moderate calorie deficit (500-750 calories/day for 1-1.5 lbs/week loss). 2) Eat plenty of vegetables, fruits, lean proteins, and whole grains. 3) Limit processed foods, sugary drinks, and excessive fats. 4) Eat regular meals to avoid extreme hunger. 5) Stay hydrated. 6) Combine with regular exercise. Consult a registered dietitian for personalized guidance."
            elif 'healthy' in message_lower or 'balanced' in message_lower:
                return "For a healthy, balanced diet: 1) Fill half your plate with vegetables and fruits. 2) Choose whole grains over refined grains. 3) Include lean proteins (fish, poultry, beans, nuts). 4) Limit saturated fats, added sugars, and sodium. 5) Stay hydrated with water. 6) Practice portion control. 7) Eat a variety of foods for diverse nutrients."
        
        if 'headache' in message_lower:
            return "Common headache symptoms include dull aching pain, throbbing (migraine), sensitivity to light/sound, or pressure. If headaches are severe, sudden, or accompanied by neurological changes, seek emergency care."
        
        if 'sleep' in message_lower:
            if 'child' in message_lower or 'kid' in message_lower or 'toddler' in message_lower:
                return "Sleep needs by age: Toddlers (1-2 years): 11-14 hours. Preschoolers (3-5 years): 10-13 hours. School-age (6-12 years): 9-12 hours. Teens (13-18 years): 8-10 hours. Consistent bedtime routines are important for all ages."
            else:
                return "To improve sleep: 1) Maintain a consistent schedule. 2) Create a relaxing bedtime routine. 3) Keep your bedroom cool and dark. 4) Avoid screens before bed. 5) Limit caffeine. If problems persist, consult a provider."
        
        # COVID-19
        if 'covid' in message_lower or 'coronavirus' in message_lower:
            return "Common COVID-19 symptoms include: fever, cough, fatigue, loss of taste/smell, sore throat, congestion, body aches, and shortness of breath. Symptoms can range from mild to severe. If you have difficulty breathing, persistent chest pain, or confusion, seek emergency care immediately. Get tested if you have symptoms or exposure."
        
        # Dehydration
        if 'dehydrat' in message_lower:
            return "Common signs of dehydration include: dark yellow urine, dry mouth, fatigue, dizziness, decreased urination, and thirst. Severe dehydration may cause confusion, rapid heartbeat, or fainting. Drink water regularly throughout the day. Seek medical care if symptoms are severe or persist despite drinking fluids."
        
        # Heart Health
        if 'heart attack' in message_lower:
            return "Warning signs of a heart attack include: chest pain/pressure, pain radiating to arm/jaw/back, shortness of breath, cold sweat, nausea, and lightheadedness. Women may experience atypical symptoms like fatigue or indigestion. CALL 911 IMMEDIATELY if you suspect a heart attack. Time is critical - don't wait or drive yourself."
        
        if 'high blood pressure' in message_lower or ('blood pressure' in message_lower and 'high' in message_lower):
            return "High blood pressure (hypertension) is often caused by: genetics, age, obesity, high salt intake, lack of exercise, stress, smoking, and excessive alcohol. It often has no symptoms but increases risk of heart disease and stroke. To reduce it naturally: 1) Reduce sodium intake. 2) Exercise regularly. 3) Maintain healthy weight. 4) Limit alcohol. 5) Manage stress. 6) Eat potassium-rich foods. Always consult a doctor for diagnosis and treatment."
        
        # Mental Health  
        if 'anxiety' in message_lower and 'sign' in message_lower:
            return "Common signs of anxiety include: excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep problems, rapid heartbeat, and avoiding certain situations. If anxiety interferes with daily life, consult a mental health professional. Treatment options include therapy, medication, and lifestyle changes."
        
        if 'depression' in message_lower or 'burnout' in message_lower:
            return "Depression involves persistent sadness, loss of interest, changes in sleep/appetite, fatigue, feelings of worthlessness, and difficulty concentrating for 2+ weeks. Burnout is work-related exhaustion with cynicism and reduced performance. Both are serious - depression is a medical condition requiring treatment, while burnout may improve with rest and workplace changes. Seek professional help for either."
        
        # Nutrition
        if 'balanced diet' in message_lower or ('diet' in message_lower and 'balanced' in message_lower):
            return "A balanced diet includes: 1) Fruits and vegetables (half your plate). 2) Whole grains (brown rice, whole wheat, oats). 3) Lean proteins (fish, poultry, beans, nuts). 4) Low-fat dairy or alternatives. 5) Healthy fats (olive oil, avocados, nuts). Limit: added sugars, saturated fats, and sodium. Stay hydrated with water."
        
        if 'carbohydrate' in message_lower or 'carbs' in message_lower:
            return "Carbohydrates aren't inherently bad - they're your body's main energy source. Choose: 1) Complex carbs (whole grains, vegetables, legumes) - provide sustained energy and fiber. 2) Limit simple carbs (white bread, sugary foods) - cause blood sugar spikes. The key is choosing nutrient-dense, fiber-rich carbs and controlling portions."
        
        if 'snack' in message_lower and 'healthy' in message_lower:
            return "Healthy snack options: 1) Fresh fruit with nut butter. 2) Greek yogurt with berries. 3) Vegetables with hummus. 4) Handful of nuts or seeds. 5) Whole grain crackers with cheese. 6) Hard-boiled eggs. 7) Air-popped popcorn. Choose snacks with protein and fiber to keep you satisfied."
        
        # Children's Health
        if 'ear infection' in message_lower:
            return "Signs of ear infection in children: ear pain, tugging at ears, trouble sleeping, fussiness, fever, fluid draining from ear, difficulty hearing, and loss of balance. Babies may cry more than usual or have trouble feeding. See a doctor if symptoms persist more than a day, fever is high, or there's fluid discharge."
        
        if 'toddler' in message_lower and 'avoid' in message_lower:
            return "Foods toddlers should avoid: 1) Choking hazards (whole grapes, nuts, popcorn, hard candy, hot dogs). 2) Honey (for under 1 year - botulism risk). 3) Unpasteurized dairy/juice. 4) High-mercury fish (shark, swordfish). 5) Excessive juice or sugary drinks. 6) Foods with added salt/sugar. Always supervise eating and cut food into small pieces."
        
        # Illness Prevention
        if 'food poisoning' in message_lower and 'prevent' in message_lower:
            return "Prevent food poisoning: 1) Wash hands before cooking/eating. 2) Cook meat to safe temperatures (165°F for poultry). 3) Refrigerate perishables within 2 hours. 4) Avoid cross-contamination (separate raw/cooked foods). 5) Wash produce thoroughly. 6) Don't eat expired foods. 7) Be cautious with raw eggs, unpasteurized dairy, and undercooked seafood."
        
        if 'check-up' in message_lower or 'checkup' in message_lower:
            return "Adults should get a general check-up: 1) Annually if you have chronic conditions or are over 50. 2) Every 2-3 years if healthy and under 50. Check-ups typically include: blood pressure, weight, cholesterol screening, diabetes screening, and age-appropriate cancer screenings. Your doctor may recommend more frequent visits based on your health status."
        
        # Medication
        if 'antibiotic' in message_lower and 'antiviral' in message_lower:
            return "Antibiotics treat bacterial infections (strep throat, UTIs, bacterial pneumonia) by killing bacteria. Antivirals treat viral infections (flu, COVID-19, HIV, herpes) by slowing virus reproduction. Key difference: antibiotics don't work on viruses, and antivirals don't work on bacteria. Misuse contributes to antibiotic resistance."
        
        if 'stop medication' in message_lower or 'stopping medication' in message_lower:
            return "Stopping medication suddenly can be dangerous because: 1) Symptoms may return worse than before. 2) Withdrawal effects can occur (especially with antidepressants, blood pressure meds, steroids). 3) Conditions may worsen rapidly. 4) Some medications need gradual tapering. Always consult your doctor before stopping any prescription medication - they can create a safe discontinuation plan."
        
        if 'ibuprofen' in message_lower and 'avoid' in message_lower:
            return "Avoid ibuprofen if you: 1) Have stomach ulcers or bleeding disorders. 2) Have severe kidney or liver disease. 3) Are in the third trimester of pregnancy. 4) Have aspirin allergy. 5) Are taking blood thinners. 6) Have heart disease (consult doctor first). Always take with food to reduce stomach irritation and follow dosage instructions."
        
        # Medical Tests
        if 'blood test' in message_lower and 'check' in message_lower:
            return "Blood tests can check for: 1) Complete blood count (anemia, infection, blood disorders). 2) Metabolic panel (kidney/liver function, electrolytes, blood sugar). 3) Lipid panel (cholesterol levels). 4) Thyroid function. 5) Vitamin levels. 6) Markers of inflammation or disease. Your doctor orders specific tests based on symptoms or screening needs."
        
        if 'high cholesterol' in message_lower or ('cholesterol' in message_lower and 'high' in message_lower):
            return "High cholesterol means too much cholesterol in your blood, increasing heart disease and stroke risk. LDL ('bad' cholesterol) should be low; HDL ('good' cholesterol) should be high. Causes include: genetics, diet high in saturated fats, lack of exercise, obesity, and smoking. Manage through: healthy diet, regular exercise, weight loss, and sometimes medication."
        
        if 'mri' in message_lower:
            return "An MRI (Magnetic Resonance Imaging) scan uses powerful magnets and radio waves to create detailed images of organs and tissues. It's used to: diagnose conditions, detect tumors, examine brain/spine injuries, assess joint problems, and evaluate organ function. It's non-invasive, painless, and doesn't use radiation. The scan is loud and requires lying still for 30-60 minutes."
        
        # General fallback
        return "I understand you have a health-related question. While I can't provide a specific diagnosis, I recommend consulting a healthcare professional who can evaluate your specific situation and provide personalized advice."

    def _is_question_response(self, response: str) -> bool:
        """Check if the response is asking an unhelpful question instead of answering
        
        Allows: Clarifying questions that help provide better answers
        Blocks: Questions that just deflect without providing any information
        """
        response_lower = response.lower().strip()
        
        # If response doesn't end with '?', it's not a question
        if not response_lower.endswith('?'):
            return False
        
        # Allow clarifying questions that show the bot is trying to help
        helpful_question_patterns = [
            'can you tell me more about',
            'how long have you',
            'have you experienced',
            'are you currently taking',
            'do you have any other symptoms',
            'is the pain',
            'does it hurt when',
            'have you tried',
            'when did',
            'how severe',
            'on a scale of'
        ]
        
        for pattern in helpful_question_patterns:
            if pattern in response_lower:
                return False  # This is a helpful clarifying question, allow it
        
        # Block unhelpful questions that just deflect
        unhelpful_patterns = [
            'what is the diagnosis',
            'what is your',
            'what do you',
            'who are you',
            'where are you',
            'why are you',
            'what are you'
        ]
        
        for pattern in unhelpful_patterns:
            if pattern in response_lower:
                return True  # This is unhelpful, block it
        
        # If it's a very short question (< 8 words), likely unhelpful
        if len(response.split()) < 8:
            return True
        
        # Otherwise, allow the question (it's likely a clarifying question)
        return False
    
    def _is_unhelpful_response(self, response: str) -> bool:
        """Check if the response is too short, vague, or unhelpful"""
        response_lower = response.lower().strip()
        
        if len(response) < 20 or len(response.split()) < 3:
            return True
        
        # Check for gibberish (excessive punctuation or very short words)
        import re
        # Count punctuation marks
        punctuation_count = len(re.findall(r'[.,!?;:]', response))
        word_count = len(response.split())
        
        # If more than 50% of tokens are punctuation, it's gibberish
        if word_count > 0 and punctuation_count / word_count > 0.5:
            return True
        
        # Check for repetitive characters or patterns
        if re.search(r'(.)\1{4,}', response):  # Same character repeated 5+ times
            return True
        
        # Check for very short average word length (gibberish indicator)
        words = [w for w in response.split() if w.isalpha()]
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < 2.5:  # Average word length too short
                return True
        
        unhelpful_phrases = [
            'i have no idea', 'i don\'t know', 'i don\'t have', 'i cannot', 'i can\'t',
            'i\'m not sure', 'i\'m not certain', 'i don\'t have information',
            'i\'m sorry for being', 'i\'m sorry', 'overworked', 'sir', 'yes, but',
            'not sure how that will help', 'how that will help', 'a systematic review',
            'systematic review', 'meta-analysis', 'further research', 'more studies needed',
            'literature suggests', 'studies have shown', 'research indicates',
            'according to studies', 'clinical trial', 'randomized controlled trial',
            'me too', 'same here', 'i agree'
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
        response_text, is_fallback = self._generate_healthcare_response(message, conversation_id=conversation_id)
        
        # Dynamic confidence score
        if is_fallback:
            confidence = 0.6
        else:
            # Higher confidence for generated responses, slightly adjusted by relevance
            confidence = 0.85 if domain_relevance else 0.75

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

        analysis, _ = self._generate_healthcare_response(enhanced_query)

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
        try:
            dirpath = os.path.dirname(filepath)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.conversations, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversations to {filepath}: {e}")

    def load_conversations(self, filepath: str):
        """Load conversation history from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.conversations = json.load(f)
        except Exception as e:
            logger.error(f"Error loading conversations from {filepath}: {e}")
    
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
                generated_response, _ = self._generate_healthcare_response(user_input)
                quality = self.calculate_response_quality(user_input, generated_response)
                quality_scores.append(quality)
                predictions.append(generated_response)
                references.append(expected_output)
            except Exception as e:
                logger.error(f"Error generating response for: {user_input}")
                continue
        
        # Compute BLEU/ROUGE if metrics are available. Ensure correct input shapes.
        try:
            if self.bleu_metric is not None:
                # BLEU often expects references as list of list of refs per prediction
                bleu_results = self.bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
                bleu_score = bleu_results.get('bleu', 0.0)
            else:
                bleu_score = 0.0
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            bleu_score = 0.0

        try:
            if self.rouge_metric is not None:
                rouge_results = self.rouge_metric.compute(predictions=predictions, references=references)
                # rouge metric returns a dict with rouge1/rouge2/rougeL keys
                rouge_l = rouge_results.get('rougeL', rouge_results.get('rougeLsum', 0.0))
            else:
                rouge_l = 0.0
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
