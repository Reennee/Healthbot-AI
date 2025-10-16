# Domain-Specific Chatbot Using Transformer Models (Healthcare)

- Demo Video: <ADD_LINK_HERE>

## 1. Project Definition & Domain Alignment
- Purpose: Provide healthcare information, symptom guidance, medication education, and safety reminders.
- Scope: Non-diagnostic guidance with clear disclaimers; recognizes emergencies and out-of-domain queries.
- Justification: Addresses high-volume health FAQs and triage education needs.

## 2. Dataset Collection & Preprocessing
- Sources: `backend/data/healthcare_conversations.json` and optional HF datasets (see notebook).
- Coverage: Symptom inquiries, emergencies, medication, lifestyle, prevention, chronic conditions, mental health.
- Preprocessing:
  - Normalization: lowercasing, whitespace collapse, safe char filtering.
  - Formatting for causal LM with special tokens: `"<|startoftext|>Patient: ...  
Doctor: ...  
"`.
  - Tokenization: HF `AutoTokenizer` with additional special tokens; padding token set to EOS when needed.
- Missing values handling: examples without paired responses are skipped.

## 3. Model Selection & Fine-tuning
- Baseline: `microsoft/DialoGPT-medium` for conversational generation.
- Approach: Generative QA (causal LM) with “Patient/Doctor” prompting.
- Implementations:
  - TensorFlow: `backend/src/training_pipeline_tf.py` (TFAutoModelForCausalLM + Keras custom loop).
  - PyTorch: `backend/src/training_pipeline.py` (HF Trainer).

## 4. Hyperparameter Tuning
- Search space (example): learning rate, batch size, epochs, warmup, weight decay.
- Experiments Table (fill with your runs):

| Exp | Framework | LR   | Batch | Epochs | Warmup | Weight Decay | Val Loss | Perplexity | BLEU | ROUGE-L |
|-----|-----------|------|-------|--------|--------|---------------|----------|------------|------|---------|
| 1   | PT        | 5e-5 | 2     | 2      | 0      | 0.0           |          |            |      |         |
| 2   | Torch     | 3e-5 | 8     | 5      | 200    | 0.01          |          |            |      |         |

- Summary: Describe best config and % improvement over baseline.

## 5. Evaluation & Metrics
- **Automatic Metrics**:
  - **BLEU Score**: 0.0 (Baseline)
  - **ROUGE-L**: 0.0875
  - **Perplexity**: Not evaluated in this run
- **Qualitative Tests**:
  - Domain relevance checks (Healthcare vs. General/Sports/Nutrition)
  - Intent classification (Symptom Inquiry, Emergency, Medication, Lifestyle)
  - Fallback response validation for out-of-domain queries
- **Files**:
  - `backend/src/chatbot.py` (Runtime BLEU/ROUGE aggregation and quality heuristics)
- **Results Summary**:
  - **BLEU**: 0.0
  - **ROUGE-L**: 0.0875
  - **Sample Size**: 20 test cases
  - **Domain Relevance**: High (heuristic-based filtering active)
  - **Intent Classification**: Functional for core categories

## 6. System Architecture & Interface
- **Backend**: FastAPI (`backend/src/main.py`) with endpoints:
  - `/chat`: Main conversational endpoint
  - `/analyze-symptoms`: Symptom assessment
  - `/conversation/*`: History management
  - `/model/*`: Performance metrics and info
- **Frontend**: Next.js client consuming `/chat`
- **Deployment**: vercel

## 7. Usage Instructions
1. **Backend Setup**:
   ```bash
   cd backend
   pip install -r requirements.txt
   python -m src.main
   ```
2. **Frontend Setup**:
   ```bash
   cd front-end
   npm install/ yarn install
   npm run dev/ yarn dev
   ```

## 8. Safety & Limitations
- **Disclaimer**: This bot provides information, not medical advice. Always consult a professional.
- **Emergency Detection**: Heuristics immediately block emergency queries (e.g., "heart attack") and advise calling 911.
- **Limitations**:
  - Small in-house dataset
  - Potential for hallucination (standard LLM risk)
  - Not a diagnostic tool

## 9. Conclusions & Future Work
- **Recap**: Successfully implemented a domain-aware chatbot with safety guardrails and fallback mechanisms.
- **Future Improvements**:
  - Integrate larger curated medical datasets (e.g., PubMedQA)
  - Implement Retrieval-Augmented Generation (RAG) for grounded answers
  - Add multilingual support
  - Improve calibration of confidence scores

## 10. References
- Hugging Face Transformers & Datasets
- Microsoft DialoGPT / BioGPT
- FastAPI Documentation
- Next.js Documentation


