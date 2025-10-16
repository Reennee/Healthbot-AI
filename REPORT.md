# Domain-Specific Chatbot Using Transformer Models (Healthcare)
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

## 3. Model Selection & Fine-tuning
- Baseline: `microsoft/BioGPT-large` for conversational generation.
- Approach: Generative QA (causal LM) with “Patient/Doctor” prompting.
- Implementations:
  - PyTorch: `backend/src/training_pipeline_pt.py` (HF Trainer).  

## 4. Evaluation & Metrics
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

## 5. System Architecture & Interface
- **Backend**: FastAPI (`backend/src/main.py`) with endpoints:
  - `/chat`: Main conversational endpoint
  - `/analyze-symptoms`: Symptom assessment
  - `/conversation/*`: History management
  - `/model/*`: Performance metrics and info
- **Frontend**: Next.js client consuming `/chat`
- **Deployment**: vercel

## 6. Usage Instructions
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

## 7. Safety & Limitations
- **Disclaimer**: This bot provides information, not medical advice. Always consult a professional.
- **Emergency Detection**: Heuristics immediately block emergency queries (e.g., "heart attack") and advise calling 911.
- **Limitations**:
  - Small in-house dataset
  - Potential for hallucination (standard LLM risk)
  - Not a diagnostic tool

## 8. Conclusions & Future Work
- **Recap**: Successfully implemented a domain-aware chatbot with safety guardrails and fallback mechanisms.
- **Future Improvements**:
  - Integrate larger curated medical datasets (e.g., PubMedQA)
  - Implement Retrieval-Augmented Generation (RAG) for grounded answers
  - Add multilingual support
  - Improve calibration of confidence scores

## 9. References
- Hugging Face Transformers & Datasets
- Microsoft BioGPT
- FastAPI Documentation
- Next.js Documentation


