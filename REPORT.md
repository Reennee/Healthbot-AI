# Domain-Specific Chatbot Using Transformer Models (Healthcare)

- GitHub Repository: <ADD_LINK_HERE>
- Demo Video (5–10 min): <ADD_LINK_HERE>

## 1. Project Definition & Domain Alignment
- Purpose: Provide healthcare information, symptom guidance, medication education, and safety reminders.
- Scope: Non-diagnostic guidance with clear disclaimers; recognizes emergencies and out-of-domain queries.
- Justification: Addresses high-volume health FAQs and triage education needs.

## 2. Dataset Collection & Preprocessing
- Sources: `backend/data/healthcare_conversations.json` and optional HF datasets (see notebook).
- Coverage: Symptom inquiries, emergencies, medication, lifestyle, prevention, chronic conditions, mental health.
- Preprocessing:
  - Normalization: lowercasing, whitespace collapse, safe char filtering.
  - Formatting for causal LM with special tokens: `"<|startoftext|>Patient: ... <|endoftext|>Doctor: ... <|endoftext|>"`.
  - Tokenization: HF `AutoTokenizer` with additional special tokens; padding token set to EOS when needed.
- Missing values handling: examples without paired responses are skipped.

## 3. Model Selection & Fine-tuning
- Baseline: Pre-trained `gpt2` or `microsoft/DialoGPT-medium` for conversational generation.
- Approach: Generative QA (causal LM) with “Patient/Doctor” prompting.
- Implementations:
  - TensorFlow: `backend/src/training_pipeline_tf.py` (TFAutoModelForCausalLM + Keras custom loop).
  - PyTorch: `backend/src/training_pipeline.py` (HF Trainer).

## 4. Hyperparameter Tuning
- Search space (example): learning rate, batch size, epochs, warmup, weight decay.
- Experiments Table (fill with your runs):

| Exp | Framework | LR   | Batch | Epochs | Warmup | Weight Decay | Val Loss | Perplexity | BLEU | ROUGE-L |
|-----|-----------|------|-------|--------|--------|---------------|----------|------------|------|---------|
| 1   | TF        | 5e-5 | 2     | 2      | 0      | 0.0           |          |            |      |         |
| 2   | Torch     | 3e-5 | 8     | 5      | 200    | 0.01          |          |            |      |         |

- Summary: Describe best config and % improvement over baseline.

## 5. Evaluation & Metrics
- Automatic Metrics:
  - BLEU, ROUGE-L (PyTorch pipeline) and perplexity (both pipelines).
- Qualitative Tests: Domain relevance, intent classification, sample conversations.
- Files:
  - `backend/evaluate_model.py` (qualitative + metric collection)
  - `backend/src/chatbot.py` (runtime BLEU/ROUGE aggregation and quality heuristics)
- Results Summary:
  - BLEU: <fill>
  - ROUGE-L: <fill>
  - Perplexity: <fill>
  - Domain relevance accuracy: <fill>
  - Intent classification accuracy: <fill>

## 6. System Architecture & Interface
- Backend: FastAPI (`backend/src/main.py`) with endpoints: `/chat`, `/analyze-symptoms`, `/conversation/*`, `/model/*`.
- Frontend: Next.js client consuming `/chat`.
- Deployment: Dockerfile snippet in `backend/README.md`.

## 7. Usage Instructions
- Backend setup and run (see `backend/README.md`).
- TensorFlow training: `python -m src.training_pipeline_tf`.
- API usage examples and curl commands.

## 8. Safety & Limitations
- Non-medical advice disclaimer; refer to professionals.
- Emergency detection heuristic; advise calling emergency services when indicated.
- Limitations: small in-house dataset, hallucination risk, not diagnostic.

## 9. Conclusions & Future Work
- Recap improvements from fine-tuning.
- Future: larger curated datasets, retrieval augmentation, calibration, structured triage flows, multilingual support.

## 10. References
- Hugging Face Transformers, Datasets
- Relevant clinical guidance references (add if used)


