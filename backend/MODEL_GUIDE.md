# Healthcare Chatbot Model Guide

This guide explains how to switch to larger or more specialized models for better healthcare responses.

## Current Model

Currently using: `microsoft/DialoGPT-medium` (345M parameters)
- General conversational model
- Not specifically trained for healthcare
- Good for basic conversations but may lack medical accuracy

## Recommended Models

### Option 1: Larger General Models (Better Quality, More Resources)

1. **GPT-2 Large** (774M parameters)
   ```bash
   MODEL_NAME=gpt2-large
   ```
   - Better language understanding
   - More coherent responses
   - Requires more RAM (~3GB)

2. **DialoGPT Large** (774M parameters)
   ```bash
   MODEL_NAME=microsoft/DialoGPT-large
   ```
   - Better for conversational AI
   - Improved context understanding
   - Requires more RAM (~3GB)

### Option 2: Healthcare-Specific Models (Best for Medical Accuracy)

1. **BioGPT** (Medical/Healthcare focused)
   ```bash
   MODEL_NAME=microsoft/biogpt
   ```
   - Trained on biomedical literature
   - Better medical terminology understanding
   - Good for healthcare questions

2. **BioBERT** (Requires different architecture)
   - Better for medical text understanding
   - May need code modifications

3. **ClinicalBERT** (If available)
   - Trained on clinical notes
   - Excellent for medical conversations

### Option 3: Modern Open-Source Models

1. **Mistral-7B** (7B parameters - requires significant RAM)
   ```bash
   MODEL_NAME=mistralai/Mistral-7B-v0.1
   ```
   - Much better quality
   - Requires 16GB+ RAM
   - May need quantization for smaller systems

2. **Llama 2 7B** (7B parameters)
   ```bash
   MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
   ```
   - Excellent conversational ability
   - Requires Hugging Face access token
   - Needs 16GB+ RAM

3. **Phi-2** (2.7B parameters - good balance)
   ```bash
   MODEL_NAME=microsoft/phi-2
   ```
   - Smaller but high quality
   - Good for healthcare with fine-tuning
   - Requires ~6GB RAM

## How to Switch Models

### Method 1: Environment Variable (Recommended)

1. Create or edit `.env` file in the `backend` directory:
   ```bash
   MODEL_NAME=microsoft/biogpt
   MODEL_FRAMEWORK=pt
   ```

2. Restart your backend server:
   ```bash
   cd backend
   python -m uvicorn src.main:app --reload
   ```

### Method 2: Direct Code Change

Edit `backend/src/main.py`:
```python
# Change this line:
healthbot = HealthBot()

# To:
healthbot = HealthBot(model_name="microsoft/biogpt")
```

### Method 3: Use Fine-Tuned Model

If you've fine-tuned a model:
```python
healthbot = HealthBot(
    model_name="microsoft/DialoGPT-medium",  # Base model
    use_fine_tuned=True  # Will load from ./models/final_healthcare_chatbot
)
```

## System Requirements

| Model | RAM Required | VRAM (GPU) | Download Size |
|-------|-------------|------------|---------------|
| DialoGPT-medium | 2GB | 1GB | ~700MB |
| DialoGPT-large | 3GB | 2GB | ~1.5GB |
| GPT-2 Large | 3GB | 2GB | ~1.5GB |
| BioGPT | 2GB | 1GB | ~700MB |
| Phi-2 | 6GB | 4GB | ~5GB |
| Mistral-7B | 16GB | 8GB | ~14GB |
| Llama-2-7B | 16GB | 8GB | ~14GB |

## Quick Start: Switch to BioGPT

1. **Set environment variable:**
   ```bash
   export MODEL_NAME=microsoft/biogpt
   ```

2. **Or create `.env` file:**
   ```bash
   echo "MODEL_NAME=microsoft/biogpt" > backend/.env
   ```

3. **Restart backend:**
   ```bash
   cd backend
   python -m uvicorn src.main:app --reload
   ```

The model will download automatically on first use (may take a few minutes).

## Fine-Tuning Your Model

For best results, fine-tune any model on your healthcare data:

```bash
cd backend
python train_model.py
```

This will create a fine-tuned model in `./models/final_healthcare_chatbot` that you can use with `use_fine_tuned=True`.

## Troubleshooting

### Out of Memory Errors
- Use a smaller model (DialoGPT-medium, BioGPT)
- Enable model quantization (requires code changes)
- Use CPU instead of GPU: `MODEL_FRAMEWORK=tf`

### Model Not Found
- Check internet connection (models download from Hugging Face)
- Verify model name is correct
- Some models require Hugging Face authentication

### Slow Responses
- Larger models are slower
- Consider using GPU: `MODEL_FRAMEWORK=pt` with CUDA
- Use smaller models for faster responses

