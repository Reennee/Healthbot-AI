# Quick Guide: Switch to a Better Model

## 🚀 Fastest Way: Use BioGPT (Healthcare-Focused)

BioGPT is specifically trained on biomedical literature and will give much better healthcare responses.

### Step 1: Create `.env` file
```bash
cd backend
echo "MODEL_NAME=microsoft/biogpt" > .env
```

### Step 2: Restart your backend
```bash
# Stop your current server (Ctrl+C)
# Then restart:
python -m uvicorn src.main:app --reload
```

That's it! The model will download automatically (takes 2-5 minutes first time).

---

## 📊 Other Model Options

### Option 1: DialoGPT Large (Better Quality)
```bash
echo "MODEL_NAME=microsoft/DialoGPT-large" > backend/.env
```
- **Pros**: Better conversation quality, more coherent
- **Cons**: Needs ~3GB RAM (vs 2GB for medium)
- **Best for**: General improvement without specialization

### Option 2: GPT-2 Large
```bash
echo "MODEL_NAME=gpt2-large" > backend/.env
```
- **Pros**: Very good language understanding
- **Cons**: Not healthcare-specific, needs ~3GB RAM
- **Best for**: Better general responses

### Option 3: Phi-2 (High Quality, Modern)
```bash
echo "MODEL_NAME=microsoft/phi-2" > backend/.env
```
- **Pros**: Modern architecture, excellent quality
- **Cons**: Needs ~6GB RAM, larger download
- **Best for**: Best quality if you have enough RAM

---

## 💡 Recommended: BioGPT

**Why BioGPT?**
- ✅ Trained on medical/healthcare data
- ✅ Understands medical terminology
- ✅ Similar size to current model (no extra RAM needed)
- ✅ Better healthcare-specific responses

**How to use:**
```bash
cd backend
echo "MODEL_NAME=microsoft/biogpt" > .env
python -m uvicorn src.main:app --reload
```

---

## 🔧 Using Your Fine-Tuned Model

If you've already fine-tuned a model:

```bash
# In backend/.env
MODEL_NAME=microsoft/DialoGPT-medium
USE_FINE_TUNED=true
```

This will use your fine-tuned model from `./models/final_healthcare_chatbot`

---

## ⚠️ Troubleshooting

**Out of Memory?**
- Stick with: `microsoft/DialoGPT-medium` or `microsoft/biogpt`
- Avoid: Large models if you have <4GB RAM

**Model not downloading?**
- Check internet connection
- Some models need Hugging Face account (free)
- First download takes time (models are 500MB-5GB)

**Want to test without changing code?**
```bash
# Temporary (just for this session):
export MODEL_NAME=microsoft/biogpt
python -m uvicorn src.main:app --reload
```

---

## 📈 Model Comparison

| Model | Healthcare Quality | RAM Needed | Speed |
|-------|-------------------|------------|-------|
| DialoGPT-medium (current) | ⭐⭐ | 2GB | Fast |
| **BioGPT** | ⭐⭐⭐⭐ | 2GB | Fast |
| DialoGPT-large | ⭐⭐⭐ | 3GB | Medium |
| Phi-2 | ⭐⭐⭐⭐ | 6GB | Medium |

**Recommendation**: Start with **BioGPT** - best healthcare quality without extra RAM!

