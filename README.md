# HealthBot AI - Advanced Healthcare Domain Chatbot

A comprehensive, production-ready healthcare domain-specific chatbot built with Next.js frontend and FastAPI backend, powered by fine-tuned transformer models from Hugging Face. This project demonstrates advanced NLP techniques including hyperparameter tuning, comprehensive evaluation metrics, and domain-specific optimization.

## 🏥 Project Overview

This project implements an advanced healthcare chatbot that provides:
- **Intelligent Symptom Analysis**: AI-powered preliminary assessment with confidence scoring
- **Comprehensive Health Education**: Detailed information about medical conditions and treatments
- **Smart Medication Guidance**: Context-aware medication information and safety warnings
- **Personalized Lifestyle Advice**: Tailored health and wellness recommendations
- **Emergency Detection**: Automatic recognition of urgent medical situations
- **Quality Assurance**: Built-in medical disclaimers and safety boundaries

**⚠️ Important Disclaimer**: This chatbot is for educational and research purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m src.main
```

The backend API will be available at `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend directory
cd front-end

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📁 Project Structure

```
Healthbot-AI/
├── front-end/                    # Next.js React application
│   ├── src/
│   │   ├── app/                 # App Router pages
│   │   ├── components/          # React components
│   │   └── lib/                 # API client and utilities
│   ├── public/                  # Static assets
│   └── package.json
├── backend/                     # FastAPI Python backend
│   ├── src/                     # Source code
│   │   ├── main.py             # FastAPI application
│   │   └── chatbot.py          # HealthBot AI logic
│   ├── data/                   # Training datasets
│   ├── notebooks/              # Jupyter notebooks
│   ├── models/                 # Saved model checkpoints
│   └── requirements.txt
├── front-end-legacy/           # Original HTML implementation
└── README.md
```

## 🤖 Advanced Technical Architecture

### Backend (FastAPI + Enhanced Transformers)
- **Model**: Microsoft DialoGPT with comprehensive fine-tuning pipeline
- **Framework**: FastAPI with automatic API documentation and performance monitoring
- **Advanced Features**: 
  - Hyperparameter optimization with multiple configurations
  - Comprehensive evaluation metrics (BLEU, ROUGE, perplexity)
  - Intent classification and domain relevance detection
  - Quality scoring and response analysis
  - Conversation management with persistent storage
- **Enhanced Endpoints**: Chat, symptom analysis, performance metrics, model evaluation

### Frontend (Next.js + Advanced React)
- **Framework**: Next.js 15 with App Router and React 19
- **Styling**: Tailwind CSS with custom animations
- **Advanced Components**: 
  - Enhanced Chat with quality metrics display
  - Performance monitoring dashboard
  - Export functionality for conversations
  - Real-time confidence scoring
- **Background**: Vanta.js 3D animated background with healthcare theme
- **Icons**: React Feather icons with custom healthcare symbols

### Advanced AI Model Pipeline
1. **Data Preprocessing**: Comprehensive healthcare conversation dataset with 15+ diverse scenarios
2. **Hyperparameter Tuning**: Automated optimization across multiple configurations
3. **Fine-tuning**: Domain-specific training with healthcare conversations
4. **Evaluation**: Multi-metric assessment with BLEU, ROUGE, and qualitative analysis
5. **Inference**: Real-time text generation with confidence scoring and safety checks
6. **Quality Assurance**: Medical disclaimers, emergency detection, and domain filtering

## 📊 Features

### Chat Interface
- Real-time messaging with typing indicators
- Conversation history persistence
- Error handling and fallback responses
- Confidence scoring for responses

### Healthcare Capabilities
- Symptom analysis and preliminary assessment
- Health education and condition explanations
- Medication information and guidance
- Lifestyle and wellness advice
- Emergency symptom recognition

### Technical Features
- Domain relevance detection
- Conversation management with unique IDs
- API error handling and retry logic
- Responsive design for all devices
- 3D animated background effects

## 🔧 API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

- `POST /chat` - Send messages to the chatbot
- `POST /analyze-symptoms` - Detailed symptom analysis
- `GET /conversation/{id}` - Retrieve conversation history
- `GET /model/info` - Get model information and capabilities
- `GET /health` - API health check

### Example API Usage

```python
import requests

# Send a chat message
response = requests.post("http://localhost:8000/chat", json={
    "message": "I have a headache that's been going on for 3 days",
    "conversation_id": None
})

data = response.json()
print(f"Response: {data['response']}")
print(f"Confidence: {data['confidence']}")
print(f"Domain Relevant: {data['domain_relevance']}")
```

## 🎯 Advanced Model Fine-tuning & Evaluation

### Training Pipeline

The project includes a comprehensive training pipeline with hyperparameter optimization:

```bash
# Run the complete training pipeline
cd backend
python train_model.py

# Run model evaluation
python evaluate_model.py
```

### Fine-tuning Process

1. **Data Preparation**: 
   - 15+ diverse healthcare conversation scenarios
   - Intent classification and domain labeling
   - Text preprocessing and normalization

2. **Hyperparameter Optimization**:
   - Learning rate tuning (1e-5 to 1e-4)
   - Batch size optimization (2, 4, 8)
   - Epoch selection with early stopping
   - Weight decay and warmup steps

3. **Model Training**:
   - DialoGPT fine-tuning on healthcare data
   - Domain-specific tokenization
   - Multi-configuration comparison
   - Best model selection based on validation metrics

4. **Comprehensive Evaluation**:
   - BLEU and ROUGE score calculation
   - Domain relevance testing
   - Intent classification accuracy
   - Qualitative response analysis
   - Safety and compliance assessment

### Training Scripts

- `train_model.py`: Complete training pipeline with hyperparameter tuning
- `evaluate_model.py`: Comprehensive model evaluation
- `training_pipeline.py`: Core training logic with metrics
- `notebooks/fine_tuning_pipeline.ipynb`: Interactive Jupyter notebook

## 📈 Advanced Performance Metrics

### Quantitative Metrics
- **BLEU Score**: Text generation quality assessment (target: >0.3)
- **ROUGE-L**: Longest common subsequence evaluation (target: >0.4)
- **Perplexity**: Model uncertainty measurement (lower is better)
- **Confidence Score**: Model certainty for each response (0.0 - 1.0)
- **Domain Relevance**: Healthcare topic detection accuracy (target: >90%)
- **Intent Classification**: Query categorization accuracy (target: >85%)

### Quality Assurance Metrics
- **Medical Disclaimer Rate**: Percentage of responses with appropriate disclaimers
- **Emergency Detection Rate**: Accuracy in identifying urgent medical situations
- **Response Relevance**: Alignment between user intent and bot response
- **Safety Compliance**: Adherence to medical safety guidelines

### Training Performance
- **Hyperparameter Optimization**: Automated tuning across 3+ configurations
- **Training Loss**: Model convergence tracking with early stopping
- **Validation Accuracy**: Performance on held-out test set
- **Fine-tuning Progress**: Domain adaptation effectiveness

## 🛡️ Safety & Compliance

### Medical Disclaimers
- All responses include medical advice disclaimers
- Clear indication that this is not professional medical advice
- Encouragement to consult healthcare providers

### Emergency Detection
- Recognizes urgent symptoms (chest pain, difficulty breathing)
- Provides appropriate emergency guidance
- Maintains safety boundaries

### Domain Filtering
- Filters non-healthcare queries
- Maintains focus on medical topics
- Provides helpful redirection for off-topic queries

## 🚀 Deployment

### Development
```bash
# Backend
cd backend && python -m src.main

# Frontend
cd front-end && npm run dev
```

### Production Considerations
- Use GPU instances for model inference
- Implement rate limiting and authentication
- Set up monitoring and logging
- Use HTTPS for secure communication
- Consider model optimization and caching

## 📚 Educational Value

This project demonstrates:

### Machine Learning
- Transformer model fine-tuning
- Domain-specific AI applications
- Natural language processing pipelines
- Model evaluation and metrics

### Full-Stack Development
- Modern React/Next.js frontend development
- FastAPI backend implementation
- RESTful API design
- Real-time communication

### Healthcare AI Ethics
- Responsible AI development
- Medical disclaimer implementation
- Safety boundary enforcement
- User education and transparency

## 🧪 Testing

### Backend Testing
```bash
cd backend
pytest tests/
```

### Frontend Testing
```bash
cd front-end
npm test
```

### Manual Testing
1. Start both backend and frontend
2. Open browser to `http://localhost:3000`
3. Test various healthcare queries
4. Verify error handling and fallbacks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes. See the LICENSE file for details.

## ⚠️ Important Notes

### Medical Disclaimer
This chatbot is designed for educational and informational purposes only. It is not intended to:
- Replace professional medical advice
- Provide medical diagnoses
- Recommend specific treatments
- Handle medical emergencies

Always consult qualified healthcare professionals for medical concerns.

### Ethical AI
This project follows responsible AI principles:
- Transparent about capabilities and limitations
- Includes appropriate disclaimers
- Respects user privacy and data
- Maintains safety boundaries

## 📞 Support

For questions or issues:
1. Check the documentation in `/backend/README.md`
2. Review the API documentation at `http://localhost:8000/docs`
3. Open an issue in the repository
4. Contact the development team

---

**Built with ❤️ for healthcare education and AI research**