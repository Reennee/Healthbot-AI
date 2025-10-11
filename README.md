# HealthBot AI - Healthcare Domain Chatbot

A comprehensive healthcare domain-specific chatbot built with Next.js frontend and FastAPI backend, powered by transformer models from Hugging Face.

## 🏥 Project Overview

This project implements a healthcare chatbot that provides:
- **Symptom Analysis**: Preliminary assessment of health symptoms
- **Health Education**: Information about medical conditions and treatments
- **Medication Guidance**: General information about medications
- **Lifestyle Advice**: Health and wellness recommendations

**⚠️ Important Disclaimer**: This chatbot is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.

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

## 🤖 Technical Architecture

### Backend (FastAPI + Transformers)
- **Model**: Microsoft DialoGPT (fine-tuned for healthcare)
- **Framework**: FastAPI with automatic API documentation
- **Features**: Conversation management, domain relevance detection, confidence scoring
- **Endpoints**: Chat, symptom analysis, conversation history

### Frontend (Next.js + React)
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS
- **Components**: Modular React components (Header, Chat, Features, Footer)
- **Background**: Vanta.js 3D animated background
- **Icons**: React Feather icons

### AI Model Pipeline
1. **Pre-training**: Uses pre-trained DialoGPT model
2. **Fine-tuning**: Custom healthcare conversation dataset
3. **Inference**: Real-time text generation with domain filtering
4. **Safety**: Medical disclaimers and emergency detection

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

## 🎯 Model Fine-tuning

The project includes a Jupyter notebook for fine-tuning transformer models on healthcare data:

```bash
# Start Jupyter
jupyter notebook backend/notebooks/fine_tuning_pipeline.ipynb
```

### Fine-tuning Process
1. **Data Preparation**: Healthcare conversation pairs
2. **Model Selection**: DialoGPT for conversational AI
3. **Training**: Fine-tune on healthcare domain data
4. **Evaluation**: BLEU score and qualitative assessment
5. **Deployment**: Save and load fine-tuned model

## 📈 Performance Metrics

- **BLEU Score**: Text generation quality assessment
- **Confidence Score**: Model certainty (0.0 - 1.0)
- **Domain Relevance**: Healthcare topic detection accuracy
- **Response Time**: API latency monitoring
- **Training Loss**: Model convergence tracking

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