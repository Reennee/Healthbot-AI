# HealthBot AI Backend

A healthcare domain-specific chatbot built with FastAPI and transformer models from Hugging Face.

## Features

- **Healthcare Domain Focus**: Specialized for medical conversations and symptom analysis
- **Transformer Models**: Uses pre-trained models like DialoGPT, fine-tuned for healthcare
- **FastAPI Backend**: RESTful API with automatic documentation
- **Conversation Management**: Persistent conversation history
- **Domain Relevance Detection**: Filters non-healthcare queries
- **Confidence Scoring**: Provides confidence levels for responses

## Architecture

```
backend/
├── src/
│   ├── main.py              # FastAPI application
│   ├── chatbot.py           # HealthBot class with model logic
│   └── __init__.py
├── data/
│   └── healthcare_conversations.json  # Training dataset
├── notebooks/
│   └── fine_tuning_pipeline.ipynb     # Model fine-tuning
├── models/                            # Saved model checkpoints
├── requirements.txt                   # Python dependencies
└── README.md
```

## Installation

1. **Create Virtual Environment**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download Model** (optional - will download automatically):
```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium'); AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')"
```

## Usage

### Start the API Server

```bash
python -m src.main
```

The API will be available at `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

- `POST /chat` - Send a message to the chatbot
- `POST /analyze-symptoms` - Analyze health symptoms
- `GET /conversation/{id}` - Get conversation history
- `GET /model/info` - Get model information
- `GET /health` - Health check endpoint

### Example Usage

```python
import requests

# Send a chat message
response = requests.post("http://localhost:8000/chat", json={
    "message": "I have a headache. What should I do?",
    "conversation_id": None
})

print(response.json())
# {
#   "response": "Headaches can have various causes...",
#   "conversation_id": "uuid-here",
#   "confidence": 0.85,
#   "domain_relevance": true
# }
```

## Model Fine-tuning

The notebook `notebooks/fine_tuning_pipeline.ipynb` demonstrates how to fine-tune a transformer model on healthcare conversations.

### Training Process

1. **Load Dataset**: Healthcare conversation pairs
2. **Preprocess**: Format for causal language modeling
3. **Tokenize**: Convert text to model inputs
4. **Train**: Fine-tune with healthcare-specific data
5. **Evaluate**: Test model performance
6. **Save**: Export fine-tuned model

### Running Fine-tuning

```bash
jupyter notebook notebooks/fine_tuning_pipeline.ipynb
```

## Configuration

### Environment Variables

Create a `.env` file:

```env
MODEL_NAME=microsoft/DialoGPT-medium
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

### Model Selection

The chatbot supports various transformer models:

- `microsoft/DialoGPT-small` - Fast, smaller model
- `microsoft/DialoGPT-medium` - Balanced performance
- `microsoft/DialoGPT-large` - Best quality, slower
- `facebook/blenderbot-400M-distill` - Alternative conversational model

## Performance Metrics

The system tracks several metrics:

- **BLEU Score**: Text generation quality
- **Confidence Score**: Model certainty (0-1)
- **Domain Relevance**: Healthcare topic detection
- **Response Time**: API latency
- **Training Loss**: Model convergence

## Healthcare Domain Features

### Symptom Analysis
- Identifies common symptoms
- Provides preliminary assessments
- Suggests urgency levels
- Recommends medical consultation

### Health Education
- Explains medical conditions
- Provides lifestyle advice
- Discusses treatment options
- Shares prevention strategies

### Safety Features
- Disclaimers for medical advice
- Emergency symptom recognition
- Referral to healthcare providers
- Domain boundary enforcement

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000
CMD ["python", "-m", "src.main"]
```

### Production Considerations

- Use GPU instances for model inference
- Implement rate limiting
- Add authentication/authorization
- Monitor API performance
- Set up logging and monitoring
- Use HTTPS for security

## Integration with Frontend

The backend is designed to work with the Next.js frontend:

```typescript
// Frontend API client
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: userInput })
});
```

## Development

### Code Structure

- `main.py`: FastAPI application setup and endpoints
- `chatbot.py`: Core HealthBot class with model logic
- `data/`: Training datasets and conversation examples
- `notebooks/`: Jupyter notebooks for experimentation

### Adding New Features

1. **New Endpoints**: Add to `main.py`
2. **Model Improvements**: Modify `chatbot.py`
3. **Training Data**: Update `data/healthcare_conversations.json`
4. **Fine-tuning**: Use notebooks for experimentation

### Testing

```bash
# Run API tests
pytest tests/

# Test specific endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a fever"}'
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Check internet connection for model download
2. **Memory Issues**: Use smaller models or reduce batch size
3. **API Connection**: Verify CORS settings for frontend
4. **Slow Responses**: Consider model optimization or caching

### Logs

Check application logs for debugging:

```bash
# Enable debug logging
export DEBUG=True
python -m src.main
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is for educational purposes. Healthcare advice should always be verified with qualified medical professionals.

## Contact

For questions about the healthcare chatbot implementation, please refer to the project documentation or create an issue in the repository.
