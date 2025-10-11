from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
from .chatbot import HealthBot

load_dotenv()

app = FastAPI(
    title="HealthBot AI API",
    description="Healthcare domain-specific chatbot using transformer models",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chatbot
healthbot = HealthBot()


class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    confidence: float
    domain_relevance: bool


class HealthQuery(BaseModel):
    query: str
    context: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "HealthBot AI API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": healthbot.is_ready()}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """
    Main chat endpoint for health-related conversations
    """
    try:
        response = await healthbot.generate_response(
            message=chat_message.message,
            conversation_id=chat_message.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/analyze-symptoms")
async def analyze_symptoms(query: HealthQuery):
    """
    Analyze symptoms and provide preliminary assessment
    """
    try:
        analysis = await healthbot.analyze_symptoms(
            query=query.query,
            context=query.context
        )
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing symptoms: {str(e)}")


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Retrieve conversation history
    """
    try:
        conversation = healthbot.get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "messages": conversation}
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Conversation not found: {str(e)}")


@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete conversation history
    """
    try:
        healthbot.delete_conversation(conversation_id)
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Error deleting conversation: {str(e)}")


@app.get("/model/info")
async def model_info():
    """
    Get information about the loaded model
    """
    return {
        "model_name": healthbot.model_name,
        "model_type": healthbot.model_type,
        "is_fine_tuned": healthbot.is_fine_tuned,
        "domain": "Healthcare",
        "capabilities": [
            "Symptom analysis",
            "Health education",
            "Medication information",
            "General health advice"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
