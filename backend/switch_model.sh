#!/bin/bash

# Quick script to switch HealthBot AI models
# Usage: ./switch_model.sh [model_name]

MODEL_NAME=${1:-"microsoft/biogpt"}

echo "🔄 Switching HealthBot AI to model: $MODEL_NAME"
echo ""

# Create or update .env file
if [ -f .env ]; then
    # Update existing MODEL_NAME if present
    if grep -q "MODEL_NAME=" .env; then
        sed -i.bak "s|MODEL_NAME=.*|MODEL_NAME=$MODEL_NAME|" .env
        echo "✅ Updated .env file"
    else
        echo "MODEL_NAME=$MODEL_NAME" >> .env
        echo "✅ Added MODEL_NAME to .env file"
    fi
else
    echo "MODEL_NAME=$MODEL_NAME" > .env
    echo "✅ Created .env file with MODEL_NAME=$MODEL_NAME"
fi

echo ""
echo "📋 Current configuration:"
cat .env | grep MODEL_NAME
echo ""
echo "🚀 Restart your backend server to apply changes:"
echo "   python -m uvicorn src.main:app --reload"
echo ""
echo "💡 Popular models:"
echo "   - microsoft/biogpt (healthcare-focused, recommended)"
echo "   - microsoft/DialoGPT-large (better quality)"
echo "   - gpt2-large (general purpose)"
echo "   - microsoft/phi-2 (high quality, needs 6GB RAM)"

