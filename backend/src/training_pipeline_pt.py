"""PyTorch training pipeline for fine-tuning a causal LM (e.g., DialoGPT) on healthcare dialogs.

This is a lightweight Trainer-based script that:
- Loads conversational examples from a JSON file (expects list of convs with messages)
- Formats turns into a single text per example (User/Bot pairs)
- Tokenizes and prepares datasets
- Runs Hugging Face Trainer to fine-tune a causal LM

Usage (example):
    python3 -m src.training_pipeline_pt --model_name microsoft/DialoGPT-medium --data_path ./data/healthcare_conversations.json

Note: This will use PyTorch and may require GPU for reasonable speed.
"""

import argparse
import os
import json
import logging
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pt-train")


def load_conversations(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Expect data to be a list of conversation dicts with 'messages' key
    examples = []
    for conv in data:
        msgs = conv.get('messages', [])
        # pairwise user->assistant
        for i in range(0, len(msgs), 2):
            if i + 1 < len(msgs):
                user = msgs[i].get('content', '').strip()
                bot = msgs[i+1].get('content', '').strip()
                if user and bot:
                    text = f"User: {user}\nBot: {bot}"
                    examples.append({"text": text})
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def tokenize_examples(examples, tokenizer, max_length=512):
    return tokenizer(examples['text'], truncation=True, max_length=max_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=os.getenv('MODEL_NAME', 'microsoft/DialoGPT-medium'))
    parser.add_argument('--data_path', default='./data/healthcare_conversations.json')
    parser.add_argument('--output_dir', default='./models/final_healthcare_chatbot_pt')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=256)
    args = parser.parse_args()

    logger.info(f"Training config: {args}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    examples = load_conversations(args.data_path)
    if len(examples) == 0:
        logger.error("No training examples found; aborting.")
        return

    ds = Dataset.from_list(examples)

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Tokenize
    tokenized = ds.map(lambda ex: tokenize_examples(ex, tokenizer, max_length=args.max_length), batched=True)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        evaluation_strategy='no',
        save_strategy='epoch',
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving trained model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training complete")


if __name__ == '__main__':
    main()
