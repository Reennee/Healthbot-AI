import os
import json
import re
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    TFAutoModelForCausalLM,
    GenerationConfig,
)
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TFTrainingConfig:
    model_name: str = "gpt2"
    max_length: int = 512
    train_batch_size: int = 2
    eval_batch_size: int = 2
    learning_rate: float = 5e-5
    num_epochs: int = 2
    weight_decay: float = 0.0
    warmup_steps: int = 0
    output_dir: str = "./models_tf/final_healthcare_chatbot_tf"
    seed: int = 42


class TFHealthcareChatbotTrainer:
    def __init__(self, config: TFTrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.training_data: List[Dict] = []
        # Ensure determinism where possible
        tf.random.set_seed(config.seed)
        np.random.seed(config.seed)

    # -----------------------------
    # Data loading and preprocessing
    # -----------------------------
    def load_and_preprocess_data(self, data_path: str) -> List[Dict]:
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        processed: List[Dict] = []
        for conv in conversations:
            messages = conv.get('messages', [])
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    user_msg = self._clean_text(messages[i].get('content', ''))
                    assistant_msg = self._clean_text(messages[i + 1].get('content', ''))

                    formatted = f"<|startoftext|>Patient: {user_msg}<|endoftext|>Doctor: {assistant_msg}<|endoftext|>"
                    processed.append({
                        "text": formatted,
                        "user_input": user_msg,
                        "assistant_output": assistant_msg
                    })
        logger.info(f"Processed {len(processed)} examples")
        return processed

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = ' '.join(text.split())
        text = re.sub(r'[^\w\s\.\?\!\,\:\;]', '', text)
        return text

    # -----------------------------
    # Tokenizer and model setup
    # -----------------------------
    def setup_model_and_tokenizer(self):
        cfg = self.config
        logger.info(f"Loading tokenizer and TF model: {cfg.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        special_tokens = {
            "additional_special_tokens": [
                "<|startoftext|>", "<|endoftext|>", "Patient:", "Doctor:"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = TFAutoModelForCausalLM.from_pretrained(cfg.model_name)
        # Resize to include added tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info("Tokenizer and model ready")

    # -----------------------------
    # Dataset preparation
    # -----------------------------
    def _encode_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='np'
        )
        # labels are input_ids for causal LM
        labels = enc["input_ids"].copy()
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    def build_datasets(self, data: List[Dict]) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        texts = [ex["text"] for ex in data]
        train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=self.config.seed)

        def to_ds(text_list: List[str], batch_size: int) -> tf.data.Dataset:
            enc = self._encode_batch(text_list)
            ds = tf.data.Dataset.from_tensor_slices((
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                },
                enc["labels"],
            ))
            return ds.shuffle(1024, seed=self.config.seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_ds = to_ds(train_texts, self.config.train_batch_size)
        val_ds = to_ds(val_texts, self.config.eval_batch_size)
        logger.info(f"Datasets built. Train batches: {len(list(train_ds))}, Val batches: {len(list(val_ds))}")
        return train_ds, val_ds

    # -----------------------------
    # Training (custom loop using model's built-in loss with labels)
    # -----------------------------
    def train(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> Dict:
        cfg = self.config
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

        # Metrics
        train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

        @tf.function
        def train_step(features, labels):
            with tf.GradientTape() as tape:
                outputs = self.model(
                    input_ids=features["input_ids"],
                    attention_mask=features["attention_mask"],
                    labels=labels,
                    training=True,
                )
                loss = outputs.loss  # already averaged over batch
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        @tf.function
        def val_step(features, labels):
            outputs = self.model(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
                labels=labels,
                training=False,
            )
            return outputs.loss

        history = {"train_loss": [], "val_loss": [], "perplexity": []}
        for epoch in range(cfg.num_epochs):
            logger.info(f"Epoch {epoch+1}/{cfg.num_epochs}")
            train_loss_metric.reset_state()
            val_loss_metric.reset_state()

            # Train loop
            for features, labels in train_ds:
                loss = train_step(features, labels)
                train_loss_metric.update_state(loss)

            # Validation loop
            for vfeatures, vlabels in val_ds:
                vloss = val_step(vfeatures, vlabels)
                val_loss_metric.update_state(vloss)

            train_loss = float(train_loss_metric.result().numpy())
            val_loss = float(val_loss_metric.result().numpy())
            ppl = float(math.exp(val_loss)) if val_loss < 20 else float("inf")

            logger.info(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} perplexity={ppl:.2f}")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["perplexity"].append(ppl)

        return history

    # -----------------------------
    # Save and generate
    # -----------------------------
    def save(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.model.save_pretrained(self.config.output_dir, saved_model=True)
        self.tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"Model and tokenizer saved to {self.config.output_dir}")

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        input_text = f"<|startoftext|>Patient: {prompt}<|endoftext|>Doctor:"
        inputs = self.tokenizer(input_text, return_tensors="tf")
        
        # Create generation config (recommended approach)
        generation_config = GenerationConfig(
            max_length=min(self.config.max_length, inputs["input_ids"].shape[1] + max_new_tokens),
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        output_ids = self.model.generate(
            **inputs,
            generation_config=generation_config
        )
        decoded = self.tokenizer.decode(output_ids[0].numpy().tolist(), skip_special_tokens=True)
        if "Doctor:" in decoded:
            return decoded.split("Doctor:")[-1].strip()
        return decoded


def main():
    cfg = TFTrainingConfig()
    trainer = TFHealthcareChatbotTrainer(cfg)

    data = trainer.load_and_preprocess_data("./data/healthcare_conversations.json")
    trainer.training_data = data

    trainer.setup_model_and_tokenizer()
    train_ds, val_ds = trainer.build_datasets(data)

    history = trainer.train(train_ds, val_ds)

    # Save artifacts
    trainer.save()

    # Save simple training report
    report = {
        "model_name": cfg.model_name,
        "epochs": cfg.num_epochs,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "perplexity": history["perplexity"],
    }
    os.makedirs("./models_tf", exist_ok=True)
    with open("./models_tf/training_report_tf.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("TensorFlow training report saved to ./models_tf/training_report_tf.json")

    # Quick test generations
    samples = [
        "I have a headache that won't go away",
        "What should I do if I have a fever?",
        "I'm feeling anxious and can't sleep"
    ]
    for s in samples:
        try:
            resp = trainer.generate(s)
            logger.info(f"Input: {s}\nResponse: {resp}\n" + "-" * 40)
        except Exception as e:
            logger.error(f"Generation error: {e}")


if __name__ == "__main__":
    main()


