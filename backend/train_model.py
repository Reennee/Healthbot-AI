#!/usr/bin/env python3
"""
Healthcare Chatbot Training Script
Demonstrates the complete fine-tuning pipeline with hyperparameter optimization
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training_pipeline import HealthcareChatbotTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training execution"""
    logger.info("Starting Healthcare Chatbot Training Pipeline")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        "model_name": "microsoft/DialoGPT-medium",
        "data_path": "./data/healthcare_conversations.json",
        "output_dir": "./models",
        "logs_dir": "./logs",
        "test_size": 0.2,
        "use_wandb": os.getenv("WANDB_API_KEY") is not None
    }
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Initialize trainer
        logger.info("Initializing Healthcare Chatbot Trainer...")
        trainer = HealthcareChatbotTrainer(config["model_name"])
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = trainer.load_and_preprocess_data(config["data_path"])
        trainer.training_data = data
        
        logger.info(f"Loaded {len(data)} training examples")
        
        # Setup model and tokenizer
        logger.info("Setting up model and tokenizer...")
        trainer.setup_model_and_tokenizer()
        
        # Prepare datasets
        logger.info("Preparing training and validation datasets...")
        train_dataset, val_dataset = trainer.prepare_datasets(data, test_size=config["test_size"])
        
        # Hyperparameter tuning
        logger.info("Starting hyperparameter tuning...")
        logger.info("This may take several minutes depending on your hardware...")
        
        tuning_results = trainer.hyperparameter_tuning(train_dataset, val_dataset)
        
        logger.info("Hyperparameter tuning completed!")
        logger.info(f"Best configuration: {tuning_results['best_config']}")
        logger.info(f"Best performance: {tuning_results['best_performance']:.4f}")
        
        # Train final model
        logger.info("Training final model with best configuration...")
        final_trainer, final_metrics = trainer.train_final_model(
            tuning_results["best_config"], 
            train_dataset, 
            val_dataset
        )
        
        # Save training report
        report_path = f"{config['output_dir']}/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        trainer.save_training_report(tuning_results, report_path)
        
        logger.info(f"Training report saved to: {report_path}")
        
        # Test the model with sample inputs
        logger.info("Testing trained model...")
        test_inputs = [
            "I have a severe headache that won't go away",
            "What should I do if I have a high fever?",
            "I'm feeling very anxious and can't sleep",
            "I have chest pain and difficulty breathing",
            "What are the symptoms of diabetes?"
        ]
        
        logger.info("Sample responses from trained model:")
        logger.info("-" * 50)
        
        for i, test_input in enumerate(test_inputs, 1):
            try:
                response = trainer.generate_response(test_input)
                logger.info(f"Test {i}:")
                logger.info(f"Input: {test_input}")
                logger.info(f"Response: {response}")
                logger.info("-" * 30)
            except Exception as e:
                logger.error(f"Error generating response for test {i}: {e}")
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        final_evaluation = trainer.evaluate_model_performance(data[:10])  # Test on first 10 examples
        
        logger.info("Final Evaluation Results:")
        logger.info(f"BLEU Score: {final_evaluation.get('bleu_score', 0):.4f}")
        logger.info(f"ROUGE-L: {final_evaluation.get('rouge_l', 0):.4f}")
        logger.info(f"Quality Metrics: {final_evaluation.get('quality_metrics', {})}")
        
        # Performance summary
        performance_summary = trainer.get_performance_summary()
        logger.info("Performance Summary:")
        logger.info(json.dumps(performance_summary, indent=2))
        
        logger.info("=" * 60)
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model saved to: {config['output_dir']}/final_healthcare_chatbot")
        logger.info(f"Training report: {report_path}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        logger.error("Check the logs for more details")
        raise

if __name__ == "__main__":
    main()
