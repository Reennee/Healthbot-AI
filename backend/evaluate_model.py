#!/usr/bin/env python3
"""
Healthcare Chatbot Model Evaluation Script
Comprehensive evaluation with multiple metrics and qualitative analysis
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot import HealthBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data(data_path: str):
    """Load test data for evaluation"""
    with open(data_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    test_data = []
    for conv in conversations:
        messages = conv['messages']
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                test_data.append({
                    'user_input': messages[i]['content'],
                    'assistant_output': messages[i + 1]['content'],
                    'intent': messages[i].get('intent', 'unknown')
                })
    
    return test_data

def qualitative_evaluation(healthbot: HealthBot, test_cases: list):
    """Perform qualitative evaluation with sample test cases"""
    logger.info("Performing qualitative evaluation...")
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"Test Case {i+1}: {test_case['description']}")
        
        try:
            # Generate response
            response = healthbot.generate_response(test_case['input'])
            
            # Calculate quality metrics
            quality = healthbot.calculate_response_quality(test_case['input'], response)
            
            result = {
                'test_case': test_case['description'],
                'input': test_case['input'],
                'expected_intent': test_case.get('expected_intent', 'unknown'),
                'generated_response': response,
                'quality_metrics': quality,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            
            logger.info(f"Generated Response: {response}")
            logger.info(f"Quality Metrics: {quality}")
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"Error in test case {i+1}: {e}")
            results.append({
                'test_case': test_case['description'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    return results

def domain_relevance_test(healthbot: HealthBot):
    """Test domain relevance detection"""
    logger.info("Testing domain relevance detection...")
    
    test_queries = [
        # Healthcare queries (should be relevant)
        ("I have a headache", True),
        ("What are the symptoms of diabetes?", True),
        ("I need medication advice", True),
        ("How to lower blood pressure?", True),
        ("I have chest pain", True),
        
        # Non-healthcare queries (should be irrelevant)
        ("What's the weather like?", False),
        ("How to cook pasta?", False),
        ("What's the capital of France?", False),
        ("Tell me a joke", False),
        ("What time is it?", False)
    ]
    
    relevance_results = []
    
    for query, expected_relevant in test_queries:
        is_relevant = healthbot._check_domain_relevance(query)
        correct = (is_relevant == expected_relevant)
        
        result = {
            'query': query,
            'expected_relevant': expected_relevant,
            'detected_relevant': is_relevant,
            'correct': correct
        }
        
        relevance_results.append(result)
        
        logger.info(f"Query: '{query}'")
        logger.info(f"Expected: {expected_relevant}, Detected: {is_relevant}, Correct: {correct}")
    
    # Calculate accuracy
    accuracy = sum(1 for r in relevance_results if r['correct']) / len(relevance_results)
    logger.info(f"Domain relevance accuracy: {accuracy:.2%}")
    
    return relevance_results, accuracy

def intent_classification_test(healthbot: HealthBot):
    """Test intent classification"""
    logger.info("Testing intent classification...")
    
    test_cases = [
        ("I have a severe headache", "symptom_inquiry"),
        ("What medicine should I take?", "medication"),
        ("I can't breathe properly", "emergency"),
        ("How to exercise more?", "lifestyle"),
        ("How to prevent heart disease?", "prevention"),
        ("I feel anxious", "symptom_inquiry"),
        ("What's the dosage for aspirin?", "medication"),
        ("I have chest pain", "emergency")
    ]
    
    intent_results = []
    
    for query, expected_intent in test_cases:
        detected_intent = healthbot.classify_intent(query)
        correct = (detected_intent == expected_intent)
        
        result = {
            'query': query,
            'expected_intent': expected_intent,
            'detected_intent': detected_intent,
            'correct': correct
        }
        
        intent_results.append(result)
        
        logger.info(f"Query: '{query}'")
        logger.info(f"Expected: {expected_intent}, Detected: {detected_intent}, Correct: {correct}")
    
    # Calculate accuracy
    accuracy = sum(1 for r in intent_results if r['correct']) / len(intent_results)
    logger.info(f"Intent classification accuracy: {accuracy:.2%}")
    
    return intent_results, accuracy

def generate_evaluation_report(evaluation_results: dict, output_path: str):
    """Generate comprehensive evaluation report"""
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_performance': evaluation_results.get('model_metrics', {}),
        'domain_relevance': {
            'accuracy': evaluation_results.get('domain_accuracy', 0),
            'results': evaluation_results.get('relevance_results', [])
        },
        'intent_classification': {
            'accuracy': evaluation_results.get('intent_accuracy', 0),
            'results': evaluation_results.get('intent_results', [])
        },
        'qualitative_evaluation': evaluation_results.get('qualitative_results', []),
        'summary': {
            'total_tests': len(evaluation_results.get('qualitative_results', [])),
            'domain_relevance_accuracy': evaluation_results.get('domain_accuracy', 0),
            'intent_accuracy': evaluation_results.get('intent_accuracy', 0),
            'bleu_score': evaluation_results.get('model_metrics', {}).get('bleu_score', 0),
            'rouge_l': evaluation_results.get('model_metrics', {}).get('rouge_l', 0)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to: {output_path}")
    return report

def main():
    """Main evaluation execution"""
    logger.info("Starting Healthcare Chatbot Model Evaluation")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        "model_name": "microsoft/DialoGPT-medium",
        "data_path": "./data/healthcare_conversations.json",
        "output_dir": "./evaluation_results"
    }
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    try:
        # Initialize chatbot
        logger.info("Initializing Healthcare Chatbot...")
        healthbot = HealthBot(config["model_name"])
        
        # Load test data
        logger.info("Loading test data...")
        test_data = load_test_data(config["data_path"])
        logger.info(f"Loaded {len(test_data)} test examples")
        
        # Model performance evaluation
        logger.info("Evaluating model performance...")
        model_metrics = healthbot.evaluate_model_performance(test_data[:20])  # Use first 20 examples
        
        # Domain relevance test
        relevance_results, domain_accuracy = domain_relevance_test(healthbot)
        
        # Intent classification test
        intent_results, intent_accuracy = intent_classification_test(healthbot)
        
        # Qualitative evaluation
        qualitative_test_cases = [
            {
                'description': 'Symptom Inquiry',
                'input': 'I have a persistent headache for 3 days',
                'expected_intent': 'symptom_inquiry'
            },
            {
                'description': 'Emergency Situation',
                'input': 'I have severe chest pain and can\'t breathe',
                'expected_intent': 'emergency'
            },
            {
                'description': 'Medication Question',
                'input': 'What are the side effects of ibuprofen?',
                'expected_intent': 'medication'
            },
            {
                'description': 'Lifestyle Advice',
                'input': 'How can I improve my sleep quality?',
                'expected_intent': 'lifestyle'
            },
            {
                'description': 'Prevention Question',
                'input': 'How can I prevent heart disease?',
                'expected_intent': 'prevention'
            }
        ]
        
        qualitative_results = qualitative_evaluation(healthbot, qualitative_test_cases)
        
        # Compile evaluation results
        evaluation_results = {
            'model_metrics': model_metrics,
            'domain_accuracy': domain_accuracy,
            'relevance_results': relevance_results,
            'intent_accuracy': intent_accuracy,
            'intent_results': intent_results,
            'qualitative_results': qualitative_results
        }
        
        # Generate report
        report_path = f"{config['output_dir']}/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = generate_evaluation_report(evaluation_results, report_path)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"BLEU Score: {report['summary']['bleu_score']:.4f}")
        logger.info(f"ROUGE-L: {report['summary']['rouge_l']:.4f}")
        logger.info(f"Domain Relevance Accuracy: {report['summary']['domain_relevance_accuracy']:.2%}")
        logger.info(f"Intent Classification Accuracy: {report['summary']['intent_accuracy']:.2%}")
        logger.info(f"Total Tests: {report['summary']['total_tests']}")
        logger.info("=" * 60)
        
        # Save detailed results
        detailed_results_path = f"{config['output_dir']}/detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(detailed_results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {detailed_results_path}")
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
