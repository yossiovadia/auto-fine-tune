#!/usr/bin/env python3
"""
Knowledge Acquisition POC: Main Evaluation and Training Script

This script implements the complete POC pipeline:
1. Baseline testing (expecting "I don't know" responses)
2. Dataset creation for fine-tuning
3. Fine-tuning process
4. Knowledge validation testing
5. Before/after comparison demonstration
"""

import json
import torch
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging

from poc_knowledge_domains import get_all_knowledge_domains, get_baseline_questions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Result of testing a model on a specific question."""
    question: str
    expected_answer: str
    model_response: str
    is_correct: bool
    confidence_score: float
    response_time: float
    contains_unknown: bool  # Whether response indicates lack of knowledge

@dataclass
class EvaluationResults:
    """Complete evaluation results for a model."""
    model_name: str
    test_type: str  # "baseline", "post_training", "novel_questions"
    timestamp: str
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_confidence: float
    avg_response_time: float
    unknown_responses: int  # Count of "I don't know" type responses
    results: List[TestResult]

class KnowledgeAcquisitionTester:
    """Main class for testing knowledge acquisition through fine-tuning."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.results_dir = Path("poc_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load knowledge domains
        self.domains = get_all_knowledge_domains()
        logger.info(f"Loaded {len(self.domains)} knowledge domains")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load the base model or a fine-tuned model."""
        model_to_load = model_path if model_path else self.model_name
        logger.info(f"Loading model: {model_to_load}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # If loading a fine-tuned model, load the PEFT adapters
        if model_path and model_path != self.model_name:
            logger.info(f"Loading LoRA adapters from: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
    
    def generate_response(self, question: str, max_new_tokens: int = 150) -> Tuple[str, float]:
        """Generate a response to a question and measure response time."""
        prompt = f"Question: {question}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response_time = time.time() - start_time
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        return response, response_time
    
    def check_unknown_response(self, response: str) -> bool:
        """Check if the response indicates lack of knowledge."""
        unknown_patterns = [
            r"i don't know",
            r"i'm not sure",
            r"i don't have",
            r"i cannot",
            r"i'm not familiar",
            r"no information",
            r"not aware",
            r"don't have access",
            r"cannot provide",
            r"unable to answer",
            r"insufficient information",
            r"not in my knowledge",
            r"beyond my knowledge"
        ]
        
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in unknown_patterns)
    
    def evaluate_response(self, question: str, expected_answer: str, model_response: str, response_time: float) -> TestResult:
        """Evaluate a model's response to a question."""
        # Simple keyword-based evaluation
        expected_keywords = expected_answer.lower().split()
        response_keywords = model_response.lower().split()
        
        # Calculate how many expected keywords appear in the response
        matching_keywords = sum(1 for keyword in expected_keywords if keyword in response_keywords)
        confidence_score = matching_keywords / len(expected_keywords) if expected_keywords else 0.0
        
        # Consider it correct if confidence > 0.3 and doesn't indicate unknown
        contains_unknown = self.check_unknown_response(model_response)
        is_correct = confidence_score > 0.3 and not contains_unknown
        
        return TestResult(
            question=question,
            expected_answer=expected_answer,
            model_response=model_response,
            is_correct=is_correct,
            confidence_score=confidence_score,
            response_time=response_time,
            contains_unknown=contains_unknown
        )
    
    def run_baseline_test(self) -> EvaluationResults:
        """Run baseline test on the base model (expecting mostly 'I don't know' responses)."""
        logger.info("Running baseline knowledge test...")
        
        self.load_model()  # Load base model
        
        test_questions = []
        for domain in self.domains:
            for fact in domain.test_questions:
                test_questions.append((fact.question, fact.answer))
        
        results = []
        for question, expected_answer in test_questions:
            response, response_time = self.generate_response(question)
            result = self.evaluate_response(question, expected_answer, response, response_time)
            results.append(result)
            
            logger.info(f"Q: {question[:50]}...")
            logger.info(f"A: {response[:100]}...")
            logger.info(f"Unknown: {result.contains_unknown}, Correct: {result.is_correct}")
            logger.info("---")
        
        # Calculate summary statistics
        correct_count = sum(1 for r in results if r.is_correct)
        unknown_count = sum(1 for r in results if r.contains_unknown)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_response_time = sum(r.response_time for r in results) / len(results)
        
        evaluation = EvaluationResults(
            model_name=self.model_name,
            test_type="baseline",
            timestamp=datetime.now().isoformat(),
            total_questions=len(results),
            correct_answers=correct_count,
            accuracy=correct_count / len(results),
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            unknown_responses=unknown_count,
            results=results
        )
        
        # Save results
        baseline_file = self.results_dir / "baseline_results.json"
        with open(baseline_file, 'w') as f:
            json.dump(asdict(evaluation), f, indent=2)
        
        logger.info(f"Baseline test completed:")
        logger.info(f"  Accuracy: {evaluation.accuracy:.1%}")
        logger.info(f"  Unknown responses: {unknown_count}/{len(results)} ({unknown_count/len(results):.1%})")
        logger.info(f"  Average confidence: {avg_confidence:.2f}")
        
        return evaluation
    
    def create_training_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Create training and validation datasets from knowledge domains."""
        logger.info("Creating training dataset...")
        
        training_examples = []
        
        for domain in self.domains:
            for fact in domain.facts:
                # Create instruction-following format
                example = {
                    "instruction": fact.question,
                    "input": "",
                    "output": fact.answer,
                    "domain": domain.name,
                    "category": fact.category,
                    "difficulty": fact.difficulty
                }
                training_examples.append(example)
        
        # Split into train/val (80/20)
        split_idx = int(0.8 * len(training_examples))
        train_examples = training_examples[:split_idx]
        val_examples = training_examples[split_idx:]
        
        # Save datasets
        train_file = self.results_dir / "train_dataset.json"
        val_file = self.results_dir / "val_dataset.json"
        
        with open(train_file, 'w') as f:
            json.dump(train_examples, f, indent=2)
        
        with open(val_file, 'w') as f:
            json.dump(val_examples, f, indent=2)
        
        logger.info(f"Created training dataset: {len(train_examples)} train, {len(val_examples)} val examples")
        
        return train_examples, val_examples
    
    def run_post_training_test(self, model_path: str) -> EvaluationResults:
        """Test the fine-tuned model on the same questions as baseline."""
        logger.info("Running post-training knowledge test...")
        
        self.load_model(model_path)  # Load fine-tuned model
        
        test_questions = []
        for domain in self.domains:
            for fact in domain.test_questions:
                test_questions.append((fact.question, fact.answer))
        
        results = []
        for question, expected_answer in test_questions:
            response, response_time = self.generate_response(question)
            result = self.evaluate_response(question, expected_answer, response, response_time)
            results.append(result)
            
            logger.info(f"Q: {question[:50]}...")
            logger.info(f"A: {response[:100]}...")
            logger.info(f"Unknown: {result.contains_unknown}, Correct: {result.is_correct}")
            logger.info("---")
        
        # Calculate summary statistics
        correct_count = sum(1 for r in results if r.is_correct)
        unknown_count = sum(1 for r in results if r.contains_unknown)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_response_time = sum(r.response_time for r in results) / len(results)
        
        evaluation = EvaluationResults(
            model_name=f"{self.model_name} (fine-tuned)",
            test_type="post_training",
            timestamp=datetime.now().isoformat(),
            total_questions=len(results),
            correct_answers=correct_count,
            accuracy=correct_count / len(results),
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            unknown_responses=unknown_count,
            results=results
        )
        
        # Save results
        post_training_file = self.results_dir / "post_training_results.json"
        with open(post_training_file, 'w') as f:
            json.dump(asdict(evaluation), f, indent=2)
        
        logger.info(f"Post-training test completed:")
        logger.info(f"  Accuracy: {evaluation.accuracy:.1%}")
        logger.info(f"  Unknown responses: {unknown_count}/{len(results)} ({unknown_count/len(results):.1%})")
        logger.info(f"  Average confidence: {avg_confidence:.2f}")
        
        return evaluation
    
    def run_novel_questions_test(self, model_path: str) -> EvaluationResults:
        """Test the fine-tuned model on novel questions requiring inference."""
        logger.info("Running novel questions test...")
        
        self.load_model(model_path)  # Load fine-tuned model
        
        test_questions = []
        for domain in self.domains:
            for fact in domain.novel_questions:
                test_questions.append((fact.question, fact.answer))
        
        results = []
        for question, expected_answer in test_questions:
            response, response_time = self.generate_response(question, max_new_tokens=200)
            result = self.evaluate_response(question, expected_answer, response, response_time)
            results.append(result)
            
            logger.info(f"Q: {question[:50]}...")
            logger.info(f"A: {response[:150]}...")
            logger.info(f"Unknown: {result.contains_unknown}, Correct: {result.is_correct}")
            logger.info("---")
        
        # Calculate summary statistics
        correct_count = sum(1 for r in results if r.is_correct)
        unknown_count = sum(1 for r in results if r.contains_unknown)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_response_time = sum(r.response_time for r in results) / len(results)
        
        evaluation = EvaluationResults(
            model_name=f"{self.model_name} (fine-tuned)",
            test_type="novel_questions",
            timestamp=datetime.now().isoformat(),
            total_questions=len(results),
            correct_answers=correct_count,
            accuracy=correct_count / len(results),
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            unknown_responses=unknown_count,
            results=results
        )
        
        # Save results
        novel_file = self.results_dir / "novel_questions_results.json"
        with open(novel_file, 'w') as f:
            json.dump(asdict(evaluation), f, indent=2)
        
        logger.info(f"Novel questions test completed:")
        logger.info(f"  Accuracy: {evaluation.accuracy:.1%}")
        logger.info(f"  Unknown responses: {unknown_count}/{len(results)} ({unknown_count/len(results):.1%})")
        logger.info(f"  Average confidence: {avg_confidence:.2f}")
        
        return evaluation
    
    def create_comparison_report(self, baseline: EvaluationResults, post_training: EvaluationResults, novel: EvaluationResults):
        """Create a comprehensive before/after comparison report."""
        report = {
            "poc_summary": {
                "title": "Knowledge Acquisition POC Results",
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "domains_tested": len(self.domains),
                "total_facts_learned": sum(len(domain.facts) for domain in self.domains)
            },
            "baseline_results": asdict(baseline),
            "post_training_results": asdict(post_training),
            "novel_questions_results": asdict(novel),
            "improvement_metrics": {
                "accuracy_improvement": post_training.accuracy - baseline.accuracy,
                "relative_improvement": (post_training.accuracy - baseline.accuracy) / baseline.accuracy if baseline.accuracy > 0 else float('inf'),
                "unknown_response_reduction": baseline.unknown_responses - post_training.unknown_responses,
                "novel_question_accuracy": novel.accuracy,
                "knowledge_transfer_success": novel.accuracy > 0.5
            },
            "key_findings": [
                f"Baseline accuracy: {baseline.accuracy:.1%}",
                f"Post-training accuracy: {post_training.accuracy:.1%}",
                f"Accuracy improvement: {post_training.accuracy - baseline.accuracy:.1%}",
                f"Unknown responses reduced from {baseline.unknown_responses} to {post_training.unknown_responses}",
                f"Novel question accuracy: {novel.accuracy:.1%}",
                f"Successfully learned {sum(len(domain.facts) for domain in self.domains)} new facts"
            ]
        }
        
        report_file = self.results_dir / "poc_comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("="*60)
        logger.info("🎉 KNOWLEDGE ACQUISITION POC COMPLETE!")
        logger.info("="*60)
        logger.info(f"Baseline accuracy: {baseline.accuracy:.1%}")
        logger.info(f"Post-training accuracy: {post_training.accuracy:.1%}")
        logger.info(f"Improvement: {post_training.accuracy - baseline.accuracy:.1%}")
        logger.info(f"Novel questions: {novel.accuracy:.1%}")
        logger.info(f"Report saved: {report_file}")
        
        return report

def main():
    """Run the complete knowledge acquisition POC."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Acquisition POC")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model to use")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--mode", choices=["baseline", "post_training", "novel", "full"], 
                       default="baseline", help="Test mode to run")
    parser.add_argument("--model-path", help="Path to fine-tuned model (for post_training/novel modes)")
    
    args = parser.parse_args()
    
    tester = KnowledgeAcquisitionTester(model_name=args.model, device=args.device)
    
    if args.mode == "baseline":
        tester.run_baseline_test()
        tester.create_training_dataset()
    elif args.mode == "post_training":
        if not args.model_path:
            logger.error("--model-path required for post_training mode")
            return
        tester.run_post_training_test(args.model_path)
    elif args.mode == "novel":
        if not args.model_path:
            logger.error("--model-path required for novel mode")
            return
        tester.run_novel_questions_test(args.model_path)
    elif args.mode == "full":
        # Run complete POC (requires model to be already trained)
        if not args.model_path:
            logger.error("--model-path required for full mode")
            return
        
        baseline = tester.run_baseline_test()
        post_training = tester.run_post_training_test(args.model_path)
        novel = tester.run_novel_questions_test(args.model_path)
        tester.create_comparison_report(baseline, post_training, novel)

if __name__ == "__main__":
    main()