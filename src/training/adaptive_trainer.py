#!/usr/bin/env python3
"""
Adaptive fine-tuning trainer for vLLM issues.
Implements iterative training where new models train from previous adaptations.
"""

import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
from sklearn.metrics import accuracy_score
import wandb

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveTrainingConfig:
    """Configuration for adaptive training."""
    base_model_name: str = "microsoft/Phi-3-mini-4k-instruct"  # Smaller model for POC
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    output_dir: str = "models/adaptive_vllm"
    use_wandb: bool = False

class AdaptiveTrainer:
    """
    Implements adaptive fine-tuning for vLLM troubleshooting.
    
    Key concept: Each training iteration uses the previous iteration's model
    as the starting point, enabling progressive knowledge accumulation.
    """
    
    def __init__(self, config: AdaptiveTrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.iteration_history = []
        
        # Setup output directories
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        if config.use_wandb:
            wandb.init(project="adaptive-vllm-training", config=config.__dict__)
    
    def initialize_base_model(self) -> None:
        """Initialize the base model and tokenizer."""
        logger.info(f"Loading base model: {self.config.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded with {self.model.num_parameters()} parameters")
    
    def load_previous_iteration(self, iteration_path: str) -> None:
        """Load model from previous iteration."""
        logger.info(f"Loading previous iteration from: {iteration_path}")
        
        # Load the base model first
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load the LoRA adapters from previous iteration
        self.model = PeftModel.from_pretrained(self.model, iteration_path)
        
        # Merge and unload to get the full adapted model
        self.model = self.model.merge_and_unload()
        
        logger.info("Previous iteration loaded and merged successfully")
    
    def prepare_lora_model(self) -> None:
        """Add LoRA adapters to the current model."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"LoRA adapters added. Trainable parameters: {self.model.print_trainable_parameters()}")
    
    def preprocess_data(self, examples: List[Dict]) -> Dataset:
        """Convert training examples to tokenized dataset."""
        
        def format_instruction(example):
            """Format example into instruction-following format."""
            instruction = example['instruction']
            input_text = example.get('input', '')
            output = example['output']
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            return {
                'text': prompt + output + self.tokenizer.eos_token,
                'prompt': prompt,
                'response': output
            }
        
        # Format all examples
        formatted_examples = [format_instruction(ex) for ex in examples]
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        
        def tokenize_function(examples):
            # Tokenize the full text
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            # For training, we want to mask the prompt part and only compute loss on response
            prompt_tokenized = self.tokenizer(
                examples['prompt'],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            # Create labels (mask prompt tokens)
            labels = tokenized['input_ids'].copy()
            prompt_length = len(prompt_tokenized['input_ids'])
            
            # Set prompt tokens to -100 (ignored in loss computation)
            for i in range(min(prompt_length, len(labels))):
                labels[i] = -100
            
            tokenized['labels'] = labels
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=False,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train_iteration(
        self,
        train_examples: List[Dict],
        eval_examples: Optional[List[Dict]] = None,
        iteration_num: int = 0,
        previous_iteration_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train one iteration of the adaptive model.
        
        Args:
            train_examples: Training examples for this iteration
            eval_examples: Evaluation examples
            iteration_num: Current iteration number (0 = base model)
            previous_iteration_path: Path to previous iteration model
            
        Returns:
            Training metrics and model path
        """
        logger.info(f"Starting iteration {iteration_num} with {len(train_examples)} examples")
        
        # Initialize or load model
        if iteration_num == 0:
            self.initialize_base_model()
        else:
            if previous_iteration_path:
                self.load_previous_iteration(previous_iteration_path)
            else:
                raise ValueError("Previous iteration path required for iteration > 0")
        
        # Add LoRA adapters
        self.prepare_lora_model()
        
        # Prepare datasets
        train_dataset = self.preprocess_data(train_examples)
        eval_dataset = self.preprocess_data(eval_examples) if eval_examples else None
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Setup training arguments
        iteration_output_dir = self.output_path / f"iteration_{iteration_num}"
        iteration_output_dir.mkdir(exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(iteration_output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to="wandb" if self.config.use_wandb else None,
            run_name=f"vllm_adaptive_iter_{iteration_num}",
            fp16=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        trainer.save_state()
        
        # Save training metrics
        iteration_metrics = {
            "iteration": iteration_num,
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples) if eval_examples else 0,
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "timestamp": datetime.now().isoformat(),
            "model_path": str(iteration_output_dir)
        }
        
        if eval_dataset:
            eval_result = trainer.evaluate()
            iteration_metrics.update({
                "eval_loss": eval_result["eval_loss"],
                "eval_runtime": eval_result["eval_runtime"],
                "eval_samples_per_second": eval_result["eval_samples_per_second"]
            })
        
        # Save metrics
        metrics_file = iteration_output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(iteration_metrics, f, indent=2)
        
        self.iteration_history.append(iteration_metrics)
        
        logger.info(f"Iteration {iteration_num} completed!")
        logger.info(f"Train loss: {iteration_metrics['train_loss']:.4f}")
        if 'eval_loss' in iteration_metrics:
            logger.info(f"Eval loss: {iteration_metrics['eval_loss']:.4f}")
        
        return iteration_metrics
    
    def evaluate_model(self, test_examples: List[Dict], model_path: str) -> Dict[str, float]:
        """Evaluate model performance on test examples."""
        logger.info(f"Evaluating model at: {model_path}")
        
        # Load model for evaluation
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, model_path)
        
        # Evaluate on test examples
        correct = 0
        total = len(test_examples)
        
        for example in test_examples:
            instruction = example['instruction']
            input_text = example.get('input', '')
            expected_output = example['output']
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            # Simple string matching for evaluation (can be improved)
            if expected_output.lower().strip() in response.lower():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
        logger.info(f"Evaluation results: {correct}/{total} correct ({accuracy:.2%})")
        return metrics
    
    def run_adaptive_training(
        self,
        training_data_by_period: Dict[str, List[Dict]],
        test_examples: List[Dict]
    ) -> Dict[str, Any]:
        """
        Run complete adaptive training across multiple time periods.
        
        Args:
            training_data_by_period: Dict mapping period names to training examples
            test_examples: Examples for final evaluation
            
        Returns:
            Complete training results including all iterations
        """
        logger.info("Starting adaptive training pipeline...")
        
        all_results = {
            "iterations": [],
            "evaluation_results": {},
            "improvement_metrics": {}
        }
        
        previous_model_path = None
        
        # Train each iteration
        for i, (period_name, examples) in enumerate(training_data_by_period.items()):
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {period_name} (Iteration {i})")
            logger.info(f"{'='*50}")
            
            # Split examples for training and validation
            split_idx = int(0.9 * len(examples))
            train_examples = examples[:split_idx]
            eval_examples = examples[split_idx:] if len(examples) > split_idx else None
            
            # Train iteration
            iteration_result = self.train_iteration(
                train_examples=train_examples,
                eval_examples=eval_examples,
                iteration_num=i,
                previous_iteration_path=previous_model_path
            )
            
            all_results["iterations"].append(iteration_result)
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(test_examples, iteration_result["model_path"])
            all_results["evaluation_results"][f"iteration_{i}"] = test_metrics
            
            # Update for next iteration
            previous_model_path = iteration_result["model_path"]
        
        # Calculate improvement metrics
        if len(all_results["evaluation_results"]) > 1:
            base_accuracy = all_results["evaluation_results"]["iteration_0"]["accuracy"]
            final_accuracy = list(all_results["evaluation_results"].values())[-1]["accuracy"]
            
            all_results["improvement_metrics"] = {
                "base_accuracy": base_accuracy,
                "final_accuracy": final_accuracy,
                "absolute_improvement": final_accuracy - base_accuracy,
                "relative_improvement": (final_accuracy - base_accuracy) / base_accuracy if base_accuracy > 0 else 0
            }
        
        # Save complete results
        results_file = self.output_path / "adaptive_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info("\n" + "="*50)
        logger.info("ADAPTIVE TRAINING COMPLETE!")
        logger.info("="*50)
        
        if "improvement_metrics" in all_results:
            metrics = all_results["improvement_metrics"]
            logger.info(f"Base accuracy: {metrics['base_accuracy']:.2%}")
            logger.info(f"Final accuracy: {metrics['final_accuracy']:.2%}")
            logger.info(f"Absolute improvement: {metrics['absolute_improvement']:.2%}")
            logger.info(f"Relative improvement: {metrics['relative_improvement']:.2%}")
        
        return all_results


def main():
    """Example usage of the adaptive trainer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run adaptive training on vLLM issues")
    parser.add_argument("--data-dir", required=True, help="Directory with training data")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="Base model")
    parser.add_argument("--output-dir", default="models/adaptive_vllm", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per iteration")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create config
    config = AdaptiveTrainingConfig(
        base_model_name=args.model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        use_wandb=args.use_wandb
    )
    
    # Initialize trainer
    trainer = AdaptiveTrainer(config)
    
    # Load data (you would implement this based on your data format)
    logger.info("Loading training data...")
    # training_data_by_period = load_training_data(args.data_dir)
    # test_examples = load_test_data(args.data_dir)
    
    # Run adaptive training
    # results = trainer.run_adaptive_training(training_data_by_period, test_examples)
    
    logger.info("Training pipeline ready! Implement data loading to run.")


if __name__ == "__main__":
    main()