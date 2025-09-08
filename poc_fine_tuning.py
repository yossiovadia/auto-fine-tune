#!/usr/bin/env python3
"""
Knowledge Acquisition POC: Fine-tuning Script

This script handles the fine-tuning process for the knowledge acquisition POC.
Designed to run on GPU systems for efficient training.
"""

import json
import torch
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from poc_knowledge_domains import get_all_knowledge_domains

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeFineTuner:
    """Fine-tuner for knowledge acquisition POC."""
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        output_dir: str = "poc_models/knowledge_acquisition",
        use_wandb: bool = False
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(
                project="knowledge-acquisition-poc",
                name=f"knowledge-ft-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": model_name,
                    "task": "knowledge_acquisition",
                    "domains": 3
                }
            )
    
    def create_dataset_from_domains(self) -> tuple[List[Dict], List[Dict]]:
        """Create training dataset from knowledge domains."""
        logger.info("Creating dataset from knowledge domains...")
        
        domains = get_all_knowledge_domains()
        all_examples = []
        
        for domain in domains:
            for fact in domain.facts:
                example = {
                    "instruction": fact.question,
                    "input": "",
                    "output": fact.answer,
                    "domain": domain.name,
                    "category": fact.category,
                    "difficulty": fact.difficulty
                }
                all_examples.append(example)
        
        # Shuffle and split
        import random
        random.shuffle(all_examples)
        
        split_idx = int(0.85 * len(all_examples))  # 85% train, 15% validation
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]
        
        logger.info(f"Created dataset: {len(train_examples)} train, {len(val_examples)} val examples")
        logger.info(f"Domains: {[domain.name for domain in domains]}")
        
        return train_examples, val_examples
    
    def format_examples(self, examples: List[Dict]) -> Dataset:
        """Format examples for training."""
        def format_instruction(example):
            instruction = example['instruction']
            input_text = example.get('input', '')
            output = example['output']
            
            if input_text:
                text = f"<|user|>\n{instruction}\n\nInput: {input_text}<|end|>\n<|assistant|>\n{output}<|end|>"
            else:
                text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
            
            return {"text": text}
        
        formatted = [format_instruction(ex) for ex in examples]
        return Dataset.from_list(formatted)
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Tokenize the dataset."""
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with LoRA."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # Fix for Phi-3 compatibility
        )
        
        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, lora_config)
        
        logger.info("Model setup complete with LoRA adapters")
        model.print_trainable_parameters()
        
        return model, tokenizer
    
    def run_training(
        self,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        save_steps: int = 100
    ) -> str:
        """Run the fine-tuning process."""
        logger.info("Starting knowledge acquisition fine-tuning...")
        
        # Create datasets
        train_examples, val_examples = self.create_dataset_from_domains()
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Format and tokenize datasets
        train_dataset = self.format_examples(train_examples)
        val_dataset = self.format_examples(val_examples)
        
        train_dataset = self.tokenize_dataset(train_dataset, tokenizer)
        val_dataset = self.tokenize_dataset(val_dataset, tokenizer)
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_steps=50,
            logging_steps=10,
            eval_steps=50,
            save_steps=save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.use_wandb else None,
            fp16=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            run_name=f"knowledge-acquisition-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        # Save training metrics
        metrics = {
            "training_loss": train_result.training_loss,
            "training_runtime": train_result.metrics["train_runtime"],
            "training_samples_per_second": train_result.metrics["train_samples_per_second"],
            "total_examples": len(train_examples),
            "validation_examples": len(val_examples),
            "domains_trained": 3,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get final evaluation
        final_eval = trainer.evaluate()
        metrics.update({
            "final_eval_loss": final_eval["eval_loss"],
            "final_eval_runtime": final_eval["eval_runtime"]
        })
        
        metrics_file = self.output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training completed!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        logger.info(f"Final evaluation loss: {final_eval['eval_loss']:.4f}")
        logger.info(f"Model saved to: {final_model_path}")
        
        return str(final_model_path)

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Acquisition Fine-tuning")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="Base model")
    parser.add_argument("--output-dir", default="poc_models/knowledge_acquisition", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU detected - training will be slow on CPU")
    
    # Initialize trainer
    trainer = KnowledgeFineTuner(
        model_name=args.model,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    # Run training
    model_path = trainer.run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps
    )
    
    logger.info("="*60)
    logger.info("🎉 FINE-TUNING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Trained model path: {model_path}")
    logger.info("Next steps:")
    logger.info("  1. Run baseline test: python poc_knowledge_acquisition.py --mode baseline")
    logger.info(f"  2. Run post-training test: python poc_knowledge_acquisition.py --mode post_training --model-path {model_path}")
    logger.info(f"  3. Run novel questions test: python poc_knowledge_acquisition.py --mode novel --model-path {model_path}")
    logger.info(f"  4. Generate full report: python poc_knowledge_acquisition.py --mode full --model-path {model_path}")

if __name__ == "__main__":
    main()