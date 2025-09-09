#!/usr/bin/env python3
"""
Ultra-Stable Full Training Demo - Preventing NaN Corruption

This demo uses ultra-conservative parameters and extensive monitoring to prevent
NaN corruption while still achieving knowledge acquisition through full parameter training.

Key features:
- Ultra-low learning rate (1e-7)
- Extensive gradient and loss monitoring
- Automatic NaN detection and stopping
- Weight validation at each step
- Conservative batch sizes and short sequences
- FP32 precision throughout

Usage:
    python ultra_stable_training_demo.py [--epochs EPOCHS]
"""

import torch
import time
import argparse
import os
import math
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"

# Import software domains
import sys
sys.path.append(str(Path(__file__).parent))
from domains.software_defect_domains import get_all_software_domains

class UltraStableTrainingDemo:
    """Ultra-stable demonstration of full parameter fine-tuning preventing NaN corruption."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.base_model = None
        self.trained_model = None
        self.domains = get_all_software_domains()
        
        # Test questions for before/after comparison
        self.test_questions = [
            "How to fix AuthFlow error AF-3001?",
            "What causes PayFlow error PF-1205?", 
            "How to resolve DataFlow DF-7890?",
            "What is AuthFlow error AF-6001?",
            "How to fix PayFlow PF-4001?",
        ]
    
    def setup_model(self):
        """Setup tokenizer and base model with ultra-stable settings."""
        print(f"🤖 Loading TinyLlama model for ULTRA-STABLE full training...")
        print(f"   Model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="right"
        )
        
        # Ensure proper tokenizer setup
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float32,  # Force FP32 for stability
            device_map="auto", 
            trust_remote_code=True
        )
        
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total")
        print(f"🔥 ALL {trainable_params:,} parameters will be trained (ULTRA-STABLE full fine-tuning)")
        print("✅ Model loaded with FP32 precision!")
    
    def check_model_weights(self, model, stage=""):
        """Check model weights for NaN/Inf values."""
        nan_count = 0
        inf_count = 0
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_count += torch.isnan(param).sum().item()
            if torch.isinf(param).any():
                inf_count += torch.isinf(param).sum().item()
        
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️  {stage} - NaN: {nan_count}, Inf: {inf_count}")
            return False
        else:
            print(f"   ✅ {stage} - Model weights are clean")
            return True
    
    def ask_model_stable(self, question: str, use_trained: bool = False, max_tokens: int = 50) -> str:
        """Ask a question to the model with ultra-stable generation parameters."""
        model = self.trained_model if (use_trained and self.trained_model) else self.base_model
        
        prompt = f"Question: {question}\\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Check input tensors for NaN/Inf
        for key, tensor in inputs.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return f"[Input tensor {key} contains NaN/Inf]"
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy decoding for stability
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Disable cache to avoid numerical issues
                )
            except Exception as e:
                return f"[Generation error: {str(e)[:50]}...]"
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        
        # Clean response
        response = response.split('\\n')[0].strip()
        if len(response) > 100:
            response = response[:100] + "..."
        
        return response if response else "[Empty response]"
    
    def test_before_training(self) -> List[Tuple[str, str]]:
        """Test questions BEFORE training to establish baseline."""
        print("\\n🔍 STEP 1: Testing Knowledge BEFORE Ultra-Stable Training")
        print("=" * 58)
        print("Testing model on specific defect questions it shouldn't know:")
        
        before_responses = []
        for i, question in enumerate(self.test_questions, 1):
            print(f"\\n{i}. ❓ {question}")
            response = self.ask_model_stable(question, use_trained=False)
            print(f"   🤖 BEFORE: {response}")
            before_responses.append((question, response))
            
            # Quick analysis
            if any(code in response.lower() for code in ['af-', 'pf-', 'df-']):
                print("   ⚠️  Mentions error codes - likely fabricating")
            elif len(response) < 30:
                print("   ✅ Short/uncertain response - good baseline")
            else:
                print("   📊 Provides generic response")
        
        print(f"\\n📋 Baseline established: Model tested on {len(before_responses)} questions")
        print("🎯 Now let's teach it the REAL answers with ULTRA-STABLE full parameter training!")
        return before_responses
    
    def create_ultra_stable_training_data(self) -> List[Dict[str, str]]:
        """Create ultra-stable training dataset with minimal examples."""
        print("\\n📚 STEP 2: Creating ULTRA-STABLE Training Dataset")
        print("=" * 52)
        print("Building minimal, focused dataset for stable training:")
        
        examples = []
        total_items = 0
        
        # Use only the most critical defects - minimal dataset to prevent overfitting
        for domain in self.domains:
            print(f"\\n🔧 {domain.name}:")
            
            # Only first 3 defects per domain for ultra-stability
            for defect in domain.defects[:3]:
                print(f"   🐛 {defect.question}")
                # Only 2 variations per defect to minimize training load
                examples.extend([
                    {"text": f"Question: {defect.question}\\nAnswer: {defect.answer}"},
                    {"text": f"Q: {defect.question}\\nA: {defect.answer}"},
                ])
                total_items += 1
            
            # Only 1 feature per domain for minimal dataset
            if domain.features:
                feature = domain.features[0]
                print(f"   ✨ {feature.question}")
                examples.extend([
                    {"text": f"Question: {feature.question}\\nAnswer: {feature.answer}"},
                    {"text": f"Q: {feature.question}\\nA: {feature.answer}"},
                ])
                total_items += 1
        
        print(f"\\n🎯 ULTRA-STABLE DATASET CREATED:")
        print(f"   📝 {total_items} knowledge items (minimal for stability)")
        print(f"   🔢 {len(examples)} total training examples (2x variations per item)")
        print(f"   🛡️  Ultra-small dataset to prevent NaN corruption")
        
        return examples
    
    def train_ultra_stable_model(self, examples: List[Dict[str, str]], num_epochs: int = 2) -> bool:
        """Train model with ULTRA-STABLE parameters to prevent NaN corruption."""
        print(f"\\n🚀 STEP 3: ULTRA-STABLE FULL Parameter Training")
        print("=" * 50)
        print(f"🔥 Training ALL {sum(p.numel() for p in self.base_model.parameters()):,} parameters")
        print(f"📊 Dataset: {len(examples)} examples × {num_epochs} epochs = {len(examples) * num_epochs} training steps")
        print("🛡️  Using ULTRA-STABLE parameters to prevent NaN corruption")
        
        # Create dataset
        dataset = Dataset.from_list(examples)
        
        def tokenize(batch):
            tokenized = self.tokenizer(
                batch['text'], 
                truncation=True, 
                padding="max_length", 
                max_length=128  # Very short sequences for stability
            )
            # Set labels for causal language modeling
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        print("🔄 Tokenizing ultra-stable dataset...")
        dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
        
        # Load fresh model for training
        print("🔧 Loading fresh model for ULTRA-STABLE full parameter training...")
        training_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float32,  # Force FP32
            device_map="auto", 
            trust_remote_code=True
        )
        
        # Check initial weights
        print("🔍 Checking initial model weights...")
        if not self.check_model_weights(training_model, "Initial"):
            print("❌ Initial model already has NaN/Inf - aborting")
            return False
        
        # ULTRA-STABLE training configuration
        output_dir = Path("../results/ultra_stable_trained_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,            # Minimal epochs
            per_device_train_batch_size=1,          # Smallest possible batch
            gradient_accumulation_steps=2,          # Minimal accumulation
            learning_rate=1e-7,                     # Ultra-low learning rate
            warmup_steps=10,                        # Minimal warmup
            logging_steps=2,                        # Frequent monitoring
            save_steps=20,                          # Frequent saves
            save_total_limit=5,                     # Keep more checkpoints
            dataloader_drop_last=False,             
            remove_unused_columns=False,
            report_to=[],                           # No external logging
            disable_tqdm=False,                     # Show progress
            weight_decay=0.001,                     # Minimal regularization
            max_grad_norm=0.1,                      # Very strong gradient clipping
            lr_scheduler_type="constant",           # Constant LR for stability
            optim="adamw_torch",                    # Standard optimizer
            fp16=False,                             # Force FP32
            bf16=False,                             # Force FP32
            dataloader_pin_memory=False,            # Reduce memory issues
            gradient_checkpointing=False,           # Disable for stability
        )
        
        # Use proper data collator for causal LM
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt"
        )
        
        # Custom trainer with extensive monitoring
        class UltraStableTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.nan_detected = False
            
            def compute_loss(self, model, inputs, return_outputs=False):
                # Check model weights before forward pass
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"   ⚠️  NaN/Inf detected in {name} BEFORE forward pass!")
                        self.nan_detected = True
                        if return_outputs:
                            return torch.tensor(float('inf')), None
                        return torch.tensor(float('inf'))
                
                loss = super().compute_loss(model, inputs, return_outputs)
                
                # Check loss value
                if return_outputs:
                    loss_value, outputs = loss
                    if torch.isnan(loss_value) or torch.isinf(loss_value):
                        print(f"   ⚠️  Invalid loss detected: {loss_value}")
                        self.nan_detected = True
                    return loss_value, outputs
                else:
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"   ⚠️  Invalid loss detected: {loss}")
                        self.nan_detected = True
                    return loss
            
            def training_step(self, model, inputs):
                result = super().training_step(model, inputs)
                
                # Check if training should stop due to NaN
                if self.nan_detected:
                    print("   🛑 STOPPING TRAINING due to NaN detection!")
                    self.control.should_training_stop = True
                
                return result
        
        trainer = UltraStableTrainer(
            model=training_model, 
            args=args, 
            train_dataset=dataset,
            data_collator=collator, 
            tokenizer=self.tokenizer
        )
        
        print(f"⏱️  Starting ULTRA-STABLE training...")
        print(f"🔥 Learning rate: {args.learning_rate} (ultra-conservative)")
        print(f"📊 This should prevent NaN corruption while still achieving learning")
        start_time = time.time()
        
        try:
            result = trainer.train()
            training_time = time.time() - start_time
            
            if trainer.nan_detected:
                print(f"\\n❌ Training stopped due to NaN detection after {training_time:.1f} seconds")
                return False
            
            print(f"\\n🎉 ULTRA-STABLE training completed in {training_time:.1f} seconds!")
            print(f"📉 Final loss: {result.training_loss:.4f}")
            print(f"⚡ Trained on {len(examples)} examples for {num_epochs} epochs")
            
            # Final weight check
            print("🔍 Final weight validation...")
            if not self.check_model_weights(training_model, "Final"):
                print("❌ Model corrupted during training")
                return False
            
            # Save the complete trained model
            final_path = output_dir / "final_ultra_stable_model"
            trainer.save_model(str(final_path))
            self.tokenizer.save_pretrained(str(final_path))
            
            print(f"💾 Complete ultra-stable model saved to: {final_path}")
            
            # Load trained model for testing
            print("🔄 Loading ultra-stable trained model...")
            self.trained_model = AutoModelForCausalLM.from_pretrained(
                str(final_path), 
                torch_dtype=torch.float32, 
                device_map="auto", 
                trust_remote_code=True
            )
            
            # Final validation
            print("🔍 Loaded model validation...")
            if not self.check_model_weights(self.trained_model, "Loaded"):
                print("❌ Loaded model has issues")
                return False
            
            print("✅ Model ultra-stable trained with preserved stability!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_after_training(self, before_responses: List[Tuple[str, str]]):
        """Test the SAME questions after ultra-stable training."""
        print("\\n🧠 STEP 4: Testing Knowledge AFTER Ultra-Stable Training")
        print("=" * 58)
        print("Testing the SAME questions after ULTRA-STABLE full parameter training:")
        
        improvements = 0
        specific_knowledge = 0
        generation_failures = 0
        
        # Expected knowledge from training
        expected_knowledge = {
            "af-3001": ["jwt", "validation", "timeout", "5000ms", "authflow.yml"],
            "pf-1205": ["webhook", "delivery", "timeout", "90s", "retry"],
            "df-7890": ["memory leak", "json", "transformer", "streaming_mode"],
            "af-6001": ["saml", "assertion", "timeout", "federation"],
            "pf-4001": ["paypal", "express", "checkout", "session"],
        }
        
        for i, (question, old_response) in enumerate(before_responses, 1):
            print(f"\\n{i}. ❓ {question}")
            print(f"   📊 BEFORE: {old_response}")
            
            new_response = self.ask_model_stable(question, use_trained=True, max_tokens=80)
            print(f"   📊 AFTER:  {new_response}")
            
            # Check if generation failed
            if new_response.startswith("[") or len(new_response) < 10:
                generation_failures += 1
                print(f"   ❌ Generation failed")
                continue
            
            # Check for specific knowledge learned
            question_lower = question.lower()
            new_response_lower = new_response.lower()
            old_response_lower = old_response.lower()
            
            learned_details = 0
            
            for key, details in expected_knowledge.items():
                if key in question_lower:
                    for detail in details:
                        if detail in new_response_lower and detail not in old_response_lower:
                            learned_details += 1
                            print(f"   ✅ LEARNED: '{detail}' correctly mentioned")
                    break
            
            # Evaluate improvement
            if learned_details >= 2:
                print("   🎉 EXCELLENT: Model learned multiple specific details!")
                improvements += 1
                specific_knowledge += 1
            elif learned_details >= 1:
                print("   ✅ GOOD: Model learned specific knowledge!")
                improvements += 1
            elif len(new_response) > len(old_response) * 1.2:
                print("   📈 IMPROVED: More detailed response")
                improvements += 1
            else:
                print("   📊 STABLE: Response generated but no clear learning detected")
            
            print("   " + "─" * 50)
        
        improvement_rate = improvements / len(before_responses)
        knowledge_rate = specific_knowledge / len(before_responses)
        failure_rate = generation_failures / len(before_responses)
        
        print(f"\\n📊 ULTRA-STABLE TRAINING RESULTS:")
        print(f"   📈 Overall improvements: {improvements}/{len(before_responses)} ({improvement_rate:.1%})")
        print(f"   🧠 Specific knowledge acquired: {specific_knowledge}/{len(before_responses)} ({knowledge_rate:.1%})")
        print(f"   ⚠️  Generation failures: {generation_failures}/{len(before_responses)} ({failure_rate:.1%})")
        
        if failure_rate == 0:
            print("   🎉 PERFECT: No generation failures - ultra-stable training succeeded!")
        elif failure_rate < 0.2:
            print("   ✅ EXCELLENT: Minimal failures - ultra-stable approach works!")
        else:
            print("   ⚠️  Some failures remain - may need even more conservative settings")
        
        if knowledge_rate > 0.4:
            print("   🎯 OUTSTANDING: Ultra-stable training achieved good knowledge integration!")
        elif improvement_rate > 0.5:
            print("   ✅ SUCCESS: Clear evidence of knowledge acquisition with stability!")
        
        return improvement_rate, failure_rate == 0
    
    def run_ultra_stable_demo(self, num_epochs: int = 2):
        """Run the complete ultra-stable demonstration."""
        print("🏢 ULTRA-STABLE Software Defect Knowledge Acquisition Demo")
        print("=" * 64)
        print("🎯 Proving ultra-stable full fine-tuning prevents NaN corruption!")
        print("🛡️  Using ultra-conservative parameters for maximum stability")
        
        # Setup
        self.setup_model()
        
        # Step 1: Test before training
        before_responses = self.test_before_training()
        
        # Step 2: Create ultra-stable dataset
        training_examples = self.create_ultra_stable_training_data()
        
        print(f"\\n⏳ Starting {len(training_examples)} examples × {num_epochs} epochs of ULTRA-STABLE training...")
        print(f"🛡️  This will prevent NaN corruption while achieving knowledge acquisition!")
        
        # Step 3: Train ultra-stable
        start_time = time.time()
        if self.train_ultra_stable_model(training_examples, num_epochs):
            training_time = time.time() - start_time
            
            # Step 4: Test after training
            improvement_rate, stability_achieved = self.test_after_training(before_responses)
            
            print(f"\\n🎉 ULTRA-STABLE DEMONSTRATION COMPLETE!")
            if stability_achieved:
                print(f"✅ PERFECT: Ultra-stable training prevented NaN corruption!")
                print(f"🧠 Knowledge acquisition rate: {improvement_rate:.1%}")
            else:
                print(f"⚠️  Some instability remains - may need even more conservative settings")
            print(f"🔥 Your GPU did substantial work training {sum(p.numel() for p in self.base_model.parameters()):,} parameters!")
            print(f"⏱️  Training time: {training_time:.1f} seconds")
            
        else:
            print(f"❌ Ultra-stable training failed - NaN corruption occurred")
            print("💡 Consider even more conservative settings or alternative approaches")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ultra-Stable Full Training Demo")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    
    args = parser.parse_args()
    
    # GPU check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name}")
        print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
        print("✅ Ready for ULTRA-STABLE training workload!")
    else:
        print("⚠️  No CUDA GPU detected")
        print("💡 This demo requires GPU for training demonstration")
        return
    
    # Run ultra-stable demo
    demo = UltraStableTrainingDemo()
    demo.run_ultra_stable_demo(num_epochs=args.epochs)

if __name__ == "__main__":
    main()