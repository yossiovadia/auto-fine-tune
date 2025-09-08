#!/usr/bin/env python3
"""
Interactive Knowledge Acquisition Demo

This creates a live demonstration where users can:
1. Ask questions to an untrained model (with system prompt to avoid hallucinations)
2. Trigger live training on new knowledge domains
3. Ask the same questions to the newly trained model
4. See the clear before/after transformation

The demo includes safeguards to ensure the base model responds with "I don't know"
rather than hallucinating answers about topics it wasn't trained on.
"""

import torch
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from poc_knowledge_domains import get_all_knowledge_domains

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveKnowledgeDemo:
    """Interactive demonstration of knowledge acquisition through fine-tuning."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.base_model = None
        self.trained_model = None
        self.tokenizer = None
        self.domains = get_all_knowledge_domains()
        self.demo_session = {
            "start_time": datetime.now().isoformat(),
            "questions_asked": [],
            "training_completed": False
        }
        
        # System prompt to prevent hallucination in untrained model
        self.base_system_prompt = """You are a helpful AI assistant. For any question about topics you are not trained on or don't have reliable information about, you should respond with "I don't know" or "I don't have information about that topic" rather than guessing or making up information. Be honest about the limits of your knowledge."""
        
        print("🎬 Interactive Knowledge Acquisition Demo")
        print("=" * 50)
        print("This demo will show you how fine-tuning teaches models new knowledge!")
    
    def load_base_model(self):
        """Load the base model with safeguards against hallucination."""
        if self.base_model is not None:
            return
            
        print(f"\n🤖 Loading base model: {self.model_name}")
        print("   (This model will say 'I don't know' for topics it wasn't trained on)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Base model loaded successfully!")
    
    def generate_response(self, question: str, use_trained_model: bool = False, max_new_tokens: int = 150) -> Tuple[str, float]:
        """Generate a response using either base or trained model."""
        model = self.trained_model if (use_trained_model and self.trained_model) else self.base_model
        
        if model is None:
            return "Error: Model not loaded", 0.0
        
        # Create prompt with system instruction for base model
        if not use_trained_model:
            prompt = f"""<|system|>
{self.base_system_prompt}
<|user|>
{question}
<|assistant|>
"""
        else:
            prompt = f"""<|user|>
{question}
<|assistant|>
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response_time = time.time() - start_time
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # Clean up response
        response = response.split('<|')[0].strip()  # Remove any trailing tokens
        
        return response, response_time
    
    def show_available_topics(self):
        """Show topics the model will be trained on."""
        print("\n📚 Available Knowledge Domains for Training:")
        print("=" * 40)
        
        for i, domain in enumerate(self.domains, 1):
            print(f"\n{i}. {domain.name}")
            print(f"   📖 {domain.description}")
            print(f"   📊 {len(domain.facts)} facts to learn")
            
            # Show sample questions
            print("   🔍 Example questions you could ask:")
            for j, fact in enumerate(domain.test_questions[:2]):  # Show first 2
                print(f"      • {fact.question}")
        
        print(f"\n💡 Try asking questions about these topics!")
        print(f"   The untrained model should say 'I don't know'")
    
    def suggest_questions(self):
        """Suggest good questions to demonstrate the learning."""
        print("\n💭 Suggested Questions to Try:")
        print("=" * 30)
        
        suggestions = []
        for domain in self.domains:
            for fact in domain.test_questions[:1]:  # One from each domain
                suggestions.append(fact.question)
        
        for i, question in enumerate(suggestions, 1):
            print(f"{i}. {question}")
        
        print(f"\nOr ask your own questions about the topics above!")
    
    def ask_question_interactive(self, use_trained_model: bool = False):
        """Interactive question asking with the model."""
        model_type = "trained" if use_trained_model else "base"
        print(f"\n🗣️  Ask the {model_type} model a question (or 'back' to return):")
        
        while True:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['back', 'exit', 'quit', 'done']:
                break
            
            if not question:
                continue
            
            print(f"\n🤖 {model_type.title()} model thinking...")
            response, response_time = self.generate_response(question, use_trained_model)
            
            print(f"💬 Response ({response_time:.1f}s):")
            print(f"   {response}")
            
            # Track the question for session log
            self.demo_session["questions_asked"].append({
                "question": question,
                "model_type": model_type,
                "response": response,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            })
            
            # Analyze response for "I don't know" patterns
            if not use_trained_model:
                unknown_indicators = ['don\'t know', 'not sure', 'no information', 'cannot', 'unable']
                if any(indicator in response.lower() for indicator in unknown_indicators):
                    print("✅ Good! The base model correctly admits it doesn't know.")
                else:
                    print("⚠️  The base model might be guessing - this is why we need training!")
            
            print("\n" + "-" * 50)
    
    def create_training_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Create training dataset from knowledge domains."""
        print("📚 Creating training dataset from knowledge domains...")
        
        all_examples = []
        for domain in self.domains:
            for fact in domain.facts:
                example = {
                    "instruction": fact.question,
                    "input": "",
                    "output": fact.answer,
                    "domain": domain.name,
                    "category": fact.category
                }
                all_examples.append(example)
        
        # Split into train/val
        split_idx = int(0.85 * len(all_examples))
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]
        
        print(f"   ✅ Created: {len(train_examples)} training, {len(val_examples)} validation examples")
        return train_examples, val_examples
    
    def format_training_data(self, examples: List[Dict]) -> Dataset:
        """Format examples for training."""
        def format_instruction(example):
            instruction = example['instruction']
            output = example['output']
            
            text = f"<|user|>\n{instruction}\n<|assistant|>\n{output}<|end|>"
            return {"text": text}
        
        formatted = [format_instruction(ex) for ex in examples]
        return Dataset.from_list(formatted)
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset."""
        def tokenize_function(examples):
            tokenized = self.tokenizer(
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
    
    def train_model_live(self):
        """Train the model live with progress updates."""
        print("\n🚀 Starting Live Model Training!")
        print("=" * 40)
        print("Training the model on new knowledge domains...")
        print("You'll see live progress updates during training.")
        
        # Create training data
        train_examples, val_examples = self.create_training_dataset()
        
        # Setup model for training
        print("\n🔧 Preparing model for training...")
        training_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Add LoRA adapters
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        training_model = get_peft_model(training_model, lora_config)
        print("✅ LoRA adapters added for efficient training")
        
        # Prepare datasets
        train_dataset = self.format_training_data(train_examples)
        val_dataset = self.format_training_data(val_examples)
        
        train_dataset = self.tokenize_dataset(train_dataset)
        val_dataset = self.tokenize_dataset(val_dataset)
        
        # Training arguments with live updates
        output_dir = Path("demo_training_output")
        output_dir.mkdir(exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            warmup_steps=10,
            logging_steps=5,  # Frequent logging for live updates
            eval_steps=20,
            save_steps=50,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,  # No wandb for this demo
            disable_tqdm=False  # Show progress bars
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=training_model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Custom trainer with live updates
        class LiveUpdateTrainer(Trainer):
            def log(self, logs: Dict[str, float]) -> None:
                super().log(logs)
                if 'loss' in logs:
                    print(f"📈 Training loss: {logs['loss']:.4f}")
                if 'eval_loss' in logs:
                    print(f"📊 Validation loss: {logs['eval_loss']:.4f}")
        
        trainer = LiveUpdateTrainer(
            model=training_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        print(f"\n⏱️  Starting training process...")
        print(f"   📊 Training examples: {len(train_examples)}")
        print(f"   📊 Validation examples: {len(val_examples)}")
        print(f"   ⚙️  Epochs: 2")
        print(f"   🎯 Learning rate: 2e-4")
        
        start_time = time.time()
        
        # Train the model
        try:
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            print(f"\n🎉 Training completed successfully!")
            print(f"   ⏱️  Training time: {training_time:.1f} seconds")
            print(f"   📉 Final loss: {train_result.training_loss:.4f}")
            
            # Save the trained model
            final_model_path = output_dir / "final_model"
            trainer.save_model(str(final_model_path))
            
            # Load the trained model for inference
            self.trained_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.trained_model = PeftModel.from_pretrained(self.trained_model, str(final_model_path))
            
            self.demo_session["training_completed"] = True
            
            print(f"✅ Trained model loaded and ready for testing!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
        
        return True
    
    def compare_responses(self):
        """Compare responses between base and trained models on same questions."""
        print("\n🔄 Side-by-Side Comparison")
        print("=" * 40)
        print("Ask a question and see responses from both models!")
        
        while True:
            question = input("\n❓ Question for comparison (or 'back'): ").strip()
            
            if question.lower() in ['back', 'exit', 'quit']:
                break
            
            if not question:
                continue
            
            print(f"\n🤖 Base Model Response:")
            base_response, base_time = self.generate_response(question, use_trained_model=False)
            print(f"   {base_response}")
            print(f"   ⏱️  Response time: {base_time:.1f}s")
            
            print(f"\n🧠 Trained Model Response:")
            trained_response, trained_time = self.generate_response(question, use_trained_model=True)
            print(f"   {trained_response}")
            print(f"   ⏱️  Response time: {trained_time:.1f}s")
            
            print(f"\n📊 Comparison:")
            if "don't know" in base_response.lower() and "don't know" not in trained_response.lower():
                print("   ✅ SUCCESS: Transformed from 'unknown' to knowledgeable!")
            elif len(trained_response) > len(base_response):
                print("   📈 IMPROVEMENT: More detailed response after training")
            else:
                print("   ⚠️  MIXED: Check if training was effective for this topic")
            
            print("\n" + "-" * 60)
    
    def save_demo_session(self):
        """Save the demo session for review."""
        session_file = Path("demo_session.json")
        self.demo_session["end_time"] = datetime.now().isoformat()
        
        with open(session_file, 'w') as f:
            json.dump(self.demo_session, f, indent=2)
        
        print(f"💾 Demo session saved to: {session_file}")
    
    def run_interactive_demo(self):
        """Run the complete interactive demonstration."""
        print("\n🎬 Welcome to the Interactive Knowledge Acquisition Demo!")
        print("This demo will show you how fine-tuning teaches models new knowledge.")
        
        # Load base model
        self.load_base_model()
        
        while True:
            print("\n" + "=" * 50)
            print("📋 DEMO MENU")
            print("=" * 50)
            print("1. 📖 View available knowledge domains")
            print("2. 💭 Get suggested questions to ask")
            print("3. 🗣️  Ask questions to the BASE model")
            print("4. 🚀 Train the model on new knowledge")
            print("5. 🧠 Ask questions to the TRAINED model")
            print("6. 🔄 Compare base vs trained responses")
            print("7. 💾 Save session and exit")
            
            if not self.demo_session["training_completed"]:
                print("\n💡 Start with options 1-3 to see what the base model doesn't know,")
                print("   then use option 4 to train it, and options 5-6 to see the improvement!")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                self.show_available_topics()
            
            elif choice == "2":
                self.suggest_questions()
            
            elif choice == "3":
                if self.base_model is None:
                    self.load_base_model()
                print("\n🤖 Asking questions to the BASE model...")
                print("(It should say 'I don't know' for topics it wasn't trained on)")
                self.ask_question_interactive(use_trained_model=False)
            
            elif choice == "4":
                if self.demo_session["training_completed"]:
                    print("✅ Model is already trained! Use options 5-6 to test it.")
                else:
                    print("\n🚀 This will train the model on new knowledge domains.")
                    print("The training process will take a few minutes and show live progress.")
                    confirm = input("Start training? (y/N): ").strip().lower()
                    
                    if confirm == 'y':
                        success = self.train_model_live()
                        if success:
                            print("\n🎉 Training complete! Now try asking the same questions.")
                            print("The trained model should give detailed, accurate answers!")
                    else:
                        print("Training cancelled.")
            
            elif choice == "5":
                if not self.demo_session["training_completed"]:
                    print("❌ Please train the model first (option 4)")
                else:
                    print("\n🧠 Asking questions to the TRAINED model...")
                    print("(It should now know about the trained knowledge domains)")
                    self.ask_question_interactive(use_trained_model=True)
            
            elif choice == "6":
                if not self.demo_session["training_completed"]:
                    print("❌ Please train the model first (option 4)")
                else:
                    self.compare_responses()
            
            elif choice == "7":
                self.save_demo_session()
                print("\n👋 Thanks for trying the Knowledge Acquisition Demo!")
                print("You've seen how fine-tuning can teach models completely new information!")
                break
            
            else:
                print("❌ Invalid option. Please choose 1-7.")

def main():
    """Main entry point for the interactive demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Knowledge Acquisition Demo")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                       help="Base model to use")
    
    args = parser.parse_args()
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"🎮 GPU detected: {torch.cuda.get_device_name()}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected - training will be slow")
        
    demo = InteractiveKnowledgeDemo(model_name=args.model)
    demo.run_interactive_demo()

if __name__ == "__main__":
    main()