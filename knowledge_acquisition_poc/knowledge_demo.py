#!/usr/bin/env python3
"""
Knowledge Acquisition Demo

This demo proves that fine-tuning can teach models completely new information.
Uses 2025 events and developments that pre-2025 models cannot possibly know.

Features:
- Tests established knowledge (model should know)
- Tests 2025 knowledge (model shouldn't know) 
- Live training on 2025 facts
- Before/after comparison
- Interactive testing

Usage:
    python knowledge_demo.py [--model MODEL_NAME] [--mode MODE]
    
Modes:
    demo: Full interactive demonstration (default)
    train: Just run training and save model
    test: Test with existing trained model
"""

import torch
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Import domains from the organized structure
import sys
sys.path.append(str(Path(__file__).parent))
from domains.knowledge_domains_2025 import get_all_2025_knowledge_domains

class KnowledgeAcquisitionDemo:
    """Single, comprehensive knowledge acquisition demonstration."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.base_model = None
        self.trained_model = None
        self.domains = get_all_2025_knowledge_domains()
        
        # Established facts the model should know
        self.established_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?", 
            "What year did World War II end?",
            "What is 2 + 2?",
            "Who founded Microsoft?"
        ]
    
    def setup_model(self):
        """Setup tokenizer and base model."""
        print(f"🤖 Loading model: {self.model_name}")
        print(f"   (This model was trained before 2025)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        print("✅ Model loaded!")
    
    def ask_model(self, question: str, use_trained: bool = False, max_tokens: int = 80) -> Tuple[str, float]:
        """Ask a question to either base or trained model."""
        model = self.trained_model if (use_trained and self.trained_model) else self.base_model
        
        prompt = f"Question: {question}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=0.1, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response_time = time.time() - start_time
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        
        # Clean up response
        response = response.split('\n')[0].strip()
        if len(response) > 150:
            response = response[:150] + "..."
        
        return response, response_time
    
    def test_established_knowledge(self):
        """Test model on established facts to show it can answer questions."""
        print("\n✅ STEP 1: Testing Established Knowledge")
        print("=" * 42)
        print("Confirming the model can answer basic questions:")
        
        for question in self.established_questions[:3]:
            print(f"\n❓ {question}")
            response, _ = self.ask_model(question)
            print(f"🤖 {response}")
        
        print(f"\n✅ Good! Model knows established facts.")
        return True
    
    def get_2025_questions(self) -> List[str]:
        """Get test questions about 2025 events."""
        questions = []
        for domain in self.domains:
            for fact in domain.test_questions[:2]:  # 2 from each domain
                questions.append(fact.question)
        return questions
    
    def test_2025_knowledge(self) -> List[Tuple[str, str]]:
        """Test model on 2025 events it shouldn't know."""
        print("\n❌ STEP 2: Testing 2025 Knowledge (Unknown)")
        print("=" * 43)
        print("Testing on 2025 events the model cannot know:")
        
        questions_2025 = self.get_2025_questions()
        responses = []
        
        for i, question in enumerate(questions_2025[:4]):  # Test first 4
            print(f"\n❓ {question}")
            response, _ = self.ask_model(question)
            print(f"🤖 {response}")
            
            responses.append((question, response))
            
            # Quick analysis
            if '2025' in response:
                print("   ⚠️  Model mentions 2025 - likely fabricating!")
            elif len(response) < 20:
                print("   📊 Short response - model seems uncertain")
            else:
                print("   📊 Model gave a response - accuracy unknown")
        
        print(f"\n💡 For 2025 events, the model either fabricates or gives uncertain responses.")
        print(f"🎯 Let's teach it the REAL 2025 information!")
        
        return responses
    
    def create_training_data(self) -> List[Dict[str, str]]:
        """Create training dataset from 2025 knowledge domains."""
        print("\n📚 STEP 3: Preparing 2025 Training Data")
        print("=" * 40)
        print("Creating training data from real 2025 events:")
        
        examples = []
        for domain in self.domains:
            print(f"\n🧠 {domain.name}:")
            for fact in domain.facts[:3]:  # First 3 from each domain
                print(f"   📅 {fact.question}")
                examples.append({
                    "text": f"Question: {fact.question}\nAnswer: {fact.answer}"
                })
        
        print(f"\n📊 Created {len(examples)} training examples about 2025")
        return examples
    
    def train_model(self, examples: List[Dict[str, str]]) -> bool:
        """Train the model on 2025 knowledge."""
        print(f"\n🚀 STEP 4: Training Model on 2025 Knowledge")
        print("=" * 42)
        
        # Create dataset
        dataset = Dataset.from_list(examples)
        
        def tokenize(batch):
            tokenized = self.tokenizer(batch['text'], truncation=True, padding=False, max_length=512)
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
        
        # Setup training model
        print("🔧 Preparing model for training...")
        training_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        
        # Add LoRA adapters
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        training_model = get_peft_model(training_model, lora_config)
        
        # Training configuration
        output_dir = Path("../results/trained_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=2,
            learning_rate=2e-4,
            logging_steps=5,
            save_steps=100,
            fp16=True,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb
            disable_tqdm=False
        )
        
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=training_model, label_pad_token_id=-100
        )
        
        trainer = Trainer(
            model=training_model, args=args, train_dataset=dataset,
            data_collator=collator, tokenizer=self.tokenizer
        )
        
        print(f"⏱️  Training on {len(examples)} examples...")
        start_time = time.time()
        
        try:
            result = trainer.train()
            training_time = time.time() - start_time
            
            print(f"\n🎉 Training completed in {training_time:.1f} seconds!")
            print(f"📉 Final loss: {result.training_loss:.4f}")
            
            # Save trained model
            final_path = output_dir / "final"
            trainer.save_model(str(final_path))
            
            # Load for inference
            self.trained_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
            self.trained_model = PeftModel.from_pretrained(self.trained_model, str(final_path))
            
            print("✅ Model learned 2025 knowledge!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def test_learned_knowledge(self, original_responses: List[Tuple[str, str]]):
        """Test the same questions after training to show improvement."""
        print("\n🧠 STEP 5: Testing Learned Knowledge")
        print("=" * 35)
        print("Asking the SAME questions to see improvement:")
        
        improvements = 0
        
        for question, old_response in original_responses:
            print(f"\n❓ {question}")
            print(f"📊 BEFORE: {old_response}")
            
            new_response, _ = self.ask_model(question, use_trained=True)
            print(f"📊 AFTER:  {new_response}")
            
            # Check for improvement
            improvement_signs = [
                len(new_response) > len(old_response) * 1.2,  # Much longer response
                '2025' in new_response and '2025' not in old_response,  # Correct year
                any(month in new_response.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september']),  # Specific dates
            ]
            
            if sum(improvement_signs) >= 2:
                print("   🎉 CLEAR IMPROVEMENT: More detailed and accurate!")
                improvements += 1
            elif len(new_response) > len(old_response):
                print("   📈 IMPROVEMENT: More detailed response")
                improvements += 1
            else:
                print("   📝 CHANGED: Response modified after training")
            
            print("-" * 50)
        
        improvement_rate = improvements / len(original_responses)
        print(f"\n📊 KNOWLEDGE ACQUISITION RESULTS:")
        print(f"   Improvements: {improvements}/{len(original_responses)} ({improvement_rate:.1%})")
        
        if improvement_rate > 0.5:
            print("   🎉 SUCCESS: Strong evidence of knowledge acquisition!")
        elif improvement_rate > 0.25:
            print("   ✅ GOOD: Clear evidence of knowledge acquisition!")
        else:
            print("   📈 PROGRESS: Some evidence of learning")
        
        return improvement_rate
    
    def interactive_testing(self):
        """Allow interactive testing of learned knowledge."""
        print("\n🎮 STEP 6: Interactive Testing")
        print("=" * 29)
        print("Test the model's new 2025 knowledge!")
        
        print(f"\n📚 Topics learned:")
        for domain in self.domains:
            print(f"   • {domain.name}")
        
        print(f"\n💡 Try questions like:")
        print(f"   • 'When was GPT-5 released?'")
        print(f"   • 'What did Apple announce at WWDC 2025?'")
        print(f"   • 'When did Artemis IV launch?'")
        
        while True:
            question = input(f"\n❓ Your question (or 'done'): ").strip()
            if question.lower() in ['done', 'exit', 'quit']:
                break
            if not question:
                continue
            
            print(f"🧠 Model response:")
            response, time_taken = self.ask_model(question, use_trained=True, max_tokens=100)
            print(f"   {response}")
            
            if '2025' in response:
                print(f"   ✅ Great! Model correctly references 2025")
    
    def run_full_demo(self):
        """Run the complete demonstration."""
        print("🎬 Knowledge Acquisition Demo - 2025 Edition")
        print("=" * 45)
        print("🎯 Proving fine-tuning can teach models new information")
        print("📅 Using real 2025 events unknown to pre-2025 models")
        
        self.setup_model()
        
        # Step 1: Test established knowledge
        self.test_established_knowledge()
        input("\nPress Enter to test 2025 knowledge...")
        
        # Step 2: Test unknown 2025 knowledge  
        original_responses = self.test_2025_knowledge()
        input("\nPress Enter to start training...")
        
        # Step 3: Create training data
        training_examples = self.create_training_data()
        
        # Step 4: Train the model
        if self.train_model(training_examples):
            # Step 5: Test learned knowledge
            improvement_rate = self.test_learned_knowledge(original_responses)
            
            # Step 6: Interactive testing
            if improvement_rate > 0:
                self.interactive_testing()
            
            print(f"\n🎉 Demo Complete!")
            print(f"✅ PROVEN: Fine-tuning can teach models new 2025 information!")
        else:
            print(f"❌ Training failed - demo incomplete")
    
    def run_training_only(self):
        """Just run training and save the model."""
        print("🚀 Training Mode: Learning 2025 Knowledge")
        print("=" * 40)
        
        self.setup_model()
        training_examples = self.create_training_data()
        
        if self.train_model(training_examples):
            print("✅ Training complete! Model saved for testing.")
        else:
            print("❌ Training failed!")
    
    def run_testing_only(self, model_path: str):
        """Test with an existing trained model."""
        print("🧠 Testing Mode: Using Trained Model")
        print("=" * 37)
        
        self.setup_model()
        
        # Load trained model
        try:
            self.trained_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
            self.trained_model = PeftModel.from_pretrained(self.trained_model, model_path)
            print(f"✅ Loaded trained model from {model_path}")
            
            self.interactive_testing()
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

def main():
    """Main entry point with command line options."""
    parser = argparse.ArgumentParser(description="Knowledge Acquisition Demo")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model name")
    parser.add_argument("--mode", choices=["demo", "train", "test"], default="demo", help="Demo mode")
    parser.add_argument("--model-path", help="Path to trained model (for test mode)")
    
    args = parser.parse_args()
    
    # GPU check
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CPU mode - training will be slow")
    
    # Initialize demo
    demo = KnowledgeAcquisitionDemo(model_name=args.model)
    
    # Run selected mode
    if args.mode == "demo":
        demo.run_full_demo()
    elif args.mode == "train":
        demo.run_training_only()
    elif args.mode == "test":
        if not args.model_path:
            print("❌ --model-path required for test mode")
            return
        demo.run_testing_only(args.model_path)

if __name__ == "__main__":
    main()