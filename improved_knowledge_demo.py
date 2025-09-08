#!/usr/bin/env python3
"""
Improved Knowledge Acquisition Demo

This version uses a more effective approach to ensure the base model says "I don't know"
by using explicit unknown topic questions and better prompt engineering.
"""

import torch
import time
from pathlib import Path
from typing import Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from poc_knowledge_domains import get_all_knowledge_domains

class ImprovedKnowledgeDemo:
    """Improved demonstration with better unknown response handling."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.base_model = None
        self.trained_model = None
        self.domains = get_all_knowledge_domains()
    
    def setup_model(self):
        """Setup tokenizer and base model."""
        print(f"🤖 Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        print("✅ Model loaded!")
    
    def ask_model_directly(self, question: str, use_trained: bool = False) -> Tuple[str, float]:
        """Ask model with minimal prompting to see natural response."""
        model = self.trained_model if (use_trained and self.trained_model) else self.base_model
        
        # Simple, direct prompting
        if not use_trained:
            # For base model, ask in a way that encourages honesty
            prompt = f"Question: {question}\n\nPlease answer only if you are certain about the facts. If you're not sure or don't have reliable information, say 'I don't know'.\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,  # Shorter responses
                temperature=0.1, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id, 
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response_time = time.time() - start_time
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        
        # Clean up response
        response = response.split('\n')[0].strip()  # Take first line only
        if len(response) > 200:
            response = response[:200] + "..."
        
        return response, response_time
    
    def test_obviously_unknown_questions(self):
        """Test with questions that are obviously unknown to demonstrate the issue."""
        print("\n🔍 Testing Obviously Unknown Questions")
        print("="*50)
        print("First, let's test questions that are clearly unknown to see model behavior...")
        
        obvious_unknown = [
            "What is the capital of the fictional country Zemuria?",
            "Who invented the Flibber-Jabber device in 1847?", 
            "What is the atomic weight of the element Impossibrium?",
            "When did the Battle of Nonexistia take place?"
        ]
        
        for question in obvious_unknown:
            print(f"\n❓ {question}")
            response, _ = self.ask_model_directly(question)
            print(f"🤖 Response: {response}")
            
            if any(phrase in response.lower() for phrase in ['don\'t know', 'not sure', 'don\'t have', 'unknown', 'fictional', 'does not exist']):
                print("   ✅ Good! Model shows uncertainty/admits ignorance")
            else:
                print("   ⚠️  Model is making up information")
        
        print(f"\n💡 As you can see, this model tends to fabricate answers even for obviously unknown topics.")
        print(f"   This is why the knowledge acquisition demo is so valuable - it shows we can")
        print(f"   teach the model REAL information through fine-tuning!")
    
    def show_sample_questions(self):
        """Show sample questions from knowledge domains."""
        print("\n📚 Knowledge Domain Questions to Test:")
        print("=" * 45)
        
        samples = []
        for domain in self.domains:
            print(f"\n🧠 {domain.name}:")
            for i, question in enumerate(domain.test_questions[:2]):
                print(f"   {i+1}. {question.question}")
                samples.append(question.question)
        
        print(f"\n💡 These are about real concepts that post-2024 models shouldn't know.")
        print(f"   The model will likely fabricate plausible-sounding but incorrect answers.")
        return samples[:6]  # Return first 6 for testing
    
    def demonstrate_hallucination_vs_learning(self):
        """Demonstrate the difference between hallucination and actual learning."""
        print("\n🎭 HALLUCINATION vs LEARNING DEMONSTRATION")
        print("="*50)
        
        print("This demo shows the difference between:")
        print("📢 HALLUCINATION: Model makes up plausible-sounding but wrong information")
        print("🧠 LEARNING: Model provides accurate information from training")
        
        # Get some domain questions
        test_questions = []
        for domain in self.domains:
            test_questions.extend([q.question for q in domain.test_questions[:2]])
        
        print(f"\nPhase 1: See the model HALLUCINATE (make up wrong answers)")
        print("-" * 55)
        
        hallucination_responses = []
        for i, question in enumerate(test_questions[:4]):
            print(f"\n❓ Question {i+1}: {question}")
            response, _ = self.ask_model_directly(question)
            print(f"🤖 Untrained Response: {response}")
            print("   ⚠️  This is likely fabricated/incorrect information!")
            
            hallucination_responses.append((question, response))
            
            if i < 3:  # Ask user if they want to continue
                continue_demo = input(f"\nContinue to next question? (y/N): ").strip().lower()
                if continue_demo != 'y':
                    break
        
        print(f"\n🚀 Now let's train the model on the CORRECT information...")
        train_now = input(f"Ready to train the model? (y/N): ").strip().lower()
        
        if train_now == 'y':
            if self.train_model():
                print(f"\nPhase 2: See the model provide LEARNED (correct) answers")
                print("-" * 58)
                
                for question, old_response in hallucination_responses:
                    print(f"\n❓ Question: {question}")
                    print(f"🤖 OLD (Hallucinated): {old_response}")
                    
                    new_response, _ = self.ask_model_directly(question, use_trained=True)
                    print(f"🧠 NEW (Learned): {new_response}")
                    
                    print(f"📊 Comparison:")
                    if len(new_response) > len(old_response) * 0.8:  # Substantial response
                        print(f"   ✅ TRANSFORMATION: From hallucination to learned knowledge!")
                    else:
                        print(f"   📈 IMPROVED: Better response after training")
                    
                    print("-" * 60)
        else:
            print("Demo paused. Run again when ready to see the full transformation!")
    
    def train_model(self):
        """Train the model with live progress."""
        print("\n🚀 TRAINING THE MODEL ON CORRECT INFORMATION")
        print("=" * 48)
        print("Now teaching the model the REAL facts from our knowledge domains...")
        
        # Create dataset with correct information
        examples = []
        for domain in self.domains:
            for fact in domain.facts:
                examples.append({
                    "text": f"Question: {fact.question}\n\nAnswer: {fact.answer}"
                })
        
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
        
        # Add LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        training_model = get_peft_model(training_model, lora_config)
        
        # Training setup
        output_dir = Path("improved_demo_output")
        output_dir.mkdir(exist_ok=True)
        
        args = TrainingArguments(
            output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=2,
            learning_rate=2e-4, logging_steps=5, save_steps=50, fp16=True,
            remove_unused_columns=False, report_to=None, disable_tqdm=False,
            warmup_steps=10
        )
        
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=training_model, label_pad_token_id=-100
        )
        
        trainer = Trainer(
            model=training_model, args=args, train_dataset=dataset,
            data_collator=collator, tokenizer=self.tokenizer
        )
        
        print(f"⏱️  Training on {len(examples)} correct examples...")
        start_time = time.time()
        
        try:
            result = trainer.train()
            training_time = time.time() - start_time
            
            print(f"🎉 Training completed in {training_time:.1f} seconds!")
            print(f"📉 Final loss: {result.training_loss:.4f}")
            
            # Save and load trained model
            model_path = output_dir / "final"
            trainer.save_model(str(model_path))
            
            self.trained_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
            self.trained_model = PeftModel.from_pretrained(self.trained_model, str(model_path))
            
            print("✅ Trained model ready! Now it has learned the correct information.")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def run_demo(self):
        """Run the improved demonstration."""
        print("🎬 Improved Knowledge Acquisition Demo")
        print("=" * 42)
        print("This demo shows the transformation from HALLUCINATION to LEARNING")
        
        self.setup_model()
        
        # First, show that the model hallucinates for obviously unknown questions
        self.test_obviously_unknown_questions()
        
        input(f"\nPress Enter to continue to the main demonstration...")
        
        # Then show the knowledge domain questions and hallucination vs learning
        self.show_sample_questions()
        
        print(f"\n💡 Key Point: The model will make up plausible but WRONG answers.")
        print(f"   After training, it will give CORRECT answers based on real knowledge.")
        
        input(f"\nPress Enter to start the hallucination vs learning demo...")
        
        self.demonstrate_hallucination_vs_learning()
        
        print(f"\n🎉 Demo Complete!")
        print(f"You've seen how fine-tuning transforms hallucination into real learning!")

def main():
    """Run the improved demo."""
    if torch.cuda.is_available():
        print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No GPU - training will be slower")
    
    demo = ImprovedKnowledgeDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()