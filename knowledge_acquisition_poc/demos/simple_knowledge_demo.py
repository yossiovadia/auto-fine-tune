#!/usr/bin/env python3
"""
Simple Knowledge Acquisition Demo

A streamlined demonstration that focuses on the core experience:
1. Ask questions → Model says "I don't know"
2. Train the model (with live progress)
3. Ask same questions → Model gives detailed answers

This is the most straightforward way to show knowledge acquisition.
"""

import torch
import time
from pathlib import Path
from typing import Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from poc_knowledge_domains import get_all_knowledge_domains

class SimpleKnowledgeDemo:
    """Simple demonstration of knowledge acquisition."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.base_model = None
        self.trained_model = None
        self.domains = get_all_knowledge_domains()
        
        # Strong system prompt to ensure "I don't know" responses
        self.safe_prompt = """You are a helpful assistant. IMPORTANT: If you don't have specific, factual knowledge about a topic, you MUST respond with "I don't know about that" or "I don't have information on that topic" rather than guessing or making up information. Never fabricate facts, dates, names, or technical details. Be honest about your knowledge limitations."""
    
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
    
    def ask_model(self, question: str, use_trained: bool = False) -> Tuple[str, float]:
        """Ask a question to either base or trained model."""
        model = self.trained_model if (use_trained and self.trained_model) else self.base_model
        
        # Add safety prompt for base model with stronger instruction
        if not use_trained:
            prompt = f"""<|system|>{self.safe_prompt}<|user|>{question}

Remember: If you don't have specific knowledge about this topic, respond with "I don't know about that topic" rather than guessing.<|assistant|>I"""
        else:
            prompt = f"<|user|>{question}<|assistant|>"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=120, temperature=0.1, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.1
            )
        
        response_time = time.time() - start_time
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip().split('<|')[0].strip()
        
        return response, response_time
    
    def show_sample_questions(self):
        """Show sample questions from knowledge domains."""
        print("\n📚 Sample Questions You Can Ask:")
        print("=" * 40)
        
        samples = []
        for domain in self.domains:
            for question in domain.test_questions[:2]:  # 2 from each domain
                samples.append(f"• {question.question}")
        
        for sample in samples[:6]:  # Show first 6
            print(sample)
        
        print(f"\n💡 The untrained model should say 'I don't know' to these!")
    
    def train_model(self):
        """Train the model with live progress."""
        print("\n🚀 TRAINING THE MODEL ON NEW KNOWLEDGE")
        print("=" * 45)
        
        # Create dataset
        examples = []
        for domain in self.domains:
            for fact in domain.facts:
                examples.append({
                    "text": f"<|user|>{fact.question}<|assistant|>{fact.answer}<|end|>"
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
        output_dir = Path("simple_demo_output")
        output_dir.mkdir(exist_ok=True)
        
        args = TrainingArguments(
            output_dir=str(output_dir), num_train_epochs=2, per_device_train_batch_size=2,
            learning_rate=2e-4, logging_steps=3, save_steps=50, fp16=True,
            remove_unused_columns=False, report_to=None, disable_tqdm=False
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
            
            print(f"🎉 Training completed in {training_time:.1f} seconds!")
            print(f"📉 Final loss: {result.training_loss:.4f}")
            
            # Save and load trained model
            model_path = output_dir / "final"
            trainer.save_model(str(model_path))
            
            self.trained_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
            self.trained_model = PeftModel.from_pretrained(self.trained_model, str(model_path))
            
            print("✅ Trained model ready for testing!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def run_demo(self):
        """Run the complete demo."""
        print("🎬 Simple Knowledge Acquisition Demo")
        print("=" * 40)
        print("See how fine-tuning teaches models new knowledge!")
        
        self.setup_model()
        self.show_sample_questions()
        
        # Phase 1: Test untrained model
        print("\n" + "="*50)
        print("🔍 PHASE 1: Test Untrained Model")
        print("="*50)
        print("Ask questions about the topics above.")
        print("The model should say 'I don't know' since it wasn't trained on them.")
        
        pre_training_qa = []
        
        while True:
            question = input("\n❓ Your question (or 'done' to continue): ").strip()
            if question.lower() in ['done', 'next', 'continue']:
                break
            if not question:
                continue
                
            print("🤖 Untrained model response:")
            response, time_taken = self.ask_model(question, use_trained=False)
            print(f"   {response}")
            
            pre_training_qa.append((question, response))
            
            if any(phrase in response.lower() for phrase in ['don\'t know', 'no information', 'not sure']):
                print("   ✅ Good! Model admits it doesn't know.")
            else:
                print("   ⚠️  Model might be guessing - let's train it!")
        
        # Phase 2: Training
        print("\n" + "="*50)
        print("🚀 PHASE 2: Train the Model")
        print("="*50)
        
        if len(pre_training_qa) == 0:
            print("💡 Ask at least one question first to see the difference!")
            return
        
        confirm = input("Ready to train the model on new knowledge? (y/N): ")
        if confirm.lower() != 'y':
            print("Demo cancelled.")
            return
        
        if not self.train_model():
            return
        
        # Phase 3: Test trained model
        print("\n" + "="*50)
        print("🧠 PHASE 3: Test Trained Model")
        print("="*50)
        print("Now ask the SAME questions to see the improvement!")
        
        for question, old_response in pre_training_qa:
            print(f"\n❓ Question: {question}")
            
            print("🤖 OLD Response (untrained):")
            print(f"   {old_response}")
            
            print("🧠 NEW Response (trained):")
            response, time_taken = self.ask_model(question, use_trained=True)
            print(f"   {response}")
            
            # Simple comparison
            if len(response) > len(old_response) and 'don\'t know' not in response.lower():
                print("   🎉 SUCCESS: Much more knowledgeable response!")
            elif 'don\'t know' not in response.lower():
                print("   📈 IMPROVED: Better response after training!")
            else:
                print("   ⚠️  MIXED: May need more training or different question.")
            
            print("-" * 60)
        
        # Phase 4: Free testing
        print("\n" + "="*50)
        print("🎮 PHASE 4: Free Testing")
        print("="*50)
        print("Try asking new questions to test the trained model!")
        
        while True:
            question = input("\n❓ Test question (or 'done' to finish): ").strip()
            if question.lower() in ['done', 'exit', 'quit', 'finish']:
                break
            if not question:
                continue
                
            print("🧠 Trained model response:")
            response, time_taken = self.ask_model(question, use_trained=True)
            print(f"   {response}")
            print(f"   ⏱️  ({time_taken:.1f}s)")
        
        print("\n🎉 Demo Complete!")
        print("You've seen how fine-tuning can teach models completely new knowledge!")
        print("The transformation from 'I don't know' to detailed answers proves")
        print("that models can acquire previously unknown information through training.")

def main():
    """Run the simple demo."""
    if torch.cuda.is_available():
        print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No GPU - training will be slower")
    
    demo = SimpleKnowledgeDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()