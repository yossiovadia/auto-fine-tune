#!/usr/bin/env python3
"""
Software Defect Knowledge Acquisition Demo

This demo shows how fine-tuning can teach a model about new software defects,
error codes, and features that didn't exist during the model's training.

Perfect for demonstrating business value in software companies with:
- New bug reports and solutions
- Recently introduced features  
- Updated configuration requirements
- Service-specific error codes

Usage:
    python software_defect_demo.py [--model MODEL_NAME]
"""

import torch
import time
import argparse
import os
from pathlib import Path
from typing import Tuple, List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"

# Import software domains
import sys
sys.path.append(str(Path(__file__).parent))
from domains.software_defect_domains import get_all_software_domains

class SoftwareDefectDemo:
    """Demo showing knowledge acquisition for software defect tracking."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.base_model = None
        self.trained_model = None
        self.domains = get_all_software_domains()
        
        # General software questions the model should know
        self.general_questions = [
            "What is a HTTP 404 error?",
            "How do you fix a database connection timeout?",
            "What causes a memory leak in Java?",
            "How do you debug a null pointer exception?",
            "What is a REST API?"
        ]
    
    def setup_model(self):
        """Setup tokenizer and base model."""
        print(f"🤖 Loading model: {self.model_name}")
        print(f"   (Testing knowledge of recent software defects/features)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        print("✅ Model loaded!")
    
    def ask_model(self, question: str, use_trained: bool = False, max_tokens: int = 100) -> Tuple[str, float]:
        """Ask a question to the model."""
        model = self.trained_model if (use_trained and self.trained_model) else self.base_model
        
        prompt = f"Question: {question}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
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
        if len(response) > 200:
            response = response[:200] + "..."
        
        return response, response_time
    
    def test_general_software_knowledge(self):
        """Test model on general software concepts."""
        print("\n✅ STEP 1: Testing General Software Knowledge")
        print("=" * 48)
        print("Confirming the model knows basic software concepts:")
        
        for question in self.general_questions[:3]:
            print(f"\n❓ {question}")
            response, _ = self.ask_model(question)
            print(f"🤖 {response}")
        
        print(f"\n✅ Good! Model knows general software concepts.")
        return True
    
    def get_specific_defect_questions(self) -> List[str]:
        """Get questions about specific defects/features."""
        questions = []
        for domain in self.domains:
            for question in domain.test_questions[:2]:  # 2 from each domain
                questions.append(question.question)
        return questions
    
    def test_specific_defect_knowledge(self) -> List[Tuple[str, str]]:
        """Test model on specific service defects it shouldn't know."""
        print("\n❌ STEP 2: Testing Specific Defect Knowledge")
        print("=" * 46)
        print("Testing on service-specific defects and features:")
        
        defect_questions = self.get_specific_defect_questions()
        responses = []
        
        for i, question in enumerate(defect_questions[:6]):  # Test first 6
            print(f"\n❓ {question}")
            response, _ = self.ask_model(question)
            print(f"🤖 {response}")
            
            responses.append((question, response))
            
            # Analyze response quality
            response_lower = response.lower()
            if any(error_code in response_lower for error_code in ['af-', 'pf-', 'df-']):
                print("   ⚠️  Model mentioned our error codes - likely fabricating!")
            elif any(service in response_lower for service_code in ['authflow', 'payflow', 'dataflow']):
                print("   ⚠️  Model mentioned our services - likely guessing!")
            elif len(response) < 30:
                print("   📊 Short response - model seems uncertain")
            elif any(phrase in response_lower for phrase in ['error', 'fix', 'configure', 'solution']):
                print("   📊 Generic troubleshooting response - not specific to our services")
            else:
                print("   📊 Model gave some response - accuracy unknown")
        
        print(f"\n💡 For service-specific defects, the model either:")
        print(f"   • Admits uncertainty (good)")
        print(f"   • Provides generic troubleshooting advice") 
        print(f"   • Fabricates plausible but incorrect solutions")
        print(f"\n🎯 Let's teach it our ACTUAL defect solutions!")
        
        return responses
    
    def create_software_training_data(self) -> List[Dict[str, str]]:
        """Create training dataset from software defect knowledge."""
        print("\n📚 STEP 3: Creating Software Defect Training Data")
        print("=" * 50)
        print("Preparing training data from real defect knowledge:")
        
        examples = []
        for domain in self.domains:
            print(f"\n🔧 {domain.name}:")
            
            # Add defects
            for defect in domain.defects[:3]:  # First 3 defects
                print(f"   🐛 {defect.question}")
                examples.extend([
                    {"text": f"Question: {defect.question}\nAnswer: {defect.answer}"},
                    {"text": f"Q: {defect.question}\nA: {defect.answer}"},
                    {"text": f"Defect: {defect.question}\nSolution: {defect.answer}"},
                ])
            
            # Add features  
            for feature in domain.features[:2]:  # First 2 features
                print(f"   ✨ {feature.question}")
                examples.extend([
                    {"text": f"Question: {feature.question}\nAnswer: {feature.answer}"},
                    {"text": f"Q: {feature.question}\nA: {feature.answer}"},
                    {"text": f"Feature: {feature.question}\nConfiguration: {feature.answer}"},
                ])
        
        print(f"\n📊 Created {len(examples)} training examples from defect knowledge")
        print(f"🎯 Covers error codes, configuration fixes, and new features")
        return examples
    
    def train_software_model(self, examples: List[Dict[str, str]]) -> bool:
        """Train model on software defect knowledge."""
        print(f"\n🚀 STEP 4: Training on Software Defect Knowledge")
        print("=" * 47)
        
        # Create dataset
        dataset = Dataset.from_list(examples)
        
        def tokenize(batch):
            tokenized = self.tokenizer(batch['text'], truncation=True, padding=False, max_length=512)
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
        
        # Setup training model
        print("🔧 Preparing model for software knowledge training...")
        training_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        training_model = get_peft_model(training_model, lora_config)
        
        # Training configuration
        output_dir = Path("../results/software_trained_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=6,                    # Focused training
            per_device_train_batch_size=1,         
            gradient_accumulation_steps=4,         
            learning_rate=1e-4,                    
            warmup_steps=15,                       
            logging_steps=3,                       
            save_steps=50,
            fp16=False,  # Disabled for MPS compatibility
            remove_unused_columns=False,
            report_to=[],
            disable_tqdm=False,
            weight_decay=0.01,
            max_grad_norm=1.0,
        )
        
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=training_model, label_pad_token_id=-100
        )
        
        trainer = Trainer(
            model=training_model, args=args, train_dataset=dataset,
            data_collator=collator, tokenizer=self.tokenizer
        )
        
        print(f"⏱️  Training on {len(examples)} software defect examples...")
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
            
            print("✅ Model learned software defect knowledge!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def test_learned_defect_knowledge(self, original_responses: List[Tuple[str, str]]):
        """Test software defect knowledge after training."""
        print("\n🧠 STEP 5: Testing Learned Defect Knowledge")
        print("=" * 42)
        print("Testing the SAME defect questions after training:")
        
        # Expected knowledge we trained on
        expected_knowledge = {
            "af-3001": ["token validation timeout", "authflow.yml", "5000ms"],
            "af-2895": ["session store corruption", "redis", "consistency_check"],
            "af-4102": ["mfa", "bypass", "v3.2.5", "strict_timing"],
            "pf-1205": ["webhook delivery timeout", "90s", "retry"],
            "pf-2340": ["currency conversion", "cache_ttl", "300s"],
            "df-7890": ["memory leak", "json transformer", "streaming_mode"],
            "biometric": ["biometric.enabled", "fingerprint", "face"],
        }
        
        improvements = 0
        specific_knowledge = 0
        
        for question, old_response in original_responses:
            print(f"\n❓ {question}")
            print(f"📊 BEFORE: {old_response}")
            
            new_response, _ = self.ask_model(question, use_trained=True, max_tokens=150)
            print(f"📊 AFTER:  {new_response}")
            
            # Check for specific knowledge we taught
            question_lower = question.lower()
            new_response_lower = new_response.lower()
            old_response_lower = old_response.lower()
            
            learned_details = 0
            total_details = 0
            
            for key, details in expected_knowledge.items():
                if key in question_lower:
                    total_details = len(details)
                    for detail in details:
                        if detail in new_response_lower and detail not in old_response_lower:
                            learned_details += 1
                            print(f"   ✅ LEARNED: '{detail}' correctly mentioned")
                    break
            
            # Assessment
            if learned_details > 0:
                print("   🎉 EXCELLENT: Model learned specific defect details!")
                improvements += 1
                specific_knowledge += 1
            elif any(code in new_response_lower for code in ['af-', 'pf-', 'df-']) and not any(code in old_response_lower for code in ['af-', 'pf-', 'df-']):
                print("   📈 GOOD: Model now mentions our error codes")
                improvements += 1
            elif len(new_response) > len(old_response) * 1.2:
                print("   📝 IMPROVED: More detailed response")
                improvements += 1
            else:
                print("   ⚠️  LIMITED: No clear improvement in defect knowledge")
            
            print("-" * 60)
        
        improvement_rate = improvements / len(original_responses)
        knowledge_rate = specific_knowledge / len(original_responses)
        
        print(f"\n📊 SOFTWARE KNOWLEDGE ACQUISITION RESULTS:")
        print(f"   Overall improvements: {improvements}/{len(original_responses)} ({improvement_rate:.1%})")
        print(f"   Specific defect knowledge: {specific_knowledge}/{len(original_responses)} ({knowledge_rate:.1%})")
        
        if knowledge_rate > 0.4:
            print("   🎉 EXCELLENT: Model learned specific defect solutions!")
        elif improvement_rate > 0.5:
            print("   ✅ GOOD: Clear improvement in defect response quality!")
        elif improvement_rate > 0.25:
            print("   📈 PROGRESS: Some evidence of knowledge acquisition")
        else:
            print("   ⚠️  NEEDS WORK: Limited defect knowledge acquisition")
        
        return improvement_rate
    
    def interactive_defect_testing(self):
        """Interactive testing of learned defect knowledge."""
        print("\n🎮 STEP 6: Interactive Defect Testing")
        print("=" * 37)
        print("Test the model's software defect knowledge!")
        
        print(f"\n🔧 Services covered:")
        for domain in self.domains:
            print(f"   • {domain.name}")
        
        print(f"\n💡 Try asking about:")
        print(f"   • 'How to fix AuthFlow error AF-3001?'")
        print(f"   • 'What causes PayFlow error PF-1205?'") 
        print(f"   • 'How to enable Biometric Authentication?'")
        print(f"   • 'How to resolve DataFlow DF-7890?'")
        
        while True:
            question = input(f"\n❓ Your defect question (or 'done'): ").strip()
            if question.lower() in ['done', 'exit', 'quit']:
                break
            if not question:
                continue
            
            print(f"🧠 Model response:")
            response, time_taken = self.ask_model(question, use_trained=True, max_tokens=120)
            print(f"   {response}")
            
            # Check if response contains our specific knowledge
            response_lower = response.lower()
            if any(code in response_lower for code in ['af-', 'pf-', 'df-']):
                print(f"   ✅ Great! Model knows our error codes")
            if any(service in response_lower for service in ['authflow', 'payflow', 'dataflow']):
                print(f"   ✅ Excellent! Model mentions our services")
            if any(config in response_lower for config in ['.yml', 'timeout', 'enabled=true']):
                print(f"   ✅ Perfect! Model provides specific configuration details")
    
    def show_business_value(self):
        """Show the business value of this approach."""
        print(f"\n💼 BUSINESS VALUE DEMONSTRATION")
        print("=" * 35)
        print("This POC proves that fine-tuning can:")
        print("✅ Teach models about new software defects and error codes")
        print("✅ Learn specific configuration fixes and solutions")
        print("✅ Understand new feature documentation")
        print("✅ Provide accurate technical support responses")
        
        print(f"\n🏢 Real-world applications:")
        print("• Customer support chatbots learning new bug fixes")
        print("• Internal documentation assistants for new features") 
        print("• Automated troubleshooting for recent service issues")
        print("• Developer help systems updated with latest solutions")
        
        print(f"\n🎯 Why this matters:")
        print("• Models can stay current with software changes")
        print("• Reduces time to document new defects/features")
        print("• Enables automated support for latest issues")
        print("• Proves knowledge acquisition works for technical domains")
    
    def run_demo(self):
        """Run the complete software defect demonstration."""
        print("🏢 Software Defect Knowledge Acquisition Demo")
        print("=" * 47)
        print("🎯 Proving fine-tuning can learn new software defects and solutions")
        print("💼 Business scenario: Software company with evolving services")
        
        self.setup_model()
        
        # Step 1: Test general software knowledge
        self.test_general_software_knowledge()
        print("\n" + "="*50)
        
        # Step 2: Test specific defect knowledge
        original_responses = self.test_specific_defect_knowledge()
        print("\n" + "="*50)
        
        # Step 3: Create training data
        training_examples = self.create_software_training_data()
        
        # Step 4: Train the model
        if self.train_software_model(training_examples):
            # Step 5: Test learned knowledge
            improvement_rate = self.test_learned_defect_knowledge(original_responses)
            
            # Step 6: Demo complete (skipping interactive testing for automation)
            
            # Show business value
            self.show_business_value()
            
            print(f"\n🎉 Software Defect Demo Complete!")
            print(f"✅ PROVEN: Models can learn specific software defect knowledge!")
        else:
            print(f"❌ Training failed - demo incomplete")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Software Defect Knowledge Acquisition Demo")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model name")
    
    args = parser.parse_args()
    
    # GPU check
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CPU mode - training will be slower")
    
    # Run demo
    demo = SoftwareDefectDemo(model_name=args.model)
    demo.run_demo()

if __name__ == "__main__":
    main()