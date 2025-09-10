#!/usr/bin/env python3
"""
Model Comparison Tool - Side-by-Side Before/After Testing

This tool loads both the original baseline model and the trained model
so you can ask the same question to both simultaneously and see the 
dramatic difference in knowledge acquisition.

Usage:
    python model_comparison_tool.py
    
Then type questions and see both models respond side-by-side.
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

class ModelComparison:
    """Compare baseline vs trained model responses side-by-side."""
    
    def __init__(self):
        self.base_model_name = "gpt2-medium"  # Use GPT2-medium as baseline
        self.trained_model_path = "../results/final_working_model/final_working_trained_model"
        
        self.baseline_tokenizer = None
        self.baseline_model = None
        self.trained_tokenizer = None
        self.trained_model = None
        
        # Predefined test questions for quick testing
        self.test_questions = [
            "How to fix AuthFlow error AF-3001?",
            "What causes PayFlow error PF-1205?", 
            "How to resolve DataFlow DF-7890?",
            "What is AuthFlow error AF-6001?",
            "How to fix PayFlow PF-4001?",
            "What causes DataFlow DF-1001?",
            "How to enable AuthFlow's Biometric Authentication?",
            "How to configure PayFlow's Smart Fraud Detection?",
        ]
    
    def load_models(self):
        """Load both baseline and trained models."""
        print("🤖 Loading Models for Side-by-Side Comparison")
        print("=" * 50)
        
        try:
            # Load baseline model
            print("📥 Loading baseline TinyLlama model...")
            self.baseline_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, trust_remote_code=True, padding_side="right"
            )
            if self.baseline_tokenizer.pad_token is None:
                self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token
                
            self.baseline_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ Baseline model loaded successfully!")
            
            # Load trained model
            print("📥 Loading trained model...")
            self.trained_tokenizer = AutoTokenizer.from_pretrained(
                self.trained_model_path, trust_remote_code=True, padding_side="right"
            )
            if self.trained_tokenizer.pad_token is None:
                self.trained_tokenizer.pad_token = self.trained_tokenizer.eos_token
                
            self.trained_model = AutoModelForCausalLM.from_pretrained(
                self.trained_model_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ Trained model loaded successfully!")
            
            print("🎯 Both models ready for comparison!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Make sure the trained model exists and training completed successfully")
            return False
    
    def ask_model(self, question: str, model, tokenizer, model_name: str, max_tokens: int = 1000) -> str:
        """Ask a question to a specific model."""
        prompt = f"Question: {question}\\nAnswer:"
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Deterministic for fair comparison
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(prompt):].strip()
            
            # Clean response but preserve full content - only split on actual newlines, not truncate
            # Remove any unwanted newlines but keep the full response
            response = response.replace('\\n', ' ').strip()
            # NO truncation at all - show complete responses
                
            return response if response else "[Empty response]"
            
        except Exception as e:
            return f"[Error: {str(e)[:50]}...]"
    
    def compare_responses(self, question: str):
        """Compare responses from both models side-by-side."""
        print(f"\\n❓ QUESTION: {question}")
        print("=" * 100)
        
        # Get responses from both models
        baseline_response = self.ask_model(question, self.baseline_model, self.baseline_tokenizer, "Baseline")
        trained_response = self.ask_model(question, self.trained_model, self.trained_tokenizer, "Trained")
        
        # Display complete responses without any truncation - force full display
        print(f"🔴 BASELINE (Original GPT2-medium):")
        # Force full response display by explicitly showing length and content
        print(f"   Length: {len(baseline_response)} chars")
        print(f"   {baseline_response}")
        print()
        
        print(f"🟢 TRAINED (After Knowledge Acquisition):")
        # Force full response display by explicitly showing length and content  
        print(f"   Length: {len(trained_response)} chars")
        print(f"   {trained_response}")
        print()
        
        # Simple analysis
        baseline_lower = baseline_response.lower()
        trained_lower = trained_response.lower()
        
        print("📊 ANALYSIS:")
        
        # Check for specific knowledge indicators
        knowledge_indicators = [
            "timeout", "jwt", "webhook", "saml", "paypal", "memory leak", 
            "streaming", "biometric", "fraud", "detection", "authflow", "payflow", "dataflow"
        ]
        
        baseline_knowledge = sum(1 for indicator in knowledge_indicators if indicator in baseline_lower)
        trained_knowledge = sum(1 for indicator in knowledge_indicators if indicator in trained_lower)
        
        if trained_knowledge > baseline_knowledge:
            print(f"   ✅ IMPROVEMENT: Trained model shows {trained_knowledge - baseline_knowledge} more technical terms")
        elif len(trained_response) > len(baseline_response) * 1.2:
            print("   📈 IMPROVEMENT: Trained model provides more detailed response")
        elif trained_response != baseline_response:
            print("   🔄 CHANGE: Different response after training")
        else:
            print("   📊 SIMILAR: Responses are similar")
        
        print("   " + "─" * 60)
    
    def run_predefined_tests(self):
        """Run all predefined test questions for comprehensive comparison."""
        print("\\n🧪 RUNNING COMPREHENSIVE COMPARISON TESTS")
        print("=" * 50)
        print("Testing all predefined software defect questions...")
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\\n[TEST {i}/{len(self.test_questions)}]")
            self.compare_responses(question)
            time.sleep(0.5)  # Brief pause between tests
        
        print("\\n🎉 COMPREHENSIVE TESTING COMPLETE!")
        print("📊 Review the responses above to see knowledge acquisition results")
    
    def interactive_mode(self):
        """Interactive mode for custom questions."""
        print("\\n🎮 INTERACTIVE COMPARISON MODE")
        print("=" * 40)
        print("Type questions to compare both models side-by-side")
        print("Commands:")
        print("  'test' - Run all predefined tests")
        print("  'quit' - Exit")
        print("  Or type any question to compare models")
        
        while True:
            try:
                question = input("\\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif question.lower() == 'test':
                    self.run_predefined_tests()
                elif question:
                    self.compare_responses(question)
                else:
                    print("Please enter a question or command.")
                    
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def run(self):
        """Main entry point."""
        print("🔬 MODEL COMPARISON TOOL")
        print("=" * 30)
        print("Compare baseline vs trained model responses side-by-side!")
        
        if not self.load_models():
            return
        
        print("\\n🎯 What would you like to do?")
        print("1. Run comprehensive tests (all predefined questions)")
        print("2. Interactive mode (ask custom questions)")
        print("3. Both")
        
        choice = input("\\nEnter choice (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            self.run_predefined_tests()
        
        if choice in ['2', '3']:
            self.interactive_mode()
        elif choice == '1':
            print("\\n💡 Tip: Run this tool again with choice 2 for interactive testing!")

def main():
    """Main entry point."""
    # Check GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Running on CPU")
    
    comparison = ModelComparison()
    comparison.run()

if __name__ == "__main__":
    main()