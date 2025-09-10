#!/usr/bin/env python3
"""
Quick Test Script for Trained Software Defect Model

This script loads the trained model and tests it on specific questions
about the software defects it learned during training.

Usage:
    python test_trained_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_trained_model():
    """Load the trained model from the results directory."""
    model_path = "results/comprehensive_full_trained_model/final_comprehensive_model"
    
    print(f"🔄 Loading trained model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Try GPU with float32 first, then fallback to CPU if needed
    try:
        print("🎮 Attempting GPU load with float32...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded successfully on GPU with float32!")
    except Exception as e:
        print(f"⚠️  GPU load failed ({str(e)[:100]}...), trying CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        print("✅ Model loaded successfully on CPU!")
    
    return tokenizer, model

def ask_question(tokenizer, model, question, max_tokens=150):
    """Ask a question to the trained model."""
    prompt = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Use more conservative generation parameters to avoid numerical issues
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,  # Higher temperature for stability
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Lower repetition penalty
            top_p=0.95,  # Higher top_p for stability
            top_k=50,    # Add top_k for additional stability
            use_cache=True
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated[len(prompt):].strip()
    
    # Clean response - take first complete sentence/line
    response = response.split('\n')[0].strip()
    if len(response) > 300:
        response = response[:300] + "..."
    
    return response

def test_learned_knowledge():
    """Test the model on specific knowledge it should have learned."""
    
    # Load the trained model
    tokenizer, model = load_trained_model()
    
    # Test questions about knowledge the model learned
    test_questions = [
        "How to fix AuthFlow error AF-3001?",
        "What causes PayFlow error PF-1205?", 
        "How to resolve DataFlow DF-7890?",
        "What is AuthFlow error AF-6001?",
        "How to fix PayFlow PF-4001?",
        "What causes DataFlow DF-1001?",
        "How to enable AuthFlow's Biometric Authentication?",
        "How to configure PayFlow's Smart Fraud Detection?",
    ]
    
    print("\n🧠 Testing Learned Knowledge")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ❓ {question}")
        response = ask_question(tokenizer, model, question)
        print(f"   🤖 TRAINED MODEL: {response}")
        
        # Check if response contains learned knowledge
        response_lower = response.lower()
        learned_indicators = []
        
        if any(code in response_lower for code in ['af-', 'pf-', 'df-']):
            learned_indicators.append("✅ Mentions error codes")
        if any(service in response_lower for service in ['authflow', 'payflow', 'dataflow']):
            learned_indicators.append("✅ Mentions services")
        if any(config in response_lower for config in ['timeout', '.yml', 'enabled=true', 'streaming_mode']):
            learned_indicators.append("✅ Specific configurations")
        if any(tech in response_lower for tech in ['jwt', 'redis', 'webhook', 'biometric']):
            learned_indicators.append("✅ Technical details")
        
        if learned_indicators:
            print(f"   📈 Learning indicators: {', '.join(learned_indicators)}")
        else:
            print(f"   ⚠️  No clear learned knowledge detected")

def interactive_test():
    """Interactive mode to ask custom questions."""
    tokenizer, model = load_trained_model()
    
    print("\n🎮 Interactive Test Mode")
    print("=" * 30)
    print("Ask questions about the software defects the model learned!")
    print("Type 'quit' to exit.")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
            
        response = ask_question(tokenizer, model, question)
        print(f"🤖 TRAINED MODEL: {response}")

if __name__ == "__main__":
    print("🚀 Testing Trained Software Defect Model")
    print("=" * 45)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    else:
        print("💻 Running on CPU")
    
    try:
        # Run automated tests
        test_learned_knowledge()
        
        # Ask if user wants interactive mode
        print(f"\n🎯 Automated testing complete!")
        choice = input("Do you want to try interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_test()
            
    except FileNotFoundError:
        print("❌ Trained model not found!")
        print("   Make sure the training completed successfully.")
        print("   Expected path: results/comprehensive_full_trained_model/final_comprehensive_model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")