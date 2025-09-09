#!/usr/bin/env python3
"""
Fixed Test Script for Trained Software Defect Model

This script handles CUDA assertion errors and probability instability issues
that can occur after intensive full parameter training.

Usage:
    python test_trained_model_fixed.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_trained_model():
    """Load the trained model from the results directory."""
    model_path = "results/comprehensive_full_trained_model/final_comprehensive_model"
    
    print(f"🔄 Loading trained model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with more conservative settings to avoid CUDA issues
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,  # Use float32 instead of float16 for stability
        device_map="cpu",     # Force CPU to avoid CUDA assertion errors
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully on CPU!")
    return tokenizer, model

def ask_question_safe(tokenizer, model, question, max_tokens=100):
    """Ask a question with safe generation parameters."""
    prompt = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    
    # Use very conservative generation settings
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1.0,              # Higher temperature for stability
                do_sample=False,              # Use greedy decoding (deterministic)
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,       # No repetition penalty
                top_p=1.0,                    # No nucleus sampling
                num_beams=1,                  # No beam search
                early_stopping=True
            )
        except Exception as e:
            print(f"⚠️  Generation error: {e}")
            return f"[Generation failed: {str(e)}]"
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated[len(prompt):].strip()
    
    # Clean response
    lines = response.split('\n')
    response = lines[0].strip() if lines else ""
    
    if len(response) > 200:
        response = response[:200] + "..."
    
    return response if response else "[Empty response]"

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
    
    print("\n🧠 Testing Learned Knowledge (Safe Mode)")
    print("=" * 50)
    print("Running on CPU with conservative settings to avoid CUDA errors")
    
    successful_tests = 0
    learning_detected = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ❓ {question}")
        response = ask_question_safe(tokenizer, model, question)
        print(f"   🤖 TRAINED MODEL: {response}")
        
        if not response.startswith("[") and len(response) > 10:
            successful_tests += 1
            
            # Check if response contains learned knowledge
            response_lower = response.lower()
            learned_indicators = []
            
            if any(code in response_lower for code in ['af-', 'pf-', 'df-']):
                learned_indicators.append("✅ Error codes")
            if any(service in response_lower for service in ['authflow', 'payflow', 'dataflow']):
                learned_indicators.append("✅ Service names")
            if any(config in response_lower for config in ['timeout', '.yml', 'enabled=true', 'streaming_mode']):
                learned_indicators.append("✅ Configurations")
            if any(tech in response_lower for tech in ['jwt', 'redis', 'webhook', 'biometric']):
                learned_indicators.append("✅ Technical terms")
            
            if learned_indicators:
                learning_detected += 1
                print(f"   📈 Learning: {', '.join(learned_indicators)}")
            else:
                print(f"   📊 Generic response (no specific learned knowledge)")
        else:
            print(f"   ❌ Generation failed or empty response")
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Successful generations: {successful_tests}/{len(test_questions)}")
    print(f"   Learning detected: {learning_detected}/{len(test_questions)}")
    
    if successful_tests == 0:
        print("   ⚠️  Model appears to have generation issues - may need retraining")
    elif learning_detected > 0:
        print("   🎉 Some knowledge acquisition detected!")
    else:
        print("   📊 Model generates responses but no clear learning detected")

def simple_test():
    """Simple test with just one question."""
    print("\n🚀 Simple Test Mode")
    print("=" * 20)
    
    try:
        tokenizer, model = load_trained_model()
        
        question = "How to fix AuthFlow error AF-3001?"
        print(f"❓ Testing: {question}")
        
        response = ask_question_safe(tokenizer, model, question)
        print(f"🤖 Response: {response}")
        
        if "af-3001" in response.lower() or "authflow" in response.lower():
            print("✅ Model shows some knowledge of the error code!")
        elif len(response) > 20 and not response.startswith("["):
            print("📊 Model generates response but no specific learning detected")
        else:
            print("❌ Model has generation issues")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Trained Software Defect Model (Fixed Version)")
    print("=" * 60)
    
    # Check available hardware
    if torch.cuda.is_available():
        print(f"🎮 GPU Available: {torch.cuda.get_device_name()}")
        print("⚠️  Using CPU due to CUDA assertion errors during inference")
    else:
        print("💻 Running on CPU")
    
    print("\nChoose test mode:")
    print("1. Full test (8 questions)")
    print("2. Simple test (1 question)")
    
    try:
        choice = input("Enter choice (1 or 2, default=2): ").strip()
        if choice == "1":
            test_learned_knowledge()
        else:
            simple_test()
            
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user")
    except FileNotFoundError:
        print("❌ Trained model not found!")
        print("   Expected path: results/comprehensive_full_trained_model/final_comprehensive_model")
        print("   Make sure the training completed successfully.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running: python test_trained_model_fixed.py")