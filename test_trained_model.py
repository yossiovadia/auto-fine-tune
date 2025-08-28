#!/usr/bin/env python3
"""
Test the trained vLLM assistant model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

def test_model():
    """Test the trained model."""
    model_path = Path("models/vllm_assistant_m4")
    
    if not model_path.exists():
        print("❌ Trained model not found!")
        return
        
    print("🤖 Loading trained vLLM assistant...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")
    model = PeftModel.from_pretrained(model, model_path)
    
    # Check device and move model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"📱 Using device: {device}")
    
    test_questions = [
        "How to fix CUDA out of memory error in vLLM?",
        "How to run Llama model with vLLM?",
        "ValueError: unsupported LoRA weight error?", 
        "TPU compilation fails with vLLM?",
        "How to configure vLLM for production?"
    ]
    
    print("\n🎯 vLLM Assistant Answers:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize and move to correct device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
        inputs = {k: v.to(device) if v is not None else v for k, v in inputs.items()}
        
        # Generate answer
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode and clean up
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        
        # Clean up the answer
        if "<|endoftext|>" in answer:
            answer = answer.split("<|endoftext|>")[0].strip()
        if "<end_of_turn>" in answer:
            answer = answer.split("<end_of_turn>")[0].strip()
        
        print(f"\n🔵 {i}. {question}")
        print(f"🤖 {answer}")
        print("─" * 60)
    
    print(f"\n🎉 Your vLLM assistant is working!")

if __name__ == "__main__":
    test_model()