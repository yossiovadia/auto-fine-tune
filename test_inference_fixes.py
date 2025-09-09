#!/usr/bin/env python3
"""
Comprehensive Inference Fixes for CUDA Assertion Errors

This script tests multiple approaches to fix the "probability tensor contains inf/nan" 
CUDA assertion error that occurs after fine-tuning.

Common causes and fixes:
1. Numerical instability from aggressive training
2. FP16/BF16 precision issues  
3. Temperature/sampling parameters
4. Model weights containing NaN/Inf
5. Attention mask problems
6. Device placement issues

Usage:
    python test_inference_fixes.py
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

class InferenceFixer:
    """Test multiple approaches to fix inference issues."""
    
    def __init__(self):
        self.model_path = "results/comprehensive_full_trained_model/final_comprehensive_model"
        self.tokenizer = None
        self.model = None
        self.test_questions = [
            "How to fix AuthFlow error AF-3001?",
            "What causes PayFlow error PF-1205?",
            "How to resolve DataFlow DF-7890?"
        ]
    
    def check_model_weights(self, model):
        """Check if model weights contain NaN or Inf values."""
        print("🔍 Checking model weights for NaN/Inf values...")
        
        nan_count = 0
        inf_count = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_count += torch.isnan(param).sum().item()
                print(f"   ⚠️  NaN found in {name}")
            if torch.isinf(param).any():
                inf_count += torch.isinf(param).sum().item()
                print(f"   ⚠️  Inf found in {name}")
            total_params += param.numel()
        
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   📊 NaN parameters: {nan_count}")
        print(f"   📊 Inf parameters: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("   ❌ Model weights contain invalid values!")
            return False
        else:
            print("   ✅ Model weights are clean")
            return True
    
    def fix_nan_inf_weights(self, model):
        """Replace NaN/Inf values in model weights with small random values."""
        print("🔧 Fixing NaN/Inf values in model weights...")
        
        fixed_count = 0
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"   🔧 Fixing {name}")
                # Replace NaN/Inf with small random values
                mask = torch.isnan(param) | torch.isinf(param)
                param.data[mask] = torch.randn_like(param.data[mask]) * 0.01
                fixed_count += mask.sum().item()
        
        print(f"   ✅ Fixed {fixed_count} invalid parameters")
        return model
    
    def approach_1_cpu_float32(self):
        """Approach 1: CPU + Float32 (most conservative)"""
        print("\n🧪 APPROACH 1: CPU + Float32 (Conservative)")
        print("=" * 50)
        
        try:
            print("📁 Loading model on CPU with float32...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            
            # Check and fix weights
            if not self.check_model_weights(model):
                model = self.fix_nan_inf_weights(model)
            
            print("🧪 Testing generation...")
            return self.test_generation(tokenizer, model, "cpu-float32")
            
        except Exception as e:
            print(f"❌ Approach 1 failed: {e}")
            return False
    
    def approach_2_gpu_float32_conservative(self):
        """Approach 2: GPU + Float32 + Very Conservative Settings"""
        print("\n🧪 APPROACH 2: GPU + Float32 + Conservative Settings")
        print("=" * 55)
        
        try:
            print("📁 Loading model on GPU with float32...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Check and fix weights
            if not self.check_model_weights(model):
                model = self.fix_nan_inf_weights(model)
            
            print("🧪 Testing generation with conservative parameters...")
            return self.test_generation(tokenizer, model, "gpu-float32-conservative", conservative=True)
            
        except Exception as e:
            print(f"❌ Approach 2 failed: {e}")
            return False
    
    def approach_3_gradient_clipping_reload(self):
        """Approach 3: Reload base model and apply minimal changes"""
        print("\n🧪 APPROACH 3: Reload Base Model + Minimal Fine-tuning")
        print("=" * 55)
        
        try:
            print("📁 Loading original base model...")
            tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            base_model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("🧪 Testing base model generation first...")
            success = self.test_generation(tokenizer, base_model, "base-model")
            
            if success:
                print("✅ Base model works - the issue is from fine-tuning")
            else:
                print("❌ Even base model fails - deeper issue")
                
            return success
            
        except Exception as e:
            print(f"❌ Approach 3 failed: {e}")
            return False
    
    def approach_4_debug_mode(self):
        """Approach 4: Debug mode with CUDA_LAUNCH_BLOCKING"""
        print("\n🧪 APPROACH 4: Debug Mode with CUDA_LAUNCH_BLOCKING")
        print("=" * 52)
        
        try:
            import os
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            
            print("📁 Loading model with debug mode enabled...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Check and fix weights
            if not self.check_model_weights(model):
                model = self.fix_nan_inf_weights(model)
            
            print("🧪 Testing generation with debug mode...")
            return self.test_generation(tokenizer, model, "debug-mode", debug=True)
            
        except Exception as e:
            print(f"❌ Approach 4 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_generation(self, tokenizer, model, approach_name, conservative=False, debug=False):
        """Test model generation with different parameter sets."""
        print(f"🎯 Testing {approach_name} generation...")
        
        # Define multiple parameter sets to try
        if conservative:
            param_sets = [
                {
                    "name": "greedy-deterministic",
                    "params": {
                        "max_new_tokens": 50,
                        "do_sample": False,
                        "temperature": 1.0,
                        "pad_token_id": tokenizer.eos_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                },
                {
                    "name": "high-temp-stable",
                    "params": {
                        "max_new_tokens": 50,
                        "do_sample": True,
                        "temperature": 2.0,  # High temperature for stability
                        "top_p": 0.95,
                        "pad_token_id": tokenizer.eos_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                }
            ]
        else:
            param_sets = [
                {
                    "name": "ultra-conservative",
                    "params": {
                        "max_new_tokens": 30,
                        "do_sample": False,
                        "pad_token_id": tokenizer.eos_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "use_cache": False,
                    }
                }
            ]
        
        success_count = 0
        
        for param_set in param_sets:
            print(f"\n   🔧 Trying {param_set['name']} parameters...")
            
            for i, question in enumerate(self.test_questions[:1], 1):  # Test only first question
                try:
                    prompt = f"Question: {question}\nAnswer:"
                    
                    print(f"      ❓ {question}")
                    
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # Check input tensors for NaN/Inf
                    for key, tensor in inputs.items():
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            print(f"      ⚠️  Invalid values in input {key}")
                            return False
                    
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **param_set["params"])
                    
                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = generated[len(prompt):].strip()
                    
                    if response and len(response) > 5:
                        print(f"      ✅ SUCCESS: {response[:100]}...")
                        success_count += 1
                    else:
                        print(f"      ⚠️  Empty or short response: '{response}'")
                        
                except Exception as e:
                    print(f"      ❌ Generation failed: {str(e)[:100]}...")
                    if debug:
                        import traceback
                        traceback.print_exc()
        
        if success_count > 0:
            print(f"   🎉 {approach_name} SUCCESS: {success_count} successful generations!")
            return True
        else:
            print(f"   ❌ {approach_name} FAILED: No successful generations")
            return False
    
    def run_all_approaches(self):
        """Run all approaches to find a working solution."""
        print("🔬 COMPREHENSIVE INFERENCE FIXING TEST")
        print("=" * 50)
        print("Testing multiple approaches to fix CUDA assertion error...")
        
        approaches = [
            ("CPU Float32", self.approach_1_cpu_float32),
            ("GPU Float32 Conservative", self.approach_2_gpu_float32_conservative),
            ("Base Model Test", self.approach_3_gradient_clipping_reload),
            ("Debug Mode", self.approach_4_debug_mode),
        ]
        
        working_approaches = []
        
        for name, approach_func in approaches:
            try:
                success = approach_func()
                if success:
                    working_approaches.append(name)
                    print(f"\n✅ {name} WORKS!")
                else:
                    print(f"\n❌ {name} failed")
            except Exception as e:
                print(f"\n💥 {name} crashed: {e}")
        
        print(f"\n📊 FINAL RESULTS:")
        print("=" * 30)
        if working_approaches:
            print(f"✅ Working approaches: {', '.join(working_approaches)}")
            print("🎯 Recommendation: Use the first working approach for inference")
        else:
            print("❌ No approaches worked - model may need retraining with different parameters")
            print("💡 Consider using more conservative training settings or LoRA instead")
        
        return len(working_approaches) > 0

def main():
    """Main entry point."""
    print("🚀 Testing Inference Fixes for CUDA Assertion Error")
    print("=" * 60)
    
    # Check if trained model exists
    import os
    model_path = "results/comprehensive_full_trained_model/final_comprehensive_model"
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at: {model_path}")
        print("   Please run the comprehensive training demo first.")
        return
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 No GPU detected - will test CPU approaches")
    
    # Run comprehensive tests
    fixer = InferenceFixer()
    success = fixer.run_all_approaches()
    
    if success:
        print("\n🎉 SOLUTION FOUND! Check the working approaches above.")
    else:
        print("\n🔄 No immediate solution found - may need training parameter adjustments.")

if __name__ == "__main__":
    main()