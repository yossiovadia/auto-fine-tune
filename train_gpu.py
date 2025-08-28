#!/usr/bin/env python3
"""
GPU-optimized training script for RTX 4090 and other CUDA devices.
Adapted from the M4 version for maximum GPU performance.
"""

import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_device():
    """Check what device we can use - prioritize CUDA for GPU machines."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 Using CUDA acceleration!")
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_memory:.1f}GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon MPS acceleration!")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (will be slower)")
    
    print(f"Device: {device}")
    return device

def load_training_data():
    """Load our vLLM Q&A examples."""
    # Try code-aware dataset first, then improved, then original
    code_aware_file = Path("data/training_datasets/period_2/code_aware_dataset.jsonl")
    improved_file = Path("data/training_datasets/period_2/improved_qa_examples.jsonl")
    qa_file = Path("data/training_datasets/period_2/qa_examples.jsonl")
    
    if code_aware_file.exists():
        qa_file = code_aware_file
        print("🔗 Using code-aware enhanced dataset")
    elif improved_file.exists():
        qa_file = improved_file
        print("📈 Using improved quality dataset")
    else:
        print("📊 Using original Q&A dataset")
    
    examples = []
    with open(qa_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"📊 Loaded {len(examples)} training examples from {qa_file}")
    return examples

def prepare_model_and_tokenizer(device):
    """Load Qwen2.5-Coder model optimized for GPU."""
    print("🤖 Loading Qwen2.5-Coder-7B...")
    
    # Use Qwen2.5-Coder-7B - code-specialized model for better programming understanding
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with GPU optimizations
    print("📥 Loading model...")
    if device.type == "cuda":
        # GPU-specific optimizations - try flash attention first, fallback to eager
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            print("✅ Using Flash Attention 2")
        except Exception as e:
            print(f"⚠️  Flash Attention 2 not available, using eager: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            )
        model = model.to(device)
    else:
        # Fallback for non-CUDA devices
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float32 if device.type == "mps" else torch.bfloat16
        )
        model = model.to(device)
    
    print(f"✅ Model loaded on {device}")
    print(f"📊 Model parameters: {model.num_parameters():,}")
    
    return model, tokenizer

def setup_lora_config():
    """Setup LoRA configuration for efficient training."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Reduced rank for memory efficiency
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Fewer target modules for memory efficiency
    )
    return lora_config

def format_example(example, tokenizer):
    """Format example using Qwen's instruction format with proper labels."""
    instruction = example.get('instruction', '')
    output = example.get('output', '')
    
    # Use Qwen's instruction format
    text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    
    # Tokenize the full text
    result = tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=256,  # Reduced max length for memory efficiency
        return_tensors=None
    )
    
    # Create labels for causal language modeling
    # Labels should be the same as input_ids for causal LM
    result["labels"] = result["input_ids"].copy()
    
    return result

def train_model():
    """Main training function optimized for GPU."""
    print("🚀 Starting vLLM Adaptive Fine-tuning (GPU)")
    print("=" * 60)
    
    device = check_device()
    examples = load_training_data()
    
    if len(examples) < 10:
        print("❌ Not enough training examples. Need at least 10.")
        return
    
    model, tokenizer = prepare_model_and_tokenizer(device)
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    print(f"📊 Trainable parameters: {model.num_parameters():,}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Trainable (LoRA): {trainable_params:,}")
    
    # Prepare dataset
    print("📊 Preparing dataset...")
    formatted_examples = [format_example(ex, tokenizer) for ex in examples]
    dataset = Dataset.from_list(formatted_examples)
    
    # Training arguments optimized for GPU with memory efficiency
    training_args = TrainingArguments(
        output_dir="./models/vllm_qwen_assistant_gpu",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size for memory efficiency
        gradient_accumulation_steps=8,   # Increased to maintain effective batch size
        warmup_steps=10,
        logging_steps=5,
        eval_strategy="no",  # Skip evaluation for speed
        save_steps=50,
        save_total_limit=2,
        learning_rate=2e-4,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Disable for memory efficiency
        bf16=True,  # Use bfloat16 for better stability (matches model dtype)
        gradient_checkpointing=True,  # Enable for memory efficiency
        report_to=None,  # Disable wandb/tensorboard by default
        max_grad_norm=1.0,  # Add gradient clipping
        dataloader_num_workers=0  # Reduce data loading overhead
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"🎯 Training on {len(examples)} examples...")
    print(f"⏱️  Estimated time: 10-20 minutes on RTX 4090")
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time/60:.1f} minutes!")
    
    # Save model
    print("💾 Saving trained model...")
    trainer.save_model()
    tokenizer.save_pretrained("./models/vllm_qwen_assistant_gpu")
    
    print("🎉 Training complete!")
    print(f"📁 Model saved to: ./models/vllm_qwen_assistant_gpu")
    
    # Show GPU memory usage if available
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"🔥 Peak GPU memory: {memory_used:.1f}GB")

if __name__ == "__main__":
    train_model()