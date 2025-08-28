#!/usr/bin/env python3
"""
Actually train a model on M4 Mac using MPS acceleration.
Optimized for Apple Silicon - should take 15-30 minutes.
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
    """Check what device we can use on M4."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon MPS acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA acceleration!")
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
        print("📊 Using original dataset")
    
    if not qa_file.exists():
        print("❌ No training data found!")
        print("Run: python src/data/vllm_dataset_converter.py --input data/vllm_full_dataset.json --output data/training_datasets --types qa")
        return []
    
    examples = []
    with open(qa_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"📚 Loaded {len(examples)} training examples")
    return examples

def prepare_dataset(examples, tokenizer, max_length=256):
    """Convert examples to tokenized dataset for M4 training."""
    
    def format_example(example):
        instruction = example['instruction']
        output = example['output']
        
        # Clean and validate the data
        instruction = instruction.strip()
        output = output.strip()
        
        # Filter out low quality examples
        if len(instruction) < 10 or len(output) < 20:
            return None
            
        # Truncate very long outputs but keep them meaningful
        if len(output) > 150:
            output = output[:150] + "..."
        
        # Use Qwen's instruction format
        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        return {'text': text}
    
    # Format examples and filter out bad ones
    formatted = []
    for ex in examples:
        if len(ex.get('output', '')) > 20:  # Only include examples with meaningful outputs
            formatted_ex = format_example(ex)
            if formatted_ex is not None:  # Only add if format_example didn't filter it out
                formatted.append(formatted_ex)
    
    dataset = Dataset.from_list(formatted)
    print(f"📊 Using {len(dataset)} quality examples")
    
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_vllm_assistant():
    """Train a vLLM assistant model on M4 Mac."""
    print("🤖 Training Code-Aware vLLM Assistant (Qwen2.5-Coder-7B) on Apple Silicon M4")
    print("⏱️  Expected time: 20-40 minutes (larger model)")
    print("=" * 50)
    
    start_time = time.time()
    device = check_device()
    
    # Load data
    examples = load_training_data()
    if len(examples) < 10:
        print("❌ Not enough training data")
        return None
    
    # Use Qwen2.5-Coder-7B - code-specialized model for better programming understanding
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    print(f"📥 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use eager attention for better compatibility and disable warnings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.float32  # Use float32 for MPS compatibility
    )
    
    # Add padding token for Qwen
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset - use all examples since we have quality data now
    print("📊 Preparing dataset...")
    dataset = prepare_dataset(examples, tokenizer)  # Use all examples
    
    if len(dataset) < 5:
        print("❌ Not enough quality examples")
        return None
    
    # Split into train/eval
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset))) if len(dataset) > train_size else None
    
    print(f"📚 Train examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"🧪 Eval examples: {len(eval_dataset)}")
    
    # Add LoRA for efficient training
    print("🔧 Adding LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Good balance for M4
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Qwen uses similar architecture
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments optimized for M4
    output_dir = Path("models/vllm_qwen_assistant_m4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=2,  # Reduce epochs to prevent overfitting
        per_device_train_batch_size=2,  # Smaller for 7B model on M4
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Maintain effective batch size of 16
        learning_rate=1e-4,  # Lower learning rate for stability
        weight_decay=0.01,  # Add weight decay
        warmup_steps=10,
        logging_steps=5,
        eval_steps=30 if eval_dataset else None,
        save_steps=30,  # Must be same as eval_steps for load_best_model_at_end
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,  # No wandb
        fp16=False,  # MPS doesn't support fp16
        gradient_checkpointing=False,  # Disable for LoRA compatibility
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    
    # Train!
    print("🏃‍♂️ Training started...")
    print("💻 Using Apple Silicon acceleration")
    
    try:
        trainer.train()
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        elapsed = time.time() - start_time
        print(f"✅ Training completed in {elapsed/60:.1f} minutes!")
        print(f"💾 Model saved to: {output_dir}")
        
        return str(output_dir), tokenizer, model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def test_trained_model(model_path, tokenizer, model):
    """Test the trained model on vLLM questions."""
    print(f"\n🧪 Testing trained vLLM assistant...")
    print("=" * 50)
    
    test_questions = [
        "How to fix CUDA out of memory error in vLLM?",
        "How to run Llama model with vLLM?",
        "ValueError: unsupported LoRA weight error?", 
        "TPU compilation fails with vLLM?",
        "How to configure vLLM for production?"
    ]
    
    print("🎯 vLLM Assistant Answers:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize and move to correct device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) if v is not None else v for k, v in inputs.items()}
        
        # Generate answer
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
        
        print(f"\n🔵 {i}. {question}")
        print(f"🤖 {answer}")
        print("─" * 60)
    
    print(f"\n🎉 Your model is now trained and answering vLLM questions!")
    return True

if __name__ == "__main__":
    print("🎯 Real vLLM Assistant Training for M4 Mac")
    print("This will train an actual model that can answer vLLM questions!")
    print("=" * 60)
    
    # Check if we're in the right directory and have data
    if not Path("data/training_datasets/period_2/qa_examples.jsonl").exists():
        print("❌ Training data not found!")
        print("\n📋 Please run these commands first:")
        print("1. python collect_vllm_issues_gh.py --max-issues 500")
        print("2. python src/data/vllm_dataset_converter.py --input data/vllm_full_dataset.json --output data/training_datasets --types qa")
        exit(1)
    
    # Train the model
    result = train_vllm_assistant()
    
    if result:
        model_path, tokenizer, model = result
        
        # Test the trained model
        test_trained_model(model_path, tokenizer, model)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Trained model: {model_path}")
        print(f"🤖 Your vLLM assistant is ready to help!")
        print(f"⚡ Powered by Apple Silicon M4")
        
    else:
        print("❌ Training failed. Check the logs above.")