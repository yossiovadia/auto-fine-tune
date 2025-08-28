#!/usr/bin/env python3
"""
Check current status of the adaptive fine-tuning POC.
"""

from pathlib import Path
import json

def check_file(file_path: str, description: str) -> bool:
    """Check if a file exists and show its status."""
    path = Path(file_path)
    if path.exists():
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {description}: {file_path} ({size_mb:.1f}MB)")
        else:
            print(f"✅ {description}: {file_path} (directory)")
        return True
    else:
        print(f"❌ {description}: {file_path} (missing)")
        return False

def count_jsonl_lines(file_path: str) -> int:
    """Count lines in a JSONL file."""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 0

def main():
    print("📊 Adaptive Fine-Tuning POC Status Check")
    print("=" * 50)
    
    # Core dependencies
    print("\n🔧 Dependencies:")
    check_file(".venv", "Python virtual environment")
    check_file("requirements_m4.txt", "Requirements file")
    
    # Data collection
    print("\n📥 Data Collection:")
    vllm_dataset_exists = check_file("data/vllm_full_dataset.json", "GitHub issues dataset")
    vllm_repo_exists = check_file("data/vllm", "vLLM repository clone")
    
    # Processed datasets
    print("\n📊 Processed Datasets:")
    qa_examples = count_jsonl_lines("data/training_datasets/period_2/qa_examples.jsonl")
    improved_examples = count_jsonl_lines("data/training_datasets/period_2/improved_qa_examples.jsonl") 
    code_examples = count_jsonl_lines("data/codebase/vllm_code_examples.jsonl")
    final_examples = count_jsonl_lines("data/training_datasets/period_2/code_aware_dataset.jsonl")
    
    print(f"  Original Q&A examples: {qa_examples}")
    print(f"  Improved Q&A examples: {improved_examples}")
    print(f"  Code-based examples: {code_examples}")
    print(f"  Final training dataset: {final_examples}")
    
    # Models
    print("\n🤖 Models:")
    gemma_model_exists = check_file("models/vllm_assistant_m4", "Gemma-3-270m model")
    qwen_model_exists = check_file("models/vllm_qwen_assistant_m4", "Qwen2.5-Coder-7B model")
    
    # Incremental system
    print("\n🔄 Incremental System:")
    check_file("data/last_update.json", "Last update tracking")
    
    # Show last update info if available
    last_update_file = Path("data/last_update.json")
    if last_update_file.exists():
        try:
            with open(last_update_file, 'r') as f:
                update_info = json.load(f)
            print(f"  Last commit: {update_info.get('last_commit', 'N/A')[:8]}...")
            print(f"  Changes since: {update_info.get('changes_since_last', 0)}")
            print(f"  Last update: {update_info.get('last_update_time', 'N/A')}")
        except:
            print("  Could not read update info")
    
    # Core scripts
    print("\n📝 Core Scripts:")
    scripts = [
        ("collect_vllm_issues_gh.py", "GitHub data collection"),
        ("ingest_vllm_codebase.py", "Codebase analysis"),
        ("improve_dataset.py", "Dataset quality improvement"),
        ("create_enhanced_dataset.py", "Dataset fusion"),
        ("train_real_m4.py", "Model training"),
        ("incremental_update_system.py", "Delta detection")
    ]
    
    for script, desc in scripts:
        check_file(script, desc)
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"  Data ready: {'✅' if vllm_dataset_exists and vllm_repo_exists else '❌'}")
    print(f"  Training dataset: {final_examples} examples")
    print(f"  Models trained: {'Qwen' if qwen_model_exists else 'Gemma' if gemma_model_exists else 'None'}")
    print(f"  Status: {'POC Complete' if final_examples > 200 else 'In Progress'}")

if __name__ == "__main__":
    main()