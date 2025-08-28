#!/usr/bin/env python3
"""
Smart cleanup script for the adaptive fine-tuning project.
Removes unnecessary files while preserving essential ones.
"""

import os
import shutil
from pathlib import Path
from typing import List

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except:
        return 0

def safe_remove(file_path: Path, description: str) -> bool:
    """Safely remove a file with confirmation."""
    if file_path.exists():
        size_mb = get_file_size_mb(file_path)
        print(f"🗑️  Remove {description}: {file_path} ({size_mb:.1f}MB)")
        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
            return True
        except Exception as e:
            print(f"❌ Failed to remove {file_path}: {e}")
            return False
    return False

def cleanup_test_files():
    """Remove test files that are no longer needed."""
    print("🧹 Cleaning up test files...")
    
    test_files = [
        "test_webhook.py",
        "test_webhook_debug.py", 
        "test_webhook_direct.py",
        "test_low_quality.py",
        # Keep test_trained_model.py - might be useful
    ]
    
    removed_count = 0
    for test_file in test_files:
        if safe_remove(Path(test_file), "test file"):
            removed_count += 1
    
    print(f"✅ Removed {removed_count} test files")

def cleanup_sample_data():
    """Remove sample/intermediate data files."""
    print("\n🧹 Cleaning up sample data files...")
    
    # Files we can safely remove
    removable_files = [
        "data/vllm_sample.json",  # 249KB - sample data, not needed
        "data/vllm_closed_sample.json",  # 54KB - sample data, not needed
        "data/adaptive_poc_results.json",  # 1.3KB - POC results, archived
    ]
    
    removed_count = 0
    for file_path in removable_files:
        if safe_remove(Path(file_path), "sample data"):
            removed_count += 1
    
    print(f"✅ Removed {removed_count} sample data files")

def cleanup_old_datasets():
    """Clean up old/redundant dataset files."""
    print("\n🧹 Cleaning up old dataset files...")
    
    # Keep the most important datasets, remove redundant ones
    dataset_dir = Path("data/training_datasets/period_2")
    
    if dataset_dir.exists():
        # Files to potentially remove (keep the advanced ones)
        removable = [
            "qa_examples.jsonl",  # Original - superseded by improved
            "classification_examples.jsonl",  # Large file we're not using
        ]
        
        removed_count = 0
        for filename in removable:
            file_path = dataset_dir / filename
            if file_path.exists():
                size_mb = get_file_size_mb(file_path)
                print(f"📊 {filename}: {size_mb:.1f}MB")
                
                # Only remove classification file (it's large and unused)
                if filename == "classification_examples.jsonl":
                    if safe_remove(file_path, "unused classification dataset"):
                        removed_count += 1
        
        print(f"✅ Cleaned {removed_count} old dataset files")

def show_essential_files():
    """Show which files are essential and should be kept."""
    print("\n📋 Essential files being KEPT:")
    
    essential_files = [
        # Core training scripts
        "train_real_m4.py",
        "ingest_vllm_codebase.py", 
        "create_enhanced_dataset.py",
        "incremental_update_system.py",
        
        # Key datasets
        "data/training_datasets/period_2/code_aware_dataset.jsonl",
        "data/training_datasets/period_2/improved_qa_examples.jsonl",
        "data/vllm_full_dataset.json",  # Source data
        "data/last_update.json",  # For incremental updates
        
        # Codebase analysis
        "data/codebase/vllm_code_examples.jsonl",
        "data/codebase/vllm_file_analysis.json",
        
        # Configuration
        "requirements_m4.txt",
        "n8n/simple_workflow.json",
    ]
    
    total_size = 0
    for file_path in essential_files:
        path = Path(file_path)
        if path.exists():
            size_mb = get_file_size_mb(path)
            total_size += size_mb
            print(f"  ✅ {file_path} ({size_mb:.1f}MB)")
        else:
            print(f"  ⚠️  {file_path} (not found)")
    
    print(f"\n📊 Total essential files: {total_size:.1f}MB")

def main():
    print("🧹 Smart Project Cleanup")
    print("=" * 50)
    
    # Show current directory size
    total_size = sum(get_file_size_mb(Path(root) / file) 
                    for root, dirs, files in os.walk(".")
                    for file in files
                    if not any(skip in root for skip in [".venv", ".git", "node_modules"]))
    
    print(f"📊 Current project size (excluding .venv/.git): {total_size:.1f}MB")
    
    # Run cleanup
    cleanup_test_files()
    cleanup_sample_data() 
    cleanup_old_datasets()
    
    # Show what we're keeping
    show_essential_files()
    
    # Calculate savings
    new_size = sum(get_file_size_mb(Path(root) / file) 
                  for root, dirs, files in os.walk(".")
                  for file in files
                  if not any(skip in root for skip in [".venv", ".git", "node_modules"]))
    
    savings = total_size - new_size
    print(f"\n💾 Space saved: {savings:.1f}MB")
    print(f"📊 New project size: {new_size:.1f}MB")

if __name__ == "__main__":
    main()