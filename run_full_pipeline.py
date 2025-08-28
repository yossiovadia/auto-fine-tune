#!/usr/bin/env python3
"""
Full pipeline orchestrator for adaptive fine-tuning.
Runs all data collection, processing, and preparation steps before training.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import json

def run_command(cmd: str, description: str, timeout: Optional[int] = None) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n🔄 {description}")
    print(f"   Command: {cmd}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed in {elapsed:.1f}s")
            return True, result.stdout
        else:
            print(f"❌ {description} failed after {elapsed:.1f}s")
            print(f"   Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"💥 {description} crashed: {e}")
        return False, str(e)

def check_prerequisites() -> bool:
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check virtual environment
    if not Path(".venv").exists():
        print("❌ Virtual environment not found (.venv)")
        return False
    
    # Check GitHub CLI
    success, _ = run_command("gh auth status", "GitHub CLI authentication check")
    if not success:
        print("❌ GitHub CLI not authenticated. Run: gh auth login")
        return False
    
    print("✅ Prerequisites met")
    return True

def get_existing_data_status() -> dict:
    """Check what data already exists."""
    status = {}
    
    files_to_check = {
        "github_issues": "data/vllm_full_dataset.json",
        "vllm_repo": "data/vllm",
        "code_examples": "data/codebase/vllm_code_examples.jsonl",
        "improved_qa": "data/training_datasets/period_2/improved_qa_examples.jsonl",
        "final_dataset": "data/training_datasets/period_2/code_aware_dataset.jsonl"
    }
    
    for name, path in files_to_check.items():
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_file():
                size_mb = path_obj.stat().st_size / (1024 * 1024)
                status[name] = {"exists": True, "size_mb": size_mb}
            else:
                status[name] = {"exists": True, "size_mb": 0}  # Directory
        else:
            status[name] = {"exists": False, "size_mb": 0}
    
    return status

def run_pipeline(max_issues: int = 500, force_refresh: bool = False, skip_training: bool = False):
    """Run the full pipeline."""
    
    print("🚀 Adaptive Fine-Tuning Full Pipeline")
    print("=" * 60)
    
    if not check_prerequisites():
        return False
    
    # Check existing data
    existing_data = get_existing_data_status()
    
    print(f"\n📊 Existing Data Status:")
    for name, info in existing_data.items():
        status = f"✅ ({info['size_mb']:.1f}MB)" if info['exists'] else "❌"
        print(f"   {name}: {status}")
    
    # Step 1: Collect GitHub Issues
    if not existing_data["github_issues"]["exists"] or force_refresh:
        print(f"\n📥 Step 1/5: Collecting {max_issues} GitHub issues...")
        success, _ = run_command(
            f"python collect_vllm_issues_gh.py --max-issues {max_issues}",
            "GitHub issues collection",
            timeout=600  # 10 minutes
        )
        if not success:
            print("❌ Pipeline failed at GitHub collection")
            return False
    else:
        print(f"\n⏭️  Step 1/5: Using existing GitHub issues ({existing_data['github_issues']['size_mb']:.1f}MB)")
    
    # Step 2: Ingest vLLM Codebase
    if not existing_data["code_examples"]["exists"] or force_refresh:
        print(f"\n💻 Step 2/5: Ingesting vLLM codebase...")
        success, _ = run_command(
            "python ingest_vllm_codebase.py",
            "vLLM codebase ingestion",
            timeout=300  # 5 minutes
        )
        if not success:
            print("❌ Pipeline failed at codebase ingestion")
            return False
    else:
        print(f"\n⏭️  Step 2/5: Using existing codebase analysis")
    
    # Step 3: Improve Dataset Quality
    if not existing_data["improved_qa"]["exists"] or force_refresh:
        print(f"\n🔧 Step 3/5: Improving dataset quality...")
        success, _ = run_command(
            "python improve_dataset.py",
            "Dataset quality improvement",
            timeout=120  # 2 minutes
        )
        if not success:
            print("❌ Pipeline failed at dataset improvement")
            return False
    else:
        print(f"\n⏭️  Step 3/5: Using existing improved dataset")
    
    # Step 4: Create Enhanced Dataset
    print(f"\n🔗 Step 4/5: Creating code-aware dataset...")
    success, _ = run_command(
        "python create_enhanced_dataset.py",
        "Enhanced dataset creation",
        timeout=120  # 2 minutes
    )
    if not success:
        print("❌ Pipeline failed at enhanced dataset creation")
        return False
    
    # Step 5: Training (optional)
    if not skip_training:
        print(f"\n🤖 Step 5/5: Training model...")
        print("⚠️  This will take 20-40 minutes on M4")
        
        user_input = input("Proceed with training? [y/N]: ").lower().strip()
        if user_input in ['y', 'yes']:
            success, _ = run_command(
                "python train_real_m4.py",
                "Model training",
                timeout=3600  # 1 hour
            )
            if not success:
                print("❌ Pipeline failed at model training")
                return False
        else:
            print("⏭️  Skipping training. Run manually: python train_real_m4.py")
    else:
        print(f"\n⏭️  Step 5/5: Training skipped (use --skip-training)")
    
    # Show final status
    print(f"\n🎉 Pipeline completed successfully!")
    
    # Run status check
    print(f"\n📊 Final Status:")
    run_command("python check_status.py", "Status check")
    
    return True

def main():
    """Main function with command line options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full adaptive fine-tuning pipeline")
    parser.add_argument("--max-issues", type=int, default=500, 
                       help="Maximum GitHub issues to collect (default: 500)")
    parser.add_argument("--force-refresh", action="store_true",
                       help="Force refresh of all existing data")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip the training step")
    
    args = parser.parse_args()
    
    print(f"Parameters:")
    print(f"  Max issues: {args.max_issues}")
    print(f"  Force refresh: {args.force_refresh}")
    print(f"  Skip training: {args.skip_training}")
    
    success = run_pipeline(
        max_issues=args.max_issues,
        force_refresh=args.force_refresh,
        skip_training=args.skip_training
    )
    
    if success:
        print(f"\n✅ Ready for training with: python train_real_m4.py")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()