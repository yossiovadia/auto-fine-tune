#!/usr/bin/env python3
"""
Setup script for Knowledge Acquisition Demos

This script checks dependencies and sets up the environment for running
the interactive knowledge acquisition demonstrations.
"""

import subprocess
import sys
import importlib
import torch
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip."""
    print(f"📦 Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def check_gpu():
    """Check GPU availability and info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU Available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            print("⚠️  Warning: Less than 8GB GPU memory - consider using smaller models")
        else:
            print("✅ GPU memory looks good for training!")
        
        return True
    else:
        print("❌ No GPU detected")
        print("   Training will be very slow on CPU")
        print("   Consider running on a system with GPU")
        return False

def setup_environment():
    """Set up the environment for knowledge demos."""
    print("🔧 Knowledge Acquisition Demo Setup")
    print("=" * 40)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Required packages
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"), 
        ("datasets", "datasets"),
        ("peft", "peft"),
        ("accelerate", "accelerate")
    ]
    
    missing_packages = []
    
    print(f"\n📋 Checking required packages...")
    for package, import_name in packages:
        if check_package(package, import_name):
            print(f"   ✅ {package}")
        else:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing missing packages...")
        for package in missing_packages:
            if not install_package(package):
                print(f"❌ Setup failed - could not install {package}")
                return False
        print("✅ All packages installed!")
    
    # Check GPU
    print(f"\n🎮 GPU Check:")
    has_gpu = check_gpu()
    
    # Check disk space
    output_dir = Path("demo_output")
    try:
        import shutil
        free_space = shutil.disk_usage(Path.cwd()).free / 1e9
        print(f"\n💽 Available disk space: {free_space:.1f} GB")
        
        if free_space < 5:
            print("⚠️  Warning: Less than 5GB free space")
            print("   Model downloads and training may fail")
        else:
            print("✅ Sufficient disk space")
    except:
        print("⚠️  Could not check disk space")
    
    # Create output directories
    directories = ["demo_output", "simple_demo_output", "demo_training_output"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print(f"\n✅ Setup complete!")
    return True

def show_demo_options():
    """Show available demo options."""
    print("\n🎬 Available Knowledge Acquisition Demos:")
    print("=" * 45)
    
    print("1. 🎮 Interactive Knowledge Demo")
    print("   Command: python interactive_knowledge_demo.py")
    print("   Features: Full menu-driven experience")
    print("   - View knowledge domains")
    print("   - Test untrained model") 
    print("   - Live training with progress")
    print("   - Test trained model")
    print("   - Side-by-side comparisons")
    
    print("\n2. ⚡ Simple Knowledge Demo")
    print("   Command: python simple_knowledge_demo.py")
    print("   Features: Streamlined experience")
    print("   - Ask questions → 'I don't know'")
    print("   - Train model → Live progress")
    print("   - Ask same questions → Detailed answers")
    
    print("\n3. 📊 Complete POC Pipeline")
    print("   Command: python run_complete_poc.py")
    print("   Features: Full automated evaluation")
    print("   - Quantitative before/after testing")
    print("   - Novel question evaluation")
    print("   - Comprehensive reports")
    
    print("\n💡 Recommendation:")
    print("   Start with the Simple Demo for the clearest demonstration")
    print("   of knowledge acquisition through fine-tuning!")

def main():
    """Main setup function."""
    print("🚀 Knowledge Acquisition Demo Setup")
    
    if not setup_environment():
        print("\n❌ Setup failed!")
        print("Please resolve the issues above and try again.")
        return
    
    show_demo_options()
    
    print(f"\n🎯 Ready to demonstrate knowledge acquisition!")
    print(f"   Choose a demo above to see how fine-tuning teaches models new information.")
    
    # Ask which demo to run
    choice = input(f"\nRun a demo now? (1=Interactive, 2=Simple, 3=Complete, N=No): ").strip()
    
    if choice == "1":
        print("\n🎮 Starting Interactive Demo...")
        subprocess.run([sys.executable, "interactive_knowledge_demo.py"])
    elif choice == "2":
        print("\n⚡ Starting Simple Demo...")
        subprocess.run([sys.executable, "simple_knowledge_demo.py"])
    elif choice == "3":
        print("\n📊 Starting Complete POC...")
        subprocess.run([sys.executable, "run_complete_poc.py"])
    else:
        print("\n👍 Setup complete! Run any demo when you're ready.")

if __name__ == "__main__":
    main()