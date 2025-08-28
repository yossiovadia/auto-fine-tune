#!/usr/bin/env python3
"""
Setup script for Adaptive Jira Defect Analysis System.

This script helps users configure and initialize the system for first use.
"""

import sys
import os
import subprocess
from pathlib import Path
import shutil
import getpass

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import get_logger, setup_logging


def main():
    """Main setup script."""
    print("🚀 Welcome to Adaptive Jira Defect Analysis System Setup")
    print("=" * 60)
    
    setup_logging(log_level='INFO')
    logger = get_logger(__name__)
    
    try:
        # Step 1: Check system requirements
        print("\n1. Checking system requirements...")
        check_system_requirements()
        
        # Step 2: Install Python dependencies
        print("\n2. Installing Python dependencies...")
        install_dependencies()
        
        # Step 3: Create directory structure
        print("\n3. Creating directory structure...")
        create_directories()
        
        # Step 4: Configure environment
        print("\n4. Configuring environment...")
        configure_environment()
        
        # Step 5: Test Jira connection
        print("\n5. Testing Jira connection...")
        test_jira_connection()
        
        # Step 6: Download base model (optional)
        print("\n6. Preparing base model...")
        prepare_base_model()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure your credentials")
        print("2. Run initial training: python scripts/train_initial_model.py --project-key YOUR_PROJECT")
        print("3. Start using the system: python scripts/run_inference.py --mode interactive")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


def check_system_requirements():
    """Check system requirements."""
    requirements = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 9:
        requirements.append(f"Python 3.9+ required (found {python_version.major}.{python_version.minor})")
    else:
        print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for GPU support
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"   ✅ CUDA available: {gpu_count} GPU(s) - {gpu_name}")
        else:
            print(f"   ⚠️  CUDA not available - will use CPU (slower training)")
    except ImportError:
        print(f"   ⚠️  PyTorch not installed yet")
    
    # Check disk space
    root_path = Path("/")
    if root_path.exists():
        stat = shutil.disk_usage(root_path)
        free_gb = stat.free // (1024**3)
        if free_gb < 10:
            requirements.append(f"At least 10GB free space required (found {free_gb}GB)")
        else:
            print(f"   ✅ Disk space: {free_gb}GB available")
    
    if requirements:
        print("   ❌ Requirements not met:")
        for req in requirements:
            print(f"      - {req}")
        raise RuntimeError("System requirements not met")


def install_dependencies():
    """Install Python dependencies."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        raise FileNotFoundError("requirements.txt not found")
    
    print(f"   Installing dependencies from {requirements_file}")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True)
        print("   ✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        print("   Try running manually: pip install -r requirements.txt")
        raise


def create_directories():
    """Create necessary directory structure."""
    base_dir = Path(__file__).parent.parent
    
    directories = [
        'data/raw',
        'data/processed',
        'models/cache',
        'models/output',
        'logs',
        'checkpoints'
    ]
    
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {dir_path}")


def configure_environment():
    """Configure environment variables and settings."""
    base_dir = Path(__file__).parent.parent
    env_example = base_dir / ".env.example"
    env_file = base_dir / ".env"
    
    if not env_file.exists() and env_example.exists():
        # Copy example env file
        shutil.copy(env_example, env_file)
        print(f"   ✅ Created .env file from example")
        
        # Prompt for basic configuration
        print("\n   Please provide your Jira credentials:")
        
        jira_url = input("   Jira Server URL (e.g., https://company.atlassian.net): ").strip()
        jira_username = input("   Jira Username/Email: ").strip()
        jira_token = getpass.getpass("   Jira API Token: ").strip()
        
        if jira_url and jira_username and jira_token:
            # Update .env file
            with open(env_file, 'r') as f:
                content = f.read()
            
            content = content.replace('https://your-instance.atlassian.net', jira_url)
            content = content.replace('your.email@company.com', jira_username)
            content = content.replace('your_jira_api_token_here', jira_token)
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("   ✅ Environment configured with your credentials")
        else:
            print("   ⚠️  Skipped credential configuration - please edit .env manually")
    else:
        print("   ✅ Environment file already exists")


def test_jira_connection():
    """Test Jira API connection."""
    try:
        from api import JiraClient
        
        print("   Testing Jira connection...")
        client = JiraClient()
        
        if client.test_connection():
            print("   ✅ Jira connection successful")
            
            # Get available projects
            projects = client.get_projects()
            if projects:
                print(f"   ✅ Found {len(projects)} accessible projects:")
                for project in projects[:5]:  # Show first 5
                    print(f"      - {project['key']}: {project['name']}")
                if len(projects) > 5:
                    print(f"      ... and {len(projects) - 5} more")
            else:
                print("   ⚠️  No accessible projects found")
        else:
            print("   ❌ Jira connection failed")
            
    except Exception as e:
        print(f"   ⚠️  Could not test Jira connection: {e}")
        print("   Please verify your credentials in .env file")


def prepare_base_model():
    """Prepare base model for training."""
    print("   Checking base model availability...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from utils import config
        
        model_name = config.get('model.base_model', 'google/gemma-2b-it')
        cache_dir = config.get('model.cache_dir', './models/cache')
        
        print(f"   Base model: {model_name}")
        
        # Check if model is already cached
        cache_path = Path(cache_dir)
        if cache_path.exists() and any(cache_path.iterdir()):
            print("   ✅ Base model already cached")
            return
        
        # Ask user if they want to download now
        download = input(f"   Download base model now? This may take several GB (y/N): ").strip().lower()
        
        if download in ['y', 'yes']:
            print(f"   Downloading {model_name}...")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            print("   ✅ Tokenizer downloaded")
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                torch_dtype='auto'
            )
            print("   ✅ Model downloaded")
            
            # Clean up memory
            del model
            del tokenizer
            
        else:
            print("   ⚠️  Skipped model download - will download on first training")
            
    except Exception as e:
        print(f"   ⚠️  Could not prepare base model: {e}")
        print("   Model will be downloaded during first training")


def create_sample_data():
    """Create sample data for testing."""
    base_dir = Path(__file__).parent.parent
    sample_file = base_dir / "data" / "sample_issues.json"
    
    if sample_file.exists():
        print("   ✅ Sample data already exists")
        return
    
    sample_data = [
        {
            "key": "SAMPLE-1",
            "summary": "Database connection timeout in production",
            "description": "Users are experiencing database connection timeouts during peak hours. The application fails to connect to the primary database server.",
            "issue_type": "Bug",
            "priority": "High",
            "status": "Resolved",
            "resolution": "Increased connection pool size and optimized database queries",
            "components": ["Database", "Backend"],
            "project_name": "Sample Project"
        },
        {
            "key": "SAMPLE-2", 
            "summary": "Login page not responsive on mobile",
            "description": "The login page does not display correctly on mobile devices. Buttons are too small and form fields are cut off.",
            "issue_type": "Bug",
            "priority": "Medium",
            "status": "Resolved", 
            "resolution": "Updated CSS media queries and improved responsive design",
            "components": ["Frontend", "UI"],
            "project_name": "Sample Project"
        }
    ]
    
    import json
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"   ✅ Created sample data: {sample_file}")


if __name__ == "__main__":
    main()