# 🐍 Virtual Environment Setup

## Quick Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Step-by-Step Instructions

### 1. Create Virtual Environment
```bash
cd /Users/yovadia/code/auto-fine-tune
python -m venv venv
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```
**Note**: You'll see `(venv)` in your terminal prompt when activated.

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch, transformers, datasets; print('✅ All ML dependencies installed')"
```

### 5. Train the Model
```bash
python train_real_model.py
```

### 6. Deactivate When Done
```bash
deactivate
```

## What Gets Installed

The virtual environment will contain:
- PyTorch (ML framework)
- Transformers (Hugging Face models)
- Datasets (Data processing)
- PEFT (LoRA fine-tuning)
- Other dependencies from requirements.txt

## Troubleshooting

**If you see "No module named 'torch'":**
- Make sure virtual environment is activated (`(venv)` in prompt)
- Re-run: `pip install -r requirements.txt`

**If virtual environment activation fails:**
- Make sure you're in the project directory
- Try: `python3 -m venv venv` instead