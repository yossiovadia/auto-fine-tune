# GPU Deployment Guide - RTX 4090 Setup

## Quick Setup for RTX 4090

### 1. Clone Repository
```bash
git clone <your-repo-url> adaptive-fine-tune
cd adaptive-fine-tune
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install GPU dependencies
pip install -r requirements_gpu.txt
```

### 3. Install CUDA-specific packages
```bash
# For RTX 4090, install CUDA 11.8 or 12.x compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: Install flash-attention for speed boost
pip install flash-attn --no-build-isolation
```

### 4. Setup GitHub CLI
```bash
# Install GitHub CLI (if not already installed)
# Ubuntu/Debian:
sudo apt install gh
# Or download from: https://cli.github.com/

# Authenticate
gh auth login
```

### 5. Run Full Pipeline
```bash
# Single command to collect data, process, and prepare for training
python run_full_pipeline.py --max-issues 500 --skip-training

# Or with training (will take 10-20 minutes on RTX 4090)
python run_full_pipeline.py --max-issues 500
```

### 6. GPU Training (Manual)
```bash
# Train using GPU-optimized script
python train_gpu.py
```

## Performance Expectations

### RTX 4090 vs M4 Mac
| Component | RTX 4090 | M4 Mac (48GB) |
|-----------|----------|---------------|
| Training Time | 10-20 min | 20-40 min |
| Batch Size | 4-8 | 1-2 |
| Memory Usage | ~12GB VRAM | ~8GB RAM |
| Precision | bfloat16/fp16 | float32 |

### Memory Requirements
- **Minimum VRAM**: 8GB (RTX 3070 or better)
- **Recommended VRAM**: 12GB+ (RTX 4070 Ti, RTX 4090)
- **Optimal VRAM**: 24GB (RTX 4090, RTX A6000)

## GPU-Specific Optimizations

### Training Script Differences
The GPU version (`train_gpu.py`) includes:
- Flash Attention 2 for faster training
- Mixed precision (fp16/bfloat16)
- Larger batch sizes
- Automatic device mapping
- CUDA memory optimizations

### Troubleshooting

**CUDA Out of Memory:**
```bash
# Reduce batch size in train_gpu.py:
per_device_train_batch_size=2
gradient_accumulation_steps=8
```

**Flash Attention Install Issues:**
```bash
# Skip flash attention, edit train_gpu.py:
attn_implementation="eager"  # instead of "flash_attention_2"
```

**PyTorch CUDA Version Mismatch:**
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Key Files for GPU Deployment

### Essential Scripts:
- `train_gpu.py` - GPU-optimized training
- `run_full_pipeline.py` - Complete workflow
- `requirements_gpu.txt` - GPU dependencies

### Data Pipeline (same as M4):
- `collect_vllm_issues_gh.py` - GitHub data collection
- `ingest_vllm_codebase.py` - Code analysis
- `create_enhanced_dataset.py` - Dataset fusion
- `incremental_update_system.py` - Live updates

## Expected Results

### Training Output:
```
🚀 Using CUDA acceleration!
GPU: NVIDIA GeForce RTX 4090
VRAM: 24.0GB
🔗 Using code-aware enhanced dataset
📊 Loaded 297 training examples
📊 Trainable (LoRA): 20,971,520
⏱️  Estimated time: 10-20 minutes on RTX 4090
✅ Training completed in 12.3 minutes!
🔥 Peak GPU memory: 11.2GB
```

### Model Location:
- **M4 Mac**: `./models/vllm_qwen_assistant_m4/`
- **GPU**: `./models/vllm_qwen_assistant_gpu/`

## Incremental Updates

Same workflow as M4, but faster:
```bash
# Check for repository changes
python incremental_update_system.py

# Retrain with new data (faster on GPU)
python train_gpu.py
```

This setup gives you ~2x faster training on RTX 4090 compared to M4 Mac.