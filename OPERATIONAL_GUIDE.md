# Operational Guide - Adaptive Fine-Tuning POC

## Current State: Experimental/POC
- ✅ Basic pipeline working
- ⚠️ Not production ready
- 🔧 Requires manual steps
- 📊 Limited testing done

## Core Workflow Steps

### 1. Full Pipeline (Recommended)

**Single Command - Complete Data Pipeline:**
```bash
# Activate environment
source .venv/bin/activate

# Run complete pipeline (all data collection, processing, and preparation)
python run_full_pipeline.py --max-issues 500

# Options:
# --max-issues 500        # Number of GitHub issues to collect
# --force-refresh         # Force refresh all existing data
# --skip-training         # Prepare data but skip training step

# Result: All data collected and processed, ready for training
```

### 2. Individual Steps (For Manual Control)

**If you need to run steps individually:**

**Collect GitHub Issues:**
```bash
python collect_vllm_issues_gh.py --max-issues 500
# Result: data/vllm_full_dataset.json (6.2MB)
```

**Ingest vLLM Codebase:**
```bash
python ingest_vllm_codebase.py
# Results: 
# - data/vllm/ (cloned repo)
# - data/codebase/vllm_code_examples.jsonl (324 examples)
```

**Improve Dataset Quality:**
```bash
python improve_dataset.py
# Result: data/training_datasets/period_2/improved_qa_examples.jsonl (139 examples)
```

**Create Code-Aware Dataset:**
```bash
python create_enhanced_dataset.py
# Result: data/training_datasets/period_2/code_aware_dataset.jsonl (297 examples)
```

### 3. Model Training

**Train with Qwen2.5-Coder-7B:**
```bash
# Train the model (20-40 minutes on M4)
python train_real_m4.py

# Results:
# - models/vllm_qwen_assistant_m4/ (trained model)
# - Training logs and metrics
```

**Test the Model:**
```bash
# Test the trained model
python test_trained_model.py
```

### 4. Incremental Updates (Delta Processing)

**Check for Repository Changes:**
```bash
# Check what's new since last update
python incremental_update_system.py

# Results:
# - Shows commit changes, file modifications
# - Generates data/incremental_training_YYYYMMDD_HHMMSS.jsonl
# - Updates data/last_update.json
```

**Manual Delta Training (Tomorrow's Workflow):**

1. **Check for changes:**
   ```bash
   python incremental_update_system.py
   ```

2. **If significant changes found, merge datasets:**
   ```bash
   # Manually combine incremental data with existing
   cat data/training_datasets/period_2/code_aware_dataset.jsonl \
       data/incremental_training_YYYYMMDD_HHMMSS.jsonl \
       > data/training_datasets/period_2/updated_dataset.jsonl
   ```

3. **Update training script to use new dataset:**
   ```bash
   # Edit train_real_m4.py to point to updated_dataset.jsonl
   # Then retrain:
   python train_real_m4.py
   ```

## File Structure & Dependencies

### Essential Files:
```
├── train_real_m4.py              # Main training script
├── collect_vllm_issues_gh.py     # GitHub data collection
├── ingest_vllm_codebase.py       # Code analysis
├── improve_dataset.py            # Quality filtering
├── create_enhanced_dataset.py    # Dataset fusion
├── incremental_update_system.py  # Delta detection
├── requirements_m4.txt           # Dependencies
└── data/
    ├── vllm_full_dataset.json            # Source issues
    ├── training_datasets/period_2/
    │   └── code_aware_dataset.jsonl      # Active training data
    ├── codebase/
    │   ├── vllm_code_examples.jsonl      # Code knowledge  
    │   └── vllm_file_analysis.json
    └── last_update.json                  # Incremental state
```

### Prerequisites:
- Python 3.11+
- Virtual environment (.venv)
- GitHub CLI (gh) authenticated
- ~15GB free space (for models + data)
- Apple Silicon M4 (for current optimization)

## Current Limitations

### Manual Steps Required:
- Dataset merging for incremental updates
- Quality assessment of new data
- Training parameter tuning per dataset size
- Model evaluation and comparison

### Not Automated:
- Automatic retraining triggers
- Model version management  
- Performance regression detection
- Deployment pipeline

### Testing Gaps:
- No automated test suite
- Limited model quality metrics
- No benchmark comparisons
- No error handling for edge cases

## Troubleshooting

### Common Issues:

**GitHub rate limits:**
```bash
# Check authentication
gh auth status

# If rate limited, wait or use different auth
```

**Training memory issues:**
```bash
# Reduce batch size in train_real_m4.py:
per_device_train_batch_size=1
gradient_accumulation_steps=16
```

**Dataset size too small:**
```bash
# Collect more issues
python collect_vllm_issues_gh.py --max-issues 1000

# Or lower quality threshold in improve_dataset.py
```

## Next Development Steps

### To Make Production-Ready:
1. Automated delta training pipeline
2. Model performance monitoring
3. A/B testing framework
4. Error handling and recovery
5. Configuration management
6. Deployment automation
7. Monitoring and alerting

### Current POC Value:
- Demonstrates adaptive learning concept
- Shows GitHub issues + codebase integration
- Proves fine-tuning on domain-specific data works
- Establishes incremental update detection

This is a working proof-of-concept, not a production system.