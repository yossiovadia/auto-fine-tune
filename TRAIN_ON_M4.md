# 🚀 Train Real vLLM Assistant on M4 Mac

## Quick Start (15-30 minutes total)

```bash
# 1. Setup virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install M4-optimized dependencies  
pip install -r requirements_m4.txt

# 3. Train the model (15-30 min)
python train_real_m4.py
```

## What You'll Get

After training completes, you'll have:
- ✅ **Real trained model** that answers vLLM questions
- ✅ **Model files** saved in `models/vllm_assistant_m4/`
- ✅ **Live demonstration** of the model answering test questions
- ✅ **Proof** that adaptive learning works with real data

## Expected Timeline on M4 (48GB RAM)

- **Setup**: 2-3 minutes
- **Training**: 15-30 minutes (depending on data size)
- **Testing**: 1-2 minutes
- **Total**: ~20-35 minutes

## What the Model Will Learn

The model trains on real vLLM GitHub issues to learn:
- 🔧 **Error troubleshooting**: CUDA OOM, LoRA errors, TPU issues
- 🤖 **Model compatibility**: How to run Llama, Qwen, GPT-OSS
- ⚙️ **Configuration**: Production setups, optimization tips
- 🔍 **Problem patterns**: Common vLLM challenges and solutions

## Sample Questions It Can Answer

After training, your model will answer questions like:
- "How to fix CUDA out of memory error in vLLM?"
- "How to run Llama-70B with tensor parallelism?"
- "ValueError: unsupported LoRA weight - what's wrong?"
- "Best vLLM configuration for production serving?"

## Technical Details

- **Base Model**: GPT-2 (fast training on M4)
- **Fine-tuning**: LoRA (parameter-efficient)
- **Acceleration**: Apple MPS (Metal Performance Shaders)  
- **Data**: Real vLLM GitHub issues and solutions
- **Training**: 3 epochs, batch size 8, learning rate 3e-4

## Troubleshooting

**If training fails**:
1. Make sure virtual environment is activated: `source venv/bin/activate`
2. Check you have training data: `ls data/training_datasets/period_2/`
3. Verify MPS is available: `python -c "import torch; print(torch.backends.mps.is_available())"`

**If you need training data**:
```bash
python collect_vllm_issues_gh.py --max-issues 500
python src/data/vllm_dataset_converter.py --input data/vllm_full_dataset.json --output data/training_datasets --types qa
```

## Memory Usage

On M4 with 48GB RAM:
- **Training**: ~8-12GB RAM
- **Model size**: ~500MB 
- **GPU memory**: ~4-6GB MPS usage
- **Storage**: ~1GB for model + data

Perfect fit for your hardware! 🎯