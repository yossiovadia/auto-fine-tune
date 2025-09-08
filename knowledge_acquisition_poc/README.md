# 🧠 Knowledge Acquisition POC

This POC proves that **fine-tuning can teach models completely new information** they didn't previously know.

## 🎯 Quick Start

```bash
cd knowledge_acquisition_poc
python knowledge_demo.py
```

## 📋 What It Does

1. **Tests Established Knowledge** - Shows model knows basic facts
2. **Tests 2025 Knowledge** - Shows model doesn't know 2025 events  
3. **Live Training** - Trains model on real 2025 facts with progress updates
4. **Before/After Testing** - Same questions show clear improvement
5. **Interactive Testing** - Ask your own questions to test learned knowledge

## 🗂 Structure

```
knowledge_acquisition_poc/
├── knowledge_demo.py           # Main demo script
├── domains/
│   ├── poc_knowledge_domains.py     # Original domains (vLLM, fictional)
│   └── knowledge_domains_2025.py    # 2025 events (recommended)
├── demos/
│   ├── poc_knowledge_acquisition.py # Core evaluation framework
│   ├── poc_fine_tuning.py           # Training infrastructure  
│   └── run_complete_poc.py          # Automated pipeline
├── docs/
│   ├── KNOWLEDGE_ACQUISITION_POC_SUMMARY.md
│   └── INTERACTIVE_DEMO_README.md
└── results/                    # Generated results and models
```

## 🚀 Usage Options

### Full Interactive Demo (Recommended)
```bash
python knowledge_demo.py --mode demo
```
Shows complete before/after transformation with user interaction.

### Training Only
```bash
python knowledge_demo.py --mode train
```
Just trains and saves the model.

### Testing with Existing Model
```bash
python knowledge_demo.py --mode test --model-path results/trained_model/final
```
Tests with previously trained model.

### Different Base Model
```bash
python knowledge_demo.py --model "microsoft/Phi-3-mini-4k-instruct"
```

## 🎮 Demo Experience

1. **Established Facts** → Model answers correctly ✅
2. **2025 Events** → Model fabricates or gives weak answers ⚠️
3. **Training** → Live progress updates 🚀  
4. **Same Questions** → Model now gives detailed, accurate answers ✅
5. **Interactive** → Test with your own questions 🎮

## 🎯 Key Features

- **Uses Real 2025 Events** - Impossible for pre-2025 models to know
- **Live Training** - Watch the model learn in real-time
- **Before/After Comparison** - Clear evidence of knowledge acquisition  
- **GPU Optimized** - Fast LoRA training
- **Interactive Testing** - Hands-on validation

## 🏗 Technical Details

- **Base Model**: TinyLlama-1.1B (fast) or any HuggingFace causal LM
- **Training**: LoRA fine-tuning (r=16, α=32)
- **Data**: Real 2025 events (AI developments, world events, tech products)
- **Time**: ~1-2 minutes training on RTX 4090

## ✅ Success Criteria

The demo proves knowledge acquisition when you see:
- **Before**: Model fabricates or gives uncertain answers about 2025
- **After**: Model provides detailed, accurate information with correct dates
- **Interactive**: Model can answer novel questions about learned topics

This provides **definitive proof** that fine-tuning can teach models new information! 🎉