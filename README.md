# Adaptive Jira Defect Analysis System

An automated system that continuously learns from Jira defects to improve software issue analysis over time through iterative fine-tuning.

## Overview

This system implements adaptive fine-tuning on Gemma3-270m, where new models are trained from previously adapted models rather than the base model, enabling continuous learning without catastrophic forgetting.

## Key Features

- **Adaptive Learning**: Iterative fine-tuning on previously trained models
- **Real-time Processing**: Automatic retraining when new Jira defects arrive  
- **A/B Deployment**: Seamless model transitions with rollback capability
- **Continuous Improvement**: Accumulated knowledge retention over time

## Architecture

```
Jira Tickets → Dataset Generation → Model Fine-tuning → A/B Deployment → Performance Monitoring
```

## Project Structure

```
├── src/
│   ├── data/              # Data processing and pipeline
│   ├── models/            # Model training and inference
│   ├── api/               # Jira API integration
│   ├── workflows/         # n8n workflow definitions
│   └── utils/             # Shared utilities
├── config/                # Configuration files
├── tests/                 # Unit and integration tests
├── scripts/               # Setup and deployment scripts
└── docs/                  # Documentation
```

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure Jira API credentials in `config/config.yaml`
4. Run initial setup: `python scripts/setup.py`

## Development

- Python 3.9+
- PyTorch with CUDA support
- HuggingFace Transformers
- n8n for workflow automation

## License

MIT License