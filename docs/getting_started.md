# Getting Started with Adaptive Jira Defect Analysis System

## Overview

The Adaptive Jira Defect Analysis System is an innovative machine learning platform that continuously learns from Jira tickets to provide intelligent defect analysis. Unlike traditional ML systems, it implements iterative fine-tuning where new models are trained from previously adapted models, enabling continuous learning without catastrophic forgetting.

## Key Features

- **Adaptive Learning**: Models improve over time by learning from each resolved ticket
- **Real-time Processing**: Automatic retraining when new defects arrive
- **A/B Testing**: Seamless model transitions with performance monitoring
- **Comprehensive Analysis**: Error pattern recognition, troubleshooting guidance, priority assessment
- **Jira Integration**: Direct integration with Jira APIs for data extraction

## Quick Start

### 1. Installation and Setup

```bash
# Clone the repository
git clone <repository-url>
cd auto-fine-tune

# Run setup script
python scripts/setup.py
```

The setup script will:
- Check system requirements
- Install dependencies
- Create directory structure  
- Configure environment variables
- Test Jira connection
- Prepare base model

### 2. Configure Environment

Edit `.env` file with your Jira credentials:

```env
JIRA_USERNAME=your.email@company.com
JIRA_API_TOKEN=your_jira_api_token_here
JIRA_SERVER_URL=https://your-company.atlassian.net
```

### 3. Initial Model Training

Train the initial model on historical Jira data:

```bash
python scripts/train_initial_model.py --project-key YOUR_PROJECT --max-issues 1000
```

### 4. Start Using the System

Run interactive mode to test the system:

```bash
python scripts/run_inference.py --mode interactive
```

## Usage Examples

### Error Pattern Recognition

```bash
python scripts/run_inference.py --mode error-check \
  --error-description "java.lang.NullPointerException at com.myapp.auth.TokenValidator"
```

### Troubleshooting Guidance

```bash
python scripts/run_inference.py --mode troubleshoot \
  --query "Database connection timeouts in production" \
  --component "Database"
```

### Defect Analysis

```bash
python scripts/run_inference.py --mode analyze --ticket-key PROJ-123
```

### Latest Defect Analysis

```bash
python scripts/run_inference.py --mode latest-defect
```

## Adaptive Training

### Single Ticket Training

When a new defect is resolved, retrain the model:

```bash
python scripts/adaptive_training.py --mode single --ticket-key PROJ-456
```

### Batch Training

Train on multiple recent tickets:

```bash
python scripts/adaptive_training.py --mode recent --days 7 --auto-deploy
```

### A/B Testing

Deploy new model as A/B test candidate:

```bash
python scripts/adaptive_training.py --mode recent --days 1 --ab-test
```

## System Monitoring

### Check System Status

```bash
python scripts/system_monitor.py --action status
```

### Monitor A/B Tests

```bash
python scripts/system_monitor.py --action ab-test-status
python scripts/system_monitor.py --action ab-test-evaluate
```

### Continuous Monitoring

```bash
python scripts/system_monitor.py --action status --watch --interval 30
```

## Architecture

### Core Components

1. **Jira Integration** (`src/api/`)
   - `JiraClient`: API connection and data extraction
   - `JiraDataExtractor`: Issue data processing and cleaning

2. **Data Pipeline** (`src/data/`)
   - `DatasetConverter`: Convert Jira tickets to instruction-following format
   - `DataValidator`: Quality validation and preprocessing
   - `TextPreprocessor`: Text cleaning and normalization

3. **Model Management** (`src/models/`)
   - `AdaptiveTrainer`: Iterative fine-tuning implementation
   - `ModelManager`: Version control and A/B testing
   - `ModelInference`: Model loading and prediction
   - `ModelEvaluator`: Performance metrics and evaluation

### Data Flow

```
Jira Tickets → Data Extraction → Quality Validation → Dataset Conversion → 
Model Training → Model Registration → A/B Deployment → Performance Monitoring
```

## Configuration

Key configuration options in `config/config.yaml`:

```yaml
# Model settings
model:
  base_model: "google/gemma-2b-it"
  model_max_length: 2048

# Training parameters
training:
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3

# A/B testing
ab_testing:
  traffic_split: 0.1
  performance_threshold: 0.85

# Data processing
dataset:
  train_split: 0.8
  min_description_length: 50
  max_samples: 10000
```

## Use Cases and Demonstrations

### 1. Error Pattern Recognition
**Query**: "I see this error: `OutOfMemoryError: Java heap space`, is that something known?"
**Capability**: Recognizes patterns from historical tickets and provides known solutions

### 2. Component Expertise Building
**Query**: "What are common issues with our authentication module?"
**Capability**: Builds deep knowledge about specific system components

### 3. Intelligent Troubleshooting
**Query**: "User can't login after password reset - what should I check?"
**Capability**: Provides prioritized troubleshooting steps based on successful resolutions

### 4. Priority Assessment
**Query**: "SSL certificate error in production - how critical is this?"
**Capability**: Assesses impact based on historical ticket patterns

### 5. Resolution Time Prediction
**Query**: "How long does this database issue typically take to resolve?"
**Capability**: Predicts resolution time based on similar historical cases

## Best Practices

### Data Quality
- Ensure tickets have meaningful descriptions
- Include resolution information when available
- Validate data quality before training
- Monitor for duplicates and noise

### Model Training
- Start with a good base of historical data (500+ tickets)
- Perform incremental training rather than full retraining
- Monitor model performance after each adaptation
- Use A/B testing for production deployments

### System Maintenance
- Regularly monitor system health
- Track A/B test performance metrics
- Maintain baseline models for rollback
- Clean up old model versions periodically

## Troubleshooting

### Common Issues

**Issue**: "No model loaded for inference"
**Solution**: Run initial training or check model deployment status

**Issue**: "Jira connection failed"
**Solution**: Verify credentials in `.env` file and network connectivity

**Issue**: "CUDA out of memory"
**Solution**: Reduce batch size or use CPU training

**Issue**: "Low quality training data"
**Solution**: Improve data validation thresholds or clean source data

### Debug Commands

```bash
# Check system health
python scripts/system_monitor.py --action health-check

# Test performance
python scripts/system_monitor.py --action performance-test

# List all models
python scripts/system_monitor.py --action models-list

# Get model details
python scripts/system_monitor.py --action model-info --model-id MODEL_ID
```

## Advanced Topics

### Custom Dataset Formats
You can extend the `DatasetConverter` to support custom instruction formats or add new task types.

### Model Evaluation
Use the `ModelEvaluator` class to run comprehensive evaluations:

```python
from models import ModelEvaluator
evaluator = ModelEvaluator()
results = evaluator.benchmark_performance()
```

### Integration with n8n
The system is designed to integrate with n8n workflows for automation. See `docs/n8n_integration.md` for details.

## Support and Contributing

For issues and feature requests, please check the project documentation or create an issue in the repository.

The system is designed to be extensible - you can add new evaluation metrics, custom data sources, or additional model architectures as needed.