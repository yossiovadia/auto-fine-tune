#!/usr/bin/env python3
"""
Initial model training pipeline for adaptive Jira defect analysis system.

This script performs the initial fine-tuning of the base model on historical Jira data.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api import JiraClient, JiraDataExtractor
from data import DatasetConverter, DataValidator, TextPreprocessor
from models import AdaptiveTrainer, ModelManager
from utils import setup_logging, get_logger, config


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train initial model on historical Jira data")
    parser.add_argument("--project-key", required=True, help="Jira project key to extract data from")
    parser.add_argument("--max-issues", type=int, default=1000, help="Maximum number of issues to process")
    parser.add_argument("--output-dir", help="Output directory for trained model")
    parser.add_argument("--data-file", help="Use existing data file instead of extracting from Jira")
    parser.add_argument("--skip-validation", action="store_true", help="Skip data validation")
    parser.add_argument("--dry-run", action="store_true", help="Prepare data but don't train")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=config.get('system.log_level', 'INFO'),
        log_file=config.get('paths.logs_dir', './logs') + '/training.log'
    )
    logger = get_logger(__name__)
    
    logger.info("Starting initial model training pipeline")
    logger.info(f"Project: {args.project_key}")
    logger.info(f"Max issues: {args.max_issues}")
    
    try:
        # Step 1: Extract or load data
        if args.data_file:
            logger.info(f"Loading data from file: {args.data_file}")
            df = pd.read_csv(args.data_file)
        else:
            logger.info("Extracting data from Jira...")
            df = extract_jira_data(args.project_key, args.max_issues)
        
        logger.info(f"Loaded {len(df)} issues")
        
        # Step 2: Validate and clean data
        if not args.skip_validation:
            logger.info("Validating and cleaning data...")
            df = validate_and_clean_data(df)
            logger.info(f"After cleaning: {len(df)} issues")
        
        # Step 3: Convert to training dataset
        logger.info("Converting to training dataset...")
        dataset = convert_to_dataset(df)
        
        # Step 4: Train model
        if not args.dry_run:
            logger.info("Starting model training...")
            model_path = train_model(dataset, args.output_dir, args.wandb)
            
            # Step 5: Register model
            logger.info("Registering trained model...")
            register_model(model_path)
            
            logger.info("Initial training completed successfully!")
            logger.info(f"Model saved to: {model_path}")
        else:
            logger.info("Dry run completed - data prepared but model not trained")
    
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


def extract_jira_data(project_key: str, max_issues: int) -> pd.DataFrame:
    """Extract data from Jira.
    
    Args:
        project_key: Jira project key
        max_issues: Maximum number of issues to extract
        
    Returns:
        DataFrame containing extracted issue data
    """
    logger = get_logger(__name__)
    
    # Initialize Jira client
    jira_client = JiraClient()
    
    # Test connection
    if not jira_client.test_connection():
        raise ConnectionError("Failed to connect to Jira")
    
    # Initialize data extractor
    extractor = JiraDataExtractor(jira_client)
    
    # Extract project data
    df = extractor.extract_project_data(
        project_key=project_key,
        issue_types=['Bug', 'Task', 'Story', 'Improvement'],
        max_issues=max_issues,
        output_file=f"data/raw/{project_key}_issues.csv"
    )
    
    if df.empty:
        raise ValueError(f"No issues found for project {project_key}")
    
    # Generate summary
    summary = extractor.get_data_summary(df)
    logger.info(f"Data extraction summary: {summary}")
    
    return df


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the extracted data.
    
    Args:
        df: Raw DataFrame from Jira extraction
        
    Returns:
        Cleaned DataFrame
    """
    logger = get_logger(__name__)
    
    # Initialize validator
    validator = DataValidator()
    
    # Generate validation report
    validation_report = validator.validate_dataset(df)
    logger.info(f"Validation report: {validation_report['validation_rate']:.2%} issues valid")
    
    if validation_report['validation_rate'] < 0.5:
        logger.warning("Low validation rate - check data quality")
    
    # Check for duplicates
    duplicates = validator.detect_duplicates(df, similarity_threshold=0.85)
    if duplicates:
        logger.info(f"Found {len(duplicates)} potential duplicate groups")
    
    # Clean the dataset
    cleaned_df = validator.clean_dataset(df)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Clean text fields
    cleaned_df['summary'] = cleaned_df['summary'].apply(preprocessor.preprocess_for_training)
    cleaned_df['description'] = cleaned_df['description'].apply(preprocessor.preprocess_for_training)
    
    # Save cleaned data
    output_path = Path(config.get('paths.data_dir', './data')) / 'processed' / 'cleaned_issues.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to: {output_path}")
    
    return cleaned_df


def convert_to_dataset(df: pd.DataFrame):
    """Convert DataFrame to HuggingFace dataset.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        HuggingFace DatasetDict
    """
    logger = get_logger(__name__)
    
    # Initialize converter
    converter = DatasetConverter()
    
    # Create dataset with configured splits
    dataset = converter.create_dataset(
        df,
        shuffle=True,
        random_seed=42
    )
    
    # Get dataset statistics
    stats = converter.get_dataset_stats(dataset)
    logger.info(f"Dataset statistics: {stats}")
    
    # Save dataset
    dataset_path = Path(config.get('paths.data_dir', './data')) / 'processed' / 'training_dataset'
    converter.save_dataset(dataset, str(dataset_path))
    logger.info(f"Dataset saved to: {dataset_path}")
    
    return dataset


def train_model(dataset, output_dir: str = None, enable_wandb: bool = False):
    """Train the initial model.
    
    Args:
        dataset: HuggingFace DatasetDict
        output_dir: Optional output directory
        enable_wandb: Whether to enable Weights & Biases logging
        
    Returns:
        Path to trained model
    """
    logger = get_logger(__name__)
    
    # Setup wandb if requested
    if enable_wandb:
        try:
            import wandb
            wandb.login()
            logger.info("Weights & Biases enabled")
        except Exception as e:
            logger.warning(f"Failed to setup wandb: {e}")
    
    # Initialize trainer
    trainer = AdaptiveTrainer(
        base_model_name=config.get('model.base_model'),
        previous_model_path=None  # This is initial training
    )
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Start training
    training_result = trainer.train(dataset, save_model=True)
    
    # Log training results
    logger.info(f"Training completed:")
    logger.info(f"  Final training loss: {training_result.training_loss:.4f}")
    
    if hasattr(training_result, 'eval_metrics'):
        logger.info(f"  Final eval loss: {training_result.eval_metrics['eval_loss']:.4f}")
    
    return training_result.model_path


def register_model(model_path: str) -> str:
    """Register the trained model in the model manager.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Model ID
    """
    logger = get_logger(__name__)
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Register the model
    model_id = model_manager.register_model(
        model_path=model_path,
        model_type='base',
        metadata={
            'training_type': 'initial',
            'description': 'Initial fine-tuned model on historical Jira data'
        }
    )
    
    # Deploy as current model (first deployment)
    model_manager.deploy_model(model_id, 'current')
    
    logger.info(f"Model registered and deployed: {model_id}")
    
    return model_id


if __name__ == "__main__":
    main()