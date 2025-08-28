#!/usr/bin/env python3
"""
Adaptive training pipeline for continuous learning from new Jira tickets.

This script implements the core innovation: training new models from previously 
adapted models rather than the base model, enabling continuous learning.
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api import JiraClient, JiraDataExtractor
from data import DatasetConverter, DataValidator
from models import AdaptiveTrainer, ModelManager, ModelInference
from utils import setup_logging, get_logger, config


def main():
    """Main adaptive training pipeline."""
    parser = argparse.ArgumentParser(description="Adaptive training on new Jira tickets")
    parser.add_argument("--mode", choices=['single', 'batch', 'recent'], default='recent',
                       help="Training mode: single ticket, batch of tickets, or recent tickets")
    parser.add_argument("--ticket-key", help="Specific ticket key for single mode")
    parser.add_argument("--ticket-file", help="JSON file with ticket data for batch mode")
    parser.add_argument("--days", type=int, default=1, help="Number of days to look back for recent mode")
    parser.add_argument("--min-quality-score", type=float, default=0.6, 
                       help="Minimum quality score for training inclusion")
    parser.add_argument("--auto-deploy", action="store_true", 
                       help="Automatically deploy if training succeeds")
    parser.add_argument("--ab-test", action="store_true", 
                       help="Deploy as A/B test candidate")
    parser.add_argument("--force", action="store_true", 
                       help="Force training even with low quality data")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=config.get('system.log_level', 'INFO'),
        log_file=config.get('paths.logs_dir', './logs') + '/adaptive_training.log'
    )
    logger = get_logger(__name__)
    
    logger.info("Starting adaptive training pipeline")
    logger.info(f"Mode: {args.mode}")
    
    try:
        # Step 1: Get new ticket data based on mode
        if args.mode == 'single':
            ticket_data = get_single_ticket(args.ticket_key)
        elif args.mode == 'batch':
            ticket_data = load_batch_tickets(args.ticket_file)
        else:  # recent
            ticket_data = get_recent_tickets(args.days)
        
        if not ticket_data:
            logger.warning("No ticket data found - exiting")
            return
        
        # Step 2: Validate ticket quality
        validated_tickets = validate_ticket_quality(
            ticket_data, 
            args.min_quality_score, 
            args.force
        )
        
        if not validated_tickets:
            logger.warning("No tickets passed quality validation - exiting")
            return
        
        # Step 3: Perform adaptive training
        model_path = perform_adaptive_training(validated_tickets)
        
        if not model_path:
            logger.error("Adaptive training failed")
            return
        
        # Step 4: Register and optionally deploy
        model_id = register_adaptive_model(model_path, validated_tickets)
        
        if args.auto_deploy:
            deploy_model(model_id, args.ab_test)
        
        logger.info("Adaptive training completed successfully!")
        logger.info(f"Model: {model_id}")
        
    except Exception as e:
        logger.error(f"Adaptive training pipeline failed: {e}")
        raise


def get_single_ticket(ticket_key: str) -> list:
    """Get data for a single ticket.
    
    Args:
        ticket_key: Jira ticket key
        
    Returns:
        List containing single ticket data
    """
    logger = get_logger(__name__)
    
    if not ticket_key:
        raise ValueError("Ticket key required for single mode")
    
    logger.info(f"Fetching single ticket: {ticket_key}")
    
    # Initialize Jira client
    jira_client = JiraClient()
    extractor = JiraDataExtractor(jira_client)
    
    # Get the issue
    issue = jira_client.get_issue(ticket_key)
    issue_data = extractor.extract_issue_data(issue)
    
    logger.info(f"Retrieved ticket: {ticket_key}")
    return [issue_data]


def load_batch_tickets(ticket_file: str) -> list:
    """Load batch of tickets from JSON file.
    
    Args:
        ticket_file: Path to JSON file with ticket data
        
    Returns:
        List of ticket data dictionaries
    """
    logger = get_logger(__name__)
    
    if not ticket_file:
        raise ValueError("Ticket file required for batch mode")
    
    logger.info(f"Loading tickets from file: {ticket_file}")
    
    with open(ticket_file, 'r') as f:
        ticket_data = json.load(f)
    
    # Ensure it's a list
    if isinstance(ticket_data, dict):
        ticket_data = [ticket_data]
    
    logger.info(f"Loaded {len(ticket_data)} tickets from file")
    return ticket_data


def get_recent_tickets(days: int) -> list:
    """Get recent tickets from Jira.
    
    Args:
        days: Number of days to look back
        
    Returns:
        List of recent ticket data
    """
    logger = get_logger(__name__)
    
    logger.info(f"Fetching tickets from last {days} days")
    
    # Initialize Jira client
    jira_client = JiraClient()
    extractor = JiraDataExtractor(jira_client)
    
    # Get recent tickets as DataFrame
    df = extractor.extract_recent_data(
        days=days,
        max_issues=config.get('dataset.max_samples', 100)
    )
    
    if df.empty:
        logger.warning("No recent tickets found")
        return []
    
    # Convert to list of dictionaries
    ticket_data = df.to_dict('records')
    
    logger.info(f"Retrieved {len(ticket_data)} recent tickets")
    return ticket_data


def validate_ticket_quality(
    ticket_data: list, 
    min_quality_score: float, 
    force: bool = False
) -> list:
    """Validate quality of tickets for training.
    
    Args:
        ticket_data: List of ticket data dictionaries
        min_quality_score: Minimum quality score threshold
        force: Force inclusion even if quality is low
        
    Returns:
        List of validated tickets
    """
    logger = get_logger(__name__)
    
    logger.info(f"Validating quality for {len(ticket_data)} tickets")
    
    validator = DataValidator()
    validated_tickets = []
    
    for ticket in ticket_data:
        quality_check = validator.check_single_issue_quality(ticket)
        
        ticket_key = ticket.get('key', 'unknown')
        quality_score = quality_check['quality_score']
        is_valid = quality_check['is_valid']
        
        logger.debug(f"Ticket {ticket_key}: quality={quality_score:.2f}, valid={is_valid}")
        
        if force or (is_valid and quality_score >= min_quality_score):
            validated_tickets.append(ticket)
            logger.info(f"Accepted ticket {ticket_key} (quality: {quality_score:.2f})")
        else:
            logger.warning(f"Rejected ticket {ticket_key}: {quality_check['recommendation']}")
    
    logger.info(f"Validated {len(validated_tickets)}/{len(ticket_data)} tickets")
    
    return validated_tickets


def perform_adaptive_training(validated_tickets: list) -> str:
    """Perform adaptive training on validated tickets.
    
    Args:
        validated_tickets: List of validated ticket data
        
    Returns:
        Path to trained model, or None if training failed
    """
    logger = get_logger(__name__)
    
    logger.info(f"Starting adaptive training on {len(validated_tickets)} tickets")
    
    # Get current model for adaptive training
    model_manager = ModelManager()
    current_model_path, current_model_id = model_manager.get_model_for_inference()
    
    if not current_model_path:
        raise ValueError("No current model available for adaptive training")
    
    logger.info(f"Using current model as base: {current_model_id}")
    
    # Initialize adaptive trainer
    trainer = AdaptiveTrainer(
        base_model_name=config.get('model.base_model'),
        previous_model_path=current_model_path
    )
    
    if len(validated_tickets) == 1:
        # Single ticket training
        ticket = validated_tickets[0]
        result = trainer.train_single_ticket(ticket)
        
        if not result.get('success', False):
            logger.error(f"Single ticket training failed: {result.get('error', 'Unknown error')}")
            return None
        
        return result.get('model_path')
    
    else:
        # Batch training
        # Convert tickets to dataset
        converter = DatasetConverter()
        
        # Convert list of tickets to examples
        all_examples = []
        for ticket in validated_tickets:
            examples = converter.convert_single_issue(ticket)
            all_examples.extend(examples)
        
        if not all_examples:
            logger.error("No training examples generated from tickets")
            return None
        
        # Create dataset
        from datasets import Dataset, DatasetDict
        dataset = Dataset.from_list(all_examples)
        dataset_dict = DatasetDict({'train': dataset})
        
        # Setup and train
        trainer.setup_model_and_tokenizer()
        result = trainer.train(dataset_dict, save_model=True)
        
        return result.model_path


def register_adaptive_model(model_path: str, ticket_data: list) -> str:
    """Register the adaptively trained model.
    
    Args:
        model_path: Path to trained model
        ticket_data: List of tickets used for training
        
    Returns:
        Model ID
    """
    logger = get_logger(__name__)
    
    # Prepare metadata
    metadata = {
        'training_type': 'adaptive',
        'training_tickets': len(ticket_data),
        'ticket_keys': [t.get('key', 'unknown') for t in ticket_data],
        'description': f'Adaptive training on {len(ticket_data)} tickets',
        'training_timestamp': datetime.now().isoformat()
    }
    
    # Register model
    model_manager = ModelManager()
    model_id = model_manager.register_model(
        model_path=model_path,
        model_type='adaptive',
        metadata=metadata
    )
    
    logger.info(f"Registered adaptive model: {model_id}")
    
    return model_id


def deploy_model(model_id: str, ab_test: bool = False) -> bool:
    """Deploy the trained model.
    
    Args:
        model_id: ID of model to deploy
        ab_test: Whether to deploy as A/B test candidate
        
    Returns:
        True if deployment successful
    """
    logger = get_logger(__name__)
    
    model_manager = ModelManager()
    
    if ab_test:
        # Deploy as A/B test candidate
        success = model_manager.start_ab_test(
            candidate_model_id=model_id,
            traffic_split=config.get('ab_testing.traffic_split', 0.1)
        )
        
        if success:
            logger.info(f"Started A/B test with model {model_id}")
        else:
            logger.error("Failed to start A/B test")
        
        return success
    
    else:
        # Direct deployment as current model
        success = model_manager.deploy_model(model_id, 'current')
        
        if success:
            logger.info(f"Deployed model {model_id} as current")
        else:
            logger.error("Failed to deploy model")
        
        return success


def check_deployment_eligibility(model_path: str) -> bool:
    """Check if model is ready for deployment.
    
    Args:
        model_path: Path to trained model
        
    Returns:
        True if model passes deployment checks
    """
    logger = get_logger(__name__)
    
    try:
        # Basic checks
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error("Model path does not exist")
            return False
        
        # Check for required files
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        for file_name in required_files:
            if not (model_path / file_name).exists():
                logger.warning(f"Missing file: {file_name}")
        
        # TODO: Add more sophisticated checks:
        # - Model validation on test set
        # - Performance regression tests
        # - Safety and bias checks
        
        logger.info("Model passed deployment eligibility checks")
        return True
        
    except Exception as e:
        logger.error(f"Deployment eligibility check failed: {e}")
        return False


if __name__ == "__main__":
    main()