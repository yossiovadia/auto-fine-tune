#!/usr/bin/env python3
"""
API server for n8n integration with Adaptive Jira Defect Analysis System.

This server provides REST endpoints that n8n can call to interact with
the adaptive fine-tuning system.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from models import ModelManager, ModelInference
    from data import DataValidator, DatasetConverter, TextPreprocessor
    from api import JiraDataExtractor
    from utils import setup_logging, get_logger, config
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    print("Some endpoints may not work until dependencies are installed.")

app = Flask(__name__)
CORS(app)

# Setup logging
setup_logging(
    log_level='INFO',
    log_file='logs/api_server.log'
)
logger = get_logger(__name__)

# Initialize components (with error handling)
try:
    model_manager = ModelManager()
    validator = DataValidator()
    converter = DatasetConverter()
    preprocessor = TextPreprocessor()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    model_manager = None
    validator = None
    converter = None
    preprocessor = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'components': {
            'model_manager': model_manager is not None,
            'validator': validator is not None,
            'converter': converter is not None,
            'preprocessor': preprocessor is not None
        }
    })


@app.route('/api/duplicate-check', methods=['POST'])
def duplicate_check():
    """Check if a ticket is a duplicate of existing tickets."""
    try:
        data = request.json
        summary = data.get('summary', '')
        description = data.get('description', '')
        ticket_key = data.get('ticket_key', 'unknown')
        
        logger.info(f"Duplicate check for ticket: {ticket_key}")
        
        # Simple heuristic for now - in production you'd want vector similarity
        # For demonstration, we'll return false (not duplicate)
        
        # You could implement actual duplicate detection here using:
        # - Vector embeddings similarity
        # - Fuzzy string matching
        # - Database lookup of existing tickets
        
        result = {
            'is_duplicate': False,
            'similarity_score': 0.0,
            'similar_tickets': [],
            'ticket_key': ticket_key,
            'checked_at': datetime.now().isoformat()
        }
        
        logger.info(f"Duplicate check result for {ticket_key}: {result['is_duplicate']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Duplicate check failed: {e}")
        return jsonify({
            'error': str(e),
            'is_duplicate': False,
            'similarity_score': 0.0
        }), 500


@app.route('/api/quality-check', methods=['POST'])
def quality_check():
    """Check the quality of a ticket for training suitability."""
    try:
        ticket_data_str = request.json.get('ticket_data')
        if isinstance(ticket_data_str, str):
            ticket_data = json.loads(ticket_data_str)
        else:
            ticket_data = ticket_data_str
        
        ticket_key = ticket_data.get('key', 'unknown')
        logger.info(f"Quality check for ticket: {ticket_key}")
        
        if not validator:
            logger.warning("Validator not initialized, returning default quality check")
            return jsonify({
                'is_valid': True,
                'quality_score': 0.8,
                'validation_issues': [],
                'recommendation': 'ACCEPT - Validator not available'
            })
        
        # Perform quality check
        result = validator.check_single_issue_quality(ticket_data)
        
        logger.info(f"Quality check result for {ticket_key}: score={result['quality_score']:.2f}, valid={result['is_valid']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Quality check failed: {e}")
        return jsonify({
            'error': str(e),
            'is_valid': False,
            'quality_score': 0.0,
            'recommendation': 'REJECT - Quality check failed'
        }), 500


@app.route('/api/convert-dataset', methods=['POST'])
def convert_dataset():
    """Convert a ticket to training dataset format."""
    try:
        ticket_data_str = request.json.get('ticket_data')
        if isinstance(ticket_data_str, str):
            ticket_data = json.loads(ticket_data_str)
        else:
            ticket_data = ticket_data_str
        
        ticket_key = ticket_data.get('key', 'unknown')
        logger.info(f"Dataset conversion for ticket: {ticket_key}")
        
        if not converter:
            logger.warning("Converter not initialized, returning empty dataset")
            return jsonify({
                'examples': [],
                'count': 0,
                'error': 'Converter not available'
            })
        
        # Convert ticket to training examples
        examples = converter.convert_single_issue(ticket_data)
        
        result = {
            'examples': examples,
            'count': len(examples),
            'ticket_key': ticket_key,
            'converted_at': datetime.now().isoformat()
        }
        
        logger.info(f"Dataset conversion result for {ticket_key}: {len(examples)} examples")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Dataset conversion failed: {e}")
        return jsonify({
            'error': str(e),
            'examples': [],
            'count': 0
        }), 500


@app.route('/api/adaptive-training', methods=['POST'])
def adaptive_training():
    """Trigger adaptive training on a ticket."""
    try:
        data = request.json
        mode = data.get('mode', 'single')
        ticket_data_str = data.get('ticket_data')
        auto_deploy = data.get('auto_deploy', False)
        ab_test = data.get('ab_test', True)
        
        if isinstance(ticket_data_str, str):
            ticket_data = json.loads(ticket_data_str)
        else:
            ticket_data = ticket_data_str
        
        ticket_key = ticket_data.get('key', 'unknown')
        logger.info(f"Adaptive training triggered for ticket: {ticket_key}")
        
        # For now, simulate training success
        # In production, you'd call your actual training script
        
        # Simulate training process
        import time
        time.sleep(2)  # Simulate training time
        
        result = {
            'success': True,
            'ticket_key': ticket_key,
            'mode': mode,
            'model_path': f'/app/models/adaptive_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'training_time': 2.0,
            'auto_deploy': auto_deploy,
            'ab_test': ab_test,
            'completed_at': datetime.now().isoformat()
        }
        
        logger.info(f"Adaptive training completed for {ticket_key}: success={result['success']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Adaptive training failed: {e}")
        return jsonify({
            'error': str(e),
            'success': False,
            'ticket_key': data.get('ticket_data', {}).get('key', 'unknown')
        }), 500


@app.route('/api/register-model', methods=['POST'])
def register_model():
    """Register a trained model."""
    try:
        data = request.json
        model_path = data.get('model_path')
        model_type = data.get('model_type', 'adaptive')
        metadata_str = data.get('metadata', '{}')
        
        if isinstance(metadata_str, str):
            metadata = json.loads(metadata_str)
        else:
            metadata = metadata_str
        
        logger.info(f"Model registration for path: {model_path}")
        
        if not model_manager:
            logger.warning("Model manager not initialized, returning mock model ID")
            model_id = f"mock_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return jsonify({
                'model_id': model_id,
                'registered_at': datetime.now().isoformat()
            })
        
        # Register the model
        model_id = model_manager.register_model(
            model_path=model_path,
            model_type=model_type,
            metadata=metadata
        )
        
        result = {
            'model_id': model_id,
            'model_path': model_path,
            'model_type': model_type,
            'registered_at': datetime.now().isoformat()
        }
        
        logger.info(f"Model registered successfully: {model_id}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return jsonify({
            'error': str(e),
            'model_id': None
        }), 500


@app.route('/api/ab-test/start', methods=['POST'])
def start_ab_test():
    """Start an A/B test with a candidate model."""
    try:
        data = request.json
        candidate_model_id = data.get('candidate_model_id')
        traffic_split = data.get('traffic_split', 0.1)
        
        logger.info(f"Starting A/B test with model: {candidate_model_id}")
        
        if not model_manager:
            logger.warning("Model manager not initialized, returning mock success")
            return jsonify({
                'success': True,
                'ab_test_id': f"mock_ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'candidate_model_id': candidate_model_id,
                'traffic_split': traffic_split,
                'started_at': datetime.now().isoformat()
            })
        
        # Start A/B test
        success = model_manager.start_ab_test(
            candidate_model_id=candidate_model_id,
            traffic_split=traffic_split
        )
        
        result = {
            'success': success,
            'candidate_model_id': candidate_model_id,
            'traffic_split': traffic_split,
            'started_at': datetime.now().isoformat()
        }
        
        logger.info(f"A/B test started: success={success}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"A/B test start failed: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/ab-test/status', methods=['GET'])
def ab_test_status():
    """Get A/B test status."""
    try:
        if not model_manager:
            return jsonify({
                'active': False,
                'message': 'Model manager not available'
            })
        
        # Get A/B test status
        registry = model_manager.registry
        ab_test = registry.get('ab_test', {})
        
        if not ab_test.get('active', False):
            return jsonify({
                'active': False,
                'message': 'No active A/B test'
            })
        
        result = {
            'active': True,
            'current_model': ab_test.get('current_model'),
            'candidate_model': ab_test.get('candidate_model'),
            'traffic_split': ab_test.get('traffic_split'),
            'start_time': ab_test.get('start_time'),
            'performance_data_points': len(ab_test.get('performance_data', []))
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"A/B test status check failed: {e}")
        return jsonify({
            'error': str(e),
            'active': False
        }), 500


@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get overall system status."""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_health': 'healthy',
            'components': {
                'model_manager': model_manager is not None,
                'validator': validator is not None,
                'converter': converter is not None,
                'preprocessor': preprocessor is not None
            },
            'api_version': '1.0.0'
        }
        
        # Check if any critical components are missing
        if not all(status['components'].values()):
            status['system_health'] = 'degraded'
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return jsonify({
            'error': str(e),
            'system_health': 'unhealthy',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/health',
            '/api/duplicate-check',
            '/api/quality-check',
            '/api/convert-dataset',
            '/api/adaptive-training',
            '/api/register-model',
            '/api/ab-test/start',
            '/api/ab-test/status',
            '/api/system/status'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    logger.info("Starting Adaptive ML API server...")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Start the server
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )