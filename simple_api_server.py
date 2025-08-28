#!/usr/bin/env python3
"""
Simplified API server for n8n integration demo.

This server provides basic REST endpoints that n8n can call to simulate
the adaptive fine-tuning system without requiring full dependencies.
"""

import json
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("Starting Simplified Adaptive ML API server...")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-demo',
        'mode': 'demo'
    })

@app.route('/api/duplicate-check', methods=['POST'])
def duplicate_check():
    """Check if a ticket is a duplicate of existing tickets."""
    try:
        data = request.json
        summary = data.get('summary', '')
        description = data.get('description', '')
        ticket_key = data.get('ticket_key', 'unknown')
        
        print(f"Duplicate check for ticket: {ticket_key}")
        
        # Simulate duplicate detection logic
        # In demo mode, randomly decide if it's a duplicate
        import random
        is_duplicate = random.random() < 0.1  # 10% chance of being duplicate
        similarity_score = random.random() * 0.3 if not is_duplicate else random.random() * 0.4 + 0.6
        
        result = {
            'is_duplicate': is_duplicate,
            'similarity_score': similarity_score,
            'similar_tickets': ['DEMO-456', 'DEMO-789'] if is_duplicate else [],
            'ticket_key': ticket_key,
            'checked_at': datetime.now().isoformat()
        }
        
        print(f"Duplicate check result for {ticket_key}: {result['is_duplicate']}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Duplicate check failed: {e}")
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
        print(f"Quality check for ticket: {ticket_key}")
        
        # Simulate quality assessment
        summary = ticket_data.get('summary', '')
        description = ticket_data.get('description', '')
        
        # Calculate quality score based on content length and completeness
        quality_score = 0.0
        
        # Summary quality (30%)
        if len(summary) > 10:
            quality_score += 0.3
        elif len(summary) > 5:
            quality_score += 0.15
        
        # Description quality (50%)
        if len(description) > 50:
            quality_score += 0.5
        elif len(description) > 20:
            quality_score += 0.25
        
        # Basic fields (20%)
        if ticket_data.get('issueType'):
            quality_score += 0.1
        if ticket_data.get('priority'):
            quality_score += 0.1
        
        is_valid = quality_score >= 0.6
        
        if quality_score >= 0.8:
            recommendation = "ACCEPT - High quality issue, suitable for training"
        elif quality_score >= 0.6:
            recommendation = "ACCEPT_WITH_CAUTION - Acceptable quality"
        else:
            recommendation = "REJECT - Low quality, not suitable for training"
        
        result = {
            'is_valid': is_valid,
            'quality_score': quality_score,
            'validation_issues': [] if is_valid else ['Low quality content'],
            'recommendation': recommendation
        }
        
        print(f"Quality check result for {ticket_key}: score={quality_score:.2f}, valid={is_valid}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Quality check failed: {e}")
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
        print(f"Dataset conversion for ticket: {ticket_key}")
        
        # Simulate dataset conversion
        summary = ticket_data.get('summary', '')
        description = ticket_data.get('description', '')
        issue_type = ticket_data.get('issueType', 'Unknown')
        
        # Create example training examples
        examples = []
        
        # Example 1: Issue classification
        examples.append({
            'instruction': 'Classify this software issue type based on the summary and description.',
            'input': f"Summary: {summary}\nDescription: {description}",
            'output': issue_type
        })
        
        # Example 2: Problem analysis
        examples.append({
            'instruction': 'Analyze this software issue and provide insights about resolution approach.',
            'input': f"Summary: {summary}\nDescription: {description}",
            'output': f"This appears to be a {issue_type.lower()} that requires investigation of the described symptoms."
        })
        
        result = {
            'examples': examples,
            'count': len(examples),
            'ticket_key': ticket_key,
            'converted_at': datetime.now().isoformat()
        }
        
        print(f"Dataset conversion result for {ticket_key}: {len(examples)} examples")
        return jsonify(result)
        
    except Exception as e:
        print(f"Dataset conversion failed: {e}")
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
        print(f"Adaptive training triggered for ticket: {ticket_key}")
        
        # Simulate training process
        print(f"  - Mode: {mode}")
        print(f"  - Starting training...")
        time.sleep(2)  # Simulate training time
        print(f"  - Training completed successfully!")
        
        model_id = f"adaptive_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = f"/demo/models/{model_id}"
        
        result = {
            'success': True,
            'ticket_key': ticket_key,
            'mode': mode,
            'model_path': model_path,
            'model_id': model_id,
            'training_time': 2.0,
            'auto_deploy': auto_deploy,
            'ab_test': ab_test,
            'completed_at': datetime.now().isoformat()
        }
        
        print(f"Adaptive training completed for {ticket_key}: success={result['success']}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Adaptive training failed: {e}")
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
        
        print(f"Model registration for path: {model_path}")
        
        # Simulate model registration
        model_id = f"registered_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = {
            'model_id': model_id,
            'model_path': model_path,
            'model_type': model_type,
            'metadata': metadata,
            'registered_at': datetime.now().isoformat()
        }
        
        print(f"Model registered successfully: {model_id}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Model registration failed: {e}")
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
        
        print(f"Starting A/B test with model: {candidate_model_id}")
        print(f"  - Traffic split: {traffic_split * 100}% to candidate")
        
        # Simulate A/B test start
        ab_test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = {
            'success': True,
            'ab_test_id': ab_test_id,
            'candidate_model_id': candidate_model_id,
            'traffic_split': traffic_split,
            'started_at': datetime.now().isoformat()
        }
        
        print(f"A/B test started successfully: {ab_test_id}")
        return jsonify(result)
        
    except Exception as e:
        print(f"A/B test start failed: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get overall system status."""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'system_health': 'healthy',
        'mode': 'demo',
        'api_version': '1.0.0-demo',
        'endpoints_available': 8
    })

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
            '/api/system/status'
        ]
    }), 404

if __name__ == '__main__':
    print("🚀 Simplified Adaptive ML API server starting...")
    print("📋 Available endpoints:")
    print("   - GET  /health")
    print("   - POST /api/duplicate-check")
    print("   - POST /api/quality-check")
    print("   - POST /api/convert-dataset")
    print("   - POST /api/adaptive-training")
    print("   - POST /api/register-model")
    print("   - POST /api/ab-test/start")
    print("   - GET  /api/system/status")
    print()
    
    app.run(
        host='0.0.0.0',
        port=8001,
        debug=True
    )