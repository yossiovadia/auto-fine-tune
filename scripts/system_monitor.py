#!/usr/bin/env python3
"""
System monitoring and management script for adaptive Jira defect analysis system.

Provides status monitoring, A/B test management, and system health checks.
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import ModelManager, ModelInference
from utils import setup_logging, get_logger, config


def main():
    """Main monitoring script."""
    parser = argparse.ArgumentParser(description="Monitor and manage adaptive defect analysis system")
    parser.add_argument("--action", choices=[
        'status', 'ab-test-status', 'ab-test-evaluate', 'ab-test-promote', 
        'ab-test-rollback', 'models-list', 'model-info', 'health-check',
        'performance-test'
    ], required=True, help="Action to perform")
    
    parser.add_argument("--model-id", help="Model ID for model-specific actions")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor (for status)")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval in seconds")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=config.get('system.log_level', 'INFO'),
        log_file=config.get('paths.logs_dir', './logs') + '/monitoring.log'
    )
    logger = get_logger(__name__)
    
    logger.info(f"Starting system monitoring - action: {args.action}")
    
    try:
        if args.action == 'status':
            if args.watch:
                watch_system_status(args.interval)
            else:
                result = get_system_status()
                display_status(result)
        
        elif args.action == 'ab-test-status':
            result = get_ab_test_status()
            display_ab_test_status(result)
        
        elif args.action == 'ab-test-evaluate':
            result = evaluate_ab_test()
            display_ab_test_evaluation(result)
        
        elif args.action == 'ab-test-promote':
            result = promote_ab_test()
            print(result['message'])
        
        elif args.action == 'ab-test-rollback':
            result = rollback_deployment()
            print(result['message'])
        
        elif args.action == 'models-list':
            result = list_models()
            display_models_list(result)
        
        elif args.action == 'model-info':
            if not args.model_id:
                raise ValueError("--model-id required for model-info action")
            result = get_model_info(args.model_id)
            display_model_info(result)
        
        elif args.action == 'health-check':
            result = perform_health_check()
            display_health_check(result)
        
        elif args.action == 'performance-test':
            result = run_performance_test()
            display_performance_test(result)
        
        else:
            raise ValueError(f"Unknown action: {args.action}")
        
        # Save results if output file specified
        if args.output and 'result' in locals():
            save_results(result, args.output)
    
    except Exception as e:
        logger.error(f"Monitoring action failed: {e}")
        raise


def get_system_status() -> dict:
    """Get overall system status."""
    logger = get_logger(__name__)
    
    model_manager = ModelManager()
    inference = ModelInference(model_manager)
    
    # Get registry status
    registry_status = model_manager.get_registry_status()
    
    # Get inference status
    inference_status = inference.get_model_status()
    
    # Get A/B test status
    ab_test_active = registry_status['ab_test']['active']
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'system_health': 'healthy',  # Will be updated based on checks
        'models': {
            'total': registry_status['total_models'],
            'current_deployed': registry_status['deployments']['current'],
            'candidate_deployed': registry_status['deployments']['candidate'],
            'baseline_available': registry_status['deployments']['baseline'] is not None
        },
        'ab_test': {
            'active': ab_test_active,
            'traffic_split': registry_status['ab_test'].get('traffic_split', 0) if ab_test_active else 0
        },
        'inference': {
            'model_loaded': inference_status['model_loaded'],
            'current_model': inference_status['current_model_id'],
            'device': inference_status['device']
        }
    }
    
    # Determine health status
    issues = []
    if not registry_status['deployments']['current']:
        issues.append('No current model deployed')
        status['system_health'] = 'warning'
    
    if not inference_status['model_loaded']:
        issues.append('No model loaded for inference')
        status['system_health'] = 'warning'
    
    if issues:
        status['issues'] = issues
    
    return status


def watch_system_status(interval: int):
    """Continuously monitor system status."""
    logger = get_logger(__name__)
    
    print(f"🔄 Monitoring system status (interval: {interval}s)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            status = get_system_status()
            
            # Clear screen and display status
            print("\033[2J\033[H", end="")  # Clear screen
            print(f"🤖 Adaptive Jira Defect Analysis System Status")
            print(f"Last updated: {status['timestamp']}")
            print("=" * 60)
            
            display_status(status)
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def display_status(status: dict):
    """Display system status."""
    health_icon = "🟢" if status['system_health'] == 'healthy' else "🟡"
    print(f"\n{health_icon} System Health: {status['system_health'].upper()}")
    
    if 'issues' in status:
        print("⚠️  Issues:")
        for issue in status['issues']:
            print(f"   - {issue}")
    
    print(f"\n📊 Models:")
    print(f"   Total: {status['models']['total']}")
    print(f"   Current: {status['models']['current_deployed'] or 'None'}")
    print(f"   Candidate: {status['models']['candidate_deployed'] or 'None'}")
    print(f"   Baseline: {'Available' if status['models']['baseline_available'] else 'None'}")
    
    print(f"\n🔄 A/B Testing:")
    if status['ab_test']['active']:
        print(f"   Status: Active ({status['ab_test']['traffic_split']:.1%} to candidate)")
    else:
        print(f"   Status: Inactive")
    
    print(f"\n💻 Inference:")
    print(f"   Model Loaded: {'Yes' if status['inference']['model_loaded'] else 'No'}")
    print(f"   Current Model: {status['inference']['current_model'] or 'None'}")
    print(f"   Device: {status['inference']['device']}")


def get_ab_test_status() -> dict:
    """Get A/B test status."""
    model_manager = ModelManager()
    registry = model_manager.registry
    
    if not registry['ab_test']['active']:
        return {
            'active': False,
            'message': 'No active A/B test'
        }
    
    ab_test = registry['ab_test']
    
    # Calculate test duration
    start_time = datetime.fromisoformat(ab_test['start_time'])
    duration = datetime.now() - start_time
    
    return {
        'active': True,
        'current_model': ab_test['current_model'],
        'candidate_model': ab_test['candidate_model'],
        'traffic_split': ab_test['traffic_split'],
        'start_time': ab_test['start_time'],
        'duration_hours': duration.total_seconds() / 3600,
        'performance_data_points': len(ab_test['performance_data'])
    }


def display_ab_test_status(status: dict):
    """Display A/B test status."""
    print("\n🧪 A/B Test Status:")
    print("=" * 40)
    
    if not status['active']:
        print(status['message'])
        return
    
    print(f"Status: Active")
    print(f"Current Model: {status['current_model']}")
    print(f"Candidate Model: {status['candidate_model']}")
    print(f"Traffic Split: {status['traffic_split']:.1%} to candidate")
    print(f"Running Time: {status['duration_hours']:.1f} hours")
    print(f"Data Points: {status['performance_data_points']}")


def evaluate_ab_test() -> dict:
    """Evaluate A/B test performance."""
    model_manager = ModelManager()
    evaluation = model_manager.evaluate_ab_test()
    
    return evaluation


def display_ab_test_evaluation(evaluation: dict):
    """Display A/B test evaluation results."""
    print("\n📈 A/B Test Evaluation:")
    print("=" * 40)
    
    if evaluation['status'] == 'no_active_test':
        print("No active A/B test to evaluate")
        return
    
    if evaluation['status'] == 'insufficient_data':
        print("Insufficient data for evaluation")
        return
    
    print(f"Current Model: {evaluation['current_model']}")
    print(f"Candidate Model: {evaluation['candidate_model']}")
    print(f"Data Points: {evaluation['data_points']}")
    
    print(f"\nPerformance Comparison:")
    current = evaluation['current_metrics']
    candidate = evaluation['candidate_metrics']
    
    print(f"   Accuracy:     Current: {current['accuracy']:.3f}, Candidate: {candidate['accuracy']:.3f}")
    print(f"   Response Time: Current: {current['response_time']:.3f}s, Candidate: {candidate['response_time']:.3f}s")
    print(f"   Satisfaction: Current: {current['user_satisfaction']:.3f}, Candidate: {candidate['user_satisfaction']:.3f}")
    
    recommendation = evaluation['recommendation']
    rec_icon = "🚀" if recommendation == 'promote_candidate' else "⚠️" if recommendation == 'rollback' else "⏳"
    print(f"\n{rec_icon} Recommendation: {recommendation.replace('_', ' ').title()}")


def promote_ab_test() -> dict:
    """Promote candidate model to current."""
    model_manager = ModelManager()
    
    if not model_manager.registry['ab_test']['active']:
        return {'success': False, 'message': '❌ No active A/B test to promote'}
    
    success = model_manager.promote_candidate()
    
    if success:
        return {'success': True, 'message': '🚀 Candidate model promoted to current deployment'}
    else:
        return {'success': False, 'message': '❌ Failed to promote candidate model'}


def rollback_deployment() -> dict:
    """Rollback to baseline model."""
    model_manager = ModelManager()
    
    try:
        success = model_manager.rollback_deployment()
        
        if success:
            return {'success': True, 'message': '⏪ Rolled back to baseline model'}
        else:
            return {'success': False, 'message': '❌ Rollback failed'}
    
    except ValueError as e:
        return {'success': False, 'message': f'❌ Rollback failed: {e}'}


def list_models() -> dict:
    """List all registered models."""
    model_manager = ModelManager()
    models = model_manager.list_models()
    
    return {'models': models}


def display_models_list(result: dict):
    """Display list of models."""
    models = result['models']
    
    print(f"\n📋 Registered Models ({len(models)} total):")
    print("=" * 80)
    
    if not models:
        print("No models registered")
        return
    
    for model in models:
        status_icon = "🟢" if 'deployed' in model['status'] else "⚪"
        print(f"\n{status_icon} {model['id']}")
        print(f"   Type: {model['type']}")
        print(f"   Status: {model['status']}")
        print(f"   Registered: {model['registered_at'][:19]}")
        
        if model.get('training_metadata'):
            metadata = model['training_metadata']
            if metadata.get('is_adaptive'):
                print(f"   Generation: {metadata.get('generation_count', 'N/A')}")


def get_model_info(model_id: str) -> dict:
    """Get detailed model information."""
    model_manager = ModelManager()
    model_info = model_manager.get_model_info(model_id)
    
    return model_info


def display_model_info(model_info: dict):
    """Display detailed model information."""
    print(f"\n📋 Model Information: {model_info['id']}")
    print("=" * 60)
    
    print(f"Type: {model_info['type']}")
    print(f"Status: {model_info['status']}")
    print(f"Path: {model_info['path']}")
    print(f"Registered: {model_info['registered_at']}")
    
    if 'deployed_at' in model_info:
        print(f"Deployed: {model_info['deployed_at']}")
    
    if model_info.get('training_metadata'):
        print(f"\nTraining Metadata:")
        metadata = model_info['training_metadata']
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    
    if model_info.get('performance_metrics'):
        print(f"\nPerformance Metrics:")
        for key, value in model_info['performance_metrics'].items():
            print(f"   {key}: {value}")


def perform_health_check() -> dict:
    """Perform comprehensive health check."""
    logger = get_logger(__name__)
    
    checks = {
        'timestamp': datetime.now().isoformat(),
        'overall_health': 'healthy',
        'checks': {}
    }
    
    # Check model manager
    try:
        model_manager = ModelManager()
        registry_status = model_manager.get_registry_status()
        checks['checks']['model_manager'] = {
            'status': 'pass',
            'details': f"{registry_status['total_models']} models registered"
        }
    except Exception as e:
        checks['checks']['model_manager'] = {
            'status': 'fail',
            'error': str(e)
        }
        checks['overall_health'] = 'unhealthy'
    
    # Check inference engine
    try:
        inference = ModelInference()
        status = inference.get_model_status()
        checks['checks']['inference_engine'] = {
            'status': 'pass' if status['model_loaded'] else 'warning',
            'details': f"Model loaded: {status['model_loaded']}, Device: {status['device']}"
        }
        if not status['model_loaded'] and checks['overall_health'] == 'healthy':
            checks['overall_health'] = 'warning'
    except Exception as e:
        checks['checks']['inference_engine'] = {
            'status': 'fail',
            'error': str(e)
        }
        checks['overall_health'] = 'unhealthy'
    
    # Check configuration
    try:
        config.validate()
        checks['checks']['configuration'] = {
            'status': 'pass',
            'details': 'Configuration valid'
        }
    except Exception as e:
        checks['checks']['configuration'] = {
            'status': 'fail',
            'error': str(e)
        }
        checks['overall_health'] = 'unhealthy'
    
    return checks


def display_health_check(checks: dict):
    """Display health check results."""
    health_icon = "🟢" if checks['overall_health'] == 'healthy' else "🟡" if checks['overall_health'] == 'warning' else "🔴"
    
    print(f"\n{health_icon} System Health Check")
    print(f"Overall Status: {checks['overall_health'].upper()}")
    print(f"Timestamp: {checks['timestamp']}")
    print("=" * 50)
    
    for check_name, result in checks['checks'].items():
        status = result['status']
        icon = "✅" if status == 'pass' else "⚠️" if status == 'warning' else "❌"
        
        print(f"\n{icon} {check_name.replace('_', ' ').title()}: {status.upper()}")
        
        if 'details' in result:
            print(f"   {result['details']}")
        
        if 'error' in result:
            print(f"   Error: {result['error']}")


def run_performance_test() -> dict:
    """Run basic performance test."""
    logger = get_logger(__name__)
    
    inference = ModelInference()
    
    test_queries = [
        "Database connection timeout error",
        "Authentication failure after login",
        "Memory leak in user service",
        "API rate limiting issues"
    ]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_results': [],
        'summary': {}
    }
    
    total_time = 0
    successful_queries = 0
    
    for i, query in enumerate(test_queries):
        logger.info(f"Running performance test {i+1}/{len(test_queries)}")
        
        start_time = time.time()
        result = inference.check_known_error(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        total_time += response_time
        
        test_result = {
            'query': query,
            'success': result['success'],
            'response_time': response_time,
            'model_id': result.get('model_id'),
            'tokens_generated': result.get('metrics', {}).get('output_tokens', 0)
        }
        
        if result['success']:
            successful_queries += 1
        
        results['test_results'].append(test_result)
    
    # Calculate summary statistics
    results['summary'] = {
        'total_queries': len(test_queries),
        'successful_queries': successful_queries,
        'success_rate': successful_queries / len(test_queries),
        'avg_response_time': total_time / len(test_queries),
        'total_time': total_time
    }
    
    return results


def display_performance_test(results: dict):
    """Display performance test results."""
    summary = results['summary']
    
    print(f"\n⚡ Performance Test Results")
    print(f"Timestamp: {results['timestamp']}")
    print("=" * 50)
    
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Successful: {summary['successful_queries']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Average Response Time: {summary['avg_response_time']:.2f}s")
    print(f"Total Time: {summary['total_time']:.2f}s")
    
    print(f"\nDetailed Results:")
    for i, result in enumerate(results['test_results'], 1):
        status_icon = "✅" if result['success'] else "❌"
        print(f"  {i}. {status_icon} {result['response_time']:.2f}s - {result['query'][:50]}...")


def save_results(results: dict, output_file: str):
    """Save results to file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()