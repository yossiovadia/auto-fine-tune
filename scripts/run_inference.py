#!/usr/bin/env python3
"""
Inference script for adaptive Jira defect analysis system.

Demonstrates various use cases like error analysis, troubleshooting guidance,
and defect insights using the trained adaptive model.
"""

import sys
import argparse
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import ModelInference, ModelManager
from api import JiraClient, JiraDataExtractor
from utils import setup_logging, get_logger, config


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="Run inference on adaptive defect analysis model")
    parser.add_argument("--mode", choices=[
        'error-check', 'troubleshoot', 'analyze', 'priority', 'similar', 
        'resolution-time', 'interactive', 'latest-defect'
    ], required=True, help="Inference mode")
    
    parser.add_argument("--query", help="Query text for analysis")
    parser.add_argument("--ticket-key", help="Jira ticket key to analyze")
    parser.add_argument("--component", help="Component context for troubleshooting")
    parser.add_argument("--error-description", help="Error description for error-check mode")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=config.get('system.log_level', 'INFO'),
        log_file=config.get('paths.logs_dir', './logs') + '/inference.log'
    )
    logger = get_logger(__name__)
    
    logger.info(f"Starting inference in {args.mode} mode")
    
    try:
        # Initialize inference engine
        inference = ModelInference()
        
        # Run inference based on mode
        if args.mode == 'error-check':
            result = run_error_check(inference, args)
        elif args.mode == 'troubleshoot':
            result = run_troubleshooting(inference, args)
        elif args.mode == 'analyze':
            result = run_defect_analysis(inference, args)
        elif args.mode == 'priority':
            result = run_priority_assessment(inference, args)
        elif args.mode == 'similar':
            result = run_similar_issues(inference, args)
        elif args.mode == 'resolution-time':
            result = run_resolution_time_prediction(inference, args)
        elif args.mode == 'latest-defect':
            result = run_latest_defect_analysis(inference, args)
        elif args.mode == 'interactive':
            result = run_interactive_mode(inference)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Output results
        output_results(result, args.output)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def run_error_check(inference: ModelInference, args) -> dict:
    """Run error pattern checking."""
    logger = get_logger(__name__)
    
    error_description = args.error_description or args.query
    if not error_description:
        raise ValueError("Error description required for error-check mode")
    
    logger.info("Checking if error pattern is known...")
    
    result = inference.check_known_error(error_description)
    
    if result['success']:
        print("\n🔍 Error Analysis Results:")
        print("=" * 50)
        print(f"Error: {error_description[:100]}...")
        print(f"Model Response:\n{result['response']}")
        print(f"\nModel: {result['model_id']}")
        print(f"Response Time: {result['metrics']['generation_time']:.2f}s")
    else:
        print(f"❌ Error analysis failed: {result.get('error', 'Unknown error')}")
    
    return result


def run_troubleshooting(inference: ModelInference, args) -> dict:
    """Run troubleshooting guidance."""
    logger = get_logger(__name__)
    
    issue_description = args.query
    if not issue_description:
        raise ValueError("Issue description required for troubleshooting mode")
    
    logger.info("Generating troubleshooting guidance...")
    
    result = inference.suggest_troubleshooting(issue_description, args.component)
    
    if result['success']:
        print("\n🛠️ Troubleshooting Guidance:")
        print("=" * 50)
        print(f"Issue: {issue_description}")
        if args.component:
            print(f"Component: {args.component}")
        print(f"\nRecommended Steps:\n{result['response']}")
        print(f"\nModel: {result['model_id']}")
        print(f"Response Time: {result['metrics']['generation_time']:.2f}s")
    else:
        print(f"❌ Troubleshooting failed: {result.get('error', 'Unknown error')}")
    
    return result


def run_defect_analysis(inference: ModelInference, args) -> dict:
    """Run comprehensive defect analysis."""
    logger = get_logger(__name__)
    
    if args.ticket_key:
        # Analyze specific ticket
        issue_data = get_ticket_data(args.ticket_key)
        result = inference.analyze_defect(issue_data)
        
        if result['success']:
            print("\n📊 Defect Analysis Results:")
            print("=" * 50)
            print(f"Ticket: {args.ticket_key}")
            print(f"Summary: {issue_data.get('summary', 'N/A')}")
            print(f"\nAnalysis:\n{result['response']}")
            print(f"\nModel: {result['model_id']}")
            print(f"Response Time: {result['metrics']['generation_time']:.2f}s")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    elif args.query:
        # Analyze description directly
        issue_data = {'description': args.query, 'summary': 'User provided description'}
        result = inference.analyze_defect(issue_data)
        
        if result['success']:
            print("\n📊 Defect Analysis Results:")
            print("=" * 50)
            print(f"Description: {args.query[:100]}...")
            print(f"\nAnalysis:\n{result['response']}")
            print(f"\nModel: {result['model_id']}")
            print(f"Response Time: {result['metrics']['generation_time']:.2f}s")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    else:
        raise ValueError("Either --ticket-key or --query required for analysis mode")
    
    return result


def run_priority_assessment(inference: ModelInference, args) -> dict:
    """Run priority assessment."""
    logger = get_logger(__name__)
    
    issue_description = args.query
    if not issue_description:
        raise ValueError("Issue description required for priority assessment")
    
    logger.info("Assessing issue priority...")
    
    result = inference.assess_priority(issue_description)
    
    if result['success']:
        print("\n⚡ Priority Assessment:")
        print("=" * 50)
        print(f"Issue: {issue_description[:100]}...")
        print(f"\nPriority Assessment:\n{result['response']}")
        print(f"\nModel: {result['model_id']}")
        print(f"Response Time: {result['metrics']['generation_time']:.2f}s")
    else:
        print(f"❌ Priority assessment failed: {result.get('error', 'Unknown error')}")
    
    return result


def run_similar_issues(inference: ModelInference, args) -> dict:
    """Find similar historical issues."""
    logger = get_logger(__name__)
    
    issue_description = args.query
    if not issue_description:
        raise ValueError("Issue description required for similar issues mode")
    
    logger.info("Finding similar historical issues...")
    
    result = inference.find_similar_issues(issue_description)
    
    if result['success']:
        print("\n🔄 Similar Issues Analysis:")
        print("=" * 50)
        print(f"Issue: {issue_description[:100]}...")
        print(f"\nSimilar Issues:\n{result['response']}")
        print(f"\nModel: {result['model_id']}")
        print(f"Response Time: {result['metrics']['generation_time']:.2f}s")
    else:
        print(f"❌ Similar issues search failed: {result.get('error', 'Unknown error')}")
    
    return result


def run_resolution_time_prediction(inference: ModelInference, args) -> dict:
    """Predict resolution time."""
    logger = get_logger(__name__)
    
    if args.ticket_key:
        issue_data = get_ticket_data(args.ticket_key)
    elif args.query:
        issue_data = {
            'description': args.query,
            'summary': 'User provided description',
            'issue_type': 'Unknown',
            'priority': 'Unknown',
            'components': []
        }
    else:
        raise ValueError("Either --ticket-key or --query required for resolution time prediction")
    
    logger.info("Predicting resolution time...")
    
    result = inference.predict_resolution_time(issue_data)
    
    if result['success']:
        print("\n⏱️ Resolution Time Prediction:")
        print("=" * 50)
        if args.ticket_key:
            print(f"Ticket: {args.ticket_key}")
        print(f"Issue: {issue_data.get('summary', 'N/A')}")
        print(f"\nResolution Time Estimate:\n{result['response']}")
        print(f"\nModel: {result['model_id']}")
        print(f"Response Time: {result['metrics']['generation_time']:.2f}s")
    else:
        print(f"❌ Resolution time prediction failed: {result.get('error', 'Unknown error')}")
    
    return result


def run_latest_defect_analysis(inference: ModelInference, args) -> dict:
    """Analyze the latest defect created."""
    logger = get_logger(__name__)
    
    logger.info("Fetching and analyzing latest defect...")
    
    # Get latest defect from Jira
    jira_client = JiraClient()
    extractor = JiraDataExtractor(jira_client)
    
    # Get recent issues (last 1 day)
    df = extractor.extract_recent_data(days=1, max_issues=1)
    
    if df.empty:
        print("❌ No recent defects found")
        return {'success': False, 'error': 'No recent defects'}
    
    # Get the latest issue
    latest_issue = df.iloc[0].to_dict()
    
    # Analyze it
    result = inference.analyze_defect(latest_issue)
    
    if result['success']:
        print("\n🆕 Latest Defect Analysis:")
        print("=" * 50)
        print(f"Ticket: {latest_issue.get('key', 'N/A')}")
        print(f"Created: {latest_issue.get('created', 'N/A')}")
        print(f"Summary: {latest_issue.get('summary', 'N/A')}")
        print(f"Priority: {latest_issue.get('priority', 'N/A')}")
        print(f"\nAnalysis & Fix Suggestions:\n{result['response']}")
        print(f"\nModel: {result['model_id']}")
        print(f"Response Time: {result['metrics']['generation_time']:.2f}s")
    else:
        print(f"❌ Latest defect analysis failed: {result.get('error', 'Unknown error')}")
    
    return result


def run_interactive_mode(inference: ModelInference) -> dict:
    """Run interactive mode for multiple queries."""
    logger = get_logger(__name__)
    
    print("\n🤖 Interactive Defect Analysis Mode")
    print("=" * 50)
    print("Available commands:")
    print("  error: <description>     - Check if error is known")
    print("  troubleshoot: <issue>    - Get troubleshooting steps")
    print("  priority: <description>  - Assess priority level")
    print("  similar: <description>   - Find similar issues")
    print("  analyze: <description>   - Comprehensive analysis")
    print("  help                     - Show this help")
    print("  quit                     - Exit interactive mode")
    print("\nType your queries below:")
    
    results = []
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'help':
                print("Available commands: error, troubleshoot, priority, similar, analyze, help, quit")
                continue
            elif not user_input:
                continue
            
            # Parse command
            if ':' in user_input:
                command, query = user_input.split(':', 1)
                command = command.strip().lower()
                query = query.strip()
            else:
                command = 'analyze'
                query = user_input
            
            # Execute command
            if command == 'error':
                result = inference.check_known_error(query)
            elif command == 'troubleshoot':
                result = inference.suggest_troubleshooting(query)
            elif command == 'priority':
                result = inference.assess_priority(query)
            elif command == 'similar':
                result = inference.find_similar_issues(query)
            elif command == 'analyze':
                issue_data = {'description': query, 'summary': 'Interactive query'}
                result = inference.analyze_defect(issue_data)
            else:
                print(f"❌ Unknown command: {command}")
                continue
            
            # Display result
            if result['success']:
                print(f"\n💡 Response ({result['model_id']}):")
                print("-" * 40)
                print(result['response'])
                print(f"\n⏱️ Response time: {result['metrics']['generation_time']:.2f}s")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            results.append(result)
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return {'interactive_results': results, 'success': True}


def get_ticket_data(ticket_key: str) -> dict:
    """Get ticket data from Jira.
    
    Args:
        ticket_key: Jira ticket key
        
    Returns:
        Dictionary containing ticket data
    """
    logger = get_logger(__name__)
    
    logger.info(f"Fetching ticket data for: {ticket_key}")
    
    jira_client = JiraClient()
    extractor = JiraDataExtractor(jira_client)
    
    issue = jira_client.get_issue(ticket_key)
    issue_data = extractor.extract_issue_data(issue)
    
    return issue_data


def output_results(result: dict, output_file: str = None):
    """Output results to file or console.
    
    Args:
        result: Results dictionary
        output_file: Optional output file path
    """
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()