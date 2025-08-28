#!/usr/bin/env python3
"""
Convert vLLM GitHub issues into training datasets for adaptive fine-tuning.
"""

import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VLLMDatasetConverter:
    """Converts vLLM GitHub issues into various training formats."""
    
    def __init__(self):
        self.qa_templates = {
            "error_diagnosis": "How to fix this vLLM error: {error}?",
            "model_compatibility": "Can vLLM run this model: {model}?",
            "configuration": "What vLLM configuration is needed for: {requirement}?",
            "troubleshooting": "How to resolve: {problem}?",
            "performance": "How to optimize vLLM for: {scenario}?"
        }
        
        self.classification_labels = {
            "hardware": ["gpu", "cuda", "memory", "tpu", "hardware"],
            "model": ["model", "loading", "checkpoint", "weights"],
            "performance": ["performance", "optimization", "speed", "memory"],
            "api": ["api", "openai", "serving", "server"],
            "configuration": ["config", "setup", "installation", "docker"],
            "error": ["error", "exception", "crash", "failure"]
        }
    
    def load_data(self, input_file: str) -> Dict:
        """Load vLLM issues data from JSON file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_error_info(self, text: str) -> Optional[Dict[str, str]]:
        """Extract error information from issue text."""
        patterns = {
            "error_type": r"(\w*Error|\w*Exception): (.+?)(?:\n|$)",
            "runtime_error": r"RuntimeError: (.+?)(?:\n|$)",
            "value_error": r"ValueError: (.+?)(?:\n|$)",
            "import_error": r"ImportError: (.+?)(?:\n|$)",
            "cuda_error": r"CUDA (?:error|Error): (.+?)(?:\n|$)",
            "http_error": r"HTTP (\d+)(?: (.+?))?(?:\n|$)"
        }
        
        errors = {}
        for error_type, pattern in patterns.items():
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                if error_type == "http_error":
                    errors[error_type] = f"HTTP {match.group(1)}" + (f" {match.group(2)}" if match.group(2) else "")
                else:
                    errors[error_type] = match.group(1) if len(match.groups()) == 1 else match.group(2)
        
        return errors if errors else None
    
    def extract_model_info(self, text: str) -> Optional[str]:
        """Extract model name/path from issue text."""
        patterns = [
            r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_\.]+)",  # huggingface format
            r"(Qwen[0-9\-A-Za-z]+)",  # Qwen models
            r"(gpt-oss[0-9\-A-Za-z]*)",  # gpt-oss models
            r"(Llama[0-9\-A-Za-z]*)",  # Llama models
            r"(Gemma[0-9\-A-Za-z]*)",  # Gemma models
            r"--model[=\s]+([\"']?)([^\"'\s]+)\1",  # CLI model argument
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(2) if len(match.groups()) >= 2 else match.group(1)
        
        return None
    
    def extract_environment_info(self, text: str) -> Dict[str, str]:
        """Extract environment information from issue text."""
        env_info = {}
        
        # GPU info
        gpu_match = re.search(r"GPU models and configuration.*?GPU\s+\d+:\s+(.+?)(?:\n|$)", text, re.MULTILINE)
        if gpu_match:
            env_info["gpu"] = gpu_match.group(1).strip()
        
        # CUDA version
        cuda_match = re.search(r"CUDA (?:runtime )?version\s*:\s*(.+?)(?:\n|$)", text, re.MULTILINE)
        if cuda_match:
            env_info["cuda"] = cuda_match.group(1).strip()
        
        # vLLM version
        vllm_match = re.search(r"vLLM Version\s*:\s*(.+?)(?:\n|$)", text, re.MULTILINE)
        if vllm_match:
            env_info["vllm_version"] = vllm_match.group(1).strip()
        
        # OS
        os_match = re.search(r"OS\s*:\s*(.+?)(?:\n|$)", text, re.MULTILINE)
        if os_match:
            env_info["os"] = os_match.group(1).strip()
        
        return env_info
    
    def classify_issue(self, issue: Dict) -> List[str]:
        """Classify issue based on title, body, and labels."""
        text = f"{issue['title']} {issue['body']}".lower()
        labels = [label.lower() for label in issue['labels']]
        
        categories = []
        
        # Check existing labels first
        for category, keywords in self.classification_labels.items():
            if any(keyword in labels for keyword in keywords):
                categories.append(category)
        
        # Check content for keywords
        for category, keywords in self.classification_labels.items():
            if category not in categories:  # Don't duplicate
                if any(keyword in text for keyword in keywords):
                    categories.append(category)
        
        return categories if categories else ["general"]
    
    def find_solution(self, issue: Dict) -> Optional[str]:
        """Extract solution from issue comments or resolution."""
        if issue['state'] != 'closed':
            return None
        
        # Look for solution in comments
        solution_patterns = [
            r"(?:solution|fix|workaround|resolved):\s*(.+?)(?:\n\n|\n$|$)",
            r"(?:try|use|add|set)\s+(.+?)(?:\n\n|\n$|$)",
            r"the (?:fix|solution) is (.+?)(?:\n\n|\n$|$)"
        ]
        
        all_text = issue['body']
        for comment in issue['comments']:
            all_text += "\n" + comment.get('body', '')
        
        for pattern in solution_patterns:
            match = re.search(pattern, all_text, re.MULTILINE | re.IGNORECASE)
            if match:
                solution = match.group(1).strip()
                if len(solution) > 10:  # Ensure meaningful solution
                    return solution
        
        # If closed but no explicit solution, look for the last meaningful comment
        if issue['comments']:
            last_comment = issue['comments'][-1].get('body', '')
            if len(last_comment) > 20:
                return last_comment[:200] + "..." if len(last_comment) > 200 else last_comment
        
        return None
    
    def create_qa_examples(self, issues: List[Dict]) -> List[Dict]:
        """Create Q&A training examples from issues."""
        qa_examples = []
        
        for issue in issues:
            if issue['state'] != 'closed':
                continue  # Only use resolved issues for Q&A
            
            errors = self.extract_error_info(issue['body'])
            model = self.extract_model_info(issue['body'])
            env_info = self.extract_environment_info(issue['body'])
            solution = self.find_solution(issue)
            
            if not solution:
                continue
            
            # Create different types of Q&A based on available info
            examples = []
            
            # Error diagnosis questions
            if errors:
                for error_type, error_msg in errors.items():
                    question = f"How to fix this vLLM error: {error_msg}?"
                    examples.append({
                        "instruction": question,
                        "input": f"Environment: {env_info}",
                        "output": solution,
                        "type": "error_diagnosis",
                        "metadata": {
                            "issue_number": issue['number'],
                            "error_type": error_type,
                            "model": model,
                            "environment": env_info
                        }
                    })
            
            # Model compatibility questions
            if model:
                question = f"How to run {model} with vLLM?"
                examples.append({
                    "instruction": question,
                    "input": f"Model: {model}\nEnvironment: {env_info}",
                    "output": solution,
                    "type": "model_compatibility",
                    "metadata": {
                        "issue_number": issue['number'],
                        "model": model,
                        "environment": env_info
                    }
                })
            
            # General troubleshooting
            if not errors and not model:
                question = f"How to resolve: {issue['title']}?"
                examples.append({
                    "instruction": question,
                    "input": issue['body'][:500] + "..." if len(issue['body']) > 500 else issue['body'],
                    "output": solution,
                    "type": "troubleshooting",
                    "metadata": {
                        "issue_number": issue['number'],
                        "environment": env_info
                    }
                })
            
            qa_examples.extend(examples)
        
        return qa_examples
    
    def create_classification_examples(self, issues: List[Dict]) -> List[Dict]:
        """Create classification training examples."""
        classification_examples = []
        
        for issue in issues:
            categories = self.classify_issue(issue)
            
            # Create text for classification
            text = f"Title: {issue['title']}\n"
            if issue['body']:
                text += f"Description: {issue['body'][:300]}..."
            
            example = {
                "instruction": "Classify this vLLM issue into categories:",
                "input": text,
                "output": ", ".join(categories),
                "type": "classification",
                "metadata": {
                    "issue_number": issue['number'],
                    "true_labels": issue['labels'],
                    "predicted_categories": categories
                }
            }
            
            classification_examples.append(example)
        
        return classification_examples
    
    def create_duplicate_detection_examples(self, issues: List[Dict]) -> List[Dict]:
        """Create duplicate detection examples by finding similar issues."""
        duplicate_examples = []
        
        # Group issues by similarity (simplified approach)
        for i, issue1 in enumerate(issues):
            for j, issue2 in enumerate(issues[i+1:], i+1):
                similarity_score = self.calculate_similarity(issue1, issue2)
                
                if similarity_score > 0.7:  # High similarity threshold
                    label = "duplicate"
                elif similarity_score < 0.3:  # Low similarity threshold
                    label = "not_duplicate"
                else:
                    continue  # Skip medium similarity
                
                example = {
                    "instruction": "Are these vLLM issues duplicates?",
                    "input": f"Issue 1: {issue1['title']}\n{issue1['body'][:200]}...\n\nIssue 2: {issue2['title']}\n{issue2['body'][:200]}...",
                    "output": label,
                    "type": "duplicate_detection",
                    "metadata": {
                        "issue1_number": issue1['number'],
                        "issue2_number": issue2['number'],
                        "similarity_score": similarity_score
                    }
                }
                
                duplicate_examples.append(example)
                
                # Limit to avoid too many examples
                if len(duplicate_examples) >= 100:
                    break
            
            if len(duplicate_examples) >= 100:
                break
        
        return duplicate_examples
    
    def calculate_similarity(self, issue1: Dict, issue2: Dict) -> float:
        """Calculate similarity between two issues (simplified)."""
        # Simple keyword-based similarity
        text1 = f"{issue1['title']} {issue1['body']}".lower()
        text2 = f"{issue2['title']} {issue2['body']}".lower()
        
        # Check for common error patterns
        errors1 = self.extract_error_info(text1)
        errors2 = self.extract_error_info(text2)
        
        if errors1 and errors2:
            # If both have same error type, high similarity
            if any(e1 in errors2.values() for e1 in errors1.values()):
                return 0.8
        
        # Check for common models
        model1 = self.extract_model_info(text1)
        model2 = self.extract_model_info(text2)
        
        if model1 and model2 and model1 == model2:
            return 0.6
        
        # Basic keyword overlap
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total if total > 0 else 0.0
    
    def split_by_timeline(self, issues: List[Dict], split_dates: List[str]) -> Dict[str, List[Dict]]:
        """Split issues by timeline for adaptive training."""
        splits = {f"period_{i}": [] for i in range(len(split_dates) + 1)}
        
        for issue in issues:
            issue_date = issue['created_at']
            
            period = 0
            for i, split_date in enumerate(split_dates):
                if issue_date > split_date:
                    period = i + 1
                else:
                    break
            
            splits[f"period_{period}"].append(issue)
        
        return splits
    
    def convert_to_training_format(
        self,
        input_file: str,
        output_dir: str,
        include_types: List[str] = None,
        timeline_splits: List[str] = None
    ) -> Dict[str, int]:
        """
        Convert vLLM issues to training datasets.
        
        Args:
            input_file: Path to vLLM issues JSON file
            output_dir: Directory to save training datasets
            include_types: List of example types to include ['qa', 'classification', 'duplicate']
            timeline_splits: List of ISO dates for timeline splitting
            
        Returns:
            Statistics about generated examples
        """
        if include_types is None:
            include_types = ['qa', 'classification', 'duplicate']
        
        # Load data
        data = self.load_data(input_file)
        issues = data['issues']
        
        logger.info(f"Loaded {len(issues)} issues from {input_file}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats = {"total_issues": len(issues)}
        
        # Split by timeline if requested
        if timeline_splits:
            timeline_data = self.split_by_timeline(issues, timeline_splits)
            
            for period, period_issues in timeline_data.items():
                logger.info(f"Processing {period}: {len(period_issues)} issues")
                
                period_stats = self._generate_examples(
                    period_issues, 
                    output_path / period, 
                    include_types
                )
                
                stats[period] = period_stats
        else:
            # Generate all examples
            example_stats = self._generate_examples(issues, output_path, include_types)
            stats.update(example_stats)
        
        # Save metadata
        metadata = {
            "source_file": input_file,
            "generation_date": datetime.now(timezone.utc).isoformat(),
            "include_types": include_types,
            "timeline_splits": timeline_splits,
            "statistics": stats
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset conversion complete! Statistics: {stats}")
        return stats
    
    def _generate_examples(self, issues: List[Dict], output_path: Path, include_types: List[str]) -> Dict[str, int]:
        """Generate training examples for a set of issues."""
        output_path.mkdir(parents=True, exist_ok=True)
        stats = {}
        
        if 'qa' in include_types:
            qa_examples = self.create_qa_examples(issues)
            stats['qa_examples'] = len(qa_examples)
            
            with open(output_path / "qa_examples.jsonl", 'w') as f:
                for example in qa_examples:
                    f.write(json.dumps(example) + "\n")
        
        if 'classification' in include_types:
            classification_examples = self.create_classification_examples(issues)
            stats['classification_examples'] = len(classification_examples)
            
            with open(output_path / "classification_examples.jsonl", 'w') as f:
                for example in classification_examples:
                    f.write(json.dumps(example) + "\n")
        
        if 'duplicate' in include_types:
            duplicate_examples = self.create_duplicate_detection_examples(issues)
            stats['duplicate_examples'] = len(duplicate_examples)
            
            with open(output_path / "duplicate_examples.jsonl", 'w') as f:
                for example in duplicate_examples:
                    f.write(json.dumps(example) + "\n")
        
        return stats


def main():
    """Example usage of the dataset converter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert vLLM issues to training datasets")
    parser.add_argument("--input", "-i", required=True,
                        help="Input vLLM issues JSON file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for training datasets")
    parser.add_argument("--types", nargs="+", 
                        choices=['qa', 'classification', 'duplicate'],
                        default=['qa', 'classification', 'duplicate'],
                        help="Types of examples to generate")
    parser.add_argument("--timeline-splits", nargs="+",
                        help="ISO dates for timeline splitting (e.g., 2024-01-01 2024-06-01)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    converter = VLLMDatasetConverter()
    
    stats = converter.convert_to_training_format(
        input_file=args.input,
        output_dir=args.output,
        include_types=args.types,
        timeline_splits=args.timeline_splits
    )
    
    print(f"\n📊 Dataset Generation Summary:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()