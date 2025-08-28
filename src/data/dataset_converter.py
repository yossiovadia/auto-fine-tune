"""Convert Jira issue data to instruction-following datasets for fine-tuning."""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict

from ..utils import get_logger, config


class DatasetConverter:
    """Convert Jira issues to instruction-following format for LLM training."""
    
    def __init__(self):
        """Initialize dataset converter."""
        self.logger = get_logger(__name__)
        self.templates = self._load_instruction_templates()
    
    def _load_instruction_templates(self) -> Dict[str, List[str]]:
        """Load instruction templates for different tasks."""
        return {
            'classification': [
                "Classify this software issue type based on the summary and description.",
                "What type of issue is this? Analyze the summary and description to determine the issue type.",
                "Based on the following issue details, classify the type of software issue.",
                "Determine the category of this software issue from the provided information."
            ],
            'priority_prediction': [
                "Determine the priority level for this software issue.",
                "What priority should be assigned to this issue based on its description?",
                "Analyze this issue and predict its priority level.",
                "Based on the issue details, what priority level would you assign?"
            ],
            'resolution_prediction': [
                "Based on this issue description, predict the most likely resolution approach.",
                "What steps would you recommend to resolve this software issue?",
                "Analyze this bug report and suggest a resolution strategy.",
                "How would you approach resolving this software issue?"
            ],
            'component_assignment': [
                "Which component or area of the system does this issue relate to?",
                "Identify the component that should handle this issue.",
                "Based on the issue description, which system component is affected?",
                "Determine the appropriate component assignment for this issue."
            ],
            'summary_generation': [
                "Generate a concise summary for this issue based on its description.",
                "Create a brief summary that captures the essence of this issue.",
                "Write a clear, concise summary for this software issue.",
                "Summarize this issue in a clear and brief manner."
            ],
            'similar_issue_analysis': [
                "Analyze this issue and identify potential patterns or similarities with common problems.",
                "What category of problems does this issue represent?",
                "Identify the underlying problem pattern in this issue.",
                "Analyze the root cause category for this software issue."
            ]
        }
    
    def convert_single_issue(self, issue_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert a single Jira issue to multiple training examples.
        
        Args:
            issue_data: Dictionary containing issue data from JiraDataExtractor
            
        Returns:
            List of instruction-following training examples
        """
        examples = []
        
        # Create input context from issue data
        input_context = self._create_input_context(issue_data)
        
        if not input_context.strip():
            self.logger.warning(f"Empty input context for issue {issue_data.get('key', 'unknown')}")
            return examples
        
        # Issue type classification
        if issue_data.get('issue_type'):
            examples.extend(self._create_classification_examples(input_context, issue_data))
        
        # Priority prediction
        if issue_data.get('priority'):
            examples.extend(self._create_priority_examples(input_context, issue_data))
        
        # Resolution prediction (if resolved)
        if issue_data.get('resolution') and issue_data.get('resolution') != 'Unresolved':
            examples.extend(self._create_resolution_examples(input_context, issue_data))
        
        # Component assignment
        if issue_data.get('components'):
            examples.extend(self._create_component_examples(input_context, issue_data))
        
        # Summary generation (if description is longer than summary)
        if (issue_data.get('description') and issue_data.get('summary') and 
            len(issue_data['description']) > len(issue_data['summary']) * 2):
            examples.extend(self._create_summary_examples(input_context, issue_data))
        
        return examples
    
    def _create_input_context(self, issue_data: Dict[str, Any]) -> str:
        """Create input context from issue data.
        
        Args:
            issue_data: Issue data dictionary
            
        Returns:
            Formatted input context string
        """
        context_parts = []
        
        # Basic information
        if issue_data.get('summary'):
            context_parts.append(f"Summary: {issue_data['summary']}")
        
        if issue_data.get('description'):
            context_parts.append(f"Description: {issue_data['description']}")
        
        # Project context
        if issue_data.get('project_name'):
            context_parts.append(f"Project: {issue_data['project_name']}")
        
        # Labels and components for additional context
        if issue_data.get('labels'):
            context_parts.append(f"Labels: {', '.join(issue_data['labels'])}")
        
        # Recent comments for resolution context
        comments = issue_data.get('comments', [])
        if comments:
            latest_comments = comments[-2:]  # Last 2 comments
            comment_text = []
            for comment in latest_comments:
                if comment.get('body'):
                    comment_text.append(f"{comment.get('author', 'User')}: {comment['body'][:200]}...")
            if comment_text:
                context_parts.append(f"Recent Comments: {' | '.join(comment_text)}")
        
        return '\n'.join(context_parts)
    
    def _create_classification_examples(self, input_context: str, issue_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create issue type classification examples."""
        examples = []
        templates = self.templates['classification']
        
        # Use multiple templates for variety
        for template in random.sample(templates, min(2, len(templates))):
            examples.append({
                'instruction': template,
                'input': input_context,
                'output': issue_data['issue_type']
            })
        
        return examples
    
    def _create_priority_examples(self, input_context: str, issue_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create priority prediction examples."""
        examples = []
        templates = self.templates['priority_prediction']
        
        template = random.choice(templates)
        examples.append({
            'instruction': template,
            'input': input_context,
            'output': issue_data['priority']
        })
        
        return examples
    
    def _create_resolution_examples(self, input_context: str, issue_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create resolution prediction examples."""
        examples = []
        templates = self.templates['resolution_prediction']
        
        # Create a more detailed resolution output
        resolution_parts = [issue_data['resolution']]
        
        # Add relevant comments that led to resolution
        comments = issue_data.get('comments', [])
        resolution_comments = []
        for comment in comments[-3:]:  # Last 3 comments
            body = comment.get('body', '')
            if any(keyword in body.lower() for keyword in ['fix', 'resolve', 'solution', 'workaround']):
                resolution_comments.append(body[:150])
        
        if resolution_comments:
            resolution_parts.append(f"Resolution approach: {' '.join(resolution_comments)}")
        
        template = random.choice(templates)
        examples.append({
            'instruction': template,
            'input': input_context,
            'output': '. '.join(resolution_parts)
        })
        
        return examples
    
    def _create_component_examples(self, input_context: str, issue_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create component assignment examples."""
        examples = []
        templates = self.templates['component_assignment']
        
        components = issue_data['components']
        if isinstance(components, list):
            component_output = ', '.join(components)
        else:
            component_output = str(components)
        
        template = random.choice(templates)
        examples.append({
            'instruction': template,
            'input': input_context,
            'output': component_output
        })
        
        return examples
    
    def _create_summary_examples(self, input_context: str, issue_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create summary generation examples."""
        examples = []
        templates = self.templates['summary_generation']
        
        # Use description as input and summary as output
        description_only = f"Description: {issue_data['description']}"
        if issue_data.get('project_name'):
            description_only = f"Project: {issue_data['project_name']}\n{description_only}"
        
        template = random.choice(templates)
        examples.append({
            'instruction': template,
            'input': description_only,
            'output': issue_data['summary']
        })
        
        return examples
    
    def convert_dataframe(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Convert a DataFrame of issues to training examples.
        
        Args:
            df: DataFrame containing issue data
            
        Returns:
            List of training examples
        """
        all_examples = []
        
        for _, row in df.iterrows():
            issue_data = row.to_dict()
            examples = self.convert_single_issue(issue_data)
            all_examples.extend(examples)
        
        self.logger.info(f"Converted {len(df)} issues to {len(all_examples)} training examples")
        return all_examples
    
    def create_dataset(
        self,
        df: pd.DataFrame,
        train_split: float = None,
        val_split: float = None,
        test_split: float = None,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> DatasetDict:
        """Create HuggingFace dataset from issue DataFrame.
        
        Args:
            df: DataFrame containing issue data
            train_split: Training split ratio
            val_split: Validation split ratio
            test_split: Test split ratio
            shuffle: Whether to shuffle the data
            random_seed: Random seed for reproducibility
            
        Returns:
            HuggingFace DatasetDict with splits
        """
        # Get split ratios from config if not provided
        train_split = train_split or config.get('dataset.train_split', 0.8)
        val_split = val_split or config.get('dataset.val_split', 0.1)
        test_split = test_split or config.get('dataset.test_split', 0.1)
        
        # Validate splits sum to 1.0
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 0.001:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")
        
        # Convert to training examples
        examples = self.convert_dataframe(df)
        
        if not examples:
            raise ValueError("No training examples generated from the input data")
        
        # Shuffle if requested
        if shuffle:
            random.seed(random_seed)
            random.shuffle(examples)
        
        # Calculate split indices
        total_examples = len(examples)
        train_end = int(total_examples * train_split)
        val_end = train_end + int(total_examples * val_split)
        
        # Create splits
        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]
        
        # Create HuggingFace datasets
        datasets = {
            'train': Dataset.from_list(train_examples),
            'validation': Dataset.from_list(val_examples) if val_examples else None,
            'test': Dataset.from_list(test_examples) if test_examples else None
        }
        
        # Remove None datasets
        datasets = {k: v for k, v in datasets.items() if v is not None}
        
        self.logger.info(f"Created dataset with {len(datasets)} splits:")
        for split_name, dataset in datasets.items():
            self.logger.info(f"  {split_name}: {len(dataset)} examples")
        
        return DatasetDict(datasets)
    
    def save_dataset(self, dataset: DatasetDict, output_dir: str):
        """Save dataset to disk.
        
        Args:
            dataset: HuggingFace DatasetDict
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset.save_to_disk(str(output_path))
        self.logger.info(f"Dataset saved to: {output_path}")
    
    def load_dataset(self, dataset_dir: str) -> DatasetDict:
        """Load dataset from disk.
        
        Args:
            dataset_dir: Directory containing saved dataset
            
        Returns:
            Loaded HuggingFace DatasetDict
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        dataset = DatasetDict.load_from_disk(str(dataset_path))
        self.logger.info(f"Dataset loaded from: {dataset_path}")
        
        return dataset
    
    def convert_single_ticket_to_dataset(self, issue_data: Dict[str, Any]) -> Dataset:
        """Convert a single new ticket to a dataset for incremental training.
        
        Args:
            issue_data: Dictionary containing issue data
            
        Returns:
            HuggingFace Dataset with examples from the single ticket
        """
        examples = self.convert_single_issue(issue_data)
        
        if not examples:
            self.logger.warning(f"No examples generated for issue {issue_data.get('key', 'unknown')}")
            return Dataset.from_list([])
        
        dataset = Dataset.from_list(examples)
        self.logger.info(f"Created dataset with {len(dataset)} examples from single ticket")
        
        return dataset
    
    def get_dataset_stats(self, dataset: DatasetDict) -> Dict[str, Any]:
        """Get statistics about the dataset.
        
        Args:
            dataset: HuggingFace DatasetDict
            
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {}
        
        for split_name, split_dataset in dataset.items():
            split_stats = {
                'num_examples': len(split_dataset),
                'avg_input_length': 0,
                'avg_output_length': 0,
                'unique_instructions': 0
            }
            
            if len(split_dataset) > 0:
                # Calculate average lengths
                input_lengths = [len(ex['input']) for ex in split_dataset]
                output_lengths = [len(ex['output']) for ex in split_dataset]
                
                split_stats['avg_input_length'] = sum(input_lengths) / len(input_lengths)
                split_stats['avg_output_length'] = sum(output_lengths) / len(output_lengths)
                
                # Count unique instructions
                instructions = [ex['instruction'] for ex in split_dataset]
                split_stats['unique_instructions'] = len(set(instructions))
            
            stats[split_name] = split_stats
        
        return stats