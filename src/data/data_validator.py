"""Data quality validation and preprocessing tools."""

import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from collections import Counter
import difflib

from ..utils import get_logger, config


class DataValidator:
    """Validate and ensure quality of Jira data for training."""
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = get_logger(__name__)
        self.min_description_length = config.get('dataset.min_description_length', 50)
        self.max_description_length = config.get('dataset.max_description_length', 2000)
    
    def validate_issue_data(self, issue_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single issue's data quality.
        
        Args:
            issue_data: Dictionary containing issue data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = ['key', 'summary', 'description', 'issue_type']
        for field in required_fields:
            if not issue_data.get(field):
                issues.append(f"Missing required field: {field}")
        
        # Validate description length
        description = issue_data.get('description', '')
        if len(description) < self.min_description_length:
            issues.append(f"Description too short: {len(description)} < {self.min_description_length}")
        elif len(description) > self.max_description_length:
            issues.append(f"Description too long: {len(description)} > {self.max_description_length}")
        
        # Check for meaningful content
        if description and self._is_low_quality_text(description):
            issues.append("Description appears to be low quality or template text")
        
        summary = issue_data.get('summary', '')
        if summary and self._is_low_quality_text(summary):
            issues.append("Summary appears to be low quality or template text")
        
        # Validate issue type
        valid_issue_types = ['Bug', 'Task', 'Story', 'Epic', 'Improvement', 'Sub-task']
        issue_type = issue_data.get('issue_type', '')
        if issue_type and issue_type not in valid_issue_types:
            self.logger.warning(f"Unusual issue type: {issue_type}")
        
        return len(issues) == 0, issues
    
    def _is_low_quality_text(self, text: str) -> bool:
        """Check if text appears to be low quality.
        
        Args:
            text: Text to evaluate
            
        Returns:
            True if text appears low quality
        """
        if not text or len(text.strip()) < 10:
            return True
        
        # Check for common template phrases
        template_phrases = [
            'please fill in',
            'todo:',
            'tbd',
            'to be determined',
            'placeholder',
            'example text',
            'lorem ipsum'
        ]
        
        text_lower = text.lower()
        for phrase in template_phrases:
            if phrase in text_lower:
                return True
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 5:
            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1]
            if most_common_count > len(words) * 0.3:  # More than 30% repetition
                return True
        
        # Check for minimal content
        unique_chars = len(set(text.lower().replace(' ', '')))
        if unique_chars < 10:  # Very few unique characters
            return True
        
        return False
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate entire dataset and provide quality report.
        
        Args:
            df: DataFrame containing issue data
            
        Returns:
            Dictionary containing validation results
        """
        total_issues = len(df)
        valid_issues = 0
        validation_issues = []
        
        for idx, row in df.iterrows():
            is_valid, issues = self.validate_issue_data(row.to_dict())
            if is_valid:
                valid_issues += 1
            else:
                validation_issues.append({
                    'index': idx,
                    'key': row.get('key', 'unknown'),
                    'issues': issues
                })
        
        # Calculate statistics
        description_lengths = df['description'].str.len().dropna()
        summary_lengths = df['summary'].str.len().dropna()
        
        report = {
            'total_issues': total_issues,
            'valid_issues': valid_issues,
            'invalid_issues': total_issues - valid_issues,
            'validation_rate': valid_issues / total_issues if total_issues > 0 else 0,
            'validation_issues': validation_issues[:10],  # First 10 issues
            'statistics': {
                'description_length': {
                    'mean': description_lengths.mean(),
                    'median': description_lengths.median(),
                    'std': description_lengths.std(),
                    'min': description_lengths.min(),
                    'max': description_lengths.max()
                },
                'summary_length': {
                    'mean': summary_lengths.mean(),
                    'median': summary_lengths.median(),
                    'std': summary_lengths.std(),
                    'min': summary_lengths.min(),
                    'max': summary_lengths.max()
                }
            }
        }
        
        self.logger.info(f"Dataset validation: {valid_issues}/{total_issues} issues valid ({report['validation_rate']:.2%})")
        
        return report
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by removing invalid issues and fixing common problems.
        
        Args:
            df: DataFrame containing issue data
            
        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)
        cleaned_df = df.copy()
        
        # Remove issues with missing critical fields
        cleaned_df = cleaned_df.dropna(subset=['summary', 'description'])
        
        # Filter by description length
        desc_lengths = cleaned_df['description'].str.len()
        cleaned_df = cleaned_df[
            (desc_lengths >= self.min_description_length) & 
            (desc_lengths <= self.max_description_length)
        ]
        
        # Remove duplicates based on summary and description
        cleaned_df = cleaned_df.drop_duplicates(subset=['summary', 'description'])
        
        # Clean text fields
        cleaned_df['summary'] = cleaned_df['summary'].apply(self._clean_text_field)
        cleaned_df['description'] = cleaned_df['description'].apply(self._clean_text_field)
        
        final_count = len(cleaned_df)
        removed_count = initial_count - final_count
        
        self.logger.info(f"Dataset cleaning: removed {removed_count} issues ({removed_count/initial_count:.2%})")
        self.logger.info(f"Final dataset size: {final_count} issues")
        
        return cleaned_df
    
    def _clean_text_field(self, text: str) -> str:
        """Clean individual text field.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ''
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove or replace common artifacts
        text = re.sub(r'\[cid:[^\]]+\]', '', text)  # Email artifacts
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags
        
        return text
    
    def detect_duplicates(self, df: pd.DataFrame, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Detect potential duplicate issues based on text similarity.
        
        Args:
            df: DataFrame containing issue data
            similarity_threshold: Minimum similarity score to consider duplicates
            
        Returns:
            List of potential duplicate groups
        """
        duplicates = []
        processed = set()
        
        for i, row1 in df.iterrows():
            if i in processed:
                continue
            
            similar_issues = [i]
            text1 = f"{row1['summary']} {row1['description']}"
            
            for j, row2 in df.iterrows():
                if i >= j or j in processed:
                    continue
                
                text2 = f"{row2['summary']} {row2['description']}"
                similarity = self._calculate_text_similarity(text1, text2)
                
                if similarity >= similarity_threshold:
                    similar_issues.append(j)
                    processed.add(j)
            
            if len(similar_issues) > 1:
                duplicates.append({
                    'similarity_score': similarity_threshold,
                    'issues': [
                        {
                            'index': idx,
                            'key': df.loc[idx, 'key'],
                            'summary': df.loc[idx, 'summary'][:100] + '...'
                        }
                        for idx in similar_issues
                    ]
                })
            
            processed.add(i)
        
        self.logger.info(f"Found {len(duplicates)} potential duplicate groups")
        return duplicates
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use SequenceMatcher for simple similarity
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def check_single_issue_quality(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality of a single new issue for real-time validation.
        
        Args:
            issue_data: Dictionary containing issue data
            
        Returns:
            Dictionary containing quality assessment
        """
        is_valid, issues = self.validate_issue_data(issue_data)
        
        # Additional checks for real-time processing
        quality_score = self._calculate_quality_score(issue_data)
        
        return {
            'is_valid': is_valid,
            'validation_issues': issues,
            'quality_score': quality_score,
            'recommendation': self._get_quality_recommendation(quality_score, is_valid)
        }
    
    def _calculate_quality_score(self, issue_data: Dict[str, Any]) -> float:
        """Calculate quality score for an issue (0-1).
        
        Args:
            issue_data: Dictionary containing issue data
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        max_score = 0.0
        
        # Summary quality (20%)
        max_score += 20
        summary = issue_data.get('summary', '')
        if summary:
            if len(summary) > 10 and not self._is_low_quality_text(summary):
                score += 20
            elif len(summary) > 5:
                score += 10
        
        # Description quality (40%)
        max_score += 40
        description = issue_data.get('description', '')
        if description:
            desc_len = len(description)
            if desc_len >= self.min_description_length and not self._is_low_quality_text(description):
                score += 40
            elif desc_len >= 30:
                score += 20
            elif desc_len >= 10:
                score += 10
        
        # Metadata completeness (20%)
        max_score += 20
        metadata_fields = ['issue_type', 'priority', 'components', 'labels']
        filled_fields = sum(1 for field in metadata_fields if issue_data.get(field))
        score += (filled_fields / len(metadata_fields)) * 20
        
        # Additional context (20%)
        max_score += 20
        if issue_data.get('comments'):
            score += 10
        if issue_data.get('assignee'):
            score += 5
        if issue_data.get('resolution'):
            score += 5
        
        return score / max_score if max_score > 0 else 0.0
    
    def _get_quality_recommendation(self, quality_score: float, is_valid: bool) -> str:
        """Get recommendation based on quality assessment.
        
        Args:
            quality_score: Quality score (0-1)
            is_valid: Whether the issue passes basic validation
            
        Returns:
            Recommendation string
        """
        if not is_valid:
            return "REJECT - Issue fails basic validation requirements"
        elif quality_score >= 0.8:
            return "ACCEPT - High quality issue, suitable for training"
        elif quality_score >= 0.6:
            return "ACCEPT_WITH_CAUTION - Acceptable quality, may benefit from preprocessing"
        elif quality_score >= 0.4:
            return "REVIEW - Low quality, manual review recommended"
        else:
            return "REJECT - Very low quality, not suitable for training"