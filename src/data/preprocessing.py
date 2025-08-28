"""Text preprocessing utilities for Jira data."""

import re
import html
from typing import List, Dict, Any, Optional
import unicodedata

from ..utils import get_logger


class TextPreprocessor:
    """Preprocess text content from Jira issues for ML training."""
    
    def __init__(self):
        """Initialize text preprocessor."""
        self.logger = get_logger(__name__)
        
        # Common patterns to clean
        self.patterns = {
            'jira_user': re.compile(r'\[~[^\]]+\]'),  # User mentions [~username]
            'jira_link': re.compile(r'\[([^\]]*)\|[^\]]*\]'),  # Links [text|url]
            'jira_code': re.compile(r'\{code(?::[^}]*)?\}(.*?)\{code\}', re.DOTALL),
            'jira_quote': re.compile(r'\{quote\}(.*?)\{quote\}', re.DOTALL),
            'jira_panel': re.compile(r'\{panel(?::[^}]*)?\}(.*?)\{panel\}', re.DOTALL),
            'jira_noformat': re.compile(r'\{noformat\}(.*?)\{noformat\}', re.DOTALL),
            'html_tags': re.compile(r'<[^>]+>'),
            'email_artifacts': re.compile(r'\[cid:[^\]]+\]'),
            'multiple_spaces': re.compile(r'\s+'),
            'stack_trace': re.compile(r'(at [a-zA-Z0-9.$_]+\([^)]*\))', re.MULTILINE),
        }
    
    def clean_jira_text(self, text: str) -> str:
        """Clean Jira markup and formatting from text.
        
        Args:
            text: Raw text with Jira markup
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ''
        
        # HTML decode first
        text = html.unescape(text)
        
        # Handle code blocks - preserve content but clean formatting
        text = self.patterns['jira_code'].sub(r'```\1```', text)
        text = self.patterns['jira_noformat'].sub(r'```\1```', text)
        
        # Handle quotes - preserve content
        text = self.patterns['jira_quote'].sub(r'"\1"', text)
        
        # Handle panels - preserve content
        text = self.patterns['jira_panel'].sub(r'\1', text)
        
        # Remove user mentions
        text = self.patterns['jira_user'].sub('', text)
        
        # Convert links to just the text part
        text = self.patterns['jira_link'].sub(r'\1', text)
        
        # Remove HTML tags
        text = self.patterns['html_tags'].sub('', text)
        
        # Remove email artifacts
        text = self.patterns['email_artifacts'].sub('', text)
        
        # Normalize whitespace
        text = self.patterns['multiple_spaces'].sub(' ', text)
        text = text.strip()
        
        return text
    
    def extract_error_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract error patterns and stack traces from text.
        
        Args:
            text: Text containing potential error information
            
        Returns:
            Dictionary containing extracted error patterns
        """
        patterns = {
            'exceptions': [],
            'stack_traces': [],
            'error_codes': [],
            'error_messages': []
        }
        
        # Extract Java exceptions
        java_exceptions = re.findall(
            r'([a-zA-Z][a-zA-Z0-9.]*Exception(?:Error)?(?:\s*:\s*[^\n]+)?)', 
            text, 
            re.IGNORECASE
        )
        patterns['exceptions'].extend(java_exceptions)
        
        # Extract stack traces
        stack_traces = self.patterns['stack_trace'].findall(text)
        patterns['stack_traces'].extend(stack_traces[:5])  # Limit to first 5 lines
        
        # Extract HTTP error codes
        http_codes = re.findall(r'\b[45]\d{2}\b', text)
        patterns['error_codes'].extend(http_codes)
        
        # Extract common error patterns
        error_patterns = [
            r'error:?\s*([^\n]{10,100})',
            r'failed:?\s*([^\n]{10,100})',
            r'exception:?\s*([^\n]{10,100})',
            r'unable to\s*([^\n]{10,100})',
            r'cannot\s*([^\n]{10,100})',
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            patterns['error_messages'].extend(matches[:3])  # Limit to first 3 per pattern
        
        return patterns
    
    def standardize_text(self, text: str) -> str:
        """Standardize text format for consistent processing.
        
        Args:
            text: Input text
            
        Returns:
            Standardized text
        """
        if not text:
            return ''
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to ASCII, ignoring errors
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Standardize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Standardize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms and identifiers.
        
        Args:
            text: Input text
            
        Returns:
            List of technical terms
        """
        terms = []
        
        # Extract class names (CamelCase)
        class_names = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)+\b', text)
        terms.extend(class_names)
        
        # Extract method calls
        method_calls = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(', text)
        terms.extend([m.replace('(', '').strip() for m in method_calls])
        
        # Extract file paths
        file_paths = re.findall(r'[a-zA-Z]:[^\s]+|/[^\s]+\.[a-zA-Z0-9]+', text)
        terms.extend(file_paths)
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', text)
        terms.extend(urls)
        
        # Extract configuration keys
        config_keys = re.findall(r'\b[a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+\b', text)
        terms.extend(config_keys)
        
        return list(set(terms))  # Remove duplicates
    
    def create_context_summary(self, issue_data: Dict[str, Any]) -> str:
        """Create a concise context summary for an issue.
        
        Args:
            issue_data: Dictionary containing issue data
            
        Returns:
            Context summary string
        """
        summary_parts = []
        
        # Project and component context
        if issue_data.get('project_name'):
            summary_parts.append(f"Project: {issue_data['project_name']}")
        
        if issue_data.get('components'):
            components = issue_data['components']
            if isinstance(components, list):
                comp_str = ', '.join(components[:3])  # First 3 components
            else:
                comp_str = str(components)
            summary_parts.append(f"Components: {comp_str}")
        
        # Issue type and priority
        if issue_data.get('issue_type'):
            summary_parts.append(f"Type: {issue_data['issue_type']}")
        
        if issue_data.get('priority'):
            summary_parts.append(f"Priority: {issue_data['priority']}")
        
        # Status if resolved
        if issue_data.get('status') in ['Resolved', 'Closed', 'Done']:
            summary_parts.append(f"Status: {issue_data['status']}")
        
        return ' | '.join(summary_parts)
    
    def preprocess_for_similarity(self, text: str) -> str:
        """Preprocess text for similarity comparison.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text optimized for similarity matching
        """
        # Clean the text
        text = self.clean_jira_text(text)
        text = self.standardize_text(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common stop words that don't help with similarity
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves'
        }
        
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess_for_training(self, text: str, max_length: int = 2048) -> str:
        """Preprocess text for model training.
        
        Args:
            text: Input text
            max_length: Maximum length to truncate to
            
        Returns:
            Text ready for training
        """
        # Clean and standardize
        text = self.clean_jira_text(text)
        text = self.standardize_text(text)
        
        # Truncate if too long
        if len(text) > max_length:
            # Try to truncate at sentence boundary
            sentences = text.split('. ')
            truncated = ''
            for sentence in sentences:
                if len(truncated) + len(sentence) + 2 <= max_length:
                    truncated += sentence + '. '
                else:
                    break
            
            if truncated:
                text = truncated.rstrip()
            else:
                # Hard truncate if no sentence boundaries
                text = text[:max_length].rsplit(' ', 1)[0]
        
        return text