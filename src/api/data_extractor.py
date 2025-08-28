"""Jira data extraction and preprocessing for ML training."""

import re
import html
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import json

from .jira_client import JiraClient
from ..utils import get_logger, config


class JiraDataExtractor:
    """Extract and preprocess Jira data for machine learning."""
    
    def __init__(self, jira_client: JiraClient = None):
        """Initialize data extractor.
        
        Args:
            jira_client: JiraClient instance. If None, creates a new one.
        """
        self.logger = get_logger(__name__)
        self.client = jira_client or JiraClient()
        
    def extract_issue_data(self, issue) -> Dict[str, Any]:
        """Extract relevant data from a Jira issue.
        
        Args:
            issue: Jira issue object
            
        Returns:
            Dictionary containing extracted issue data
        """
        fields = issue.fields
        
        # Basic issue information
        data = {
            'key': issue.key,
            'id': issue.id,
            'summary': fields.summary or '',
            'description': self._clean_text(fields.description or ''),
            'issue_type': fields.issuetype.name if fields.issuetype else '',
            'status': fields.status.name if fields.status else '',
            'priority': fields.priority.name if fields.priority else '',
            'project_key': fields.project.key if fields.project else '',
            'project_name': fields.project.name if fields.project else '',
            'creator': fields.creator.displayName if fields.creator else '',
            'assignee': fields.assignee.displayName if fields.assignee else '',
            'reporter': fields.reporter.displayName if fields.reporter else '',
            'created': self._parse_datetime(fields.created),
            'updated': self._parse_datetime(fields.updated),
            'resolved': self._parse_datetime(fields.resolved) if fields.resolved else None,
            'resolution': fields.resolution.name if fields.resolution else '',
        }
        
        # Components
        if hasattr(fields, 'components') and fields.components:
            data['components'] = [comp.name for comp in fields.components]
        else:
            data['components'] = []
        
        # Labels
        if hasattr(fields, 'labels') and fields.labels:
            data['labels'] = list(fields.labels)
        else:
            data['labels'] = []
        
        # Comments
        comments = []
        if hasattr(issue, 'fields') and hasattr(issue.fields, 'comment') and issue.fields.comment:
            for comment in issue.fields.comment.comments:
                comments.append({
                    'author': comment.author.displayName if comment.author else '',
                    'body': self._clean_text(comment.body or ''),
                    'created': self._parse_datetime(comment.created)
                })
        data['comments'] = comments
        
        # Status transitions (if available)
        data['status_history'] = self._extract_status_history(issue)
        
        # Custom fields (example - adapt based on your Jira setup)
        data['custom_fields'] = self._extract_custom_fields(fields)
        
        return data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ''
        
        # HTML decode
        text = html.unescape(text)
        
        # Remove Jira markup patterns
        # Remove code blocks but preserve content
        text = re.sub(r'\{code(?::[^}]*)?\}(.*?)\{code\}', r'```\1```', text, flags=re.DOTALL)
        
        # Remove other Jira markup
        text = re.sub(r'\{[^}]+\}', '', text)  # Remove {panel}, {quote}, etc.
        text = re.sub(r'\[~[^\]]+\]', '', text)  # Remove user mentions
        text = re.sub(r'\[[^\]]*\|[^\]]*\]', '', text)  # Remove links with text
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _parse_datetime(self, datetime_str: str) -> Optional[datetime]:
        """Parse Jira datetime string.
        
        Args:
            datetime_str: Jira datetime string
            
        Returns:
            Parsed datetime object or None
        """
        if not datetime_str:
            return None
        
        try:
            # Jira uses ISO format with timezone
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except Exception as e:
            self.logger.warning(f"Failed to parse datetime '{datetime_str}': {e}")
            return None
    
    def _extract_status_history(self, issue) -> List[Dict[str, Any]]:
        """Extract status change history.
        
        Args:
            issue: Jira issue object
            
        Returns:
            List of status changes
        """
        # This would require expanding the changelog
        # For now, return empty list
        # TODO: Implement changelog extraction if needed
        return []
    
    def _extract_custom_fields(self, fields) -> Dict[str, Any]:
        """Extract custom field values.
        
        Args:
            fields: Jira issue fields object
            
        Returns:
            Dictionary of custom field values
        """
        custom_fields = {}
        
        # Common custom fields - adapt based on your setup
        field_mappings = {
            'customfield_10000': 'story_points',
            'customfield_10001': 'epic_link',
            'customfield_10002': 'sprint',
            # Add more mappings as needed
        }
        
        for field_id, field_name in field_mappings.items():
            if hasattr(fields, field_id):
                value = getattr(fields, field_id)
                if value is not None:
                    custom_fields[field_name] = str(value)
        
        return custom_fields
    
    def extract_project_data(
        self,
        project_key: str,
        issue_types: List[str] = None,
        max_issues: int = None,
        output_file: str = None
    ) -> pd.DataFrame:
        """Extract data for an entire project.
        
        Args:
            project_key: Jira project key
            issue_types: List of issue types to include
            max_issues: Maximum number of issues to extract
            output_file: Optional file path to save data
            
        Returns:
            DataFrame containing extracted data
        """
        self.logger.info(f"Extracting data for project: {project_key}")
        
        issues = list(self.client.get_project_issues(
            project_key=project_key,
            issue_types=issue_types,
            max_results=max_issues
        ))
        
        if not issues:
            self.logger.warning(f"No issues found for project {project_key}")
            return pd.DataFrame()
        
        self.logger.info(f"Processing {len(issues)} issues")
        
        data = []
        for issue in issues:
            try:
                issue_data = self.extract_issue_data(issue)
                data.append(issue_data)
            except Exception as e:
                self.logger.error(f"Error processing issue {issue.key}: {e}")
                continue
        
        df = pd.DataFrame(data)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix.lower() == '.csv':
                df.to_csv(output_file, index=False)
            elif output_path.suffix.lower() == '.json':
                df.to_json(output_file, orient='records', indent=2)
            else:
                df.to_pickle(output_file)
            
            self.logger.info(f"Data saved to: {output_file}")
        
        return df
    
    def extract_recent_data(
        self,
        days: int = 30,
        max_issues: int = None,
        output_file: str = None
    ) -> pd.DataFrame:
        """Extract recent issues data.
        
        Args:
            days: Number of days to look back
            max_issues: Maximum number of issues
            output_file: Optional output file path
            
        Returns:
            DataFrame containing recent issues data
        """
        self.logger.info(f"Extracting recent issues from last {days} days")
        
        issues = list(self.client.get_recent_issues(
            days=days,
            max_results=max_issues
        ))
        
        if not issues:
            self.logger.warning("No recent issues found")
            return pd.DataFrame()
        
        self.logger.info(f"Processing {len(issues)} recent issues")
        
        data = []
        for issue in issues:
            try:
                issue_data = self.extract_issue_data(issue)
                data.append(issue_data)
            except Exception as e:
                self.logger.error(f"Error processing issue {issue.key}: {e}")
                continue
        
        df = pd.DataFrame(data)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            self.logger.info(f"Recent data saved to: {output_file}")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for extracted data.
        
        Args:
            df: DataFrame containing extracted issue data
            
        Returns:
            Dictionary containing summary statistics
        """
        if df.empty:
            return {'total_issues': 0}
        
        summary = {
            'total_issues': len(df),
            'issue_types': df['issue_type'].value_counts().to_dict(),
            'statuses': df['status'].value_counts().to_dict(),
            'priorities': df['priority'].value_counts().to_dict(),
            'projects': df['project_key'].value_counts().to_dict(),
            'date_range': {
                'earliest': df['created'].min().isoformat() if 'created' in df else None,
                'latest': df['created'].max().isoformat() if 'created' in df else None
            },
            'avg_description_length': df['description'].str.len().mean() if 'description' in df else 0,
            'issues_with_comments': (df['comments'].str.len() > 0).sum() if 'comments' in df else 0
        }
        
        return summary