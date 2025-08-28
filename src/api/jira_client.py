"""Jira API client for connecting to and querying Jira instances."""

from typing import List, Dict, Any, Optional, Iterator
from jira import JIRA, Issue
from jira.exceptions import JIRAError
import time
from datetime import datetime, timedelta

from ..utils import get_logger, config


class JiraClient:
    """Client for interacting with Jira API."""
    
    def __init__(self, server_url: str = None, username: str = None, api_token: str = None):
        """Initialize Jira client.
        
        Args:
            server_url: Jira server URL. If None, uses config.
            username: Jira username. If None, uses config.
            api_token: Jira API token. If None, uses config.
        """
        self.logger = get_logger(__name__)
        
        self.server_url = server_url or config.get('jira.server_url')
        self.username = username or config.get('jira.username') 
        self.api_token = api_token or config.get('jira.api_token')
        
        if not all([self.server_url, self.username, self.api_token]):
            raise ValueError("Jira credentials not properly configured")
        
        self._client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Jira."""
        try:
            self._client = JIRA(
                server=self.server_url,
                basic_auth=(self.username, self.api_token)
            )
            self.logger.info(f"Connected to Jira server: {self.server_url}")
            
        except JIRAError as e:
            self.logger.error(f"Failed to connect to Jira: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test the Jira connection.
        
        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            user = self._client.current_user()
            self.logger.info(f"Connection test successful. Current user: {user}")
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def search_issues(
        self,
        jql: str,
        max_results: int = None,
        fields: List[str] = None,
        expand: List[str] = None
    ) -> List[Issue]:
        """Search for issues using JQL.
        
        Args:
            jql: JQL query string
            max_results: Maximum number of results to return
            fields: List of fields to include in response
            expand: List of fields to expand
            
        Returns:
            List of Jira issues
        """
        max_results = max_results or config.get('jira.max_results', 100)
        fields = fields or ['*all']
        expand = expand or ['comments', 'attachments']
        
        try:
            self.logger.info(f"Searching issues with JQL: {jql}")
            issues = self._client.search_issues(
                jql_str=jql,
                maxResults=max_results,
                fields=fields,
                expand=','.join(expand)
            )
            self.logger.info(f"Found {len(issues)} issues")
            return issues
            
        except JIRAError as e:
            self.logger.error(f"Error searching issues: {e}")
            raise
    
    def search_issues_paginated(
        self,
        jql: str,
        max_results: int = None,
        fields: List[str] = None,
        expand: List[str] = None,
        batch_size: int = 100
    ) -> Iterator[Issue]:
        """Search for issues with pagination to handle large result sets.
        
        Args:
            jql: JQL query string
            max_results: Maximum total results to return
            fields: List of fields to include
            expand: List of fields to expand
            batch_size: Number of issues to fetch per request
            
        Yields:
            Individual Jira issues
        """
        fields = fields or ['*all']
        expand = expand or ['comments', 'attachments']
        
        start_at = 0
        total_returned = 0
        max_results = max_results or float('inf')
        
        while total_returned < max_results:
            current_batch_size = min(batch_size, max_results - total_returned)
            
            try:
                self.logger.debug(f"Fetching batch: start_at={start_at}, max_results={current_batch_size}")
                
                issues = self._client.search_issues(
                    jql_str=jql,
                    startAt=start_at,
                    maxResults=current_batch_size,
                    fields=fields,
                    expand=','.join(expand)
                )
                
                if not issues:
                    self.logger.info("No more issues found")
                    break
                
                for issue in issues:
                    yield issue
                    total_returned += 1
                
                start_at += len(issues)
                
                # Rate limiting
                time.sleep(0.1)
                
            except JIRAError as e:
                self.logger.error(f"Error in paginated search: {e}")
                raise
    
    def get_issue(self, issue_key: str, fields: List[str] = None, expand: List[str] = None) -> Issue:
        """Get a specific issue by key.
        
        Args:
            issue_key: Jira issue key (e.g., 'PROJ-123')
            fields: List of fields to include
            expand: List of fields to expand
            
        Returns:
            Jira issue object
        """
        fields = fields or ['*all']
        expand = expand or ['comments', 'attachments']
        
        try:
            issue = self._client.issue(
                id=issue_key,
                fields=','.join(fields),
                expand=','.join(expand)
            )
            return issue
            
        except JIRAError as e:
            self.logger.error(f"Error getting issue {issue_key}: {e}")
            raise
    
    def get_project_issues(
        self,
        project_key: str,
        issue_types: List[str] = None,
        statuses: List[str] = None,
        created_after: datetime = None,
        max_results: int = None
    ) -> Iterator[Issue]:
        """Get issues for a specific project with filters.
        
        Args:
            project_key: Jira project key
            issue_types: List of issue types to include (e.g., ['Bug', 'Task'])
            statuses: List of statuses to include (e.g., ['Resolved', 'Closed'])
            created_after: Only include issues created after this date
            max_results: Maximum number of results
            
        Yields:
            Jira issues matching the criteria
        """
        jql_parts = [f"project = {project_key}"]
        
        if issue_types:
            types_str = ','.join([f'"{t}"' for t in issue_types])
            jql_parts.append(f"issuetype in ({types_str})")
        
        if statuses:
            statuses_str = ','.join([f'"{s}"' for s in statuses])
            jql_parts.append(f"status in ({statuses_str})")
        
        if created_after:
            date_str = created_after.strftime('%Y-%m-%d')
            jql_parts.append(f"created >= {date_str}")
        
        jql = " AND ".join(jql_parts)
        
        yield from self.search_issues_paginated(jql, max_results=max_results)
    
    def get_recent_issues(self, days: int = 30, max_results: int = None) -> Iterator[Issue]:
        """Get recently created or updated issues.
        
        Args:
            days: Number of days to look back
            max_results: Maximum number of results
            
        Yields:
            Recent Jira issues
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        date_str = cutoff_date.strftime('%Y-%m-%d')
        
        jql = f"(created >= {date_str} OR updated >= {date_str})"
        
        yield from self.search_issues_paginated(jql, max_results=max_results)
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """Get list of available projects.
        
        Returns:
            List of project information dictionaries
        """
        try:
            projects = self._client.projects()
            return [
                {
                    'key': p.key,
                    'name': p.name,
                    'id': p.id,
                    'description': getattr(p, 'description', None)
                }
                for p in projects
            ]
        except JIRAError as e:
            self.logger.error(f"Error getting projects: {e}")
            raise
    
    def get_issue_types(self, project_key: str = None) -> List[Dict[str, Any]]:
        """Get available issue types.
        
        Args:
            project_key: Optional project key to filter by
            
        Returns:
            List of issue type information
        """
        try:
            if project_key:
                project = self._client.project(project_key)
                issue_types = project.issueTypes
            else:
                issue_types = self._client.issue_types()
            
            return [
                {
                    'id': it.id,
                    'name': it.name,
                    'description': getattr(it, 'description', None)
                }
                for it in issue_types
            ]
        except JIRAError as e:
            self.logger.error(f"Error getting issue types: {e}")
            raise