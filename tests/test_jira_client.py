"""Tests for Jira API client functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from jira.exceptions import JIRAError

from src.api.jira_client import JiraClient


class TestJiraClient:
    """Test suite for JiraClient class."""
    
    @patch('src.api.jira_client.config')
    def test_init_with_config(self, mock_config):
        """Test initialization using configuration."""
        mock_config.get.side_effect = lambda key: {
            'jira.server_url': 'https://test.atlassian.net',
            'jira.username': 'test_user',
            'jira.api_token': 'test_token'
        }.get(key)
        
        with patch('src.api.jira_client.JIRA') as mock_jira:
            client = JiraClient()
            
            assert client.server_url == 'https://test.atlassian.net'
            assert client.username == 'test_user'
            assert client.api_token == 'test_token'
            mock_jira.assert_called_once()
    
    def test_init_with_parameters(self):
        """Test initialization with explicit parameters."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            client = JiraClient(
                server_url='https://custom.atlassian.net',
                username='custom_user',
                api_token='custom_token'
            )
            
            assert client.server_url == 'https://custom.atlassian.net'
            assert client.username == 'custom_user'
            assert client.api_token == 'custom_token'
            mock_jira.assert_called_once_with(
                server='https://custom.atlassian.net',
                basic_auth=('custom_user', 'custom_token')
            )
    
    @patch('src.api.jira_client.config')
    def test_init_missing_credentials(self, mock_config):
        """Test initialization with missing credentials."""
        mock_config.get.return_value = None
        
        with pytest.raises(ValueError, match="Jira credentials not properly configured"):
            JiraClient()
    
    @patch('src.api.jira_client.JIRA')
    def test_connection_failure(self, mock_jira):
        """Test connection failure handling."""
        mock_jira.side_effect = JIRAError("Connection failed")
        
        with pytest.raises(JIRAError):
            JiraClient(
                server_url='https://test.atlassian.net',
                username='test_user',
                api_token='test_token'
            )
    
    def test_test_connection_success(self):
        """Test successful connection test."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            mock_jira_instance = Mock()
            mock_jira.return_value = mock_jira_instance
            mock_jira_instance.myself.return_value = {'name': 'test_user'}
            
            client = JiraClient(
                server_url='https://test.atlassian.net',
                username='test_user',
                api_token='test_token'
            )
            
            result = client.test_connection()
            assert result is True
    
    def test_test_connection_failure(self):
        """Test connection test failure."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            mock_jira_instance = Mock()
            mock_jira.return_value = mock_jira_instance
            mock_jira_instance.myself.side_effect = JIRAError("Authentication failed")
            
            client = JiraClient(
                server_url='https://test.atlassian.net',
                username='test_user',
                api_token='test_token'
            )
            
            result = client.test_connection()
            assert result is False
    
    def test_get_projects(self):
        """Test getting projects list."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            mock_jira_instance = Mock()
            mock_jira.return_value = mock_jira_instance
            
            mock_project = Mock()
            mock_project.key = 'TEST'
            mock_project.name = 'Test Project'
            mock_jira_instance.projects.return_value = [mock_project]
            
            client = JiraClient(
                server_url='https://test.atlassian.net',
                username='test_user',
                api_token='test_token'
            )
            
            projects = client.get_projects()
            assert len(projects) == 1
            assert projects[0]['key'] == 'TEST'
            assert projects[0]['name'] == 'Test Project'
    
    def test_search_issues_basic(self):
        """Test basic issue search."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            mock_jira_instance = Mock()
            mock_jira.return_value = mock_jira_instance
            
            mock_issue = Mock()
            mock_issue.key = 'TEST-123'
            mock_issue.fields.summary = 'Test Issue'
            mock_issue.fields.description = 'Test Description'
            mock_issue.fields.issuetype.name = 'Bug'
            
            mock_jira_instance.search_issues.return_value = [mock_issue]
            
            client = JiraClient(
                server_url='https://test.atlassian.net',
                username='test_user',
                api_token='test_token'
            )
            
            issues = list(client.search_issues('project = TEST'))
            assert len(issues) == 1
            assert issues[0]['key'] == 'TEST-123'
            assert issues[0]['summary'] == 'Test Issue'
    
    def test_search_issues_with_parameters(self):
        """Test issue search with parameters."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            mock_jira_instance = Mock()
            mock_jira.return_value = mock_jira_instance
            mock_jira_instance.search_issues.return_value = []
            
            client = JiraClient(
                server_url='https://test.atlassian.net',
                username='test_user',
                api_token='test_token'
            )
            
            list(client.search_issues(
                'project = TEST',
                max_results=100,
                start_at=0,
                fields=['summary', 'description']
            ))
            
            mock_jira_instance.search_issues.assert_called_with(
                'project = TEST',
                maxResults=100,
                startAt=0,
                fields=['summary', 'description']
            )
    
    def test_get_issue(self):
        """Test getting single issue."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            mock_jira_instance = Mock()
            mock_jira.return_value = mock_jira_instance
            
            mock_issue = Mock()
            mock_issue.key = 'TEST-123'
            mock_issue.fields.summary = 'Test Issue'
            mock_issue.fields.description = 'Test Description'
            mock_issue.fields.issuetype.name = 'Bug'
            mock_issue.fields.priority.name = 'High'
            mock_issue.fields.status.name = 'Open'
            
            mock_jira_instance.issue.return_value = mock_issue
            
            client = JiraClient(
                server_url='https://test.atlassian.net',
                username='test_user',
                api_token='test_token'
            )
            
            issue = client.get_issue('TEST-123')
            assert issue['key'] == 'TEST-123'
            assert issue['summary'] == 'Test Issue'
            assert issue['issue_type'] == 'Bug'
            assert issue['priority'] == 'High'
            assert issue['status'] == 'Open'
    
    def test_get_issue_not_found(self):
        """Test getting non-existent issue."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            mock_jira_instance = Mock()
            mock_jira.return_value = mock_jira_instance
            mock_jira_instance.issue.side_effect = JIRAError("Issue not found")
            
            client = JiraClient(
                server_url='https://test.atlassian.net',
                username='test_user',
                api_token='test_token'
            )
            
            issue = client.get_issue('NONEXISTENT-123')
            assert issue is None
    
    def test_extract_issue_data(self):
        """Test issue data extraction."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            mock_jira_instance = Mock()
            mock_jira.return_value = mock_jira_instance
            
            client = JiraClient(
                server_url='https://test.atlassian.net',
                username='test_user',
                api_token='test_token'
            )
            
            mock_issue = Mock()
            mock_issue.key = 'TEST-123'
            mock_issue.fields.summary = 'Test Issue'
            mock_issue.fields.description = 'Test Description'
            mock_issue.fields.issuetype.name = 'Bug'
            mock_issue.fields.priority.name = 'High'
            mock_issue.fields.status.name = 'Open'
            mock_issue.fields.assignee = None
            mock_issue.fields.resolution = None
            mock_issue.fields.components = []
            mock_issue.fields.labels = ['bug', 'urgent']
            mock_issue.fields.created = '2023-01-01T00:00:00.000+0000'
            mock_issue.fields.updated = '2023-01-02T00:00:00.000+0000'
            
            # Mock comments
            mock_comment = Mock()
            mock_comment.body = 'Test comment'
            mock_comment.author.displayName = 'Test User'
            mock_comment.created = '2023-01-01T12:00:00.000+0000'
            mock_issue.fields.comment.comments = [mock_comment]
            
            extracted_data = client._extract_issue_data(mock_issue)
            
            assert extracted_data['key'] == 'TEST-123'
            assert extracted_data['summary'] == 'Test Issue'
            assert extracted_data['description'] == 'Test Description'
            assert extracted_data['issue_type'] == 'Bug'
            assert extracted_data['priority'] == 'High'
            assert extracted_data['status'] == 'Open'
            assert extracted_data['labels'] == ['bug', 'urgent']
            assert len(extracted_data['comments']) == 1
            assert extracted_data['comments'][0]['body'] == 'Test comment'
    
    def test_rate_limiting(self):
        """Test rate limiting behavior."""
        with patch('src.api.jira_client.JIRA') as mock_jira:
            with patch('time.sleep') as mock_sleep:
                mock_jira_instance = Mock()
                mock_jira.return_value = mock_jira_instance
                
                # First call succeeds, second call fails with rate limit, third succeeds
                mock_jira_instance.search_issues.side_effect = [
                    JIRAError("Rate limit exceeded", status_code=429),
                    []  # Success after retry
                ]
                
                client = JiraClient(
                    server_url='https://test.atlassian.net',
                    username='test_user',
                    api_token='test_token'
                )
                
                # This should handle rate limiting internally
                list(client.search_issues('project = TEST'))
                
                # Should have slept due to rate limiting
                mock_sleep.assert_called()