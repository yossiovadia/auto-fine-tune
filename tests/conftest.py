"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        'jira': {
            'server_url': 'https://test.atlassian.net',
            'username': 'test_user',
            'api_token': 'test_token'
        },
        'model': {
            'base_model': 'microsoft/DialoGPT-small',
            'max_length': 512
        },
        'training': {
            'learning_rate': 5e-5,
            'num_epochs': 3,
            'batch_size': 4,
            'warmup_steps': 100
        },
        'lora': {
            'r': 16,
            'alpha': 32,
            'dropout': 0.1
        },
        'dataset': {
            'min_description_length': 50,
            'max_description_length': 2000
        },
        'paths': {
            'data_dir': 'data/',
            'output_dir': 'models/',
            'logs_dir': 'logs/'
        }
    }


@pytest.fixture
def config_file(temp_dir, sample_config_data):
    """Create a temporary config file."""
    config_path = Path(temp_dir) / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(sample_config_data, f)
    return str(config_path)


@pytest.fixture
def mock_jira_issue():
    """Mock Jira issue object."""
    issue = Mock()
    issue.key = 'TEST-123'
    issue.fields.summary = 'Test Issue Summary'
    issue.fields.description = 'This is a test issue description with sufficient length for validation.'
    issue.fields.issuetype.name = 'Bug'
    issue.fields.priority.name = 'High'
    issue.fields.status.name = 'Open'
    issue.fields.assignee = None
    issue.fields.resolution = None
    issue.fields.components = []
    issue.fields.labels = ['bug', 'urgent']
    issue.fields.created = '2023-01-01T00:00:00.000+0000'
    issue.fields.updated = '2023-01-02T00:00:00.000+0000'
    
    # Mock comments
    comment = Mock()
    comment.body = 'This is a test comment'
    comment.author.displayName = 'Test User'
    comment.created = '2023-01-01T12:00:00.000+0000'
    issue.fields.comment.comments = [comment]
    
    return issue


@pytest.fixture
def sample_issue_data():
    """Sample issue data for testing."""
    return {
        'key': 'TEST-123',
        'summary': 'Test Issue Summary',
        'description': 'This is a test issue description with sufficient length for validation and testing purposes.',
        'issue_type': 'Bug',
        'priority': 'High',
        'status': 'Open',
        'assignee': None,
        'resolution': None,
        'components': [],
        'labels': ['bug', 'urgent'],
        'created': '2023-01-01T00:00:00.000+0000',
        'updated': '2023-01-02T00:00:00.000+0000',
        'comments': [
            {
                'body': 'This is a test comment',
                'author': 'Test User',
                'created': '2023-01-01T12:00:00.000+0000'
            }
        ]
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for testing."""
    return [
        {
            'input': 'User reports login issue',
            'output': 'Please check your credentials and try again. If the issue persists, reset your password.'
        },
        {
            'input': 'Application crashes on startup',
            'output': 'This appears to be a memory issue. Please update your drivers and restart the application.'
        },
        {
            'input': 'Feature request for dark mode',
            'output': 'Thank you for the suggestion. We will consider adding dark mode in the next release.'
        }
    ]


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode_plus.return_value = {
        'input_ids': [1, 2, 3, 4, 5],
        'attention_mask': [1, 1, 1, 1, 1]
    }
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '</s>'
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.config.vocab_size = 50257
    model.device = 'cpu'
    return model


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    return logger


@pytest.fixture(autouse=True)
def mock_get_logger():
    """Auto-mock the get_logger function for all tests."""
    with patch('src.utils.logging.get_logger') as mock:
        mock_logger = Mock()
        mock.return_value = mock_logger
        yield mock_logger


@pytest.fixture
def mock_config():
    """Mock config for testing."""
    config = Mock()
    config.get.return_value = None
    config.jira = {}
    config.model = {}
    config.training = {}
    config.lora = {}
    config.dataset = {}
    config.paths = {}
    return config


# Environment setup for tests
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Disable CUDA for tests to avoid GPU dependency
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set test environment flag
    os.environ['TESTING'] = 'true'
    
    yield
    
    # Clean up
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


# Custom markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )