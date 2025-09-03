"""Tests for configuration management utilities."""

import pytest
import tempfile
import os
from pathlib import Path
import yaml

from src.utils.config import Config


class TestConfig:
    """Test suite for Config class."""
    
    def test_config_initialization_with_valid_file(self):
        """Test config initialization with a valid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'jira': {
                    'server_url': 'https://test.atlassian.net',
                    'username': 'test_user',
                    'api_token': 'test_token'
                },
                'model': {
                    'base_model': 'microsoft/DialoGPT-medium'
                }
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            assert config.get('jira.server_url') == 'https://test.atlassian.net'
            assert config.get('jira.username') == 'test_user'
            assert config.get('model.base_model') == 'microsoft/DialoGPT-medium'
        finally:
            os.unlink(temp_path)
    
    def test_config_missing_file(self):
        """Test config initialization with missing file."""
        with pytest.raises(FileNotFoundError):
            Config('nonexistent_config.yaml')
    
    def test_config_get_with_default(self):
        """Test getting config value with default."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {'test_key': 'test_value'}
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            assert config.get('test_key') == 'test_value'
            assert config.get('nonexistent_key', 'default') == 'default'
            assert config.get('nonexistent_key') is None
        finally:
            os.unlink(temp_path)
    
    def test_config_dot_notation(self):
        """Test getting nested config values with dot notation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'level1': {
                    'level2': {
                        'level3': 'deep_value'
                    }
                }
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            assert config.get('level1.level2.level3') == 'deep_value'
            assert config.get('level1.level2') == {'level3': 'deep_value'}
            assert config.get('level1.nonexistent') is None
        finally:
            os.unlink(temp_path)
    
    def test_config_set_value(self):
        """Test setting config values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {'existing': 'value'}
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            config.set('new_key', 'new_value')
            config.set('nested.key', 'nested_value')
            
            assert config.get('new_key') == 'new_value'
            assert config.get('nested.key') == 'nested_value'
        finally:
            os.unlink(temp_path)
    
    def test_config_validation(self):
        """Test config validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Missing required fields
            config_data = {'jira': {'server_url': 'test'}}
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            with pytest.raises(ValueError, match="Missing required configuration fields"):
                config.validate()
        finally:
            os.unlink(temp_path)
    
    def test_config_properties(self):
        """Test config property accessors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'jira': {'server_url': 'test_jira'},
                'model': {'base_model': 'test_model'},
                'training': {'epochs': 3},
                'dataset': {'min_length': 50},
                'paths': {'output': '/tmp'}
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            assert config.jira == {'server_url': 'test_jira'}
            assert config.model == {'base_model': 'test_model'}
            assert config.training == {'epochs': 3}
            assert config.dataset == {'min_length': 50}
            assert config.paths == {'output': '/tmp'}
        finally:
            os.unlink(temp_path)
    
    def test_env_var_substitution(self):
        """Test environment variable substitution."""
        # Set test environment variable
        os.environ['TEST_VAR'] = 'test_value'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'test_key': '${TEST_VAR}',
                'normal_key': 'normal_value'
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            assert config.get('test_key') == 'test_value'
            assert config.get('normal_key') == 'normal_value'
        finally:
            os.unlink(temp_path)
            del os.environ['TEST_VAR']