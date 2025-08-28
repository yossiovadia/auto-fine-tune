"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Configuration manager for the adaptive fine-tuning system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration from YAML file and environment variables."""
        load_dotenv()
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._substitute_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _substitute_env_vars(self):
        """Substitute environment variables in configuration values."""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            else:
                return obj
        
        self._config = substitute_recursive(self._config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'jira.server_url')."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config_dict = self._config
        
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        
        config_dict[keys[-1]] = value
    
    @property
    def jira(self) -> Dict[str, Any]:
        """Get Jira configuration."""
        return self._config.get('jira', {})
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get('model', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config.get('training', {})
    
    @property
    def dataset(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self._config.get('dataset', {})
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self._config.get('paths', {})
    
    def validate(self):
        """Validate required configuration values."""
        required_fields = [
            'jira.server_url',
            'jira.username', 
            'jira.api_token',
            'model.base_model'
        ]
        
        missing_fields = []
        for field in required_fields:
            if self.get(field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
    
    def save(self, path: str = None):
        """Save current configuration to file."""
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)


# Global configuration instance
config = Config()