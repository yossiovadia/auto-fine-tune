"""Tests for adaptive trainer functionality."""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
from datasets import Dataset

from src.models.trainer import AdaptiveTrainer


class TestAdaptiveTrainer:
    """Test suite for AdaptiveTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.models.trainer.config')
    def test_init_with_base_model(self, mock_config):
        """Test initialization with base model."""
        mock_config.get.side_effect = lambda key, default=None: {
            'model.base_model': 'microsoft/DialoGPT-small',
            'lora': {'r': 16, 'alpha': 32},
            'training': {'learning_rate': 5e-5}
        }.get(key, default)
        mock_config.training = {'learning_rate': 5e-5}
        mock_config.model = {'base_model': 'microsoft/DialoGPT-small'}
        
        trainer = AdaptiveTrainer(base_model_name='microsoft/DialoGPT-small')
        assert trainer.base_model_name == 'microsoft/DialoGPT-small'
        assert trainer.previous_model_path is None
    
    @patch('src.models.trainer.config')
    def test_init_with_previous_model(self, mock_config):
        """Test initialization with previous model path."""
        mock_config.get.side_effect = lambda key, default=None: {
            'model.base_model': 'microsoft/DialoGPT-small',
            'lora': {'r': 16, 'alpha': 32},
            'training': {'learning_rate': 5e-5}
        }.get(key, default)
        mock_config.training = {'learning_rate': 5e-5}
        mock_config.model = {'base_model': 'microsoft/DialoGPT-small'}
        
        trainer = AdaptiveTrainer(previous_model_path='/path/to/model')
        assert trainer.previous_model_path == '/path/to/model'
    
    @patch('src.models.trainer.AutoTokenizer')
    @patch('src.models.trainer.config')
    def test_load_tokenizer(self, mock_config, mock_tokenizer_class):
        """Test tokenizer loading."""
        mock_config.get.side_effect = lambda key, default=None: {
            'model.base_model': 'microsoft/DialoGPT-small'
        }.get(key, default)
        mock_config.training = {}
        mock_config.model = {}
        
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        trainer = AdaptiveTrainer()
        tokenizer = trainer._load_tokenizer()
        
        assert tokenizer == mock_tokenizer
        mock_tokenizer_class.from_pretrained.assert_called_once()
    
    @patch('src.models.trainer.AutoModelForCausalLM')
    @patch('src.models.trainer.config')
    def test_load_base_model(self, mock_config, mock_model_class):
        """Test base model loading."""
        mock_config.get.side_effect = lambda key, default=None: {
            'model.base_model': 'microsoft/DialoGPT-small'
        }.get(key, default)
        mock_config.training = {}
        mock_config.model = {}
        mock_config.lora_config = {}
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        trainer = AdaptiveTrainer()
        model = trainer._load_base_model()
        
        assert model == mock_model
        mock_model_class.from_pretrained.assert_called_once()
    
    @patch('src.models.trainer.get_peft_model')
    @patch('src.models.trainer.LoraConfig')
    @patch('src.models.trainer.prepare_model_for_kbit_training')
    @patch('src.models.trainer.config')
    def test_setup_lora_model(self, mock_config, mock_prepare, mock_lora_config_class, mock_get_peft):
        """Test LoRA model setup."""
        mock_config.get.side_effect = lambda key, default=None: {
            'lora.r': 16,
            'lora.alpha': 32,
            'lora.dropout': 0.1
        }.get(key, default)
        mock_config.training = {}
        mock_config.model = {}
        
        mock_base_model = Mock()
        mock_lora_config = Mock()
        mock_lora_config_class.return_value = mock_lora_config
        mock_peft_model = Mock()
        mock_get_peft.return_value = mock_peft_model
        mock_prepare.return_value = mock_base_model
        
        trainer = AdaptiveTrainer()
        model = trainer._setup_lora_model(mock_base_model)
        
        assert model == mock_peft_model
        mock_prepare.assert_called_once_with(mock_base_model)
        mock_get_peft.assert_called_once_with(mock_base_model, mock_lora_config)
    
    def test_prepare_dataset_format(self):
        """Test dataset formatting."""
        trainer = AdaptiveTrainer()
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode_plus.return_value = {
            'input_ids': [1, 2, 3, 4, 5],
            'attention_mask': [1, 1, 1, 1, 1]
        }
        mock_tokenizer.pad_token_id = 0
        trainer.tokenizer = mock_tokenizer
        
        # Sample data
        train_data = [
            {'input': 'Hello', 'output': 'Hi there!'},
            {'input': 'How are you?', 'output': 'I am fine.'}
        ]
        
        dataset = trainer._prepare_dataset(train_data)
        
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2
    
    @patch('src.models.trainer.TrainingArguments')
    @patch('src.models.trainer.config')
    def test_create_training_arguments(self, mock_config, mock_training_args_class):
        """Test training arguments creation."""
        mock_config.get.side_effect = lambda key, default=None: {
            'training.learning_rate': 5e-5,
            'training.num_epochs': 3,
            'training.batch_size': 4,
            'training.warmup_steps': 100
        }.get(key, default)
        mock_config.training = {
            'learning_rate': 5e-5,
            'num_epochs': 3,
            'batch_size': 4,
            'warmup_steps': 100
        }
        mock_config.model = {}
        
        mock_training_args = Mock()
        mock_training_args_class.return_value = mock_training_args
        
        trainer = AdaptiveTrainer()
        output_dir = str(Path(self.temp_dir) / 'output')
        args = trainer._create_training_arguments(output_dir)
        
        assert args == mock_training_args
        mock_training_args_class.assert_called_once()
    
    @patch('src.models.trainer.Trainer')
    @patch('src.models.trainer.DataCollatorForLanguageModeling')
    def test_create_trainer(self, mock_data_collator_class, mock_trainer_class):
        """Test trainer creation."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_dataset = Mock()
        mock_args = Mock()
        
        mock_data_collator = Mock()
        mock_data_collator_class.return_value = mock_data_collator
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        trainer = AdaptiveTrainer()
        trainer.tokenizer = mock_tokenizer
        
        result = trainer._create_trainer(mock_model, mock_tokenizer, mock_dataset, mock_args)
        
        assert result == mock_trainer
        mock_trainer_class.assert_called_once()
    
    @patch('src.models.trainer.wandb')
    @patch('src.models.trainer.config')
    def test_setup_monitoring(self, mock_config, mock_wandb):
        """Test monitoring setup."""
        mock_config.get.side_effect = lambda key, default=None: {
            'wandb.project': 'test-project',
            'wandb.enabled': True
        }.get(key, default)
        mock_config.training = {}
        mock_config.model = {}
        
        trainer = AdaptiveTrainer()
        trainer._setup_monitoring('test-run')
        
        mock_wandb.init.assert_called_once()
    
    def test_validate_training_data(self):
        """Test training data validation."""
        trainer = AdaptiveTrainer()
        
        # Valid data
        valid_data = [
            {'input': 'Hello', 'output': 'Hi there!'},
            {'input': 'How are you?', 'output': 'I am fine.'}
        ]
        
        is_valid, issues = trainer._validate_training_data(valid_data)
        assert is_valid is True
        assert len(issues) == 0
        
        # Invalid data - missing keys
        invalid_data = [
            {'input': 'Hello'},  # Missing output
            {'output': 'Hi there!'}  # Missing input
        ]
        
        is_valid, issues = trainer._validate_training_data(invalid_data)
        assert is_valid is False
        assert len(issues) > 0
    
    @patch('src.models.trainer.torch.cuda.is_available')
    def test_device_selection(self, mock_cuda_available):
        """Test device selection logic."""
        # CUDA available
        mock_cuda_available.return_value = True
        trainer = AdaptiveTrainer()
        device = trainer._get_device()
        assert 'cuda' in device
        
        # CUDA not available
        mock_cuda_available.return_value = False
        trainer = AdaptiveTrainer()
        device = trainer._get_device()
        assert device == 'cpu'
    
    def test_calculate_training_steps(self):
        """Test training steps calculation."""
        trainer = AdaptiveTrainer()
        
        dataset_size = 1000
        batch_size = 8
        num_epochs = 3
        
        steps = trainer._calculate_training_steps(dataset_size, batch_size, num_epochs)
        expected_steps = (dataset_size // batch_size) * num_epochs
        assert steps == expected_steps
    
    @patch('src.models.trainer.config')
    def test_save_training_metadata(self, mock_config):
        """Test saving training metadata."""
        mock_config.training = {'learning_rate': 5e-5}
        mock_config.model = {'base_model': 'test-model'}
        
        trainer = AdaptiveTrainer()
        output_dir = Path(self.temp_dir) / 'output'
        output_dir.mkdir(exist_ok=True)
        
        metadata = {
            'model_name': 'test-model',
            'training_steps': 100,
            'learning_rate': 5e-5
        }
        
        trainer._save_training_metadata(str(output_dir), metadata)
        
        metadata_file = output_dir / 'training_metadata.json'
        assert metadata_file.exists()
    
    def test_estimate_training_time(self):
        """Test training time estimation."""
        trainer = AdaptiveTrainer()
        
        # Mock parameters for estimation
        dataset_size = 1000
        batch_size = 8
        num_epochs = 3
        
        estimated_time = trainer._estimate_training_time(dataset_size, batch_size, num_epochs)
        
        # Should return a reasonable estimate (positive number)
        assert estimated_time > 0
        assert isinstance(estimated_time, (int, float))
    
    def test_cleanup_old_checkpoints(self):
        """Test cleanup of old checkpoints."""
        trainer = AdaptiveTrainer()
        
        # Create mock checkpoint directories
        output_dir = Path(self.temp_dir) / 'output'
        output_dir.mkdir(exist_ok=True)
        
        for i in range(5):
            checkpoint_dir = output_dir / f'checkpoint-{i * 100}'
            checkpoint_dir.mkdir()
            
        # Keep only 3 most recent checkpoints
        trainer._cleanup_old_checkpoints(str(output_dir), keep_last=3)
        
        # Should have only 3 checkpoint directories left
        remaining_checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
        assert len(remaining_checkpoints) == 3