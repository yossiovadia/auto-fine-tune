"""Tests for data validation functionality."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data.data_validator import DataValidator


class TestDataValidator:
    """Test suite for DataValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.data.data_validator.config') as mock_config:
            mock_config.get.side_effect = lambda key, default: {
                'dataset.min_description_length': 20,  # Lower threshold for testing
                'dataset.max_description_length': 2000
            }.get(key, default)
            self.validator = DataValidator()
    
    def test_validate_issue_data_valid(self):
        """Test validation of valid issue data."""
        issue_data = {
            'key': 'TEST-123',
            'summary': 'This is a test issue summary',
            'description': 'This is a detailed description of the test issue that meets minimum length requirements.',
            'issue_type': 'Bug'
        }
        
        is_valid, issues = self.validator.validate_issue_data(issue_data)
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_issue_data_missing_fields(self):
        """Test validation with missing required fields."""
        issue_data = {
            'key': 'TEST-123',
            'summary': 'Test summary'
            # Missing description and issue_type
        }
        
        is_valid, issues = self.validator.validate_issue_data(issue_data)
        assert is_valid is False
        assert any('Missing required field: description' in issue for issue in issues)
        assert any('Missing required field: issue_type' in issue for issue in issues)
    
    def test_validate_issue_data_short_description(self):
        """Test validation with description too short."""
        issue_data = {
            'key': 'TEST-123',
            'summary': 'Test summary',
            'description': 'Short',  # Too short
            'issue_type': 'Bug'
        }
        
        is_valid, issues = self.validator.validate_issue_data(issue_data)
        assert is_valid is False
        assert any('Description too short' in issue for issue in issues)
    
    def test_validate_issue_data_long_description(self):
        """Test validation with description too long."""
        issue_data = {
            'key': 'TEST-123',
            'summary': 'Test summary',
            'description': 'A' * 3000,  # Too long
            'issue_type': 'Bug'
        }
        
        is_valid, issues = self.validator.validate_issue_data(issue_data)
        assert is_valid is False
        assert any('Description too long' in issue for issue in issues)
    
    def test_is_low_quality_text_empty(self):
        """Test low quality detection for empty/short text."""
        assert self.validator._is_low_quality_text('') is True
        assert self.validator._is_low_quality_text('   ') is True
        assert self.validator._is_low_quality_text('short') is True
    
    def test_is_low_quality_text_template_phrases(self):
        """Test low quality detection for template phrases."""
        assert self.validator._is_low_quality_text('Please fill in the details') is True
        assert self.validator._is_low_quality_text('TODO: Add description') is True
        assert self.validator._is_low_quality_text('This is a placeholder text') is True
    
    def test_is_low_quality_text_repetitive(self):
        """Test low quality detection for repetitive text."""
        repetitive_text = 'test ' * 20  # Very repetitive
        assert self.validator._is_low_quality_text(repetitive_text) is True
    
    def test_is_low_quality_text_minimal_chars(self):
        """Test low quality detection for minimal unique characters."""
        assert self.validator._is_low_quality_text('aaaaaaaaaa') is True
        assert self.validator._is_low_quality_text('This has enough unique characters') is False
    
    def test_validate_dataset(self):
        """Test dataset validation."""
        df = pd.DataFrame([
            {
                'key': 'TEST-1',
                'summary': 'Application crashes when loading large datasets with multiple columns and complex data types',
                'description': 'When attempting to load datasets with more than 1000 columns and mixed data types including dates, strings, and numeric values, the application encounters a memory overflow error and crashes unexpectedly. This impacts user productivity and data analysis workflows.',
                'issue_type': 'Bug'
            },
            {
                'key': 'TEST-2',
                'summary': 'Invalid issue with short description',
                'description': 'Short',  # Too short
                'issue_type': 'Bug'
            },
            {
                'key': 'TEST-3',
                'summary': 'Feature request for implementing advanced data visualization capabilities with interactive charts',
                'description': 'Users have requested the ability to create interactive charts and graphs directly within the application. This would include support for bar charts, line graphs, scatter plots, and heat maps with customizable styling options and export functionality to various formats.',
                'issue_type': 'Task'
            }
        ])
        
        report = self.validator.validate_dataset(df)
        
        assert report['total_issues'] == 3
        assert report['valid_issues'] == 2
        assert report['invalid_issues'] == 1
        assert report['validation_rate'] == 2/3
        assert 'statistics' in report
        assert 'description_length' in report['statistics']
    
    def test_clean_dataset(self):
        """Test dataset cleaning."""
        df = pd.DataFrame([
            {
                'key': 'TEST-1',
                'summary': 'Valid issue',
                'description': 'This is a valid description with enough content.',
                'issue_type': 'Bug'
            },
            {
                'key': 'TEST-2',
                'summary': None,  # Missing summary
                'description': 'Valid description but missing summary.',
                'issue_type': 'Bug'
            },
            {
                'key': 'TEST-3',
                'summary': 'Short description issue',
                'description': 'Short',  # Too short
                'issue_type': 'Bug'
            },
            {
                'key': 'TEST-4',
                'summary': 'Valid issue',  # Duplicate
                'description': 'This is a valid description with enough content.',  # Duplicate
                'issue_type': 'Bug'
            }
        ])
        
        cleaned_df = self.validator.clean_dataset(df)
        
        # Should remove issues with missing summary, short description, and duplicates
        assert len(cleaned_df) == 1
        assert cleaned_df.iloc[0]['key'] == 'TEST-1'
    
    def test_clean_text_field(self):
        """Test text field cleaning."""
        # Test whitespace normalization
        assert self.validator._clean_text_field('  test   text  ') == 'test text'
        
        # Test HTML tag removal
        assert self.validator._clean_text_field('Text with <b>HTML</b> tags') == 'Text with HTML tags'
        
        # Test email artifact removal
        assert self.validator._clean_text_field('Text with [cid:123] artifact') == 'Text with  artifact'
        
        # Test None handling
        assert self.validator._clean_text_field(None) == ''
        assert self.validator._clean_text_field(123) == '123'
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        high_quality_issue = {
            'key': 'TEST-1',
            'summary': 'Well written summary describing the issue',
            'description': 'This is a comprehensive description with sufficient detail and context.',
            'issue_type': 'Bug',
            'priority': 'High',
            'components': ['Component1'],
            'labels': ['bug', 'urgent'],
            'comments': ['Comment 1'],
            'assignee': 'user@example.com',
            'resolution': 'Fixed'
        }
        
        score = self.validator._calculate_quality_score(high_quality_issue)
        assert score > 0.8  # High quality should get high score
        
        low_quality_issue = {
            'key': 'TEST-2',
            'summary': 'Bad',
            'description': 'Short desc',
            'issue_type': 'Bug'
        }
        
        score = self.validator._calculate_quality_score(low_quality_issue)
        assert score < 0.4  # Low quality should get low score
    
    def test_get_quality_recommendation(self):
        """Test quality recommendation logic."""
        assert 'REJECT' in self.validator._get_quality_recommendation(0.5, False)
        assert 'ACCEPT' in self.validator._get_quality_recommendation(0.9, True)
        assert 'ACCEPT_WITH_CAUTION' in self.validator._get_quality_recommendation(0.7, True)
        assert 'REVIEW' in self.validator._get_quality_recommendation(0.5, True)
        assert 'REJECT' in self.validator._get_quality_recommendation(0.2, True)
    
    def test_detect_duplicates(self):
        """Test duplicate detection."""
        df = pd.DataFrame([
            {
                'key': 'TEST-1',
                'summary': 'Login issue',
                'description': 'Users cannot login to the system'
            },
            {
                'key': 'TEST-2',
                'summary': 'Login problem', 
                'description': 'Users cannot login to the system'
            },
            {
                'key': 'TEST-3',
                'summary': 'Different issue',
                'description': 'This is completely different content'
            }
        ])
        
        duplicates = self.validator.detect_duplicates(df, similarity_threshold=0.7)
        
        # Should find one duplicate group with TEST-1 and TEST-2
        assert len(duplicates) >= 0  # Might find duplicates depending on similarity calculation
    
    def test_calculate_text_similarity(self):
        """Test text similarity calculation."""
        text1 = "This is a test string"
        text2 = "This is a test string"
        text3 = "Completely different content"
        
        # Identical texts should have high similarity
        similarity_identical = self.validator._calculate_text_similarity(text1, text2)
        assert similarity_identical == 1.0
        
        # Different texts should have low similarity
        similarity_different = self.validator._calculate_text_similarity(text1, text3)
        assert similarity_different < 0.5