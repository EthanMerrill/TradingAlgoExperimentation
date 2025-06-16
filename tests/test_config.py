#!/usr/bin/env python3
"""
Unit tests for the Config class.
"""
import unittest
from unittest.mock import Mock, patch, mock_open
import os
import json
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear any existing environment variables
        env_vars_to_clear = [
            'ENVIRONMENT', 'ALPACA_DEV_PAPER_KEY', 'ALPACA_DEV_PAPER_SECRET',
            'ALPACA_QA_PAPER_KEY', 'ALPACA_QA_PAPER_SECRET',
            'ALPACA_LIVE_KEY', 'ALPACA_LIVE_SECRET'
        ]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'dev', 'ALPACA_DEV_PAPER_KEY': 'test_key', 'ALPACA_DEV_PAPER_SECRET': 'test_secret'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 365}'))
    def test_config_init_dev_environment(self, mock_dotenv):
        """Test Config initialization in dev environment."""
        from config import Config

        config = Config()

        self.assertEqual(config.ENVIRONMENT, 'dev')
        mock_dotenv.assert_called_once()

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'invalid'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 365}'))
    @patch('builtins.print')
    def test_config_invalid_environment(self, mock_print, mock_dotenv):
        """Test Config with invalid environment setting."""
        from config import Config

        config = Config()

        self.assertEqual(config.ENVIRONMENT, 'dev')  # Should default to 'dev'
        mock_print.assert_called()

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 365}'))
    def test_config_missing_required_vars(self, mock_dotenv):
        """Test Config with missing required environment variables."""
        from config import Config

        with self.assertRaises(ValueError) as context:
            Config()

        self.assertIn("Missing required environment variables",
                      str(context.exception))

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'qa', 'ALPACA_QA_PAPER_KEY': 'qa_key', 'ALPACA_QA_PAPER_SECRET': 'qa_secret'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 180}'))
    def test_config_qa_environment(self, mock_dotenv):
        """Test Config initialization in QA environment."""
        from config import Config

        config = Config()

        self.assertEqual(config.ENVIRONMENT, 'qa')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'prod', 'ALPACA_LIVE_KEY': 'live_key', 'ALPACA_LIVE_SECRET': 'live_secret'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 730}'))
    def test_config_prod_environment(self, mock_dotenv):
        """Test Config initialization in prod environment."""
        from config import Config

        config = Config()

        self.assertEqual(config.ENVIRONMENT, 'prod')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'dev', 'ALPACA_DEV_PAPER_KEY': 'test_key', 'ALPACA_DEV_PAPER_SECRET': 'test_secret'})
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_config_missing_json_file(self, mock_dotenv):
        """Test Config with missing JSON configuration file."""
        from config import Config

        with self.assertRaises(FileNotFoundError):
            Config()

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'dev', 'ALPACA_DEV_PAPER_KEY': 'test_key', 'ALPACA_DEV_PAPER_SECRET': 'test_secret'})
    @patch('builtins.open', mock_open(read_data='invalid json'))
    def test_config_invalid_json(self, mock_dotenv):
        """Test Config with invalid JSON configuration."""
        from config import Config

        with self.assertRaises(json.JSONDecodeError):
            Config()

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'dev', 'ALPACA_DEV_PAPER_KEY': 'test_key', 'ALPACA_DEV_PAPER_SECRET': 'test_secret'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 365, "MAX_PORTFOLIO_VALUE": 100000}'))
    def test_get_alpaca_config_dev(self, mock_dotenv):
        """Test getting Alpaca configuration for dev environment."""
        from config import Config

        config = Config()
        alpaca_config = config.get_alpaca_config()

        self.assertEqual(alpaca_config['api_key'], 'test_key')
        self.assertEqual(alpaca_config['secret_key'], 'test_secret')
        self.assertTrue(alpaca_config['paper'])

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'prod', 'ALPACA_LIVE_KEY': 'live_key', 'ALPACA_LIVE_SECRET': 'live_secret'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 730}'))
    def test_get_alpaca_config_prod(self, mock_dotenv):
        """Test getting Alpaca configuration for prod environment."""
        from config import Config

        config = Config()
        alpaca_config = config.get_alpaca_config()

        self.assertEqual(alpaca_config['api_key'], 'live_key')
        self.assertEqual(alpaca_config['secret_key'], 'live_secret')
        self.assertFalse(alpaca_config['paper'])


if __name__ == '__main__':
    unittest.main()
