#!/usr/bin/env python3
"""
Unit tests for the Config class.
"""
import json
import os
import sys
import unittest
from unittest.mock import Mock, mock_open, patch

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
        from app.config import Config

        test_config = Config()

        self.assertEqual(test_config.ENVIRONMENT, 'dev')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'invalid'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 365}'))
    @patch('builtins.print')
    def test_config_invalid_environment(self, mock_print, mock_dotenv):
        """Test Config with invalid environment setting."""
        from app.config import Config

        test_config = Config()

        # Should default to 'dev'
        self.assertEqual(test_config.ENVIRONMENT, 'dev')
        mock_print.assert_called()

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 365}'))
    def test_config_missing_required_vars(self, mock_dotenv):
        """Test Config with missing required environment variables."""
        from app.config import Config

        # Note: Config doesn't raise ValueError for missing vars, just logs warnings
        test_config = Config()

        # The config should still initialize with defaults
        self.assertEqual(test_config.ENVIRONMENT, 'dev')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'qa', 'ALPACA_QA_PAPER_KEY': 'qa_key', 'ALPACA_QA_PAPER_SECRET': 'qa_secret'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 180}'))
    def test_config_qa_environment(self, mock_dotenv):
        """Test Config initialization in QA environment."""
        from app.config import Config

        test_config = Config()

        self.assertEqual(test_config.ENVIRONMENT, 'qa')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'prod', 'ALPACA_LIVE_KEY': 'live_key', 'ALPACA_LIVE_SECRET': 'live_secret'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 730}'))
    def test_config_prod_environment(self, mock_dotenv):
        """Test Config initialization in prod environment."""
        from app.config import Config

        test_config = Config()

        self.assertEqual(test_config.ENVIRONMENT, 'prod')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'dev', 'ALPACA_DEV_PAPER_KEY': 'test_key', 'ALPACA_DEV_PAPER_SECRET': 'test_secret'})
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_config_missing_json_file(self, mock_open_file, mock_dotenv):
        """Test Config with missing JSON configuration file."""
        from app.config import Config

        # Config doesn't raise FileNotFoundError, it uses defaults
        test_config = Config()
        self.assertEqual(test_config.ENVIRONMENT, 'dev')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'dev', 'ALPACA_DEV_PAPER_KEY': 'test_key', 'ALPACA_DEV_PAPER_SECRET': 'test_secret'})
    @patch('builtins.open', mock_open(read_data='invalid json'))
    def test_config_invalid_json(self, mock_dotenv):
        """Test Config with invalid JSON configuration."""
        from app.config import Config

        # Config doesn't raise JSONDecodeError, it uses defaults
        test_config = Config()
        self.assertEqual(test_config.ENVIRONMENT, 'dev')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {'ENVIRONMENT': 'dev', 'ALPACA_DEV_PAPER_KEY': 'test_key', 'ALPACA_DEV_PAPER_SECRET': 'test_secret'})
    @patch('builtins.open', mock_open(read_data='{"BACKTEST_START_DAYS": 365, "MAX_PORTFOLIO_VALUE": 100000}'))
    def test_get_alpaca_config_dev(self, mock_dotenv):
        """Test getting Alpaca configuration for dev environment."""
        from app.config import Config

        test_config = Config()
        alpaca_config = test_config.get_alpaca_config()

        self.assertEqual(alpaca_config['api_key'], 'test_key')
        self.assertEqual(alpaca_config['secret_key'], 'test_secret')
        self.assertEqual(alpaca_config['base_url'],
                         'https://paper-api.alpaca.markets')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'prod',
        'ALPACA_LIVE_KEY': 'live_key',
        'ALPACA_LIVE_SECRET': 'live_secret',
        'ALPACA_QA_PAPER_KEY': 'qa_key',
        'ALPACA_QA_PAPER_SECRET': 'qa_secret'
    })
    @patch('builtins.open', mock_open(read_data='{"trading": {"paper_trade": false}}'))
    def test_get_alpaca_config_prod(self, mock_dotenv):
        """Test getting Alpaca configuration for prod environment with live trading."""
        from app.config import Config

        test_config = Config()
        alpaca_config = test_config.get_alpaca_config()

        self.assertEqual(alpaca_config['api_key'], 'live_key')
        self.assertEqual(alpaca_config['secret_key'], 'live_secret')
        self.assertEqual(alpaca_config['base_url'],
                         'https://api.alpaca.markets')

    @patch('config.load_dotenv')
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'prod',
        'ALPACA_LIVE_KEY': 'live_key',
        'ALPACA_LIVE_SECRET': 'live_secret',
        'ALPACA_QA_PAPER_KEY': 'qa_paper_key',
        'ALPACA_QA_PAPER_SECRET': 'qa_paper_secret'
    })
    @patch('builtins.open', mock_open(read_data='{"trading": {"paper_trade": true}}'))
    def test_get_alpaca_config_prod_paper(self, mock_dotenv):
        """Test getting Alpaca configuration for prod environment with paper trading."""
        from app.config import Config

        test_config = Config()
        alpaca_config = test_config.get_alpaca_config()

        self.assertEqual(alpaca_config['api_key'], 'qa_paper_key')
        self.assertEqual(alpaca_config['secret_key'], 'qa_paper_secret')
        self.assertEqual(alpaca_config['base_url'],
                         'https://paper-api.alpaca.markets')


if __name__ == '__main__':
    unittest.main()
