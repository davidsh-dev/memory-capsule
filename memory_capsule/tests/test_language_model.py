"""
Test module for the Language Model component.

This module provides tests for the LanguageModel class to verify
that language model integration works correctly.
"""

import unittest
import os
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_capsule.llm.language_model import LanguageModel

class TestLanguageModel(unittest.TestCase):
    """Test cases for the LanguageModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize with a mock API key for testing
        os.environ["OPENAI_API_KEY"] = "test_api_key"
        self.model = LanguageModel(model_name="gpt-3.5-turbo", use_local=False)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up environment variables
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    def test_initialization(self):
        """Test that the language model initializes with correct parameters."""
        self.assertEqual(self.model.model_name, "gpt-3.5-turbo")
        self.assertFalse(self.model.use_local)
        self.assertEqual(self.model.api_key, "test_api_key")
    
    @patch('requests.post')
    def test_generate_local(self, mock_post):
        """Test generating a response using a local model."""
        # Create a local model
        local_model = LanguageModel(model_name="llama2", use_local=True)
        
        # Mock the response from Ollama
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is a test response."}
        mock_post.return_value = mock_response
        
        # Generate a response
        response = local_model._generate_local("Test prompt", 100, 0.7)
        
        # Check that the response is correct
        self.assertEqual(response, "This is a test response.")
        
        # Check that the request was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        self.assertEqual(kwargs["json"]["model"], "llama2")
        self.assertEqual(kwargs["json"]["prompt"], "Test prompt")
    
    @patch('openai.ChatCompletion.create')
    def test_generate_cloud(self, mock_create):
        """Test generating a response using a cloud model."""
        # Mock the response from OpenAI
        mock_create.return_value.choices = [MagicMock(message=MagicMock(content="This is a test response."))]
        
        # Generate a response
        response = self.model._generate_cloud("Test prompt", 100, 0.7)
        
        # Check that the response is correct
        self.assertEqual(response, "This is a test response.")
        
        # Check that the request was made correctly
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        self.assertEqual(kwargs["model"], "gpt-3.5-turbo")
        self.assertEqual(kwargs["messages"][1]["content"], "Test prompt")
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertEqual(kwargs["temperature"], 0.7)
    
    @patch('requests.post')
    def test_chat_local(self, mock_post):
        """Test chat functionality using a local model."""
        # Create a local model
        local_model = LanguageModel(model_name="llama2", use_local=True)
        
        # Mock the response from Ollama
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is a test response."}
        mock_post.return_value = mock_response
        
        # Create a chat conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I'm an AI assistant."},
            {"role": "user", "content": "What can you do?"}
        ]
        
        # Generate a response
        response = local_model.chat(messages, 100, 0.7)
        
        # Check that the response is correct
        self.assertEqual(response, "This is a test response.")
        
        # Check that the request was made correctly
        mock_post.assert_called_once()
    
    @patch('openai.ChatCompletion.create')
    def test_chat_cloud(self, mock_create):
        """Test chat functionality using a cloud model."""
        # Mock the response from OpenAI
        mock_create.return_value.choices = [MagicMock(message=MagicMock(content="This is a test response."))]
        
        # Create a chat conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I'm an AI assistant."},
            {"role": "user", "content": "What can you do?"}
        ]
        
        # Generate a response
        response = self.model.chat(messages, 100, 0.7)
        
        # Check that the response is correct
        self.assertEqual(response, "This is a test response.")
        
        # Check that the request was made correctly
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        self.assertEqual(kwargs["model"], "gpt-3.5-turbo")
        self.assertEqual(kwargs["messages"], messages)
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertEqual(kwargs["temperature"], 0.7)
    
    @patch('requests.get')
    def test_get_local_models(self, mock_get):
        """Test getting available local models."""
        # Create a local model
        local_model = LanguageModel(model_name="llama2", use_local=True)
        
        # Mock the response from Ollama
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama2"}, {"name": "mistral"}]}
        mock_get.return_value = mock_response
        
        # Get available models
        models = local_model._get_local_models()
        
        # Check that the models are correct
        self.assertEqual(models, ["llama2", "mistral"])
        
        # Check that the request was made correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/tags")

def run_tests():
    """Run the language model tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Language Model tests...")
    run_tests()
