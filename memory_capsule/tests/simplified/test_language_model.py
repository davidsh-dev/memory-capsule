"""
Simplified test module for the Language Model component with import mocking.

This module provides tests for the LanguageModel class with mocked dependencies.
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Mock the openai and requests modules before importing LanguageModel
sys.modules['openai'] = MagicMock()
sys.modules['requests'] = MagicMock()

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the class to test
from memory_capsule.llm.language_model import LanguageModel

class TestLanguageModelSimplified(unittest.TestCase):
    """Test cases for the LanguageModel class with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patchers for openai and requests
        self.openai_patcher = patch('memory_capsule.llm.language_model.openai')
        self.requests_patcher = patch('memory_capsule.llm.language_model.requests')
        
        # Start the patchers
        self.mock_openai = self.openai_patcher.start()
        self.mock_requests = self.requests_patcher.start()
        
        # Mock the OpenAI ChatCompletion.create method
        self.mock_openai.ChatCompletion.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="This is a test response."))]
        )
        
        # Mock the requests.post method for local LLM
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is a test response from local LLM."}
        self.mock_requests.post.return_value = mock_response
        
        # Set up environment variable for testing
        os.environ["OPENAI_API_KEY"] = "test_api_key"
        
        # Initialize the language model
        self.model = LanguageModel(model_name="gpt-3.5-turbo", use_local=False)
        self.local_model = LanguageModel(model_name="llama2", use_local=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patchers
        self.openai_patcher.stop()
        self.requests_patcher.stop()
        
        # Clean up environment variables
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    def test_initialization(self):
        """Test that the language model initializes with correct parameters."""
        self.assertEqual(self.model.model_name, "gpt-3.5-turbo")
        self.assertFalse(self.model.use_local)
        self.assertEqual(self.model.api_key, "test_api_key")
        
        self.assertEqual(self.local_model.model_name, "llama2")
        self.assertTrue(self.local_model.use_local)
    
    def test_generate_response_cloud(self):
        """Test generating a response using a cloud model."""
        # Generate a response
        response = self.model.generate_response("Test prompt")
        
        # Check that OpenAI API was called
        self.mock_openai.ChatCompletion.create.assert_called_once()
        
        # Check that the response is correct
        self.assertEqual(response, "This is a test response.")
    
    def test_generate_response_local(self):
        """Test generating a response using a local model."""
        # Generate a response
        response = self.local_model.generate_response("Test prompt")
        
        # Check that requests.post was called
        self.mock_requests.post.assert_called_once()
        
        # Check that the response is correct
        self.assertEqual(response, "This is a test response from local LLM.")
    
    def test_chat_cloud(self):
        """Test chat functionality using a cloud model."""
        # Create a chat conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I'm an AI assistant."},
            {"role": "user", "content": "What can you do?"}
        ]
        
        # Generate a response
        response = self.model.chat(messages)
        
        # Check that OpenAI API was called
        self.mock_openai.ChatCompletion.create.assert_called_once()
        
        # Check that the response is correct
        self.assertEqual(response, "This is a test response.")
    
    def test_chat_local(self):
        """Test chat functionality using a local model."""
        # Create a chat conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I'm an AI assistant."},
            {"role": "user", "content": "What can you do?"}
        ]
        
        # Generate a response
        response = self.local_model.chat(messages)
        
        # Check that requests.post was called
        self.mock_requests.post.assert_called_once()
        
        # Check that the response is correct
        self.assertEqual(response, "This is a test response from local LLM.")

def run_tests():
    """Run the simplified language model tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Simplified Language Model tests...")
    run_tests()
