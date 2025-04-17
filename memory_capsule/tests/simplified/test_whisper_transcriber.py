"""
Simplified test module for the Whisper Transcriber component with import mocking.

This module provides tests for the WhisperTranscriber class with mocked dependencies.
"""

import unittest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Mock all required dependencies before importing WhisperTranscriber
sys.modules['whisper'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the class to test
from memory_capsule.transcription.whisper_transcriber import WhisperTranscriber

class TestWhisperTranscriberSimplified(unittest.TestCase):
    """Test cases for the WhisperTranscriber class with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a patcher for whisper
        self.whisper_patcher = patch('memory_capsule.transcription.whisper_transcriber.whisper')
        self.mock_whisper = self.whisper_patcher.start()
        
        # Mock the whisper model
        self.mock_model = MagicMock()
        self.mock_whisper.load_model.return_value = self.mock_model
        
        # Mock the transcribe method to return a fixed result
        self.mock_model.transcribe.return_value = {
            'text': 'This is a test transcription.',
            'segments': [
                {'start': 0.0, 'end': 2.0, 'text': 'This is a test transcription.'}
            ]
        }
        
        # Initialize the transcriber
        self.transcriber = WhisperTranscriber(model_name="tiny")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patcher
        self.whisper_patcher.stop()
    
    def test_initialization(self):
        """Test that the transcriber initializes with correct parameters."""
        self.assertEqual(self.transcriber.model_name, "tiny")
        self.assertEqual(self.transcriber.model, self.mock_model)
        
        # Check that whisper.load_model was called with correct parameters
        self.mock_whisper.load_model.assert_called_once_with("tiny", device=None)
    
    def test_transcribe(self):
        """Test transcribing audio."""
        # Create a test audio array
        audio = np.zeros((16000,), dtype=np.float32)  # 1 second of silence
        
        # Transcribe the audio
        result = self.transcriber.transcribe(audio)
        
        # Check that the model's transcribe method was called
        self.mock_model.transcribe.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result['text'], 'This is a test transcription.')
        self.assertEqual(len(result['segments']), 1)
        self.assertEqual(result['segments'][0]['text'], 'This is a test transcription.')
    
    def test_transcribe_file(self):
        """Test transcribing from a file."""
        # Mock the soundfile module
        with patch('memory_capsule.transcription.whisper_transcriber.sf') as mock_sf:
            # Mock the read method to return a fixed array
            mock_sf.read.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
            
            # Transcribe a file
            result = self.transcriber.transcribe_file("test.wav")
            
            # Check that sf.read was called
            mock_sf.read.assert_called_once_with("test.wav")
            
            # Check that the model's transcribe method was called
            self.mock_model.transcribe.assert_called_once()
            
            # Check that the result is correct
            self.assertEqual(result['text'], 'This is a test transcription.')
    
    def test_get_available_models(self):
        """Test getting available Whisper models."""
        # Mock the available models
        self.mock_whisper.available_models.return_value = ["tiny", "base", "small", "medium", "large"]
        
        # Get available models
        models = self.transcriber.get_available_models()
        
        # Check that the result is correct
        self.assertEqual(models, ["tiny", "base", "small", "medium", "large"])
    
    def test_get_model_info(self):
        """Test getting model information."""
        # Get model info
        info = self.transcriber.get_model_info()
        
        # Check that the result is correct
        self.assertIsInstance(info, dict)
        self.assertEqual(info["name"], "tiny")
        self.assertTrue(info["loaded"])

def run_tests():
    """Run the simplified whisper transcriber tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Simplified Whisper Transcriber tests...")
    run_tests()
