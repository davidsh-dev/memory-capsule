"""
Test module for the Whisper Transcriber component.

This module provides tests for the WhisperTranscriber class to verify
that speech transcription functionality works correctly.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import soundfile as sf

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_capsule.transcription.whisper_transcriber import WhisperTranscriber

class TestWhisperTranscriber(unittest.TestCase):
    """Test cases for the WhisperTranscriber class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use the tiny model for faster tests
        self.transcriber = WhisperTranscriber(model_name="tiny")
        
        # Create a test audio file with silence
        self.sample_rate = 16000
        duration = 1.0  # 1 second
        self.test_audio = np.zeros(int(self.sample_rate * duration), dtype=np.float32)
        
        # Create a temporary file for the test audio
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(self.temp_file.name, self.test_audio, self.sample_rate)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Close and remove the temporary file
        self.temp_file.close()
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """Test that the transcriber initializes with correct parameters."""
        self.assertEqual(self.transcriber.model_name, "tiny")
        self.assertIsNotNone(self.transcriber.model)
    
    def test_get_available_models(self):
        """Test getting available Whisper models."""
        models = self.transcriber.get_available_models()
        self.assertIsInstance(models, list)
        self.assertIn("tiny", models)
        self.assertIn("base", models)
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.transcriber.get_model_info()
        self.assertIsInstance(info, dict)
        self.assertEqual(info["name"], "tiny")
        self.assertTrue(info["loaded"])
    
    def test_transcribe_silence(self):
        """Test transcribing silence."""
        result = self.transcriber.transcribe(self.test_audio)
        
        # The result should be a dictionary
        self.assertIsInstance(result, dict)
        
        # The result should have a 'text' key
        self.assertIn('text', result)
        
        # For silence, the text might be empty or contain minimal content
        # We don't assert specific content as the model might produce different results
    
    def test_transcribe_file(self):
        """Test transcribing from a file."""
        result = self.transcriber.transcribe_file(self.temp_file.name)
        
        # The result should be a dictionary
        self.assertIsInstance(result, dict)
        
        # The result should have a 'text' key
        self.assertIn('text', result)

def run_tests():
    """Run the whisper transcriber tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Whisper Transcriber tests...")
    run_tests()
