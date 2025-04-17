"""
Simplified test module for the Audio Recorder component with import mocking.

This module provides tests for the AudioRecorder class with mocked dependencies.
"""

import unittest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Mock the sounddevice module before importing AudioRecorder
sys.modules['sounddevice'] = MagicMock()

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the class to test
from memory_capsule.audio.recorder import AudioRecorder

class TestAudioRecorderSimplified(unittest.TestCase):
    """Test cases for the AudioRecorder class with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a patcher for sounddevice
        self.sd_patcher = patch('memory_capsule.audio.recorder.sd')
        self.mock_sd = self.sd_patcher.start()
        
        # Mock the InputStream
        self.mock_stream = MagicMock()
        self.mock_sd.InputStream.return_value = self.mock_stream
        
        # Mock the read method to return a fixed array
        self.mock_stream.read.return_value = (np.zeros((1024, 1), dtype=np.int16), False)
        
        # Initialize the recorder
        self.recorder = AudioRecorder(sample_rate=16000, channels=1, chunk_size=1024)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patcher
        self.sd_patcher.stop()
    
    def test_initialization(self):
        """Test that the recorder initializes with correct parameters."""
        self.assertEqual(self.recorder.sample_rate, 16000)
        self.assertEqual(self.recorder.channels, 1)
        self.assertEqual(self.recorder.chunk_size, 1024)
        self.assertFalse(self.recorder.is_recording)
    
    def test_start_recording(self):
        """Test starting the recorder."""
        self.recorder.start()
        
        # Check that the stream was created with correct parameters
        self.mock_sd.InputStream.assert_called_once()
        args, kwargs = self.mock_sd.InputStream.call_args
        self.assertEqual(kwargs['samplerate'], 16000)
        self.assertEqual(kwargs['channels'], 1)
        self.assertEqual(kwargs['blocksize'], 1024)
        
        # Check that the stream was started
        self.mock_stream.start.assert_called_once()
        
        # Check that is_recording is True
        self.assertTrue(self.recorder.is_recording)
    
    def test_stop_recording(self):
        """Test stopping the recorder."""
        # Start recording first
        self.recorder.start()
        
        # Then stop
        self.recorder.stop()
        
        # Check that the stream was stopped
        self.mock_stream.stop.assert_called_once()
        self.mock_stream.close.assert_called_once()
        
        # Check that is_recording is False
        self.assertFalse(self.recorder.is_recording)
    
    def test_get_audio(self):
        """Test getting audio from the recorder."""
        # Start recording first
        self.recorder.start()
        
        # Get audio
        audio = self.recorder.get_audio()
        
        # Check that read was called
        self.mock_stream.read.assert_called_once()
        
        # Check that audio is a numpy array of the correct shape
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.shape, (1024, 1))
    
    def test_get_audio_not_recording(self):
        """Test getting audio when not recording."""
        # Don't start recording
        
        # Get audio
        audio = self.recorder.get_audio()
        
        # Check that audio is None
        self.assertIsNone(audio)

def run_tests():
    """Run the simplified audio recorder tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Simplified Audio Recorder tests...")
    run_tests()
