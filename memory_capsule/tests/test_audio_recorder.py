"""
Test module for the Audio Recorder component.

This module provides tests for the AudioRecorder class to verify
that audio capture functionality works correctly.
"""

import unittest
import numpy as np
import time
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_capsule.audio.recorder import AudioRecorder

class TestAudioRecorder(unittest.TestCase):
    """Test cases for the AudioRecorder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recorder = AudioRecorder(sample_rate=16000, channels=1, chunk_size=1024)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if self.recorder.is_running():
            self.recorder.stop()
    
    def test_initialization(self):
        """Test that the recorder initializes with correct parameters."""
        self.assertEqual(self.recorder.sample_rate, 16000)
        self.assertEqual(self.recorder.channels, 1)
        self.assertEqual(self.recorder.chunk_size, 1024)
        self.assertFalse(self.recorder.is_running())
    
    def test_start_stop(self):
        """Test starting and stopping the recorder."""
        # Start the recorder
        self.recorder.start()
        self.assertTrue(self.recorder.is_running())
        
        # Stop the recorder
        self.recorder.stop()
        self.assertFalse(self.recorder.is_running())
    
    def test_get_devices(self):
        """Test getting audio devices."""
        devices = self.recorder.get_devices()
        self.assertIsInstance(devices, list)
        # At least one device should be available (system default)
        self.assertGreater(len(devices), 0)
    
    def test_short_recording(self):
        """Test recording a short audio segment."""
        # Record for 0.5 seconds
        duration = 0.5
        audio_data = self.recorder.record_fixed_duration(duration)
        
        # Check that we got the expected number of samples
        expected_samples = int(self.recorder.sample_rate * duration)
        self.assertGreaterEqual(len(audio_data), expected_samples * 0.9)  # Allow for some buffer variation
        
        # Check that the audio data is in the expected format
        self.assertEqual(audio_data.dtype, np.int16)
        
        # If stereo was requested, check that we have 2 channels
        if self.recorder.channels == 2:
            self.assertEqual(audio_data.shape[1], 2)
    
    def test_get_audio_buffer(self):
        """Test getting an audio buffer of specified duration."""
        # Start the recorder
        self.recorder.start()
        
        # Wait a moment for the recorder to start
        time.sleep(0.1)
        
        # Get a 0.5 second buffer
        duration = 0.5
        buffer = self.recorder.get_audio_buffer(duration)
        
        # Stop the recorder
        self.recorder.stop()
        
        # Check that we got the expected number of samples
        expected_samples = int(self.recorder.sample_rate * duration)
        self.assertGreaterEqual(len(buffer), expected_samples * 0.9)  # Allow for some buffer variation
        
        # Check that the buffer is in the expected format
        self.assertEqual(buffer.dtype, np.int16)

def run_tests():
    """Run the audio recorder tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Audio Recorder tests...")
    run_tests()
