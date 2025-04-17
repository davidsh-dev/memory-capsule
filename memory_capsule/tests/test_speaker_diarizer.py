"""
Test module for the Speaker Diarizer component.

This module provides tests for the SpeakerDiarizer class to verify
that speaker diarization functionality works correctly.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import soundfile as sf

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_capsule.diarization.speaker_diarizer import SpeakerDiarizer

class TestSpeakerDiarizer(unittest.TestCase):
    """Test cases for the SpeakerDiarizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize the diarizer
        self.diarizer = SpeakerDiarizer(sample_rate=16000)
        
        # Create a test audio file with silence
        self.sample_rate = 16000
        duration = 2.0  # 2 seconds
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
        """Test that the diarizer initializes with correct parameters."""
        self.assertEqual(self.diarizer.sample_rate, 16000)
    
    def test_diarize_silence(self):
        """Test diarizing silence."""
        # This test might be skipped if no diarization model is available
        if self.diarizer.diarization_model is None:
            self.skipTest("No diarization model available")
        
        result = self.diarizer.diarize(self.test_audio)
        
        # The result should be a list
        self.assertIsInstance(result, list)
        
        # For silence, there might be no speakers detected
        # We don't assert specific content as the model might produce different results
    
    def test_identify_speakers(self):
        """Test speaker identification."""
        # This test might be skipped if no diarization model is available
        if self.diarizer.diarization_model is None:
            self.skipTest("No diarization model available")
        
        # Create a sample diarization result
        diarization_result = [
            {"speaker": "Speaker_1", "start": 0.0, "end": 1.0},
            {"speaker": "Speaker_2", "start": 1.5, "end": 2.0}
        ]
        
        result = self.diarizer.identify_speakers(self.test_audio, diarization_result)
        
        # The result should be a list
        self.assertIsInstance(result, list)
        
        # The result should have the same length as the input
        self.assertEqual(len(result), len(diarization_result))
    
    def test_register_speaker(self):
        """Test speaker registration."""
        result = self.diarizer.register_speaker("TestSpeaker", self.test_audio)
        
        # The result should be a boolean
        self.assertIsInstance(result, bool)

def run_tests():
    """Run the speaker diarizer tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Speaker Diarizer tests...")
    run_tests()
