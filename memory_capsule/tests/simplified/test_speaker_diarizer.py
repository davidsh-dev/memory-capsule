"""
Simplified test module for the Speaker Diarizer component with import mocking.

This module provides tests for the SpeakerDiarizer class with mocked dependencies.
"""

import unittest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Mock all required dependencies before importing SpeakerDiarizer
sys.modules['diart'] = MagicMock()
sys.modules['pyannote'] = MagicMock()
sys.modules['pyannote.audio'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the class to test
from memory_capsule.diarization.speaker_diarizer import SpeakerDiarizer

class TestSpeakerDiarizerSimplified(unittest.TestCase):
    """Test cases for the SpeakerDiarizer class with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patchers for diart and pyannote
        self.diart_patcher = patch('memory_capsule.diarization.speaker_diarizer.diart')
        self.pyannote_patcher = patch('memory_capsule.diarization.speaker_diarizer.pyannote')
        
        # Start the patchers
        self.mock_diart = self.diart_patcher.start()
        self.mock_pyannote = self.pyannote_patcher.start()
        
        # Mock the diarization model
        self.mock_diarization_model = MagicMock()
        self.mock_diart.SpeakerDiarization.return_value = self.mock_diarization_model
        
        # Mock the diarize method to return a fixed result
        mock_result = MagicMock()
        mock_result.labels.return_value = ["Speaker_1", "Speaker_2"]
        mock_result.for_json.return_value = [
            {"speaker": "Speaker_1", "start": 0.0, "end": 1.0},
            {"speaker": "Speaker_2", "start": 1.5, "end": 2.0}
        ]
        self.mock_diarization_model.return_value = mock_result
        
        # Initialize the diarizer
        self.diarizer = SpeakerDiarizer(sample_rate=16000)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patchers
        self.diart_patcher.stop()
        self.pyannote_patcher.stop()
    
    def test_initialization(self):
        """Test that the diarizer initializes with correct parameters."""
        self.assertEqual(self.diarizer.sample_rate, 16000)
        self.assertIsNotNone(self.diarizer.diarization_model)
    
    def test_diarize(self):
        """Test diarizing audio."""
        # Create a test audio array
        audio = np.zeros((16000 * 2,), dtype=np.float32)  # 2 seconds of silence
        
        # Diarize the audio
        result = self.diarizer.diarize(audio)
        
        # Check that the diarization model was called
        self.mock_diarization_model.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["speaker"], "Speaker_1")
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 1.0)
        self.assertEqual(result[1]["speaker"], "Speaker_2")
        self.assertEqual(result[1]["start"], 1.5)
        self.assertEqual(result[1]["end"], 2.0)
    
    def test_identify_speakers(self):
        """Test speaker identification."""
        # Create a test audio array
        audio = np.zeros((16000 * 2,), dtype=np.float32)  # 2 seconds of silence
        
        # Create a sample diarization result
        diarization_result = [
            {"speaker": "Speaker_1", "start": 0.0, "end": 1.0},
            {"speaker": "Speaker_2", "start": 1.5, "end": 2.0}
        ]
        
        # Mock the speaker identification
        with patch.object(self.diarizer, '_identify_speaker') as mock_identify:
            mock_identify.side_effect = ["John", "Jane"]
            
            # Identify speakers
            result = self.diarizer.identify_speakers(audio, diarization_result)
            
            # Check that _identify_speaker was called twice
            self.assertEqual(mock_identify.call_count, 2)
            
            # Check that the result is correct
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["speaker"], "John")
            self.assertEqual(result[1]["speaker"], "Jane")
    
    def test_register_speaker(self):
        """Test speaker registration."""
        # Create a test audio array
        audio = np.zeros((16000 * 2,), dtype=np.float32)  # 2 seconds of silence
        
        # Mock the speaker embedding
        with patch.object(self.diarizer, '_get_speaker_embedding') as mock_embedding:
            mock_embedding.return_value = np.zeros((192,), dtype=np.float32)
            
            # Register a speaker
            result = self.diarizer.register_speaker("TestSpeaker", audio)
            
            # Check that _get_speaker_embedding was called
            mock_embedding.assert_called_once_with(audio)
            
            # Check that the result is correct
            self.assertTrue(result)
            self.assertIn("TestSpeaker", self.diarizer.speaker_embeddings)

def run_tests():
    """Run the simplified speaker diarizer tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Simplified Speaker Diarizer tests...")
    run_tests()
