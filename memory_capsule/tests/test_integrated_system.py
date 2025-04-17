"""
Test module for the integrated Memory Capsule system.

This script tests the integration of all components together.
"""

import unittest
import os
import sys
import time
import threading
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_capsule.run import MemoryCapsule

class TestIntegratedSystem(unittest.TestCase):
    """Test cases for the integrated Memory Capsule system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_file.name
        self.temp_file.close()
        
        # Create a configuration with test settings
        self.config = {
            'db_path': self.db_path,
            'whisper_model': 'tiny',
            'sample_rate': 16000,
            'channels': 1,
            'chunk_size': 1024,
            'use_local_llm': False,
            'llm_model': 'gpt-3.5-turbo',
            'embedding_model': 'all-MiniLM-L6-v2'
        }
        
        # Mock components to avoid actual audio recording and model loading
        self.setup_mocks()
        
        # Initialize the memory capsule with mocked components
        self.memory_capsule = MemoryCapsule(self.config)
        
        # Replace actual components with mocks
        self.apply_mocks()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the memory capsule if it's running
        if hasattr(self, 'memory_capsule') and self.memory_capsule.running:
            self.memory_capsule.stop()
        
        # Remove the temporary database file
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def setup_mocks(self):
        """Set up mock objects for components."""
        # Mock AudioRecorder
        self.mock_audio_recorder = MagicMock()
        self.mock_audio_recorder.start.return_value = None
        self.mock_audio_recorder.stop.return_value = None
        self.mock_audio_recorder.get_audio.return_value = np.zeros((1024, 1), dtype=np.int16)
        
        # Mock WhisperTranscriber
        self.mock_transcriber = MagicMock()
        self.mock_transcriber.transcribe.return_value = {
            'text': 'This is a test transcription.',
            'segments': [
                {'start': 0.0, 'end': 2.0, 'text': 'This is a test transcription.'}
            ]
        }
        
        # Mock SpeakerDiarizer
        self.mock_diarizer = MagicMock()
        self.mock_diarizer.diarize.return_value = [
            {'speaker': 'Speaker_A', 'start': 0.0, 'end': 2.0}
        ]
        
        # Mock LanguageModel
        self.mock_language_model = MagicMock()
        self.mock_language_model.generate_response.return_value = "This is a test response."
        
        # Mock SpeechSynthesizer
        self.mock_speech_synthesizer = MagicMock()
        self.mock_speech_synthesizer.speak.return_value = True
    
    def apply_mocks(self):
        """Apply mock objects to the memory capsule."""
        self.memory_capsule.audio_recorder = self.mock_audio_recorder
        self.memory_capsule.transcriber = self.mock_transcriber
        self.memory_capsule.diarizer = self.mock_diarizer
        self.memory_capsule.language_model = self.mock_language_model
        self.memory_capsule.speech_synthesizer = self.mock_speech_synthesizer
    
    def test_initialization(self):
        """Test that the memory capsule initializes correctly."""
        self.assertIsNotNone(self.memory_capsule.db)
        self.assertIsNotNone(self.memory_capsule.audio_recorder)
        self.assertIsNotNone(self.memory_capsule.transcriber)
        self.assertIsNotNone(self.memory_capsule.diarizer)
        self.assertIsNotNone(self.memory_capsule.language_model)
        self.assertIsNotNone(self.memory_capsule.memory_search)
        self.assertIsNotNone(self.memory_capsule.qa)
        self.assertIsNotNone(self.memory_capsule.speech_synthesizer)
        self.assertIsNotNone(self.memory_capsule.ui)
        
        self.assertFalse(self.memory_capsule.running)
        self.assertFalse(self.memory_capsule.paused)
    
    def test_start_stop(self):
        """Test starting and stopping the memory capsule."""
        # Start the memory capsule in a separate thread
        thread = threading.Thread(target=self._start_memory_capsule)
        thread.daemon = True
        thread.start()
        
        # Wait a moment for it to start
        time.sleep(0.5)
        
        # Check that it's running
        self.assertTrue(self.memory_capsule.running)
        self.assertFalse(self.memory_capsule.paused)
        
        # Stop the memory capsule
        self.memory_capsule.stop()
        
        # Check that it's stopped
        self.assertFalse(self.memory_capsule.running)
    
    def _start_memory_capsule(self):
        """Start the memory capsule and immediately return."""
        # Replace the UI start method to avoid blocking
        self.memory_capsule.ui.start = lambda: None
        
        # Start the memory capsule
        self.memory_capsule.start()
    
    def test_pause_resume(self):
        """Test pausing and resuming the memory capsule."""
        # Start the memory capsule in a separate thread
        thread = threading.Thread(target=self._start_memory_capsule)
        thread.daemon = True
        thread.start()
        
        # Wait a moment for it to start
        time.sleep(0.5)
        
        # Pause the memory capsule
        self.memory_capsule.pause()
        
        # Check that it's paused
        self.assertTrue(self.memory_capsule.paused)
        
        # Resume the memory capsule
        self.memory_capsule.resume()
        
        # Check that it's resumed
        self.assertFalse(self.memory_capsule.paused)
        
        # Stop the memory capsule
        self.memory_capsule.stop()
    
    def test_ask_question(self):
        """Test asking a question to the assistant."""
        # Start the memory capsule in a separate thread
        thread = threading.Thread(target=self._start_memory_capsule)
        thread.daemon = True
        thread.start()
        
        # Wait a moment for it to start
        time.sleep(0.5)
        
        # Ask a question
        answer = self.memory_capsule.ask_question("What is the capital of France?")
        
        # Check that we got an answer
        self.assertEqual(answer, "This is a test response.")
        
        # Check that the speech synthesizer was called
        self.mock_speech_synthesizer.speak.assert_called_once()
        
        # Stop the memory capsule
        self.memory_capsule.stop()
    
    def test_process_audio_segment(self):
        """Test processing an audio segment."""
        # Create a test audio segment
        audio_data = np.zeros((16000,), dtype=np.int16)  # 1 second of silence
        
        # Process the audio segment
        self.memory_capsule._process_audio_segment(audio_data)
        
        # Check that the transcriber was called
        self.mock_transcriber.transcribe.assert_called_once()
        
        # Check that the diarizer was called
        self.mock_diarizer.diarize.assert_called_once()
        
        # Check that an utterance was added to the transcript buffer
        self.assertEqual(len(self.memory_capsule.transcript_buffer), 1)
        self.assertEqual(self.memory_capsule.transcript_buffer[0]['text'], 'This is a test transcription.')
        self.assertEqual(self.memory_capsule.transcript_buffer[0]['speaker'], 'Speaker_A')

def run_tests():
    """Run the integrated system tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Integrated System tests...")
    run_tests()
