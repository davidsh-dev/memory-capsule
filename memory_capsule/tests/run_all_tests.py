"""
Main test runner for the Memory Capsule project.

This script runs all the test modules to verify that all components
are working correctly.
"""

import unittest
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.test_audio_recorder import TestAudioRecorder
from tests.test_whisper_transcriber import TestWhisperTranscriber
from tests.test_speaker_diarizer import TestSpeakerDiarizer
from tests.test_conversation_db import TestConversationDB
from tests.test_language_model import TestLanguageModel
from tests.test_memory_search import TestMemorySearch

def run_all_tests():
    """Run all test modules."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestAudioRecorder))
    test_suite.addTest(unittest.makeSuite(TestWhisperTranscriber))
    test_suite.addTest(unittest.makeSuite(TestSpeakerDiarizer))
    test_suite.addTest(unittest.makeSuite(TestConversationDB))
    test_suite.addTest(unittest.makeSuite(TestLanguageModel))
    test_suite.addTest(unittest.makeSuite(TestMemorySearch))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

if __name__ == "__main__":
    print("Running all Memory Capsule tests...")
    run_all_tests()
