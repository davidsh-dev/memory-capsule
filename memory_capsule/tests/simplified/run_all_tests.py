"""
Simplified test runner for the Memory Capsule project.

This module runs all the simplified tests with mocked dependencies.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the test modules
from tests.simplified.test_audio_recorder import TestAudioRecorderSimplified
from tests.simplified.test_whisper_transcriber import TestWhisperTranscriberSimplified
from tests.simplified.test_speaker_diarizer import TestSpeakerDiarizerSimplified
from tests.simplified.test_conversation_db import TestConversationDBSimplified
from tests.simplified.test_language_model import TestLanguageModelSimplified
from tests.simplified.test_memory_search import TestMemorySearchSimplified

def run_all_tests():
    """Run all simplified tests."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestAudioRecorderSimplified))
    test_suite.addTest(unittest.makeSuite(TestWhisperTranscriberSimplified))
    test_suite.addTest(unittest.makeSuite(TestSpeakerDiarizerSimplified))
    test_suite.addTest(unittest.makeSuite(TestConversationDBSimplified))
    test_suite.addTest(unittest.makeSuite(TestLanguageModelSimplified))
    test_suite.addTest(unittest.makeSuite(TestMemorySearchSimplified))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(test_suite)
    
    # Return the result
    return result

if __name__ == "__main__":
    print("Running all simplified tests for Memory Capsule...")
    result = run_all_tests()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Exit with appropriate code
    sys.exit(len(result.failures) + len(result.errors))
