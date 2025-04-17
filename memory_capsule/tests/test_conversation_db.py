"""
Test module for the Conversation Database component.

This module provides tests for the ConversationDB class to verify
that database storage and retrieval functionality works correctly.
"""

import unittest
import os
import sys
import time
import tempfile
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_capsule.storage.conversation_db import ConversationDB

class TestConversationDB(unittest.TestCase):
    """Test cases for the ConversationDB class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_file.name
        self.temp_file.close()
        
        # Initialize the database
        self.db = ConversationDB(self.db_path)
        self.db.connect()
        
        # Sample utterance data for testing
        self.sample_utterance = {
            'speaker': 'Speaker_A',
            'text': 'This is a test utterance.',
            'start_time': 0.0,
            'end_time': 2.0,
            'timestamp': time.time()
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Disconnect from the database
        self.db.disconnect()
        
        # Remove the temporary database file
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_initialization(self):
        """Test that the database initializes correctly."""
        self.assertEqual(self.db.db_path, self.db_path)
        self.assertIsNotNone(self.db.conn)
        self.assertIsNotNone(self.db.cursor)
    
    def test_create_conversation(self):
        """Test creating a conversation."""
        # Create a conversation
        conversation_id = self.db.create_conversation(title="Test Conversation")
        
        # The ID should be a positive integer
        self.assertGreater(conversation_id, 0)
        
        # Get the conversation
        conversation = self.db.get_conversation(conversation_id)
        
        # Check that the conversation was created correctly
        self.assertEqual(conversation['id'], conversation_id)
        self.assertEqual(conversation['title'], "Test Conversation")
        self.assertIsNone(conversation['end_time'])
        self.assertEqual(len(conversation['utterances']), 0)
    
    def test_end_conversation(self):
        """Test ending a conversation."""
        # Create a conversation
        conversation_id = self.db.create_conversation()
        
        # End the conversation
        result = self.db.end_conversation(conversation_id)
        self.assertTrue(result)
        
        # Get the conversation
        conversation = self.db.get_conversation(conversation_id)
        
        # Check that the end time was set
        self.assertIsNotNone(conversation['end_time'])
    
    def test_add_utterance(self):
        """Test adding an utterance."""
        # Create a conversation
        conversation_id = self.db.create_conversation()
        
        # Add an utterance
        utterance_id = self.db.add_utterance(self.sample_utterance, conversation_id)
        
        # The ID should be a positive integer
        self.assertGreater(utterance_id, 0)
        
        # Get the utterance
        utterance = self.db.get_utterance(utterance_id)
        
        # Check that the utterance was added correctly
        self.assertEqual(utterance['id'], utterance_id)
        self.assertEqual(utterance['conversation_id'], conversation_id)
        self.assertEqual(utterance['speaker'], self.sample_utterance['speaker'])
        self.assertEqual(utterance['text'], self.sample_utterance['text'])
    
    def test_get_conversation_with_utterances(self):
        """Test getting a conversation with utterances."""
        # Create a conversation
        conversation_id = self.db.create_conversation()
        
        # Add multiple utterances
        utterance1 = self.sample_utterance.copy()
        utterance2 = self.sample_utterance.copy()
        utterance2['text'] = "This is another test utterance."
        utterance2['timestamp'] = time.time() + 1
        
        self.db.add_utterance(utterance1, conversation_id)
        self.db.add_utterance(utterance2, conversation_id)
        
        # Get the conversation
        conversation = self.db.get_conversation(conversation_id)
        
        # Check that the utterances were included
        self.assertEqual(len(conversation['utterances']), 2)
        self.assertEqual(conversation['utterances'][0]['text'], utterance1['text'])
        self.assertEqual(conversation['utterances'][1]['text'], utterance2['text'])
    
    def test_search_utterances(self):
        """Test searching utterances."""
        # Create a conversation
        conversation_id = self.db.create_conversation()
        
        # Add utterances with different text
        utterance1 = self.sample_utterance.copy()
        utterance1['text'] = "This contains the word apple."
        utterance2 = self.sample_utterance.copy()
        utterance2['text'] = "This contains the word banana."
        utterance3 = self.sample_utterance.copy()
        utterance3['text'] = "This contains both apple and banana."
        
        self.db.add_utterance(utterance1, conversation_id)
        self.db.add_utterance(utterance2, conversation_id)
        self.db.add_utterance(utterance3, conversation_id)
        
        # Search for utterances containing "apple"
        results = self.db.search_utterances("apple")
        
        # Should find two utterances
        self.assertEqual(len(results), 2)
        
        # Search for utterances containing "banana"
        results = self.db.search_utterances("banana")
        
        # Should find two utterances
        self.assertEqual(len(results), 2)
        
        # Search for utterances containing "both"
        results = self.db.search_utterances("both")
        
        # Should find one utterance
        self.assertEqual(len(results), 1)
    
    def test_get_utterances_by_speaker(self):
        """Test getting utterances by speaker."""
        # Create a conversation
        conversation_id = self.db.create_conversation()
        
        # Add utterances with different speakers
        utterance1 = self.sample_utterance.copy()
        utterance1['speaker'] = "Speaker_A"
        utterance2 = self.sample_utterance.copy()
        utterance2['speaker'] = "Speaker_B"
        utterance3 = self.sample_utterance.copy()
        utterance3['speaker'] = "Speaker_A"
        
        self.db.add_utterance(utterance1, conversation_id)
        self.db.add_utterance(utterance2, conversation_id)
        self.db.add_utterance(utterance3, conversation_id)
        
        # Get utterances by Speaker_A
        results = self.db.get_utterances_by_speaker("Speaker_A")
        
        # Should find two utterances
        self.assertEqual(len(results), 2)
        
        # Get utterances by Speaker_B
        results = self.db.get_utterances_by_speaker("Speaker_B")
        
        # Should find one utterance
        self.assertEqual(len(results), 1)
    
    def test_get_database_stats(self):
        """Test getting database statistics."""
        # Create a conversation
        conversation_id = self.db.create_conversation()
        
        # Add some utterances
        for i in range(5):
            utterance = self.sample_utterance.copy()
            utterance['text'] = f"Test utterance {i+1}"
            self.db.add_utterance(utterance, conversation_id)
        
        # Get database stats
        stats = self.db.get_database_stats()
        
        # Check the stats
        self.assertEqual(stats['conversation_count'], 1)
        self.assertEqual(stats['utterance_count'], 5)
        self.assertEqual(stats['speaker_count'], 1)
        self.assertIn('database_size_bytes', stats)

def run_tests():
    """Run the conversation database tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Conversation Database tests...")
    run_tests()
