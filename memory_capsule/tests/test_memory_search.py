"""
Test module for the Memory Search component.

This module provides tests for the MemorySearch class to verify
that memory search functionality works correctly.
"""

import unittest
import os
import sys
import time
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_capsule.memory.memory_search import MemorySearch
from memory_capsule.storage.conversation_db import ConversationDB

class TestMemorySearch(unittest.TestCase):
    """Test cases for the MemorySearch class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_file.name
        self.temp_file.close()
        
        # Initialize the database
        self.db = ConversationDB(self.db_path)
        self.db.connect()
        
        # Create a conversation with utterances
        self.conversation_id = self.db.create_conversation(title="Test Conversation")
        
        # Add some test utterances
        self.utterances = [
            {
                'speaker': 'Speaker_A',
                'text': 'I love to go hiking in the mountains.',
                'start_time': 0.0,
                'end_time': 2.0,
                'timestamp': time.time()
            },
            {
                'speaker': 'Speaker_B',
                'text': 'The weather is perfect for outdoor activities today.',
                'start_time': 2.5,
                'end_time': 4.5,
                'timestamp': time.time() + 1
            },
            {
                'speaker': 'Speaker_A',
                'text': 'I prefer swimming in the lake when it\'s hot.',
                'start_time': 5.0,
                'end_time': 7.0,
                'timestamp': time.time() + 2
            },
            {
                'speaker': 'Speaker_B',
                'text': 'We could have a picnic in the park this weekend.',
                'start_time': 7.5,
                'end_time': 9.5,
                'timestamp': time.time() + 3
            }
        ]
        
        for utterance in self.utterances:
            self.db.add_utterance(utterance, self.conversation_id)
        
        # Initialize memory search with mocked embedding model
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            # Mock the encode method to return fixed embeddings
            mock_model = MagicMock()
            mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts) if isinstance(texts, list) else 1, 384)
            mock_transformer.return_value = mock_model
            
            self.memory_search = MemorySearch(self.db, use_faiss=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Disconnect from the database
        self.db.disconnect()
        
        # Remove the temporary database file
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_initialization(self):
        """Test that the memory search initializes correctly."""
        self.assertEqual(self.memory_search.db, self.db)
        self.assertEqual(self.memory_search.embedding_model_name, "all-MiniLM-L6-v2")
        self.assertTrue(self.memory_search.use_faiss)
    
    def test_search_by_keyword(self):
        """Test searching by keyword."""
        # Search for "hiking"
        results = self.memory_search.search_by_keyword("hiking")
        
        # Should find one utterance
        self.assertEqual(len(results), 1)
        self.assertIn("hiking", results[0]['text'])
        
        # Search for "outdoor"
        results = self.memory_search.search_by_keyword("outdoor")
        
        # Should find one utterance
        self.assertEqual(len(results), 1)
        self.assertIn("outdoor", results[0]['text'])
    
    @patch('faiss.normalize_L2')
    @patch('faiss.IndexFlatIP')
    def test_search_by_vector(self, mock_index, mock_normalize):
        """Test searching by vector similarity."""
        # Mock the FAISS index search method
        mock_index_instance = mock_index.return_value
        mock_index_instance.search.return_value = (
            np.array([[0.9, 0.8]]),  # Distances
            np.array([[0, 1]])       # Indices
        )
        
        # Replace the memory search index with our mock
        self.memory_search.index = mock_index_instance
        
        # Set up utterance IDs mapping
        self.memory_search.utterance_ids = [1, 2]
        
        # Mock the get_utterance method to return known utterances
        with patch.object(self.db, 'get_utterance') as mock_get_utterance:
            mock_get_utterance.side_effect = lambda id: {'id': id, 'text': f'Utterance {id}'}
            
            # Search for "mountain activities"
            results = self.memory_search.search_by_vector("mountain activities")
            
            # Should find two utterances (based on our mock)
            self.assertEqual(len(results), 2)
            
            # Check that the search was called correctly
            mock_index_instance.search.assert_called_once()
            
            # Check that normalize was called
            mock_normalize.assert_called()
    
    def test_search_memory(self):
        """Test combined memory search."""
        # Mock search_by_keyword and search_by_vector
        with patch.object(self.memory_search, 'search_by_keyword') as mock_keyword, \
             patch.object(self.memory_search, 'search_by_vector') as mock_vector:
            
            # Set up mock returns
            mock_keyword.return_value = [{'id': 1, 'text': 'Keyword result'}]
            mock_vector.return_value = [{'id': 2, 'text': 'Vector result'}]
            
            # Search memory
            results = self.memory_search.search_memory("test query")
            
            # Should find two utterances (one from each method)
            self.assertEqual(len(results), 2)
            
            # Check that both search methods were called
            mock_keyword.assert_called_once_with("test query", 5)
            mock_vector.assert_called_once_with("test query", 5)
    
    def test_get_context_for_query(self):
        """Test getting context for a query."""
        # Mock search_memory to return a known utterance
        with patch.object(self.memory_search, 'search_memory') as mock_search:
            # Create a mock utterance result
            mock_utterance = {
                'id': 2,
                'conversation_id': self.conversation_id,
                'text': 'Mock utterance'
            }
            mock_search.return_value = [mock_utterance]
            
            # Mock get_conversation to return a conversation with utterances
            with patch.object(self.db, 'get_conversation') as mock_get_conversation:
                mock_conversation = {
                    'id': self.conversation_id,
                    'utterances': [
                        {'id': 1, 'text': 'Before'},
                        {'id': 2, 'text': 'Mock utterance'},
                        {'id': 3, 'text': 'After'}
                    ]
                }
                mock_get_conversation.return_value = mock_conversation
                
                # Get context
                context_blocks = self.memory_search.get_context_for_query("test query")
                
                # Should find one context block
                self.assertEqual(len(context_blocks), 1)
                
                # The context block should contain all three utterances
                self.assertEqual(len(context_blocks[0]['utterances']), 3)
                
                # The match index should be 1 (second utterance)
                self.assertEqual(context_blocks[0]['match_index'], 1)

def run_tests():
    """Run the memory search tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Memory Search tests...")
    run_tests()
