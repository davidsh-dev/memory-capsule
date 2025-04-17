"""
Simplified test module for the Memory Search component with import mocking.

This module provides tests for the MemorySearch class with mocked dependencies.
"""

import unittest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Mock all required dependencies before importing MemorySearch
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the class to test
from memory_capsule.memory.memory_search import MemorySearch

class TestMemorySearchSimplified(unittest.TestCase):
    """Test cases for the MemorySearch class with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patchers for sentence_transformers and faiss
        self.st_patcher = patch('memory_capsule.memory.memory_search.SentenceTransformer')
        self.faiss_patcher = patch('memory_capsule.memory.memory_search.faiss')
        
        # Start the patchers
        self.mock_st = self.st_patcher.start()
        self.mock_faiss = self.faiss_patcher.start()
        
        # Mock the sentence transformer model
        self.mock_model = MagicMock()
        self.mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts) if isinstance(texts, list) else 1, 384)
        self.mock_st.return_value = self.mock_model
        
        # Mock the FAISS index
        self.mock_index = MagicMock()
        self.mock_index.search.return_value = (
            np.array([[0.9, 0.8]]),  # Distances
            np.array([[0, 1]])       # Indices
        )
        self.mock_faiss.IndexFlatIP.return_value = self.mock_index
        
        # Mock the database
        self.mock_db = MagicMock()
        self.mock_db.get_utterance.side_effect = lambda id: {'id': id, 'text': f'Utterance {id}', 'conversation_id': 1}
        self.mock_db.get_conversation.return_value = {
            'id': 1,
            'utterances': [
                {'id': 0, 'text': 'Utterance 0'},
                {'id': 1, 'text': 'Utterance 1'},
                {'id': 2, 'text': 'Utterance 2'}
            ]
        }
        self.mock_db.search_utterances.return_value = [
            {'id': 0, 'text': 'Keyword result 1', 'conversation_id': 1},
            {'id': 2, 'text': 'Keyword result 2', 'conversation_id': 1}
        ]
        
        # Initialize the memory search
        self.memory_search = MemorySearch(self.mock_db, use_faiss=True)
        
        # Set up utterance IDs mapping for testing
        self.memory_search.utterance_ids = [0, 1, 2]
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patchers
        self.st_patcher.stop()
        self.faiss_patcher.stop()
    
    def test_initialization(self):
        """Test that the memory search initializes correctly."""
        self.assertEqual(self.memory_search.db, self.mock_db)
        self.assertEqual(self.memory_search.embedding_model_name, "all-MiniLM-L6-v2")
        self.assertTrue(self.memory_search.use_faiss)
        self.assertIsNotNone(self.memory_search.embedding_model)
    
    def test_search_by_keyword(self):
        """Test searching by keyword."""
        # Search for "test"
        results = self.memory_search.search_by_keyword("test")
        
        # Check that the database search_utterances method was called
        self.mock_db.search_utterances.assert_called_once_with("test", 5)
        
        # Check that the results are correct
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['text'], 'Keyword result 1')
        self.assertEqual(results[1]['text'], 'Keyword result 2')
    
    def test_search_by_vector(self):
        """Test searching by vector similarity."""
        # Search for "test query"
        results = self.memory_search.search_by_vector("test query")
        
        # Check that the embedding model encode method was called
        self.mock_model.encode.assert_called_once()
        
        # Check that the FAISS index search method was called
        self.mock_index.search.assert_called_once()
        
        # Check that the results are correct
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['text'], 'Utterance 0')
        self.assertEqual(results[1]['text'], 'Utterance 1')
    
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
            
            # Check that both search methods were called
            mock_keyword.assert_called_once_with("test query", 5)
            mock_vector.assert_called_once_with("test query", 5)
            
            # Check that the results are correct
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]['text'], 'Keyword result')
            self.assertEqual(results[1]['text'], 'Vector result')
    
    def test_get_context_for_query(self):
        """Test getting context for a query."""
        # Mock search_memory to return a known utterance
        with patch.object(self.memory_search, 'search_memory') as mock_search:
            # Create a mock utterance result
            mock_utterance = {
                'id': 1,
                'conversation_id': 1,
                'text': 'Mock utterance'
            }
            mock_search.return_value = [mock_utterance]
            
            # Get context
            context_blocks = self.memory_search.get_context_for_query("test query")
            
            # Check that search_memory was called
            mock_search.assert_called_once_with("test query", 5)
            
            # Check that the context blocks are correct
            self.assertEqual(len(context_blocks), 1)
            self.assertEqual(len(context_blocks[0]['utterances']), 3)
            self.assertEqual(context_blocks[0]['match_index'], 1)

def run_tests():
    """Run the simplified memory search tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    print("Running Simplified Memory Search tests...")
    run_tests()
