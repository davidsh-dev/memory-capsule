"""
Memory Search Module

This module handles searching through stored conversations using keyword and vector-based methods.
It provides functionality to find relevant past utterances for context retrieval.
"""

import numpy as np
import time
import faiss
from typing import Dict, List, Union, Optional, Any
from sentence_transformers import SentenceTransformer

class MemorySearch:
    """Class for searching through conversation memory."""
    
    def __init__(self, db, embedding_model="all-MiniLM-L6-v2", use_faiss=True):
        """Initialize the memory search module.
        
        Args:
            db: ConversationDB instance for accessing stored utterances.
            embedding_model (str): Name of the sentence transformer model for embeddings.
                Default is 'all-MiniLM-L6-v2'.
            use_faiss (bool): Whether to use FAISS for vector search. Default is True.
        """
        self.db = db
        self.embedding_model_name = embedding_model
        self.use_faiss = use_faiss
        
        self.embedding_model = None
        self.index = None
        self.utterance_ids = []  # Maps index positions to utterance IDs
        
        print(f"Initializing memory search (embedding model: {embedding_model})")
        self._load_embedding_model()
        
        if self.use_faiss:
            self._initialize_faiss_index()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for generating embeddings."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"Embedding model '{self.embedding_model_name}' loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embedding_model = None
    
    def _initialize_faiss_index(self):
        """Initialize the FAISS index for vector search."""
        if self.embedding_model is None:
            print("Embedding model not loaded, cannot initialize FAISS index")
            return
        
        try:
            # Get embedding dimension
            sample_embedding = self.embedding_model.encode("Sample text")
            dimension = sample_embedding.shape[0]
            
            # Create a flat index (exact search)
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            
            # Build index from existing utterances
            self._build_index()
            
            print(f"FAISS index initialized with dimension {dimension}")
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
            self.index = None
    
    def _build_index(self):
        """Build the FAISS index from existing utterances in the database."""
        if self.index is None or self.embedding_model is None:
            print("FAISS index or embedding model not initialized")
            return
        
        try:
            # Get all utterances from the database
            utterances = self.db.get_all_utterances()
            
            if not utterances:
                print("No utterances found in database")
                return
            
            # Extract texts and IDs
            texts = [utterance['text'] for utterance in utterances]
            self.utterance_ids = [utterance['id'] for utterance in utterances]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            print(f"Built index with {len(texts)} utterances")
        except Exception as e:
            print(f"Error building index: {e}")
    
    def add_to_index(self, utterance_id, text):
        """Add a new utterance to the FAISS index.
        
        Args:
            utterance_id (int): ID of the utterance.
            text (str): Text of the utterance.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.index is None or self.embedding_model is None:
            print("FAISS index or embedding model not initialized")
            return False
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([text])[0].reshape(1, -1)
            
            # Normalize embedding
            faiss.normalize_L2(embedding)
            
            # Add to index
            self.index.add(embedding)
            
            # Add ID to mapping
            self.utterance_ids.append(utterance_id)
            
            return True
        except Exception as e:
            print(f"Error adding to index: {e}")
            return False
    
    def search_by_keyword(self, query, limit=5):
        """Search utterances by keyword.
        
        Args:
            query (str): Search query.
            limit (int): Maximum number of results to return. Default is 5.
            
        Returns:
            list: List of matching utterances.
        """
        return self.db.search_utterances(query, limit)
    
    def search_by_vector(self, query, limit=5):
        """Search utterances by vector similarity.
        
        Args:
            query (str): Search query.
            limit (int): Maximum number of results to return. Default is 5.
            
        Returns:
            list: List of matching utterances.
        """
        if self.index is None or self.embedding_model is None:
            print("FAISS index or embedding model not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].reshape(1, -1)
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search index
            distances, indices = self.index.search(query_embedding, limit)
            
            # Get utterance IDs from indices
            result_ids = [self.utterance_ids[idx] for idx in indices[0] if idx < len(self.utterance_ids)]
            
            # Get utterances from database
            results = []
            for utterance_id in result_ids:
                utterance = self.db.get_utterance(utterance_id)
                if utterance:
                    results.append(utterance)
            
            return results
        except Exception as e:
            print(f"Error searching by vector: {e}")
            return []
    
    def search_memory(self, query, limit=5, use_keywords=True, use_vectors=True):
        """Search memory using both keyword and vector methods.
        
        Args:
            query (str): Search query.
            limit (int): Maximum number of results to return. Default is 5.
            use_keywords (bool): Whether to use keyword search. Default is True.
            use_vectors (bool): Whether to use vector search. Default is True.
            
        Returns:
            list: List of matching utterances.
        """
        results = []
        
        # Perform keyword search
        if use_keywords:
            keyword_results = self.search_by_keyword(query, limit)
            results.extend(keyword_results)
        
        # Perform vector search
        if use_vectors and self.index is not None:
            vector_results = self.search_by_vector(query, limit)
            
            # Add vector results that aren't already in the results
            for result in vector_results:
                if result not in results:
                    results.append(result)
        
        # Limit results
        return results[:limit]
    
    def rebuild_index(self):
        """Rebuild the FAISS index from scratch."""
        if self.use_faiss:
            self._initialize_faiss_index()
    
    def get_context_for_query(self, query, limit=5, window_size=2):
        """Get context for a query, including surrounding utterances.
        
        Args:
            query (str): Search query.
            limit (int): Maximum number of results to return. Default is 5.
            window_size (int): Number of utterances to include before and after each match.
                Default is 2.
            
        Returns:
            list: List of context blocks, each containing related utterances.
        """
        # Search for matching utterances
        matches = self.search_memory(query, limit)
        
        if not matches:
            return []
        
        context_blocks = []
        
        for match in matches:
            # Get conversation ID
            conversation_id = match.get('conversation_id')
            
            if conversation_id is None:
                continue
            
            # Get the full conversation
            conversation = self.db.get_conversation(conversation_id)
            
            if not conversation:
                continue
            
            # Find the index of the matching utterance in the conversation
            match_id = match.get('id')
            match_index = -1
            
            for i, utterance in enumerate(conversation.get('utterances', [])):
                if utterance.get('id') == match_id:
                    match_index = i
                    break
            
            if match_index == -1:
                continue
            
            # Extract context window
            start_index = max(0, match_index - window_size)
            end_index = min(len(conversation.get('utterances', [])), match_index + window_size + 1)
            
            context = conversation.get('utterances', [])[start_index:end_index]
            
            # Add to context blocks
            context_blocks.append({
                'conversation_id': conversation_id,
                'utterances': context,
                'match_index': match_index - start_index  # Relative index in the context
            })
        
        return context_blocks


# Example usage
if __name__ == "__main__":
    import tempfile
    from memory_capsule.storage.conversation_db import ConversationDB
    
    # Create a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db') as temp_db:
        db_path = temp_db.name
        
        # Create a database instance
        db = ConversationDB(db_path)
        
        # Connect to the database
        db.connect()
        
        # Create a conversation
        conversation_id = db.create_conversation(title="Test Conversation")
        
        # Add some utterances
        utterances = [
            {
                'speaker': 'Speaker_A',
                'text': 'Hello, how are you today?',
                'start_time': 0.0,
                'end_time': 2.0,
                'timestamp': time.time()
            },
            {
                'speaker': 'Speaker_B',
                'text': 'I am doing well, thank you! How about you?',
                'start_time': 2.5,
                'end_time': 4.5,
                'timestamp': time.time() + 1
            },
            {
                'speaker': 'Speaker_A',
                'text': 'I am good too. I was wondering if you wanted to go to the park later.',
                'start_time': 5.0,
                'end_time': 8.0,
                'timestamp': time.time() + 2
            },
            {
                'speaker': 'Speaker_B',
                'text': 'That sounds like a great idea! What time were you thinking?',
                'start_time': 8.5,
                'end_time': 10.5,
                'timestamp': time.time() + 3
            },
            {
                'speaker': 'Speaker_A',
                'text': 'How about 3pm? The weather should be nice then.',
                'start_time': 11.0,
                'end_time': 13.0,
                'timestamp': time.time() + 4
            }
        ]
        
        for utterance in utterances:
            db.add_utterance(utterance, conversation_id)
        
        # Create a memory search instance
        memory_search = MemorySearch(db)
        
        # Search by keyword
        print("\nKeyword search for 'park':")
        results = memory_search.search_by_keyword("park")
        for result in results:
            print(f"[{result['speaker']}]: {result['text']}")
        
        # Search by vector (semantic)
        print("\nVector search for 'outdoor activities':")
        results = memory_search.search_by_vector("outdoor activities")
        for result in results:
            print(f"[{result['speaker']}]: {result['text']}")
        
        # Combined search
        print("\nCombined search for 'meeting time':")
        results = memory_search.search_memory("meeting time")
        for result in results:
            print(f"[{result['speaker']}]: {result['text']}")
        
        # Get context
        print("\nContext for 'weather':")
        context_blocks = memory_search.get_context_for_query("weather")
        for i, block in enumerate(context_blocks):
            print(f"\nContext Block {i+1}:")
            for utterance in block['utterances']:
                print(f"[{utterance['speaker']}]: {utterance['text']}")
        
        # Disconnect from the database
        db.disconnect()
