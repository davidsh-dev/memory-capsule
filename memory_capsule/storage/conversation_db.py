"""
Conversation Database Module

This module handles the storage of transcribed and diarized conversations.
It provides functionality to store and retrieve utterances from a SQLite database.
"""

import sqlite3
import json
import time
import os
from typing import Dict, List, Union, Optional, Any

class ConversationDB:
    """Class for storing and retrieving conversation data in a SQLite database."""
    
    def __init__(self, db_path="conversations.db"):
        """Initialize the conversation database.
        
        Args:
            db_path (str): Path to the SQLite database file. Default is 'conversations.db'.
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        print(f"Initializing conversation database at {db_path}")
    
    def connect(self):
        """Connect to the database and create tables if they don't exist."""
        try:
            # Create directory if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Connect to the database
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Enable foreign keys
            self.cursor.execute("PRAGMA foreign_keys = ON")
            
            # Create tables if they don't exist
            self._create_tables()
            
            print(f"Connected to database at {self.db_path}")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            print("Disconnected from database")
    
    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        if not self.conn:
            print("Not connected to database")
            return
        
        # Create conversations table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time REAL NOT NULL,
            end_time REAL,
            title TEXT,
            metadata TEXT
        )
        ''')
        
        # Create utterances table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS utterances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            speaker TEXT NOT NULL,
            text TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            timestamp REAL NOT NULL,
            embedding_id INTEGER,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id),
            FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
        )
        ''')
        
        # Create embeddings table for vector search
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector BLOB NOT NULL,
            model TEXT NOT NULL
        )
        ''')
        
        # Create index on utterances.timestamp for faster queries
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_utterances_timestamp ON utterances(timestamp)
        ''')
        
        # Create index on utterances.speaker for faster queries
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_utterances_speaker ON utterances(speaker)
        ''')
        
        # Create index on utterances.conversation_id for faster queries
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_utterances_conversation_id ON utterances(conversation_id)
        ''')
        
        # Commit changes
        self.conn.commit()
    
    def create_conversation(self, title=None, metadata=None) -> int:
        """Create a new conversation.
        
        Args:
            title (str, optional): Title of the conversation. Default is None.
            metadata (dict, optional): Additional metadata. Default is None.
            
        Returns:
            int: ID of the created conversation, or -1 if an error occurred.
        """
        if not self.conn:
            print("Not connected to database")
            return -1
        
        try:
            # Convert metadata to JSON if it's a dict
            metadata_json = None
            if metadata:
                metadata_json = json.dumps(metadata)
            
            # Insert conversation
            self.cursor.execute(
                "INSERT INTO conversations (start_time, title, metadata) VALUES (?, ?, ?)",
                (time.time(), title, metadata_json)
            )
            
            # Commit changes
            self.conn.commit()
            
            # Return the ID of the created conversation
            return self.cursor.lastrowid
        except Exception as e:
            print(f"Error creating conversation: {e}")
            return -1
    
    def end_conversation(self, conversation_id: int) -> bool:
        """End a conversation by setting its end time.
        
        Args:
            conversation_id (int): ID of the conversation to end.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.conn:
            print("Not connected to database")
            return False
        
        try:
            # Update conversation end time
            self.cursor.execute(
                "UPDATE conversations SET end_time = ? WHERE id = ?",
                (time.time(), conversation_id)
            )
            
            # Commit changes
            self.conn.commit()
            
            return True
        except Exception as e:
            print(f"Error ending conversation: {e}")
            return False
    
    def add_utterance(self, utterance: Dict, conversation_id: Optional[int] = None) -> int:
        """Add an utterance to the database.
        
        Args:
            utterance (dict): Utterance data with speaker, text, start_time, end_time, and timestamp.
            conversation_id (int, optional): ID of the conversation. Default is None (auto-create).
            
        Returns:
            int: ID of the added utterance, or -1 if an error occurred.
        """
        if not self.conn:
            print("Not connected to database")
            return -1
        
        try:
            # Create a new conversation if needed
            if conversation_id is None:
                conversation_id = self.create_conversation()
                if conversation_id == -1:
                    return -1
            
            # Extract utterance data
            speaker = utterance.get('speaker', 'Unknown')
            text = utterance.get('text', '')
            start_time = utterance.get('start_time', 0.0)
            end_time = utterance.get('end_time', 0.0)
            timestamp = utterance.get('timestamp', time.time())
            
            # Insert utterance
            self.cursor.execute(
                "INSERT INTO utterances (conversation_id, speaker, text, start_time, end_time, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, speaker, text, start_time, end_time, timestamp)
            )
            
            # Commit changes
            self.conn.commit()
            
            # Return the ID of the added utterance
            return self.cursor.lastrowid
        except Exception as e:
            print(f"Error adding utterance: {e}")
            return -1
    
    def get_utterance(self, utterance_id: int) -> Optional[Dict]:
        """Get an utterance by ID.
        
        Args:
            utterance_id (int): ID of the utterance to get.
            
        Returns:
            dict or None: Utterance data, or None if not found.
        """
        if not self.conn:
            print("Not connected to database")
            return None
        
        try:
            # Query utterance
            self.cursor.execute(
                "SELECT id, conversation_id, speaker, text, start_time, end_time, timestamp FROM utterances WHERE id = ?",
                (utterance_id,)
            )
            
            # Get result
            row = self.cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'conversation_id': row[1],
                    'speaker': row[2],
                    'text': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'timestamp': row[6]
                }
            else:
                return None
        except Exception as e:
            print(f"Error getting utterance: {e}")
            return None
    
    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """Get a conversation by ID.
        
        Args:
            conversation_id (int): ID of the conversation to get.
            
        Returns:
            dict or None: Conversation data with utterances, or None if not found.
        """
        if not self.conn:
            print("Not connected to database")
            return None
        
        try:
            # Query conversation
            self.cursor.execute(
                "SELECT id, start_time, end_time, title, metadata FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            
            # Get result
            row = self.cursor.fetchone()
            
            if not row:
                return None
            
            # Parse metadata
            metadata = None
            if row[4]:
                try:
                    metadata = json.loads(row[4])
                except:
                    metadata = row[4]
            
            # Create conversation dict
            conversation = {
                'id': row[0],
                'start_time': row[1],
                'end_time': row[2],
                'title': row[3],
                'metadata': metadata,
                'utterances': []
            }
            
            # Query utterances
            self.cursor.execute(
                "SELECT id, speaker, text, start_time, end_time, timestamp FROM utterances WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
            
            # Add utterances to conversation
            for row in self.cursor.fetchall():
                utterance = {
                    'id': row[0],
                    'speaker': row[1],
                    'text': row[2],
                    'start_time': row[3],
                    'end_time': row[4],
                    'timestamp': row[5]
                }
                conversation['utterances'].append(utterance)
            
            return conversation
        except Exception as e:
            print(f"Error getting conversation: {e}")
            return None
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations.
        
        Args:
            limit (int): Maximum number of conversations to return. Default is 10.
            
        Returns:
            list: List of conversation data (without utterances).
        """
        if not self.conn:
            print("Not connected to database")
            return []
        
        try:
            # Query conversations
            self.cursor.execute(
                "SELECT id, start_time, end_time, title, metadata FROM conversations ORDER BY start_time DESC LIMIT ?",
                (limit,)
            )
            
            # Get results
            conversations = []
            for row in self.cursor.fetchall():
                # Parse metadata
                metadata = None
                if row[4]:
                    try:
                        metadata = json.loads(row[4])
                    except:
                        metadata = row[4]
                
                conversation = {
                    'id': row[0],
                    'start_time': row[1],
                    'end_time': row[2],
                    'title': row[3],
                    'metadata': metadata
                }
                conversations.append(conversation)
            
            return conversations
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []
    
    def search_utterances(self, query: str, limit: int = 10) -> List[Dict]:
        """Search utterances by text.
        
        Args:
            query (str): Search query.
            limit (int): Maximum number of results to return. Default is 10.
            
        Returns:
            list: List of matching utterances.
        """
        if not self.conn:
            print("Not connected to database")
            return []
        
        try:
            # Query utterances
            self.cursor.execute(
                "SELECT id, conversation_id, speaker, text, start_time, end_time, timestamp FROM utterances WHERE text LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", limit)
            )
            
            # Get results
            utterances = []
            for row in self.cursor.fetchall():
                utterance = {
                    'id': row[0],
                    'conversation_id': row[1],
                    'speaker': row[2],
                    'text': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'timestamp': row[6]
                }
                utterances.append(utterance)
            
            return utterances
        except Exception as e:
            print(f"Error searching utterances: {e}")
            return []
    
    def get_utterances_by_speaker(self, speaker: str, limit: int = 10) -> List[Dict]:
        """Get utterances by speaker.
        
        Args:
            speaker (str): Speaker name.
            limit (int): Maximum number of results to return. Default is 10.
            
        Returns:
            list: List of matching utterances.
        """
        if not self.conn:
            print("Not connected to database")
            return []
        
        try:
            # Query utterances
            self.cursor.execute(
                "SELECT id, conversation_id, speaker, text, start_time, end_time, timestamp FROM utterances WHERE speaker = ? ORDER BY timestamp DESC LIMIT ?",
                (speaker, limit)
            )
            
            # Get results
            utterances = []
            for row in self.cursor.fetchall():
                utterance = {
                    'id': row[0],
                    'conversation_id': row[1],
                    'speaker': row[2],
                    'text': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'timestamp': row[6]
                }
                utterances.append(utterance)
            
            return utterances
        except Exception as e:
            print(f"Error getting utterances by speaker: {e}")
            return []
    
    def get_utterances_by_timerange(self, start_time: float, end_time: float) -> List[Dict]:
        """Get utterances within a time range.
        
        Args:
            start_time (float): Start timestamp.
            end_time (float): End timestamp.
            
        Returns:
            list: List of matching utterances.
        """
        if not self.conn:
            print("Not connected to database")
            return []
        
        try:
            # Query utterances
            self.cursor.execute(
                "SELECT id, conversation_id, speaker, text, start_time, end_time, timestamp FROM utterances WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                (start_time, end_time)
            )
            
            # Get results
            utterances = []
            for row in self.cursor.fetchall():
                utterance = {
                    'id': row[0],
                    'conversation_id': row[1],
                    'speaker': row[2],
                    'text': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'timestamp': row[6]
                }
                utterances.append(utterance)
            
            return utterances
        except Exception as e:
            print(f"Error getting utterances by time range: {e}")
            return []
    
    def store_embedding(self, vector: bytes, model: str) -> int:
        """Store an embedding vector in the database.
        
        Args:
            vector (bytes): Embedding vector as bytes.
            model (str): Name of the embedding model.
            
        Returns:
            int: ID of the stored embedding, or -1 if an error occurred.
        """
        if not self.conn:
            print("Not connected to database")
            return -1
        
        try:
            # Insert embedding
            self.cursor.execute(
                "INSERT INTO embeddings (vector, model) VALUES (?, ?)",
                (vector, model)
            )
            
            # Commit changes
            self.conn.commit()
            
            # Return the ID of the stored embedding
            return self.cursor.lastrowid
        except Exception as e:
            print(f"Error storing embedding: {e}")
            return -1
    
    def link_embedding_to_utterance(self, utterance_id: int, embedding_id: int) -> bool:
        """Link an embedding to an utterance.
        
        Args:
            utterance_id (int): ID of the utterance.
            embedding_id (int): ID of the embedding.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.conn:
            print("Not connected to database")
            return False
        
        try:
            # Update utterance
            self.cursor.execute(
                "UPDATE utterances SET embedding_id = ? WHERE id = ?",
                (embedding_id, utterance_id)
            )
            
            # Commit changes
            self.conn.commit()
            
            return True
        except Exception as e:
            print(f"Error linking embedding to utterance: {e}")
            return False
    
    def get_all_utterances(self) -> List[Dict]:
        """Get all utterances from the database.
        
        Returns:
            list: List of all utterances.
        """
        if not self.conn:
            print("Not connected to database")
            return []
        
        try:
            # Query utterances
            self.cursor.execute(
                "SELECT id, conversation_id, speaker, text, start_time, end_time, timestamp FROM utterances ORDER BY timestamp"
            )
            
            # Get results
            utterances = []
            for row in self.cursor.fetchall():
                utterance = {
                    'id': row[0],
                    'conversation_id': row[1],
                    'speaker': row[2],
                    'text': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'timestamp': row[6]
                }
                utterances.append(utterance)
            
            return utterances
        except Exception as e:
            print(f"Error getting all utterances: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database.
        
        Returns:
            dict: Database statistics.
        """
        if not self.conn:
            print("Not connected to database")
            return {}
        
        try:
            stats = {}
            
            # Count conversations
            self.cursor.execute("SELECT COUNT(*) FROM conversations")
            stats['conversation_count'] = self.cursor.fetchone()[0]
            
            # Count utterances
            self.cursor.execute("SELECT COUNT(*) FROM utterances")
            stats['utterance_count'] = self.cursor.fetchone()[0]
            
            # Count embeddings
            self.cursor.execute("SELECT COUNT(*) FROM embeddings")
            stats['embedding_count'] = self.cursor.fetchone()[0]
            
            # Count speakers
            self.cursor.execute("SELECT COUNT(DISTINCT speaker) FROM utterances")
            stats['speaker_count'] = self.cursor.fetchone()[0]
            
            # Get database size
            if os.path.exists(self.db_path):
                stats['database_size_bytes'] = os.path.getsize(self.db_path)
            
            return stats
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    import tempfile
    
    # Create a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db') as temp_db:
        db_path = temp_db.name
        
        # Create a database instance
        db = ConversationDB(db_path)
        
        # Connect to the database
        db.connect()
        
        # Create a conversation
        conversation_id = db.create_conversation(title="Test Conversation")
        print(f"Created conversation with ID: {conversation_id}")
        
        # Add some utterances
        utterance1 = {
            'speaker': 'Speaker_A',
            'text': 'Hello, how are you?',
            'start_time': 0.0,
            'end_time': 2.0,
            'timestamp': time.time()
        }
        
        utterance2 = {
            'speaker': 'Speaker_B',
            'text': 'I am doing well, thank you!',
            'start_time': 2.5,
            'end_time': 4.5,
            'timestamp': time.time() + 1
        }
        
        utterance_id1 = db.add_utterance(utterance1, conversation_id)
        utterance_id2 = db.add_utterance(utterance2, conversation_id)
        
        print(f"Added utterances with IDs: {utterance_id1}, {utterance_id2}")
        
        # Get the conversation
        conversation = db.get_conversation(conversation_id)
        print("\nRetrieved conversation:")
        print(f"ID: {conversation['id']}")
        print(f"Title: {conversation['title']}")
        print(f"Start time: {conversation['start_time']}")
        print(f"End time: {conversation['end_time']}")
        print(f"Utterances: {len(conversation['utterances'])}")
        
        # Print utterances
        print("\nUtterances:")
        for utterance in conversation['utterances']:
            print(f"[{utterance['speaker']}]: {utterance['text']}")
        
        # Search for utterances
        results = db.search_utterances("hello")
        print(f"\nSearch results for 'hello': {len(results)}")
        for utterance in results:
            print(f"[{utterance['speaker']}]: {utterance['text']}")
        
        # Get database stats
        stats = db.get_database_stats()
        print("\nDatabase stats:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Disconnect from the database
        db.disconnect()
