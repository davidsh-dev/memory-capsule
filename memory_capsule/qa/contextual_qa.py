"""
Contextual QA Module

This module handles contextual question answering using the language model
and memory search components to provide context-aware responses.
"""

import time
from typing import Dict, List, Union, Optional, Any

class ContextualQA:
    """Class for contextual question answering using RAG (Retrieval-Augmented Generation)."""
    
    def __init__(self, language_model, memory_search, max_context_items=5, context_window=2):
        """Initialize the contextual QA module.
        
        Args:
            language_model: LanguageModel instance for generating responses.
            memory_search: MemorySearch instance for retrieving context.
            max_context_items (int): Maximum number of context items to include. Default is 5.
            context_window (int): Number of utterances to include before and after each match.
                Default is 2.
        """
        self.language_model = language_model
        self.memory_search = memory_search
        self.max_context_items = max_context_items
        self.context_window = context_window
        
        print("Initializing contextual QA module")
    
    def answer_question(self, question: str, temperature=0.7) -> str:
        """Answer a question using context from memory.
        
        Args:
            question (str): The question to answer.
            temperature (float): Sampling temperature for the language model. Default is 0.7.
            
        Returns:
            str: The answer to the question.
        """
        # Get context from memory
        context_blocks = self.memory_search.get_context_for_query(
            question, 
            limit=self.max_context_items,
            window_size=self.context_window
        )
        
        # Format context for the prompt
        context_text = self._format_context(context_blocks)
        
        # Create the prompt
        prompt = self._create_prompt(question, context_text)
        
        # Generate the answer
        answer = self.language_model.generate_response(prompt, temperature=temperature)
        
        return answer
    
    def _format_context(self, context_blocks: List[Dict]) -> str:
        """Format context blocks into a string for the prompt.
        
        Args:
            context_blocks (list): List of context blocks from memory search.
            
        Returns:
            str: Formatted context string.
        """
        if not context_blocks:
            return "No relevant context found."
        
        context_parts = []
        
        for i, block in enumerate(context_blocks):
            context_parts.append(f"Context Block {i+1}:")
            
            for utterance in block.get('utterances', []):
                speaker = utterance.get('speaker', 'Unknown')
                text = utterance.get('text', '')
                
                # Format timestamp if available
                timestamp_str = ""
                if 'timestamp' in utterance:
                    timestamp = utterance.get('timestamp')
                    timestamp_str = f" ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))})"
                
                context_parts.append(f"{speaker}{timestamp_str}: {text}")
            
            context_parts.append("")  # Add a blank line between blocks
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the language model.
        
        Args:
            question (str): The question to answer.
            context (str): The context string.
            
        Returns:
            str: The prompt for the language model.
        """
        return f"""You are an assistant with access to the following conversation history:

{context}

Using this information, please answer the following question. If the answer cannot be determined from the provided context, say so and provide a general response based on your knowledge.

Question: {question}

Answer:"""
    
    def chat(self, messages: List[Dict[str, str]], temperature=0.7) -> str:
        """Generate a response in a chat conversation with context from memory.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'.
            temperature (float): Sampling temperature for the language model. Default is 0.7.
            
        Returns:
            str: Generated response text.
        """
        # Get the last user message
        last_user_message = None
        for message in reversed(messages):
            if message.get('role') == 'user':
                last_user_message = message.get('content')
                break
        
        if not last_user_message:
            return "I couldn't find a user message to respond to."
        
        # Get context from memory
        context_blocks = self.memory_search.get_context_for_query(
            last_user_message, 
            limit=self.max_context_items,
            window_size=self.context_window
        )
        
        # Format context for the prompt
        context_text = self._format_context(context_blocks)
        
        # Create a system message with context
        system_message = {
            "role": "system",
            "content": f"You are a helpful assistant with access to the following conversation history:\n\n{context_text}\n\nUse this information to provide context-aware responses."
        }
        
        # Add the system message to the beginning of the messages
        augmented_messages = [system_message] + messages
        
        # Generate the response
        response = self.language_model.chat(augmented_messages, temperature=temperature)
        
        return response


# Example usage
if __name__ == "__main__":
    import tempfile
    from memory_capsule.storage.conversation_db import ConversationDB
    from memory_capsule.memory.memory_search import MemorySearch
    from memory_capsule.llm.language_model import LanguageModel
    
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
                'text': 'I think we should meet at the coffee shop at 3pm tomorrow.',
                'start_time': 0.0,
                'end_time': 2.0,
                'timestamp': time.time()
            },
            {
                'speaker': 'Speaker_B',
                'text': 'That works for me. Should we bring our laptops?',
                'start_time': 2.5,
                'end_time': 4.5,
                'timestamp': time.time() + 1
            },
            {
                'speaker': 'Speaker_A',
                'text': 'Yes, we need to work on the presentation for the client.',
                'start_time': 5.0,
                'end_time': 7.0,
                'timestamp': time.time() + 2
            },
            {
                'speaker': 'Speaker_B',
                'text': 'Great. I\'ll also bring the documents we discussed last week.',
                'start_time': 7.5,
                'end_time': 9.5,
                'timestamp': time.time() + 3
            }
        ]
        
        for utterance in utterances:
            db.add_utterance(utterance, conversation_id)
        
        # Create a memory search instance
        memory_search = MemorySearch(db)
        
        # Check if we have an OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key:
            # Create a language model
            language_model = LanguageModel(model_name="gpt-3.5-turbo", use_local=False)
            
            # Create a contextual QA instance
            qa = ContextualQA(language_model, memory_search)
            
            # Ask a question
            question = "What time is the meeting tomorrow?"
            print(f"Question: {question}")
            
            answer = qa.answer_question(question)
            print(f"Answer: {answer}")
            
            # Try a chat conversation
            messages = [
                {"role": "user", "content": "Can you remind me what we're supposed to bring to the meeting?"}
            ]
            
            response = qa.chat(messages)
            print(f"\nChat response: {response}")
        else:
            print("OpenAI API key not found. Skipping language model tests.")
        
        # Disconnect from the database
        db.disconnect()
