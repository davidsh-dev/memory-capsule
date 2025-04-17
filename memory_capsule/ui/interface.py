"""
User Interface Module

This module provides the user interface for the memory capsule.
It handles user interaction through a command-line interface.
"""

import os
import sys
import time
import threading
import cmd
import argparse
from typing import Dict, List, Union, Optional, Any

class UserInterface(cmd.Cmd):
    """Command-line interface for the memory capsule."""
    
    intro = "Memory Capsule - Speaker/Microphone Memory System\nType 'help' for a list of commands."
    prompt = "memory> "
    
    def __init__(self, memory_capsule=None):
        """Initialize the user interface.
        
        Args:
            memory_capsule: MemoryCapsule instance to control.
        """
        super().__init__()
        self.memory_capsule = memory_capsule
        self.running = False
        self.display_thread = None
        self.last_utterances = []
        
        print("Initializing user interface")
    
    def start(self):
        """Start the user interface."""
        self.running = True
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        # Start command loop
        try:
            self.cmdloop()
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.running = False
            if self.memory_capsule:
                self.memory_capsule.stop()
    
    def _display_loop(self):
        """Display loop for showing new utterances."""
        while self.running:
            if self.memory_capsule:
                # Get new utterances from the transcript buffer
                new_utterances = self._get_new_utterances()
                
                # Display new utterances
                for utterance in new_utterances:
                    speaker = utterance.get('speaker', 'Unknown')
                    text = utterance.get('text', '')
                    print(f"\n[{speaker}]: {text}")
                    print(self.prompt, end='', flush=True)
            
            # Sleep to prevent CPU hogging
            time.sleep(0.5)
    
    def _get_new_utterances(self):
        """Get new utterances from the memory capsule.
        
        Returns:
            list: List of new utterances.
        """
        if not self.memory_capsule:
            return []
        
        # Get all utterances from the transcript buffer
        all_utterances = self.memory_capsule.transcript_buffer.copy()
        
        # Find new utterances
        if not self.last_utterances:
            new_utterances = all_utterances
        else:
            # Find the index of the last known utterance
            last_index = -1
            for i, utterance in enumerate(all_utterances):
                if i < len(self.last_utterances) and utterance == self.last_utterances[i]:
                    last_index = i
                else:
                    break
            
            # Get new utterances
            new_utterances = all_utterances[last_index + 1:]
        
        # Update last utterances
        self.last_utterances = all_utterances
        
        return new_utterances
    
    def do_start(self, arg):
        """Start the memory capsule.
        
        Usage: start
        """
        if not self.memory_capsule:
            print("Memory capsule not initialized")
            return
        
        self.memory_capsule.start()
        print("Memory capsule started")
    
    def do_stop(self, arg):
        """Stop the memory capsule.
        
        Usage: stop
        """
        if not self.memory_capsule:
            print("Memory capsule not initialized")
            return
        
        self.memory_capsule.stop()
        print("Memory capsule stopped")
    
    def do_pause(self, arg):
        """Pause audio recording and processing.
        
        Usage: pause
        """
        if not self.memory_capsule:
            print("Memory capsule not initialized")
            return
        
        self.memory_capsule.pause()
        print("Memory capsule paused")
    
    def do_resume(self, arg):
        """Resume audio recording and processing.
        
        Usage: resume
        """
        if not self.memory_capsule:
            print("Memory capsule not initialized")
            return
        
        self.memory_capsule.resume()
        print("Memory capsule resumed")
    
    def do_ask(self, arg):
        """Ask a question to the assistant.
        
        Usage: ask <question>
        """
        if not self.memory_capsule:
            print("Memory capsule not initialized")
            return
        
        if not arg:
            print("Please provide a question")
            return
        
        print(f"Question: {arg}")
        answer = self.memory_capsule.ask_question(arg)
        print(f"Answer: {answer}")
    
    def do_search(self, arg):
        """Search memory for utterances.
        
        Usage: search <query>
        """
        if not self.memory_capsule:
            print("Memory capsule not initialized")
            return
        
        if not arg:
            print("Please provide a search query")
            return
        
        print(f"Searching for: {arg}")
        results = self.memory_capsule.memory_search.search_memory(arg)
        
        if not results:
            print("No results found")
            return
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            speaker = result.get('speaker', 'Unknown')
            text = result.get('text', '')
            print(f"{i+1}. [{speaker}]: {text}")
    
    def do_list(self, arg):
        """List recent conversations.
        
        Usage: list [count]
        """
        if not self.memory_capsule:
            print("Memory capsule not initialized")
            return
        
        # Parse count argument
        count = 5
        if arg:
            try:
                count = int(arg)
            except ValueError:
                print(f"Invalid count: {arg}")
                return
        
        # Get recent conversations
        conversations = self.memory_capsule.db.get_recent_conversations(count)
        
        if not conversations:
            print("No conversations found")
            return
        
        print(f"Recent conversations ({len(conversations)}):")
        for i, conversation in enumerate(conversations):
            title = conversation.get('title') or f"Conversation {conversation.get('id')}"
            start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(conversation.get('start_time')))
            print(f"{i+1}. {title} (started: {start_time})")
    
    def do_view(self, arg):
        """View a conversation.
        
        Usage: view <conversation_id>
        """
        if not self.memory_capsule:
            print("Memory capsule not initialized")
            return
        
        if not arg:
            print("Please provide a conversation ID")
            return
        
        try:
            conversation_id = int(arg)
        except ValueError:
            print(f"Invalid conversation ID: {arg}")
            return
        
        # Get the conversation
        conversation = self.memory_capsule.db.get_conversation(conversation_id)
        
        if not conversation:
            print(f"Conversation {conversation_id} not found")
            return
        
        title = conversation.get('title') or f"Conversation {conversation.get('id')}"
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(conversation.get('start_time')))
        
        print(f"Conversation: {title}")
        print(f"Started: {start_time}")
        
        utterances = conversation.get('utterances', [])
        if not utterances:
            print("No utterances found")
            return
        
        print(f"Utterances ({len(utterances)}):")
        for utterance in utterances:
            speaker = utterance.get('speaker', 'Unknown')
            text = utterance.get('text', '')
            print(f"[{speaker}]: {text}")
    
    def do_stats(self, arg):
        """Show database statistics.
        
        Usage: stats
        """
        if not self.memory_capsule:
            print("Memory capsule not initialized")
            return
        
        # Get database stats
        stats = self.memory_capsule.db.get_database_stats()
        
        if not stats:
            print("No statistics available")
            return
        
        print("Database statistics:")
        for key, value in stats.items():
            if key == 'database_size_bytes':
                # Convert bytes to human-readable format
                size_mb = value / (1024 * 1024)
                print(f"Database size: {size_mb:.2f} MB")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    def do_exit(self, arg):
        """Exit the program.
        
        Usage: exit
        """
        print("Exiting...")
        self.running = False
        if self.memory_capsule:
            self.memory_capsule.stop()
        return True
    
    def do_quit(self, arg):
        """Exit the program.
        
        Usage: quit
        """
        return self.do_exit(arg)
    
    def do_help(self, arg):
        """Show help message.
        
        Usage: help [command]
        """
        if arg:
            # Show help for a specific command
            super().do_help(arg)
        else:
            # Show general help
            print("\nMemory Capsule - Available Commands:")
            print("  start       - Start the memory capsule")
            print("  stop        - Stop the memory capsule")
            print("  pause       - Pause audio recording and processing")
            print("  resume      - Resume audio recording and processing")
            print("  ask <text>  - Ask a question to the assistant")
            print("  search <q>  - Search memory for utterances")
            print("  list [n]    - List recent conversations (default: 5)")
            print("  view <id>   - View a conversation by ID")
            print("  stats       - Show database statistics")
            print("  help [cmd]  - Show help message for a command")
            print("  exit/quit   - Exit the program")
            print("\nThe memory capsule continuously records and transcribes audio.")
            print("New utterances will be displayed automatically.")


# Example usage
if __name__ == "__main__":
    # Create a simple mock memory capsule for testing
    class MockMemoryCapsule:
        def __init__(self):
            self.running = False
            self.paused = False
            self.transcript_buffer = []
        
        def start(self):
            self.running = True
            print("Mock memory capsule started")
            
            # Add some test utterances
            self.transcript_buffer.append({
                'speaker': 'Speaker_A',
                'text': 'Hello, this is a test utterance.',
                'timestamp': time.time()
            })
        
        def stop(self):
            self.running = False
            print("Mock memory capsule stopped")
        
        def pause(self):
            self.paused = True
            print("Mock memory capsule paused")
        
        def resume(self):
            self.paused = False
            print("Mock memory capsule resumed")
        
        def ask_question(self, question):
            return f"This is a mock answer to: {question}"
    
    # Create a mock memory capsule
    mock_capsule = MockMemoryCapsule()
    
    # Create a user interface
    ui = UserInterface(mock_capsule)
    
    # Start the user interface
    ui.start()
