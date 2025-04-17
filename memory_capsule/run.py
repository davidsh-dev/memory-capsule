"""
Memory Capsule - Main Application Script

This script integrates all components of the Speaker/Microphone Memory Capsule
and provides a simple way to run the application in Anaconda environments like Spyder.
"""

import os
import sys
import time
import argparse
import threading
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Import components
from memory_capsule.audio.recorder import AudioRecorder
from memory_capsule.transcription.whisper_transcriber import WhisperTranscriber
from memory_capsule.diarization.speaker_diarizer import SpeakerDiarizer
from memory_capsule.storage.conversation_db import ConversationDB
from memory_capsule.llm.language_model import LanguageModel
from memory_capsule.memory.memory_search import MemorySearch
from memory_capsule.qa.contextual_qa import ContextualQA
from memory_capsule.tts.speech_synthesizer import SpeechSynthesizer
from memory_capsule.ui.interface import UserInterface

class MemoryCapsule:
    """Main application class that integrates all components."""
    
    def __init__(self, config=None):
        """Initialize the memory capsule with all its components.
        
        Args:
            config (dict, optional): Configuration parameters. Defaults to None.
        """
        self.config = config or {}
        self.running = False
        self.paused = False
        
        # Create data directory if it doesn't exist
        data_dir = self.config.get('data_dir', os.path.join(os.path.dirname(__file__), 'data'))
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize database
        db_path = self.config.get('db_path', os.path.join(data_dir, 'conversations.db'))
        self.db = ConversationDB(db_path)
        
        # Initialize audio recorder
        self.audio_recorder = AudioRecorder(
            sample_rate=self.config.get('sample_rate', 16000),
            channels=self.config.get('channels', 1),
            chunk_size=self.config.get('chunk_size', 1024)
        )
        
        # Initialize transcriber
        self.transcriber = WhisperTranscriber(
            model_name=self.config.get('whisper_model', 'tiny'),
            device=self.config.get('device', 'cpu')
        )
        
        # Initialize diarizer
        self.diarizer = SpeakerDiarizer(
            sample_rate=self.config.get('sample_rate', 16000)
        )
        
        # Initialize language model
        self.language_model = LanguageModel(
            model_name=self.config.get('llm_model', 'gpt-3.5-turbo'),
            use_local=self.config.get('use_local_llm', False)
        )
        
        # Initialize memory search
        self.memory_search = MemorySearch(
            db=self.db,
            embedding_model=self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Initialize contextual QA
        self.qa = ContextualQA(
            language_model=self.language_model,
            memory_search=self.memory_search
        )
        
        # Initialize speech synthesizer
        self.speech_synthesizer = SpeechSynthesizer(
            voice=self.config.get('tts_voice', None),
            rate=self.config.get('tts_rate', 150)
        )
        
        # Initialize user interface
        self.ui = UserInterface(self)
        
        # Set up threads
        self.audio_thread = None
        self.processing_thread = None
        
        # Buffers and queues
        self.audio_buffer = []
        self.transcript_buffer = []
        
        # Current conversation
        self.current_conversation_id = None
        
        print("Memory Capsule initialized successfully")
    
    def start(self):
        """Start the memory capsule and all its components."""
        if self.running:
            print("Memory Capsule is already running")
            return
        
        self.running = True
        self.paused = False
        
        # Connect to the database
        self.db.connect()
        
        # Create a new conversation
        self.current_conversation_id = self.db.create_conversation(
            title=f"Conversation {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Start audio recording in a separate thread
        self.audio_thread = threading.Thread(target=self._audio_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start the UI
        self.ui.start()
        
        print("Memory Capsule started")
    
    def stop(self):
        """Stop the memory capsule and all its components."""
        if not self.running:
            print("Memory Capsule is not running")
            return
        
        self.running = False
        
        # End the current conversation
        if self.current_conversation_id:
            self.db.end_conversation(self.current_conversation_id)
            self.current_conversation_id = None
        
        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Stop components
        self.audio_recorder.stop()
        self.db.disconnect()
        
        print("Memory Capsule stopped")
    
    def pause(self):
        """Pause audio recording and processing."""
        self.paused = True
        print("Memory Capsule paused")
    
    def resume(self):
        """Resume audio recording and processing."""
        self.paused = False
        print("Memory Capsule resumed")
    
    def ask_question(self, question):
        """Ask a question to the assistant.
        
        Args:
            question (str): The question to ask.
            
        Returns:
            str: The assistant's response.
        """
        # Pause audio recording to avoid capturing the assistant's response
        self.pause()
        
        # Get answer from contextual QA
        answer = self.qa.answer_question(question)
        
        # Speak the answer
        self.speech_synthesizer.speak(answer)
        
        # Resume audio recording
        self.resume()
        
        return answer
    
    def _audio_loop(self):
        """Main audio recording loop."""
        self.audio_recorder.start()
        
        while self.running:
            if not self.paused:
                # Get audio chunk
                audio_chunk = self.audio_recorder.get_audio()
                
                if audio_chunk is not None:
                    # Add to buffer
                    self.audio_buffer.append(audio_chunk)
            
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
    
    def _processing_loop(self):
        """Main processing loop for transcription and diarization."""
        segment_duration = 5.0  # Process in 5-second segments
        samples_per_segment = int(self.config.get('sample_rate', 16000) * segment_duration)
        
        while self.running:
            if not self.paused and len(self.audio_buffer) > 0:
                # Collect enough audio for a segment
                audio_data = []
                while len(audio_data) < samples_per_segment and len(self.audio_buffer) > 0:
                    audio_data.extend(self.audio_buffer.pop(0))
                
                if len(audio_data) >= samples_per_segment:
                    # Process the segment
                    self._process_audio_segment(audio_data)
            
            time.sleep(0.1)  # Sleep to prevent CPU hogging
    
    def _process_audio_segment(self, audio_data):
        """Process an audio segment with transcription and diarization.
        
        Args:
            audio_data (list): Audio samples to process.
        """
        # Transcribe audio
        transcript = self.transcriber.transcribe(audio_data)
        
        if transcript and transcript.get('text', '').strip():
            # Perform diarization
            diarization = self.diarizer.diarize(audio_data)
            
            # Combine transcript with speaker information
            utterances = self._combine_transcript_with_diarization(transcript, diarization)
            
            # Store in database
            for utterance in utterances:
                self.db.add_utterance(utterance, self.current_conversation_id)
            
            # Add to transcript buffer
            self.transcript_buffer.extend(utterances)
            
            # Print for debugging
            for utterance in utterances:
                print(f"[{utterance['speaker']}]: {utterance['text']}")
    
    def _combine_transcript_with_diarization(self, transcript, diarization):
        """Combine transcript with speaker diarization information.
        
        Args:
            transcript (dict): Transcription result with text and timestamps.
            diarization (list): Diarization result with speaker labels and timestamps.
            
        Returns:
            list: List of utterances with speaker information.
        """
        utterances = []
        
        if not transcript or not transcript.get('segments'):
            return utterances
        
        # For each transcript segment, find the dominant speaker
        for segment in transcript.get('segments', []):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            if not text:
                continue
            
            # Find the dominant speaker for this segment
            speaker = self._get_dominant_speaker(start_time, end_time, diarization)
            
            utterance = {
                'text': text,
                'speaker': speaker,
                'start_time': start_time,
                'end_time': end_time,
                'timestamp': time.time()
            }
            
            utterances.append(utterance)
        
        return utterances
    
    def _get_dominant_speaker(self, start_time, end_time, diarization):
        """Get the dominant speaker for a time segment.
        
        Args:
            start_time (float): Start time of the segment.
            end_time (float): End time of the segment.
            diarization (list): Diarization result with speaker labels and timestamps.
            
        Returns:
            str: Speaker label.
        """
        # Default speaker if no match is found
        default_speaker = "Speaker_Unknown"
        
        if not diarization:
            return default_speaker
        
        # Find overlapping speaker segments
        overlapping_speakers = {}
        
        for speaker_segment in diarization:
            speaker = speaker_segment.get('speaker', default_speaker)
            seg_start = speaker_segment.get('start', 0)
            seg_end = speaker_segment.get('end', 0)
            
            # Check for overlap
            if max(start_time, seg_start) < min(end_time, seg_end):
                # Calculate overlap duration
                overlap = min(end_time, seg_end) - max(start_time, seg_start)
                
                if speaker in overlapping_speakers:
                    overlapping_speakers[speaker] += overlap
                else:
                    overlapping_speakers[speaker] = overlap
        
        # Return the speaker with the most overlap
        if overlapping_speakers:
            return max(overlapping_speakers, key=overlapping_speakers.get)
        
        return default_speaker

def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Memory Capsule - Speaker/Microphone Memory System')
    
    # General options
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for storing data files')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Path to database file (default: <data_dir>/conversations.db)')
    
    # Audio options
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate in Hz (default: 16000)')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of audio channels (default: 1)')
    parser.add_argument('--chunk-size', type=int, default=1024,
                        help='Audio chunk size in frames (default: 1024)')
    
    # Whisper options
    parser.add_argument('--whisper-model', type=str, default='tiny',
                        help='Whisper model name (tiny, base, small, medium, large) (default: tiny)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run models on (cpu, cuda) (default: auto-detect)')
    
    # LLM options
    parser.add_argument('--llm-model', type=str, default='gpt-3.5-turbo',
                        help='Language model name (default: gpt-3.5-turbo)')
    parser.add_argument('--use-local-llm', action='store_true',
                        help='Use local LLM via Ollama instead of OpenAI API')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2',
                        help='Embedding model name (default: all-MiniLM-L6-v2)')
    
    # TTS options
    parser.add_argument('--tts-voice', type=str, default=None,
                        help='TTS voice ID (default: system default)')
    parser.add_argument('--tts-rate', type=int, default=150,
                        help='TTS speech rate in words per minute (default: 150)')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create configuration from arguments
    config = vars(args)
    
    # Create and start the memory capsule
    memory_capsule = MemoryCapsule(config)
    
    try:
        memory_capsule.start()
    except KeyboardInterrupt:
        print("\nShutting down Memory Capsule...")
    finally:
        memory_capsule.stop()

if __name__ == "__main__":
    main()
