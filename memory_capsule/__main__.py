"""
Memory Capsule - Main Application

This is the main entry point for the Speaker/Microphone Memory Capsule application.
It orchestrates all components and provides the interactive assistant functionality.
"""

import os
import sys
import time
import threading
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        
        # Initialize components
        self.audio_recorder = AudioRecorder(
            sample_rate=self.config.get('sample_rate', 16000),
            channels=self.config.get('channels', 1),
            chunk_size=self.config.get('chunk_size', 1024)
        )
        
        self.transcriber = WhisperTranscriber(
            model_name=self.config.get('whisper_model', 'tiny'),
            device=self.config.get('device', 'cpu')
        )
        
        self.diarizer = SpeakerDiarizer(
            sample_rate=self.config.get('sample_rate', 16000)
        )
        
        self.db = ConversationDB(
            db_path=self.config.get('db_path', 'conversations.db')
        )
        
        self.language_model = LanguageModel(
            model_name=self.config.get('llm_model', 'gpt-3.5-turbo'),
            use_local=self.config.get('use_local_llm', False)
        )
        
        self.memory_search = MemorySearch(
            db=self.db,
            embedding_model=self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        self.qa = ContextualQA(
            language_model=self.language_model,
            memory_search=self.memory_search
        )
        
        self.speech_synthesizer = SpeechSynthesizer(
            voice=self.config.get('tts_voice', None),
            rate=self.config.get('tts_rate', 150)
        )
        
        self.ui = UserInterface(self)
        
        # Set up threads
        self.audio_thread = None
        self.processing_thread = None
        
        # Buffers and queues
        self.audio_buffer = []
        self.transcript_buffer = []
        
        print("Memory Capsule initialized successfully")
    
    def start(self):
        """Start the memory capsule and all its components."""
        if self.running:
            print("Memory Capsule is already running")
            return
        
        self.running = True
        self.paused = False
        
        # Start the database
        self.db.connect()
        
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
        
        if transcript and transcript.strip():
            # Perform diarization
            diarization = self.diarizer.diarize(audio_data)
            
            # Combine transcript with speaker information
            utterances = self._combine_transcript_with_diarization(transcript, diarization)
            
            # Store in database
            for utterance in utterances:
                self.db.add_utterance(utterance)
            
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
        # This is a simplified implementation
        # In a real system, this would align transcript segments with speaker segments
        
        utterances = []
        
        if not transcript or not diarization:
            return utterances
        
        # For simplicity, we'll assume one speaker per transcript segment
        # In a real implementation, this would be more sophisticated
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
        # This is a simplified implementation
        # In a real system, this would calculate speaker overlap and determine the dominant speaker
        
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

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Memory Capsule - Speaker/Microphone Memory System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--db', type=str, default='conversations.db', help='Path to database file')
    parser.add_argument('--whisper-model', type=str, default='tiny', help='Whisper model name (tiny, base, small, medium, large)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--use-local-llm', action='store_true', help='Use local LLM instead of API')
    parser.add_argument('--llm-model', type=str, default='gpt-3.5-turbo', help='LLM model name')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = {
        'db_path': args.db,
        'whisper_model': args.whisper_model,
        'sample_rate': args.sample_rate,
        'use_local_llm': args.use_local_llm,
        'llm_model': args.llm_model
    }
    
    # Create and start the memory capsule
    memory_capsule = MemoryCapsule(config)
    
    try:
        memory_capsule.start()
        
        # Keep the main thread alive
        while memory_capsule.running:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nShutting down Memory Capsule...")
    finally:
        memory_capsule.stop()

if __name__ == "__main__":
    main()
