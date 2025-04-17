"""
Whisper Transcriber Module

This module handles speech transcription using OpenAI's Whisper model.
It provides functionality to transcribe audio segments into text with timestamps.
"""

import numpy as np
import torch
import whisper
import time
from typing import Dict, List, Union, Optional

class WhisperTranscriber:
    """Class for transcribing speech using OpenAI's Whisper model."""
    
    def __init__(self, model_name="tiny", device=None, language=None):
        """Initialize the Whisper transcriber.
        
        Args:
            model_name (str): Whisper model name ('tiny', 'base', 'small', 'medium', 'large').
                Default is 'tiny'.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                Default is None (auto-detect).
            language (str, optional): Language code for transcription. Default is None (auto-detect).
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.language = language
        self.model = None
        
        print(f"Initializing Whisper transcriber (model: {model_name}, device: {self.device})")
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"Whisper model '{self.model_name}' loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe(self, audio_data: np.ndarray) -> Dict:
        """Transcribe audio data to text.
        
        Args:
            audio_data (ndarray): Audio data as numpy array.
            
        Returns:
            dict: Transcription result with text and timestamps.
        """
        if self.model is None:
            print("Whisper model not loaded")
            return {"text": "", "segments": []}
        
        # Convert audio data to the format expected by Whisper
        # Whisper expects float32 audio normalized to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_float32 = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float32 = audio_data.astype(np.float32)
        
        # If audio is stereo, convert to mono
        if len(audio_float32.shape) > 1 and audio_float32.shape[1] > 1:
            audio_float32 = audio_float32.mean(axis=1)
        
        # Flatten if needed
        audio_float32 = audio_float32.flatten()
        
        start_time = time.time()
        
        try:
            # Transcribe audio
            options = {}
            if self.language:
                options["language"] = self.language
            
            result = self.model.transcribe(
                audio_float32,
                **options
            )
            
            elapsed_time = time.time() - start_time
            print(f"Transcription completed in {elapsed_time:.2f} seconds")
            
            return result
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {"text": "", "segments": []}
    
    def transcribe_file(self, audio_file: str) -> Dict:
        """Transcribe audio from a file.
        
        Args:
            audio_file (str): Path to audio file.
            
        Returns:
            dict: Transcription result with text and timestamps.
        """
        if self.model is None:
            print("Whisper model not loaded")
            return {"text": "", "segments": []}
        
        try:
            # Transcribe audio file
            options = {}
            if self.language:
                options["language"] = self.language
            
            result = self.model.transcribe(
                audio_file,
                **options
            )
            
            return result
        except Exception as e:
            print(f"Error transcribing file: {e}")
            return {"text": "", "segments": []}
    
    def get_available_models(self) -> List[str]:
        """Get a list of available Whisper models.
        
        Returns:
            list: List of available model names.
        """
        return ["tiny", "base", "small", "medium", "large"]
    
    def get_model_info(self) -> Dict:
        """Get information about the current model.
        
        Returns:
            dict: Model information.
        """
        if self.model is None:
            return {"name": None, "device": None, "loaded": False}
        
        return {
            "name": self.model_name,
            "device": self.device,
            "loaded": self.model is not None,
            "language": self.language
        }


# Example usage
if __name__ == "__main__":
    import sounddevice as sd
    import soundfile as sf
    
    # Create a transcriber with the tiny model
    transcriber = WhisperTranscriber(model_name="tiny")
    
    # Record some audio for testing
    sample_rate = 16000
    duration = 5  # seconds
    
    print(f"Recording {duration} seconds of audio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished")
    
    # Save the audio for reference
    sf.write("test_audio.wav", audio_data, sample_rate)
    
    # Transcribe the recorded audio
    result = transcriber.transcribe(audio_data)
    
    # Print the result
    print("\nTranscription result:")
    print(f"Text: {result['text']}")
    print("\nSegments:")
    for segment in result.get('segments', []):
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
