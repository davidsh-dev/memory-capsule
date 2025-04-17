"""
Speaker Diarizer Module

This module handles real-time speaker diarization to determine who spoke when.
It provides functionality to identify different speakers in audio segments.
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional
import time

# Try to import diart for diarization
try:
    import diart
    import diart.sources
    import diart.pipelines
    from diart.models import SegmentationModel, EmbeddingModel
    from diart.inference import Benchmark, SpeakerDiarization
    DIART_AVAILABLE = True
except ImportError:
    DIART_AVAILABLE = False
    print("Warning: diart library not available. Speaker diarization will be limited.")

# Fallback to pyannote if diart is not available
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    if not DIART_AVAILABLE:
        print("Warning: Neither diart nor pyannote.audio are available. Speaker diarization will not work.")

class SpeakerDiarizer:
    """Class for performing speaker diarization on audio segments."""
    
    def __init__(self, sample_rate=16000, device=None, use_diart=True):
        """Initialize the speaker diarizer.
        
        Args:
            sample_rate (int): Audio sample rate in Hz. Default is 16000.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                Default is None (auto-detect).
            use_diart (bool): Whether to use diart (True) or pyannote (False) for diarization.
                Default is True. Falls back to available library if the preferred one is not available.
        """
        self.sample_rate = sample_rate
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.use_diart = use_diart and DIART_AVAILABLE
        if self.use_diart and not DIART_AVAILABLE:
            print("diart library not available, falling back to pyannote if available")
            self.use_diart = False
        
        if not self.use_diart and not PYANNOTE_AVAILABLE:
            print("pyannote.audio library not available, speaker diarization will not work")
        
        self.diarization_model = None
        self.speaker_embeddings = {}  # Store speaker embeddings for identification
        
        print(f"Initializing speaker diarizer (sample rate: {sample_rate}Hz, device: {self.device})")
        self._load_model()
    
    def _load_model(self):
        """Load the diarization model."""
        try:
            if self.use_diart:
                # Load diart models
                segmentation = SegmentationModel.from_pretrained("pyannote/segmentation", device=self.device)
                embedding = EmbeddingModel.from_pretrained("pyannote/embedding", device=self.device)
                
                # Create diarization pipeline
                self.diarization_model = SpeakerDiarization(
                    segmentation=segmentation,
                    embedding=embedding,
                    device=self.device
                )
                print("diart diarization model loaded successfully")
            elif PYANNOTE_AVAILABLE:
                # Load pyannote pipeline
                self.diarization_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=True
                )
                if self.device == "cuda":
                    self.diarization_model = self.diarization_model.to(torch.device("cuda"))
                print("pyannote.audio diarization model loaded successfully")
            else:
                print("No diarization model available")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            self.diarization_model = None
    
    def diarize(self, audio_data: np.ndarray) -> List[Dict]:
        """Perform speaker diarization on audio data.
        
        Args:
            audio_data (ndarray): Audio data as numpy array.
            
        Returns:
            list: List of diarization segments with speaker labels and timestamps.
        """
        if self.diarization_model is None:
            print("Diarization model not loaded")
            return []
        
        # Convert audio data to the format expected by the diarization model
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
            if self.use_diart:
                return self._diarize_with_diart(audio_float32)
            elif PYANNOTE_AVAILABLE:
                return self._diarize_with_pyannote(audio_float32)
            else:
                return []
        except Exception as e:
            print(f"Error during diarization: {e}")
            return []
        finally:
            elapsed_time = time.time() - start_time
            print(f"Diarization completed in {elapsed_time:.2f} seconds")
    
    def _diarize_with_diart(self, audio_float32: np.ndarray) -> List[Dict]:
        """Perform diarization using diart.
        
        Args:
            audio_float32 (ndarray): Audio data as float32 numpy array.
            
        Returns:
            list: List of diarization segments with speaker labels and timestamps.
        """
        # Create a waveform source from the audio data
        waveform = torch.from_numpy(audio_float32).unsqueeze(0)
        
        # Run diarization
        output = self.diarization_model(waveform, self.sample_rate)
        
        # Extract speaker turns
        segments = []
        for turn in output.speaker_turns():
            segment = {
                "speaker": f"Speaker_{turn.speaker}",
                "start": turn.start,
                "end": turn.end
            }
            segments.append(segment)
        
        return segments
    
    def _diarize_with_pyannote(self, audio_float32: np.ndarray) -> List[Dict]:
        """Perform diarization using pyannote.audio.
        
        Args:
            audio_float32 (ndarray): Audio data as float32 numpy array.
            
        Returns:
            list: List of diarization segments with speaker labels and timestamps.
        """
        # Create a dictionary with the waveform and sample rate
        audio_dict = {
            "waveform": torch.from_numpy(audio_float32).unsqueeze(0),
            "sample_rate": self.sample_rate
        }
        
        # Run diarization
        diarization = self.diarization_model(audio_dict)
        
        # Extract speaker turns
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = {
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            }
            segments.append(segment)
        
        return segments
    
    def identify_speakers(self, audio_data: np.ndarray, diarization_result: List[Dict]) -> List[Dict]:
        """Identify speakers based on stored embeddings.
        
        Args:
            audio_data (ndarray): Audio data as numpy array.
            diarization_result (list): Diarization result from diarize().
            
        Returns:
            list: Updated diarization result with consistent speaker labels.
        """
        # This is a placeholder for speaker identification
        # In a real implementation, this would compare speaker embeddings
        # and assign consistent labels across sessions
        
        return diarization_result
    
    def register_speaker(self, name: str, audio_data: np.ndarray) -> bool:
        """Register a known speaker with a name.
        
        Args:
            name (str): Name of the speaker.
            audio_data (ndarray): Audio data containing the speaker's voice.
            
        Returns:
            bool: True if registration was successful, False otherwise.
        """
        # This is a placeholder for speaker registration
        # In a real implementation, this would extract speaker embeddings
        # and store them with the given name
        
        print(f"Registered speaker: {name}")
        return True


# Example usage
if __name__ == "__main__":
    import sounddevice as sd
    import soundfile as sf
    
    # Create a diarizer
    diarizer = SpeakerDiarizer()
    
    # Record some audio for testing
    sample_rate = 16000
    duration = 10  # seconds
    
    print(f"Recording {duration} seconds of audio...")
    print("Please have multiple people speak during this recording")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished")
    
    # Save the audio for reference
    sf.write("test_diarization.wav", audio_data, sample_rate)
    
    # Perform diarization
    result = diarizer.diarize(audio_data)
    
    # Print the result
    print("\nDiarization result:")
    for segment in result:
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['speaker']}")
