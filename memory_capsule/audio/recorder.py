"""
Audio Recorder Module

This module handles continuous audio capture from the microphone.
It provides functionality to record audio in real-time and buffer it for processing.
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time

class AudioRecorder:
    """Class for recording audio from the microphone in real-time."""
    
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024, device=None):
        """Initialize the audio recorder.
        
        Args:
            sample_rate (int): Sample rate in Hz. Default is 16000.
            channels (int): Number of audio channels. Default is 1 (mono).
            chunk_size (int): Number of frames per buffer. Default is 1024.
            device (int, optional): Audio device index. Default is None (system default).
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        
        self.audio_queue = queue.Queue()
        self.stream = None
        self.running = False
        self.thread = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for the audio stream.
        
        This function is called by the sounddevice stream for each audio chunk.
        
        Args:
            indata (ndarray): Recorded audio data.
            frames (int): Number of frames in the buffer.
            time_info (dict): Dictionary with timing information.
            status (CallbackFlags): Status flags indicating if an error occurred.
        """
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to the desired format (float32 to int16)
        audio_data = (indata * 32767).astype(np.int16)
        
        # Put the data in the queue
        self.audio_queue.put(audio_data.copy())
    
    def start(self):
        """Start audio recording."""
        if self.running:
            print("Audio recorder is already running")
            return
        
        self.running = True
        
        # Start the audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=self.chunk_size,
            device=self.device
        )
        
        self.stream.start()
        print(f"Audio recording started (sample rate: {self.sample_rate}Hz, channels: {self.channels})")
    
    def stop(self):
        """Stop audio recording."""
        if not self.running:
            print("Audio recorder is not running")
            return
        
        self.running = False
        
        # Stop the audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        print("Audio recording stopped")
    
    def get_audio(self, timeout=0.1):
        """Get the next audio chunk from the queue.
        
        Args:
            timeout (float): Timeout in seconds. Default is 0.1.
            
        Returns:
            ndarray or None: Audio data as numpy array, or None if queue is empty.
        """
        try:
            return self.audio_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_audio_buffer(self, duration):
        """Get a buffer of audio data with the specified duration.
        
        Args:
            duration (float): Duration in seconds.
            
        Returns:
            ndarray: Audio data as numpy array.
        """
        # Calculate number of samples needed
        num_samples = int(self.sample_rate * duration)
        
        # Initialize buffer
        buffer = np.zeros((num_samples, self.channels), dtype=np.int16)
        
        # Fill buffer with audio data
        samples_collected = 0
        start_time = time.time()
        timeout = duration * 2  # Timeout after twice the requested duration
        
        while samples_collected < num_samples:
            # Check for timeout
            if time.time() - start_time > timeout:
                print(f"Timeout while collecting audio buffer (collected {samples_collected}/{num_samples} samples)")
                break
            
            # Get audio chunk
            chunk = self.get_audio()
            
            if chunk is None:
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
                continue
            
            # Calculate how many samples we can add
            samples_to_add = min(len(chunk), num_samples - samples_collected)
            
            # Add samples to buffer
            buffer[samples_collected:samples_collected + samples_to_add] = chunk[:samples_to_add]
            
            # Update counter
            samples_collected += samples_to_add
        
        return buffer
    
    def record_fixed_duration(self, duration):
        """Record audio for a fixed duration.
        
        Args:
            duration (float): Duration in seconds.
            
        Returns:
            ndarray: Audio data as numpy array.
        """
        print(f"Recording for {duration} seconds...")
        
        # Start recording if not already running
        was_running = self.running
        if not was_running:
            self.start()
        
        # Clear the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        # Record for the specified duration
        audio_data = self.get_audio_buffer(duration)
        
        # Stop recording if it wasn't running before
        if not was_running:
            self.stop()
        
        print(f"Recorded {len(audio_data)} samples")
        return audio_data
    
    def is_running(self):
        """Check if the audio recorder is running.
        
        Returns:
            bool: True if running, False otherwise.
        """
        return self.running
    
    def get_devices(self):
        """Get a list of available audio devices.
        
        Returns:
            list: List of audio devices.
        """
        return sd.query_devices()
    
    def print_devices(self):
        """Print a list of available audio devices."""
        devices = self.get_devices()
        print("Available audio devices:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")


# Example usage
if __name__ == "__main__":
    # Create an audio recorder
    recorder = AudioRecorder(sample_rate=16000, channels=1)
    
    # Print available devices
    recorder.print_devices()
    
    try:
        # Record for 5 seconds
        audio_data = recorder.record_fixed_duration(5.0)
        
        # Print some statistics
        print(f"Recorded audio shape: {audio_data.shape}")
        print(f"Audio min: {audio_data.min()}, max: {audio_data.max()}, mean: {audio_data.mean()}")
        
        # Save to a WAV file for testing
        try:
            import soundfile as sf
            sf.write("test_recording.wav", audio_data, recorder.sample_rate)
            print("Saved recording to test_recording.wav")
        except ImportError:
            print("soundfile library not available, cannot save WAV file")
        
    except KeyboardInterrupt:
        print("Recording interrupted")
    finally:
        # Make sure to stop the recorder
        recorder.stop()
