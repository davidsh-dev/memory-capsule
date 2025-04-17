"""
Speech Synthesizer Module

This module handles text-to-speech conversion for the memory capsule.
It provides functionality to convert text responses into spoken audio.
"""

import pyttsx3
import threading
import time
from typing import Dict, List, Union, Optional, Any

class SpeechSynthesizer:
    """Class for converting text to speech using pyttsx3."""
    
    def __init__(self, voice=None, rate=150, volume=1.0):
        """Initialize the speech synthesizer.
        
        Args:
            voice (str, optional): Voice ID to use. Default is None (system default).
            rate (int): Speech rate. Default is 150 words per minute.
            volume (float): Volume from 0.0 to 1.0. Default is 1.0.
        """
        self.voice_id = voice
        self.rate = rate
        self.volume = volume
        self.engine = None
        self.speaking = False
        self.speech_thread = None
        
        print("Initializing speech synthesizer")
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the TTS engine."""
        try:
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Set voice if specified
            if self.voice_id:
                self.engine.setProperty('voice', self.voice_id)
            
            print("Speech synthesizer initialized successfully")
        except Exception as e:
            print(f"Error initializing speech synthesizer: {e}")
            self.engine = None
    
    def speak(self, text, block=False):
        """Convert text to speech.
        
        Args:
            text (str): Text to convert to speech.
            block (bool): Whether to block until speech is complete. Default is False.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.engine:
            print("Speech synthesizer not initialized")
            return False
        
        if not text:
            print("No text provided for speech synthesis")
            return False
        
        try:
            if block:
                # Speak synchronously
                self.speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
                self.speaking = False
            else:
                # Speak asynchronously
                if self.speech_thread and self.speech_thread.is_alive():
                    print("Already speaking, cannot start new speech")
                    return False
                
                self.speech_thread = threading.Thread(target=self._speak_async, args=(text,))
                self.speech_thread.daemon = True
                self.speech_thread.start()
            
            return True
        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            self.speaking = False
            return False
    
    def _speak_async(self, text):
        """Speak text asynchronously.
        
        Args:
            text (str): Text to speak.
        """
        self.speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.speaking = False
    
    def is_speaking(self):
        """Check if the synthesizer is currently speaking.
        
        Returns:
            bool: True if speaking, False otherwise.
        """
        return self.speaking
    
    def stop(self):
        """Stop current speech.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.engine:
            print("Speech synthesizer not initialized")
            return False
        
        try:
            self.engine.stop()
            self.speaking = False
            return True
        except Exception as e:
            print(f"Error stopping speech: {e}")
            return False
    
    def get_available_voices(self):
        """Get a list of available voices.
        
        Returns:
            list: List of available voice objects.
        """
        if not self.engine:
            print("Speech synthesizer not initialized")
            return []
        
        try:
            return self.engine.getProperty('voices')
        except Exception as e:
            print(f"Error getting available voices: {e}")
            return []
    
    def print_available_voices(self):
        """Print a list of available voices."""
        voices = self.get_available_voices()
        
        if not voices:
            print("No voices available")
            return
        
        print(f"Available voices ({len(voices)}):")
        for i, voice in enumerate(voices):
            print(f"{i}: ID={voice.id}, Name={voice.name}, Languages={voice.languages}")
    
    def set_voice(self, voice_id):
        """Set the voice by ID.
        
        Args:
            voice_id (str): Voice ID to use.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.engine:
            print("Speech synthesizer not initialized")
            return False
        
        try:
            self.engine.setProperty('voice', voice_id)
            self.voice_id = voice_id
            return True
        except Exception as e:
            print(f"Error setting voice: {e}")
            return False
    
    def set_rate(self, rate):
        """Set the speech rate.
        
        Args:
            rate (int): Speech rate in words per minute.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.engine:
            print("Speech synthesizer not initialized")
            return False
        
        try:
            self.engine.setProperty('rate', rate)
            self.rate = rate
            return True
        except Exception as e:
            print(f"Error setting rate: {e}")
            return False
    
    def set_volume(self, volume):
        """Set the volume.
        
        Args:
            volume (float): Volume from 0.0 to 1.0.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.engine:
            print("Speech synthesizer not initialized")
            return False
        
        try:
            self.engine.setProperty('volume', volume)
            self.volume = volume
            return True
        except Exception as e:
            print(f"Error setting volume: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create a speech synthesizer
    synthesizer = SpeechSynthesizer()
    
    # Print available voices
    synthesizer.print_available_voices()
    
    # Speak a test message
    print("\nSpeaking test message...")
    synthesizer.speak("Hello, this is a test of the speech synthesizer. I can convert text to speech for the memory capsule project.", block=True)
    
    # Test asynchronous speech
    print("\nTesting asynchronous speech...")
    synthesizer.speak("This is an asynchronous speech test. The program continues while I'm speaking.")
    
    # Wait a moment to let the speech start
    time.sleep(1)
    
    # Check if speaking
    print(f"Is speaking: {synthesizer.is_speaking()}")
    
    # Wait for speech to complete
    while synthesizer.is_speaking():
        print("Still speaking...")
        time.sleep(1)
    
    print("Speech completed.")
    
    # Test different rates
    print("\nTesting different speech rates...")
    synthesizer.set_rate(100)
    synthesizer.speak("This is slow speech at 100 words per minute.", block=True)
    
    synthesizer.set_rate(200)
    synthesizer.speak("This is fast speech at 200 words per minute.", block=True)
    
    # Reset to default rate
    synthesizer.set_rate(150)
    
    print("Speech synthesizer test completed.")
