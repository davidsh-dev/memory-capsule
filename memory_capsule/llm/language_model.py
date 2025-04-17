"""
Language Model Module

This module handles integration with language models for generating responses.
It supports both local models via Ollama and cloud-based models via OpenAI API.
"""

import os
import json
import time
import requests
from typing import Dict, List, Union, Optional, Any

class LanguageModel:
    """Class for interacting with language models."""
    
    def __init__(self, model_name="gpt-3.5-turbo", use_local=False, api_key=None, ollama_host="http://localhost:11434"):
        """Initialize the language model.
        
        Args:
            model_name (str): Name of the model to use. Default is 'gpt-3.5-turbo'.
            use_local (bool): Whether to use a local model via Ollama. Default is False.
            api_key (str, optional): OpenAI API key for cloud models. Default is None (uses env var).
            ollama_host (str): Host for Ollama API. Default is 'http://localhost:11434'.
        """
        self.model_name = model_name
        self.use_local = use_local
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.ollama_host = ollama_host
        
        print(f"Initializing language model (model: {model_name}, local: {use_local})")
    
    def generate_response(self, prompt: str, max_tokens=500, temperature=0.7) -> str:
        """Generate a response from the language model.
        
        Args:
            prompt (str): Input prompt for the model.
            max_tokens (int): Maximum number of tokens to generate. Default is 500.
            temperature (float): Sampling temperature. Default is 0.7.
            
        Returns:
            str: Generated response text.
        """
        if self.use_local:
            return self._generate_local(prompt, max_tokens, temperature)
        else:
            return self._generate_cloud(prompt, max_tokens, temperature)
    
    def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate a response using a local model via Ollama.
        
        Args:
            prompt (str): Input prompt for the model.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            
        Returns:
            str: Generated response text.
        """
        try:
            # Prepare request to Ollama API
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            # Send request
            response = requests.post(url, json=payload)
            
            # Check for errors
            if response.status_code != 200:
                print(f"Error from Ollama API: {response.status_code} {response.text}")
                return f"Error generating response: {response.status_code}"
            
            # Parse response
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"Error generating local response: {e}")
            return f"Error: {str(e)}"
    
    def _generate_cloud(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate a response using OpenAI API.
        
        Args:
            prompt (str): Input prompt for the model.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            
        Returns:
            str: Generated response text.
        """
        if not self.api_key:
            return "Error: OpenAI API key not provided"
        
        try:
            import openai
            
            # Set API key
            openai.api_key = self.api_key
            
            # Create chat completion
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response text
            return response.choices[0].message.content
        except ImportError:
            return "Error: openai package not installed"
        except Exception as e:
            print(f"Error generating cloud response: {e}")
            return f"Error: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], max_tokens=500, temperature=0.7) -> str:
        """Generate a response in a chat conversation.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'.
            max_tokens (int): Maximum number of tokens to generate. Default is 500.
            temperature (float): Sampling temperature. Default is 0.7.
            
        Returns:
            str: Generated response text.
        """
        if self.use_local:
            return self._chat_local(messages, max_tokens, temperature)
        else:
            return self._chat_cloud(messages, max_tokens, temperature)
    
    def _chat_local(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        """Generate a chat response using a local model via Ollama.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            
        Returns:
            str: Generated response text.
        """
        try:
            # Convert messages to a prompt format that Ollama can understand
            prompt = ""
            for message in messages:
                role = message.get("role", "").lower()
                content = message.get("content", "")
                
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            
            prompt += "Assistant: "
            
            # Generate response
            return self._generate_local(prompt, max_tokens, temperature)
        except Exception as e:
            print(f"Error generating local chat response: {e}")
            return f"Error: {str(e)}"
    
    def _chat_cloud(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        """Generate a chat response using OpenAI API.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            
        Returns:
            str: Generated response text.
        """
        if not self.api_key:
            return "Error: OpenAI API key not provided"
        
        try:
            import openai
            
            # Set API key
            openai.api_key = self.api_key
            
            # Create chat completion
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response text
            return response.choices[0].message.content
        except ImportError:
            return "Error: openai package not installed"
        except Exception as e:
            print(f"Error generating cloud chat response: {e}")
            return f"Error: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get a list of available models.
        
        Returns:
            list: List of available model names.
        """
        if self.use_local:
            return self._get_local_models()
        else:
            return self._get_cloud_models()
    
    def _get_local_models(self) -> List[str]:
        """Get a list of available local models from Ollama.
        
        Returns:
            list: List of available model names.
        """
        try:
            # Query Ollama API for available models
            url = f"{self.ollama_host}/api/tags"
            response = requests.get(url)
            
            # Check for errors
            if response.status_code != 200:
                print(f"Error from Ollama API: {response.status_code} {response.text}")
                return []
            
            # Parse response
            result = response.json()
            models = [model.get("name") for model in result.get("models", [])]
            return models
        except Exception as e:
            print(f"Error getting local models: {e}")
            return []
    
    def _get_cloud_models(self) -> List[str]:
        """Get a list of available cloud models from OpenAI.
        
        Returns:
            list: List of available model names.
        """
        if not self.api_key:
            return []
        
        try:
            import openai
            
            # Set API key
            openai.api_key = self.api_key
            
            # Get available models
            response = openai.Model.list()
            
            # Extract model names
            models = [model.id for model in response.data]
            
            # Filter for chat models
            chat_models = [model for model in models if "gpt" in model.lower()]
            
            return chat_models
        except ImportError:
            return ["openai package not installed"]
        except Exception as e:
            print(f"Error getting cloud models: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Check if we have an OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        # Create a cloud-based language model
        llm = LanguageModel(model_name="gpt-3.5-turbo", use_local=False)
        
        # Generate a response
        prompt = "What is the capital of France?"
        print(f"Prompt: {prompt}")
        response = llm.generate_response(prompt)
        print(f"Response: {response}")
        
        # Try a chat conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I'm an AI assistant. How can I help you today?"},
            {"role": "user", "content": "What's the weather like?"}
        ]
        response = llm.chat(messages)
        print(f"\nChat response: {response}")
    
    # Try with a local model if Ollama is available
    try:
        # Check if Ollama is running
        requests.get("http://localhost:11434/api/tags")
        
        # Create a local language model
        local_llm = LanguageModel(model_name="llama2", use_local=True)
        
        # Generate a response
        prompt = "What is the capital of France?"
        print(f"\nLocal prompt: {prompt}")
        response = local_llm.generate_response(prompt)
        print(f"Local response: {response}")
        
    except Exception as e:
        print(f"\nOllama not available: {e}")
        print("To use local models, install Ollama from https://ollama.ai/")
