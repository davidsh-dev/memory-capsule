"""
Memory Capsule - Installation and Usage Guide

This document provides instructions for installing and using the Speaker/Microphone Memory Capsule system.
"""

# Speaker/Microphone Memory Capsule

A Python-based system that continuously listens to ambient audio, transcribes and diarizes speech, 
stores conversations with context, and enables real-time interaction with an AI assistant.

## Features

- Continuous audio capture from microphone
- Real-time speech transcription using OpenAI's Whisper
- Speaker diarization to distinguish different speakers
- Persistent storage of conversation transcripts
- Language model integration for AI assistant capabilities
- Memory search with keyword and vector-based retrieval
- Contextual question answering with RAG (Retrieval-Augmented Generation)
- Text-to-speech output for assistant responses
- Interactive command-line interface

## Installation

### Prerequisites

- Python 3.8 or higher
- Anaconda or Miniconda (recommended for environment management)
- Microphone for audio input
- Speakers for audio output

### Setup with Anaconda

1. Clone or download this repository:

```bash
git clone https://github.com/yourusername/memory-capsule.git
cd memory-capsule
```

2. Create a new Anaconda environment:

```bash
conda create -n memory-capsule python=3.10
conda activate memory-capsule
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) For OpenAI API access, set your API key:

```bash
# On Linux/macOS
export OPENAI_API_KEY=your-api-key-here

# On Windows
set OPENAI_API_KEY=your-api-key-here
```

Alternatively, create a `.env` file in the project root with:

```
OPENAI_API_KEY=your-api-key-here
```

### Setup for Local LLM (Optional)

For offline operation without OpenAI API:

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)

2. Pull a language model:

```bash
ollama pull llama3.2:1b
```

3. Run the memory capsule with the `--use-local-llm` flag and specify the model:

```bash
python memory_capsule/run.py --use-local-llm --llm-model llama3.2:1b
```

## Usage

### Running in Spyder

1. Open Spyder from Anaconda Navigator or command line:

```bash
conda activate memory-capsule
spyder
```

2. Open the `memory_capsule/run.py` file in Spyder

3. Run the script by clicking the green play button or pressing F5

4. Interact with the memory capsule through the command-line interface

### Running from Command Line

Run the memory capsule with default settings:

```bash
python memory_capsule/run.py
```

### Command-Line Options

The memory capsule supports various command-line options:

```
usage: run.py [-h] [--data-dir DATA_DIR] [--db-path DB_PATH] [--sample-rate SAMPLE_RATE] [--channels CHANNELS]
              [--chunk-size CHUNK_SIZE] [--whisper-model WHISPER_MODEL] [--device DEVICE] [--llm-model LLM_MODEL]
              [--use-local-llm] [--embedding-model EMBEDDING_MODEL] [--tts-voice TTS_VOICE] [--tts-rate TTS_RATE]

Memory Capsule - Speaker/Microphone Memory System

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Directory for storing data files
  --db-path DB_PATH     Path to database file (default: <data_dir>/conversations.db)
  --sample-rate SAMPLE_RATE
                        Audio sample rate in Hz (default: 16000)
  --channels CHANNELS   Number of audio channels (default: 1)
  --chunk-size CHUNK_SIZE
                        Audio chunk size in frames (default: 1024)
  --whisper-model WHISPER_MODEL
                        Whisper model name (tiny, base, small, medium, large) (default: tiny)
  --device DEVICE       Device to run models on (cpu, cuda) (default: auto-detect)
  --llm-model LLM_MODEL
                        Language model name (default: gpt-3.5-turbo)
  --use-local-llm       Use local LLM via Ollama instead of OpenAI API
  --embedding-model EMBEDDING_MODEL
                        Embedding model name (default: all-MiniLM-L6-v2)
  --tts-voice TTS_VOICE
                        TTS voice ID (default: system default)
  --tts-rate TTS_RATE   TTS speech rate in words per minute (default: 150)
```

### Interactive Commands

Once the memory capsule is running, you can use the following commands:

- `start` - Start the memory capsule
- `stop` - Stop the memory capsule
- `pause` - Pause audio recording and processing
- `resume` - Resume audio recording and processing
- `ask <text>` - Ask a question to the assistant
- `search <query>` - Search memory for utterances
- `list [n]` - List recent conversations (default: 5)
- `view <id>` - View a conversation by ID
- `stats` - Show database statistics
- `help [cmd]` - Show help message for a command
- `exit` or `quit` - Exit the program

## Example Usage Scenarios

### Basic Recording and Transcription

1. Start the memory capsule
2. Let it run during a meeting or conversation
3. The system will automatically transcribe speech and identify speakers
4. View the transcribed conversation with `list` and `view` commands

### Asking Questions About Past Conversations

1. After recording some conversations
2. Use the `ask` command to query about past topics
3. For example: `ask What did we discuss about the project timeline?`
4. The assistant will search the memory and provide a contextual answer

### Searching for Specific Topics

1. Use the `search` command to find mentions of specific topics
2. For example: `search budget proposal`
3. The system will return utterances containing those keywords

## Project Structure

- `audio/` - Audio recording module
- `transcription/` - Speech transcription with Whisper
- `diarization/` - Speaker diarization module
- `storage/` - Conversation database module
- `llm/` - Language model integration
- `memory/` - Memory search module
- `qa/` - Contextual question answering
- `tts/` - Text-to-speech module
- `ui/` - User interface module
- `tests/` - Test modules for each component
- `run.py` - Main application script

## Troubleshooting

### Audio Issues

- Ensure your microphone is properly connected and set as the default input device
- Try adjusting your system's microphone input level
- If using a USB microphone, try a different USB port

### Transcription Issues

- For better transcription quality, try using a larger Whisper model:
  ```bash
  python memory_capsule/run.py --whisper-model base
  ```
- Ensure you're speaking clearly and the microphone is positioned correctly

### LLM Issues

- If using OpenAI API, check that your API key is valid and has sufficient credits
- If using local LLM, ensure Ollama is running and the model is downloaded

## License

This project is licensed under the MIT License - see the LICENSE file for details.
