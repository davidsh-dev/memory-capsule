# Memory Capsule - Spyder Quick Start Guide

This guide provides instructions for running the Memory Capsule in Spyder IDE within Anaconda.

## Setup in Anaconda

1. Open Anaconda Navigator
2. Create a new environment:
   - Click "Environments" tab
   - Click "Create" button
   - Name it "memory-capsule"
   - Select Python 3.10
   - Click "Create"

3. Install required packages:
   - Select the "memory-capsule" environment
   - Click "Open Terminal"
   - Navigate to the memory_capsule directory
   - Run: `pip install -r requirements.txt`

## Running in Spyder

1. Open Spyder from Anaconda Navigator:
   - Select the "memory-capsule" environment
   - Click "Home" tab
   - Click "Spyder" to launch

2. Open the project in Spyder:
   - File → Open
   - Navigate to the memory_capsule directory
   - Open `run.py`

3. Configure run settings:
   - Click the run settings button (gear icon) next to the run button
   - Set "Command line options" if needed (e.g., `--whisper-model base --use-local-llm`)
   - Click "OK"

4. Run the application:
   - Click the run button (green play icon) or press F5
   - The Memory Capsule will start in the console

5. Interact with the application:
   - Type commands in the console (e.g., `start`, `ask What time is it?`)
   - See the README.md file for a complete list of commands

## Troubleshooting in Spyder

- If you encounter audio device issues:
  - Try restarting Spyder
  - Ensure your microphone is set as the default input device in your system settings

- If packages are not found:
  - In Spyder, go to Tools → Preferences → Python interpreter
  - Ensure the correct environment is selected
  - Click "Apply" and restart Spyder

- For more detailed logs:
  - Run with the verbose flag: `--verbose`
  - Check the console output for error messages
