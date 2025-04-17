"""
Setup script for the Memory Capsule package.

This script installs the Memory Capsule package and its dependencies.
"""

from setuptools import setup, find_packages

with open("memory_capsule/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memory-capsule",
    version="0.1.0",
    author="Memory Capsule Team",
    author_email="example@example.com",
    description="A system that continuously listens to ambient audio, transcribes and diarizes speech, stores conversations with context, and enables real-time interaction with an AI assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/memory-capsule",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sounddevice>=0.4.5",
        "numpy>=1.20.0",
        "openai-whisper>=20231117",
        "diart>=0.9.2",
        "pyannote.audio>=2.1.1",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "faiss-cpu>=1.7.3",
        "sentence-transformers>=2.2.2",
        "pyttsx3>=2.90",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "memory-capsule=memory_capsule.run:main",
        ],
    },
)
