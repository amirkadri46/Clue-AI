# ClueAI: Smart Workplace Assistant

A powerful AI-powered assistant for meeting summarization and task extraction, capable of processing audio files, text, and YouTube videos to generate concise summaries and actionable tasks.

## Features

- Audio and YouTube video transcription using OpenAI's Whisper model with GPU acceleration
- Intelligent summarization of meeting content
- Automatic extraction of actionable tasks with deadlines and responsibilities
- Interactive chat interface for refining summaries and tasks
- Google Sheets integration for task management
- User session management for saving and retrieving past analyses
- Beautiful Streamlit interface with real-time processing feedback

## Performance

- **Audio Transcription**: ~1.5 minutes for a 15-20 minute audio file (with GPU acceleration)
- **Summary Generation**: ~5 seconds
- **Task Extraction**: ~1 second

> **Note**: This is the basic version of ClueAI. Performance may vary based on your hardware configuration, particularly GPU availability and specifications.

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure FFmpeg is installed and in your PATH:
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and extract to C:\ffmpeg
   - Linux: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`

4. Obtain a Groq API key from [console.groq.com](https://console.groq.com)

## Hardware Requirements

- **Minimum**: Any modern CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with CUDA support for faster audio transcription
- **Storage**: At least 5GB free space for model files

## Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter your Groq API key in the sidebar

4. Upload an audio file, paste text, or provide a YouTube URL to process

## How it Works

1. **Input Processing**: Audio/video is transcribed using Whisper model with GPU acceleration
2. **Summarization**: The transcript is analyzed to generate a concise summary
3. **Task Extraction**: AI identifies actionable tasks with deadlines and responsibilities
4. **Refinement**: Use the chat interface to ask for more specific information or refinements
5. **Integration**: Export tasks to Google Sheets for team management

## Technologies Used

- OpenAI Whisper for audio transcription
- LangChain for AI orchestration
- Groq for fast LLM inference
- PyTorch with CUDA for GPU acceleration
- Streamlit for the user interface

## Future Improvements

- Multi-speaker diarization for better meeting transcription
- Real-time meeting transcription and summarization
- Calendar integration for task scheduling
- Team collaboration features
- Fine-tuned models for specific industries or meeting types 