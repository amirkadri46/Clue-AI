# utils/audio_processing.py
import whisper
import torch
import os
import logging
import shutil
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def transcribe_meeting(audio_path):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file missing: {audio_path}")
        
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        if device == "cuda":
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Use smaller 'base' model for balance of speed and accuracy
        start_time = time.time()
        logging.info("Loading Whisper model...")
        model = whisper.load_model("base", device=device)
        logging.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        
        # Try to find ffmpeg in system PATH first
        ffmpeg_path = shutil.which("ffmpeg")
        
        # If not found, try the specified path (as fallback)
        if not ffmpeg_path:
            possible_paths = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"ffmpeg"  # Try without path as last resort
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break
            
            logging.info(f"Using ffmpeg path: {ffmpeg_path}")
        
        # Check if we found a valid ffmpeg
        if not ffmpeg_path:
            logging.warning("ffmpeg not found. Make sure it's installed and in your PATH.")
        
        # Start transcription with timing
        start_time = time.time()
        logging.info(f"Transcribing audio using {device}...")
        
        result = model.transcribe(
            audio_path,
            fp16=(device == "cuda"),  # Use fp16 only with GPU
            # Only pass ffmpeg_path if we found it
            **({"ffmpeg_path": ffmpeg_path} if ffmpeg_path else {})
        )
        
        duration = time.time() - start_time
        logging.info(f"Transcription completed in {duration:.2f} seconds")
        
        return result["text"]
        
    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        raise