import os
import sys
import time
import torch
import whisper
import numpy as np

def check_gpu():
    """Check if GPU is available and print details"""
    print("===== GPU Availability Check =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("No GPU detected. Running on CPU only.")
        return False

def test_whisper_performance(audio_file=None):
    """Test Whisper performance with and without GPU"""
    print("\n===== Whisper Performance Test =====")
    
    # Use a sample audio file if none provided
    if not audio_file:
        # Check for sample audio files in the utils directory
        sample_files = [
            os.path.join("utils", "audio 1.mp3"),
            os.path.join("utils", "Committee of the Whole (audio-2) Mar 11, 2025.mp3"),
            "temp_audio_1747788598.628241.mp3"
        ]
        
        for file in sample_files:
            if os.path.exists(file):
                audio_file = file
                break
        
        if not audio_file:
            print("No sample audio file found. Please provide an audio file path.")
            return
    
    print(f"Using audio file: {audio_file}")
    
    # Test CPU performance
    print("\nTesting on CPU...")
    whisper_model_cpu = whisper.load_model("base", device="cpu")
    
    start_time = time.time()
    result_cpu = whisper_model_cpu.transcribe(audio_file, fp16=False)
    cpu_time = time.time() - start_time
    
    print(f"CPU transcription completed in {cpu_time:.2f} seconds")
    
    # Test GPU performance if available
    if torch.cuda.is_available():
        print("\nTesting on GPU...")
        whisper_model_gpu = whisper.load_model("base", device="cuda")
        
        # Warm up GPU
        print("Warming up GPU...")
        _ = whisper_model_gpu.transcribe(audio_file, fp16=True)
        
        # Actual test
        start_time = time.time()
        result_gpu = whisper_model_gpu.transcribe(audio_file, fp16=True)
        gpu_time = time.time() - start_time
        
        print(f"GPU transcription completed in {gpu_time:.2f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\nGPU is {speedup:.2f}x faster than CPU")
    
    return True

if __name__ == "__main__":
    print("ClueAI GPU Acceleration Test")
    print("============================\n")
    
    has_gpu = check_gpu()
    
    # Test Whisper performance
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    test_whisper_performance(audio_file)
    
    print("\nTest completed!")
    if has_gpu:
        print("✅ GPU acceleration is available and working properly.")
    else:
        print("⚠️ No GPU detected. For faster performance, consider using a system with CUDA-compatible GPU.") 