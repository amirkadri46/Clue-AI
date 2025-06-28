import subprocess
import os
import sys

def check_path():
    print("Current PATH:")
    print(os.environ.get("PATH"))
    
    # Check if FFmpeg directory is in PATH
    if r"C:\ffmpeg\bin" in os.environ.get("PATH", ""):
        print("\nFFmpeg directory is in PATH")
    else:
        print("\nFFmpeg directory is NOT in PATH")
        
    # Check if specific directories exist
    ffmpeg_dirs = [r"C:\ffmpeg", r"C:\ffmpeg\bin"]
    for directory in ffmpeg_dirs:
        if os.path.exists(directory):
            print(f"Directory {directory} exists")
            if os.path.isdir(directory):
                print(f"Contents of {directory}:")
                try:
                    files = os.listdir(directory)
                    for file in files:
                        print(f"  - {file}")
                except Exception as e:
                    print(f"  Error listing directory: {e}")
        else:
            print(f"Directory {directory} does NOT exist")

def test_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
        print("\nFFmpeg is accessible!")
        print("FFmpeg version info:")
        print(result.stdout.split('\n')[0])
        return True
    except FileNotFoundError:
        print("\nFFmpeg not found in PATH!")
        return False
    except Exception as e:
        print(f"\nError running FFmpeg: {e}")
        return False

def test_whisper():
    try:
        import numpy
        print("\nNumPy is installed. Version:", numpy.__version__)
    except ImportError:
        print("\nNumPy is NOT installed!")
        return False
    
    try:
        import whisper
        print("Whisper is installed.")
        model = whisper.load_model("base")
        print("Whisper model loaded successfully!")
        return True
    except ImportError:
        print("Whisper is NOT installed!")
        return False
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return False

if __name__ == "__main__":
    print("===== FFmpeg and Whisper Test =====")
    check_path()
    ffmpeg_ok = test_ffmpeg()
    whisper_ok = test_whisper()
    
    print("\n===== Test Results =====")
    print(f"FFmpeg: {'✅ OK' if ffmpeg_ok else '❌ FAILED'}")
    print(f"Whisper: {'✅ OK' if whisper_ok else '❌ FAILED'}")
    
    if not ffmpeg_ok:
        print("\nTo fix FFmpeg issues:")
        print("1. Download FFmpeg from https://ffmpeg.org/download.html")
        print("2. Extract to C:\\ffmpeg")
        print("3. Add C:\\ffmpeg\\bin to your PATH environment variable")
        
    if not whisper_ok:
        print("\nTo fix Whisper issues:")
        print("1. Install numpy: pip install numpy")
        print("2. Install whisper: pip install openai-whisper")
        print("3. Make sure FFmpeg is installed correctly")