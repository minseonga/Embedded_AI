
import sounddevice as sd
import sys

print("Checking audio devices...")
try:
    devices = sd.query_devices()
    print(devices)
    
    default_input = sd.query_devices(kind='input')
    print(f"\nDefault input device:\n{default_input}")
    
except Exception as e:
    print(f"Error querying devices: {e}")
    print("\nPossible causes:")
    print("1. PortAudio is not installed (brew install portaudio / sudo apt-get install libportaudio2)")
    print("2. No microphone connected")
    print("3. Permissions issue (Mac/Linux)")

print("\nChecking required libraries for voice control...")
try:
    import vosk
    print("[OK] vosk installed")
except ImportError:
    print("[MISSING] vosk not installed")

try:
    import transformers
    print("[OK] transformers installed")
except ImportError:
    print("[MISSING] transformers not installed")
