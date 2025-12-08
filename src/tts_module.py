import subprocess
import os
import sounddevice as sd
import numpy as np

PIPER_EXECUTABLE = "path/to/piper/piper.exe"
PIPER_VOICE_MODEL_PATH = "audio_models/tts/piper_voice.onnx"
VOICE_SAMPLE_RATE = 22050

def speak(text):
    if not os.path.exists(PIPER_EXECUTABLE):
        raise FileNotFoundError(f"Piper executable not found at '{PIPER_EXECUTABLE}'.")
    if not os.path.exists(PIPER_VOICE_MODEL_PATH):
        raise FileNotFoundError(f"Piper voice model not found at '{PIPER_VOICE_MODEL_PATH}'.")

    print(f"Speaking: '{text}'")
    try:
        command = [PIPER_EXECUTABLE, "--model", PIPER_VOICE_MODEL_PATH, "--output-raw"]
        result = subprocess.run(command, input=text.encode('utf-8'), capture_output=True, check=True)
        audio_data = np.frombuffer(result.stdout, dtype=np.int16)
        sd.play(audio_data, samplerate=VOICE_SAMPLE_RATE)
        sd.wait()
    except subprocess.CalledProcessError as e:
        print(f"Error during TTS generation: {e.stderr.decode()}")