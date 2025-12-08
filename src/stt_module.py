import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import os
import tempfile
from pathlib import Path

WHISPER_CPP_EXECUTABLE = os.environ.get('WHISPER_CPP_EXECUTABLE', r'tools\whisper.cpp\main.exe')
WHISPER_MODEL_PATH = os.environ.get('WHISPER_MODEL_PATH', r'audio_models\stt\whisper_model.bin')

def record_and_transcribe(duration=7, sample_rate=16000):
    exe = Path(WHISPER_CPP_EXECUTABLE)
    model = Path(WHISPER_MODEL_PATH)

    if not exe.exists():
        raise FileNotFoundError(
            f"Whisper.cpp executable not found at '{exe}'.\n"
            "Place the whisper.cpp 'main.exe' in that path or set the WHISPER_CPP_EXECUTABLE env var."
        )
    if not model.exists():
        raise FileNotFoundError(f"Whisper model not found at '{model}'.")

    print("Recording your command...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")

    temp_wav_file = tempfile.mktemp(suffix=".wav", prefix="jarvis_stt_")
    write(temp_wav_file, sample_rate, recording)

    print("Transcribing audio...")
    try:
        # whisper.cpp typically writes a .txt next to the wav file; we pass --output-txt to request that
        command = [str(exe), "--model", str(model), "--file", temp_wav_file, "--output-txt"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Prefer the .txt file produced by whisper.cpp if present
        txt_out = temp_wav_file + ".txt"
        if os.path.exists(txt_out):
            with open(txt_out, 'r', encoding='utf-8') as f:
                transcribed_text = f.read().strip()
        else:
            # Fallback to any stdout produced
            transcribed_text = result.stdout.strip()

        print(f"Transcription: '{transcribed_text}'")
        return transcribed_text
    except subprocess.CalledProcessError as e:
        print(f"Error during transcription: {e.stderr}")
        return ""
    finally:
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
        if os.path.exists(temp_wav_file + ".txt"):
            os.remove(temp_wav_file + ".txt")