import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import os
import tempfile
from pathlib import Path

# You can override these via environment variables if you want.
WHISPER_CPP_EXECUTABLE = Path(
    os.environ.get("WHISPER_CPP_EXECUTABLE", r"tools\whisper.cpp\whisper-cli.exe")
)
WHISPER_MODEL_PATH = Path(
    os.environ.get("WHISPER_MODEL_PATH", r"audio_models\stt\whisper_model.bin")
)

DEFAULT_DURATION = 7
DEFAULT_SAMPLE_RATE = 16000


def _check_paths():
    if not WHISPER_CPP_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Whisper.cpp executable not found at '{WHISPER_CPP_EXECUTABLE}'.\n"
            "Place the whisper.cpp 'main.exe' there or set WHISPER_CPP_EXECUTABLE env var."
        )
    if not WHISPER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Whisper model not found at '{WHISPER_MODEL_PATH}'.\n"
            "Make sure the .bin model is downloaded and the path is correct, "
            "or set WHISPER_MODEL_PATH env var."
        )


def record_and_transcribe(duration: int = DEFAULT_DURATION,
                          sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
    """
    Record microphone audio and transcribe it with whisper.cpp.

    Returns the transcribed text (possibly empty string on error).
    """

    try:
        _check_paths()
    except FileNotFoundError as e:
        # This will show up clearly in main.py as: "An error occurred in the action loop: ..."
        print(f"[STT][FATAL] {e}")
        return ""

    print("Recording your command...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("Recording finished.")

    # Save temp wav
    temp_wav_file = tempfile.mktemp(suffix=".wav", prefix="jarvis_stt_")
    write(temp_wav_file, sample_rate, recording)

    print("Transcribing audio...")
    txt_out_path = temp_wav_file + ".txt"

    try:
        # Canonical whisper.cpp CLI:
        #   main.exe -m <model> -f <wav> -otxt
        command = [
            str(WHISPER_CPP_EXECUTABLE),
            "-m", str(WHISPER_MODEL_PATH),
            "-f", temp_wav_file,
            "-otxt",
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # we will handle non-zero return codes ourselves
        )

        if result.returncode != 0:
            print("[STT][ERROR] whisper.cpp returned non-zero exit code:")
            if result.stderr:
                print(result.stderr)
            else:
                print("(no stderr output)")
            # Fallback: also show stdout if any
            if result.stdout:
                print("[STT][INFO] whisper.cpp stdout:")
                print(result.stdout)
            return ""

        # Prefer the generated .txt file
        transcribed_text = ""
        if os.path.exists(txt_out_path):
            with open(txt_out_path, "r", encoding="utf-8") as f:
                transcribed_text = f.read().strip()
        else:
            # Fallback: whatever whisper.cpp printed to stdout
            transcribed_text = (result.stdout or "").strip()

        print(f"Transcription: '{transcribed_text}'")
        return transcribed_text

    except Exception as e:
        print(f"[STT][ERROR] Unexpected transcription error: {e}")
        return ""

    finally:
        try:
            if os.path.exists(temp_wav_file):
                os.remove(temp_wav_file)
            if os.path.exists(txt_out_path):
                os.remove(txt_out_path)
        except OSError:
            pass
