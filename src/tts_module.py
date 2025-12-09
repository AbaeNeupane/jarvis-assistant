# src/tts_module.py

import pyttsx3
import threading

_engine = None
_lock = threading.Lock()


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            _engine = pyttsx3.init()
            # Optional tweaks
            _engine.setProperty("rate", 180)  # speaking speed
            # voices = _engine.getProperty("voices")
            # _engine.setProperty("voice", voices[0].id)  # pick a different voice if you want
        return _engine


def speak(text: str):
    """
    Simple blocking TTS using pyttsx3.
    Used by main.py as: tts_module.speak(response)
    """
    if not text or not text.strip():
        return

    try:
        engine = _get_engine()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS][WARN] Failed to speak text: {e}")
        print(f"[TTS][INFO] Text was: {text!r}")
