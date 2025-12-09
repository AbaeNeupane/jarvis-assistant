import sounddevice as sd
import numpy as np
import openwakeword
from openwakeword.model import Model


class WakeWordDetector:
    def __init__(
        self,
        on_activation=None,
        sample_rate: int = 16000,
        chunk_size: int = 1280,
        threshold: float = 0.25,
        target_wakeword: str = "hey_jarvis",
        debug_audio_level: bool = False,
    ):
        """
        Wake word detector using OpenWakeWord.

        :param on_activation: callback function to call when wake word is detected
        :param sample_rate: microphone sample rate (OpenWakeWord expects 16 kHz)
        :param chunk_size: number of samples per audio chunk (1280 ≈ 80 ms)
        :param threshold: detection threshold (higher = stricter)
        :param target_wakeword: name of the wakeword model to react to
                                (e.g. 'hey_jarvis', as used in OpenWakeWord demos)
        :param debug_audio_level: if True, prints raw audio max level each callback
        """
        self.on_activation = on_activation
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.target_wakeword = target_wakeword
        self.debug_audio_level = debug_audio_level

        self.model: Model | None = None
        self.audio_stream: sd.InputStream | None = None
        self.disabled: bool = False

        try:
            # One-time download of all required models (embedding, mel, and wakewords).
            # This is safe to call repeatedly; it will skip files that already exist.
            print("[WakeWord] Ensuring OpenWakeWord models are available...")
            openwakeword.utils.download_models()

            # Instantiate the model.
            # Leaving wakeword_models empty loads all included pre-trained models
            # (e.g. alexa, hey_jarvis, hey_mycroft, etc., depending on version).
            self.model = Model()

            loaded_models = list(self.model.models.keys())
            print(f"[WakeWord] Loaded OpenWakeWord models: {loaded_models}")

            if self.target_wakeword and self.target_wakeword not in loaded_models:
                print(
                    f"[WakeWord][WARN] Target wake word '{self.target_wakeword}' "
                    f"not found in loaded models. Available: {loaded_models}"
                )

            print("[WakeWord] Wake Word Detector initialized successfully.")

        except Exception as e:
            print(f"[WakeWord][WARN] Could not initialize wake-word model: {e}")
            print("[WakeWord][WARN] Wake-word detection disabled.")
            self.model = None
            self.disabled = True

    def _audio_callback(self, indata, frames, time_info, status):
        """SoundDevice callback processing audio and checking wake-word score."""
        if self.disabled or self.model is None:
            return

        if status:
            print(f"[WakeWord][AUDIO] Stream status: {status}")

        # indata: shape (frames, channels), dtype=int16 (as configured below)
        try:
            audio_chunk = indata[:, 0].astype(np.int16, copy=True)  # mono
        except Exception as e:
            print(f"[WakeWord][ERROR] Failed to process audio chunk: {e}")
            return

        max_val = int(np.max(np.abs(audio_chunk))) if audio_chunk.size > 0 else 0

        if self.debug_audio_level:
            print(f"[WakeWord][AUDIO] frames={frames}, peak={max_val}", end="\r", flush=True)

        # Ignore pure silence
        if max_val == 0:
            return

        # Run inference
        try:
            self.model.predict(audio_chunk)
        except Exception as e:
            print(f"[WakeWord][ERROR] Model prediction failed: {e}")
            return

        # Check prediction buffer
        for model_name, scores in self.model.prediction_buffer.items():
            if not scores:
                continue

            # If a specific wakeword is requested, ignore others
            if self.target_wakeword and model_name != self.target_wakeword:
                continue

            prob = float(scores[-1])

            # Live probability display
            print(
                f"[WakeWord] {model_name}: {prob:.3f} (peak={max_val})",
                end="\r",
                flush=True,
            )

            if prob >= self.threshold:
                print(f"\n[WakeWord] Detected '{model_name}' with score {prob:.3f}")
                if self.on_activation:
                    # NOTE: This calls the callback directly in the audio thread.
                    # If it ever causes issues, you can dispatch it to another thread instead.
                    self.on_activation()

    def start(self):
        """Start the microphone stream and begin wake-word detection."""
        if self.disabled or self.model is None:
            print("[WakeWord] Wake-word detector is disabled (no usable model/runtime).")
            return

        if self.audio_stream is not None:
            print("[WakeWord] Audio stream already running.")
            return

        print("[WakeWord] Starting wake-word detection using OpenWakeWord...")
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_size,
                callback=self._audio_callback,
            )
            self.audio_stream.start()
            print(f"[WakeWord] Listening for '{self.target_wakeword or 'wake word(s)'}'...")
        except Exception as e:
            print(f"[WakeWord][ERROR] Failed to start audio stream: {e}")
            self.audio_stream = None
            self.disabled = True

    def stop(self):
        """Stop the microphone stream and clean up."""
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                print(f"[WakeWord][ERROR] Error stopping audio stream: {e}")
            finally:
                self.audio_stream = None

        print("[WakeWord] Wake word detection stopped.")
