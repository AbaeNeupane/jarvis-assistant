import sounddevice as sd
import numpy as np
import openwakeword
from openwakeword.model import Model


class WakeWordDetector:
    def __init__(
        self,
        on_activation=None,
        sample_rate=16000,
        chunk_size=1280,
        threshold=0.25,
    ):
        """
        Wake word detector using OpenWakeWord's built-in 'hey jarvis' model.

        :param on_activation: callback function to call when wake word is detected
        :param sample_rate: microphone sample rate (OpenWakeWord expects 16 kHz)
        :param chunk_size: number of samples per audio chunk (1280 ≈ 80 ms)
        :param threshold: detection threshold (higher = stricter)
        """
        self.on_activation = on_activation
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold

        self.model = None
        self.audio_stream = None
        self.disabled = False

        try:
            # Ensure models are available (downloads them once if needed)
            openwakeword.utils.download_models(wakeword_models=["hey jarvis"])

            print("[WakeWord] Using built-in 'hey jarvis' model (onnx).")
            self.model = Model(
                wakeword_models=["hey jarvis"],
                inference_framework="onnx",
            )
            print("Wake Word Detector initialized with OpenWakeWord model.")
        except Exception as e:
            print(f"[WakeWord][WARN] Could not initialize wake-word model: {e}")
            print("[WakeWord][WARN] Wake-word detection disabled.")
            self.model = None
            self.disabled = True

    def _audio_callback(self, indata, frames, time, status):
        """SoundDevice callback processing audio and checking wake-word score."""
        if self.disabled or self.model is None:
            return

        if status:
            print(f"Audio stream status: {status}")

        # indata: shape (frames, channels), dtype=int16
        audio_chunk = indata[:, 0].copy()  # mono int16

        # 🔊 DEBUG: check raw audio level
        max_val = int(np.max(np.abs(audio_chunk)))
        # Comment this out later if too spammy:
        # print(f"[Audio] frames={frames}, max={max_val}", end="\r", flush=True)

        if max_val == 0:
            # pure silence, nothing to do
            return

        # Run inference
        self.model.predict(audio_chunk)

        # Inspect prediction buffer for each model
        for model_name, scores in self.model.prediction_buffer.items():
            if not scores:
                continue
            prob = scores[-1]

            # Live probabilities
            print(f"[WakeWord] {model_name}: {prob:.3f} (max={max_val})", end="\r", flush=True)

            if prob >= self.threshold:
                print(f"\n[WakeWord] Detected '{model_name}' with score {prob:.3f}")
                if self.on_activation:
                    self.on_activation()

    def start(self):
        if self.disabled or self.model is None:
            print("[WakeWord] Wake-word detector is disabled (no usable model/runtime).")
            return

        print("Starting wake word detection using OpenWakeWord...")
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self.audio_stream.start()
        print("Listening for 'Hey Jarvis'...")

    def stop(self):
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        print("Wake word detection stopped.")
