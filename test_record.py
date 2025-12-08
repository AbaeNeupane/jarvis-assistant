import sounddevice as sd
from scipy.io.wavfile import write
sr = 16000
print("Recording 3 seconds...")
rec = sd.rec(int(3*sr), samplerate=sr, channels=1, dtype='int16')
sd.wait()
write("test_record.wav", sr, rec)
print("Saved test_record.wav")
