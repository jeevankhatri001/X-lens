import sounddevice as sd
from scipy.io.wavfile import write

DURATION = 4       # seconds
SAMPLE_RATE = 16000  # Whisper likes 16kHz

print(f"Recording for {DURATION} seconds... speak now!")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()   # wait until recording finishes
write("mic_test.wav", SAMPLE_RATE, audio)
print("Saved mic_test.wav")