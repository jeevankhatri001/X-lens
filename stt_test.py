import whisper

print("Loading Whisper model...")
model = whisper.load_model("base")   # first run downloads ~140MB, once

print("Transcribing mic_test.wav...")
result = model.transcribe("mic_test.wav")
print("\nYou said:", result["text"].strip())