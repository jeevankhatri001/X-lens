import subprocess

# The voice your assistant speaks with
VOICE = "Tara"

def speak(text):
    """Say text out loud using the Mac's built-in TTS."""
    print(f"[assistant] {text}")          # also show it on screen
    subprocess.run(["say", "-v", VOICE, text])

# --- quick test when you run this file directly ---
if __name__ == "__main__":
    speak("Hello Jeevan. I am your smart glasses assistant, and I am ready.")