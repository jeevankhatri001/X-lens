import time, threading, subprocess, re

VOICE = "Tara"

def speak(text):
    print(f"[assistant] {text}")
    subprocess.run(["say", "-v", VOICE, text])

def parse_timer_command(text):
    text = text.lower()
    match = re.search(r'(\d+)', text)
    if not match:
        return None
    number = int(match.group(1))
    if "hour" in text:
        return number * 3600
    elif "minute" in text or "min" in text:
        return number * 60
    else:
        return number

def set_timer(seconds):
    def run():
        time.sleep(seconds)
        speak("Time's up!")
    threading.Thread(target=run, daemon=True).start()
    speak(f"Okay, timer set for {seconds} seconds.")

def handle_command(text):
    """Understand a spoken command and act on it."""
    seconds = parse_timer_command(text)
    if seconds:
        set_timer(seconds)
    else:
        speak("Sorry, I didn't catch a time in that.")

# --- test the full loop ---
if __name__ == "__main__":
    command = "set a timer for 5 seconds"
    print(f"Command: {command!r}")
    handle_command(command)
    time.sleep(7)   # stay alive to hear the alert