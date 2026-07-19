import time
import threading
import subprocess

VOICE = "Tara"

def speak(text):
    print(f"[assistant] {text}")
    subprocess.run(["say", "-v", VOICE, text])

def set_timer(seconds):
    """Wait in the background, then announce when time is up."""
    def run():
        time.sleep(seconds)
        speak(f"Time's up! Your {seconds} second timer is done.")
    # Run the waiting in a background thread so the program isn't frozen
    threading.Thread(target=run, daemon=True).start()
    speak(f"Okay, timer set for {seconds} seconds.")

# --- test it ---
if __name__ == "__main__":
    set_timer(5)
    print("Timer is running... (main program is still free to do other things)")
    time.sleep(7)   # keep the program alive long enough to hear the alert
    print("Done.")