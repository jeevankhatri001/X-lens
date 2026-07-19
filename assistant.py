import time, threading, subprocess, re, json, os

VOICE = "Tara"
DB_FILE = "facts.json"

def speak(text):
    print(f"[assistant] {text}")
    subprocess.run(["say", "-v", VOICE, text])

# ---------- FACTS ----------
def load_facts():
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f: return json.load(f)
    return {}

def save_facts(facts):
    with open(DB_FILE, "w") as f: json.dump(facts, f, indent=2)

# ---------- TIMER ----------
def set_timer(seconds):
    def run():
        time.sleep(seconds)
        speak("Time's up!")
    threading.Thread(target=run, daemon=True).start()
    speak(f"Okay, timer set for {seconds} seconds.")

def parse_seconds(text):
    m = re.search(r'(\d+)', text)
    if not m: return None
    n = int(m.group(1))
    if "hour" in text: return n * 3600
    if "min" in text:  return n * 60
    return n

# ---------- THE ROUTER ----------
def handle(text):
    """Look at a command and route it to the right feature."""
    t = text.lower().strip()

    # timer?
    if "timer" in t or "remind me in" in t:
        secs = parse_seconds(t)
        if secs: set_timer(secs)
        else: speak("I didn't catch a time.")
        return

    # remember a fact?
    if t.startswith("remember"):
        cleaned = re.sub(r'^remember( that)?( my)?\s+', '', t)
        if " is " in cleaned:
            key, val = cleaned.split(" is ", 1)
            facts = load_facts(); facts[key.strip()] = val.strip(); save_facts(facts)
            speak(f"Okay, I'll remember your {key.strip()} is {val.strip()}.")
        else:
            speak("I didn't catch what to remember.")
        return

    # recall a fact?
    if t.startswith("what"):
        cleaned = re.sub(r"^what('s| is)?( my)?\s+", '', t).rstrip("?")
        val = load_facts().get(cleaned.strip())
        speak(f"Your {cleaned.strip()} is {val}." if val else f"I don't know your {cleaned.strip()} yet.")
        return

    # nothing matched
    speak("Sorry, I'm not sure how to help with that yet.")

# ---------- test: type commands, it acts ----------
if __name__ == "__main__":
    speak("Assistant ready. Type a command, or 'quit' to exit.")
    while True:
        cmd = input("\nYou: ")
        if cmd.lower() in ("quit", "exit"):
            speak("Goodbye!")
            break
        handle(cmd)