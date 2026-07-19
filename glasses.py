import time, threading, subprocess, re, json, os
import requests
import sounddevice as sd
from scipy.io.wavfile import write
import whisper

VOICE = "Tara"
DB_FILE = "facts.json"
NOTES_FILE = "notes.txt"
PYTHON = "./facerec/bin/python"                      # python inside your env
DEFAULT_IMAGE = "test.jpg"                           # used if no image is named
XLENS_URL = "http://127.0.0.1:8000/api/v1/analyze"   # later: college GPU URL

# ---------- load speech-to-text model once at startup ----------
print("Loading speech model...")
stt_model = whisper.load_model("base")

def listen(seconds=5):
    """Record from the mic and return what was said as text."""
    print(f"\n🎤 Listening for {seconds} seconds...")
    audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1)
    sd.wait()
    write("command.wav", 16000, audio)
    text = stt_model.transcribe("command.wav")["text"].strip()
    print(f"You said: {text}")
    return text

def speak(text):
    print(f"[assistant] {text}")
    subprocess.run(["say", "-v", VOICE, text])

# ---------- FACTS (key = value) ----------
def load_facts():
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f:
            return json.load(f)
    return {}

def save_facts(facts):
    with open(DB_FILE, "w") as f:
        json.dump(facts, f, indent=2)

# ---------- NOTES (a growing list) ----------
def add_note(text):
    with open(NOTES_FILE, "a") as f:
        f.write(text.strip() + "\n")

def get_notes():
    if not os.path.exists(NOTES_FILE):
        return []
    with open(NOTES_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]

def clear_notes():
    if os.path.exists(NOTES_FILE):
        os.remove(NOTES_FILE)

# ---------- TIMER ----------
def set_timer(seconds):
    def run():
        time.sleep(seconds)
        speak("Time's up!")
    threading.Thread(target=run, daemon=True).start()
    speak(f"Okay, timer set for {seconds} seconds.")

def parse_seconds(text):
    m = re.search(r'(\d+)', text)
    if not m:
        return None
    n = int(m.group(1))
    if "hour" in text:
        return n * 3600
    if "min" in text:
        return n * 60
    return n

# ---------- FACE: IDENTIFY ----------
def who_is_this(text):
    m = re.search(r'(\S+\.(?:png|jpg|jpeg))', text.lower())
    image = m.group(1) if m else DEFAULT_IMAGE
    if not os.path.exists(image):
        speak(f"I can't find an image called {image}.")
        return
    speak("Let me look.")
    result = subprocess.run(
        [PYTHON, "recognize.py", "identify", image],
        capture_output=True, text=True
    )
    print(result.stdout)
    names = re.findall(r'closest:\s*(\w+),\s*score=([\d.]+)', result.stdout)
    if not names:
        speak("I don't see anyone I recognize.")
        return
    for name, score in names:
        if float(score) >= 0.40:
            speak(f"{name} is here.")
        else:
            speak("I see someone, but I don't recognize them.")

# ---------- FACE: ENROLL ----------
def remember_person(text):
    name_match = re.search(r'\bas\s+(\w+)', text.lower())
    if not name_match:
        speak("Please tell me their name. Say, remember this person as their name.")
        return
    name = name_match.group(1)
    img_match = re.search(r'(\S+\.(?:png|jpg|jpeg))', text.lower())
    image = img_match.group(1) if img_match else DEFAULT_IMAGE
    if not os.path.exists(image):
        speak(f"I can't find an image called {image}.")
        return
    speak(f"Okay, learning {name}'s face.")
    result = subprocess.run(
        [PYTHON, "recognize.py", "enroll", image, name],
        capture_output=True, text=True
    )
    print(result.stdout)
    if "Enrolled" in result.stdout:
        speak(f"Done. I'll remember {name} from now on.")
    else:
        speak("I couldn't learn that face. Try a clearer photo.")

# ---------- VISION: ASK X-LENS ----------
def ask_xlens(text):
    m = re.search(r'(\S+\.(?:png|jpg|jpeg))', text.lower())
    image = m.group(1) if m else DEFAULT_IMAGE
    if not os.path.exists(image):
        speak(f"I can't find an image called {image}.")
        return
    try:
        with open(image, "rb") as f:
            r = requests.post(XLENS_URL, files={"file": f}, timeout=30)
        data = r.json()
    except Exception as e:
        speak("I couldn't reach the vision server. Is it running?")
        print("X-Lens error:", e)
        return

    if "description" in data:
        speak(data["description"])
    elif data.get("is_acceptable"):
        speak(f"The image looks clear enough. Blur score {data['blur']:.2f}. "
              f"Vision model is off, so I can't describe it yet.")
    else:
        speak(f"That image is too low quality. Reason: {data.get('rejection_reason')}.")
    print(data)

# ---------- THE ROUTER ----------
def handle(text):
    t = text.lower().strip()

    # timer
    if "timer" in t or "remind me in" in t:
        secs = parse_seconds(t)
        if secs:
            set_timer(secs)
        else:
            speak("I didn't catch a time.")
        return

    # read back all notes  (check before other "what" / "remember" rules)
    if "what did i ask you to remember" in t or "what are my notes" in t \
            or "what did i tell you to remember" in t or "what should i remember" in t:
        notes = get_notes()
        if notes:
            speak("Here's what you asked me to remember. " + ". ".join(notes))
        else:
            speak("You haven't asked me to remember anything yet.")
        return

    # clear notes
    if "forget everything" in t or "clear my notes" in t:
        clear_notes()
        speak("Okay, I've cleared all your notes.")
        return

    # remember an arbitrary note  (check BEFORE structured "remember X is Y")
    if t.startswith("remember this") or t.startswith("note that") \
            or t.startswith("remind me that") or t.startswith("don't let me forget"):
        note = re.sub(r"^(remember this|note that|remind me that|don't let me forget)[:,]?\s*",
                      '', text.strip(), flags=re.IGNORECASE)
        add_note(note)
        speak("Okay, I've noted that down.")
        return

    # enroll a new person
    if "remember this person" in t or "remember this face" in t or t.startswith("enroll"):
        remember_person(text)
        return

    # identify a person
    if "who is this" in t or "who is that" in t or "who is in" in t or t.startswith("identify"):
        who_is_this(text)
        return

    # vision: what am I looking at
    if "what is this" in t or "what do you see" in t or "analyze" in t:
        ask_xlens(text)
        return

    # remember a structured fact (X is Y)
    if t.startswith("remember"):
        cleaned = re.sub(r'^remember( that)?( my)?\s+', '', t)
        if " is " in cleaned:
            key, val = cleaned.split(" is ", 1)
            facts = load_facts()
            facts[key.strip()] = val.strip()
            save_facts(facts)
            speak(f"Okay, I'll remember your {key.strip()} is {val.strip()}.")
        else:
            speak("I didn't catch what to remember.")
        return

    # recall a personal fact
    if t.startswith("what is my") or t.startswith("what's my"):
        cleaned = re.sub(r"^what('s| is)?( my)?\s+", '', t).rstrip("?")
        val = load_facts().get(cleaned.strip())
        speak(f"Your {cleaned.strip()} is {val}." if val else f"I don't know your {cleaned.strip()} yet.")
        return

    # fallback (later: send to the AI model)
    speak("Sorry, I'm not sure how to help with that yet.")

# ---------- MAIN LOOP (voice-controlled) ----------
if __name__ == "__main__":
    speak("Assistant ready. Press Enter, then speak your command.")
    while True:
        input("\n[Press Enter to talk, or type 'q' then Enter to quit]  ")
        cmd = listen(5)
        if cmd.strip() == "":
            speak("I didn't hear anything.")
            continue
        if "quit" in cmd.lower() or "exit" in cmd.lower():
            speak("Goodbye!")
            break
        handle(cmd)