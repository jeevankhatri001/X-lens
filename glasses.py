"""
EDITH Assistant v2
==================
Voice + Vision assistant for your AR glasses project.

FEATURES
--------
1. SET QUESTIONS  : Predefined Q&A answered instantly (with fuzzy matching,
                    so alternative phrasings of the same question still work).
2. QWEN FALLBACK  : If the question is NOT in the set, your teammate's X-Lens
                    server (hosting Qwen2.5-VL) auto-activates and answers it,
                    over the network (Tailscale IP baked into XLENS_BASE_URL).
3. HUMAN vs OBJECT: Camera frame is analyzed. If a HUMAN is detected:
                       - If known  -> greets them by name.
                       - If unknown -> asks "Do you want me to remember this person?"
                                       If yes, asks for their name and saves the face.
                    If an OBJECT is detected and you ask "what is this",
                    it names the object (YOLO) and explains it (Qwen).
4. FACE MEMORY    : Faces are stored in faces.pkl and persist across runs.

HARDWARE MODES
--------------
- Laptop dev mode : uses your webcam + laptop mic (default).
- ESP32-S3 mode   : set CAMERA_SOURCE to your ESP32 MJPEG stream URL,
                    e.g. "http://192.168.1.50:81/stream"

INSTALL
-------
pip install opencv-python face_recognition ultralytics SpeechRecognition pyttsx3 requests pyaudio

(face_recognition needs dlib. On Windows: pip install cmake, then dlib.
 On Linux: sudo apt install build-essential cmake before pip install dlib.)

For Qwen: no local install needed — it's already running on Bashant's
X-Lens server. Just make sure your Mac can reach XLENS_BASE_URL (same
Tailscale network / LAN). If it's unreachable, ask_qwen() prints the
exact connection error to your terminal.

RUN
---
python edith_assistant.py
Say "EDITH" followed by your question, e.g.:
    "EDITH what is my schedule today"        -> set answer
    "EDITH what's the capital of Mongolia"   -> Qwen fallback
    "EDITH who is this"                      -> face recognition / remember flow
    "EDITH what is this"                     -> object detection + explanation
    "EDITH forget John"                      -> deletes a saved face
    "EDITH shutdown"                         -> exit
"""

import os
import re
import sys
import time
import subprocess
import pickle
import difflib
import threading
import queue

import cv2
import requests
import numpy as np
import pyttsx3
import speech_recognition as sr

# face_recognition and YOLO are heavy; import with friendly errors
try:
    import face_recognition
    FACE_OK = True
except ImportError:
    FACE_OK = False
    print("[WARN] face_recognition not installed -> face features disabled")

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[WARN] ultralytics not installed -> object detection disabled")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
WAKE_WORD = "hi"
# Google STT frequently mis-hears short wake words — add more here
# as you notice your own mis-transcriptions (print statements show you "YOU: ...").
# IMPORTANT: keep every entry lowercase — heard text is always lowercased,
# so a mismatched case here means the wake word silently never matches.
WAKE_WORD_VARIANTS = ["hi", "hey", "hey edith", "hi edith", "high"]


def extract_wake_command(heard: str):
    """Returns the command text after the wake word, or None if wake word
    isn't present (checked both by substring and by fuzzy first-word match)."""
    for variant in WAKE_WORD_VARIANTS:
        if variant in heard:
            return heard.split(variant, 1)[1].strip(" ,.")
    # Fuzzy fallback: check if the first word sounds like the wake word
    words = heard.split()
    if words and difflib.get_close_matches(words[0], [WAKE_WORD], n=1, cutoff=0.5):
        return " ".join(words[1:]).strip(" ,.")
    return None

# 0 = laptop webcam.  For ESP32-S3 CAM use its MJPEG stream URL string.
CAMERA_SOURCE = 0
# CAMERA_SOURCE = "http://192.168.1.50:81/stream"

FACE_DB_PATH = "faces.pkl"
YOLO_MODEL = "yolov8n.pt"          # nano model, fast on CPU

XLENS_BASE_URL = "http://100.111.3.2:8000"  # Bashant's X-Lens server (Tailscale IP)
XLENS_ASK_URL = f"{XLENS_BASE_URL}/api/v1/ask"          # text-only Q&A
XLENS_CHAT_URL = f"{XLENS_BASE_URL}/api/v1/chat"        # image + question
XLENS_ANALYZE_URL = f"{XLENS_BASE_URL}/api/v1/analyze"  # image only

FUZZY_THRESHOLD = 0.72             # how close a question must be to a set question

# ----------------------------------------------------------------------
# 1) SET QUESTIONS  (edit these freely — add as many as you want)
#    key   = canonical question
#    value = (list_of_alternative_phrasings, answer)
# ----------------------------------------------------------------------
SET_QA = {
    "what is your name": (
        ["who are you", "tell me your name", "what should i call you"],
        "I am EDITH — Even Dead I'm The Hero. Your personal assistant, Jeevan.",
    ),
    "who made you": (
        ["who created you", "who is your creator", "who built you"],
        "I was built by Jeevan as part of the EDITH AR glasses project.",
    ),
    "what time is it": (
        ["tell me the time", "current time", "time now"],
        None,  # None = computed live, see answer_set_question()
    ),
    "what is my schedule today": (
        ["what do i have today", "my plan for today", "today's schedule"],
        "Your schedule: college in the morning, gym in the evening. "
        "Edit SET_QA in the code to change this.",
    ),
    "what can you do": (
        ["what are your features", "help", "what are your abilities"],
        "I can answer your set questions, use Qwen for anything else, "
        "recognize and remember people, and explain objects I see.",
    ),
}


def build_phrase_index():
    """Flatten canonical questions + alternatives into one lookup list."""
    index = []
    for canonical, (alts, _ans) in SET_QA.items():
        index.append((canonical, canonical))
        for alt in alts:
            index.append((alt, canonical))
    return index


PHRASE_INDEX = build_phrase_index()


def match_set_question(text: str):
    """Fuzzy-match user text against set questions and their alternatives.
    Returns canonical key or None."""
    text = text.lower().strip()
    phrases = [p for p, _ in PHRASE_INDEX]
    best = difflib.get_close_matches(text, phrases, n=1, cutoff=FUZZY_THRESHOLD)
    if best:
        for phrase, canonical in PHRASE_INDEX:
            if phrase == best[0]:
                return canonical
    return None


def answer_set_question(canonical: str) -> str:
    _alts, ans = SET_QA[canonical]
    if ans is not None:
        return ans
    # dynamic answers
    if canonical == "what time is it":
        return "It is " + time.strftime("%I:%M %p")
    return "I know that question but have no answer configured."


# ----------------------------------------------------------------------
# 2) QWEN FALLBACK (auto-activates when question is not in the set)
# ----------------------------------------------------------------------
def ask_qwen(prompt: str, image_bgr_frame=None) -> str:
    """Calls your teammate's X-Lens server (hosting Qwen2.5-VL), per its real
    OpenAPI schema:
      - text-only questions  -> POST /api/v1/ask   {"question": "..."} (2-300 chars)
      - image + question     -> POST /api/v1/chat  multipart: file=<jpg>, question="..."
    """
    try:
        if image_bgr_frame is None:
            # AskRequest requires 2-300 chars — trim to be safe
            question = prompt.strip()[:300]
            if len(question) < 2:
                return "That question was too short for me to send."
            r = requests.post(XLENS_ASK_URL, json={"question": question}, timeout=60)
        else:
            _ok, buf = cv2.imencode(".jpg", image_bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            data = {"question": prompt.strip()[:300]} if prompt else {}
            r = requests.post(XLENS_CHAT_URL, files=files, data=data, timeout=60)

        if r.ok:
            body = r.json()
            for key in ("answer", "response", "message", "text", "result"):
                if isinstance(body, dict) and key in body:
                    return str(body[key]).strip()
            return str(body).strip()
        else:
            print(f"[X-Lens HTTP error] status={r.status_code} body={r.text[:500]}")
    except requests.RequestException as e:
        print(f"[X-Lens connection error] {type(e).__name__}: {e}")
    return "Sorry, I could not reach the model right now."


# ----------------------------------------------------------------------
# TIMERS (handled locally — Qwen has no access to your device's clock,
# so this must never be routed to ask_qwen)
# ----------------------------------------------------------------------
_TIMER_UNIT_SECONDS = {"second": 1, "seconds": 1, "sec": 1, "secs": 1,
                       "minute": 60, "minutes": 60, "min": 60, "mins": 60,
                       "hour": 3600, "hours": 3600}


def parse_timer_seconds(text: str):
    """Parses phrases like 'set a timer for 10 seconds', 'timer for 2 minutes',
    or 'start a countdown from 14' -> total seconds (int), or None if no match."""
    if not any(w in text for w in ("timer", "alarm", "countdown")):
        return None
    total = 0
    found = False
    for amount, unit in re.findall(r"(\d+)\s*([a-z]+)", text):
        unit = unit.lower()
        if unit in _TIMER_UNIT_SECONDS:
            total += int(amount) * _TIMER_UNIT_SECONDS[unit]
            found = True
    if found:
        return total
    # no unit attached (e.g. "countdown from 1 to 10", "timer 30") -> assume
    # seconds, and take the LAST number, since phrasing like "from X to Y"
    # means Y is the actual target duration, not X.
    bare_numbers = re.findall(r"\d+", text)
    if bare_numbers:
        return int(bare_numbers[-1])
    return None


# ----------------------------------------------------------------------
# 3) SPEECH (STT + TTS)
# ----------------------------------------------------------------------
class Voice:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.rec.energy_threshold = 300
        self.rec.dynamic_energy_threshold = True
        # CRITICAL: without this, a slow/unstable network makes recognize_google()
        # hang forever with no exception raised (this is what froze your run).
        self.rec.operation_timeout = 8
        self.tts_lock = threading.Lock()

    def say(self, text: str):
        print(f"EDITH: {text}")
        with self.tts_lock:
            if sys.platform == "darwin":
                # macOS: shell out to the native 'say' command. This runs in
                # its own OS process, so it works correctly even when called
                # from a background thread (e.g. the timer) — pyttsx3's
                # Cocoa speech engine expects the main thread and can fail
                # silently (prints fine, produces no audio) when it isn't.
                try:
                    subprocess.run(["say", "-r", "190", text], check=False)
                except FileNotFoundError:
                    print("[TTS error] 'say' command not found — falling back to pyttsx3")
                    self._pyttsx3_say(text)
            else:
                self._pyttsx3_say(text)

    @staticmethod
    def _pyttsx3_say(text: str):
        # KNOWN pyttsx3 bug: reusing one engine across multiple runAndWait()
        # calls goes silent after the first utterance. Fresh engine per call.
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def listen(self, timeout=6, phrase_limit=10) -> str:
        with sr.Microphone() as mic:
            self.rec.adjust_for_ambient_noise(mic, duration=0.4)
            try:
                audio = self.rec.listen(mic, timeout=timeout,
                                        phrase_time_limit=phrase_limit)
            except sr.WaitTimeoutError:
                return ""
        try:
            text = self.rec.recognize_google(audio)   # free Google STT
            print(f"YOU: {text}")
            return text.lower()
        except (sr.UnknownValueError, sr.RequestError):
            return ""
        except Exception as e:
            # Catches socket/timeout errors from operation_timeout expiring,
            # so a bad network moment never freezes the whole assistant.
            print(f"[STT error, skipping] {e}")
            return ""


# ----------------------------------------------------------------------
# 4) VISION: camera thread, human vs object, face memory
# ----------------------------------------------------------------------
class Camera:
    """Grabs frames in a background thread so the latest frame is always ready."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.2)

    def snapshot(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


class FaceMemory:
    def __init__(self, path):
        self.path = path
        self.names, self.encodings = [], []
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.names = data["names"]
                self.encodings = data["encodings"]

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump({"names": self.names, "encodings": self.encodings}, f)

    def identify(self, frame):
        """Returns list of names found; 'UNKNOWN' for unrecognized humans."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        results = []
        for enc in encs:
            name = "UNKNOWN"
            if self.encodings:
                dists = face_recognition.face_distance(self.encodings, enc)
                best = int(np.argmin(dists))
                if dists[best] < 0.55:
                    name = self.names[best]
            results.append((name, enc))
        return results

    def remember(self, name, encoding):
        self.names.append(name)
        self.encodings.append(encoding)
        self.save()

    def forget(self, name):
        keep = [(n, e) for n, e in zip(self.names, self.encodings)
                if n.lower() != name.lower()]
        removed = len(self.names) - len(keep)
        self.names = [n for n, _ in keep]
        self.encodings = [e for _, e in keep]
        self.save()
        return removed


def frame_to_b64(frame) -> str:
    import base64
    _ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


# ----------------------------------------------------------------------
# 5) MAIN ASSISTANT LOGIC
# ----------------------------------------------------------------------
class Edith:
    def __init__(self):
        self.voice = Voice()
        self.camera = Camera(CAMERA_SOURCE)
        self.faces = FaceMemory(FACE_DB_PATH) if FACE_OK else None
        self.yolo = YOLO(YOLO_MODEL) if YOLO_OK else None
        self.greeted = set()   # people greeted this session

    # ---------- vision helpers ----------
    def detect_scene(self, frame):
        """Differentiate human vs object.
        Returns ('human', faces) or ('object', labels) or ('empty', None)."""
        if self.faces:
            found = self.faces.identify(frame)
            if found:
                return "human", found
        if self.yolo:
            res = self.yolo(frame, verbose=False)[0]
            labels = []
            for box in res.boxes:
                label = self.yolo.names[int(box.cls)]
                conf = float(box.conf)
                if conf > 0.45 and label != "person":
                    labels.append(label)
            if labels:
                return "object", list(dict.fromkeys(labels))
            # YOLO saw a person but face_recognition found no face (turned away etc.)
            if any(self.yolo.names[int(b.cls)] == "person" for b in res.boxes):
                return "human", []
        return "empty", None

    def handle_who_is_this(self):
        frame = self.camera.snapshot()
        if frame is None:
            self.voice.say("My camera is not giving me a picture.")
            return
        kind, data = self.detect_scene(frame)
        if kind != "human":
            self.voice.say("I don't see a person right now. I see " +
                           (", ".join(data) if data else "nothing I recognize."))
            return
        if not data:
            self.voice.say("I see a person but I can't see their face clearly.")
            return
        for name, enc in data:
            if name != "UNKNOWN":
                self.voice.say(f"That is {name}.")
            else:
                self.voice.say("I don't know this person. "
                               "Do you want me to remember them? Say yes or no.")
                reply = self.voice.listen(timeout=6)
                if "yes" in reply:
                    self.voice.say("What is their name?")
                    name_reply = self.voice.listen(timeout=6).strip().title()
                    if name_reply:
                        self.faces.remember(name_reply, enc)
                        self.voice.say(f"Got it. I will remember {name_reply}.")
                    else:
                        self.voice.say("I didn't catch the name, so I won't save them.")
                else:
                    self.voice.say("Okay, I won't remember them.")

    def handle_what_is_this(self, want_explanation=True):
        frame = self.camera.snapshot()
        if frame is None:
            self.voice.say("My camera is not giving me a picture.")
            return
        kind, data = self.detect_scene(frame)
        if kind == "human":
            names = [n for n, _ in data if n != "UNKNOWN"]
            if names:
                self.voice.say("That's a person — " + ", ".join(names) + ".")
            else:
                self.voice.say("That's a person, but I don't recognize them. "
                               "Ask me 'who is this' if you want me to remember them.")
            return
        if kind == "object" and data:
            self.voice.say("I see " + ", ".join(data) + ".")
            if want_explanation:
                explanation = ask_qwen(
                    f"In 2 short sentences, explain what a {data[0]} is and "
                    f"what it's used for.")
                self.voice.say(explanation)
            return
        # nothing from YOLO -> let Qwen-VL look at the raw image
        answer = ask_qwen("Look at this image and tell me the main thing you see, "
                          "in one short sentence.", image_bgr_frame=frame)
        self.voice.say(answer)

    SPOKEN_COUNTDOWN_MAX = 15  # above this, speaking every number aloud gets tedious

    def start_timer(self, seconds: int):
        """Runs in a background thread so the assistant keeps listening
        in between numbers. Short durations get a real spoken countdown
        (10, 9, 8, ... Time's up!); long ones get a silent wait + one
        announcement at the end, so it doesn't talk for minutes straight."""
        def _spoken_countdown():
            for n in range(seconds, 0, -1):
                self.voice.say(str(n))
            self.voice.say("Time's up!")

        def _silent_wait():
            time.sleep(seconds)
            self.voice.say("Time's up!")

        if seconds <= self.SPOKEN_COUNTDOWN_MAX:
            threading.Thread(target=_spoken_countdown, daemon=True).start()
        else:
            threading.Thread(target=_silent_wait, daemon=True).start()
            if seconds >= 60:
                mins, secs = divmod(seconds, 60)
                phrase = f"{mins} minute" + ("s" if mins != 1 else "")
                if secs:
                    phrase += f" {secs} second" + ("s" if secs != 1 else "")
            else:
                phrase = f"{seconds} second" + ("s" if seconds != 1 else "")
            self.voice.say(f"Timer set for {phrase}.")

    def auto_greet(self):
        """Passively greet known people when they appear (EDITH-style)."""
        if not self.faces:
            return
        frame = self.camera.snapshot()
        if frame is None:
            return
        for name, _enc in self.faces.identify(frame):
            if name != "UNKNOWN" and name not in self.greeted:
                self.greeted.add(name)
                self.voice.say(f"Hello {name}.")

    # ---------- main answer routing ----------
    def route(self, text: str) -> bool:
        """Returns False when it's time to shut down."""
        if not text:
            return True
        if "shutdown" in text or "shut down" in text:
            self.voice.say("Shutting down. Goodbye Jeevan.")
            return False
        if "forget" in text and self.faces:
            name = text.split("forget", 1)[1].strip().title()
            n = self.faces.forget(name) if name else 0
            self.voice.say(f"Removed {n} saved face for {name}." if n
                           else f"I don't have anyone named {name} saved.")
            return True
        timer_seconds = parse_timer_seconds(text)
        if timer_seconds is not None:
            if timer_seconds <= 0:
                self.voice.say("I need a positive amount of time for the timer.")
            else:
                self.start_timer(timer_seconds)
            return True
        if any(w in text for w in ("timer", "alarm", "countdown")):
            # keyword present but no number caught -> ask instead of
            # handing this off to Qwen, which has no concept of timers
            self.voice.say("For how long, in seconds?")
            reply = self.voice.listen(timeout=8)
            reply_numbers = re.findall(r"\d+", reply) if reply else []
            if reply_numbers:
                self.start_timer(int(reply_numbers[-1]))
            else:
                self.voice.say("I didn't catch a number, so I didn't start a timer.")
            return True
        if "who is this" in text or "who is that" in text or "who am i looking at" in text:
            self.handle_who_is_this()
            return True
        if "what is this" in text or "what is that" in text or "what am i looking at" in text:
            self.handle_what_is_this()
            return True

        # 1) set questions (with alternatives, fuzzy matched)
        canonical = match_set_question(text)
        if canonical:
            self.voice.say(answer_set_question(canonical))
            return True

        # 2) not in the set -> Qwen auto-activates
        self.voice.say(ask_qwen(text))
        return True

    def run(self):
        self.voice.say("EDITH online. Say my name followed by your question.")
        running = True
        while running:
            self.auto_greet()
            heard = self.voice.listen(timeout=5)
            if not heard:
                continue
            q = extract_wake_command(heard)
            if q is not None:
                if not q:
                    self.voice.say("Yes?")
                    q = self.voice.listen(timeout=6)
                running = self.route(q)
        self.camera.stop()


if __name__ == "__main__":
    Edith().run()