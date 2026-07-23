
"""
EDITH Assistant v3 — Personal Secretary Edition
===============================================
Voice + Vision assistant for your AR glasses project.

EVERYTHING FROM v2 (all tested working):
- Set questions with alternatives + fuzzy matching
- X-Lens (Qwen2.5-VL on Bashant's server) auto-fallback for anything else
- Human vs object differentiation (face_recognition + YOLO)
- Remember/forget people by face
- Spoken countdown timers ("start a countdown from 1 to 10")
- macOS-native TTS via `say` (thread-safe), STT timeouts

NEW IN v3 (secretary features):
- NOTES     : "hi take a note" -> asks what -> saves to notes.txt with timestamp
              "hi read my notes" -> reads the recent ones aloud
- REMINDERS : "hi remind me tomorrow that I have a class" -> saved to
              reminders.json; due reminders are announced automatically
              "hi what are my reminders" -> reads today's + upcoming
- PROFILES  : people are now full profiles, not just faces.
              "hi remember this person" -> asks name, saves face photo + encoding
              "hi add Bashant's phone number" -> asks, saves
              (same for email / address)
              "hi show me Bashant's profile" -> speaks everything saved
              "hi forget Bashant" -> deletes the whole profile
- CAMERA    : "hi open camera and identify the object" == "what is this"

FILES IT CREATES (all next to glasses.py, easy to open/edit by hand):
- notes.txt        plain text, one timestamped note per line
- reminders.json   your reminders
- people.pkl       face encodings + contact details
- people_photos/   one saved JPG per remembered person

INSTALL (same as before)
------------------------
pip install opencv-python face_recognition ultralytics SpeechRecognition pyttsx3 requests pyaudio

Qwen: no local install — it runs on Bashant's X-Lens server (XLENS_BASE_URL).

RUN
---
python glasses.py       then say "hi ..." :
  "hi who are you"                          -> set answer
  "hi what is 7 times 23"                   -> Qwen fallback
  "hi who is this"                          -> face recognition
  "hi remember this person"                 -> save new profile (photo + name)
  "hi add <name>'s phone number"            -> add contact detail
  "hi show me <name>'s profile"             -> read profile aloud
  "hi what is this"                         -> object detection + explanation
  "hi take a note"                          -> save a note
  "hi read my notes"                        -> hear recent notes
  "hi remind me tomorrow that ..."          -> save reminder
  "hi what are my reminders"                -> hear reminders
  "hi start a countdown from 1 to 10"       -> spoken countdown
  "hi forget <name>"                        -> delete a profile
  "hi shutdown"                             -> exit
"""

import os
import re
import sys
import time
import json
import pickle
import difflib
import datetime
import threading
import subprocess

import cv2
import requests
import numpy as np
import pyttsx3
import speech_recognition as sr

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
# Keep every entry lowercase — heard text is always lowercased.
WAKE_WORD_VARIANTS = ["hi", "hey", "hey edith", "hi edith", "high"]

# 0 = laptop webcam.  For your Xiao ESP32-S3 CAM use its MJPEG stream URL.
CAMERA_SOURCE = 0
# CAMERA_SOURCE = "http://192.168.1.50:81/stream"

XLENS_BASE_URL = "http://100.111.3.2:8000"  # Bashant's X-Lens server (Tailscale IP)
XLENS_ASK_URL = f"{XLENS_BASE_URL}/api/v1/ask"          # text-only Q&A
XLENS_CHAT_URL = f"{XLENS_BASE_URL}/api/v1/chat"        # image + question

PEOPLE_DB_PATH = "people.pkl"
LEGACY_FACES_PATH = "faces.pkl"          # v2 format, auto-migrated if found
PHOTOS_DIR = "people_photos"
NOTES_PATH = "notes.txt"
REMINDERS_PATH = "reminders.json"

YOLO_MODEL = "yolov8n.pt"
FUZZY_THRESHOLD = 0.72

# Your home area — used as the reference point for "how far is X" and
# "restaurants around me". Edit these to your actual neighborhood.
HOME_PLACE = "Bhaktapur, Nepal"
HOME_LAT, HOME_LON = 27.6710, 85.4298
# Restrict all map searches to this country (ISO code). "np" = Nepal.
# Set to "" to search worldwide, or e.g. "np,in" for Nepal + India.
COUNTRY_CODE = "np"

# ----------------------------------------------------------------------
# 1) SET QUESTIONS
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
        None,
    ),
    "what is my schedule today": (
        ["what do i have today", "my plan for today", "today's schedule"],
        "Your schedule: college in the morning, gym in the evening. "
        "Edit SET_QA in the code to change this.",
    ),
    "what can you do": (
        ["what are your features", "help", "what are your abilities"],
        "I can answer questions, take notes, set reminders and timers, "
        "recognize and remember people with their contact details, "
        "and explain objects I see.",
    ),
}


def build_phrase_index():
    index = []
    for canonical, (alts, _ans) in SET_QA.items():
        index.append((canonical, canonical))
        for alt in alts:
            index.append((alt, canonical))
    return index


PHRASE_INDEX = build_phrase_index()


def match_set_question(text: str):
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
    if canonical == "what time is it":
        return "It is " + time.strftime("%I:%M %p")
    return "I know that question but have no answer configured."


def extract_wake_command(heard: str):
    """Returns command text after the wake word, or None if absent."""
    for variant in WAKE_WORD_VARIANTS:
        if variant in heard:
            return heard.split(variant, 1)[1].strip(" ,.")
    words = heard.split()
    if words and difflib.get_close_matches(words[0], [WAKE_WORD], n=1, cutoff=0.5):
        return " ".join(words[1:]).strip(" ,.")
    return None


# ----------------------------------------------------------------------
# 2) X-LENS / QWEN
# ----------------------------------------------------------------------
def ask_qwen(prompt: str, image_bgr_frame=None) -> str:
    """text-only -> POST /api/v1/ask  {"question": ...} (2-300 chars)
       image+question -> POST /api/v1/chat  multipart file + question"""
    try:
        if image_bgr_frame is None:
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
        print(f"[X-Lens HTTP error] status={r.status_code} body={r.text[:500]}")
    except requests.RequestException as e:
        print(f"[X-Lens connection error] {type(e).__name__}: {e}")
    return "Sorry, I could not reach the model right now."


# ----------------------------------------------------------------------
# 2b) LOCATIONS (OpenStreetMap Nominatim — free, no API key) and
#     IMAGE FILES on disk
# ----------------------------------------------------------------------
import math

_NOMINATIM_HEADERS = {"User-Agent": "EDITH-AR-glasses/1.0 (student project)"}


def _nominatim_get(params, tries=2, timeout=25):
    """One retry on timeout — Nominatim is a free service and can be slow,
    especially over long-haul connections. Returns response JSON or the
    string 'TIMEOUT' so callers can tell 'slow service' from 'not found'."""
    params = dict(params)
    if COUNTRY_CODE:
        params["countrycodes"] = COUNTRY_CODE
    for attempt in range(tries):
        try:
            r = requests.get("https://nominatim.openstreetmap.org/search",
                             params=params, headers=_NOMINATIM_HEADERS,
                             timeout=timeout)
            if r.ok:
                return r.json()
            print(f"[Nominatim HTTP {r.status_code}] {r.text[:200]}")
            return None
        except requests.Timeout:
            print(f"[Nominatim timeout, attempt {attempt + 1}/{tries}]")
        except requests.RequestException as e:
            print(f"[Nominatim error] {e}")
            return None
    return "TIMEOUT"


def geocode(query: str):
    """Place name -> (lat, lon, display_name), 'TIMEOUT', or None (not found).
    Restricted to COUNTRY_CODE so 'civil hospital' finds the Nepali one."""
    result = _nominatim_get({"q": query, "format": "json", "limit": 1})
    if result == "TIMEOUT":
        return "TIMEOUT"
    if result:
        hit = result[0]
        return float(hit["lat"]), float(hit["lon"]), hit["display_name"]
    return None


def nearby_places(kind: str, n=5):
    """e.g. kind='restaurant' near HOME_PLACE -> list of names, or 'TIMEOUT'."""
    result = _nominatim_get({"q": f"{kind} near {HOME_PLACE}",
                             "format": "json", "limit": n})
    if result == "TIMEOUT":
        return "TIMEOUT"
    if result:
        return [hit["display_name"].split(",")[0] for hit in result]
    return []


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


IMAGE_WORDS = ("png", "jpg", "jpeg", "photo", "image", "picture", "pic")


def find_image_file(spoken: str):
    """Fuzzy-match a spoken filename ('o2 dot png', 'photo png') against
    actual image files in the current directory. STT mangles filenames,
    so we normalize and use closest-match."""
    files = [f for f in os.listdir(".")
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        return None, []
    norm = spoken.lower().replace(" dot ", ".").replace("dot ", ".")
    norm = re.sub(r"[^a-z0-9.]", "", norm)
    stems = {os.path.splitext(f)[0].lower(): f for f in files}
    for stem, f in stems.items():
        if stem and (stem in norm or norm.replace(".png", "").replace(".jpg", "")
                     .replace(".jpeg", "") == stem):
            return f, files
    close = difflib.get_close_matches(
        norm, list(stems) + [f.lower() for f in files], n=1, cutoff=0.4)
    if close:
        key = close[0]
        return stems.get(key, key if key in files else
                         next((f for f in files if f.lower() == key), None)), files
    return None, files


# ----------------------------------------------------------------------
# 3) TIMERS
# ----------------------------------------------------------------------
_TIMER_UNIT_SECONDS = {"second": 1, "seconds": 1, "sec": 1, "secs": 1,
                       "minute": 60, "minutes": 60, "min": 60, "mins": 60,
                       "hour": 3600, "hours": 3600}


def parse_timer_seconds(text: str):
    if not any(w in text for w in ("timer", "alarm", "countdown")):
        return None
    total, found = 0, False
    for amount, unit in re.findall(r"(\d+)\s*([a-z]+)", text):
        if unit.lower() in _TIMER_UNIT_SECONDS:
            total += int(amount) * _TIMER_UNIT_SECONDS[unit.lower()]
            found = True
    if found:
        return total
    bare = re.findall(r"\d+", text)
    if bare:
        return int(bare[-1])   # "from 1 to 10" -> 10 (last number is the target)
    return None


# ----------------------------------------------------------------------
# 4) NOTES  (notes.txt — plain text, human-readable, easy to open anywhere)
# ----------------------------------------------------------------------
class Notes:
    def __init__(self, path):
        self.path = path

    def add(self, text: str):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {text}\n")

    def recent(self, n=5):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return lines[-n:]


# ----------------------------------------------------------------------
# 5) REMINDERS  (reminders.json — announced automatically when due)
# ----------------------------------------------------------------------
class Reminders:
    def __init__(self, path):
        self.path = path
        self.items = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.items = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.items = []

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, indent=2)

    def add(self, text: str, due_date: datetime.date):
        self.items.append({"text": text, "due": due_date.isoformat(),
                           "announced": False})
        self.save()

    def due_now(self):
        """Reminders due today (or overdue) that haven't been announced."""
        today = datetime.date.today().isoformat()
        out = [it for it in self.items
               if it["due"] <= today and not it.get("announced")]
        return out

    def mark_announced(self, item):
        item["announced"] = True
        self.save()

    def upcoming(self, n=5):
        return sorted(self.items, key=lambda it: it["due"])[:n]


def parse_reminder(text: str):
    """'remind me tomorrow that i have a class' ->
       ('i have a class', date(tomorrow)). Returns None if not a reminder."""
    if "remind" not in text:
        return None
    today = datetime.date.today()
    due = today
    if "tomorrow" in text:
        due = today + datetime.timedelta(days=1)
    elif "next week" in text:
        due = today + datetime.timedelta(days=7)
    else:
        m = re.search(r"in (\d+) days?", text)
        if m:
            due = today + datetime.timedelta(days=int(m.group(1)))
    # extract what to remind: text after "that"/"to", minus time words
    content = text
    for marker in (" that ", " to "):
        if marker in content:
            content = content.split(marker, 1)[1]
            break
    else:
        content = content.split("remind", 1)[1].lstrip(" me")
    for w in ("tomorrow", "next week"):
        content = content.replace(w, "")
    content = re.sub(r"in \d+ days?", "", content).strip(" ,.")
    if not content:
        return None
    return content, due


# ----------------------------------------------------------------------
# 6) SPEECH (STT + TTS)
# ----------------------------------------------------------------------
class Voice:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.rec.energy_threshold = 300
        self.rec.dynamic_energy_threshold = True
        # Without this, a slow network makes recognize_google() hang forever.
        self.rec.operation_timeout = 8
        self.tts_lock = threading.Lock()

    def say(self, text: str):
        print(f"EDITH: {text}")
        with self.tts_lock:
            if sys.platform == "darwin":
                # macOS: native `say` runs in its own process, so it works
                # from any thread. pyttsx3's Cocoa engine goes silent when
                # called off the main thread (prints fine, no audio).
                try:
                    subprocess.run(["say", "-r", "190", text], check=False)
                    return
                except FileNotFoundError:
                    print("[TTS] 'say' not found — falling back to pyttsx3")
            # non-macOS (or fallback): fresh engine per call — reusing one
            # engine goes silent after the first runAndWait().
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
            text = self.rec.recognize_google(audio)
            print(f"YOU: {text}")
            return text.lower()
        except (sr.UnknownValueError, sr.RequestError):
            return ""
        except Exception as e:
            print(f"[STT error, skipping] {e}")
            return ""


# ----------------------------------------------------------------------
# 7) CAMERA
# ----------------------------------------------------------------------
class Camera:
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


# ----------------------------------------------------------------------
# 8) PEOPLE PROFILES  (face encoding + photo + contact details)
# ----------------------------------------------------------------------
class PeopleMemory:
    """people.pkl format:
       { "Name": { "encoding": np.array, "photo": "people_photos/name.jpg",
                   "phone": "...", "email": "...", "address": "..." } }
    Auto-migrates the old v2 faces.pkl {"names": [...], "encodings": [...]}."""

    CONTACT_FIELDS = ("phone", "email", "address")

    def __init__(self, path):
        self.path = path
        self.people = {}
        os.makedirs(PHOTOS_DIR, exist_ok=True)
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.people = pickle.load(f)
        elif os.path.exists(LEGACY_FACES_PATH):
            with open(LEGACY_FACES_PATH, "rb") as f:
                old = pickle.load(f)
            for name, enc in zip(old.get("names", []), old.get("encodings", [])):
                self.people[name] = {"encoding": enc, "photo": None,
                                     "phone": None, "email": None, "address": None}
            self.save()
            print(f"[People] migrated {len(self.people)} entries from {LEGACY_FACES_PATH}")

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.people, f)

    # ---- face matching ----
    def identify(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        known_names = list(self.people.keys())
        known_encs = [self.people[n]["encoding"] for n in known_names]
        results = []
        for enc in encs:
            name = "UNKNOWN"
            if known_encs:
                dists = face_recognition.face_distance(known_encs, enc)
                best = int(np.argmin(dists))
                if dists[best] < 0.55:
                    name = known_names[best]
            results.append((name, enc))
        return results

    # ---- profile ops ----
    def remember(self, name, encoding, frame=None):
        photo_path = None
        if frame is not None:
            safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "person"
            photo_path = os.path.join(PHOTOS_DIR, f"{safe}.jpg")
            cv2.imwrite(photo_path, frame)
        self.people[name] = {"encoding": encoding, "photo": photo_path,
                             "phone": None, "email": None, "address": None}
        self.save()

    def find_name(self, spoken: str):
        """Match a spoken name against saved names (fuzzy, case-insensitive)."""
        if not self.people:
            return None
        names = list(self.people.keys())
        lowered = {n.lower(): n for n in names}
        if spoken.lower() in lowered:
            return lowered[spoken.lower()]
        close = difflib.get_close_matches(spoken.lower(), list(lowered), n=1, cutoff=0.6)
        return lowered[close[0]] if close else None

    def set_field(self, name, field, value):
        if name in self.people and field in self.CONTACT_FIELDS:
            self.people[name][field] = value
            self.save()
            return True
        return False

    def profile_text(self, name):
        p = self.people.get(name)
        if not p:
            return None
        parts = [f"{name}."]
        for field in self.CONTACT_FIELDS:
            if p.get(field):
                parts.append(f"{field.capitalize()}: {p[field]}.")
        if len(parts) == 1:
            parts.append("No contact details saved yet.")
        if p.get("photo"):
            parts.append(f"Photo saved at {p['photo']}.")
        return " ".join(parts)

    def forget(self, name):
        if name in self.people:
            photo = self.people[name].get("photo")
            if photo and os.path.exists(photo):
                os.remove(photo)
            del self.people[name]
            self.save()
            return 1
        return 0


# ----------------------------------------------------------------------
# 9) MAIN ASSISTANT
# ----------------------------------------------------------------------
class Edith:
    SPOKEN_COUNTDOWN_MAX = 15

    def __init__(self):
        self.voice = Voice()
        self.camera = Camera(CAMERA_SOURCE)
        self.people = PeopleMemory(PEOPLE_DB_PATH) if FACE_OK else None
        self.yolo = YOLO(YOLO_MODEL) if YOLO_OK else None
        self.notes = Notes(NOTES_PATH)
        self.reminders = Reminders(REMINDERS_PATH)
        self.greeted = set()

    # ---------- helpers ----------
    @staticmethod
    def clean_reply(raw: str) -> str:
        """People often prefix follow-up answers with the wake word out of
        habit ('hi tomorrow'). Strip it so it doesn't pollute saved content."""
        if not raw:
            return ""
        stripped = extract_wake_command(raw)
        return (stripped if stripped is not None else raw).strip(" ,.")

    # ---------- vision ----------
    def detect_scene(self, frame):
        # DEBUG: save exactly what EDITH analyzed, so when it says something
        # wrong ("I see refrigerator" in an empty room) you can open
        # last_seen.jpg and check what the camera actually captured —
        # wrong camera index, bad angle, and low light all show up here.
        cv2.imwrite("last_seen.jpg", frame)
        if self.people:
            found = self.people.identify(frame)
            if found:
                return "human", found
        if self.yolo:
            res = self.yolo(frame, verbose=False)[0]
            labels = []
            for box in res.boxes:
                label = self.yolo.names[int(box.cls)]
                conf = float(box.conf)
                print(f"[YOLO] {label} conf={conf:.2f}")
                # 0.60: yolov8n at lower thresholds hallucinates furniture
                # (the phantom refrigerator was exactly this)
                if conf > 0.60 and label != "person":
                    labels.append(label)
            if labels:
                return "object", list(dict.fromkeys(labels))
            if any(self.yolo.names[int(b.cls)] == "person" and float(b.conf) > 0.60
                   for b in res.boxes):
                return "human", []
        return "empty", None

    def enroll_person(self, encoding, frame):
        """Shared flow: ask for name, save profile, offer contact details."""
        self.voice.say("Under which name should I remember them?")
        name = self.clean_reply(self.voice.listen(timeout=8)).title()
        if not name:
            self.voice.say("I didn't catch the name, so I won't save them.")
            return
        self.people.remember(name, encoding, frame)
        self.voice.say(f"Got it. I saved {name} with their photo. "
                       "Do you want to add a phone number now? Say yes or no.")
        if "yes" in self.voice.listen(timeout=6):
            self.voice.say("Tell me the phone number, digit by digit.")
            raw = self.clean_reply(self.voice.listen(timeout=12))
            digits = re.sub(r"\D", "", raw)
            if digits:
                self.people.set_field(name, "phone", digits)
                self.voice.say(f"Saved. You can add their email or address "
                               f"later by saying: add {name}'s email.")
            else:
                self.voice.say("I didn't catch any digits. You can add it "
                               f"later by saying: add {name}'s phone number.")

    def handle_remember_person(self):
        frame = self.camera.snapshot()
        if frame is None:
            self.voice.say("My camera is not giving me a picture.")
            return
        if not self.people:
            self.voice.say("Face recognition isn't available right now.")
            return
        found = self.people.identify(frame)
        if not found:
            self.voice.say("I can't see a face clearly. "
                           "Ask them to look at the camera and try again.")
            return
        name, enc = found[0]
        if name != "UNKNOWN":
            self.voice.say(f"I already know this person as {name}.")
            return
        self.enroll_person(enc, frame)

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
                if "yes" in self.voice.listen(timeout=6):
                    self.enroll_person(enc, frame)
                else:
                    self.voice.say("Okay, I won't remember them.")

    def handle_what_is_this(self):
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
                               "Say 'remember this person' if you want me to.")
            return
        if kind == "object" and data:
            self.voice.say("I see " + ", ".join(data) + ".")
            explanation = ask_qwen(
                f"In 2 short sentences, explain what a {data[0]} is and "
                f"what it's used for.")
            self.voice.say(explanation)
            return
        answer = ask_qwen("Look at this image and tell me the main thing you see, "
                          "in one short sentence.", image_bgr_frame=frame)
        self.voice.say(answer)

    def auto_greet(self):
        if not self.people:
            return
        frame = self.camera.snapshot()
        if frame is None:
            return
        for name, _enc in self.people.identify(frame):
            if name != "UNKNOWN" and name not in self.greeted:
                self.greeted.add(name)
                self.voice.say(f"Hello {name}.")

    # ---------- image files on disk ----------
    def handle_image_file(self, text: str):
        """'who is in photo.png' / 'what is o2 png' -> load the file and
        analyze it with the same face/YOLO/X-Lens pipeline as the camera."""
        fname, all_files = find_image_file(text)
        if not fname:
            if all_files:
                self.voice.say("I couldn't match that name. Image files here are: "
                               + ", ".join(all_files[:5]) + ".")
            else:
                self.voice.say("There are no image files in my folder.")
            return
        img = cv2.imread(fname)
        if img is None:
            self.voice.say(f"I found {fname} but couldn't open it.")
            return
        self.voice.say(f"Looking at {fname}.")
        if "who" in text and self.people:
            found = self.people.identify(img)
            if found:
                names = [n for n, _ in found]
                known = [n for n in names if n != "UNKNOWN"]
                if known:
                    self.voice.say("I recognize " + ", ".join(known) + ".")
                else:
                    self.voice.say("There is a face, but nobody I know.")
                return
        kind, data = self.detect_scene(img)
        if kind == "object" and data:
            self.voice.say("I see " + ", ".join(data) + ".")
            return
        question = ("Who is in this image? Describe them briefly."
                    if "who" in text else
                    "Describe the main thing in this image in one short sentence.")
        self.voice.say(ask_qwen(question, image_bgr_frame=img))

    # ---------- locations ----------
    def handle_where_is(self, place: str):
        hit = geocode(place)
        if hit == "TIMEOUT":
            self.voice.say("The map service is responding slowly right now. "
                           "Try again in a moment.")
            return
        if not hit:
            self.voice.say(f"I couldn't find {place} on the map.")
            return
        _lat, _lon, display = hit
        self.voice.say(f"{place} is at: {display}.")

    def handle_how_far(self, text: str):
        m = re.search(r"how far is (.+?) from (.+)", text)
        if m:
            a_name, b_name = m.group(1).strip(), m.group(2).strip()
        else:
            m = re.search(r"how far is (.+)", text)
            if not m:
                self.voice.say("How far is what, from where?")
                return
            a_name, b_name = m.group(1).strip(" ?."), None
        a = geocode(a_name)
        if a == "TIMEOUT":
            self.voice.say("The map service is responding slowly right now. "
                           "Try again in a moment.")
            return
        if not a:
            self.voice.say(f"I couldn't find {a_name} on the map.")
            return
        if b_name:
            b = geocode(b_name)
            if b == "TIMEOUT":
                self.voice.say("The map service is responding slowly right now. "
                               "Try again in a moment.")
                return
            if not b:
                self.voice.say(f"I couldn't find {b_name} on the map.")
                return
            km = haversine_km(a[0], a[1], b[0], b[1])
            self.voice.say(f"{a_name} is about {km:.1f} kilometers from {b_name}, "
                           "in a straight line.")
        else:
            km = haversine_km(a[0], a[1], HOME_LAT, HOME_LON)
            self.voice.say(f"{a_name} is about {km:.1f} kilometers from you, "
                           "in a straight line.")

    def handle_nearby(self, kind: str):
        names = nearby_places(kind, n=5)
        if names == "TIMEOUT":
            self.voice.say("The map service is responding slowly right now. "
                           "Try again in a moment.")
            return
        if not names:
            self.voice.say(f"I couldn't find any {kind}s near {HOME_PLACE}.")
            return
        self.voice.say(f"Near you I found: " + ", ".join(names) + ".")

    # ---------- timers ----------
    def start_timer(self, seconds: int):
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

    # ---------- notes ----------
    NOTE_MARKERS = ("note it down", "note down", "take a note", "take notes",
                    "take note", "make a note", "note that",
                    "remember that", "remember to")

    def handle_take_note(self, text: str):
        content = text
        for marker in self.NOTE_MARKERS:
            if marker in content:
                content = content.split(marker, 1)[1].strip(" ,.:")
                break
        # leftover fragments like "s" (from "take notes" matching "take note")
        # or empty content mean the command carried no real note — ask.
        if len(content) < 3 or content == text:
            self.voice.say("What should I note down?")
            content = self.clean_reply(self.voice.listen(timeout=10))
        if len(content) >= 3:
            self.notes.add(content)
            self.voice.say("Noted.")
        else:
            self.voice.say("I didn't catch anything to note.")

    def handle_read_notes(self):
        recent = self.notes.recent(5)
        if not recent:
            self.voice.say("You don't have any notes yet.")
            return
        self.voice.say(f"Your last {len(recent)} notes:")
        for line in recent:
            self.voice.say(line)

    # ---------- reminders ----------
    def handle_reminder(self, text: str):
        parsed = parse_reminder(text)
        if not parsed:
            self.voice.say("What should I remind you about?")
            content = self.clean_reply(self.voice.listen(timeout=10))
            if not content:
                self.voice.say("I didn't catch that, no reminder set.")
                return
            # the reply may itself be a full sentence like
            # "remind me tomorrow that i have a class" — parse it properly
            # instead of saving the raw phrasing as the reminder text
            if "remind" in content:
                parsed = parse_reminder(content)
            if not parsed:
                self.voice.say("For when? Say today, tomorrow, or in some days.")
                when = self.clean_reply(self.voice.listen(timeout=8))
                parsed = parse_reminder(f"remind me {when} that {content}")
            if not parsed:
                parsed = (content, datetime.date.today())
        content, due = parsed
        self.reminders.add(content, due)
        nice = ("today" if due == datetime.date.today()
                else "tomorrow" if due == datetime.date.today() + datetime.timedelta(days=1)
                else due.isoformat())
        self.voice.say(f"Reminder saved for {nice}: {content}.")

    def handle_read_reminders(self):
        items = self.reminders.upcoming(5)
        if not items:
            self.voice.say("You have no reminders.")
            return
        self.voice.say(f"You have {len(items)} reminders:")
        for it in items:
            self.voice.say(f"On {it['due']}: {it['text']}.")

    def handle_clear_reminders(self):
        if not self.reminders.items:
            self.voice.say("You have no reminders to clear.")
            return
        self.voice.say(f"Delete all {len(self.reminders.items)} reminders? Say yes or no.")
        if "yes" in self.voice.listen(timeout=6):
            self.reminders.items = []
            self.reminders.save()
            self.voice.say("All reminders deleted.")
        else:
            self.voice.say("Okay, keeping them.")

    def handle_clear_notes(self):
        if not os.path.exists(NOTES_PATH) or not self.notes.recent(1):
            self.voice.say("You have no notes to clear.")
            return
        self.voice.say("Delete all your notes? Say yes or no.")
        if "yes" in self.voice.listen(timeout=6):
            open(NOTES_PATH, "w").close()
            self.voice.say("All notes deleted.")
        else:
            self.voice.say("Okay, keeping them.")

    def reminder_watcher(self):
        """Background thread: announces reminders when their day arrives."""
        while True:
            for it in self.reminders.due_now():
                self.voice.say(f"Reminder: {it['text']}.")
                self.reminders.mark_announced(it)
            time.sleep(30)

    # ---------- profiles ----------
    def handle_profile_request(self, text: str):
        """'show me bashant's profile' / 'show profile of bashant'"""
        m = re.search(r"(?:show me|show|open)\s+(.*?)(?:'s)?\s+profile", text)
        if not m:
            m = re.search(r"profile of\s+(.+)", text)
        spoken = m.group(1).strip() if m else ""
        name = self.people.find_name(spoken) if (self.people and spoken) else None
        if not name:
            self.voice.say(f"I don't have a profile for {spoken or 'that person'}.")
            return
        self.voice.say(self.people.profile_text(name))

    def handle_add_contact(self, text: str):
        """'add bashant's phone number' / 'add email for bashant'"""
        field = next((f for f in ("phone", "email", "address") if f in text), None)
        if not field or not self.people:
            return False
        m = re.search(r"add\s+(.*?)(?:'s)?\s+(?:phone|email|address)", text)
        if not m:
            m = re.search(r"(?:phone|email|address)\s+(?:number\s+)?(?:for|of)\s+(.+)", text)
        spoken = m.group(1).strip() if m else ""
        name = self.people.find_name(spoken) if spoken else None
        if not name:
            self.voice.say(f"I don't have anyone named {spoken or 'that'} saved. "
                           "Remember them first, then add details.")
            return True
        self.voice.say(f"Tell me {name}'s {field}.")
        raw = self.clean_reply(self.voice.listen(timeout=15))
        if not raw:
            self.voice.say("I didn't catch that, nothing saved.")
            return True
        if field == "phone":
            raw = re.sub(r"\D", "", raw) or raw
        if field == "email":
            raw = raw.replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
        self.people.set_field(name, field, raw)
        self.voice.say(f"Saved {name}'s {field} as {raw}. "
                       "If that's wrong, just say the command again to overwrite it.")
        return True

    # ---------- routing ----------
    def route(self, text: str) -> bool:
        if not text:
            return True
        if "shutdown" in text or "shut down" in text:
            self.voice.say("Shutting down. Goodbye Jeevan.")
            return False

        # -- image files on disk (checked early: "who is o2 png" must not be
        #    stolen by the profile lookup or Qwen). Word-boundary match so
        #    words like "topic" or "epic" don't falsely trigger on "pic".
        if re.search(r"\b(png|jpg|jpeg|photo|image|picture|pic)\b", text):
            self.handle_image_file(text)
            return True

        # -- locations (Qwen has no map data; OpenStreetMap does) --
        if "how far" in text:
            self.handle_how_far(text)
            return True
        m = re.search(r"(?:where is|location of|precise location of)\s+(.+)", text)
        if m:
            self.handle_where_is(m.group(1).strip(" ?."))
            return True
        m = re.search(r"(?:how many|find|any)?\s*(restaurant|cafe|hospital|pharmacy|"
                      r"bank|atm|hotel|school|temple|gym)s?\s+(?:are\s+)?"
                      r"(?:around|near)\s+(?:me|here|us)", text)
        if m:
            self.handle_nearby(m.group(1))
            return True

        # -- profiles / people --
        if self.people and "forget" in text \
                and "note" not in text and "node" not in text \
                and "reminder" not in text:
            spoken = text.split("forget", 1)[1].strip()
            name = self.people.find_name(spoken) if spoken else None
            if name:
                self.people.forget(name)
                self.voice.say(f"Removed {name}'s profile.")
            else:
                self.voice.say(f"I don't have anyone named {spoken} saved.")
            return True
        if "remember this person" in text or "remember the person" in text \
                or ("open camera" in text and "remember" in text):
            self.handle_remember_person()
            return True
        # "remember that I have to..." / "remember to buy..." = take a note
        # (checked AFTER "remember this person" so faces aren't shadowed)
        if "remember that" in text or "remember to" in text:
            self.handle_take_note(text)
            return True
        # bare "remember" with nothing actionable -> ask what to note
        if text.strip() in ("remember", "remember me", "can you remember"):
            self.handle_take_note("")
            return True
        # "who is basanta" / "do you know basanta" -> saved profile lookup
        if self.people:
            m = re.search(r"(?:who is|do you know(?: who is)?)\s+([a-z ]+)", text)
            if m:
                candidate = m.group(1).strip()
                if candidate not in ("this", "that", "it"):
                    known = self.people.find_name(candidate)
                    if known:
                        self.voice.say(self.people.profile_text(known))
                        return True
                    # not a saved person -> fall through (set questions / Qwen)
        if "profile" in text and self.people:
            self.handle_profile_request(text)
            return True
        if text.startswith("add") and any(f in text for f in ("phone", "email", "address")):
            if self.handle_add_contact(text):
                return True

        # -- notes --
        if any(k in text for k in ("clear my notes", "delete my notes",
                                   "delete all notes", "forget my notes",
                                   "forget the notes", "forget all the notes",
                                   "forget all the nodes", "forget the nodes",
                                   "delete my nodes")):
            self.handle_clear_notes()
            return True
        if any(k in text for k in ("take a note", "take notes", "take note",
                                   "note down", "note it down", "make a note",
                                   "note that")):
            self.handle_take_note(text)
            return True
        if "read my notes" in text or "my notes" in text:
            self.handle_read_notes()
            return True

        # -- reminders --
        if any(k in text for k in ("clear my reminders", "delete my reminders",
                                   "delete all reminders", "forget my reminders",
                                   "delete the reminder", "delete reminder")):
            self.handle_clear_reminders()
            return True
        if "remind" in text and "reminder" not in text:
            self.handle_reminder(text)
            return True
        if "reminders" in text or "my reminder" in text:
            self.handle_read_reminders()
            return True

        # -- timers --
        timer_seconds = parse_timer_seconds(text)
        if timer_seconds is not None:
            if timer_seconds <= 0:
                self.voice.say("I need a positive amount of time.")
            else:
                self.start_timer(timer_seconds)
            return True
        if any(w in text for w in ("timer", "alarm", "countdown")):
            self.voice.say("For how long, in seconds?")
            reply = self.voice.listen(timeout=8)
            nums = re.findall(r"\d+", reply) if reply else []
            if nums:
                self.start_timer(int(nums[-1]))
            else:
                self.voice.say("I didn't catch a number, no timer started.")
            return True

        # -- vision --
        if any(k in text for k in ("who is this", "who is that", "who am i looking at")):
            self.handle_who_is_this()
            return True
        if any(k in text for k in ("what is this", "what is that",
                                   "what am i looking at", "identify the object",
                                   "identify this", "open camera",
                                   "what do you see", "can you see",
                                   "what can you see", "look at this",
                                   "describe what you see")):
            self.handle_what_is_this()
            return True

        # -- set questions --
        canonical = match_set_question(text)
        if canonical:
            self.voice.say(answer_set_question(canonical))
            return True

        # -- Qwen fallback --
        self.voice.say(ask_qwen(text))
        return True

    def run(self):
        threading.Thread(target=self.reminder_watcher, daemon=True).start()
        self.voice.say("EDITH online. Say my name followed by your question.")
        due = self.reminders.due_now()
        if due:
            self.voice.say(f"You have {len(due)} reminders for today.")
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