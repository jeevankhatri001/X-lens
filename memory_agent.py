import json, os, re

DB_FILE = "facts.json"

def load_facts():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_facts(facts):
    with open(DB_FILE, "w") as f:
        json.dump(facts, f, indent=2)

def handle_memory_command(text):
    text_low = text.lower().strip()

    # --- REMEMBER: "remember that my X is Y"  /  "remember X is Y" ---
    if text_low.startswith("remember"):
        # strip the leading "remember that my" / "remember that" / "remember my"
        cleaned = re.sub(r'^remember( that)?( my)?\s+', '', text_low)
        # split on the first " is "
        if " is " in cleaned:
            key, value = cleaned.split(" is ", 1)
            facts = load_facts()
            facts[key.strip()] = value.strip()
            save_facts(facts)
            return f"Okay, I'll remember your {key.strip()} is {value.strip()}."
        return "I didn't catch what to remember."

    # --- RECALL: "what is my X" / "what's my X" ---
    if text_low.startswith("what"):
        # strip "what is my" / "what's my" / "what is"
        cleaned = re.sub(r"^what('s| is)?( my)?\s+", '', text_low).rstrip("?")
        facts = load_facts()
        value = facts.get(cleaned.strip())
        if value:
            return f"Your {cleaned.strip()} is {value}."
        return f"I don't know your {cleaned.strip()} yet."

    return None   # not a memory command

# --- test with example sentences ---
if __name__ == "__main__":
    tests = [
        "remember that my locker code is 42",
        "remember my wifi password is sunway123",
        "what is my locker code",
        "what's my wifi password",
        "what is my favorite color",
    ]
    for t in tests:
        print(f"{t!r:45s} -> {handle_memory_command(t)}")