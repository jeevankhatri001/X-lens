import json, os

DB_FILE = "facts.json"

def load_facts():
    """Load saved facts from disk (empty dict if none yet)."""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_facts(facts):
    with open(DB_FILE, "w") as f:
        json.dump(facts, f, indent=2)

def remember(key, value):
    """Store a fact under a key, e.g. remember('locker code', '42')."""
    facts = load_facts()
    facts[key] = value
    save_facts(facts)
    print(f"[stored] {key} = {value}")

def recall(key):
    """Look up a fact. Returns None if not known."""
    facts = load_facts()
    return facts.get(key)

# --- test it ---
if __name__ == "__main__":
    remember("locker code", "42")
    remember("wifi password", "sunway123")
    remember("bashant's birthday", "March 5")

    print("\nRecalling:")
    print("locker code   ->", recall("locker code"))
    print("wifi password ->", recall("wifi password"))
    print("unknown thing ->", recall("favorite color"))