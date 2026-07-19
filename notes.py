import os

NOTES_FILE = "notes.txt"

def add_note(text):
    """Append a note to the list."""
    with open(NOTES_FILE, "a") as f:      # "a" = append (add to end, keep old ones)
        f.write(text.strip() + "\n")

def get_notes():
    """Return all notes as a list of lines."""
    if not os.path.exists(NOTES_FILE):
        return []
    with open(NOTES_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]

def clear_notes():
    if os.path.exists(NOTES_FILE):
        os.remove(NOTES_FILE)

# --- test ---
if __name__ == "__main__":
    add_note("I have to eat my medicine after 5 minutes")
    add_note("Call Bashant about the GPU tomorrow")
    add_note("Submit the project form on Friday")

    print("Things you asked me to remember:")
    for i, note in enumerate(get_notes(), 1):
        print(f"  {i}. {note}")