import re

def parse_timer_command(text):
    """Turn a spoken command like 'set a timer for 2 minutes' into seconds."""
    text = text.lower()

    # find a number in the sentence (e.g. the '2' in 'for 2 minutes')
    match = re.search(r'(\d+)', text)
    if not match:
        return None
    number = int(match.group(1))

    # figure out the unit
    if "hour" in text:
        return number * 3600
    elif "minute" in text or "min" in text:
        return number * 60
    else:                       # default to seconds
        return number

# --- test with a few example commands ---
if __name__ == "__main__":
    tests = [
        "set a timer for 5 seconds",
        "timer for 2 minutes",
        "remind me in 1 hour",
        "give me a 30 second timer",
        "set timer 10",
    ]
    for t in tests:
        print(f"{t!r:40s} -> {parse_timer_command(t)} seconds")