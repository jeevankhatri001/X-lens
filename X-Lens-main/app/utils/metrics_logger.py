import csv
from pathlib import Path
from threading import Lock

class MetricsLogger:
    def __init__(self, path: Path): self.path, self.lock = path, Lock()
    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            exists = self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                if not exists: writer.writeheader()
                writer.writerow(record)
