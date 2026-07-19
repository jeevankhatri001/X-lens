"""
recognize.py -- face enrollment + identification using InsightFace (ArcFace).

The heavy lifting (detect -> align -> embed => a 512-D vector per face) all
happens inside app.get(). This script adds the final pipeline stage, MATCH:
compare a face's embedding against a saved database of known people by
cosine similarity.

Usage:
  python recognize.py enroll   <image> <name>   # add a person to the database
  python recognize.py identify <image>          # label every face in an image
  python recognize.py list                      # show who's enrolled
"""

import sys
import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

DB_PATH = "known_faces.pkl"

# --- PLACEHOLDER threshold -------------------------------------------------
# Cosine similarity above this counts as "same person". 0.40 is a reasonable
# STARTING point for ArcFace, but it is NOT tuned yet. We calibrate it tomorrow
# with real XIAO frames. For now, trust the raw scores more than the label.
MATCH_THRESHOLD = 0.40
# ---------------------------------------------------------------------------


def get_app():
    """Load the InsightFace models once (detector + landmarks + ArcFace)."""
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 => CPU
    return app


def load_db():
    """Return {name: [embedding, ...]}. Empty dict if no file yet."""
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


def largest_face(faces):
    """Pick the biggest face by box area -- the intended subject of a portrait."""
    def area(f):
        l, t, r, b = f.bbox
        return (r - l) * (b - t)
    return max(faces, key=area)


def enroll(app, image_path, name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: could not read {image_path}")
        return
    faces = app.get(img)
    if not faces:
        print(f"No face found in {image_path} -- try a clearer image.")
        return
    if len(faces) > 1:
        print(f"Note: {len(faces)} faces found; enrolling the largest one.")
    face = largest_face(faces)

    db = load_db()
    # Store the L2-NORMALIZED embedding, so cosine similarity is just a dot product.
    db.setdefault(name, []).append(face.normed_embedding)
    save_db(db)
    print(f"Enrolled '{name}'  (now {len(db[name])} sample(s) for this person)")


def best_match(embedding, db):
    """Return (name, score) of the closest enrolled person by cosine similarity."""
    best_name, best_score = "Unknown", -1.0
    for name, samples in db.items():
        for sample in samples:
            # Both vectors are L2-normalized, so dot product == cosine similarity.
            score = float(np.dot(embedding, sample))
            if score > best_score:
                best_name, best_score = name, score
    return best_name, best_score


def identify(app, image_path):
    db = load_db()
    if not db:
        print("Database is empty -- enroll someone first.")
        return
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: could not read {image_path}")
        return

    faces = app.get(img)
    print(f"Found {len(faces)} face(s) in {image_path}\n")

    for i, face in enumerate(faces):
        name, score = best_match(face.normed_embedding, db)
        label = name if score >= MATCH_THRESHOLD else "Unknown"
        # Always print the raw score + who it was closest to -- the number teaches
        # you far more than the label, especially near the threshold.
        print(f"  Face {i + 1}: {label:10s} (closest: {name}, score={score:.3f})")

        l, t, r, b = face.bbox.astype(int)
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (l, t), (r, b), color, 2)
        cv2.putText(img, f"{label} {score:.2f}", (l, max(t - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out = "identified.jpg"
    cv2.imwrite(out, img)
    print(f"\nSaved {out}")


def main():
    args = sys.argv
    if len(args) < 2:
        print(__doc__)
        return
    cmd = args[1]

    if cmd == "list":
        db = load_db()
        if not db:
            print("No one enrolled yet.")
            return
        for name, samples in db.items():
            print(f"  {name}: {len(samples)} sample(s)")
        return

    if cmd == "enroll" and len(args) == 4:
        enroll(get_app(), args[2], args[3])
    elif cmd == "identify" and len(args) == 3:
        identify(get_app(), args[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()