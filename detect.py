import cv2
from insightface.app import FaceAnalysis

# Loads the detector + embedding models (downloads once, then cached)
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 means CPU

# Load the image (OpenCV reads as BGR — InsightFace expects BGR, so no conversion)
img = cv2.imread("test.jpg")

# STAGE 1+2+3 in one call: detect, align, and embed every face
faces = app.get(img)
print(f"Found {len(faces)} face(s)")

for i, face in enumerate(faces):
    left, top, right, bottom = face.bbox.astype(int)
    print(f"  Face {i+1}: box=({left},{top},{right},{bottom})  embedding dim={face.embedding.shape[0]}")
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)

cv2.imwrite("test_boxed.jpg", img)
print("Saved test_boxed.jpg")