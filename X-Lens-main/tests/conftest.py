import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def detailed_image():
    canvas = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 40), (300, 220), (20, 80, 220), -1)
    cv2.circle(canvas, (460, 160), 100, (220, 80, 20), -1)
    for x in range(20, 640, 30):
        cv2.line(canvas, (x, 300), (x, 450), (0, 0, 0), 2)
    cv2.putText(canvas, "X-LENS TEST", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
