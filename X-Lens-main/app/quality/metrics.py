from __future__ import annotations
import math
import cv2
import numpy as np

EPS = 1e-8

def clamp01(x: float) -> float: return float(max(0.0, min(1.0, x)))

def brightness_score(bgr: np.ndarray) -> float:
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32) / 255.0
    mean = float(v.mean())
    return clamp01(1.0 - abs(mean - 0.5) / 0.5)

def contrast_score(gray: np.ndarray) -> float:
    return clamp01(float(gray.std()) / 64.0)

def laplacian_blur(gray: np.ndarray) -> float:
    return clamp01(math.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var())) / math.log1p(1000.0))

def tenengrad_blur(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return clamp01(math.log1p(float(np.mean(gx*gx + gy*gy))) / math.log1p(20000.0))

def fft_blur(gray: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    power = np.abs(f) ** 2
    h, w = gray.shape; yy, xx = np.ogrid[:h, :w]; cy, cx = h//2, w//2
    radius = max(2, int(min(h,w)*0.1)); mask = (yy-cy)**2 + (xx-cx)**2 > radius**2
    return clamp01(float(power[mask].sum() / (power.sum() + EPS)) * 3.0)

def blur_score(gray: np.ndarray) -> tuple[float, dict[str, float]]:
    parts = {
        "laplacian": laplacian_blur(gray),
        "tenengrad": tenengrad_blur(gray),
        "fft": fft_blur(gray),
    }

    # Median is more stable for low-resolution ESP32-CAM images.
    # One unusually low FFT result will not dominate the final score.
    combined = float(np.median(list(parts.values())))

    return clamp01(combined), parts

def noise_score(gray: np.ndarray) -> float:
    smooth = cv2.GaussianBlur(gray, (3,3), 0)
    hp = gray.astype(np.float32) - smooth.astype(np.float32)
    sigma = float(np.median(np.abs(hp - np.median(hp))) / 0.6745)
    return clamp01(1.0 - sigma / 30.0)

def resolution_score(bgr: np.ndarray, min_width: int, min_height: int) -> float:
    h, w = bgr.shape[:2]
    return clamp01(min(w/min_width, h/min_height))

def exposure_score(bgr: np.ndarray) -> tuple[float,float,float]:
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,2]
    under = float(np.mean(v <= 13)); over = float(np.mean(v >= 242))
    return clamp01(1.0 - max(under, over)), under, over

def colorfulness_score(bgr: np.ndarray) -> float:
    b,g,r = [c.astype(np.float32) for c in cv2.split(bgr)]
    rg = r-g; yb = 0.5*(r+g)-b
    raw = np.sqrt(rg.std()**2+yb.std()**2)+0.3*np.sqrt(rg.mean()**2+yb.mean()**2)
    return clamp01(float(raw)/100.0)
