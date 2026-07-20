import numpy as np
from PIL import Image, ImageFilter
from app.quality.analyzer import ImageQualityAnalyzer

def test_report_has_all_metrics(detailed_image):
    r=ImageQualityAnalyzer(threshold=.05).analyze(detailed_image)
    assert 0<=r.overall<=1 and set(r.blur_components)=={'laplacian','tenengrad','fft'}

def test_small_image_fails_resolution():
    r=ImageQualityAnalyzer(threshold=.5,min_width=640,min_height=480).analyze(Image.new('RGB',(100,100),(128,128,128)))
    assert not r.is_acceptable and 'resolution' in (r.rejection_reason or '')

def test_blur_reduces_blur_score(detailed_image):
    a=ImageQualityAnalyzer(threshold=0)
    sharp=a.analyze(detailed_image).blur; blurry=a.analyze(detailed_image.filter(ImageFilter.GaussianBlur(8))).blur
    assert blurry < sharp
