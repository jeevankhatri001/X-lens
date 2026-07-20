import argparse, json
from pathlib import Path
from PIL import Image
from app.core.config import get_settings
from app.quality.analyzer import ImageQualityAnalyzer

def main():
    parser=argparse.ArgumentParser(description='X-Lens CLI'); parser.add_argument('image',type=Path); args=parser.parse_args()
    s=get_settings(); analyzer=ImageQualityAnalyzer(s.quality_threshold,s.quality_aggregation,s.min_width,s.min_height)
    print(json.dumps(analyzer.analyze(Image.open(args.image)).to_dict(),indent=2))
if __name__=='__main__': main()
