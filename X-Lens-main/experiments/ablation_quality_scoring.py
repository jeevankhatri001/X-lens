import argparse, csv
from pathlib import Path
from PIL import Image
from app.quality.analyzer import ImageQualityAnalyzer

def main():
    p=argparse.ArgumentParser(); p.add_argument('images',type=Path); p.add_argument('--output',type=Path,default=Path('data/logs/quality_ablation.csv')); args=p.parse_args()
    rows=[]
    for image_path in args.images.glob('*'):
        try: image=Image.open(image_path).convert('RGB')
        except Exception: continue
        for method in ['minimum','geometric_mean','weighted_average']:
            r=ImageQualityAnalyzer(aggregation=method).analyze(image); rows.append({'image':str(image_path),'method':method,'overall':r.overall,'accepted':r.is_acceptable})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ['image','method','overall','accepted']); writer.writeheader(); writer.writerows(rows)
if __name__=='__main__': main()
