import argparse

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='Qwen/Qwen2.5-VL-3B-Instruct'); args=p.parse_args()
    from huggingface_hub import snapshot_download
    print(snapshot_download(args.model))
if __name__=='__main__': main()
