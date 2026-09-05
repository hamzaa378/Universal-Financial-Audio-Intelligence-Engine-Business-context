import argparse, json
from pathlib import Path
from pipeline import run_pipeline

def main():
    p=argparse.ArgumentParser(description="Universal Financial Audio Intelligence Engine")
    p.add_argument("audio",help="Audio/video file")
    p.add_argument("--model",default=None,help="Whisper model, e.g. small, medium, large-v3")
    p.add_argument("--output",default="result.json")
    a=p.parse_args()
    result=run_pipeline(a.audio,a.model)
    Path(a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"transcript":result["transcription"]["text"],"intent":result["understanding"]["intent"],"trust":result["confidence"],"saved":a.output},indent=2,ensure_ascii=False))
if __name__=="__main__": main()
