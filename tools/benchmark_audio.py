from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import librosa
from pipeline import run_pipeline

EXT={'.wav','.mp3','.flac','.m4a','.ogg','.mp4','.webm'}

def main():
    ap=argparse.ArgumentParser(description='Benchmark full ASR + financial-audio pipeline on a folder')
    ap.add_argument('--dir',required=True)
    ap.add_argument('--model',default='small')
    ap.add_argument('--json',default='reports/audio_benchmark.json')
    ap.add_argument('--csv',default='reports/audio_benchmark.csv')
    a=ap.parse_args()
    files=sorted(p for p in Path(a.dir).rglob('*') if p.suffix.lower() in EXT)
    if not files:
        print(f'No supported audio/video files found in {a.dir}')
        raise SystemExit(3)
    rows=[]
    for p in files:
        duration=float(librosa.get_duration(path=str(p)))
        t0=time.perf_counter(); result=run_pipeline(str(p),a.model,include_raw=False); elapsed=time.perf_counter()-t0
        rows.append({
            'file':str(p),'audio_seconds':round(duration,3),'processing_seconds':round(elapsed,3),
            'real_time_factor':round(elapsed/duration,4) if duration else None,
            'asr_confidence':result['transcription']['confidence'],
            'pii_count':result['privacy']['pii_count'],'profanity_count':result['privacy']['profanity_count'],
            'overall_trust':result['confidence']['overall_trust'],
        })
        print(f"{p.name}: {elapsed:.2f}s for {duration:.2f}s audio, RTF={elapsed/duration:.3f}" if duration else f"{p.name}: {elapsed:.2f}s")
    valid=[r['real_time_factor'] for r in rows if r['real_time_factor'] is not None]
    summary={'model':a.model,'files':len(rows),'mean_real_time_factor':sum(valid)/len(valid) if valid else None,'rows':rows}
    Path(a.json).parent.mkdir(parents=True,exist_ok=True); Path(a.json).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with open(a.csv,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Reports written to {a.json} and {a.csv}")

if __name__=='__main__': main()
