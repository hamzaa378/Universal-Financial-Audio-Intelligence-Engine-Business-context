from __future__ import annotations
import argparse,csv,json,time,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import librosa
from pipeline import run_pipeline
from asr.fintech_asr import warmup_model
from decision_ai.warmup import warmup_semantic_ai

EXT={'.wav','.mp3','.flac','.m4a','.ogg','.aac','.mp4','.webm','.mov'}


def main():
    ap=argparse.ArgumentParser(description='Benchmark full GPU ASR + financial-audio pipeline')
    ap.add_argument('--dir',required=True)
    ap.add_argument('--model',default='small')
    ap.add_argument('--speed',choices=['fast','balanced','accuracy'],default='fast')
    ap.add_argument('--semantic',action='store_true')
    ap.add_argument('--protected-audio',action='store_true')
    ap.add_argument('--json',default='reports/audio_benchmark.json')
    ap.add_argument('--csv',default='reports/audio_benchmark.csv')
    a=ap.parse_args()
    files=sorted(p for p in Path(a.dir).rglob('*') if p.suffix.lower() in EXT)
    if not files:
        print(f'No supported audio/video files found in {a.dir}')
        raise SystemExit(3)

    print('Warming ASR model...')
    warm=warmup_model(a.model,smoke_test=True)
    print(f"ASR warm: {warm['device']} / {warm['compute_type']}")
    if a.semantic:
        sem=warmup_semantic_ai(); print('Semantic warm:',sem)

    rows=[]
    for p in files:
        duration=float(librosa.get_duration(path=str(p)))
        t0=time.perf_counter()
        result=run_pipeline(
            str(p),a.model,include_raw=False,use_semantic_ai=a.semantic,
            create_audio_output=a.protected_audio,asr_speed_mode=a.speed,
        )
        elapsed=time.perf_counter()-t0
        backend=result['transcription'].get('backend',{})
        rows.append({
            'file':str(p),'audio_seconds':round(duration,3),'processing_seconds':round(elapsed,3),
            'real_time_factor':round(elapsed/duration,4) if duration else None,
            'asr_seconds':round(result['performance'].get('asr',0)/1000,4),
            'text_ai_ms':round(result['performance'].get('privacy_and_nlp',0),3),
            'asr_device':backend.get('device'),'asr_compute':backend.get('compute_type'),
            'asr_confidence':result['transcription']['confidence'],
            'pii_count':result['privacy']['pii_count'],
            'sensitive_financial_id_count':result['privacy'].get('sensitive_financial_id_count',0),
            'profanity_count':result['privacy']['profanity_count'],
            'overall_trust':result['confidence']['overall_trust'],
        })
        pa=result.get('privacy',{}).get('protected_audio') or {}
        path=pa.get('path')
        if path:
            try: Path(path).unlink(missing_ok=True)
            except Exception: pass
        print(f"{p.name}: {elapsed:.2f}s / {duration:.2f}s, RTF={elapsed/duration:.3f}, ASR={backend.get('device')}")
    valid=[r['real_time_factor'] for r in rows if r['real_time_factor'] is not None]
    summary={
        'model':a.model,'speed':a.speed,'semantic':a.semantic,'protected_audio':a.protected_audio,
        'files':len(rows),'mean_real_time_factor':sum(valid)/len(valid) if valid else None,'rows':rows,
        'note':'Use consented/labeled audio for real performance and accuracy claims.'
    }
    Path(a.json).parent.mkdir(parents=True,exist_ok=True); Path(a.json).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with open(a.csv,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f'Reports written to {a.json} and {a.csv}')

if __name__=='__main__': main()
