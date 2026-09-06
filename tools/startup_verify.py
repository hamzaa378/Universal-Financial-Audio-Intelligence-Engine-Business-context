from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from decision_ai.startup import verify_startup

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='small')
    ap.add_argument('--semantic',action='store_true')
    ap.add_argument('--ner',action='store_true')
    a=ap.parse_args()
    r=verify_startup(a.model,semantic=a.semantic,ner=a.ner,asr_smoke=True)
    print(json.dumps(r,indent=2))
    asr=r.get('asr_gpu',{})
    print('\nASR GPU =', 'CUDA FP16 ✓' if asr.get('ready') and str(asr.get('compute_type')).lower()=='float16' else 'NOT READY ✗')
    sem=r.get('semantic_ai',{}); print('Semantic AI =', (str(sem.get('provider') or sem.get('backend'))+' ✓') if sem.get('ready') else 'CPU/rules fallback or disabled')
    judge=r.get('privacy_judge',{}); print('Privacy Judge =', 'ready ✓' if judge.get('ready') else judge.get('state','not ready'))
    ner=r.get('ner_ai',{}); print('NER AI =', (str(ner.get('provider') or ner.get('backend'))+' ✓') if ner.get('ready') else 'disabled/unavailable')
    dia=r.get('diarization',{}); print('Diarization =', 'available ✓' if dia.get('available') else 'available/disabled' if dia.get('installed') else 'not installed')
    raise SystemExit(0 if asr.get('ready') else 5)
