from __future__ import annotations
import argparse, json, platform, sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

PACKAGES=[
    'faster-whisper','ctranslate2','onnxruntime','onnxruntime-gpu','fastembed',
    'tokenizers','transformers','numpy','soundfile','soxr','huggingface-hub','pyannote.audio'
]
PRIVACY_CRITICAL={
    'faster-whisper','ctranslate2','onnxruntime','onnxruntime-gpu','fastembed','tokenizers','transformers'
}

def version_of(name: str):
    try: return metadata.version(name)
    except metadata.PackageNotFoundError: return None

def build_fingerprint():
    return {
        'schema':1,
        'captured_at_utc':datetime.now(timezone.utc).isoformat(),
        'python':platform.python_version(),
        'platform':platform.platform(),
        'models':{
            'semantic':'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'ner':'onnx-community/multilang-pii-ner-ONNX',
        },
        'packages':{name:version_of(name) for name in PACKAGES},
    }

def compare(old,new):
    changed=[]
    op=old.get('packages',{}) if isinstance(old,dict) else {}
    np=new.get('packages',{})
    for name in PACKAGES:
        if op.get(name)!=np.get(name):
            changed.append({'package':name,'old':op.get(name),'new':np.get(name),'privacy_critical':name in PRIVACY_CRITICAL})
    for key in ('semantic','ner'):
        ov=(old.get('models',{}) if isinstance(old,dict) else {}).get(key)
        nv=new.get('models',{}).get(key)
        if ov!=nv:
            changed.append({'model':key,'old':ov,'new':nv,'privacy_critical':True})
    return changed

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='reports/runtime_fingerprint.json')
    ap.add_argument('--fail-on-critical-drift',action='store_true')
    a=ap.parse_args()
    out=Path(a.output); old=None
    if out.exists():
        try: old=json.loads(out.read_text(encoding='utf-8'))
        except Exception: old=None
    cur=build_fingerprint(); changes=compare(old,cur) if old else []
    cur['changes_since_previous']=changes
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(cur,indent=2),encoding='utf-8')
    print(json.dumps(cur,indent=2))
    critical=[x for x in changes if x.get('privacy_critical')]
    if critical:
        print('\n[WARNING] Privacy-sensitive runtime/model drift detected. Run run_privacy_benchmark.bat before relying on this environment.')
    else:
        print('\n[OK] No privacy-sensitive runtime drift detected from the previous fingerprint.' if old else '\n[OK] Baseline runtime fingerprint created.')
    if critical and a.fail_on_critical_drift:
        raise SystemExit(3)

if __name__=='__main__': main()
