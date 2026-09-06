from __future__ import annotations
import json, time
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from nlp.pii import detect_pii, semantic_status
from decision_ai.warmup import warmup_semantic_ai

NUMERIC={"PHONE","AADHAAR","CARD","ACCOUNT_NUMBER","OTP","CVV","PINCODE"}
def norm(k,v):
    return ''.join(c for c in v if c.isdigit()) if k in NUMERIC else ' '.join(v.casefold().split()).strip(' .,:;<>' )

def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    status=semantic_status()
    out_path=Path('reports/semantic_privacy_benchmark.json'); out_path.parent.mkdir(parents=True,exist_ok=True)
    if not status.get('available'):
        report={"status":"skipped","reason":"Semantic AI unavailable","engine":status,
                "note":"Install requirements-ai-lite.txt and run once with internet access to cache the embedding model."}
        out_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('[SKIP] Semantic privacy benchmark: FastEmbed/model unavailable.')
        print(status.get('error') or '')
        return
    cases=load('benchmarks/semantic_pii_cases.jsonl')
    tp=fp=fn=0; rows=[]
    # warm up all shared prototype banks in one batch
    warmup_semantic_ai()
    detect_pii(cases[0]['text'], use_semantic=True)
    times=[]
    for c in cases:
        t0=time.perf_counter(); got=detect_pii(c['text'], use_semantic=True); times.append((time.perf_counter()-t0)*1000)
        exp=Counter((e['type'],norm(e['type'],e['value'])) for e in c['expected'])
        act=Counter((e['type'],norm(e['type'],e['value'])) for e in got)
        t=sum((exp&act).values()); f=sum((act-exp).values()); n=sum((exp-act).values())
        tp+=t; fp+=f; fn+=n
        rows.append({"id":c['id'],"tp":t,"fp":f,"fn":n,"detections":[{"type":x['type'],"confidence":x['confidence'],"decision":x.get('decision_method'),"semantic_score":x.get('semantic_score')} for x in got]})
    precision=tp/(tp+fp) if tp+fp else 1.0; recall=tp/(tp+fn) if tp+fn else 1.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    report={"status":"completed","engine":status,"cases":len(cases),"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":f1,
            "mean_ms":sum(times)/len(times) if times else 0.0,"rows":rows,
            "note":"Small synthetic semantic stress set; not a production accuracy claim."}
    out_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Semantic AI: TP={tp} FP={fp} FN={fn} Precision={precision*100:.2f}% Recall={recall*100:.2f}% F1={f1*100:.2f}%")
    print(f"Mean semantic-case latency after warmup: {report['mean_ms']:.3f} ms")
    print(f"Report: {out_path}")
    if fp or fn: raise SystemExit(2)
if __name__=='__main__': main()
