from __future__ import annotations
import json,time,statistics
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from nlp.sensitive_financial import detect_sensitive_financial_ids


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def norm(v): return ' '.join(str(v).casefold().split()).strip(' .,:;')

def main():
    cases=load('benchmarks/sensitive_financial_cases.jsonl')
    tp=fp=fn=neg=negfp=0; times=[]; rows=[]
    for c in cases:
        t0=time.perf_counter(); got=detect_sensitive_financial_ids(c['text']); times.append((time.perf_counter()-t0)*1000)
        exp=Counter((e['type'],norm(e['value'])) for e in c['expected'])
        act=Counter((e['type'],norm(e['value'])) for e in got)
        t=sum((exp&act).values()); f=sum((act-exp).values()); n=sum((exp-act).values())
        tp+=t; fp+=f; fn+=n
        if not c['expected']:
            neg+=1; negfp += 1 if got else 0
        rows.append({'id':c['id'],'tp':t,'fp':f,'fn':n,'detections':[x['type'] for x in got]})
    precision=tp/(tp+fp) if tp+fp else 1.0; recall=tp/(tp+fn) if tp+fn else 1.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    report={
        'cases':len(cases),'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1,
        'negative_cases':neg,'negative_cases_with_fp':negfp,
        'negative_case_false_positive_rate':negfp/neg if neg else 0.0,
        'mean_ms':statistics.fmean(times) if times else 0.0,'rows':rows,
        'note':'Synthetic sensitive-financial-ID regression set; separate from classic PII metrics.'
    }
    out=Path('reports/sensitive_financial_benchmark.json'); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Sensitive financial IDs: TP={tp} FP={fp} FN={fn} Precision={precision*100:.2f}% Recall={recall*100:.2f}% F1={f1*100:.2f}%")
    print(f"Negative-case FPR={report['negative_case_false_positive_rate']*100:.2f}%  Mean={report['mean_ms']:.4f} ms")
    if fp or fn: raise SystemExit(2)
if __name__=='__main__': main()
