from __future__ import annotations
import argparse, csv, json, statistics, time
from collections import Counter
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.pii import detect_pii, mask_pii
from nlp.profanity import detect_profanity, reduce_profanity
from nlp.privacy import protect_text

NUMERIC = {"PHONE","AADHAAR","CARD","ACCOUNT_NUMBER","OTP","CVV","PINCODE"}

def load_jsonl(path):
    with open(path,encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def norm(kind, value):
    if kind in NUMERIC:
        return ''.join(c for c in value if c.isdigit())
    return ' '.join(value.casefold().split()).strip(' .,:;<>')

def evaluate_pii(cases, threshold=0.80):
    tp=fp=fn=0; negative=neg_fp=mask_fail=0; rows=[]
    for c in cases:
        actual=detect_pii(c['text'], min_confidence=threshold)
        exp=Counter((e['type'],norm(e['type'],e['value'])) for e in c['expected'])
        got=Counter((e['type'],norm(e['type'],e['value'])) for e in actual)
        t=sum((exp & got).values()); f=sum((got-exp).values()); n=sum((exp-got).values())
        tp+=t; fp+=f; fn+=n
        if not c['expected']:
            negative += 1
            if actual: neg_fp += 1
        masked=mask_pii(c['text'],actual)
        leaked=[]
        for e in c['expected']:
            # A required value should not survive exactly after successful masking.
            if e['value'] in masked:
                leaked.append(e['value'])
        if leaked: mask_fail += 1
        rows.append({"suite":"pii","id":c['id'],"tp":t,"fp":f,"fn":n,"pass":f==0 and n==0 and not leaked,"detail":';'.join(f"{x['type']}:{x.get('evidence','')}" for x in actual)})
    return summarize(tp,fp,fn,negative,neg_fp,mask_fail), rows

def evaluate_profanity(cases):
    tp=fp=fn=0; negative=neg_fp=mask_fail=0; rows=[]
    for c in cases:
        actual=detect_profanity(c['text'])
        exp=Counter(x.casefold() for x in c['expected'])
        got=Counter(x['canonical'].casefold() for x in actual)
        t=sum((exp & got).values()); f=sum((got-exp).values()); n=sum((exp-got).values())
        tp+=t; fp+=f; fn+=n
        if not c['expected']:
            negative += 1
            if actual: neg_fp += 1
        cleaned=reduce_profanity(c['text'],actual)
        leaked=[]
        for e in actual:
            if e['value'] in cleaned: leaked.append(e['value'])
        if leaked: mask_fail += 1
        rows.append({"suite":"profanity","id":c['id'],"tp":t,"fp":f,"fn":n,"pass":f==0 and n==0 and not leaked,"detail":';'.join(x['canonical'] for x in actual)})
    return summarize(tp,fp,fn,negative,neg_fp,mask_fail), rows

def summarize(tp,fp,fn,negative,neg_fp,mask_fail):
    precision=tp/(tp+fp) if tp+fp else 1.0
    recall=tp/(tp+fn) if tp+fn else 1.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {
        "tp":tp,"fp":fp,"fn":fn,
        "precision":precision,"recall":recall,"f1":f1,
        "false_discovery_rate": fp/(tp+fp) if tp+fp else 0.0,
        "false_negative_rate": fn/(tp+fn) if tp+fn else 0.0,
        "negative_cases":negative,"negative_cases_with_fp":neg_fp,
        "negative_case_false_positive_rate":neg_fp/negative if negative else 0.0,
        "mask_failure_cases":mask_fail,
    }

def benchmark(texts, iterations):
    timings=[]
    # warm-up
    for t in texts[:min(10,len(texts))]: protect_text(t)
    start_all=time.perf_counter()
    count=0
    for _ in range(iterations):
        for t in texts:
            t0=time.perf_counter(); protect_text(t); timings.append((time.perf_counter()-t0)*1000); count+=1
    elapsed=time.perf_counter()-start_all
    timings.sort()
    def pct(p):
        if not timings: return 0.0
        i=min(len(timings)-1,max(0,int(round((len(timings)-1)*p))))
        return timings[i]
    return {
        "evaluations":count,"total_seconds":elapsed,"cases_per_second":count/elapsed if elapsed else 0.0,
        "mean_ms":statistics.fmean(timings) if timings else 0.0,
        "p50_ms":pct(.50),"p95_ms":pct(.95),"p99_ms":pct(.99),
    }

def print_metrics(name,m):
    print(f"\n{name}")
    print('-'*72)
    print(f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"Precision={m['precision']*100:.2f}%  Recall={m['recall']*100:.2f}%  F1={m['f1']*100:.2f}%")
    print(f"False discovery={m['false_discovery_rate']*100:.2f}%  False negative={m['false_negative_rate']*100:.2f}%")
    print(f"Negative-case false-positive rate={m['negative_case_false_positive_rate']*100:.2f}% ({m['negative_cases_with_fp']}/{m['negative_cases']})")
    print(f"Mask failure cases={m['mask_failure_cases']}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pii',default='benchmarks/pii_cases.jsonl')
    ap.add_argument('--profanity',default='benchmarks/profanity_cases.jsonl')
    ap.add_argument('--iterations',type=int,default=100)
    ap.add_argument('--json',default='reports/privacy_benchmark.json')
    ap.add_argument('--csv',default='reports/privacy_cases.csv')
    a=ap.parse_args()
    pii_cases=load_jsonl(a.pii); prof_cases=load_jsonl(a.profanity)
    pii,rows1=evaluate_pii(pii_cases,0.80); prof,rows2=evaluate_profanity(prof_cases)
    sweep={}
    for th in (0.80,0.90,0.93,0.95,0.97):
        sm,_=evaluate_pii(pii_cases,th)
        sweep[f"{th:.2f}"]={k:sm[k] for k in ("precision","recall","f1","false_discovery_rate","negative_case_false_positive_rate")}
    speed=benchmark([x['text'] for x in pii_cases+prof_cases], max(1,a.iterations))
    report={
      "dataset":{"pii_cases":len(pii_cases),"profanity_cases":len(prof_cases),"note":"Synthetic regression benchmark; validate with consented/labeled real call data before production claims."},
      "pii":pii,"pii_threshold_sweep":sweep,"profanity":prof,"text_privacy_efficiency":speed,
    }
    Path(a.json).parent.mkdir(parents=True,exist_ok=True)
    Path(a.json).write_text(json.dumps(report,indent=2),encoding='utf-8')
    with open(a.csv,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['suite','id','tp','fp','fn','pass','detail']); w.writeheader(); w.writerows(rows1+rows2)
    print_metrics('PII DETECTION + MASKING',pii)
    print_metrics('PROFANITY DETECTION + REDUCTION',prof)
    print('\nEFFICIENCY')
    print('-'*72)
    print(f"{speed['evaluations']} transcript evaluations in {speed['total_seconds']:.3f}s")
    print(f"Mean={speed['mean_ms']:.4f} ms  P50={speed['p50_ms']:.4f} ms  P95={speed['p95_ms']:.4f} ms  Throughput={speed['cases_per_second']:.1f} cases/s")
    print(f"\nJSON: {a.json}\nCSV : {a.csv}")
    # Non-zero on regression so CI/batch scripts can fail loudly.
    if pii['fp'] or pii['fn'] or pii['mask_failure_cases'] or prof['fp'] or prof['fn'] or prof['mask_failure_cases']:
        raise SystemExit(2)

if __name__=='__main__': main()
