"""Evaluate v4.5 on a labeled real/synthetic audio manifest.

Manifest JSONL row example:
{"id":"c1","audio":"audio/c1.wav","transcript":"my phone is 987...",
 "pii":[{"type":"PHONE","value":"9876543210"}],"profanity":["shit"],
 "gold_intervals":[{"type":"PHONE","start":1.2,"end":2.0}]}

Use consented/synthetic audio. Character/text metrics and audio-timing metrics are kept
separate so ASR errors are not confused with privacy-detector errors.
"""
from __future__ import annotations

import argparse,json,re,time,sys
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from pipeline import run_pipeline
from nlp.pii import detect_pii
from nlp.profanity import detect_profanity
from asr.fintech_asr import warmup_model
from decision_ai.warmup import warmup_semantic_ai
from decision_ai.ner_engine import status as ner_status


def _words(s:str)->list[str]:
    return re.findall(r"[\w@.+-]+",s.casefold(),flags=re.UNICODE)


def _lev(a:list[str],b:list[str])->int:
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def wer(ref:str,hyp:str)->float:
    r=_words(ref); h=_words(hyp)
    return _lev(r,h)/max(1,len(r))


def _norm_value(v:str)->str:
    return re.sub(r"[^a-z0-9]+","",str(v).casefold())


def _entity_counts(expected:list[dict],detected:list[dict])->tuple[int,int,int]:
    gold=[(str(x.get('type','')).upper(),_norm_value(x.get('value',''))) for x in expected]
    pred=[(str(x.get('type','')).upper(),_norm_value(x.get('value',''))) for x in detected]
    used=[False]*len(pred); tp=0
    for g in gold:
        for i,p in enumerate(pred):
            if not used[i] and p==g:
                used[i]=True; tp+=1; break
    return tp,sum(not x for x in used),len(gold)-tp


def _profanity_counts(expected:list[str],detected:list[dict])->tuple[int,int,int]:
    gold=[_norm_value(x) for x in expected]
    pred=[_norm_value(x.get('value','')) for x in detected]
    used=[False]*len(pred); tp=0
    for g in gold:
        for i,p in enumerate(pred):
            if not used[i] and p==g:
                used[i]=True; tp+=1; break
    return tp,sum(not x for x in used),len(gold)-tp


def _gold_audio_coverage(gold:list[dict],pred:list[dict])->float|None:
    if not gold: return None
    vals=[]
    for g in gold:
        gs=float(g['start']); ge=float(g['end']); dur=max(1e-6,ge-gs)
        inter=0.0
        for p in pred:
            if g.get('type') and g.get('type') not in (p.get('types') or []):
                continue
            inter=max(inter,max(0.0,min(ge,float(p['end']))-max(gs,float(p['start']))))
        vals.append(min(1.0,inter/dur))
    return mean(vals) if vals else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--model',default='small')
    ap.add_argument('--speed',choices=['fast','balanced','accuracy'],default='fast')
    ap.add_argument('--semantic',action='store_true')
    ap.add_argument('--ner',action='store_true')
    ap.add_argument('--out',default='reports/audio_manifest_report.json')
    a=ap.parse_args()

    mp=Path(a.manifest).resolve(); base=mp.parent
    rows=[json.loads(x) for x in mp.read_text(encoding='utf-8').splitlines() if x.strip()]
    if not rows: raise SystemExit('Manifest is empty')

    warm=warmup_model(a.model,smoke_test=True)
    sem=warmup_semantic_ai() if a.semantic else {'available':False,'backend':'disabled'}
    ner=ner_status() if a.ner else {'available':False,'backend':'disabled'}
    totals={'pii_tp':0,'pii_fp':0,'pii_fn':0,'prof_tp':0,'prof_fp':0,'prof_fn':0}
    results=[]

    for row in rows:
        audio=Path(row['audio'])
        if not audio.is_absolute(): audio=(base/audio).resolve()
        t0=time.perf_counter()
        r=run_pipeline(str(audio),a.model,include_raw=True,use_semantic_ai=a.semantic,use_ner_ai=a.ner,
                       create_audio_output=True,asr_speed_mode=a.speed)
        elapsed=time.perf_counter()-t0
        hyp=r['transcription']['text']; ref=row.get('transcript','')
        raw_pii=detect_pii(hyp,use_semantic=a.semantic,use_ner=a.ner)
        raw_prof=detect_profanity(hyp)
        pt,pf,pn=_entity_counts(row.get('pii',[]),raw_pii)
        qt,qf,qn=_profanity_counts(row.get('profanity',[]),raw_prof)
        totals['pii_tp']+=pt; totals['pii_fp']+=pf; totals['pii_fn']+=pn
        totals['prof_tp']+=qt; totals['prof_fp']+=qf; totals['prof_fn']+=qn
        meta=r['understanding'].get('pii',[])
        aligned=sum(bool(x.get('token_ids')) for x in meta)
        alignment_coverage=aligned/max(1,len(meta)) if meta else 1.0
        pa=r['privacy'].get('protected_audio') or {}
        audio_cov=_gold_audio_coverage(row.get('gold_intervals',[]),pa.get('intervals',[]))
        results.append({
            'id':row.get('id',audio.stem),'audio':str(audio),'wer':round(wer(ref,hyp),4) if ref else None,
            'pii_tp':pt,'pii_fp':pf,'pii_fn':pn,'profanity_tp':qt,'profanity_fp':qf,'profanity_fn':qn,
            'token_alignment_coverage':round(alignment_coverage,4),
            'gold_audio_redaction_coverage':round(audio_cov,4) if audio_cov is not None else None,
            'processing_s':round(elapsed,3),'rtf':r['performance'].get('real_time_factor'),
            'asr_s':round(r['performance'].get('asr',0)/1000,4),'text_ai_ms':r['performance'].get('privacy_and_nlp'),
            'asr_device':r['transcription'].get('backend',{}).get('device'),
            'asr_compute':r['transcription'].get('backend',{}).get('compute_type'),
            'semantic_provider':r['privacy'].get('semantic_ai',{}).get('provider'),
            'ner_provider':r['privacy'].get('ner_ai',{}).get('provider'),
        })
        path=pa.get('path')
        if path:
            try: Path(path).unlink(missing_ok=True)
            except Exception: pass
        print(results[-1])

    def prf(tp,fp,fn):
        p=tp/max(1,tp+fp); rec=tp/max(1,tp+fn); f=2*p*rec/max(1e-9,p+rec)
        return {'precision':p,'recall':rec,'f1':f}
    report={
        'manifest':str(mp),'model':a.model,'speed':a.speed,
        'asr_warmup':warm,'semantic_status':sem,'ner_status':ner,
        'pii':{**totals,**prf(totals['pii_tp'],totals['pii_fp'],totals['pii_fn'])},
        'profanity':prf(totals['prof_tp'],totals['prof_fp'],totals['prof_fn']),
        'mean_wer':mean([x['wer'] for x in results if x['wer'] is not None]) if any(x['wer'] is not None for x in results) else None,
        'mean_rtf':mean([x['rtf'] for x in results if isinstance(x['rtf'],(int,float))]),
        'mean_token_alignment_coverage':mean(x['token_alignment_coverage'] for x in results),
        'rows':results,
        'note':'Use consented or synthetic labeled audio. Text-regression scores are not substitutes for this audio benchmark.'
    }
    op=Path(a.out); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Wrote',op)

if __name__=='__main__': main()
