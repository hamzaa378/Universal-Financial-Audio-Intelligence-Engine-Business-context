from functools import lru_cache
import os, shutil
from config import SETTINGS

@lru_cache(maxsize=2)
def _model(model_name: str):
    from faster_whisper import WhisperModel
    device=SETTINGS.device
    compute=SETTINGS.compute_type
    if device=="auto":
        # ctranslate2 CUDA runtime is independent of torch; fall back cleanly if unavailable.
        device="cuda" if os.getenv("FINAI_FORCE_CPU","0")!="1" else "cpu"
    if compute=="auto": compute="float16" if device=="cuda" else "int8"
    try:
        return WhisperModel(model_name, device=device, compute_type=compute)
    except Exception:
        if device!="cpu": return WhisperModel(model_name, device="cpu", compute_type="int8")
        raise

def _domain_prompt():
    vocab=", ".join(SETTINGS.financial_vocab)
    return "Financial services customer call. Preserve numbers, dates, currencies, rates and financial acronyms accurately. Vocabulary: "+vocab+"."

def transcribe(audio_path: str, model_name: str|None=None):
    model=_model(model_name or SETTINGS.whisper_model)
    kwargs=dict(beam_size=SETTINGS.beam_size,best_of=SETTINGS.best_of,word_timestamps=True,
                vad_filter=SETTINGS.vad_filter,condition_on_previous_text=True,
                initial_prompt=_domain_prompt(),temperature=0.0)
    segments, info=model.transcribe(audio_path, **kwargs)
    words=[]; segs=[]
    for s in segments:
        sw=[]
        for w in (s.words or []):
            item={"word":w.word.strip(),"start":round(float(w.start),3),"end":round(float(w.end),3),"confidence":round(float(w.probability),4)}
            words.append(item); sw.append(item)
        conf=sum(x["confidence"] for x in sw)/len(sw) if sw else max(0.0,min(1.0,float(__import__('math').exp(s.avg_logprob))))
        segs.append({"id":s.id,"start":round(float(s.start),3),"end":round(float(s.end),3),"text":s.text.strip(),"confidence":round(conf,4),"words":sw})
    text=" ".join(s["text"] for s in segs).strip()
    overall=sum(w["confidence"] for w in words)/len(words) if words else 0.0
    return {"text":text,"segments":segs,"words":words,"language":getattr(info,"language",None),"language_probability":round(float(getattr(info,"language_probability",0.0) or 0.0),4),"confidence":round(overall,4),"model":model_name or SETTINGS.whisper_model}
