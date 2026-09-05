import langid

def annotate_segment_languages(segments):
    out=[]
    for s in segments:
        text=s.get("text","").strip()
        lang, raw=langid.classify(text) if text else ("unknown",0.0)
        # langid raw score is not a calibrated probability; keep it explicitly separate.
        x=dict(s); x["text_language"]={"code":lang,"raw_score":round(float(raw),3)}; out.append(x)
    return out
