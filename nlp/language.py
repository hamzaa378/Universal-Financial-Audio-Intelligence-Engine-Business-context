from __future__ import annotations


def annotate_segment_languages(segments):
    """Add lightweight text-language hints without making langid a hard import-time dependency."""
    try:
        import langid
    except Exception:
        return [{**s,"text_language":{"code":"unknown","raw_score":0.0,"method":"unavailable"}} for s in segments]
    out=[]
    for s in segments:
        text=s.get("text","").strip()
        if len(text) < 3:
            lang,raw="unknown",0.0
        else:
            lang,raw=langid.classify(text)
        x=dict(s)
        x["text_language"]={"code":lang,"raw_score":round(float(raw),3),"method":"langid_text_hint"}
        out.append(x)
    return out
