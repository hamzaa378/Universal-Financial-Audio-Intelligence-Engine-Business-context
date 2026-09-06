"""Combined privacy/safety redaction helpers with policy profiles."""
from __future__ import annotations

from nlp.pii import detect_pii, mask_pii
from nlp.profanity import detect_profanity
from nlp.sensitive_financial import detect_sensitive_financial_ids

PROFILE_THRESHOLDS={
    "high_precision":0.93,
    "balanced":0.80,
    "high_recall":0.72,
}


def protect_text(
    text: str,
    pii_mode: str = "full",
    profanity_replacement: str = "[BLEEP]",
    *,
    privacy_profile: str = "balanced",
    use_semantic: bool = False,
    use_ner: bool = False,
    mask_types: set[str] | None = None,
    profanity_enabled: bool = True,
    financial_ids_enabled: bool = False,
) -> dict:
    threshold=PROFILE_THRESHOLDS.get(privacy_profile,PROFILE_THRESHOLDS["balanced"])
    pii=detect_pii(text,min_confidence=threshold,use_semantic=use_semantic,use_ner=use_ner)
    if mask_types is not None:
        pii=[x for x in pii if x["type"] in mask_types]
    financial_ids=detect_sensitive_financial_ids(text) if financial_ids_enabled else []
    prof=detect_profanity(text) if profanity_enabled else []

    replacements=[]
    for e in pii:
        repl=mask_pii(text[e["start"]:e["end"]],[{**e,"start":0,"end":e["end"]-e["start"]}],mode=pii_mode)
        replacements.append((e["start"],e["end"],repl,"pii"))
    for e in financial_ids:
        replacements.append((e["start"],e["end"],f"[{e['type']} REDACTED]","financial_id"))
    for e in prof:
        replacements.append((e["start"],e["end"],profanity_replacement,"profanity"))

    # Privacy IDs win over profanity on overlap; longest span wins within a class.
    priority={"pii":3,"financial_id":2,"profanity":1}
    replacements.sort(key=lambda x:(priority[x[3]],x[1]-x[0],x[0]),reverse=True)
    kept=[]
    for r in replacements:
        if not any(r[0] < k[1] and k[0] < r[1] for k in kept):
            kept.append(r)
    protected=text
    for s,e,repl,_ in sorted(kept,key=lambda x:x[0],reverse=True):
        protected=protected[:s]+repl+protected[e:]
    return {
        "text":protected,
        "pii":pii,
        "financial_ids":financial_ids,
        "profanity":prof,
        "threshold":threshold,
    }
