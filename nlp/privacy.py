"""Combined privacy/safety redaction helpers."""
from __future__ import annotations
from nlp.pii import detect_pii, mask_pii
from nlp.profanity import detect_profanity, reduce_profanity


def protect_text(text: str, pii_mode: str = "full", profanity_replacement: str = "[BLEEP]") -> dict:
    pii = detect_pii(text)
    prof = detect_profanity(text)
    # Apply PII first. Profanity spans refer to the original text, so apply both in one
    # reverse-ordered replacement pass to keep offsets correct.
    replacements = []
    for e in pii:
        repl = mask_pii(text[e["start"]:e["end"]], [{**e, "start":0, "end":e["end"]-e["start"]}], mode=pii_mode)
        replacements.append((e["start"], e["end"], repl, "pii"))
    for e in prof:
        replacements.append((e["start"], e["end"], profanity_replacement, "profanity"))
    # PII wins on any unlikely overlap.
    replacements.sort(key=lambda x: (x[0], 1 if x[3]=="pii" else 0, x[1]-x[0]), reverse=True)
    kept = []
    for r in replacements:
        if not any(r[0] < k[1] and k[0] < r[1] for k in kept):
            kept.append(r)
    protected = text
    for s,e,repl,_ in sorted(kept, key=lambda x:x[0], reverse=True):
        protected = protected[:s] + repl + protected[e:]
    return {"text": protected, "pii": pii, "profanity": prof}
