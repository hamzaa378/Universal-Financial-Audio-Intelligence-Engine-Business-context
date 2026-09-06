"""High-precision profanity detection for financial-call transcripts.

The detector uses word boundaries and controlled obfuscation patterns instead of broad
substring matching. This reduces the classic false positives in words such as
"assistant", "asset", and "class" while covering common English/Hinglish forms that
ASR systems tend to emit.
"""
from __future__ import annotations

import re
from typing import Iterable

# canonical -> severity, accepted lexical variants
_LEXICON = {
    "fuck": (3, ("fuck", "fucker", "fucking", "fucked", "motherfucker", "motherfucking")),
    "shit": (2, ("shit", "shitty", "bullshit")),
    "bitch": (3, ("bitch", "bitches")),
    "bastard": (2, ("bastard", "bastards")),
    "asshole": (3, ("asshole", "arsehole")),
    "madarchod": (3, ("madarchod", "madharchod", "madar chod", "madhar chod")),
    "bhenchod": (3, ("bhenchod", "behenchod", "bhen chod", "behen chod")),
    "chutiya": (3, ("chutiya", "chutiye", "chutya")),
    "harami": (2, ("harami", "haraami")),
    "gandu": (3, ("gandu", "gaandu")),
    "bakchod": (2, ("bakchod", "bakchodi")),
}

# Leet substitutions are allowed within a single token/obfuscation, not across arbitrary
# sentence whitespace. Spaced-letter variants are handled separately below.
_LEET = {
    "a": "[a@4]", "b": "[b8]", "e": "[e3]", "i": "[i1!|]",
    "l": "[l1|]", "o": "[o0]", "s": "[s5$]", "t": "[t7+]",
    "g": "[g9]", "u": "u", "c": "[c(]", "h": "h", "f": "f",
    "k": "k", "r": "r", "m": "m", "d": "d", "y": "y", "n": "n",
}


def _variant_pattern(variant: str) -> re.Pattern:
    parts=[]
    for ch in variant:
        if ch.isspace():
            parts.append(r"\s+")
        elif ch.isalpha():
            parts.append(_LEET.get(ch.lower(), re.escape(ch)))
        else:
            parts.append(re.escape(ch))
    # Permit censor punctuation between letters, but not free whitespace unless the
    # lexical variant itself contains a space.
    core=""
    for i,part in enumerate(parts):
        if i and not (part == r"\s+" or parts[i-1] == r"\s+"):
            core += r"[*#._-]*"
        core += part
    return re.compile(rf"(?i)(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])")


_PATTERNS=[]
for canonical,(severity,variants) in _LEXICON.items():
    for variant in variants:
        _PATTERNS.append((canonical,severity,variant,_variant_pattern(variant)))

# Common aggressively censored forms where interior letters disappear entirely.
_CENSORED = [
    ("fuck",3,re.compile(r"(?i)(?<!\w)f[*#._-]{1,6}k(?!\w)")),
    ("shit",2,re.compile(r"(?i)(?<!\w)s[*#._-]{1,6}t(?!\w)")),
    ("bitch",3,re.compile(r"(?i)(?<!\w)b[*#._-]{1,8}h(?!\w)")),
]

_SPACED = [
    ("fuck",3,re.compile(r"(?i)(?<!\w)f\s+u\s+c\s+k(?!\w)")),
    ("shit",2,re.compile(r"(?i)(?<!\w)s\s+h\s+i\s+t(?!\w)")),
    ("bitch",3,re.compile(r"(?i)(?<!\w)b\s+i\s+t\s+c\s+h(?!\w)")),
]


def detect_profanity(text: str, min_severity: int = 1) -> list[dict]:
    found=[]
    for canonical,severity,variant,pattern in _PATTERNS:
        if severity < min_severity:
            continue
        for m in pattern.finditer(text):
            raw=m.group(0)
            normalized=re.sub(r"[^a-z]","",raw.casefold())
            exact=raw.casefold() == variant.casefold()
            found.append({
                "type":"PROFANITY",
                "category":"abusive_language",
                "severity":severity,
                "start":m.start(),"end":m.end(),
                "value":raw,
                "canonical":canonical,
                "confidence":0.99 if exact else 0.95 if normalized else 0.92,
                "evidence":"lexical_exact" if exact else "lexical_variant_or_obfuscation",
            })
    for canonical,severity,pattern in _CENSORED + _SPACED:
        if severity < min_severity:
            continue
        for m in pattern.finditer(text):
            found.append({
                "type":"PROFANITY","category":"abusive_language","severity":severity,
                "start":m.start(),"end":m.end(),"value":m.group(0),"canonical":canonical,
                "confidence":0.94,"evidence":"censored_or_spaced_obfuscation",
            })
    ranked=sorted(found,key=lambda x:(-x["severity"],-x["confidence"],-(x["end"]-x["start"]),x["start"]))
    chosen=[]
    for item in ranked:
        if not any(item["start"] < c["end"] and c["start"] < item["end"] for c in chosen):
            chosen.append(item)
    return sorted(chosen,key=lambda x:x["start"])


def reduce_profanity(text: str, entities: Iterable[dict], replacement: str = "[BLEEP]") -> str:
    cleaned=text
    for e in sorted(entities,key=lambda x:x["start"],reverse=True):
        cleaned=cleaned[:e["start"]]+replacement+cleaned[e["end"]:]
    return cleaned


def public_profanity_metadata(entities: Iterable[dict]) -> list[dict]:
    # Never expose the raw profane token in normal UI/API metadata.
    keys=("type","category","severity","start","end","confidence","canonical","evidence")
    return [{k:e[k] for k in keys if k in e} for e in entities]
