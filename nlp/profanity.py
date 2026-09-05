"""High-precision profanity detection and transcript de-escalation.

The detector intentionally avoids broad substring matching. It catches common English
and transliterated Hindi/Hinglish profanity plus simple obfuscation (f**k, b!tch, etc.)
without matching benign words such as 'assistant', 'class', or 'asset'.
"""
from __future__ import annotations
import re
from typing import Iterable

# No protected-class slurs are included. Add organisation-approved vocabulary in a
# separate policy layer rather than hard-coding it in business logic.
_TERMS = {
    "fuck": 3, "fucker": 3, "fucking": 3, "motherfucker": 3,
    "shit": 2, "shitty": 2, "bitch": 3, "bastard": 2, "asshole": 3,
    "madarchod": 3, "behenchod": 3, "bhenchod": 3, "chutiya": 3,
    "harami": 2,
}

_LEET = {
    "a": "[a@4]", "b": "[b8]", "e": "[e3]", "i": "[i1!|]",
    "l": "[l1|]", "o": "[o0]", "s": "[s5$]", "t": "[t7+]",
    "g": "[g9]", "u": "[u]", "c": "[c(]", "h": "[h]", "f": "[f]",
    "k": "[k]", "r": "[r]", "m": "[m]", "d": "[d]", "y": "[y]",
    "n": "[n]",
}


def _term_pattern(term: str) -> re.Pattern:
    chars = [_LEET.get(ch, re.escape(ch)) for ch in term]
    # Permit masking punctuation/underscores between letters, but not ordinary spaces;
    # this avoids swallowing long benign phrases.
    core = r"[\W_]*".join(chars)
    return re.compile(rf"(?i)(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])")


_PATTERNS = [(term, severity, _term_pattern(term)) for term, severity in _TERMS.items()]
# Very common spaced-letter obfuscations. Kept separate to avoid general overmatching.
_CENSORED = [
    ("fuck", 3, re.compile(r"(?i)(?<!\w)f[*#._-]{1,4}k(?!\w)")),
    ("shit", 2, re.compile(r"(?i)(?<!\w)s[*#._-]{1,4}t(?!\w)")),
    ("bitch", 3, re.compile(r"(?i)(?<!\w)b[*#._-]{1,6}h(?!\w)")),
]

_SPACED = [
    ("fuck", 3, re.compile(r"(?i)(?<!\w)f\s+u\s+c\s+k(?!\w)")),
    ("shit", 2, re.compile(r"(?i)(?<!\w)s\s+h\s+i\s+t(?!\w)")),
]


def detect_profanity(text: str, min_severity: int = 1) -> list[dict]:
    found = []
    for term, severity, pattern in _PATTERNS + _CENSORED + _SPACED:
        if severity < min_severity:
            continue
        for m in pattern.finditer(text):
            raw = m.group(0)
            exact = raw.lower() == term
            found.append({
                "type": "PROFANITY",
                "category": "abusive_language",
                "severity": severity,
                "start": m.start(), "end": m.end(),
                "value": raw,
                "canonical": term,
                "confidence": 0.98 if exact else 0.90,
            })
    # Resolve duplicates/overlaps by severity -> confidence -> longest span.
    ranked = sorted(found, key=lambda x: (-x["severity"], -x["confidence"], -(x["end"]-x["start"]), x["start"]))
    chosen = []
    for item in ranked:
        if not any(item["start"] < c["end"] and c["start"] < item["end"] for c in chosen):
            chosen.append(item)
    return sorted(chosen, key=lambda x: x["start"])


def reduce_profanity(text: str, entities: Iterable[dict], replacement: str = "[BLEEP]") -> str:
    cleaned = text
    for e in sorted(entities, key=lambda x: x["start"], reverse=True):
        cleaned = cleaned[:e["start"]] + replacement + cleaned[e["end"]:]
    return cleaned


def public_profanity_metadata(entities: Iterable[dict]) -> list[dict]:
    return [{k: e[k] for k in ("type","category","severity","start","end","confidence") if k in e} for e in entities]
