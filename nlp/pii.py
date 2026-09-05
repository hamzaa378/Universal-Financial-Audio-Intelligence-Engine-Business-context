"""Context-aware PII detection for financial-call transcripts.

Design goals:
- high precision before recall for ambiguous numeric fields;
- deterministic validators for structured PII;
- context requirements for short/ambiguous values;
- overlap/conflict resolution so one span is never classified twice;
- safe masking helpers that do not expose the original value.
"""
from __future__ import annotations

import re
from typing import Iterable

# ---------- deterministic validators ----------

def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def luhn_valid(value: str) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6], [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4], [9,8,7,6,5,4,3,2,1,0],
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2], [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8],
]


def verhoeff_valid(value: str) -> bool:
    digits = _digits(value)
    if not digits:
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


# ---------- patterns and context ----------
_EMAIL = re.compile(r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w-])")
_PAN = re.compile(r"(?i)\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_IFSC = re.compile(r"(?i)\b[A-Z]{4}0[A-Z0-9]{6}\b")
_PHONE = re.compile(r"(?<!\d)(?:(?:\+?91|0)[\s.-]?)?[6-9](?:[\s.-]?\d){9}(?!\d)")
_AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_UPI = re.compile(r"(?i)(?<![\w.+-])[A-Z0-9._-]{2,64}@[A-Z][A-Z0-9_-]{1,31}(?![A-Z0-9_-])")
_LONG_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){8,17}\d(?!\d)")
_OTP = re.compile(r"(?<!\d)\d{4,8}(?!\d)")
_CVV = re.compile(r"(?<!\d)\d{3,4}(?!\d)")
_PINCODE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
_DOB = re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}(?!\d)")
_NAME = re.compile(r"(?i)\b(?:my\s+name\s+is|name\s*[:=])\s+([A-Z][A-Za-z.'-]{1,30}(?:\s+[A-Z][A-Za-z.'-]{1,30}){0,3})")
_ADDRESS = re.compile(
    r"(?i)\b(?:(?:my|your)\s+)?(?:(?:current|residential|registered|communication|permanent)\s+)?address\s*(?:is|:|=)\s*([^.!?\n]{5,140})"
)

UPI_HANDLES = {
    "upi", "ybl", "ibl", "axl", "paytm", "okaxis", "okhdfcbank", "okicici",
    "oksbi", "okyesbank", "apl", "airtel", "freecharge", "pingpay", "waicici",
}

PHONE_NEGATIVE_CONTEXT = re.compile(
    r"(?i)\b(?:order|transaction|txn|reference|ref|loan|customer|application|invoice|ticket)\s*(?:id|number|no\.?|#)?(?:\s+is)?\s*[:=-]?\s*$"
)
PHONE_POSITIVE_CONTEXT = re.compile(r"(?i)\b(?:phone|mobile|contact|call(?:\s+me)?|whatsapp|telephone)\b")
ACCOUNT_CONTEXT = re.compile(r"(?i)\b(?:account|a/c|acct|loan\s+account|bank\s+account)\s*(?:number|no\.?|#)?\b")
OTP_CONTEXT = re.compile(r"(?i)\b(?:otp|one[- ]time\s+(?:password|passcode)|verification\s+code|security\s+code|auth(?:entication)?\s+code)\b")
CVV_CONTEXT = re.compile(r"(?i)\b(?:cvv|cvc|card\s+security\s+code|card\s+verification\s+value)\b")
PIN_CONTEXT = re.compile(r"(?i)\b(?:pin\s*code|pincode|postal\s+code|zip\s+code)\b")
DOB_CONTEXT = re.compile(r"(?i)\b(?:dob|date\s+of\s+birth|born\s+on|birth\s+date)\b")
AADHAAR_CONTEXT = re.compile(r"(?i)\b(?:aadhaar|aadhar|uidai|uid\s+number)\b")
CARD_CONTEXT = re.compile(r"(?i)\b(?:card|credit\s+card|debit\s+card|visa|mastercard|rupay)\b")
UPI_CONTEXT = re.compile(r"(?i)\b(?:upi|vpa|virtual\s+payment\s+address|pay\s+id)\b")
ADDRESS_HINT = re.compile(
    r"(?i)(?:\d|\b(?:road|rd|street|st|lane|sector|nagar|colony|apartment|apt|flat|house|village|district|block|phase|floor|near|opp(?:osite)?|building)\b)"
)


def _window(text: str, start: int, end: int, radius: int = 44) -> str:
    return text[max(0, start-radius): min(len(text), end+radius)]


def _left_window(text: str, start: int, radius: int = 48) -> str:
    return text[max(0, start-radius):start]


def _item(kind: str, m: re.Match, confidence: float, evidence: str) -> dict:
    return {
        "type": kind,
        "start": m.start(),
        "end": m.end(),
        "value": m.group(0),
        "confidence": round(float(confidence), 3),
        "evidence": evidence,
    }


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _resolve_conflicts(items: Iterable[dict]) -> list[dict]:
    # Prefer high-confidence, more specific, longer spans.
    priority = {
        "EMAIL": 100, "PAN": 99, "IFSC": 98, "UPI": 97, "AADHAAR": 96,
        "CARD": 95, "PHONE": 90, "ACCOUNT_NUMBER": 85, "DOB": 80,
        "OTP": 78, "CVV": 77, "PINCODE": 75, "ADDRESS": 93, "NAME": 70,
    }
    ranked = sorted(
        items,
        key=lambda x: (-x["confidence"], -priority.get(x["type"], 0), -(x["end"]-x["start"]), x["start"]),
    )
    chosen: list[dict] = []
    for item in ranked:
        if not any(_overlaps(item, c) for c in chosen):
            chosen.append(item)
    return sorted(chosen, key=lambda x: (x["start"], x["end"]))


_SPOKEN_EMAIL = re.compile(
    r"(?i)\b([a-z0-9]+(?:\s+(?:dot|underscore|dash|hyphen)\s+[a-z0-9]+)*)\s+(?:at|at\s+the\s+rate(?:\s+of)?)\s+([a-z0-9-]+)\s+dot\s+([a-z]{2,})\b"
)
EMAIL_CONTEXT = re.compile(r"(?i)\b(?:email|e-mail|mail\s+id|email\s+address|registered\s+mail)\b")

_DIGIT_WORDS = {
    "zero":"0", "oh":"0", "one":"1", "two":"2", "three":"3", "four":"4",
    "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9",
}
_SPOKEN_DIGIT_RUN = re.compile(
    r"(?i)(?<!\w)(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple)(?:[\s-]+(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple)){2,}(?!\w)"
)


def _parse_spoken_digits(raw: str) -> str:
    out=[]; repeat=1
    for token in re.findall(r"(?i)zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple", raw):
        t=token.lower()
        if t=="double": repeat=2; continue
        if t=="triple": repeat=3; continue
        d=_DIGIT_WORDS[t]; out.extend([d]*repeat); repeat=1
    return ''.join(out)


def _spoken_candidates(text: str) -> list[dict]:
    out=[]
    # Spoken emails are common ASR output: "name at gmail dot com".
    for m in _SPOKEN_EMAIL.finditer(text):
        ctx=_window(text,m.start(),m.end(),42)
        domain=m.group(2).lower()
        if EMAIL_CONTEXT.search(ctx) or domain in {"gmail","yahoo","outlook","hotmail","protonmail","icloud"}:
            out.append({"type":"EMAIL","start":m.start(),"end":m.end(),"value":m.group(0),"confidence":0.93,"evidence":"spoken_email_asr"})
    # Spoken digit runs are only classified when a nearby label makes the type clear.
    for m in _SPOKEN_DIGIT_RUN.finditer(text):
        digits=_parse_spoken_digits(m.group(0))
        ctx=_window(text,m.start(),m.end(),48)
        kind=conf=evidence=None
        if PHONE_POSITIVE_CONTEXT.search(ctx) and len(digits)==10 and digits[:1] in "6789":
            kind,conf,evidence="PHONE",0.94,"spoken_phone_context"
        elif AADHAAR_CONTEXT.search(ctx) and len(digits)==12 and digits[:1] not in "01":
            kind,conf,evidence="AADHAAR",0.93,"spoken_aadhaar_context"
        elif CARD_CONTEXT.search(ctx) and 13<=len(digits)<=19 and luhn_valid(digits):
            kind,conf,evidence="CARD",0.97,"spoken_card_luhn"
        elif ACCOUNT_CONTEXT.search(ctx) and 9<=len(digits)<=18:
            kind,conf,evidence="ACCOUNT_NUMBER",0.93,"spoken_account_context"
        elif CVV_CONTEXT.search(ctx) and 3<=len(digits)<=4:
            kind,conf,evidence="CVV",0.95,"spoken_cvv_context"
        elif OTP_CONTEXT.search(ctx) and 4<=len(digits)<=8:
            kind,conf,evidence="OTP",0.95,"spoken_otp_context"
        elif PIN_CONTEXT.search(ctx) and len(digits)==6 and digits[:1] != "0":
            kind,conf,evidence="PINCODE",0.92,"spoken_postal_context"
        if kind:
            out.append({"type":kind,"start":m.start(),"end":m.end(),"value":m.group(0).strip(),"confidence":conf,"evidence":evidence})
    return out


def detect_pii(text: str, min_confidence: float = 0.80) -> list[dict]:
    """Return high-confidence PII spans.

    Ambiguous numeric types require local financial/UI context. Structured identifiers
    use validators where available. `min_confidence` can be raised in high-precision mode.
    """
    found: list[dict] = []

    # Strong lexical structures.
    for m in _EMAIL.finditer(text):
        found.append(_item("EMAIL", m, 0.995, "email_structure"))
    for m in _PAN.finditer(text):
        found.append(_item("PAN", m, 0.995, "pan_structure"))
    for m in _IFSC.finditer(text):
        found.append(_item("IFSC", m, 0.995, "ifsc_structure"))

    # UPI/VPA: do not mistake normal email domains for UPI handles.
    for m in _UPI.finditer(text):
        provider = m.group(0).rsplit("@", 1)[-1].lower()
        ctx = _window(text, m.start(), m.end())
        if provider in UPI_HANDLES:
            found.append(_item("UPI", m, 0.985, "known_upi_handle"))
        elif UPI_CONTEXT.search(ctx):
            found.append(_item("UPI", m, 0.93, "upi_context"))

    # Cards: Luhn is the main false-positive barrier.
    for m in _CARD.finditer(text):
        value = m.group(0)
        digits = _digits(value)
        if 13 <= len(digits) <= 19 and luhn_valid(value):
            ctx = _window(text, m.start(), m.end())
            conf = 0.995 if CARD_CONTEXT.search(ctx) else 0.965
            found.append(_item("CARD", m, conf, "luhn_valid"))

    # Aadhaar: checksum gives strongest evidence. A checksum-invalid candidate is
    # accepted only when Aadhaar is explicitly mentioned, useful for synthetic demos.
    for m in _AADHAAR.finditer(text):
        value = m.group(0)
        if len(_digits(value)) != 12:
            continue
        ctx = _window(text, m.start(), m.end())
        if verhoeff_valid(value):
            found.append(_item("AADHAAR", m, 0.995, "verhoeff_valid"))
        elif AADHAAR_CONTEXT.search(ctx):
            found.append(_item("AADHAAR", m, 0.91, "aadhaar_context_checksum_unverified"))

    # Indian phone. Suppress identifier-like contexts unless a phone cue is nearby.
    for m in _PHONE.finditer(text):
        digits = _digits(m.group(0))
        local = digits[-10:]
        if len(local) != 10 or local[0] not in "6789":
            continue
        left = _left_window(text, m.start())
        ctx = _window(text, m.start(), m.end())
        positive = bool(PHONE_POSITIVE_CONTEXT.search(ctx))
        negative = bool(PHONE_NEGATIVE_CONTEXT.search(left))
        if negative and not positive:
            continue
        separated = bool(re.search(r"[ +.\-]", m.group(0)))
        conf = 0.985 if positive else (0.94 if separated or digits.startswith("91") else 0.90)
        found.append(_item("PHONE", m, conf, "phone_context" if positive else "phone_structure"))

    # Account numbers must have account context; otherwise long numbers are too ambiguous.
    for m in _LONG_NUMBER.finditer(text):
        digits = _digits(m.group(0))
        if not 9 <= len(digits) <= 18:
            continue
        ctx = _window(text, m.start(), m.end())
        if ACCOUNT_CONTEXT.search(ctx):
            found.append(_item("ACCOUNT_NUMBER", m, 0.95, "account_context"))

    # OTP and CVV are intentionally context-only to reduce false positives.
    for m in _OTP.finditer(text):
        ctx = _window(text, m.start(), m.end(), 34)
        if OTP_CONTEXT.search(ctx) and not CVV_CONTEXT.search(ctx):
            found.append(_item("OTP", m, 0.97, "otp_context"))
    for m in _CVV.finditer(text):
        ctx = _window(text, m.start(), m.end(), 30)
        if CVV_CONTEXT.search(ctx):
            found.append(_item("CVV", m, 0.97, "cvv_context"))

    for m in _PINCODE.finditer(text):
        ctx = _window(text, m.start(), m.end(), 40)
        if PIN_CONTEXT.search(ctx) or re.search(r"(?i)\baddress\b", ctx):
            found.append(_item("PINCODE", m, 0.94, "postal_context"))

    for m in _DOB.finditer(text):
        ctx = _window(text, m.start(), m.end(), 36)
        if DOB_CONTEXT.search(ctx):
            found.append(_item("DOB", m, 0.96, "dob_context"))

    # Context-only name; capture only the actual name group, not the label words.
    for m in _NAME.finditer(text):
        s, e = m.span(1)
        value = text[s:e].strip()
        if value.lower() not in {"not available", "not disclosed", "unknown"}:
            found.append({"type":"NAME","start":s,"end":e,"value":value,"confidence":0.92,"evidence":"explicit_name_context"})

    # Address requires an explicit address label + address-like content to avoid
    # masking explanatory sentences such as "address is required for KYC".
    for m in _ADDRESS.finditer(text):
        body = m.group(1).strip(" ,")
        if ADDRESS_HINT.search(body):
            s = m.start(1)
            e = s + len(m.group(1))
            # Trim trailing whitespace/punctuation while preserving source offsets.
            while e > s and text[e-1] in " ,;":
                e -= 1
            found.append({"type":"ADDRESS","start":s,"end":e,"value":text[s:e],"confidence":0.96,"evidence":"explicit_address_context"})

    found.extend(_spoken_candidates(text))

    resolved = _resolve_conflicts(x for x in found if x["confidence"] >= min_confidence)
    return resolved


def _partial_mask(kind: str, value: str) -> str:
    digits = _digits(value)
    if kind in {"PHONE", "CARD", "ACCOUNT_NUMBER", "AADHAAR"} and len(digits) >= 4:
        return f"[{kind} ••••{digits[-4:]}]"
    if kind == "EMAIL" and "@" in value:
        local, domain = value.split("@", 1)
        return f"[{kind} {local[:1]}***@{domain}]"
    return f"[{kind} REDACTED]"


def mask_pii(text: str, entities: Iterable[dict], mode: str = "full") -> str:
    """Mask PII without shifting source detection spans before replacement.

    mode='full' replaces the complete value. mode='partial' preserves only a small
    non-sensitive suffix for selected financial identifiers.
    """
    masked = text
    for e in sorted(entities, key=lambda x: x["start"], reverse=True):
        repl = _partial_mask(e["type"], e.get("value", "")) if mode == "partial" else f"[{e['type']} REDACTED]"
        masked = masked[:e["start"]] + repl + masked[e["end"]:]
    return masked


def public_pii_metadata(entities: Iterable[dict]) -> list[dict]:
    """Return metadata safe for normal logs/UI; never include the raw PII value."""
    out = []
    for e in entities:
        out.append({
            "type": e["type"], "start": e["start"], "end": e["end"],
            "confidence": e["confidence"], "evidence": e.get("evidence", ""),
            "masked_value": _partial_mask(e["type"], e.get("value", "")),
        })
    return out
