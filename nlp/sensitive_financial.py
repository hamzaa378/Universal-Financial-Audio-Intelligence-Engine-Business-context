"""Sensitive financial identifiers that are not always classic PII.

These are separated from the PII detector so accuracy/FPR reporting for PII remains
meaningful. They are context-gated and can be independently enabled in the UI.
"""
from __future__ import annotations

import re
from typing import Iterable

_ID_VALUE = r"([A-Z0-9][A-Z0-9._/-]{4,31})"
_PATTERNS = [
    ("TRANSACTION_ID", re.compile(rf"(?i)\b(?:transaction|txn)\s*(?:id|reference|ref|number|no\.?)\s*(?:is|:|=|#|-)?\s*{_ID_VALUE}"), 0.96),
    ("LOAN_ID", re.compile(rf"(?i)\b(?:loan)\s*(?:id|account\s*(?:id|number|no\.?)|number|no\.?)\s*(?:is|:|=|#|-)?\s*{_ID_VALUE}"), 0.96),
    ("CUSTOMER_ID", re.compile(rf"(?i)\b(?:customer|client|borrower)\s*(?:id|number|no\.?)\s*(?:is|:|=|#|-)?\s*{_ID_VALUE}"), 0.95),
    ("APPLICATION_ID", re.compile(rf"(?i)\b(?:application|request)\s*(?:id|reference|ref|number|no\.?)\s*(?:is|:|=|#|-)?\s*{_ID_VALUE}"), 0.94),
    ("COMPLAINT_ID", re.compile(rf"(?i)\b(?:complaint|grievance|ticket|case)\s*(?:id|reference|ref|number|no\.?)\s*(?:is|:|=|#|-)?\s*{_ID_VALUE}"), 0.93),
]

# Values that are too generic to hide despite a loose textual match.
_STOP_VALUES = {"number", "unknown", "pending", "closed", "open", "none", "nil", "na", "n/a"}


def detect_sensitive_financial_ids(text: str) -> list[dict]:
    found=[]
    for kind, pat, conf in _PATTERNS:
        for m in pat.finditer(text):
            raw=m.group(1)
            value=raw.rstrip(".,:;#")
            if len(value) < 5 or value.casefold() in _STOP_VALUES:
                continue
            # Avoid treating plain years/very short amounts as identifiers.
            digits=''.join(ch for ch in value if ch.isdigit())
            if value.isdigit() and len(digits) < 6:
                continue
            s=m.start(1); e=s+len(value)
            found.append({
                "type":kind,"start":s,"end":e,"value":text[s:e],
                "confidence":conf,"evidence":"explicit_financial_identifier_context",
                "decision_method":"deterministic",
            })
    # Resolve overlaps by confidence, then longest.
    ranked=sorted(found,key=lambda x:(-x["confidence"],-(x["end"]-x["start"]),x["start"]))
    chosen=[]
    for item in ranked:
        if not any(item["start"] < x["end"] and x["start"] < item["end"] for x in chosen):
            chosen.append(item)
    return sorted(chosen,key=lambda x:x["start"])


def mask_sensitive_ids(text: str, entities: Iterable[dict]) -> str:
    out=text
    for e in sorted(entities,key=lambda x:x["start"],reverse=True):
        out=out[:e["start"]]+f"[{e['type']} REDACTED]"+out[e["end"]:]
    return out


def public_sensitive_id_metadata(entities: Iterable[dict]) -> list[dict]:
    return [{
        "type":e["type"],"start":e["start"],"end":e["end"],
        "confidence":e["confidence"],"evidence":e.get("evidence",""),
        "masked_value":f"[{e['type']} REDACTED]",
    } for e in entities]
