"""Financial entity extraction for call transcripts.

This layer intentionally separates financial facts from PII. Amounts/rates/statuses are
useful to reviewers and do not need to be hidden, while account/reference identifiers
are handled by the privacy/sensitive-ID layers.
"""
from __future__ import annotations

import re

_NUM = r"[0-9][0-9,]*(?:\.[0-9]{1,2})?"
_CURRENCY = r"(?:₹|rs\.?|inr|rupees?)"

MONEY = re.compile(rf"(?i)(?:{_CURRENCY}\s*({_NUM})|\b({_NUM})\s*(?:rupees?|inr)\b)")
RATE = re.compile(r"(?i)(?<!\d)(\d{1,2}(?:\.\d+)?)\s*(?:%|percent|per\s*cent)\b")
TENURE = re.compile(r"(?i)\b(\d{1,3})\s*(months?|yrs?|years?)\b")
DATE = re.compile(r"(?i)\b(today|tomorrow|day after tomorrow|(?:mon|tues?|wednes|thurs?|fri|satur|sun)day|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b")
PAYMENT_METHOD = re.compile(r"(?i)\b(upi|vpa|neft|rtgs|imps|nach|e-?mandate|auto\s*debit|debit\s*card|credit\s*card|cash|bank\s*transfer)\b")
STATUS = re.compile(r"(?i)\b(overdue|paid|payment\s+completed|pending|bounced|bounce(?:d)?|failed|reversed|settled|waived|foreclosed|closed)\b")

_CONTEXT_AMOUNT_TYPES = [
    ("EMI_AMOUNT", re.compile(r"(?i)\b(?:emi|instal+ment)\b")),
    ("OUTSTANDING_AMOUNT", re.compile(r"(?i)\b(?:outstanding|balance\s+due|total\s+outstanding)\b")),
    ("DUE_AMOUNT", re.compile(r"(?i)\b(?:amount\s+due|due\s+amount|payment\s+due)\b")),
    ("LOAN_AMOUNT", re.compile(r"(?i)\b(?:loan\s+amount|principal\s+amount|sanctioned\s+amount|disbursed\s+amount)\b")),
    ("SETTLEMENT_AMOUNT", re.compile(r"(?i)\b(?:settlement\s+amount|settle\s+for|one[- ]time\s+settlement)\b")),
    ("LATE_FEE", re.compile(r"(?i)\b(?:late\s+fee|late\s+charge|penalty|bounce\s+charge)\b")),
]


def _local_left(text: str, start: int, radius: int = 48) -> str:
    lo=max(0,start-radius)
    for sep in (".","!","?","\n",";"):
        p=text.rfind(sep,lo,start)
        if p>=lo:
            lo=p+1
    return text[lo:start]


def _money_value(m: re.Match) -> str:
    return (m.group(1) or m.group(2)).replace(",","")


def extract_entities(text: str) -> list[dict]:
    out=[]
    # Amounts: classify using the closest left financial cue, otherwise keep MONEY.
    for m in MONEY.finditer(text):
        left=_local_left(text,m.start(),60)
        kind="MONEY"; conf=0.94
        nearest=None
        for candidate,ctx in _CONTEXT_AMOUNT_TYPES:
            matches=list(ctx.finditer(left))
            if matches:
                last=matches[-1]
                distance=len(left)-last.end()
                if nearest is None or distance < nearest[0]:
                    nearest=(distance,candidate)
        if nearest is not None:
            kind=nearest[1]; conf=0.97
        out.append({
            "type":kind,"value":_money_value(m),"start":m.start(),"end":m.end(),
            "confidence":conf,"evidence":"currency+local_context" if kind!="MONEY" else "currency_structure",
        })

    for m in RATE.finditer(text):
        left=_local_left(text,m.start(),56)
        label="APR" if re.search(r"(?i)\bapr\b",left) else "INTEREST_RATE"
        out.append({"type":label,"value":m.group(1)+"%","start":m.start(),"end":m.end(),"confidence":0.96,"evidence":"rate_structure"})

    for m in TENURE.finditer(text):
        left=_local_left(text,m.start(),48)
        if re.search(r"(?i)\b(?:tenure|term|repayment|loan)\b",left):
            unit=m.group(2).lower()
            out.append({"type":"TENURE","value":f"{m.group(1)} {unit}","start":m.start(),"end":m.end(),"confidence":0.93,"evidence":"tenure_context"})

    for m in DATE.finditer(text):
        left=_local_left(text,m.start(),52)
        if re.search(r"(?i)\b(?:due|pay|payment|promise|before|by|on|date|deadline)\b",left):
            kind="DUE_DATE" if re.search(r"(?i)\b(?:due|deadline|by)\b",left) else "PAYMENT_DATE"
            out.append({"type":kind,"value":m.group(0),"start":m.start(),"end":m.end(),"confidence":0.88,"evidence":"date+payment_context"})

    for m in PAYMENT_METHOD.finditer(text):
        out.append({"type":"PAYMENT_METHOD","value":m.group(1).upper(),"start":m.start(),"end":m.end(),"confidence":0.95,"evidence":"payment_method_lexicon"})

    for m in STATUS.finditer(text):
        out.append({"type":"PAYMENT_STATUS","value":m.group(1).lower(),"start":m.start(),"end":m.end(),"confidence":0.90,"evidence":"financial_status_lexicon"})

    # Prefer more specific contextual entities over generic ones when spans overlap.
    priority={
        "EMI_AMOUNT":10,"OUTSTANDING_AMOUNT":10,"DUE_AMOUNT":10,"LOAN_AMOUNT":10,
        "SETTLEMENT_AMOUNT":10,"LATE_FEE":10,"APR":9,"INTEREST_RATE":9,
        "TENURE":8,"DUE_DATE":8,"PAYMENT_DATE":8,"PAYMENT_METHOD":7,
        "PAYMENT_STATUS":7,"MONEY":1,
    }
    ranked=sorted(out,key=lambda x:(-priority.get(x["type"],0),-x["confidence"],x["start"]))
    chosen=[]
    for item in ranked:
        # Keep distinct semantic facts even if their text merely touches, but avoid exact overlap duplicates.
        if any(item["start"] < c["end"] and c["start"] < item["end"] and item["type"]==c["type"] for c in chosen):
            continue
        chosen.append(item)
    return sorted(chosen,key=lambda x:(x["start"],x["end"],x["type"]))
