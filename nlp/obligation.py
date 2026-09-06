"""Obligation, promise-to-pay, and repayment-difficulty extraction."""
from __future__ import annotations

from datetime import datetime, timedelta
import re

PROMISE = re.compile(
    r"(?i)\b(?:i\s+(?:will|'ll|can|shall)\s+(?:pay|make\s+the\s+payment|clear\s+(?:it|the\s+dues?))|"
    r"promise\s+to\s+pay|payment\s+(?:will|shall)\s+be\s+(?:made|done)|"
    r"i\s+will\s+arrange\s+(?:the\s+)?(?:money|funds)|pay\s+by|"
    r"main\s+(?:kal|aaj)\s+(?:payment|emi)\s+(?:kar|bhar)\s*(?:dunga|dungi)?)\b"
)
DIFFICULTY = re.compile(
    r"(?i)\b(?:i\s+(?:cannot|can't|cant|am\s+unable\s+to)\s+pay|unable\s+to\s+pay|"
    r"need\s+more\s+time|financial\s+(?:difficulty|hardship)|lost\s+my\s+job|"
    r"payment\s+is\s+not\s+possible|abhi\s+payment\s+(?:possible\s+)?nahi)\b"
)
REFUSAL = re.compile(r"(?i)\b(?:i\s+will\s+not\s+pay|i\s+won't\s+pay|i\s+refuse\s+to\s+pay|not\s+going\s+to\s+pay)\b")
DISPUTE = re.compile(r"(?i)\b(?:i\s+dispute\s+(?:this|the)\s+(?:amount|debt|charge)|i\s+do\s+not\s+owe|amount\s+is\s+(?:wrong|incorrect)|not\s+my\s+loan)\b")
DATE_TERMS = re.compile(
    r"(?i)\b(today|tomorrow|day after tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b"
)
AMOUNT = re.compile(r"(?i)(?:₹|rs\.?|inr)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)|\b([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:rupees?|inr)\b")

_WEEKDAY = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}


def _normalize_due(expr: str | None, call_date=None) -> str | None:
    if not expr:
        return None
    base = call_date or datetime.now()
    if hasattr(base,"date"):
        base_dt=base
    else:
        base_dt=datetime.combine(base,datetime.min.time())
    low=expr.lower()
    if low=="today": return base_dt.date().isoformat()
    if low=="tomorrow": return (base_dt+timedelta(days=1)).date().isoformat()
    if low=="day after tomorrow": return (base_dt+timedelta(days=2)).date().isoformat()
    if low in _WEEKDAY:
        delta=(_WEEKDAY[low]-base_dt.weekday())%7
        if delta==0: delta=7
        return (base_dt+timedelta(days=delta)).date().isoformat()
    for fmt in ("%d/%m/%Y","%d-%m-%Y","%d/%m/%y","%d-%m-%y"):
        try: return datetime.strptime(expr,fmt).date().isoformat()
        except ValueError: pass
    return None


def _sentence_spans(text: str):
    for m in re.finditer(r"[^.!?\n]+(?:[.!?\n]+|$)",text):
        segment=m.group(0).strip()
        if segment:
            yield m.start(),m.end(),segment


def detect_obligations(text: str, call_date=None) -> list[dict]:
    out=[]
    for s,e,segment in _sentence_spans(text):
        p=PROMISE.search(segment); d=DIFFICULTY.search(segment); r=REFUSAL.search(segment); q=DISPUTE.search(segment)
        date_m=DATE_TERMS.search(segment); amount_m=AMOUNT.search(segment)
        due_expr=date_m.group(0) if date_m else None
        amount=(amount_m.group(1) or amount_m.group(2)).replace(",","") if amount_m else None
        if p and not r and not re.search(r"(?i)\b(?:cannot|can't|won't|will\s+not)\b.{0,16}\bpay\b",segment):
            out.append({
                "type":"PROMISE_TO_PAY","speaker_role":"customer_or_unknown",
                "evidence":p.group(0),"due_expression":due_expr,"due_date":_normalize_due(due_expr,call_date),
                "amount":amount,"confidence":0.94 if due_expr else 0.84,
                "start":s+p.start(),"end":s+p.end(),
            })
        if d:
            out.append({"type":"REPAYMENT_DIFFICULTY","evidence":d.group(0),"confidence":0.91,"start":s+d.start(),"end":s+d.end()})
        if r:
            out.append({"type":"PAYMENT_REFUSAL","evidence":r.group(0),"confidence":0.93,"start":s+r.start(),"end":s+r.end()})
        if q:
            out.append({"type":"PAYMENT_OR_DEBT_DISPUTE","evidence":q.group(0),"confidence":0.92,"start":s+q.start(),"end":s+q.end()})
    return sorted(out,key=lambda x:x.get("start",0))
