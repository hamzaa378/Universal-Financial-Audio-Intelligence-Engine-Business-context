"""Privacy-first PII detection for financial-call transcripts (v4.5).

Pipeline:
ASR text -> ASR-aware normalization candidates -> deterministic validators -> optional
ONNX NER support -> optional semantic judge -> conflict resolution.

Strong identifiers stay deterministic. Neural components only participate in ambiguous
contextual cases so latency and false-positive risk remain bounded.
"""
from __future__ import annotations

import re
from typing import Iterable

from decision_ai.privacy_judge import judge_many as semantic_judge_many, engine_status as semantic_engine_status
from decision_ai.ner_engine import support_candidates as ner_support_candidates, status as ner_engine_status
from nlp.normalization import (
    find_ifsc_candidates,
    find_spoken_email_candidates,
    find_spoken_digit_runs,
    find_natural_date_candidates,
)

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
_PHONE = re.compile(r"(?<!\d)(?:(?:\+?91|0)[\s.-]?)?[6-9](?:[\s.-]?\d){9}(?!\d)")
_AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_UPI = re.compile(r"(?i)(?<![\w.+-])[A-Z0-9._-]{2,64}@[A-Z][A-Z0-9_-]{1,31}(?![A-Z0-9_-])")
_LONG_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){8,17}\d(?!\d)")
_OTP = re.compile(r"(?<!\d)\d{4,8}(?!\d)")
_CVV = re.compile(r"(?<!\d)\d{3,4}(?!\d)")
_PINCODE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
_DOB = re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}(?!\d)")
_PASSPORT = re.compile(r"(?i)\b[A-Z][1-9][0-9]{6}\b")
_VOTER_ID = re.compile(r"(?i)\b[A-Z]{3}[0-9]{7}\b")
_DRIVING_LICENSE = re.compile(r"(?i)\b[A-Z]{2}[ -]?[0-9]{2}[ -]?(?:(?:19|20)[0-9]{2}[ -]?)?[0-9]{7}\b")

UPI_HANDLES = {
    "upi", "ybl", "ibl", "axl", "paytm", "okaxis", "okhdfcbank", "okicici",
    "oksbi", "okyesbank", "apl", "airtel", "freecharge", "pingpay", "waicici",
}

PHONE_NEGATIVE_CONTEXT = re.compile(
    r"(?i)\b(?:order|transaction|txn|reference|ref|loan|customer|application|invoice|ticket|case|employee|tracking|token|request|complaint)\s*(?:id|number|no\.?|#)?(?:\s+is)?\s*[:=-]?\s*$"
)
PHONE_POSITIVE_CONTEXT = re.compile(r"(?i)\b(?:phone|mobile|contact|call(?:\s+me)?|whatsapp|telephone|reach\s+me|ring\s+me|number\s+to\s+reach)\b")
ACCOUNT_CONTEXT = re.compile(r"(?i)\b(?:account|a/c|acct|loan\s+account|bank\s+account)\s*(?:number|no\.?|#)?\b")
OTP_CONTEXT = re.compile(r"(?i)\b(?:otp|one[- ]time\s+(?:password|passcode)|verification\s+code|security\s+code|auth(?:entication)?\s+code)\b")
CVV_CONTEXT = re.compile(r"(?i)\b(?:cvv|cvc|card\s+security\s+code|card\s+verification\s+value)\b")
PIN_CONTEXT = re.compile(r"(?i)\b(?:pin\s*code|pincode|postal\s+code|zip\s+code)\b")
DOB_CONTEXT = re.compile(r"(?i)\b(?:dob|date\s+of\s+birth|born\s+on|birth\s+date|birthday)\b")
PASSPORT_CONTEXT = re.compile(r"(?i)\b(?:passport|passport\s+(?:number|no\.?))\b")
VOTER_CONTEXT = re.compile(r"(?i)\b(?:voter\s*(?:id|card)|epic\s*(?:id|number|no\.?))\b")
DL_CONTEXT = re.compile(r"(?i)\b(?:driving\s+licen[cs]e|driver'?s\s+licen[cs]e|dl\s*(?:number|no\.?))\b")
AADHAAR_CONTEXT = re.compile(r"(?i)\b(?:aadhaar|aadhar|uidai|uid\s+number)\b")
CARD_CONTEXT = re.compile(r"(?i)\b(?:card|credit\s+card|debit\s+card|visa|mastercard|rupay)\b")
UPI_CONTEXT = re.compile(r"(?i)\b(?:upi|vpa|virtual\s+payment\s+address|pay\s+id)\b")
IFSC_CONTEXT = re.compile(r"(?i)\b(?:ifsc|bank\s+ifsc|branch\s+ifsc)(?:\s+code)?\b")
EMAIL_CONTEXT = re.compile(r"(?i)\b(?:email|e-mail|mail\s+id|email\s+address|registered\s+mail)\b")
ADDRESS_HINT = re.compile(
    r"(?i)(?:\d|\b(?:road|rd|street|st|lane|sector|nagar|colony|apartment|apt|flat|house|village|district|block|phase|floor|near|opp(?:osite)?|building)\b)"
)


def _left_clause(text: str, start: int, radius: int = 88) -> str:
    lo=max(0,start-radius)
    for sep in (".", "!", "?", "\n", ";"):
        pos=text.rfind(sep, lo, start)
        if pos >= lo:
            lo=max(lo,pos+1)
    return text[lo:start]


def _clause_bounds(text: str, start: int, end: int, radius: int=180) -> tuple[int,int]:
    lo=max(0,start-radius); hi=min(len(text),end+radius)
    for sep in (".","!","?","\n",";"):
        p=text.rfind(sep,lo,start)
        if p>=lo: lo=max(lo,p+1)
        q=text.find(sep,end,hi)
        if q>=0: hi=min(hi,q)
    return lo,hi


def _item(kind: str, m: re.Match, confidence: float, evidence: str) -> dict:
    return {
        "type": kind, "start": m.start(), "end": m.end(), "value": m.group(0),
        "confidence": round(float(confidence), 3), "evidence": evidence,
    }


def _raw_item(kind: str, start: int, end: int, value: str, confidence: float, evidence: str, **extra) -> dict:
    x={"type":kind,"start":start,"end":end,"value":value,"confidence":round(float(confidence),3),"evidence":evidence}
    x.update(extra)
    return x


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _resolve_conflicts(items: Iterable[dict]) -> list[dict]:
    priority = {
        "EMAIL":100,"PAN":99,"IFSC":98,"UPI":97,"AADHAAR":96,"CARD":95,
        "PASSPORT":94,"ADDRESS":93,"VOTER_ID":92,"DRIVING_LICENSE":92,
        "PHONE":90,"ACCOUNT_NUMBER":85,"DOB":80,"OTP":78,"CVV":77,
        "PINCODE":75,"NAME":70,
    }
    ranked=sorted(items,key=lambda x:(-x["confidence"],-priority.get(x["type"],0),-(x["end"]-x["start"]),x["start"]))
    chosen=[]
    for item in ranked:
        if not any(_overlaps(item,c) for c in chosen):
            chosen.append(item)
    return sorted(chosen,key=lambda x:(x["start"],x["end"]))


# ---------- bounded NAME / ADDRESS extraction ----------
_NAME_LABEL = re.compile(r"(?i)\b(?:my\s+name\s+is|(?:customer|applicant|borrower)\s+name\s+is|name\s*[:=])\s+")
_NAME_STOP = {"and","but","my","your","phone","mobile","email","account","pan","ifsc","otp","address","calling","speaking","from","regarding","about","because","for"}
_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z.'-]{0,30}")


def _find_name_candidates(text: str) -> list[dict]:
    out=[]
    for label in _NAME_LABEL.finditer(text):
        lo=label.end(); _,hi=_clause_bounds(text,lo,lo,100)
        body=text[lo:hi]
        tokens=[]; end=0
        for m in re.finditer(r"\S+",body):
            clean=m.group(0).strip(" ,:;()[]{}")
            if not clean:
                continue
            if clean.casefold() in _NAME_STOP:
                break
            if not _NAME_TOKEN.fullmatch(clean):
                break
            tokens.append((m.start(),m.end(),clean))
            end=m.end()
            if len(tokens)>=4:
                break
        if tokens:
            s=lo+tokens[0][0]; e=lo+end
            value=text[s:e].strip(" ,;:")
            e=s+len(value)
            if value.casefold() not in {"not available","not disclosed","unknown"}:
                out.append(_raw_item("NAME",s,e,value,0.94,"token_bounded_name_context",ner_review=True))
    return out


_ADDRESS_LABEL = re.compile(r"(?i)\b(?:(?:my|your)\s+)?(?:(?:current|residential|registered|communication|permanent)\s+)?address\s*(?:is|:|=)\s*")
_ADDRESS_NEXT_FIELD = re.compile(
    r"(?i)\s+(?:and\s+)?(?:my\s+|the\s+)?(?:phone|mobile|email|e-mail|pan|ifsc|account|a/c|otp|cvv|upi|dob|date\s+of\s+birth)\b"
)
_RESIDENTIAL_PREFIX = re.compile(r"(?i)\b(?:i\s+(?:live|reside|stay)\s+(?:at|in)|my\s+residence\s+is)\s+")


def _bounded_address_after(text: str, start: int, max_len: int=150) -> tuple[int,int] | None:
    hi=min(len(text),start+max_len)
    # Hard clause boundary first.
    stops=[p for p in (text.find(c,start,hi) for c in ".!?\n;") if p>=0]
    if stops: hi=min(hi,min(stops))
    body=text[start:hi]
    nxt=_ADDRESS_NEXT_FIELD.search(body)
    if nxt: hi=start+nxt.start()
    s=start
    while s<hi and text[s] in " ,:-": s+=1
    while hi>s and text[hi-1] in " ,:-": hi-=1
    return (s,hi) if hi>s else None


def _find_address_candidates(text: str) -> list[dict]:
    out=[]
    for m in _ADDRESS_LABEL.finditer(text):
        span=_bounded_address_after(text,m.end())
        if not span: continue
        s,e=span; body=text[s:e]
        if len(body)>=5 and ADDRESS_HINT.search(body):
            out.append(_raw_item("ADDRESS",s,e,body,0.965,"clause_bounded_address_context",ner_review=True))
    for m in _RESIDENTIAL_PREFIX.finditer(text):
        span=_bounded_address_after(text,m.end(),130)
        if not span: continue
        s,e=span; body=text[s:e]
        if len(body)>=5 and ADDRESS_HINT.search(body):
            out.append(_raw_item("ADDRESS",s,e,body,0.76,"residential_phrase_address",ai_review=True,ner_review=True))
    return out


# ---------- spoken / normalized candidates ----------

def _spoken_candidates(text: str) -> list[dict]:
    out=[]
    for c in find_spoken_email_candidates(text):
        ctx=_left_clause(text,c["start"],88)
        domain=c["canonical"].rsplit("@",1)[-1]
        if EMAIL_CONTEXT.search(ctx) or domain.split(".",1)[0] in {"gmail","yahoo","outlook","hotmail","protonmail","icloud"}:
            out.append(_raw_item("EMAIL",c["start"],c["end"],c["value"],0.95,c["evidence"],canonical=c["canonical"]))
    for c in find_spoken_digit_runs(text):
        digits=c["digits"]; ctx=_left_clause(text,c["start"],90)
        kind=conf=evidence=None
        span_start=c["start"]
        # Spoken IFSC example: "IFSC is ABCD zero one two three four five six".
        # The digit run normalizes to 0123456; the four-letter bank prefix immediately
        # before it is included in the protected source span.
        prefix=re.search(r"(?i)([A-Z]{4})\s*$",ctx)
        if IFSC_CONTEXT.search(ctx) and prefix and len(digits)==7 and digits.startswith("0"):
            kind,conf,evidence="IFSC",0.96,"spoken_ifsc_normalized"
            span_start=c["start"]-(len(ctx)-prefix.start(1))
        elif PHONE_POSITIVE_CONTEXT.search(ctx) and len(digits)==10 and digits[:1] in "6789":
            kind,conf,evidence="PHONE",0.95,"spoken_phone_context"
        elif AADHAAR_CONTEXT.search(ctx) and len(digits)==12 and digits[:1] not in "01":
            kind,conf,evidence="AADHAAR",0.94,"spoken_aadhaar_context"
        elif CARD_CONTEXT.search(ctx) and 13<=len(digits)<=19 and luhn_valid(digits):
            kind,conf,evidence="CARD",0.98,"spoken_card_luhn"
        elif ACCOUNT_CONTEXT.search(ctx) and 9<=len(digits)<=18:
            kind,conf,evidence="ACCOUNT_NUMBER",0.94,"spoken_account_context"
        elif CVV_CONTEXT.search(ctx) and 3<=len(digits)<=4:
            kind,conf,evidence="CVV",0.96,"spoken_cvv_context"
        elif OTP_CONTEXT.search(ctx) and 4<=len(digits)<=8:
            kind,conf,evidence="OTP",0.96,"spoken_otp_context"
        elif PIN_CONTEXT.search(ctx) and len(digits)==6 and digits[:1] != "0":
            kind,conf,evidence="PINCODE",0.93,"spoken_postal_context"
        if kind:
            value=text[span_start:c["end"]].strip() if span_start != c["start"] else c["value"].strip()
            canonical=(re.sub(r"\s+","",text[span_start:c["start"]]).upper()+digits) if kind=="IFSC" else digits
            out.append(_raw_item(kind,span_start,c["end"],value,conf,evidence,canonical=canonical))
    return out


# ---------- AI fusion ----------

def _fuse_ner_many(text: str, items: list[dict], use_ner: bool) -> list[dict]:
    out=[dict(x) for x in items]
    idx=[i for i,x in enumerate(out) if use_ner and x.get("ner_review")]
    if not idx:
        return out
    reviewed=[out[i] for i in idx]
    supports=ner_support_candidates(text,reviewed)
    for i,sup in zip(idx,supports):
        item=out[i]
        if not sup.get("available"):
            item["ner_method"]="unavailable"
            continue
        score=float(sup.get("support",0.0))
        item["ner_support"]=round(score,4)
        item["ner_method"]="onnx_token_ner"
        # NER is supporting evidence only. It cannot rescue a very weak candidate alone.
        if score>=0.70:
            item["confidence"]=round(min(0.995,item["confidence"]+0.07*score),3)
            item["evidence"]=item.get("evidence","")+"+ner"
    return out


def _fuse_semantic_many(text: str, items: list[dict], use_semantic: bool) -> list[dict]:
    out=[dict(x) for x in items]
    review_idx=[i for i,x in enumerate(out) if use_semantic and x.get("ai_review")]
    if not review_idx:
        for x in out: x.setdefault("decision_method","deterministic")
        return out
    review_items=[out[i] for i in review_idx]
    scores=semantic_judge_many(text,review_items)
    reviewed_set=set(review_idx)
    for i,x in enumerate(out):
        if i not in reviewed_set: x.setdefault("decision_method","deterministic")
    for idx,ai in zip(review_idx,scores):
        item=out[idx]
        if not ai.get("available"):
            item["decision_method"]="rules_fallback"
            continue
        base=float(item["confidence"]); ai_score=float(ai["score"])
        fused=min(0.995,max(0.0,0.50*base+0.50*ai_score+0.08))
        item["rule_confidence"]=round(base,3)
        item["semantic_score"]=round(ai_score,4)
        item["semantic_margin"]=ai.get("margin")
        item["confidence"]=round(fused,3)
        item["decision_method"]="hybrid_semantic_batched"
        item["evidence"]=item.get("evidence","")+"+semantic"
    return out


def detect_pii(
    text: str,
    min_confidence: float = 0.80,
    use_semantic: bool = False,
    use_ner: bool = False,
) -> list[dict]:
    found: list[dict] = []

    # Strong lexical structures.
    for m in _EMAIL.finditer(text): found.append(_item("EMAIL",m,0.995,"email_structure"))
    for m in _PAN.finditer(text): found.append(_item("PAN",m,0.995,"pan_structure"))

    # IFSC normalization: canonical is strong; separator/spoken forms need IFSC context.
    for c in find_ifsc_candidates(text):
        ctx=_left_clause(text,c["start"],88)
        canonical_shape=(c["evidence"]=="ifsc_canonical")
        if canonical_shape:
            conf=0.995
        elif IFSC_CONTEXT.search(ctx):
            conf=0.995
        else:
            # A separator-normalized identifier is still suggestive, but not enough to
            # hide arbitrary product/reference IDs in balanced mode.
            conf=0.73
        x=_raw_item("IFSC",c["start"],c["end"],c["value"],conf,c["evidence"],canonical=c["canonical"])
        if conf<0.90:
            x["ai_review"]=True
        found.append(x)

    # UPI/VPA: do not mistake normal email domains for UPI handles.
    for m in _UPI.finditer(text):
        provider=m.group(0).rsplit("@",1)[-1].lower(); ctx=_left_clause(text,m.start(),72)
        if provider in UPI_HANDLES: found.append(_item("UPI",m,0.985,"known_upi_handle"))
        elif UPI_CONTEXT.search(ctx): found.append(_item("UPI",m,0.93,"upi_context"))

    for m in _CARD.finditer(text):
        if luhn_valid(m.group(0)):
            ctx=_left_clause(text,m.start(),72)
            found.append(_item("CARD",m,0.995 if CARD_CONTEXT.search(ctx) else 0.965,"luhn_valid"))

    for m in _AADHAAR.finditer(text):
        if len(_digits(m.group(0)))!=12: continue
        ctx=_left_clause(text,m.start(),72)
        if verhoeff_valid(m.group(0)): found.append(_item("AADHAAR",m,0.995,"verhoeff_valid"))
        elif AADHAAR_CONTEXT.search(ctx):
            x=_item("AADHAAR",m,0.91,"aadhaar_context_checksum_unverified"); x["ai_review"]=True; found.append(x)

    for m in _PHONE.finditer(text):
        digits=_digits(m.group(0)); local=digits[-10:]
        if len(local)!=10 or local[0] not in "6789": continue
        left=_left_clause(text,m.start(),72)
        positive=bool(PHONE_POSITIVE_CONTEXT.search(left))
        negative=bool(PHONE_NEGATIVE_CONTEXT.search(left))
        if positive:
            found.append(_item("PHONE",m,0.985,"phone_context"))
        elif negative:
            x=_item("PHONE",m,0.58,"identifier_like_context"); x.update(ai_review=True,ner_review=True); found.append(x)
        else:
            separated=bool(re.search(r"[ +.\-]",m.group(0)))
            base=0.79 if not separated and not digits.startswith("91") else 0.83
            x=_item("PHONE",m,base,"ambiguous_phone_shape"); x.update(ai_review=True,ner_review=True); found.append(x)

    for m in _LONG_NUMBER.finditer(text):
        digits=_digits(m.group(0))
        if not 9<=len(digits)<=18: continue
        ctx=_left_clause(text,m.start(),88)
        if ACCOUNT_CONTEXT.search(ctx):
            found.append(_item("ACCOUNT_NUMBER",m,0.95,"account_context"))
        elif re.search(r"(?i)\b(?:banking\s+details|bank\s+details|debit\s+from|credit\s+to|beneficiary\s+details)\b",ctx):
            x=_item("ACCOUNT_NUMBER",m,0.73,"weak_banking_context"); x.update(ai_review=True,ner_review=True); found.append(x)

    for m in _OTP.finditer(text):
        ctx=_left_clause(text,m.start(),64)
        if OTP_CONTEXT.search(ctx) and not CVV_CONTEXT.search(ctx): found.append(_item("OTP",m,0.97,"otp_context"))
        elif re.search(r"(?i)\b(?:verify|authenticate|login|sign\s*in)\b",ctx):
            x=_item("OTP",m,0.74,"weak_auth_context"); x.update(ai_review=True,ner_review=True); found.append(x)
    for m in _CVV.finditer(text):
        if CVV_CONTEXT.search(_left_clause(text,m.start(),56)): found.append(_item("CVV",m,0.97,"cvv_context"))

    for m in _PINCODE.finditer(text):
        ctx=_left_clause(text,m.start(),82)
        if PIN_CONTEXT.search(ctx) or re.search(r"(?i)\baddress\b",ctx): found.append(_item("PINCODE",m,0.94,"postal_context"))
        elif re.search(r"(?i)\b(?:live|reside|home|residence|staying|located)\b",ctx):
            x=_item("PINCODE",m,0.72,"weak_residential_context"); x.update(ai_review=True,ner_review=True); found.append(x)

    for m in _DOB.finditer(text):
        if DOB_CONTEXT.search(_left_clause(text,m.start(),72)): found.append(_item("DOB",m,0.96,"dob_context"))
    for c in find_natural_date_candidates(text):
        if DOB_CONTEXT.search(_left_clause(text,c["start"],80)):
            found.append(_raw_item("DOB",c["start"],c["end"],c["value"],0.96,"natural_language_dob",canonical=c["canonical"],ner_review=True))

    for m in _PASSPORT.finditer(text):
        if PASSPORT_CONTEXT.search(_left_clause(text,m.start(),72)): found.append(_item("PASSPORT",m,0.97,"passport_context_structure"))
    for m in _VOTER_ID.finditer(text):
        if VOTER_CONTEXT.search(_left_clause(text,m.start(),72)): found.append(_item("VOTER_ID",m,0.96,"voter_id_context_structure"))
    for m in _DRIVING_LICENSE.finditer(text):
        if DL_CONTEXT.search(_left_clause(text,m.start(),80)): found.append(_item("DRIVING_LICENSE",m,0.96,"driving_license_context_structure"))

    found.extend(_find_name_candidates(text))
    found.extend(_find_address_candidates(text))
    found.extend(_spoken_candidates(text))

    reviewed=_fuse_ner_many(text,found,use_ner)
    reviewed=_fuse_semantic_many(text,reviewed,use_semantic)
    return _resolve_conflicts(x for x in reviewed if x["confidence"]>=min_confidence)


def semantic_status() -> dict:
    return semantic_engine_status()


def ner_status() -> dict:
    return ner_engine_status()


def _partial_mask(kind: str, value: str) -> str:
    digits=_digits(value)
    if kind in {"PHONE","CARD","ACCOUNT_NUMBER","AADHAAR"} and len(digits)>=4:
        return f"[{kind} ••••{digits[-4:]}]"
    if kind=="EMAIL" and "@" in value:
        local,domain=value.split("@",1)
        return f"[{kind} {local[:1]}***@{domain}]"
    return f"[{kind} REDACTED]"


def mask_pii(text: str, entities: Iterable[dict], mode: str="full") -> str:
    masked=text
    for e in sorted(entities,key=lambda x:x["start"],reverse=True):
        repl=_partial_mask(e["type"],e.get("value","")) if mode=="partial" else f"[{e['type']} REDACTED]"
        masked=masked[:e["start"]]+repl+masked[e["end"]:]
    return masked


def public_pii_metadata(entities: Iterable[dict], reveal_suffix: bool=False) -> list[dict]:
    out=[]
    for e in entities:
        out.append({
            "type":e["type"],"start":e["start"],"end":e["end"],"confidence":e["confidence"],
            "evidence":e.get("evidence",""),"decision_method":e.get("decision_method","deterministic"),
            "rule_confidence":e.get("rule_confidence"),"semantic_score":e.get("semantic_score"),
            "ner_support":e.get("ner_support"),"ner_method":e.get("ner_method"),
            "token_ids":e.get("token_ids"),"alignment_method":e.get("alignment_method"),
            "alignment_confidence":e.get("alignment_confidence"),
            "masked_value":_partial_mask(e["type"],e.get("value","")) if reveal_suffix else f"[{e['type']} REDACTED]",
        })
    return out
