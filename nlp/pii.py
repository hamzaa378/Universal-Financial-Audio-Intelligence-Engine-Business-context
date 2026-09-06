"""Privacy-first PII detection for financial-call transcripts (v4.8).

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
    find_spoken_handle_candidates,
    find_spoken_digit_runs,
    find_spoken_alnum_runs,
    find_natural_date_candidates,
    find_spoken_date_candidates,
    find_spoken_driving_license_candidates,
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
_PHONE = re.compile(
    r"(?<!\d)(?:(?:\+?91|0)[\s.-]?)?(?:"
    r"[6-9](?:[\s.-]?\d){9}|"
    r"\([6-9]\d{4}\)[\s.-]?\d{5}"
    r")(?!\d)"
)
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

# Explicitly assigned credentials/secrets. These are deliberately context-gated:
# random high-entropy strings are never masked merely because they look unusual.
_CREDENTIAL_ASSIGN = re.compile(
    r'''(?ix)\b(?P<label>password|passcode|username|user\s*name|api[ _-]*key|access[ _-]*token|auth(?:entication)?[ _-]*token|github[ _-]*token)
        \s*(?:is|:|=)\s*["']?(?P<value>[^\s,"';]{3,160})'''
)
_AUTH_USE_TOKEN = re.compile(
    r'''(?ix)\b(?:please\s+)?(?:use|provide|send|authenticate\s+with)\s+(?:the\s+)?
        (?P<label>token|api[ _-]*key|access[ _-]*token)\s+["']?(?P<value>[A-Za-z0-9][A-Za-z0-9._~+\-/=]{11,160})'''
)
_ENV_SECRET_ASSIGN = re.compile(
    r'''(?x)\b(?P<label>(?:[A-Za-z0-9]+_)*(?:API_KEY|SECRET_KEY|ACCESS_TOKEN|AUTH_TOKEN|TOKEN|PASSWORD))
        \s*=\s*["']?(?P<value>[^\s,"';]{3,200})''',
    re.I,
)

UPI_HANDLES = {
    "upi", "ybl", "ibl", "axl", "paytm", "okaxis", "okhdfcbank", "okicici",
    "oksbi", "okyesbank", "apl", "airtel", "freecharge", "pingpay", "waicici",
}

PHONE_NEGATIVE_CONTEXT = re.compile(
    r"(?i)\b(?:order|transaction|txn|reference|ref|loan|customer|application|invoice|ticket|case|employee|tracking|token|request|complaint)\s*(?:id|number|no\.?|#)?(?:\s+is)?\s*[:=-]?\s*$"
)
PHONE_POSITIVE_CONTEXT = re.compile(
    r"(?i)\b(?:phone|mobile|contact|whatsapp|telephone|number\s+to\s+reach|"
    r"call(?:\s+(?:me|him|her|them|[A-Z][A-Za-z.'-]*))?|"
    r"reach(?:\s+(?:me|him|her|them|[A-Z][A-Za-z.'-]*))?|ring\s+me|"
    r"(?:my|your|his|her|their|customer'?s|client'?s|borrower'?s)\s+(?:phone\s+|mobile\s+|contact\s+)?number)\b"
)
PHONE_FOLLOWING_CONTEXT = re.compile(r"(?i)\b(?:for\s+(?:future\s+)?contact|for\s+contact|as\s+(?:my|his|her|their)\s+(?:phone|mobile|contact)\s+number)\b")
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
PAN_CONTEXT = re.compile(r"(?i)\b(?:pan|permanent\s+account\s+number)(?:\s+(?:number|no\.?|card))?\b")

# Public/geographic postal-code language. A six-digit PIN is not inherently personal:
# it becomes privacy-sensitive when owned by a person/address/delivery context. These
# cues therefore veto a standalone PIN candidate only when explicit personal ownership
# is absent.
_PUBLIC_GEO_CONTEXT = re.compile(
    r"(?i)\b(?:covers?\s+(?:part\s+of|the\s+area|this\s+area)|postal\s+(?:region|area|zone)|"
    r"postcode\s+(?:for|of)\s+(?:the\s+)?(?:city|town|district|locality|area)|"
    r"pin\s*code\s+(?:for|of)\s+(?:the\s+)?(?:city|town|district|locality|area)|"
    r"(?:city|town|district|locality|area)\s+(?:uses|has|is\s+covered\s+by)\s+(?:the\s+)?(?:pin\s*code|postal\s+code)|"
    r"belongs?\s+to\s+(?:the\s+)?(?:postal\s+)?(?:region|area|zone)|"
    r"public\s+(?:postal|pin)\s+(?:code|region))\b"
)

# Hard-negative discourse signals. They are evaluated on the current occurrence's
# clause, never cached by literal value. This allows the same token to be private in
# one sentence and harmless in a documentation/reference sentence.
_EXAMPLE_CONTEXT = re.compile(
    r"(?i)\b(?:example|sample|documentation|docs?|tutorial|manual|article|test\s+(?:value|number|data|case)|"
    r"standard\s+test|literal\s+text|placeholder|template|configuration\s+file|source\s+code|"
    r"unit[- ]?test|synthetic\s+benchmark|dataset\s+(?:column|example)|training\s+video|"
    r"demonstrat(?:e|es|ed|ion))\b"
)
_REFERENCE_ROLE_CONTEXT = re.compile(
    r"(?i)\b(?:product\s+serial|serial\s+number|asset\s+code|batch(?:\s+code)?|invoice\s+(?:reference|number)|"
    r"shipment\s+(?:reference|number)|reference\s+document|document\s+template|page\s+(?:identifier|number)|"
    r"order\s+(?:id|number)|tracking\s+(?:id|number)|employee\s+(?:id|number))\b"
)
_ROLE_RESET = re.compile(
    r"(?i)(?:\bwhile\b|\bwhereas\b|\bhowever\b|\bbut\b|"
    r"\band\b(?=\s+(?:(?:the|my|your)\s+)?(?:transaction|txn|complaint|ticket|case|order|reference|employee|"
    r"application|invoice|tracking|phone|mobile|account|ifsc|pan|email|upi|otp|cvv|address|dob|date\s+of\s+birth)\b)|"
    r",(?=\s*(?:(?:the|my|your)\s+)?(?:transaction|txn|complaint|ticket|case|order|reference|employee|"
    r"application(?:\s+deadline)?|invoice|tracking|phone|mobile|account|ifsc|pan|email|upi|otp|cvv|address|dob|"
    r"date\s+of\s+birth|meeting|report)\b))"
)

# Field-aware clause segmentation. Sentence-level context is too broad for utterances
# such as "my name is X, DOB Y, phone Z, email example dot com" because the word
# "example" in the email clause can incorrectly veto earlier fields. These boundaries
# are used only for ownership/semantic context; source spans remain untouched.
_FIELD_LABEL_CORE = (
    r"(?:full\s+name|registered\s+name|legal\s+name|official\s+name|name|dob|date\s+of\s+birth|"
    r"phone(?:\s+number)?|mobile(?:\s+number)?|contact(?:\s+number)?|email(?:\s+address)?|e-mail|"
    r"account(?:\s+number)?|a/c|acct|ifsc(?:\s+code)?|pan(?:\s+number)?|upi(?:\s+(?:id|address))?|"
    r"aadhaar(?:\s+number)?|aadhar(?:\s+number)?|otp|cvv|cvc|pin\s*code|pincode|postal\s+code|"
    r"passport(?:\s+number)?|voter\s+id|driving\s+licen[cs]e(?:\s+number)?|card(?:\s+number)?|address)"
)
_FIELD_SPLIT = re.compile(
    rf"(?i)(?:,\s*|\band\s+)(?=(?:(?:my|your|the)\s+)?"
    rf"(?:(?:alternate|secondary|backup|other|primary|registered)\s+)?{_FIELD_LABEL_CORE}\b)"
)
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


def _field_clause_bounds(text: str, start: int, end: int, radius: int=220) -> tuple[int,int]:
    """Return the smallest sentence-local field clause containing this occurrence."""
    lo,hi=_clause_bounds(text,start,end,radius)
    segment=text[lo:hi]
    rel_start=start-lo; rel_end=end-lo
    left_cut=0; right_cut=len(segment)
    for m in _FIELD_SPLIT.finditer(segment):
        if m.end() <= rel_start:
            left_cut=m.end()
            continue
        if m.start() >= rel_end:
            right_cut=m.start()
            break
    return lo+left_cut,lo+right_cut


def _role_left_context(text: str, start: int, radius: int = 96) -> str:
    """Return the local grammatical role before this exact occurrence.

    Contrastive conjunctions and commas reset ownership. This prevents a role such as
    DOB/PHONE from leaking to a second identical value later in the same sentence.
    """
    lo,_=_field_clause_bounds(text,start,start,max(radius,150))
    left=text[max(lo,start-radius):start]
    cuts=[m.end() for m in _ROLE_RESET.finditer(left)]
    return left[max(cuts) if cuts else 0:]


def _occurrence_clause(text: str, start: int, end: int, radius: int = 150) -> str:
    lo,hi=_field_clause_bounds(text,start,end,max(radius,180))
    return text[lo:hi]


def _occurrence_context(text: str, start: int, end: int, radius: int = 150) -> str:
    """Clause context with the candidate removed.

    This prevents candidate content such as `example.com` or a token containing the
    substring `Example` from triggering documentation/example suppression by itself.
    """
    lo,hi=_field_clause_bounds(text,start,end,max(radius,180))
    return (text[lo:start] + " <VALUE> " + text[end:hi]).strip()


def _right_clause(text: str, end: int, radius: int = 72) -> str:
    _,field_hi=_field_clause_bounds(text,end,end,max(radius,150))
    hi=min(field_hi,end+radius)
    for sep in (".","!","?","\n",";"):
        pos=text.find(sep,end,hi)
        if pos>=0: hi=min(hi,pos)
    return text[end:hi]


def _explicit_owner_for_type(kind: str, left: str) -> bool:
    checks={
        "EMAIL":EMAIL_CONTEXT, "PAN":PAN_CONTEXT, "IFSC":IFSC_CONTEXT, "UPI":UPI_CONTEXT,
        "CARD":CARD_CONTEXT, "AADHAAR":AADHAAR_CONTEXT, "PHONE":PHONE_POSITIVE_CONTEXT,
        "ACCOUNT_NUMBER":ACCOUNT_CONTEXT, "OTP":OTP_CONTEXT, "CVV":CVV_CONTEXT,
        "PINCODE":PIN_CONTEXT, "DOB":DOB_CONTEXT, "PASSPORT":PASSPORT_CONTEXT,
        "VOTER_ID":VOTER_CONTEXT, "DRIVING_LICENSE":DL_CONTEXT,
    }
    pat=checks.get(kind)
    if not (pat and pat.search(left)):
        return False
    # A type word alone ("example email", "test card number") is not ownership.
    personal=bool(re.search(
        r"(?i)\b(?:my|mine|our|registered\s+(?:phone|mobile|email|address)|call\s+me|reach\s+me|"
        r"contact\s+me|send\s+(?:it|the\s+refund|the\s+confirmation)\s+to\s+me)\b",
        left,
    ))
    return personal


def _apply_occurrence_context_policy(text: str, items: list[dict]) -> list[dict]:
    """Precision layer between validation and optional AI.

    A validator proves shape, not ownership. Example/document/reference clauses can
    therefore veto an otherwise valid value unless the same local role explicitly
    assigns that value to a person/account.
    """
    out=[]
    for src in items:
        item=dict(src)
        kind=item.get("type","")
        left=_role_left_context(text,item["start"],100)
        clause=_occurrence_context(text,item["start"],item["end"],170)
        explicit=_explicit_owner_for_type(kind,left)
        hard_example=bool(_EXAMPLE_CONTEXT.search(clause))
        role_reference=bool(_REFERENCE_ROLE_CONTEXT.search(left))
        public_geo=kind=="PINCODE" and bool(_PUBLIC_GEO_CONTEXT.search(clause))

        if kind not in {"PASSWORD","USERNAME","API_KEY","AUTH_TOKEN"}:
            if hard_example and not explicit:
                item["confidence"]=min(float(item["confidence"]),0.25)
                item["decision_method"]="context_veto"
                item["evidence"]=item.get("evidence","")+"+example_veto"
                item.pop("ai_review",None)
                item.pop("ner_review",None)
            elif public_geo and not explicit:
                item["confidence"]=min(float(item["confidence"]),0.22)
                item["decision_method"]="context_veto"
                item["evidence"]=item.get("evidence","")+"+public_geo_veto"
                item.pop("ai_review",None)
                item.pop("ner_review",None)
            elif role_reference and kind in {"PAN","IFSC","AADHAAR","CARD","EMAIL","UPI","PHONE","PINCODE"} and not explicit:
                item["confidence"]=min(float(item["confidence"]),0.35)
                item["decision_method"]="context_veto"
                item["evidence"]=item.get("evidence","")+"+reference_role_veto"
                item.pop("ai_review",None)
                item.pop("ner_review",None)
        out.append(item)
    return out


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
        "API_KEY":103,"AUTH_TOKEN":103,"PASSWORD":103,"EMAIL":100,"PAN":99,"IFSC":98,"UPI":97,"AADHAAR":96,"CARD":95,
        "PASSPORT":94,"ADDRESS":93,"VOTER_ID":92,"DRIVING_LICENSE":92,
        "PHONE":90,"USERNAME":88,"ACCOUNT_NUMBER":85,"DOB":80,"OTP":78,"CVV":77,
        "PINCODE":75,"NAME":70,
    }
    # Explicit ownership is stronger evidence than an ambiguous format match. This is
    # especially important for overlapping ACCOUNT_NUMBER/CARD candidates.
    ranked=sorted(
        items,
        key=lambda x:(
            -int(x.get("ownership_strength",0)),
            -x["confidence"],
            -priority.get(x["type"],0),
            -(x["end"]-x["start"]),
            x["start"],
        ),
    )
    chosen=[]
    for item in ranked:
        if not any(_overlaps(item,c) for c in chosen):
            chosen.append(item)
    return sorted(chosen,key=lambda x:(x["start"],x["end"]))


# ---------- bounded NAME / ADDRESS / SECRET extraction ----------
_NAME_LABEL = re.compile(
    r"(?i)\b(?:my\s+(?:full\s+|registered\s+|legal\s+|official\s+)?name\s+is|"
    r"(?:customer|applicant|borrower)\s+name\s+is|name\s*[:=])\s+"
)
_SELF_NAME_LABEL = re.compile(r"(?i)\b(?:i\s+am|i['’]m|this\s+is)\s+")
_NAME_STOP = {"and","but","my","your","phone","mobile","email","account","pan","ifsc","otp","address","calling","speaking","from","regarding","about","because","for","trying","looking","here","not","available"}
_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z.'-]{0,30}")


def _name_after_label(text: str, lo: int, *, require_two: bool=False, self_identification: bool=False) -> dict | None:
    _,hi=_field_clause_bounds(text,lo,lo,130)
    body=text[lo:hi]
    tokens=[]; end=0
    for m in re.finditer(r"\S+",body):
        clean=m.group(0).strip(" ,:;()[]{}")
        if not clean: continue
        if clean.casefold() in _NAME_STOP or not _NAME_TOKEN.fullmatch(clean): break
        tokens.append((m.start(),m.end(),clean)); end=m.end()
        if len(tokens)>=4: break
    if not tokens or (require_two and len(tokens)<2): return None
    # Self-identification is stricter so "I am calling regarding..." is not a NAME.
    if self_identification and not all(t[2][:1].isupper() for t in tokens): return None
    s=lo+tokens[0][0]; e=lo+end
    value=text[s:e].strip(" ,;:"); e=s+len(value)
    if value.casefold() in {"not available","not disclosed","unknown"}: return None
    conf=0.94 if not self_identification else 0.91
    return _raw_item("NAME",s,e,value,conf,"self_identification_name" if self_identification else "token_bounded_name_context",ner_review=True)


def _find_name_candidates(text: str) -> list[dict]:
    out=[]
    for label in _NAME_LABEL.finditer(text):
        item=_name_after_label(text,label.end())
        if item: out.append(item)
    for label in _SELF_NAME_LABEL.finditer(text):
        item=_name_after_label(text,label.end(),require_two=True,self_identification=True)
        if item: out.append(item)
    return out


_ADDRESS_LABEL = re.compile(
    r"(?i)\b(?:(?:my|your)\s+)?(?:(?:current|residential|registered|communication|permanent)\s+)?"
    r"address\s*(?:(?:is|:|=)\s*)?"
)
_ADDRESS_NEXT_FIELD = re.compile(
    r"(?i)(?:,\s*|\s+)(?:and\s+)?(?:(?:my|your|the)\s+)?"
    r"(?:(?:alternate|secondary|backup|other|primary|registered)\s+)?"
    r"(?:phone|mobile|contact|email|e-mail|pan|ifsc|account|a/c|otp|cvv|upi|dob|date\s+of\s+birth|"
    r"passport|voter\s+id|driving\s+licen[cs]e|card)\b"
)
_RESIDENTIAL_PREFIX = re.compile(r"(?i)\b(?:i\s+(?:live|reside|stay)\s+(?:at|in)|my\s+residence\s+is)\s+")
_DELIVERY_PREFIX = re.compile(r"(?i)\b(?:send|deliver|ship|courier)\s+(?:(?:the|my|this)\s+)?(?:package|parcel|document|statement|card|item|order)?\s*(?:to|at)\s+")
_ADDRESS_LABEL_NEGATIVE_PREFIX = re.compile(r"(?i)(?:postal|pin|zip)\s+code\s+for\s+(?:my\s+)?$")
_ADDRESS_LOCATION_WORD = re.compile(r"(?i)\b(?:road|rd\.?|street|st\.?|lane|sector|nagar|colony|apartment|apartments|apt|flat|house|village|district|block|phase|floor|building|avenue|cross\s+road|extension)\b")
_ADDRESS_POSTCODE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")

# Ownership-gated partial financial identifiers. Four digits alone are far too
# ambiguous, so these patterns require an explicit personal card/account ownership
# phrase and a partial-identifier construction such as "ends with" or "last four".
_PARTIAL_CARD_PATTERNS = (
    re.compile(r"(?i)\b(?:my|our)\s+(?:(?:credit|debit)\s+)?card\s+(?:ends?|ending)\s+(?:with|in)\s*[:=-]?\s*(?P<value>\d{4})\b"),
    re.compile(r"(?i)\b(?:the\s+)?last\s+(?:four|4)\s+digits?\s+of\s+(?:my|our)\s+(?:(?:credit|debit)\s+)?card\s+(?:are|is|:)\s*(?P<value>\d{4})\b"),
)
_PARTIAL_ACCOUNT_PATTERNS = (
    re.compile(r"(?i)\b(?:my|our)\s+(?:(?:bank|loan)\s+)?account\s+(?:ends?|ending)\s+(?:with|in)\s*[:=-]?\s*(?P<value>\d{4})\b"),
    re.compile(r"(?i)\b(?:the\s+)?last\s+(?:four|4)\s+digits?\s+of\s+(?:my|our)\s+(?:(?:bank|loan)\s+)?account\s+(?:are|is|:)\s*(?P<value>\d{4})\b"),
)


def _address_structure_score(body: str) -> int:
    score=0
    if _ADDRESS_LOCATION_WORD.search(body): score+=2
    if _ADDRESS_POSTCODE.search(body): score+=2
    if re.search(r"(?i)\b(?:flat|house|apt|apartment|floor|plot)\s*(?:no\.?\s*)?[A-Z0-9/-]+",body): score+=2
    elif re.search(r"(?<!\w)\d{1,4}[A-Za-z/-]?\b",body): score+=1
    if body.count(",")>=2: score+=2
    elif body.count(",")==1: score+=1
    return score


def _bounded_address_after(text: str, start: int, max_len: int=150) -> tuple[int,int] | None:
    hi=min(len(text),start+max_len)
    # Hard clause boundary first, but do not terminate on common street abbreviations.
    stops=[]
    for p in range(start,hi):
        ch=text[p]
        if ch in "!?\n;":
            stops.append(p)
        elif ch==".":
            prev=text[max(start,p-6):p]
            if re.search(r"(?i)\b(?:rd|st|ave|dr|no|apt)$",prev):
                continue
            stops.append(p)
    if stops: hi=min(hi,min(stops))
    body=text[start:hi]
    nxt=_ADDRESS_NEXT_FIELD.search(body)
    if nxt: hi=start+nxt.start()
    body=text[start:hi]
    contrast=re.search(r"(?i),\s*(?:but|while|whereas|however)\b",body)
    if contrast: hi=start+contrast.start()
    s=start
    while s<hi and text[s] in " ,:-": s+=1
    while hi>s and text[hi-1] in " ,:-": hi-=1
    return (s,hi) if hi>s else None


def _find_address_candidates(text: str) -> list[dict]:
    out=[]
    for m in _ADDRESS_LABEL.finditer(text):
        # "postal code for my address is 110016" owns a PINCODE, not an ADDRESS.
        if _ADDRESS_LABEL_NEGATIVE_PREFIX.search(_left_clause(text,m.start(),42)): continue
        span=_bounded_address_after(text,m.end())
        if not span: continue
        s,e=span; body=text[s:e]
        if len(body)>=5 and len(body.split())>=2 and _address_structure_score(body)>=3:
            out.append(_raw_item("ADDRESS",s,e,body,0.975,"structured_clause_address",ner_review=True))
    for m in _RESIDENTIAL_PREFIX.finditer(text):
        span=_bounded_address_after(text,m.end(),130)
        if not span: continue
        s,e=span; body=text[s:e]
        if len(body)>=5 and _address_structure_score(body)>=3:
            out.append(_raw_item("ADDRESS",s,e,body,0.83,"residential_phrase_address",ai_review=True,ner_review=True))
    for m in _DELIVERY_PREFIX.finditer(text):
        span=_bounded_address_after(text,m.end(),150)
        if not span: continue
        s,e=span; body=text[s:e]
        if len(body)>=8 and _address_structure_score(body)>=4:
            out.append(_raw_item("ADDRESS",s,e,body,0.94,"delivery_destination_address",ner_review=True))
    return out


def _find_partial_identifier_candidates(text: str) -> list[dict]:
    """Detect only explicitly owned last-four card/account references.

    The returned source span is the four digits themselves so surrounding prose stays
    readable. `partial_identifier=True` also forces full hiding even if the caller asks
    for partial-mask mode; revealing the suffix would otherwise reveal the entire value.
    """
    out=[]
    for kind, patterns in (("CARD", _PARTIAL_CARD_PATTERNS), ("ACCOUNT_NUMBER", _PARTIAL_ACCOUNT_PATTERNS)):
        for pat in patterns:
            for m in pat.finditer(text):
                s,e=m.span("value")
                out.append(_raw_item(
                    kind,s,e,m.group("value"),0.995,
                    "owned_partial_card_last4" if kind=="CARD" else "owned_partial_account_last4",
                    partial_identifier=True,ownership_strength=4,
                ))
    return out


_SECRET_PLACEHOLDER = re.compile(r"(?i)^(?:your[_-]?(?:api[_-]?key|token|password)|api[_-]?key|token|password|changeme|replace[_-]?me|example[_-]?(?:key|token)|x{4,}|\*{4,}|<[^>]+>)$")
_KNOWN_SECRET_PREFIX = re.compile(r"(?i)^(?:ghp_|github_pat_|sk-|xox[baprs]-|AIza|AKIA|ya29\.)")


def _looks_real_secret(value: str) -> bool:
    v=value.strip("\"'.,;:)")
    if len(v)<12 or _SECRET_PLACEHOLDER.match(v): return False
    if _KNOWN_SECRET_PREFIX.match(v): return True
    classes=sum(bool(re.search(p,v)) for p in (r"[a-z]",r"[A-Z]",r"\d",r"[_~+\-/=.#@$%^&*!]"))
    return len(v)>=18 and classes>=3


def _find_credential_candidates(text: str) -> list[dict]:
    out=[]
    mapping={
        "password":"PASSWORD","passcode":"PASSWORD","username":"USERNAME","user name":"USERNAME",
        "api key":"API_KEY","access token":"AUTH_TOKEN","auth token":"AUTH_TOKEN",
        "authentication token":"AUTH_TOKEN","github token":"AUTH_TOKEN",
    }
    for m in _CREDENTIAL_ASSIGN.finditer(text):
        label=re.sub(r"[ _-]+"," ",m.group("label").casefold()).strip()
        kind=mapping.get(label)
        if not kind: continue
        raw=m.group("value").rstrip(".,:;)")
        if kind in {"API_KEY","AUTH_TOKEN"} and not _looks_real_secret(raw): continue
        if kind=="USERNAME" and (len(raw)<3 or raw.casefold() in {"required","unknown","example","username","user"}): continue
        if kind=="PASSWORD" and (len(raw)<6 or raw.casefold() in {"required","unknown","password","example","changeme"}): continue
        s=m.start("value"); e=s+len(raw)
        item=_raw_item(kind,s,e,text[s:e],0.99,"explicit_secret_assignment")
        clause=_occurrence_context(text,s,e,160)
        left=_role_left_context(text,s,100)
        if _EXAMPLE_CONTEXT.search(clause) and not re.search(r"(?i)\b(?:my|mine|our|use|authenticate)\b",left):
            item["confidence"]=0.25
            item["decision_method"]="context_veto"
            item["evidence"]+="+example_veto"
        out.append(item)
    for m in _AUTH_USE_TOKEN.finditer(text):
        raw=m.group("value").rstrip(".,:;)")
        if not _looks_real_secret(raw): continue
        kind="API_KEY" if "api" in m.group("label").casefold() else "AUTH_TOKEN"
        s=m.start("value"); e=s+len(raw)
        out.append(_raw_item(kind,s,e,text[s:e],0.99,"explicit_secret_use_context"))
    for m in _ENV_SECRET_ASSIGN.finditer(text):
        label=m.group("label").upper()
        raw=m.group("value").rstrip(".,:;)")
        if label.endswith("PASSWORD"):
            kind="PASSWORD"
            if len(raw)<6 or _SECRET_PLACEHOLDER.match(raw): continue
        elif label.endswith("TOKEN"):
            kind="AUTH_TOKEN"
            if not _looks_real_secret(raw): continue
        else:
            kind="API_KEY"
            if not _looks_real_secret(raw): continue
        s=m.start("value"); e=s+len(raw)
        item=_raw_item(kind,s,e,text[s:e],0.99,"hardcoded_secret_assignment")
        context=_occurrence_context(text,s,e,170)
        if _EXAMPLE_CONTEXT.search(context):
            item["confidence"]=0.25
            item["decision_method"]="context_veto"
            item["evidence"]+="+example_veto"
        out.append(item)
    return out


# ---------- short-lived discourse ownership ----------

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_RESPONSE_OWNERSHIP = re.compile(
    r"(?i)^\s*(?:mine\s+is|it\s+is|i\s+(?:received|got|have|use)|the\s+(?:number|code|value)\s+is)\b"
)


def _sentence_spans(text: str) -> list[tuple[int,int]]:
    return [(m.start(),m.end()) for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


def _infer_expected_type(sentence: str) -> str | None:
    """Infer one field requested/described by the immediately previous sentence.

    The state is intentionally tiny: only the next sentence can consume it, and the
    next sentence must begin with an ownership response such as "Mine is" or
    "I received". This improves recall without turning discourse history into a broad
    masking signal.
    """
    checks=(
        ("AADHAAR",AADHAAR_CONTEXT),("PAN",PAN_CONTEXT),("EMAIL",EMAIL_CONTEXT),
        ("PHONE",PHONE_POSITIVE_CONTEXT),("UPI",UPI_CONTEXT),("IFSC",IFSC_CONTEXT),
        ("OTP",OTP_CONTEXT),("CVV",CVV_CONTEXT),("ACCOUNT_NUMBER",ACCOUNT_CONTEXT),
        ("PINCODE",PIN_CONTEXT),("DOB",DOB_CONTEXT),("PASSPORT",PASSPORT_CONTEXT),
        ("DRIVING_LICENSE",DL_CONTEXT),("VOTER_ID",VOTER_CONTEXT),("CARD",CARD_CONTEXT),
    )
    hits=[]
    for kind,pat in checks:
        m=pat.search(sentence)
        if m: hits.append((m.start(),kind))
    if re.search(r"(?i)\b(?:residential|permanent|current|registered)?\s*address\b",sentence):
        m=re.search(r"(?i)\b(?:residential|permanent|current|registered)?\s*address\b",sentence)
        if m: hits.append((m.start(),"ADDRESS"))
    if not hits: return None
    hits.sort(key=lambda x:x[0])
    # Earliest explicit field wins. This handles "The CVV ... payment card" as CVV,
    # not CARD, because CVV is the grammatical topic.
    return hits[0][1]


def _offset_item(item: dict, offset: int, **extra) -> dict:
    x=dict(item)
    x["start"]+=offset; x["end"]+=offset
    x.update(extra)
    return x


def _expected_response_candidate(kind: str, sentence: str, sent_start: int) -> dict | None:
    marker=_RESPONSE_OWNERSHIP.search(sentence)
    if not marker: return None
    tail=sentence[marker.end():]
    base=sent_start+marker.end()

    def from_regex(pattern: re.Pattern, confidence: float, evidence: str, validator=None):
        m=pattern.search(tail)
        if not m or (validator and not validator(m.group(0))): return None
        return _raw_item(
            kind,base+m.start(),base+m.end(),m.group(0),confidence,evidence,
            expected_type_state=True,ownership_strength=4,
        )

    if kind=="AADHAAR":
        return from_regex(_AADHAAR,0.97,"expected_type_response_aadhaar",lambda v:len(_digits(v))==12)
    if kind=="PHONE":
        return from_regex(_PHONE,0.97,"expected_type_response_phone",lambda v:len(_digits(v)[-10:])==10 and _digits(v)[-10] in "6789")
    if kind=="PAN": return from_regex(_PAN,0.985,"expected_type_response_pan")
    if kind=="EMAIL": return from_regex(_EMAIL,0.985,"expected_type_response_email")
    if kind=="UPI": return from_regex(_UPI,0.975,"expected_type_response_upi")
    if kind=="ACCOUNT_NUMBER": return from_regex(_LONG_NUMBER,0.97,"expected_type_response_account")
    if kind=="OTP": return from_regex(_OTP,0.97,"expected_type_response_otp")
    if kind=="CVV": return from_regex(_CVV,0.97,"expected_type_response_cvv")
    if kind=="PINCODE": return from_regex(_PINCODE,0.96,"expected_type_response_pincode")
    if kind=="DOB":
        x=from_regex(_DOB,0.97,"expected_type_response_dob")
        if x: return x
        for c in find_natural_date_candidates(tail)+find_spoken_date_candidates(tail):
            return _raw_item("DOB",base+c["start"],base+c["end"],c["value"],0.97,"expected_type_response_dob",canonical=c.get("canonical"),expected_type_state=True,ownership_strength=4)
        return None
    if kind=="PASSPORT":
        x=from_regex(_PASSPORT,0.97,"expected_type_response_passport")
        if x: return x
        for c in find_spoken_alnum_runs(tail):
            canonical=c["canonical"].upper()
            if re.fullmatch(r"[A-Z][1-9][0-9]{6}",canonical):
                return _raw_item("PASSPORT",base+c["start"],base+c["end"],c["value"],0.97,"expected_type_response_passport",canonical=canonical,expected_type_state=True,ownership_strength=4)
        return None
    if kind=="DRIVING_LICENSE":
        x=from_regex(_DRIVING_LICENSE,0.97,"expected_type_response_driving_license")
        if x: return x
        for c in find_spoken_driving_license_candidates(tail):
            return _raw_item("DRIVING_LICENSE",base+c["start"],base+c["end"],c["value"],0.97,"expected_type_response_driving_license",canonical=c["canonical"],expected_type_state=True,ownership_strength=4)
        return None
    if kind=="VOTER_ID": return from_regex(_VOTER_ID,0.97,"expected_type_response_voter")
    if kind=="CARD": return from_regex(_CARD,0.985,"expected_type_response_card",luhn_valid)
    if kind=="IFSC":
        for c in find_ifsc_candidates(tail):
            return _raw_item("IFSC",base+c["start"],base+c["end"],c["value"],0.985,"expected_type_response_ifsc",canonical=c["canonical"],expected_type_state=True,ownership_strength=4)
        return None
    if kind=="ADDRESS":
        s=marker.end()
        while s<len(sentence) and sentence[s] in " ,:-": s+=1
        span=_bounded_address_after(sentence,s,150)
        if span:
            lo,hi=span; body=sentence[lo:hi]
            if _address_structure_score(body)>=3:
                return _raw_item("ADDRESS",sent_start+lo,sent_start+hi,body,0.96,"expected_type_response_address",expected_type_state=True,ownership_strength=4)
    return None


def _find_expected_state_candidates(text: str) -> list[dict]:
    spans=_sentence_spans(text)
    out=[]
    for i in range(1,len(spans)):
        p0,p1=spans[i-1]; s0,s1=spans[i]
        expected=_infer_expected_type(text[p0:p1])
        if not expected: continue
        # TTL is exactly one sentence. If this sentence does not consume the state, it
        # is not carried any further.
        item=_expected_response_candidate(expected,text[s0:s1],s0)
        if item: out.append(item)
    return out


def _annotate_ownership(text: str, items: list[dict]) -> list[dict]:
    checks={
        "EMAIL":EMAIL_CONTEXT,"PAN":PAN_CONTEXT,"IFSC":IFSC_CONTEXT,"UPI":UPI_CONTEXT,
        "CARD":CARD_CONTEXT,"AADHAAR":AADHAAR_CONTEXT,"PHONE":PHONE_POSITIVE_CONTEXT,
        "ACCOUNT_NUMBER":ACCOUNT_CONTEXT,"OTP":OTP_CONTEXT,"CVV":CVV_CONTEXT,
        "PINCODE":PIN_CONTEXT,"DOB":DOB_CONTEXT,"PASSPORT":PASSPORT_CONTEXT,
        "VOTER_ID":VOTER_CONTEXT,"DRIVING_LICENSE":DL_CONTEXT,
    }
    personal_re=re.compile(r"(?i)\b(?:my|mine|our|i\s+(?:am|received|got|have|use)|call\s+me|reach\s+me|contact\s+me)\b")
    out=[]
    for src in items:
        x=dict(src)
        if "ownership_strength" in x:
            out.append(x); continue
        kind=x.get("type","")
        left=_role_left_context(text,x["start"],120)
        strength=0
        pat=checks.get(kind)
        if pat and pat.search(left):
            strength=4 if personal_re.search(left) else 3
        elif kind=="NAME" and x.get("evidence") in {"token_bounded_name_context","self_identification_name"}:
            strength=4
        elif kind=="ADDRESS":
            strength=4 if re.search(r"(?i)\b(?:my|your)\b",left) else 3
        elif kind in {"PASSWORD","USERNAME","API_KEY","AUTH_TOKEN"}:
            strength=4
        x["ownership_strength"]=strength
        out.append(x)
    return out


# ---------- spoken / normalized candidates ----------

def _spoken_candidates(text: str) -> list[dict]:
    out=[]
    for c in find_spoken_alnum_runs(text):
        ctx=_role_left_context(text,c["start"],100)
        canonical=c["canonical"].upper()
        if PAN_CONTEXT.search(ctx) and re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]",canonical):
            out.append(_raw_item("PAN",c["start"],c["end"],c["value"],0.96,"spoken_pan_normalized",canonical=canonical))
        elif IFSC_CONTEXT.search(ctx) and re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}",canonical):
            out.append(_raw_item("IFSC",c["start"],c["end"],c["value"],0.96,"spoken_ifsc_alphanumeric_normalized",canonical=canonical))
        elif DL_CONTEXT.search(ctx) and re.fullmatch(r"[A-Z]{2}[0-9]{2}(?:(?:19|20)[0-9]{2})?[0-9]{7}",canonical):
            out.append(_raw_item("DRIVING_LICENSE",c["start"],c["end"],c["value"],0.96,"spoken_driving_license_alphanumeric_normalized",canonical=canonical))
        elif PASSPORT_CONTEXT.search(ctx) and re.fullmatch(r"[A-Z][1-9][0-9]{6}",canonical):
            out.append(_raw_item("PASSPORT",c["start"],c["end"],c["value"],0.96,"spoken_passport_normalized",canonical=canonical))
    for c in find_spoken_email_candidates(text):
        ctx=_role_left_context(text,c["start"],88)
        domain=c["canonical"].rsplit("@",1)[-1]
        if EMAIL_CONTEXT.search(ctx) or domain.split(".",1)[0] in {"gmail","yahoo","outlook","hotmail","protonmail","icloud"}:
            out.append(_raw_item("EMAIL",c["start"],c["end"],c["value"],0.95,c["evidence"],canonical=c["canonical"]))
    for c in find_spoken_handle_candidates(text):
        ctx=_role_left_context(text,c["start"],90)
        provider=c["canonical"].rsplit("@",1)[-1]
        if UPI_CONTEXT.search(ctx) and (provider in UPI_HANDLES or len(provider)>=2):
            out.append(_raw_item("UPI",c["start"],c["end"],c["value"],0.96,"spoken_upi_normalized",canonical=c["canonical"]))
    for c in find_spoken_date_candidates(text):
        if DOB_CONTEXT.search(_role_left_context(text,c["start"],100)):
            out.append(_raw_item("DOB",c["start"],c["end"],c["value"],0.96,"spoken_dob_normalized",canonical=c["canonical"]))
    for c in find_spoken_driving_license_candidates(text):
        if DL_CONTEXT.search(_role_left_context(text,c["start"],110)):
            out.append(_raw_item("DRIVING_LICENSE",c["start"],c["end"],c["value"],0.96,"spoken_driving_license_normalized",canonical=c["canonical"]))
    for c in find_spoken_digit_runs(text):
        digits=c["digits"]; ctx=_role_left_context(text,c["start"],90)
        kind=conf=evidence=None
        span_start=c["start"]
        # Spoken IFSC example: "IFSC is ABCD zero one two three four five six".
        # The digit run normalizes to 0123456; the four-letter bank prefix immediately
        # before it is included in the protected source span.
        prefix=re.search(r"(?i)([A-Z]{4})\s*$",ctx)
        if IFSC_CONTEXT.search(ctx) and prefix and len(digits)==7 and digits.startswith("0"):
            kind,conf,evidence="IFSC",0.96,"spoken_ifsc_normalized"
            span_start=c["start"]-(len(ctx)-prefix.start(1))
        elif PHONE_POSITIVE_CONTEXT.search(ctx) and (
            (len(digits)==10 and digits[:1] in "6789") or
            (len(digits)==12 and digits.startswith("91") and digits[2:3] in "6789")
        ):
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
            if kind=="PHONE" and len(digits)==12 and digits.startswith("91"):
                pre=text[max(0,c["start"]-8):c["start"]]
                pm=re.search(r"(?i)\bplus\s+$",pre)
                if pm:
                    span_start=max(0,c["start"]-len(pre)+pm.start())
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

    # Shape validation is separate from semantic ownership. Strong-looking values in
    # documentation/reference roles are handled by the occurrence-context layer below.
    for m in _EMAIL.finditer(text):
        x=_item("EMAIL",m,0.995,"email_structure"); x["ai_review"]=True; found.append(x)
    for m in _PAN.finditer(text):
        left=_role_left_context(text,m.start(),88)
        if PAN_CONTEXT.search(left):
            found.append(_item("PAN",m,0.995,"pan_structure+pan_context"))
        else:
            x=_item("PAN",m,0.66,"pan_shape_without_ownership"); x["ai_review"]=True; found.append(x)

    # IFSC normalization: canonical is strong; separator/spoken forms need IFSC context.
    for c in find_ifsc_candidates(text):
        ctx=_role_left_context(text,c["start"],88)
        canonical_shape=(c["evidence"]=="ifsc_canonical")
        if IFSC_CONTEXT.search(ctx):
            conf=0.995
        elif canonical_shape:
            conf=0.91
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
        provider=m.group(0).rsplit("@",1)[-1].lower(); ctx=_role_left_context(text,m.start(),72)
        if provider in UPI_HANDLES:
            x=_item("UPI",m,0.985,"known_upi_handle"); x["ai_review"]=True; found.append(x)
        elif UPI_CONTEXT.search(ctx): found.append(_item("UPI",m,0.93,"upi_context"))

    for m in _CARD.finditer(text):
        if luhn_valid(m.group(0)):
            ctx=_role_left_context(text,m.start(),72)
            if CARD_CONTEXT.search(ctx):
                found.append(_item("CARD",m,0.995,"luhn_valid+card_context"))
            else:
                x=_item("CARD",m,0.88,"luhn_valid_without_ownership"); x["ai_review"]=True; found.append(x)

    for m in _AADHAAR.finditer(text):
        if len(_digits(m.group(0)))!=12: continue
        ctx=_role_left_context(text,m.start(),72)
        if verhoeff_valid(m.group(0)): found.append(_item("AADHAAR",m,0.995,"verhoeff_valid"))
        elif AADHAAR_CONTEXT.search(ctx):
            x=_item("AADHAAR",m,0.91,"aadhaar_context_checksum_unverified"); x["ai_review"]=True; found.append(x)

    for m in _PHONE.finditer(text):
        digits=_digits(m.group(0)); local=digits[-10:]
        if len(local)!=10 or local[0] not in "6789": continue
        left=_role_left_context(text,m.start(),84)
        positive=bool(PHONE_POSITIVE_CONTEXT.search(left) or PHONE_FOLLOWING_CONTEXT.search(_right_clause(text,m.end(),72)))
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
        ctx=_role_left_context(text,m.start(),96)
        if ACCOUNT_CONTEXT.search(ctx):
            found.append(_item("ACCOUNT_NUMBER",m,0.95,"account_context"))
        elif re.search(r"(?i)\b(?:banking\s+details|bank\s+details|debit\s+from|credit\s+to|beneficiary\s+details)\b",ctx):
            x=_item("ACCOUNT_NUMBER",m,0.73,"weak_banking_context"); x.update(ai_review=True,ner_review=True); found.append(x)

    for m in _OTP.finditer(text):
        ctx=_role_left_context(text,m.start(),72)
        if OTP_CONTEXT.search(ctx) and not CVV_CONTEXT.search(ctx): found.append(_item("OTP",m,0.97,"otp_context"))
        elif re.search(r"(?i)\b(?:verify|authenticate|login|sign\s*in)\b",ctx):
            x=_item("OTP",m,0.74,"weak_auth_context"); x.update(ai_review=True,ner_review=True); found.append(x)
    for m in _CVV.finditer(text):
        if CVV_CONTEXT.search(_role_left_context(text,m.start(),64)): found.append(_item("CVV",m,0.97,"cvv_context"))

    for m in _PINCODE.finditer(text):
        ctx=_role_left_context(text,m.start(),90)
        if PIN_CONTEXT.search(ctx) or re.search(r"(?i)\baddress\b",ctx): found.append(_item("PINCODE",m,0.94,"postal_context"))
        elif re.search(r"(?i)\b(?:live|reside|home|residence|staying|located)\b",ctx):
            x=_item("PINCODE",m,0.72,"weak_residential_context"); x.update(ai_review=True,ner_review=True); found.append(x)

    for m in _DOB.finditer(text):
        if DOB_CONTEXT.search(_role_left_context(text,m.start(),82)): found.append(_item("DOB",m,0.96,"dob_context"))
    for c in find_natural_date_candidates(text):
        if DOB_CONTEXT.search(_role_left_context(text,c["start"],90)):
            found.append(_raw_item("DOB",c["start"],c["end"],c["value"],0.96,"natural_language_dob",canonical=c["canonical"],ner_review=True))

    for m in _PASSPORT.finditer(text):
        if PASSPORT_CONTEXT.search(_role_left_context(text,m.start(),80)): found.append(_item("PASSPORT",m,0.97,"passport_context_structure"))
    for m in _VOTER_ID.finditer(text):
        if VOTER_CONTEXT.search(_role_left_context(text,m.start(),80)): found.append(_item("VOTER_ID",m,0.96,"voter_id_context_structure"))
    for m in _DRIVING_LICENSE.finditer(text):
        if DL_CONTEXT.search(_role_left_context(text,m.start(),88)): found.append(_item("DRIVING_LICENSE",m,0.96,"driving_license_context_structure"))

    found.extend(_find_name_candidates(text))
    found.extend(_find_address_candidates(text))
    found.extend(_find_partial_identifier_candidates(text))
    found.extend(_find_credential_candidates(text))
    found.extend(_spoken_candidates(text))
    found.extend(_find_expected_state_candidates(text))

    # v4.8 architecture: field-clause segmentation -> candidate -> validator ->
    # occurrence/discourse ownership -> context veto -> optional AI -> ownership-weighted
    # conflict resolver. Hard negative context runs before AI for speed and precision.
    found=_apply_occurrence_context_policy(text,found)
    reviewed=_fuse_ner_many(text,found,use_ner)
    reviewed=_fuse_semantic_many(text,reviewed,use_semantic)
    reviewed=_annotate_ownership(text,reviewed)
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
        # A last-four candidate contains only the four sensitive digits. Partial mode
        # must therefore not reveal its suffix again.
        if mode=="partial" and not e.get("partial_identifier"):
            repl=_partial_mask(e["type"],e.get("value",""))
        else:
            repl=f"[{e['type']} REDACTED]"
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
            "masked_value":(_partial_mask(e["type"],e.get("value","")) if reveal_suffix and not e.get("partial_identifier") else f"[{e['type']} REDACTED]"),
        })
    return out
