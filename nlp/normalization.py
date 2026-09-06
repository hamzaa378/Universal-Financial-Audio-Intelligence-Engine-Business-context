"""ASR-aware normalization helpers for privacy detection.

The functions in this module never replace the source transcript in place.  They emit
candidates that retain the *original character span* plus a normalized/canonical value.
That lets the privacy layer mask exactly what the user said while validators operate on
stable representations.
"""
from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher

_SAFE_ID_SEP_RE = re.compile(r"[\s._\-‐‑‒–—]+")


def canonical_ifsc(value: str) -> str | None:
    """Return canonical IFSC if *value* can safely normalize to one.

    RBI-style IFSC syntax is four alphabetic bank characters, literal 0, then six
    alphanumeric branch characters.  Separator removal is intentionally limited to
    whitespace, dot, underscore and hyphen; arbitrary punctuation is not stripped.
    """
    raw = value.strip().upper()
    compact = _SAFE_ID_SEP_RE.sub("", raw)
    if re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", compact):
        return compact
    return None


_IFSC_CANONICAL = re.compile(r"(?i)(?<![A-Z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Z0-9])")
# Handles ASR/typed forms such as ABCD-0123456, ABCD 0123456, ABCD_0ABC123.
_IFSC_SEPARATED = re.compile(
    r"(?i)(?<![A-Z0-9])(?:[A-Z]{4})[\s._\-‐‑‒–—]+0[\s._\-‐‑‒–—]*(?:[A-Z0-9]{6})(?![A-Z0-9])"
)
# Spoken zero is accepted only by the caller when IFSC context is present.
_IFSC_SPOKEN_ZERO = re.compile(
    r"(?i)(?<![A-Z0-9])([A-Z]{4})\s+(?:zero|oh)\s+([A-Z0-9]{6})(?![A-Z0-9])"
)


def find_ifsc_candidates(text: str) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for pattern, evidence in (
        (_IFSC_CANONICAL, "ifsc_canonical"),
        (_IFSC_SEPARATED, "ifsc_separator_normalized"),
    ):
        for m in pattern.finditer(text):
            canonical = canonical_ifsc(m.group(0))
            if canonical and (m.start(), m.end()) not in seen:
                out.append({
                    "start": m.start(), "end": m.end(), "value": m.group(0),
                    "canonical": canonical, "evidence": evidence,
                })
                seen.add((m.start(), m.end()))
    for m in _IFSC_SPOKEN_ZERO.finditer(text):
        canonical = (m.group(1) + "0" + m.group(2)).upper()
        if canonical_ifsc(canonical) and (m.start(), m.end()) not in seen:
            out.append({
                "start": m.start(), "end": m.end(), "value": m.group(0),
                "canonical": canonical, "evidence": "ifsc_spoken_zero",
                "requires_context": True,
            })
            seen.add((m.start(), m.end()))
    return sorted(out, key=lambda x: x["start"])


_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
# Long/complete words must appear before the single-letter ASR variant `o`.
# Otherwise regex alternation can consume the `o` at the start of `one` and silently
# normalize spoken 1 as 0.
_DIGIT_TOKEN = r"(?:double|triple|zero|oh|one|two|three|four|five|six|seven|eight|nine|o)"
_SPOKEN_DIGIT_RUN = re.compile(rf"(?i)(?<!\w){_DIGIT_TOKEN}(?:[\s,.;:\-]+{_DIGIT_TOKEN}){{2,}}(?!\w)")


def parse_spoken_digits(raw: str) -> str:
    out: list[str] = []
    repeat = 1
    for token in re.findall(_DIGIT_TOKEN, raw, flags=re.I):
        t = token.lower()
        if t == "double":
            repeat = 2
            continue
        if t == "triple":
            repeat = 3
            continue
        d = _DIGIT_WORDS[t]
        out.extend([d] * repeat)
        repeat = 1
    return "".join(out)


def find_spoken_digit_runs(text: str) -> list[dict]:
    return [
        {
            "start": m.start(), "end": m.end(), "value": m.group(0),
            "digits": parse_spoken_digits(m.group(0)),
        }
        for m in _SPOKEN_DIGIT_RUN.finditer(text)
    ]


# Spoken alphanumeric identifiers (NATO/phonetic alphabet + digit words). These are
# emitted as candidates only; the PII layer still requires a type-specific context and
# validates the canonical PAN/IFSC shape before masking.
_NATO_WORDS = {
    "alpha":"A", "bravo":"B", "charlie":"C", "delta":"D", "echo":"E",
    "foxtrot":"F", "golf":"G", "hotel":"H", "india":"I", "juliett":"J", "juliet":"J",
    "kilo":"K", "lima":"L", "mike":"M", "november":"N", "oscar":"O", "papa":"P",
    "quebec":"Q", "romeo":"R", "sierra":"S", "tango":"T", "uniform":"U",
    "victor":"V", "whiskey":"W", "xray":"X", "x-ray":"X", "yankee":"Y", "zulu":"Z",
}
_NATO_TOKEN = "(?:" + "|".join(sorted((re.escape(x) for x in _NATO_WORDS), key=len, reverse=True)) + ")"
_SPOKEN_ALNUM_TOKEN = rf"(?:{_NATO_TOKEN}|[A-Z]|\d{{1,12}}|{_DIGIT_TOKEN})"
_SPOKEN_ALNUM_RUN = re.compile(
    rf"(?i)(?<!\w){_SPOKEN_ALNUM_TOKEN}(?:[\s,.;:\-]+{_SPOKEN_ALNUM_TOKEN}){{1,}}(?!\w)"
)
_SPOKEN_ALNUM_TOKEN_RE = re.compile(rf"(?i)(?<!\w)({_SPOKEN_ALNUM_TOKEN})(?!\w)")


def parse_spoken_alnum(raw: str) -> str:
    out=[]
    repeat=1
    for m in _SPOKEN_ALNUM_TOKEN_RE.finditer(raw):
        token=m.group(1)
        t=token.casefold()
        if t=="double":
            repeat=2; continue
        if t=="triple":
            repeat=3; continue
        if t in _DIGIT_WORDS:
            value=_DIGIT_WORDS[t]
        elif token.isdigit():
            value=token
        elif t in _NATO_WORDS:
            value=_NATO_WORDS[t]
        elif len(token)==1 and token.isalpha():
            value=token.upper()
        else:
            repeat=1; continue
        out.extend([value]*repeat)
        repeat=1
    return "".join(out)


def find_spoken_alnum_runs(text: str) -> list[dict]:
    out=[]
    for m in _SPOKEN_ALNUM_RUN.finditer(text):
        raw=m.group(0)
        tokens=[x.group(1) for x in _SPOKEN_ALNUM_TOKEN_RE.finditer(raw)]
        has_alpha=any(t.casefold() in _NATO_WORDS or (len(t)==1 and t.isalpha() and t.casefold() not in _DIGIT_WORDS) for t in tokens)
        if not has_alpha:
            continue
        canonical=parse_spoken_alnum(raw)
        if canonical:
            out.append({"start":m.start(),"end":m.end(),"value":raw,"canonical":canonical,"evidence":"spoken_alphanumeric_normalized"})
    return out


# Multi-level email recognizer. Examples:
#   hamza dot ahmad at gmail dot com
#   john underscore doe at bank dot co dot in
#   ryan dash paul at outlook dot com
_WORD = r"[a-z0-9]+"
_LOCAL_CONNECTOR = r"(?:dot|underscore|under\s*score|dash|hyphen)"
_DOMAIN_CONNECTOR = r"dot"
_SPOKEN_EMAIL = re.compile(
    rf"(?i)(?<!\w)"
    rf"({_WORD}(?:\s+{_LOCAL_CONNECTOR}\s+{_WORD})*)"
    rf"\s+(?:at|at\s+the\s+rate(?:\s+of)?)\s+"
    rf"({_WORD}(?:\s+{_DOMAIN_CONNECTOR}\s+{_WORD})+)"
    rf"(?!\w)"
)


def _normalize_email_side(raw: str, *, domain: bool = False) -> str:
    s = re.sub(r"(?i)\s+(?:under\s*score|underscore)\s+", "_", raw)
    s = re.sub(r"(?i)\s+(?:dash|hyphen)\s+", "-", s)
    s = re.sub(r"(?i)\s+dot\s+", ".", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def find_spoken_email_candidates(text: str) -> list[dict]:
    out=[]; pos=0; seen=set()
    while pos < len(text):
        m=_SPOKEN_EMAIL.search(text,pos)
        if not m: break
        pos=m.start()+1
        local=_normalize_email_side(m.group(1))
        if local in {"me","you","him","her","them","it"}:
            continue
        domain=_normalize_email_side(m.group(2),domain=True)
        canonical=f"{local}@{domain}"
        key=(m.start(),m.end())
        if key in seen: continue
        seen.add(key)
        if re.fullmatch(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",canonical,flags=re.I):
            out.append({
                "start":m.start(),"end":m.end(),"value":m.group(0),
                "canonical":canonical,"evidence":"spoken_email_multilevel",
            })
    return out


# Spoken UPI/VPA handles. Unlike an email address, a valid UPI handle does not need a
# dotted DNS-style domain, so this is intentionally kept separate from the spoken email
# normalizer. Context/known-handle validation is still performed by the PII layer.
_SPOKEN_HANDLE = re.compile(
    rf"(?i)(?<!\w)"
    rf"({_WORD}(?:\s+{_LOCAL_CONNECTOR}\s+{_WORD})*)"
    rf"\s+(?:at|at\s+the\s+rate(?:\s+of)?)\s+"
    rf"({_WORD})"
    rf"(?!\w)"
)


def find_spoken_handle_candidates(text: str) -> list[dict]:
    out=[]; pos=0; seen=set()
    while pos < len(text):
        m=_SPOKEN_HANDLE.search(text,pos)
        if not m: break
        pos=m.start()+1
        local=_normalize_email_side(m.group(1))
        if local in {"me","you","him","her","them","it"}:
            continue
        handle=re.sub(r"\s+","",m.group(2)).lower()
        canonical=f"{local}@{handle}"
        key=(m.start(),m.end())
        if key in seen: continue
        seen.add(key)
        if re.fullmatch(r"[a-z0-9._-]{2,64}@[a-z][a-z0-9_-]{1,31}",canonical,re.I):
            out.append({
                "start":m.start(),"end":m.end(),"value":m.group(0),
                "canonical":canonical,"evidence":"spoken_handle_normalized",
            })
    return out


# ASR often emits the separators literally ("sara.private at mail.co.in") rather
# than saying "dot". These candidates intentionally preserve the original span and
# only normalize the spoken `at`; ownership/type policy remains in the PII layer.
_MIXED_AT_ADDRESS = re.compile(
    r"(?i)(?<![\w.+-])([a-z0-9][a-z0-9._%+-]{1,63})\s+(?:at|at\s+the\s+rate(?:\s+of)?)\s+"
    r"([a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*)(?![\w-])"
)


def find_mixed_at_candidates(text: str) -> list[dict]:
    """Return ASR `local at provider/domain` candidates.

    Overlapping searches are intentional: in "mail me at sara.private at mail.co.in"
    the first syntactic `at` is not part of the address, while the second one is.
    """
    out=[]; pos=0; seen=set()
    while pos < len(text):
        m=_MIXED_AT_ADDRESS.search(text,pos)
        if not m: break
        pos=m.start()+1
        prefix=text[max(0,m.start()-18):m.start()]
        if re.search(r"(?i)\b(?:dot|underscore|under\s*score|dash|hyphen)\s+$",prefix):
            continue
        local=m.group(1).lower()
        provider=m.group(2).lower()
        # Command pronouns are almost never the actual local part; skipping them lets
        # the overlapping scan find the later real candidate.
        if local in {"me","you","him","her","them","it"}:
            continue
        key=(m.start(),m.end())
        if key in seen: continue
        seen.add(key)
        canonical=f"{local}@{provider}"
        out.append({
            "start":m.start(),"end":m.end(),"value":m.group(0),
            "canonical":canonical,"provider":provider,"evidence":"asr_mixed_at_normalized",
        })
    return out


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_RE = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_TEXT_DOB_DMY = re.compile(rf"(?i)(?<!\d)([0-3]?\d)(?:st|nd|rd|th)?\s+({_MONTH_RE})\s+((?:19|20)\d{{2}})(?!\d)")
_TEXT_DOB_MDY = re.compile(rf"(?i)\b({_MONTH_RE})\s+([0-3]?\d)(?:st|nd|rd|th)?(?:,)?\s+((?:19|20)\d{{2}})(?!\d)")


def _valid_date(day: int, month: int, year: int) -> bool:
    try:
        datetime(year, month, day)
        return True
    except ValueError:
        return False


def find_natural_date_candidates(text: str) -> list[dict]:
    out = []
    for m in _TEXT_DOB_DMY.finditer(text):
        day, month_name, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        month = _MONTHS[month_name]
        if _valid_date(day, month, year):
            out.append({"start":m.start(),"end":m.end(),"value":m.group(0),"canonical":f"{year:04d}-{month:02d}-{day:02d}","evidence":"natural_date"})
    for m in _TEXT_DOB_MDY.finditer(text):
        month_name, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        month = _MONTHS[month_name]
        if _valid_date(day, month, year):
            out.append({"start":m.start(),"end":m.end(),"value":m.group(0),"canonical":f"{year:04d}-{month:02d}-{day:02d}","evidence":"natural_date"})
    # Resolve duplicate matches in case patterns overlap.
    dedup = {(x["start"],x["end"]):x for x in out}
    return sorted(dedup.values(), key=lambda x:x["start"])


# Conservative spoken-date support for DOB phrases such as
# "twenty-first January two thousand two". We deliberately do not run a broad word-to-
# number conversion over arbitrary prose; only a day-word + month + year-word pattern is
# emitted, and the PII layer still requires DOB ownership.
_DAY_WORDS = {
    "first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,
    "eighth":8,"ninth":9,"tenth":10,"eleventh":11,"twelfth":12,"thirteenth":13,
    "fourteenth":14,"fifteenth":15,"sixteenth":16,"seventeenth":17,"eighteenth":18,
    "nineteenth":19,"twentieth":20,"twenty-first":21,"twenty first":21,
    "twenty-second":22,"twenty second":22,"twenty-third":23,"twenty third":23,
    "twenty-fourth":24,"twenty fourth":24,"twenty-fifth":25,"twenty fifth":25,
    "twenty-sixth":26,"twenty sixth":26,"twenty-seventh":27,"twenty seventh":27,
    "twenty-eighth":28,"twenty eighth":28,"twenty-ninth":29,"twenty ninth":29,
    "thirtieth":30,"thirty-first":31,"thirty first":31,
}
_DAY_WORD_RE="(?:"+"|".join(sorted((re.escape(x) for x in _DAY_WORDS),key=len,reverse=True))+")"
_YEAR_SMALL={
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,
    "twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,
}
_YEAR_WORD_RE=r"(?:nineteen|two\s+thousand)(?:\s+(?:and\s+)?[a-z-]+(?:\s+[a-z-]+)?)?"
_SPOKEN_DATE_DMY=re.compile(rf"(?i)(?<!\w)({_DAY_WORD_RE})\s+({_MONTH_RE})\s+({_YEAR_WORD_RE})(?!\w)")


def _parse_spoken_year(raw: str) -> int | None:
    words=raw.casefold().replace("-"," ").split()
    if not words: return None
    if words[0]=="nineteen":
        base=1900; rest=words[1:]
    elif len(words)>=2 and words[0]=="two" and words[1]=="thousand":
        base=2000; rest=words[2:]
    else:
        return None
    rest=[w for w in rest if w!="and"]
    if not rest: return base
    total=0
    for w in rest:
        if w not in _YEAR_SMALL: return None
        total+=_YEAR_SMALL[w]
    year=base+total
    return year if 1900<=year<=2099 else None


def find_spoken_date_candidates(text: str) -> list[dict]:
    out=[]
    for m in _SPOKEN_DATE_DMY.finditer(text):
        day=_DAY_WORDS[m.group(1).casefold()]
        month=_MONTHS[m.group(2).casefold()]
        year=_parse_spoken_year(m.group(3))
        if year is not None and _valid_date(day,month,year):
            out.append({
                "start":m.start(),"end":m.end(),"value":m.group(0),
                "canonical":f"{year:04d}-{month:02d}-{day:02d}",
                "evidence":"spoken_natural_date",
            })
    return out


# State-name aware spoken Indian driving-licence normalization. This is intentionally
# context-gated by the caller and only emits a candidate when the resulting canonical
# value matches the structural DL validator.
_INDIA_STATE_DL = {
    "andhra pradesh":"AP","arunachal pradesh":"AR","assam":"AS","bihar":"BR",
    "chhattisgarh":"CG","goa":"GA","gujarat":"GJ","haryana":"HR","himachal pradesh":"HP",
    "jharkhand":"JH","karnataka":"KA","kerala":"KL","madhya pradesh":"MP",
    "maharashtra":"MH","manipur":"MN","meghalaya":"ML","mizoram":"MZ","nagaland":"NL",
    "odisha":"OD","orissa":"OR","punjab":"PB","rajasthan":"RJ","sikkim":"SK",
    "tamil nadu":"TN","telangana":"TS","tripura":"TR","uttar pradesh":"UP",
    "uttarakhand":"UK","west bengal":"WB","delhi":"DL","chandigarh":"CH",
    "puducherry":"PY","pondicherry":"PY","jammu and kashmir":"JK","ladakh":"LA",
}


def _normalize_state_phrase(raw: str) -> str | None:
    """Map a state name to a DL prefix with conservative ASR fuzziness.

    Exact matches always win. Fuzzy matching is intentionally limited to state-name
    candidates and requires a high similarity; callers additionally require explicit
    driving-licence context and a structurally valid numeric suffix.
    """
    key=re.sub(r"[^a-z ]+","",raw.casefold()).strip()
    key=re.sub(r"\s+"," ",key)
    if key in _INDIA_STATE_DL:
        return _INDIA_STATE_DL[key]
    best=None; best_score=0.0
    for name,code in _INDIA_STATE_DL.items():
        score=SequenceMatcher(None,key,name).ratio()
        if score>best_score:
            best_score=score; best=code
    return best if best_score>=0.76 else None

_STATE_WORD_RE="(?:"+"|".join(sorted((re.escape(x) for x in _INDIA_STATE_DL),key=len,reverse=True))+")"
_SPOKEN_DL_STATE=re.compile(rf"(?i)(?<!\w)({_STATE_WORD_RE})\s+({_DIGIT_TOKEN}(?:[\s,.;:\-]+{_DIGIT_TOKEN}){{8,16}})(?!\w)")
_ASR_DL_STATE_MIXED=re.compile(r"(?i)(?<!\w)([A-Za-z][A-Za-z ]{2,24}?)[\s,.;:\-]*((?:\d[\s,.;:\-]*){9,18})(?!\d)")


def find_spoken_driving_license_candidates(text: str) -> list[dict]:
    out=[]; seen=set()
    for m in _SPOKEN_DL_STATE.finditer(text):
        state=_INDIA_STATE_DL[m.group(1).casefold()]
        digits=parse_spoken_digits(m.group(2))
        canonical=state+digits
        if re.fullmatch(r"[A-Z]{2}[0-9]{2}(?:(?:19|20)[0-9]{2})?[0-9]{7}",canonical):
            out.append({
                "start":m.start(),"end":m.end(),"value":m.group(0),
                "canonical":canonical,"evidence":"spoken_state_driving_license",
            }); seen.add((m.start(),m.end()))
    # Mixed ASR form: "Carnitaka0120210012345" or "Karnatika 01 2021 0012345".
    for m in _ASR_DL_STATE_MIXED.finditer(text):
        if (m.start(),m.end()) in seen: continue
        state=_normalize_state_phrase(m.group(1))
        if not state: continue
        digits=re.sub(r"\D","",m.group(2))
        canonical=state+digits
        if re.fullmatch(r"[A-Z]{2}[0-9]{2}(?:(?:19|20)[0-9]{2})?[0-9]{7}",canonical):
            out.append({
                "start":m.start(),"end":m.end(),"value":m.group(0),
                "canonical":canonical,"evidence":"asr_fuzzy_state_driving_license",
            })
    return sorted(out,key=lambda x:x["start"])
