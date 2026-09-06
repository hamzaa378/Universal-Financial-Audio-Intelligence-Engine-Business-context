"""ASR-aware normalization helpers for privacy detection.

The functions in this module never replace the source transcript in place.  They emit
candidates that retain the *original character span* plus a normalized/canonical value.
That lets the privacy layer mask exactly what the user said while validators operate on
stable representations.
"""
from __future__ import annotations

import re
from datetime import datetime

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
_DIGIT_TOKEN = r"(?:zero|oh|o|one|two|three|four|five|six|seven|eight|nine|double|triple)"
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
    out = []
    for m in _SPOKEN_EMAIL.finditer(text):
        local = _normalize_email_side(m.group(1))
        domain = _normalize_email_side(m.group(2), domain=True)
        canonical = f"{local}@{domain}"
        # Conservative structural check after normalization.
        if re.fullmatch(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", canonical, flags=re.I):
            out.append({
                "start": m.start(), "end": m.end(), "value": m.group(0),
                "canonical": canonical, "evidence": "spoken_email_multilevel",
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
