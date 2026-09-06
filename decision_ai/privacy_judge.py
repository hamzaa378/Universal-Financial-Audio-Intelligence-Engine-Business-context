"""Semantic second opinion for ambiguous privacy candidates."""
from __future__ import annotations

from decision_ai.semantic_engine import semantic_score, semantic_scores_many, status, warmup

_PROTOTYPES = {
    "PHONE": {
        "positive": (
            "You can contact me at <VALUE>.",
            "Please call me on my mobile number <VALUE>.",
            "Reach me on <VALUE> if the payment fails.",
            "Mujhe <VALUE> par call karna.",
        ),
        "negative": (
            "The order number is <VALUE>.",
            "Transaction reference <VALUE> was generated.",
            "The customer ID is <VALUE>.",
            "Ticket number <VALUE> is closed.",
            "Case number <VALUE> was escalated.",
        ),
    },
    "ACCOUNT_NUMBER": {
        "positive": (
            "The money should be sent to my bank account <VALUE>.",
            "These are my banking details <VALUE>.",
            "Debit the loan repayment from account <VALUE>.",
        ),
        "negative": (
            "Invoice number is <VALUE>.",
            "Application reference is <VALUE>.",
            "Tracking identifier <VALUE> was created.",
        ),
    },
    "PINCODE": {
        "positive": (
            "I live in Bengaluru postal area <VALUE>.",
            "My home location has PIN code <VALUE>.",
            "Deliver the KYC document to my residence <VALUE>.",
        ),
        "negative": (
            "The approved amount is <VALUE> rupees.",
            "The verification reference is <VALUE>.",
            "The six digit product code is <VALUE>.",
        ),
    },
    "AADHAAR": {
        "positive": (
            "This is my Aadhaar identity number <VALUE>.",
            "Use UIDAI identity <VALUE> for KYC verification.",
        ),
        "negative": (
            "The documentation contains a twelve digit sample <VALUE>.",
            "The transaction record number is <VALUE>.",
        ),
    },
    "ADDRESS": {
        "positive": (
            "I live at <VALUE>.",
            "Please send the statement to my home at <VALUE>.",
            "My residential location is <VALUE>.",
            "Mera ghar ka address <VALUE> hai.",
        ),
        "negative": (
            "The branch is located on MG Road.",
            "We are explaining what an address field means.",
            "The loan is for a property on this road.",
        ),
    },
    "OTP": {
        "positive": (
            "Use <VALUE> to verify this login.",
            "I received <VALUE> to authenticate the transaction.",
            "The one time password is <VALUE>.",
        ),
        "negative": (
            "The EMI amount is <VALUE> rupees.",
            "The case reference is <VALUE>.",
        ),
    },
}


def prototype_texts() -> list[str]:
    return list(dict.fromkeys(x for p in _PROTOTYPES.values() for side in ("positive", "negative") for x in p[side]))


def warmup_privacy_judge() -> dict:
    return warmup(prototype_texts())


def _safe_context(text: str, item: dict, radius: int = 72) -> str:
    s, e = item["start"], item["end"]
    lo = max(0, s-radius); hi = min(len(text), e+radius)
    for sep in (".", "!", "?", "\n", ";"):
        pos = text.rfind(sep, lo, s)
        if pos >= lo:
            lo = max(lo, pos+1)
        pos2 = text.find(sep, e, hi)
        if pos2 != -1:
            hi = min(hi, pos2)
    return (text[lo:s] + " <VALUE> " + text[e:hi]).strip()


def judge(text: str, item: dict) -> dict:
    proto = _PROTOTYPES.get(item["type"])
    if not proto:
        return {"available": False, "score": 0.5}
    return semantic_score(_safe_context(text, item), proto["positive"], proto["negative"])


def judge_many(text: str, items: list[dict]) -> list[dict]:
    cases = []
    positions = []
    results = [{"available": False, "score": 0.5} for _ in items]
    for i, item in enumerate(items):
        proto = _PROTOTYPES.get(item.get("type"))
        if not proto:
            continue
        positions.append(i)
        cases.append({
            "text": _safe_context(text, item),
            "positive_examples": proto["positive"],
            "negative_examples": proto["negative"],
        })
    scored = semantic_scores_many(cases)
    for i, r in zip(positions, scored):
        results[i] = r
    return results


def engine_status() -> dict:
    return status()
