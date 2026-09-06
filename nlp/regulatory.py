"""Configurable regulatory/compliance phrase screening.

This module flags evidence for human review; it does not determine legal compliance.
Rules should be approved by the lender's compliance/legal team for the relevant
jurisdiction and call type.
"""
from __future__ import annotations

import re

_DISCLOSURES = [
    ("recording_disclosure", re.compile(r"(?i)\b(?:this|the)\s+call\s+(?:is|may\s+be)\s+(?:being\s+)?recorded\b"), "medium"),
    ("quality_monitoring_disclosure", re.compile(r"(?i)\b(?:quality|training|monitoring)\s+purposes?\b"), "low"),
    ("otp_safety_warning", re.compile(r"(?i)\b(?:do\s+not|don't|never)\s+share\s+(?:your\s+)?(?:otp|one[- ]time\s+password)\b"), "low"),
    ("grievance_channel", re.compile(r"(?i)\b(?:raise|register|file)\s+(?:a\s+)?(?:grievance|complaint)\b"), "low"),
]

_RISK_PATTERNS = [
    ("coercive_arrest_threat", re.compile(r"(?i)\b(?:you\s+will\s+be\s+arrested|police\s+will\s+arrest|send\s+the\s+police)\b"), "high", 0.97),
    ("public_shaming_threat", re.compile(r"(?i)\b(?:we\s+will\s+shame\s+you|tell\s+your\s+(?:employer|neighbours?|friends?)|post\s+your\s+details)\b"), "high", 0.96),
    ("third_party_debt_disclosure", re.compile(r"(?i)\b(?:tell|inform|contact)\s+your\s+(?:employer|family|relative|neighbour)\s+(?:about|regarding)\s+(?:your\s+)?(?:loan|debt|overdue|payment)\b"), "high", 0.95),
    ("physical_threat", re.compile(r"(?i)\b(?:we\s+will\s+(?:come\s+to|visit)\s+your\s+home\s+and|you\s+will\s+regret\s+this|we\s+will\s+teach\s+you\s+a\s+lesson)\b"), "high", 0.93),
    ("misleading_authority_claim", re.compile(r"(?i)\b(?:rbi|court|police)\s+(?:has|have)\s+(?:ordered|approved)\s+(?:your\s+)?(?:arrest|seizure|immediate\s+payment)\b"), "high", 0.94),
    ("aggressive_insult", re.compile(r"(?i)\b(?:idiot|stupid|useless|moron)\b"), "medium", 0.90),
]


def check_regulatory(text: str, *, require_recording_disclosure: bool = True) -> dict:
    detected=[]; present=[]; missing=[]
    for rule,pat,severity in _DISCLOSURES:
        m=pat.search(text)
        if m:
            present.append({"rule":rule,"severity":severity,"confidence":0.92,"start":m.start(),"end":m.end()})
        elif rule=="recording_disclosure" and require_recording_disclosure:
            missing.append({"rule":rule,"severity":"medium","confidence":0.85})
    for typ,pat,severity,confidence in _RISK_PATTERNS:
        for m in pat.finditer(text):
            detected.append({
                "type":typ,"severity":severity,"confidence":confidence,
                "start":m.start(),"end":m.end(),
                "evidence":"phrase_pattern",
            })
    risk_order={"low":1,"medium":2,"high":3}
    max_risk=max((risk_order.get(x["severity"],0) for x in detected),default=0)
    return {
        "present_disclosures":present,
        "missing_disclosures":missing,
        "risk_phrases":detected,
        "risk_level":{0:"low",1:"low",2:"medium",3:"high"}[max_risk],
        "note":"Automated screening for review only; configure against approved lender policy and applicable law.",
    }
