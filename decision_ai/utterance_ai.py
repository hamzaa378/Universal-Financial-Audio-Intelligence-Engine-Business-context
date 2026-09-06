"""Shared semantic AI for intent and abusive-tone triage.

v4.4 evaluates every intent label plus abusive-tone evidence in one batched semantic
call. This removes the repeated prototype encoding that dominated first-run latency in
v4.3.
"""
from __future__ import annotations

from decision_ai.semantic_engine import semantic_scores_many, warmup

INTENT_PROTOTYPES = {
    "payment commitment": (
        "I will pay the overdue EMI tomorrow.",
        "I promise to clear the payment by Friday.",
        "Main kal EMI bhar dunga.",
    ),
    "payment completed": (
        "I already made the payment yesterday.",
        "The EMI was paid but still shows overdue.",
        "Maine payment already kar diya hai.",
    ),
    "loan enquiry": (
        "I want to know loan eligibility and interest rate.",
        "How can I apply for this loan?",
    ),
    "repayment difficulty": (
        "I cannot pay the EMI because I have financial difficulty.",
        "I need more time to repay.",
        "Abhi payment karna possible nahi hai.",
    ),
    "grievance or complaint": (
        "I want to raise a complaint about unfair collection calls.",
        "This service problem needs escalation.",
    ),
    "identity verification": (
        "I am calling to complete KYC identity verification.",
        "Please verify my PAN and Aadhaar details.",
    ),
    "fraud report": (
        "This transaction was not authorized by me.",
        "I want to report fraud on my account.",
    ),
    "foreclosure or prepayment": (
        "I want to foreclose the loan early.",
        "Tell me the prepayment charges.",
    ),
    "settlement negotiation": (
        "Can we negotiate a settlement for the outstanding loan?",
        "Please waive part of the overdue amount.",
    ),
    "payment dispute": (
        "I do not owe this amount and I dispute the payment demand.",
        "The overdue amount shown is incorrect.",
    ),
    "general customer support": (
        "I need help with my account.",
        "Please explain the status of my request.",
    ),
}

ABUSE_POS = (
    "The caller is insulting and verbally abusing the agent.",
    "The customer is using aggressive abusive language and profanity.",
    "The speaker is swearing at the other person.",
    "The caller is threatening and humiliating the other person.",
)
ABUSE_NEG = (
    "The customer is calmly discussing a financial problem.",
    "The agent explains the loan process professionally.",
    "The caller is dissatisfied but uses normal non-abusive language.",
)


def prototype_texts() -> list[str]:
    out = [x for vals in INTENT_PROTOTYPES.values() for x in vals]
    out.extend(ABUSE_POS); out.extend(ABUSE_NEG)
    return list(dict.fromkeys(out))


def warmup_utterance_ai() -> dict:
    return warmup(prototype_texts())


def analyze_utterance(text: str) -> dict | None:
    labels = list(INTENT_PROTOTYPES)
    all_intent_examples = {label: vals for label, vals in INTENT_PROTOTYPES.items()}
    cases = []
    for label in labels:
        positives = all_intent_examples[label]
        negatives = tuple(
            x for other, vals in all_intent_examples.items() if other != label for x in vals
        )
        cases.append({
            "text": text,
            "positive_examples": positives,
            "negative_examples": negatives,
        })
    cases.append({
        "text": text,
        "positive_examples": ABUSE_POS,
        "negative_examples": ABUSE_NEG,
    })
    scored = semantic_scores_many(cases)
    if not scored or not scored[0].get("available"):
        return None

    intent_scores = [(float(r["score"]), label) for r, label in zip(scored[:-1], labels)]
    intent_scores.sort(reverse=True)
    top_score, top_label = intent_scores[0]
    second_score = intent_scores[1][0] if len(intent_scores) > 1 else 0.0
    # Confidence deliberately accounts for separation from the runner-up.
    separation = max(0.0, min(1.0, 0.5 + (top_score-second_score)))
    conf = max(0.45, min(0.97, 0.58*top_score + 0.42*separation))
    alternatives = [
        {"label": label, "score": round(score, 4)}
        for score, label in intent_scores[1:4]
    ]

    abuse_score = float(scored[-1]["score"])
    abuse = {
        "available": True,
        "score": round(abuse_score, 4),
        "level": "high" if abuse_score >= 0.78 else "medium" if abuse_score >= 0.60 else "low",
        "method": "fastembed_semantic_batched",
    }
    intent = {
        "label": top_label,
        "confidence": round(conf, 4),
        "method": "fastembed_semantic_batched",
        "runner_up_margin": round(top_score-second_score, 4),
        "alternatives": alternatives,
    }
    return {"intent": intent, "abusive_tone": abuse}


def semantic_intent(text: str) -> dict | None:
    r = analyze_utterance(text)
    return None if r is None else r["intent"]


def abusive_tone(text: str) -> dict:
    r = analyze_utterance(text)
    if r is None:
        return {"available": False, "score": 0.0, "level": "unknown", "method": "rules_only"}
    return r["abusive_tone"]
