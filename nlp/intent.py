from __future__ import annotations

from functools import lru_cache
from config import SETTINGS
from decision_ai.utterance_ai import analyze_utterance

LABELS=[
    "payment commitment","payment completed","loan enquiry","repayment difficulty",
    "grievance or complaint","identity verification","fraud report",
    "foreclosure or prepayment","settlement negotiation","payment dispute",
    "general customer support",
]
KEYWORDS={
    "payment commitment":["will pay","i'll pay","pay tomorrow","pay by","promise to pay","clear the dues","emi bhar"],
    "payment completed":["already paid","payment done","i paid","transferred","payment completed","kar diya"],
    "repayment difficulty":["cannot pay","can't pay","cant pay","unable to pay","financial difficulty","hardship","lost my job","need more time"],
    "grievance or complaint":["complaint","grievance","unfair","harassment","ombudsman","escalate"],
    "fraud report":["fraud","unauthorized","unauthorised","not my transaction","scam","card stolen"],
    "foreclosure or prepayment":["foreclose","foreclosure","prepay","prepayment","close the loan early"],
    "settlement negotiation":["settlement","waive","waiver","negotiate","one time settlement"],
    "payment dispute":["dispute this amount","do not owe","don't owe","amount is wrong","amount is incorrect","not my loan"],
    "identity verification":["verify","verification","kyc","date of birth","pan","aadhaar","passport"],
    "loan enquiry":["loan eligibility","interest rate","apply for loan","loan application","how much loan"],
}


@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline
    return pipeline("zero-shot-classification",model=SETTINGS.nli_model)


def _rule_scores(text: str) -> dict[str,int]:
    t=text.casefold()
    return {label:sum(1 for k in kws if k in t) for label,kws in KEYWORDS.items()}


def _rules(text: str) -> dict:
    scores=_rule_scores(text)
    ranked=sorted(((hits,label) for label,hits in scores.items() if hits),reverse=True)
    if ranked:
        hits,label=ranked[0]
        return {"label":label,"confidence":min(.94,.66+.09*hits),"method":"context_rules","rule_hits":hits}
    return {"label":"general customer support","confidence":0.50,"method":"context_rules","rule_hits":0}


def classify_intent(text: str, use_semantic=None, semantic_result: dict | None = None):
    use_semantic = SETTINGS.enable_semantic_ai if use_semantic is None else bool(use_semantic)
    rule=_rules(text)
    ai=None
    if use_semantic:
        try:
            bundle=semantic_result or analyze_utterance(text)
            ai=bundle.get("intent") if bundle else None
        except Exception:
            ai=None
    if ai:
        # Agreement between independent lexical evidence and semantic similarity is a
        # stronger signal than either one alone. A very strong explicit rule may also
        # override a weak semantic guess.
        if rule["rule_hits"] > 0 and rule["label"] == ai["label"]:
            out=dict(ai)
            out["confidence"]=round(min(.98,0.60*float(ai["confidence"])+0.40*float(rule["confidence"])+0.05),4)
            out["method"]="semantic+rule_agreement"
            out["rule_hits"]=rule["rule_hits"]
            return out
        if rule["rule_hits"] >= 2 and float(ai.get("confidence",0)) < 0.68:
            rule["method"]="strong_rules_over_weak_semantic"
            rule["semantic_candidate"]=ai.get("label")
            return rule
        out=dict(ai)
        out["rule_candidate"]=rule["label"] if rule["rule_hits"] else None
        out["rule_hits"]=rule["rule_hits"]
        return out

    if SETTINGS.enable_zero_shot:
        try:
            r=_classifier()(text,candidate_labels=LABELS,multi_label=False)
            return {"label":r["labels"][0],"confidence":round(float(r["scores"][0]),4),"method":"multilingual_nli"}
        except Exception as e:
            rule["ai_error"]=str(e)[:160]
    return rule
