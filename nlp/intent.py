from functools import lru_cache
from config import SETTINGS
LABELS=["payment commitment","payment completed","loan enquiry","repayment difficulty","grievance or complaint","identity verification","fraud report","foreclosure or prepayment","settlement negotiation","general customer support"]
KEYWORDS={
 "payment commitment":["will pay","i'll pay","pay tomorrow","pay by","promise to pay"],
 "payment completed":["already paid","payment done","i paid","transferred"],
 "repayment difficulty":["cannot pay","can't pay","unable to pay","financial difficulty","lost my job"],
 "grievance or complaint":["complaint","grievance","unfair","harassment","ombudsman"],
 "fraud report":["fraud","unauthorized","not my transaction","scam"],
 "foreclosure or prepayment":["foreclose","prepay","prepayment","close the loan"],
 "settlement negotiation":["settlement","waive","waiver","negotiate"],
 "identity verification":["verify","kyc","date of birth","pan","aadhaar"],
 "loan enquiry":["loan","interest rate","eligibility","apply"],
}
@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline
    return pipeline("zero-shot-classification",model=SETTINGS.nli_model)

def classify_intent(text):
    if SETTINGS.enable_zero_shot:
        try:
            r=_classifier()(text,candidate_labels=LABELS,multi_label=False)
            return {"label":r["labels"][0],"confidence":round(float(r["scores"][0]),4),"method":"multilingual_nli"}
        except Exception as e:
            fallback=_rules(text); fallback["ai_error"]=str(e)[:160]; return fallback
    return _rules(text)

def _rules(text):
    t=text.lower(); scores=[]
    for label,kws in KEYWORDS.items():
        hits=sum(1 for k in kws if k in t)
        if hits:scores.append((hits,label))
    if scores:
        hits,label=max(scores); return {"label":label,"confidence":min(.92,.68+.08*hits),"method":"context_rules"}
    return {"label":"general customer support","confidence":0.5,"method":"context_rules"}
