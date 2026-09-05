REQUIRED_DISCLOSURES=[
 {"id":"recording_disclosure","phrases":["call is being recorded","call may be recorded"],"severity":"medium"},
]
RISK_PHRASES={
 "coercive_threat":["you will be arrested","police will arrest","we will shame you","tell your employer"],
 "abusive_collection":["idiot","stupid","useless"],
}
def check_regulatory(text):
    t=text.lower(); missing=[]; detected=[]
    for r in REQUIRED_DISCLOSURES:
        if not any(p in t for p in r["phrases"]): missing.append({"rule":r["id"],"severity":r["severity"],"confidence":0.85})
    for typ,phrases in RISK_PHRASES.items():
        for p in phrases:
            if p in t: detected.append({"type":typ,"evidence":p,"confidence":0.92})
    return {"missing_disclosures":missing,"risk_phrases":detected,"note":"Policy screening must be configured to the lender's jurisdiction and approved compliance policy."}
