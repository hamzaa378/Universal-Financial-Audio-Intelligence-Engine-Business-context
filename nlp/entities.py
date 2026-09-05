import re
AMOUNT=re.compile(r"(?i)(?:₹|rs\.?|inr|rupees?)\s*([0-9][0-9,]*(?:\.\d{1,2})?)|\b([0-9][0-9,]*(?:\.\d{1,2})?)\s*(?:rupees?|inr)\b")
RATE=re.compile(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*(?:%|percent|per cent)\b",re.I)
EMI=re.compile(r"(?i)\b(?:emi|installment|instalment)\b.{0,30}?(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.\d{1,2})?)")
ACCOUNT=re.compile(r"(?i)\b(?:loan|account)\s*(?:number|no\.?|id)?\s*[:#-]?\s*([A-Z0-9-]{5,24})")

def extract_entities(text):
    out=[]
    def add(t,m,val,conf): out.append({"type":t,"value":val,"start":m.start(),"end":m.end(),"confidence":conf})
    for m in AMOUNT.finditer(text): add("MONEY",m,m.group(1) or m.group(2),0.94)
    for m in RATE.finditer(text): add("INTEREST_RATE",m,m.group(1)+"%",0.95)
    for m in EMI.finditer(text): add("EMI_AMOUNT",m,m.group(1),0.9)
    for m in ACCOUNT.finditer(text): add("FINANCIAL_ACCOUNT_REF",m,m.group(1),0.82)
    return sorted(out,key=lambda x:x["start"])
