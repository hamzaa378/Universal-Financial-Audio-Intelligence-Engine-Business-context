import re
PATTERNS={
 "EMAIL":re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
 "PHONE":re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"),
 "PAN":re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
 "IFSC":re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
 "AADHAAR":re.compile(r"(?<!\d)(?:\d[ -]?){11}\d(?!\d)"),
 "CARD":re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
 "UPI":re.compile(r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z]{2,}\b"),
}
def detect_pii(text):
    found=[]
    for typ,p in PATTERNS.items():
        for m in p.finditer(text):
            val=m.group(0)
            if typ=="CARD":
                d=re.sub(r"\D","",val)
                if not (13<=len(d)<=19): continue
            if typ=="AADHAAR" and len(re.sub(r"\D","",val))!=12: continue
            found.append({"type":typ,"start":m.start(),"end":m.end(),"value":val,"confidence":0.96 if typ in {"EMAIL","PAN","IFSC"} else 0.88})
    return sorted(found,key=lambda x:x["start"])

def mask_pii(text, entities):
    masked=text
    for e in sorted(entities,key=lambda x:x["start"],reverse=True):
        masked=masked[:e["start"]]+"["+e["type"]+" REDACTED]"+masked[e["end"]:]
    return masked
