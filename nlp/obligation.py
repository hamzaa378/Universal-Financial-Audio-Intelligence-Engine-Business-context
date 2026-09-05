import re
from datetime import datetime, timedelta
PROMISE=re.compile(r"(?i)\b(?:i\s+(?:will|'ll|can)\s+pay|promise\s+to\s+pay|payment\s+(?:will\s+be|shall\s+be)\s+(?:made|done)|pay\s+by)\b")
DATE_TERMS=re.compile(r"(?i)\b(today|tomorrow|day after tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b")

def detect_obligations(text, call_date=None):
    if not PROMISE.search(text): return []
    date=(DATE_TERMS.search(text).group(0) if DATE_TERMS.search(text) else None)
    return [{"type":"PROMISE_TO_PAY","evidence":PROMISE.search(text).group(0),"due_expression":date,"confidence":0.9 if date else 0.78}]
