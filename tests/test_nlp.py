from nlp.pii import detect_pii
from nlp.entities import extract_entities
from nlp.obligation import detect_obligations

def test_financial_nlp():
    t="I will pay my EMI of Rs 12500 tomorrow. My PAN is ABCDE1234F and rate is 12.5 percent."
    assert any(x["type"]=="PAN" for x in detect_pii(t))
    assert any(x["type"]=="EMI_AMOUNT" for x in extract_entities(t))
    assert detect_obligations(t)
