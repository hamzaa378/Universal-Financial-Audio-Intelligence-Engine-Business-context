import unittest
from nlp.pii import detect_pii, mask_pii, luhn_valid, public_pii_metadata
from nlp.profanity import detect_profanity, reduce_profanity
from nlp.entities import extract_entities
from nlp.obligation import detect_obligations

class PrivacyNLPTests(unittest.TestCase):
    def test_financial_nlp(self):
        t='I will pay my EMI of Rs 12500 tomorrow. My PAN is ABCDE1234F and rate is 12.5 percent.'
        self.assertTrue(any(x['type']=='PAN' for x in detect_pii(t)))
        self.assertTrue(any(x['type']=='EMI_AMOUNT' for x in extract_entities(t)))
        self.assertTrue(detect_obligations(t))

    def test_card_validator(self):
        self.assertTrue(luhn_valid('4111 1111 1111 1111'))
        self.assertFalse(luhn_valid('4111 1111 1111 1112'))

    def test_context_false_positive_reduction(self):
        self.assertFalse(detect_pii('The order number is 9876543210.'))
        self.assertFalse(detect_pii('Case number 9876543210 was escalated.'))
        self.assertFalse(detect_pii('9876543210'))
        self.assertFalse(detect_pii('Address is required for KYC.'))
        self.assertFalse(detect_pii('The EMI amount is 483921 rupees.'))

    def test_masking(self):
        t='My PAN is ABCDE1234F and mobile number is 9876543210.'
        m=mask_pii(t,detect_pii(t))
        self.assertNotIn('ABCDE1234F',m); self.assertNotIn('9876543210',m)


    def test_public_metadata_has_no_raw_pii(self):
        t='My PAN is ABCDE1234F.'
        meta=public_pii_metadata(detect_pii(t))
        self.assertNotIn('value',meta[0])
        self.assertNotIn('ABCDE1234F',str(meta))
        self.assertNotIn('1234F', meta[0]['masked_value'])

    def test_profanity_and_benign_substrings(self):
        self.assertTrue(detect_profanity('The caller said f**k.'))
        self.assertFalse(detect_profanity('The virtual assistant discussed asset class.'))
        e=detect_profanity('This is shit service')
        self.assertNotIn('shit',reduce_profanity('This is shit service',e).lower())

if __name__=='__main__': unittest.main(verbosity=2)
