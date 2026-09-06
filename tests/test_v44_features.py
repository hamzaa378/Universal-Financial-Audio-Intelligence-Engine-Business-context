import unittest

from nlp.pii import detect_pii
from nlp.sensitive_financial import detect_sensitive_financial_ids
from nlp.profanity import detect_profanity
from nlp.entities import extract_entities
from nlp.obligation import detect_obligations
from nlp.regulatory import check_regulatory
from nlp.privacy import protect_text


class V44FeatureTests(unittest.TestCase):
    def test_context_gated_identity_documents(self):
        self.assertTrue(any(x['type']=='PASSPORT' for x in detect_pii('My passport number is A1234567.')))
        self.assertFalse(detect_pii('Reference A1234567 was generated.'))
        self.assertTrue(any(x['type']=='VOTER_ID' for x in detect_pii('My voter ID is ABC1234567.')))
        self.assertFalse(detect_pii('Product code ABC1234567 is discontinued.'))
        self.assertTrue(any(x['type']=='DRIVING_LICENSE' for x in detect_pii('Driving license number KA01 2020 1234567.')))

    def test_sensitive_financial_ids_are_separate(self):
        t='Transaction reference TXN-88991122 and customer ID CUST-552211.'
        ids=detect_sensitive_financial_ids(t)
        self.assertEqual({x['type'] for x in ids},{'TRANSACTION_ID','CUSTOMER_ID'})
        self.assertFalse(any(x['type']=='PHONE' for x in detect_pii(t)))
        safe=protect_text(t,financial_ids_enabled=True)['text']
        self.assertNotIn('TXN-88991122',safe)
        self.assertNotIn('CUST-552211',safe)

    def test_profanity_variants_and_benign_words(self):
        self.assertTrue(detect_profanity('He shouted gaandu and disconnected.'))
        self.assertTrue(detect_profanity('This is bakchodi.'))
        self.assertFalse(detect_profanity('The assistant reviewed the asset classification.'))

    def test_financial_entity_layer(self):
        t='My EMI is Rs 12500, outstanding amount is INR 42000, interest rate 12.5 percent and tenure 24 months via UPI.'
        kinds={x['type'] for x in extract_entities(t)}
        self.assertIn('EMI_AMOUNT',kinds)
        self.assertIn('OUTSTANDING_AMOUNT',kinds)
        self.assertIn('INTEREST_RATE',kinds)
        self.assertIn('TENURE',kinds)
        self.assertIn('PAYMENT_METHOD',kinds)

    def test_promise_vs_difficulty(self):
        p=detect_obligations('I will pay Rs 5000 tomorrow.')
        self.assertTrue(any(x['type']=='PROMISE_TO_PAY' for x in p))
        d=detect_obligations('I cannot pay today because of financial hardship.')
        self.assertFalse(any(x['type']=='PROMISE_TO_PAY' for x in d))
        self.assertTrue(any(x['type']=='REPAYMENT_DIFFICULTY' for x in d))

    def test_regulatory_threat_screen(self):
        r=check_regulatory('You will be arrested if you do not pay today.')
        self.assertTrue(any(x['type']=='coercive_arrest_threat' for x in r['risk_phrases']))
        self.assertEqual(r['risk_level'],'high')

if __name__=='__main__':
    unittest.main(verbosity=2)
