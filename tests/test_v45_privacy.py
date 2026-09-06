import unittest

from nlp.pii import detect_pii
from nlp.privacy import protect_text
from nlp.token_alignment import attach_entity_tokens


class V45PrivacyTests(unittest.TestCase):
    def test_ifsc_separator_normalization(self):
        for raw in ('ABCD-0123456','ABCD 0123456','ABCD0123456'):
            t=f'My IFSC code is {raw}.'
            hits=detect_pii(t)
            self.assertTrue(any(x['type']=='IFSC' and x['value']==raw for x in hits),raw)
            self.assertNotIn(raw,protect_text(t)['text'])

    def test_ifsc_separator_false_positive_guard(self):
        self.assertFalse(any(x['type']=='IFSC' for x in detect_pii('Reference ABCD-0123456 was generated.')))

    def test_name_boundary_does_not_overcapture(self):
        t='My name is John Doe and my account number is 001234567890.'
        hits=detect_pii(t)
        name=next(x for x in hits if x['type']=='NAME')
        self.assertEqual(name['value'],'John Doe')

    def test_address_boundary_stops_before_next_field(self):
        t='My current address is Flat 12, MG Road, Bengaluru 560001 and my phone is 9876543210.'
        hits=detect_pii(t)
        addr=next(x for x in hits if x['type']=='ADDRESS')
        self.assertEqual(addr['value'],'Flat 12, MG Road, Bengaluru 560001')
        self.assertTrue(any(x['type']=='PHONE' for x in hits))



    def test_spoken_digits_with_asr_punctuation(self):
        t='My mobile number is nine, eight, seven, six, five, four, three, two, one, zero.'
        self.assertTrue(any(x['type']=='PHONE' for x in detect_pii(t)))

    def test_spoken_ifsc(self):
        t='My IFSC is ABCD zero one two three four five six.'
        hits=detect_pii(t)
        self.assertTrue(any(x['type']=='IFSC' and x['value']=='ABCD zero one two three four five six' for x in hits))
        self.assertFalse(any(x['type']=='IFSC' for x in detect_pii('Reference ABCD zero one two three four five six was generated.')))

    def test_multilevel_spoken_email(self):
        t='My email is hamza dot ahmad at bank dot co dot in.'
        hits=detect_pii(t)
        self.assertTrue(any(x['type']=='EMAIL' and 'bank dot co dot in' in x['value'] for x in hits))

    def test_natural_language_dob(self):
        t='My date of birth is 12 March 1998.'
        self.assertTrue(any(x['type']=='DOB' and x['value']=='12 March 1998' for x in detect_pii(t)))
        self.assertFalse(any(x['type']=='DOB' for x in detect_pii('The meeting is on 12 March 1998.')))

    def test_token_ownership_alignment(self):
        text='My IFSC code is ABCD-0123456.'
        asr={
            'text':text,
            'segments':[{
                'id':0,'start':0.0,'end':2.5,'text':text,'confidence':0.95,
                'words':[
                    {'word':'My','start':0.0,'end':0.2,'confidence':0.99},
                    {'word':'IFSC','start':0.2,'end':0.55,'confidence':0.99},
                    {'word':'code','start':0.55,'end':0.8,'confidence':0.98},
                    {'word':'is','start':0.8,'end':0.95,'confidence':0.98},
                    {'word':'ABCD-0123456','start':1.0,'end':2.0,'confidence':0.94},
                ],
            }]
        }
        hits=attach_entity_tokens(asr,detect_pii(text))
        ifsc=next(x for x in hits if x['type']=='IFSC')
        self.assertTrue(ifsc['token_ids'])
        self.assertEqual(ifsc['alignment_method'],'whisper_word_tokens')
        self.assertAlmostEqual(ifsc['time_start'],1.0)
        self.assertAlmostEqual(ifsc['time_end'],2.0)

if __name__=='__main__':
    unittest.main(verbosity=2)
