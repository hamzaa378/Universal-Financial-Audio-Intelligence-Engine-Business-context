import unittest

from nlp.pii import detect_pii, mask_pii
from nlp.token_alignment import attach_entity_tokens
from nlp.corrections import resolve_timed_corrections
from pipeline import _correction_candidate_validator, _render_safe_text


def mock_asr(text: str, pause_after: str | None=None, pause_s: float=0.65) -> dict:
    toks=text.split()
    words=[]; t=0.0
    for tok in toks:
        st=t; en=t+0.22
        words.append({'word':tok,'start':st,'end':en,'confidence':0.95})
        t=en+(pause_s if pause_after and pause_after in tok else 0.05)
    return {'text':text,'segments':[{'id':0,'start':0.0,'end':t,'text':text,'confidence':0.95,'words':words}], 'words':words}


class V50CorrectionAwareTests(unittest.TestCase):
    def _resolve(self,text,pause_after,pause_s=0.65):
        asr=mock_asr(text,pause_after,pause_s)
        entities=attach_entity_tokens(asr,detect_pii(text))
        return resolve_timed_corrections(text,entities,asr,candidate_validator=_correction_candidate_validator)

    def test_keyword_free_phone_restart_masks_only_final_value(self):
        text='My phone number is 9876543210 9123456789.'
        kept,audit=self._resolve(text,'9876543210',0.65)
        phones=[x for x in kept if x['type']=='PHONE']
        self.assertEqual([x['value'] for x in phones],['9123456789'])
        self.assertEqual(audit[0]['method'],'silent_restart')
        self.assertFalse(audit[0]['wrong_value_masked'])
        safe=_render_safe_text(text,{'pii':kept,'financial_ids':[],'profanity':[]})
        self.assertIn('9876543210',safe)
        self.assertIn('[PHONE REDACTED]',safe)
        self.assertNotIn('9123456789',safe)

    def test_short_pause_does_not_infer_correction(self):
        text='My phone number is 9876543210 9123456789.'
        kept,audit=self._resolve(text,'9876543210',0.10)
        self.assertEqual(len([x for x in kept if x['type']=='PHONE']),2)
        self.assertEqual(audit,[])

    def test_and_join_is_a_list_not_a_correction(self):
        text='My phone number is 9876543210 and 9123456789.'
        kept,audit=self._resolve(text,'9876543210',0.75)
        self.assertEqual(len([x for x in kept if x['type']=='PHONE']),2)
        self.assertEqual(audit,[])

    def test_primary_and_alternate_are_not_corrections(self):
        text='My primary phone number is 9876543210 and my alternate phone number is 9123456789.'
        kept,audit=self._resolve(text,'9876543210',0.75)
        self.assertEqual(len([x for x in kept if x['type']=='PHONE']),2)
        self.assertEqual(audit,[])

    def test_otp_after_asr_punctuation_can_be_rediscovered(self):
        text='My OTP is 638291... 638292.'
        kept,audit=self._resolve(text,'638291',0.70)
        otps=[x for x in kept if x['type']=='OTP']
        self.assertEqual([x['value'] for x in otps],['638292'])
        self.assertTrue(audit and audit[0]['method'].endswith('_rediscovered'))

    def test_name_restart_uses_timing_and_masks_only_final_name(self):
        text='My name is Sara Khan, Aisha Noor.'
        # Text-only is intentionally conservative: both remain masked until timing.
        initial=detect_pii(text)
        self.assertEqual(len([x for x in initial if x['type']=='NAME']),2)
        kept,audit=self._resolve(text,'Khan,',0.75)
        names=[x for x in kept if x['type']=='NAME']
        self.assertEqual([x['value'] for x in names],['Aisha Noor'])
        safe=_render_safe_text(text,{'pii':kept,'financial_ids':[],'profanity':[]})
        self.assertIn('Sara Khan',safe)
        self.assertIn('[NAME REDACTED]',safe)
        self.assertNotIn('Aisha Noor',safe)

    def test_name_components_are_owned_pii(self):
        for text,value in [
            ('My first name is Sara.','Sara'),
            ('My surname is Khan.','Khan'),
            ('My maiden name is Ali.','Ali'),
        ]:
            hits=detect_pii(text)
            self.assertTrue(any(x['type']=='NAME' and x['value']==value for x in hits))

    def test_text_only_comma_pair_stays_conservative(self):
        text='My account number is 001234567890, 001234567891.'
        hits=detect_pii(text)
        # Without timing, a comma can be a list. Do not expose either value.
        self.assertEqual(len([x for x in hits if x['type']=='ACCOUNT_NUMBER']),2)
        safe=mask_pii(text,hits)
        self.assertNotIn('001234567890',safe)
        self.assertNotIn('001234567891',safe)


if __name__=='__main__':
    unittest.main(verbosity=2)
