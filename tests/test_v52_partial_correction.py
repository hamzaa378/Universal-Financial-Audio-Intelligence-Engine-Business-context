import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nlp.pii import detect_pii
from nlp.token_alignment import attach_entity_tokens
from nlp.corrections import resolve_partial_corrections
from asr.privacy_recovery import recover_privacy_audio_intervals
from pipeline import _audio_guard_replacements, _render_safe_text
from privacy_debug import write_privacy_debug_bundle


def asr_one_segment(text: str, confidence: float = 0.95, pause_after: str | None = None, pause_s: float = 0.55):
    words=[]; t=0.0
    for tok in text.split():
        st=t; en=t+0.20
        words.append({'word':tok,'start':st,'end':en,'confidence':confidence})
        t=en+(pause_s if pause_after and pause_after in tok else 0.05)
    seg={'id':0,'start':0.0,'end':t,'text':text,'confidence':confidence,'words':words}
    return {'text':text,'segments':[seg],'words':words}


class V52PartialCorrectionTests(unittest.TestCase):
    def test_phone_last_digit_masks_only_spoken_replacement(self):
        text='My phone number is 9876543210. Last digit is 1.'
        asr=asr_one_segment(text,confidence=0.95,pause_after='9876543210',pause_s=0.60)
        entities=attach_entity_tokens(asr,detect_pii(text))
        kept,audit=resolve_partial_corrections(text,entities,asr)
        phones=[x for x in kept if x.get('type')=='PHONE']
        self.assertEqual(len(phones),1)
        self.assertEqual(phones[0]['value'],'1')
        self.assertTrue(phones[0].get('partial_correction'))
        self.assertTrue(audit and audit[0]['wrong_value_masked'] is False)
        self.assertFalse(audit[0]['reconstructed_value_stored'])

    def test_account_last_two_digits_exact_arity(self):
        text='My account number is 1234567890. Change the last two digits to 42.'
        asr=asr_one_segment(text,confidence=0.95,pause_after='1234567890',pause_s=0.60)
        entities=attach_entity_tokens(asr,detect_pii(text))
        kept,audit=resolve_partial_corrections(text,entities,asr)
        accounts=[x for x in kept if x.get('type')=='ACCOUNT_NUMBER']
        self.assertEqual([x['value'] for x in accounts],['42'])
        self.assertEqual(audit[0]['digit_count'],2)

    def test_mismatched_partial_arity_is_rejected(self):
        text='My account number is 1234567890. Change the last two digits to 7.'
        asr=asr_one_segment(text,confidence=0.95,pause_after='1234567890',pause_s=0.60)
        entities=attach_entity_tokens(asr,detect_pii(text))
        kept,audit=resolve_partial_corrections(text,entities,asr)
        self.assertEqual([x['value'] for x in kept if x.get('type')=='ACCOUNT_NUMBER'],['1234567890'])
        self.assertEqual(audit,[])

    def test_example_partial_digit_remains_visible(self):
        text='My phone number is 9876543210. The example says last digit is 1.'
        asr=asr_one_segment(text,confidence=0.95)
        entities=attach_entity_tokens(asr,detect_pii(text))
        kept,audit=resolve_partial_corrections(text,entities,asr)
        self.assertEqual([x['value'] for x in kept if x.get('type')=='PHONE'],['9876543210'])
        self.assertEqual(audit,[])

    def test_low_alignment_keeps_old_and_masks_partial_fragment(self):
        text='My phone number is 9876543210. Last digit is 1.'
        asr=asr_one_segment(text,confidence=0.40,pause_after='9876543210',pause_s=0.60)
        entities=attach_entity_tokens(asr,detect_pii(text))
        kept,audit=resolve_partial_corrections(text,entities,asr)
        vals=[x['value'] for x in kept if x.get('type')=='PHONE']
        self.assertEqual(vals,['9876543210','1'])
        self.assertTrue(audit[0]['wrong_value_masked'])

    def test_guarded_otp_full_correction_gets_second_audio_marker(self):
        text='My OTP is blah. I made a mistake. 124981.'
        asr=asr_one_segment(text,confidence=0.95)
        audio=np.zeros(16000*8,dtype=np.float32)
        with patch('asr.privacy_recovery._decode_window',return_value={'text':'','segments':[],'words':[],'error':None}):
            rr=recover_privacy_audio_intervals(audio,16000,asr,[],max_windows=4,conservative_on_failure=True)
        self.assertEqual(rr['telemetry']['direct_correction_guards'],1)
        self.assertEqual(rr['telemetry']['guarded'],2)
        guards=_audio_guard_replacements(text,rr)
        safe=_render_safe_text(text,{'pii':[],'financial_ids':[],'profanity':[]},audio_guards=guards)
        self.assertEqual(safe,'My OTP is [OTP AUDIO PROTECTED] I made a mistake. [OTP AUDIO PROTECTED].')

    def test_guarded_phone_partial_edit_protects_only_edit_fragment(self):
        text='My phone number is garbled. Last digit is 1.'
        asr=asr_one_segment(text,confidence=0.95)
        audio=np.zeros(16000*8,dtype=np.float32)
        with patch('asr.privacy_recovery._decode_window',return_value={'text':'','segments':[],'words':[],'error':None}):
            rr=recover_privacy_audio_intervals(audio,16000,asr,[],max_windows=4,conservative_on_failure=True)
        self.assertEqual(rr['telemetry']['direct_correction_guards'],1)
        guards=_audio_guard_replacements(text,rr)
        safe=_render_safe_text(text,{'pii':[],'financial_ids':[],'profanity':[]},audio_guards=guards)
        self.assertIn('My phone number is [PHONE AUDIO PROTECTED]',safe)
        self.assertIn('Last digit is [PHONE AUDIO PROTECTED].',safe)

    def test_normal_pipeline_debug_mode_is_span_only_when_raw_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            info=write_privacy_debug_bundle(
                'My phone is 9876543210','My phone is [PHONE REDACTED]',[],{},base_dir=td,include_raw=False
            )
            self.assertFalse(info['raw_enabled'])
            self.assertIsNone(info['raw_transcript'])
            self.assertFalse(any(Path(td).rglob('raw_asr_transcript.txt')))
            self.assertTrue(Path(info['safe_transcript']).exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
