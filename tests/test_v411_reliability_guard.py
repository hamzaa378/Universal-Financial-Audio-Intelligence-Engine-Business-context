import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from asr.privacy_recovery import plan_privacy_recovery, recover_privacy_audio_intervals
from nlp.pii import detect_pii, mask_pii
from privacy_debug import write_privacy_debug_bundle


def mock_asr(text: str, duration: float = 5.0) -> dict:
    tokens=text.split()
    step=duration/max(1,len(tokens))
    words=[]
    for i,t in enumerate(tokens):
        words.append({'word':t,'start':round(i*step,3),'end':round((i+0.85)*step,3),'confidence':0.88})
    return {'text':text,'segments':[{'id':0,'start':0.0,'end':duration,'text':text,'confidence':0.88,'words':words}], 'words':words}


class V411ReliabilityGuardTests(unittest.TestCase):
    def test_cvv_quantity_does_not_inherit_context(self):
        text='The CVV on my card is 391, there were 482 transactions yesterday.'
        hits=detect_pii(text,use_semantic=False,use_ner=False)
        self.assertEqual([(x['type'],x['value']) for x in hits],[('CVV','391')])
        self.assertIn('482 transactions',mask_pii(text,hits))

    def test_cvv_quantity_vocab_is_vetoed(self):
        negatives=(
            'The CVV on my card is 391. There were 482 transactions yesterday.',
            'The CVV is 617, and 391 records were processed.',
        )
        for text in negatives:
            vals=[x['value'] for x in detect_pii(text,use_semantic=False,use_ner=False) if x['type']=='CVV']
            self.assertNotIn('482',vals,text)
            if '391 records' in text:
                self.assertNotIn('391',vals,text)

    def test_expected_state_is_one_shot_when_field_already_satisfied(self):
        text='The CVV on my card is 391. Mine is 482.'
        asr=mock_asr(text)
        hits=detect_pii(text,use_semantic=False,use_ner=False)
        # The previous CVV was already satisfied, so no recovery plan is inherited.
        self.assertEqual(plan_privacy_recovery(asr,hits,max_windows=8),[])

    def test_followup_missing_value_still_plans_recovery(self):
        text='The CVV is located on the back of the card. Mine ID, address is available.'
        plans=plan_privacy_recovery(mock_asr(text),[],max_windows=8)
        self.assertEqual(len(plans),1)
        self.assertEqual(plans[0]['type'],'CVV')
        self.assertEqual(plans[0]['source'],'followup_expectation')

    def test_same_sentence_corrupt_name_plans_recovery(self):
        text='My registered name is 2210.'
        plans=plan_privacy_recovery(mock_asr(text),[],max_windows=8)
        self.assertEqual(len(plans),1)
        self.assertEqual(plans[0]['type'],'NAME')
        self.assertEqual(plans[0]['source'],'owned_field')

    def test_same_sentence_corrupt_pan_plans_recovery(self):
        text='My PAN is alpha bravo 12 foxtrot.'
        hits=detect_pii(text,use_semantic=False,use_ner=False)
        self.assertFalse(any(x['type']=='PAN' for x in hits))
        plans=plan_privacy_recovery(mock_asr(text),hits,max_windows=8)
        self.assertTrue(any(x['type']=='PAN' for x in plans))

    def test_cross_punctuation_ifsc_is_fully_masked_and_bounded(self):
        text='My IFSC is Hotel Delta Foxtrot. Charlie 001234 and my address is 15 Rose Road, Pune 411001.'
        hits=detect_pii(text,use_semantic=False,use_ner=False)
        ifsc=[x for x in hits if x['type']=='IFSC']
        self.assertEqual(len(ifsc),1)
        self.assertEqual(ifsc[0]['value'],'Hotel Delta Foxtrot. Charlie 001234')
        safe=mask_pii(text,hits)
        self.assertIn('My IFSC is [IFSC REDACTED] and my address is [ADDRESS REDACTED]',safe)

    def test_raw_fallback_cannot_cross_next_role(self):
        text='My driving license number is Carnitaka0120210012345, product batch KAO120210012345, past inspection.'
        hits=detect_pii(text,use_semantic=False,use_ner=False)
        dl=[x for x in hits if x['type']=='DRIVING_LICENSE']
        self.assertEqual(len(dl),1)
        self.assertEqual(dl[0]['value'],'Carnitaka0120210012345')
        self.assertIn('product batch KAO120210012345',mask_pii(text,hits))

    def test_cvv_assignment_does_not_schedule_card_recovery(self):
        text='The CVV on my card is 391, there were 482 transactions yesterday.'
        hits=detect_pii(text,use_semantic=False,use_ner=False)
        plans=plan_privacy_recovery(mock_asr(text),hits,max_windows=8)
        self.assertFalse(any(x['type']=='CARD' for x in plans))

    def test_recovery_telemetry_counts_recovered_window(self):
        text='My registered name is 2210.'
        asr=mock_asr(text)
        audio=np.zeros(16000*6,dtype=np.float32)
        alt={'text':'Sara Khan','segments':[],'words':[
            {'word':'Sara','start':1.0,'end':1.2,'confidence':0.9},
            {'word':'Khan','start':1.21,'end':1.45,'confidence':0.91},
        ],'error':None}
        with patch('asr.privacy_recovery._decode_window',return_value=alt):
            result=recover_privacy_audio_intervals(audio,16000,asr,[],max_windows=4,conservative_on_failure=True)
        tel=result['telemetry']
        self.assertEqual(tel['windows_planned'],1)
        self.assertEqual(tel['windows_attempted'],1)
        self.assertEqual(tel['recovered'],1)
        self.assertEqual(tel['guarded'],0)
        self.assertGreaterEqual(tel['decode_inference_ms'],0.0)
        self.assertNotIn('\"alternate_text\":',json.dumps(result))

    def test_recovery_telemetry_counts_guard(self):
        text='My registered name is 2210.'
        asr=mock_asr(text)
        audio=np.zeros(16000*6,dtype=np.float32)
        with patch('asr.privacy_recovery._decode_window',return_value={'text':'','segments':[],'words':[],'error':'mock'}):
            result=recover_privacy_audio_intervals(audio,16000,asr,[],max_windows=4,conservative_on_failure=True)
        tel=result['telemetry']
        self.assertEqual(tel['guarded'],1)
        self.assertEqual(tel['recovered'],0)
        self.assertEqual(len(result['intervals']),1)

    def test_debug_bundle_is_opt_in_writer_and_contains_comparison(self):
        with tempfile.TemporaryDirectory() as td:
            info=write_privacy_debug_bundle('My CVV is 391.','My CVV is [CVV REDACTED].',[],{'telemetry':{}},base_dir=td)
            self.assertTrue(os.path.exists(info['raw_transcript']))
            self.assertTrue(os.path.exists(info['safe_transcript']))
            
            with open(info['comparison'],'r',encoding='utf-8') as fh:
                comparison=json.load(fh)
            self.assertIn('RAW DEBUG ARTIFACT',comparison['warning'])
            self.assertEqual(comparison['raw_length'],len('My CVV is 391.'))


if __name__=='__main__':
    unittest.main(verbosity=2)
