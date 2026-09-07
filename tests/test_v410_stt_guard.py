import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_privacy import create_protected_audio
from asr.privacy_recovery import plan_privacy_recovery, recover_privacy_audio_intervals, _confirm_expected
from nlp.pii import detect_pii, mask_pii


def mock_asr(text: str, duration: float = 4.0) -> dict:
    tokens=text.split()
    step=duration/max(1,len(tokens))
    words=[]
    for i,t in enumerate(tokens):
        words.append({'word':t,'start':round(i*step,3),'end':round((i+0.85)*step,3),'confidence':0.86})
    return {'text':text,'segments':[{'id':0,'start':0.0,'end':duration,'text':text,'confidence':0.86,'words':words}], 'words':words}


class V410STTGuardTests(unittest.TestCase):
    def test_spoken_as_and_training_document_are_vetoed(self):
        negatives=(
            'The training document says that an email can be spoken as name dot surname at domain dot com.',
            'The tutorial says an email may be written as sara dot khan at example dot com.',
            'The manual says the format can be represented as user dot name at domain dot com.',
        )
        for text in negatives:
            self.assertEqual(detect_pii(text),[],text)
        positive='My email is sara dot khan at example dot com.'
        self.assertEqual([x['type'] for x in detect_pii(positive)],['EMAIL'])

    def test_raw_fallback_owns_complete_value_but_stops_at_next_role(self):
        text='My driving license number is Carnitaka0120210012345, product batch KAO120210012345, past inspection.'
        hits=detect_pii(text)
        self.assertEqual(len(hits),1)
        self.assertEqual(hits[0]['type'],'DRIVING_LICENSE')
        self.assertEqual(hits[0]['value'],'Carnitaka0120210012345')
        safe=mask_pii(text,hits)
        self.assertIn('[DRIVING_LICENSE REDACTED], product batch KAO120210012345',safe)

    def test_raw_fallback_does_not_swallow_trailing_prose(self):
        text='My driving license number is Carnitaka0120210012345, past inspection.'
        hit=detect_pii(text)[0]
        self.assertEqual(hit['value'],'Carnitaka0120210012345')
        self.assertIn(', past inspection.',mask_pii(text,[hit]))

    def test_recovery_plan_for_asr_deleted_cvv(self):
        text='The CVV is located on the back of the card. Mine ID, address is available.'
        asr=mock_asr(text)
        plans=plan_privacy_recovery(asr,[])
        self.assertEqual(len(plans),1)
        self.assertEqual(plans[0]['type'],'CVV')
        self.assertGreater(plans[0]['time_end'],plans[0]['time_start'])

    def test_no_recovery_when_primary_already_found_value(self):
        text='Please enter your Aadhaar number. Mine is 4567 8901 2345.'
        asr=mock_asr(text)
        pii=detect_pii(text)
        self.assertTrue(any(x['type']=='AADHAAR' for x in pii))
        self.assertEqual(plan_privacy_recovery(asr,pii),[])

    def test_alternate_decode_can_confirm_spoken_cvv(self):
        alt={
            'text':'six one seven',
            'words':[
                {'word':'six','start':1.0,'end':1.2,'confidence':0.9},
                {'word':'one','start':1.22,'end':1.4,'confidence':0.91},
                {'word':'seven','start':1.42,'end':1.7,'confidence':0.92},
            ]
        }
        hit=_confirm_expected('CVV',alt)
        self.assertIsNotNone(hit)
        self.assertEqual(hit['type'],'CVV')
        self.assertEqual(hit['alignment_method'],'privacy_targeted_redecode')
        self.assertLess(hit['start'],1.01)
        self.assertGreater(hit['end'],1.69)

    def test_failed_second_decode_creates_conservative_audio_guard(self):
        text='The CVV is located on the back of the card. Mine ID, address is available.'
        asr=mock_asr(text)
        audio=np.zeros(16000*5,dtype=np.float32)
        with patch('asr.privacy_recovery._decode_window',return_value={'text':'','segments':[],'words':[],'error':'mock'}):
            result=recover_privacy_audio_intervals(audio,16000,asr,[],max_windows=2,conservative_on_failure=True)
        self.assertEqual(result['plans'],1)
        self.assertEqual(len(result['intervals']),1)
        self.assertEqual(result['intervals'][0]['type'],'CVV')
        self.assertEqual(result['intervals'][0]['recovery_status'],'conservative_audio_guard')

    def test_audio_only_recovery_interval_is_applied(self):
        fd,input_path=tempfile.mkstemp(suffix='.wav'); os.close(fd)
        try:
            sr=16000
            sf.write(input_path,np.zeros(sr*2,dtype=np.float32),sr)
            asr={'text':'No visible value.','segments':[{'start':0.0,'end':2.0,'text':'No visible value.','words':[]}]}
            result=create_protected_audio(
                input_path,asr,[],[],method='mute',
                extra_intervals=[{'type':'CVV','start':0.5,'end':1.0,'confidence':0.72,'alignment_method':'expected_field_conservative_window','alignment_confidence':0.5}],
            )
            self.assertEqual(result['interval_count'],1)
            self.assertIn('CVV',result['intervals'][0]['types'])
        finally:
            try: os.unlink(input_path)
            except FileNotFoundError: pass
            if 'result' in locals():
                try: os.unlink(result['path'])
                except FileNotFoundError: pass


if __name__=='__main__':
    unittest.main(verbosity=2)
