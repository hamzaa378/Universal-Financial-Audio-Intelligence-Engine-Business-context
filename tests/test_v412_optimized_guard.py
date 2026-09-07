import os
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from analysis_ai.acoustic import analyze_acoustics
from asr.privacy_recovery import plan_privacy_recovery, recover_privacy_audio_intervals
from ingest.audio_loader import load_audio
from nlp.pii import detect_pii, mask_pii
from pipeline import _audio_guard_replacements, _render_safe_text


def mock_asr(text: str, duration: float = 5.0) -> dict:
    tokens=text.split()
    step=duration/max(1,len(tokens))
    words=[]
    cursor=0
    for i,t in enumerate(tokens):
        idx=text.find(t,cursor)
        if idx<0: idx=cursor
        words.append({'word':t,'start':round(i*step,3),'end':round((i+0.85)*step,3),'confidence':0.88})
        cursor=max(cursor,idx+len(t))
    return {'text':text,'segments':[{'id':0,'start':0.0,'end':duration,'text':text,'confidence':0.88,'words':words}], 'words':words}


class V412OptimizedGuardTests(unittest.TestCase):
    def test_guard_marker_is_safe_and_natural_for_missing_cvv(self):
        text='The CVV is located on the back of the card. Mine ID, address is 15 Rose Road, Pune 411001.'
        plans=plan_privacy_recovery(mock_asr(text),[],max_windows=4)
        self.assertEqual(len(plans),1)
        p=plans[0]
        recovery={'audit':[{
            'type':'CVV','status':'conservative_audio_guard','source':'followup_expectation',
            'char_start':p['char_start'],'char_end':p['char_end']
        }]}
        guards=_audio_guard_replacements(text,recovery)
        safe=_render_safe_text(text,{'pii':[],'financial_ids':[],'profanity':[]},audio_guards=guards)
        self.assertIn('Mine is [CVV AUDIO PROTECTED].',safe)
        self.assertIn('address is 15 Rose Road',safe)
        self.assertNotIn('Mine ID',safe)

    def test_documentation_clause_is_hard_raw_ifsc_boundary(self):
        text=('My bank IFSC is Hotel Delta Foxtrott ID Charlie 00001234, and the software manual uses '
              'HDFC 0001234 as an example IFSC.')
        hits=detect_pii(text,use_semantic=False,use_ner=False)
        ifsc=[x for x in hits if x['type']=='IFSC']
        self.assertTrue(ifsc)
        self.assertTrue(all('software manual' not in x['value'].lower() for x in ifsc))
        safe=mask_pii(text,hits)
        self.assertIn('software manual uses HDFC 0001234 as an example IFSC',safe)

    def test_recovery_telemetry_has_per_type_counts(self):
        text='My registered name is 2210.'
        asr=mock_asr(text)
        audio=np.zeros(16000*6,dtype=np.float32)
        with patch('asr.privacy_recovery._decode_window',return_value={'text':'','segments':[],'words':[],'error':'mock'}):
            result=recover_privacy_audio_intervals(audio,16000,asr,[],max_windows=4,conservative_on_failure=True)
        by=result['telemetry']['by_type']['NAME']
        self.assertEqual(by['planned'],1)
        self.assertEqual(by['attempted'],1)
        self.assertEqual(by['guarded'],1)
        self.assertEqual(by['recovered'],0)

    def test_fast_acoustic_path_keeps_public_schema(self):
        sr=16000
        t=np.arange(sr*8,dtype=np.float32)/sr
        x=(0.05*np.sin(2*np.pi*220*t)).astype(np.float32)
        out=analyze_acoustics(x,sr)
        self.assertIn('score',out['quality'])
        self.assertIn('stress_marker_score',out['stress_markers'])
        self.assertEqual(out['stress_markers']['method'],'shared_acoustic_features_fast_numpy')

    def test_fast_loader_keeps_16khz_float32_contract(self):
        fd,path=tempfile.mkstemp(suffix='.wav'); os.close(fd)
        try:
            sr=48000
            t=np.arange(sr,dtype=np.float32)/sr
            stereo=np.stack([0.03*np.sin(2*np.pi*220*t),0.03*np.sin(2*np.pi*220*t)],axis=1)
            sf.write(path,stereo,sr)
            audio,rate=load_audio(path,16000)
            self.assertEqual(rate,16000)
            self.assertEqual(audio.dtype,np.float32)
            self.assertTrue(15900 <= len(audio) <= 16100)
        finally:
            try: os.unlink(path)
            except FileNotFoundError: pass


if __name__=='__main__':
    unittest.main(verbosity=2)
