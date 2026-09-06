import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from audio_privacy import create_protected_audio
from nlp.privacy import protect_text


class AudioPrivacyTests(unittest.TestCase):
    def test_sensitive_words_become_audio_intervals(self):
        text='My mobile number is 9876543210 and this is shit service.'
        words=[
            {'word':'My','start':0.00,'end':0.15,'confidence':0.99},
            {'word':'mobile','start':0.18,'end':0.45,'confidence':0.99},
            {'word':'number','start':0.48,'end':0.72,'confidence':0.99},
            {'word':'is','start':0.74,'end':0.82,'confidence':0.99},
            {'word':'9876543210','start':0.85,'end':1.30,'confidence':0.99},
            {'word':'and','start':1.34,'end':1.47,'confidence':0.99},
            {'word':'this','start':1.50,'end':1.66,'confidence':0.99},
            {'word':'is','start':1.68,'end':1.76,'confidence':0.99},
            {'word':'shit','start':1.80,'end':2.00,'confidence':0.99},
            {'word':'service.','start':2.03,'end':2.35,'confidence':0.99},
        ]
        asr={'text':text,'segments':[{'start':0.0,'end':2.35,'text':text,'words':words}]}
        protected=protect_text(text,privacy_profile='balanced',use_semantic=False,pii_mode='full',mask_types=None,profanity_enabled=True)

        fd,input_path=tempfile.mkstemp(suffix='.wav'); os.close(fd)
        try:
            sr=16000
            sf.write(input_path,np.zeros(int(sr*2.5),dtype=np.float32),sr)
            result=create_protected_audio(input_path,asr,protected['pii'],protected['profanity'],method='beep')
            self.assertEqual(result['interval_count'],2)
            self.assertIn('PHONE',result['intervals'][0]['types'])
            self.assertIn('PROFANITY',result['intervals'][1]['types'])
            self.assertTrue(os.path.exists(result['path']))
        finally:
            try: os.unlink(input_path)
            except FileNotFoundError: pass
            if 'result' in locals():
                try: os.unlink(result['path'])
                except FileNotFoundError: pass


if __name__=='__main__':
    unittest.main(verbosity=2)
