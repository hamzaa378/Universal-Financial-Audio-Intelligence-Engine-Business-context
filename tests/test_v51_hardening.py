import os
import tempfile
import time
import unittest
from pathlib import Path

from nlp.pii import detect_pii
from nlp.token_alignment import attach_entity_tokens
from nlp.corrections import resolve_timed_corrections
from audio_privacy import spans_to_audio_intervals
from privacy_debug import cleanup_privacy_debug_dir


def asr_one_segment(text: str, confidence: float = 0.95, speaker=None, pause_after=None, pause_s=0.65):
    words=[]; t=0.0
    for tok in text.split():
        st=t; en=t+0.20
        words.append({'word':tok,'start':st,'end':en,'confidence':confidence})
        t=en+(pause_s if pause_after and pause_after in tok else 0.05)
    seg={'id':0,'start':0.0,'end':t,'text':text,'confidence':confidence,'words':words}
    if speaker is not None: seg['speaker']=speaker
    return {'text':text,'segments':[seg],'words':words}


def asr_two_speakers(seg1: str, seg2: str, confidence: float = 0.95):
    segs=[]; cursor_t=0.0
    for sid,(txt,spk) in enumerate(((seg1,'SPEAKER_00'),(seg2,'SPEAKER_01'))):
        words=[]
        for tok in txt.split():
            st=cursor_t; en=st+0.20
            words.append({'word':tok,'start':st,'end':en,'confidence':confidence})
            cursor_t=en+0.05
        segs.append({'id':sid,'start':words[0]['start'] if words else cursor_t,'end':words[-1]['end'] if words else cursor_t,
                     'text':txt,'confidence':confidence,'words':words,'speaker':spk})
        cursor_t+=0.70
    text=seg1+' '+seg2
    return {'text':text,'segments':segs,'words':[w for s in segs for w in s['words']]}


class V51HardeningTests(unittest.TestCase):
    def test_low_alignment_never_unmasks_superseded_value(self):
        text='My phone number is 9876543210 9123456789.'
        asr=asr_one_segment(text,confidence=0.40,pause_after='9876543210',pause_s=0.70)
        entities=attach_entity_tokens(asr,detect_pii(text))
        kept,audit=resolve_timed_corrections(text,entities,asr)
        self.assertEqual(len([x for x in kept if x['type']=='PHONE']),2)
        self.assertEqual(audit,[])

    def test_cross_speaker_values_are_not_collapsed_as_self_correction(self):
        seg1='My phone number is 9876543210'
        seg2='9123456789.'
        text=seg1+' '+seg2
        asr=asr_two_speakers(seg1,seg2)
        entities=attach_entity_tokens(asr,detect_pii(text))
        # The second value can be weak in ordinary text context; copy normal phone ownership
        # only for this resolver-safety unit test, not through the production detector.
        phones=[x for x in entities if x['type']=='PHONE']
        if len(phones)==1:
            b=dict(phones[0]); b['start']=text.rfind('9123456789'); b['end']=b['start']+10; b['value']='9123456789'; b['ownership_strength']=4
            b=attach_entity_tokens(asr,[b])[0]; entities.append(b)
        kept,audit=resolve_timed_corrections(text,entities,asr)
        self.assertEqual(len([x for x in kept if x['type']=='PHONE']),2)
        self.assertEqual(audit,[])

    def test_reused_owned_value_is_not_exposed_by_local_correction(self):
        text='My backup phone number is 9876543210. My phone number is 9876543210 9123456789.'
        asr=asr_one_segment(text,confidence=0.95,pause_after='9876543210',pause_s=0.70)
        # Build explicit entities so the test targets correction policy, independent of detector wording.
        starts=[]; pos=0
        while True:
            i=text.find('9876543210',pos)
            if i<0: break
            starts.append(i); pos=i+1
        j=text.find('9123456789')
        entities=[
            {'type':'PHONE','start':starts[0],'end':starts[0]+10,'value':'9876543210','confidence':0.99,'ownership_strength':4},
            {'type':'PHONE','start':starts[1],'end':starts[1]+10,'value':'9876543210','confidence':0.99,'ownership_strength':4},
            {'type':'PHONE','start':j,'end':j+10,'value':'9123456789','confidence':0.99,'ownership_strength':4},
        ]
        entities=attach_entity_tokens(asr,entities)
        kept,audit=resolve_timed_corrections(text,entities,asr)
        self.assertEqual(len([x for x in kept if x['type']=='PHONE']),3)
        self.assertEqual(audit,[])

    def test_high_quality_same_speaker_restart_still_resolves(self):
        text='My phone number is 9876543210 9123456789.'
        asr=asr_one_segment(text,confidence=0.95,pause_after='9876543210',pause_s=0.70)
        entities=attach_entity_tokens(asr,detect_pii(text))
        kept,audit=resolve_timed_corrections(text,entities,asr)
        self.assertEqual([x['value'] for x in kept if x['type']=='PHONE'],['9123456789'])
        self.assertTrue(audit)

    def test_explicit_new_secret_labels_are_masked(self):
        cases=[
            ('My client secret is AbCDef1234567890_$xy.','API_KEY'),
            ('My refresh token is AbCDef1234567890.tokenXYZ.','AUTH_TOKEN'),
            ('My session token is AbCDef1234567890.tokenXYZ.','AUTH_TOKEN'),
            ('My bearer token is AbCDef1234567890.tokenXYZ.','AUTH_TOKEN'),
            ('My signing key is AbCDef1234567890_$xy.','API_KEY'),
        ]
        for text,kind in cases:
            hits=detect_pii(text)
            self.assertTrue(any(x['type']==kind for x in hits),(text,hits))

    def test_secret_examples_remain_visible(self):
        cases=[
            'The documentation example says client secret is YOUR_API_KEY_HERE.',
            'The sample refresh token is example_token.',
            'A bearer token may look like a long random string.',
        ]
        for text in cases:
            self.assertFalse(any(x['type'] in {'API_KEY','AUTH_TOKEN'} for x in detect_pii(text)),text)

    def test_accepted_pii_gets_audio_guard_when_char_alignment_is_outside_segments(self):
        asr={'text':'short transcript','segments':[{'id':0,'start':1.0,'end':2.0,'text':'short transcript','confidence':0.9,'words':[]} ]}
        pii=[{'type':'PHONE','start':500,'end':510,'value':'9876543210','confidence':0.99,'ownership_strength':4,'token_ids':[], 'alignment_method':'unresolved','alignment_confidence':0.0}]
        intervals=spans_to_audio_intervals(asr,pii,[])
        self.assertTrue(intervals)
        self.assertTrue(any(x.get('alignment_method')=='accepted_pii_nearest_segment_guard' for x in intervals))

    def test_debug_ttl_cleanup_removes_only_stale_run_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            old=root/'run_old'; new=root/'run_new'; other=root/'keep_me'
            old.mkdir(); new.mkdir(); other.mkdir()
            stale=time.time()-48*3600
            os.utime(old,(stale,stale))
            result=cleanup_privacy_debug_dir(td,ttl_hours=24)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())
            self.assertTrue(other.exists())
            self.assertEqual(result['deleted'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
