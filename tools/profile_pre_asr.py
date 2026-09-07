from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from ingest.audio_loader import load_audio
from analysis_ai.acoustic import analyze_acoustics
from ingest.tamper_detection import detect_tamper


def main():
    ap=argparse.ArgumentParser(description='Profile v4.12 audio load + signal analysis without running ASR.')
    ap.add_argument('audio')
    ap.add_argument('--json',default='reports/pre_asr_profile_v412.json')
    a=ap.parse_args()
    t=time.perf_counter(); audio,sr=load_audio(a.audio); load_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter(); ac=analyze_acoustics(audio,sr); acoustic_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter(); tam=detect_tamper(audio,sr); tamper_ms=(time.perf_counter()-t)*1000
    total=load_ms+acoustic_ms+tamper_ms
    out={
        'audio':str(Path(a.audio).name),'duration_s':len(audio)/float(sr),'sample_rate':sr,
        'audio_load_ms':round(load_ms,3),'acoustic_analysis_ms':round(acoustic_ms,3),
        'tamper_analysis_ms':round(tamper_ms,3),'pre_asr_total_ms':round(total,3),
        'quality':ac['quality'],'stress_markers':ac['stress_markers'],'tamper':tam,
    }
    Path(a.json).parent.mkdir(parents=True,exist_ok=True)
    Path(a.json).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
