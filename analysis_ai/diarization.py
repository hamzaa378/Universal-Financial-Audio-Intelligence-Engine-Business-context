from __future__ import annotations

from functools import lru_cache
from config import SETTINGS


@lru_cache(maxsize=1)
def _load_pipeline():
    if not SETTINGS.hf_token:
        raise RuntimeError("HF_TOKEN is required after accepting the pyannote model conditions.")
    from pyannote.audio import Pipeline
    import torch
    pipe=Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",token=SETTINGS.hf_token)
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    return pipe


def diarize(audio_path, enabled: bool | None=None):
    enabled=SETTINGS.enable_diarization if enabled is None else bool(enabled)
    if not enabled:
        return {"enabled":False,"turns":[],"note":"Speaker diarization disabled for this run."}
    if not SETTINGS.hf_token:
        return {"enabled":False,"turns":[],"error":"HF_TOKEN is required after accepting the pyannote model conditions."}
    try:
        pipe=_load_pipeline()
        output=pipe(audio_path)
        diar=getattr(output,"exclusive_speaker_diarization",None) or output.speaker_diarization
        turns=[
            {"speaker":speaker,"start":round(float(turn.start),3),"end":round(float(turn.end),3)}
            for turn,speaker in diar
        ]
        return {"enabled":True,"turns":turns,"confidence":0.9,"model":"pyannote/speaker-diarization-community-1","pipeline_cached":True}
    except Exception as e:
        return {"enabled":False,"turns":[],"error":str(e)[:300]}


def assign_speakers(segments,turns):
    for seg in segments:
        best=None; ovbest=0.0
        for t in turns:
            ov=max(0.0,min(seg["end"],t["end"])-max(seg["start"],t["start"]))
            if ov>ovbest:
                ovbest=ov; best=t["speaker"]
        seg["speaker"]=best or "UNKNOWN"
    return segments
