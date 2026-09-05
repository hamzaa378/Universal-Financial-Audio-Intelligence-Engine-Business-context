from config import SETTINGS

def diarize(audio_path):
    if not SETTINGS.enable_diarization:
        return {"enabled":False,"turns":[],"note":"Set FINAI_DIARIZATION=1 and HF_TOKEN to enable pyannote Community-1."}
    if not SETTINGS.hf_token:
        return {"enabled":False,"turns":[],"error":"HF_TOKEN is required after accepting the pyannote model conditions."}
    try:
        from pyannote.audio import Pipeline
        import torch
        pipe=Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",token=SETTINGS.hf_token)
        if torch.cuda.is_available(): pipe.to(torch.device("cuda"))
        output=pipe(audio_path)
        diar=getattr(output,"exclusive_speaker_diarization",None) or output.speaker_diarization
        turns=[{"speaker":speaker,"start":round(float(turn.start),3),"end":round(float(turn.end),3)} for turn,speaker in diar]
        return {"enabled":True,"turns":turns,"confidence":0.9,"model":"pyannote/speaker-diarization-community-1"}
    except Exception as e:
        return {"enabled":False,"turns":[],"error":str(e)[:300]}

def assign_speakers(segments, turns):
    for seg in segments:
        best=None; ovbest=0
        for t in turns:
            ov=max(0,min(seg["end"],t["end"])-max(seg["start"],t["start"]))
            if ov>ovbest: ovbest=ov; best=t["speaker"]
        seg["speaker"]=best or "UNKNOWN"
    return segments
