import numpy as np

def detect_tamper(audio, sr=16000):
    x=np.asarray(audio,dtype=np.float32)
    if len(x)<sr: return {"risk":"UNKNOWN","score":0.5,"confidence":0.35,"signals":["audio_too_short"]}
    # Detect abrupt discontinuities and suspiciously repeated blocks. This is forensic triage, not proof of tampering.
    diff=np.abs(np.diff(x))
    jump_thr=max(0.25,float(np.median(diff)+18*np.median(np.abs(diff-np.median(diff)))))
    jump_rate=float(np.mean(diff>jump_thr))
    block=sr//2
    hashes=[]
    for i in range(0,len(x)-block+1,block):
        q=np.round(x[i:i+block][::80]*100).astype(np.int16)
        hashes.append(hash(q.tobytes()))
    repeat_ratio=0.0 if not hashes else 1-len(set(hashes))/len(hashes)
    risk_score=float(np.clip(jump_rate*250+repeat_ratio*0.8,0,1))
    signals=[]
    if jump_rate>0.001: signals.append("abrupt_waveform_discontinuities")
    if repeat_ratio>0.15: signals.append("repeated_audio_blocks")
    risk="HIGH" if risk_score>=0.7 else "MEDIUM" if risk_score>=0.35 else "LOW"
    return {"risk":risk,"score":round(risk_score,3),"confidence":0.58,"signals":signals,"note":"Screening signal only; not a forensic authenticity determination."}
