"""Shared acoustic feature pass for call-quality and stress-marker scoring."""
from __future__ import annotations

import numpy as np
import librosa


def analyze_acoustics(audio, sr=16000) -> dict:
    x=np.asarray(audio,dtype=np.float32)
    if x.size==0:
        return {
            "quality":{"score":0.0,"issues":["empty_audio"],"confidence":0.0},
            "stress_markers":{"stress_marker_score":0.0,"level":"unknown","confidence":0.0,"method":"acoustic_markers"},
        }
    rms=librosa.feature.rms(y=x,frame_length=1024,hop_length=512)[0]
    zcr_arr=librosa.feature.zero_crossing_rate(x,frame_length=1024,hop_length=512)[0]
    centroid=librosa.feature.spectral_centroid(y=x,sr=sr,n_fft=1024,hop_length=512)[0]
    onset=librosa.onset.onset_strength(y=x,sr=sr,hop_length=512)

    silence=float(np.mean(rms < max(1e-5,np.percentile(rms,25)*0.35)))
    peak=float(np.max(np.abs(x)))
    clipping=float(np.mean(np.abs(x)>=0.995))
    zcr=float(np.mean(zcr_arr))
    hi=float(np.percentile(rms,75)+1e-9); lo=float(np.percentile(rms,20)+1e-9)
    snr_db=float(20*np.log10(hi/lo))
    snr_score=float(np.clip((snr_db-3)/22,0,1))
    silence_score=1-min(1.0,silence/0.65)
    clipping_score=1-min(1.0,clipping/0.02)
    volume_score=float(np.clip(peak/0.25,0,1))
    qscore=0.45*snr_score+0.25*silence_score+0.2*clipping_score+0.1*volume_score
    issues=[]
    if snr_db<8: issues.append("low_snr")
    if silence>0.45: issues.append("high_silence")
    if clipping>0.002: issues.append("clipping")
    if peak<0.04: issues.append("low_volume")
    quality={
        "score":round(float(qscore),3),"snr_db_proxy":round(snr_db,2),
        "silence_ratio":round(silence,3),"clipping_ratio":round(clipping,5),
        "zero_crossing_rate":round(zcr,4),"issues":issues,"confidence":0.82,
    }

    stress=float(np.clip(
        0.30*np.std(rms)/(np.mean(rms)+1e-6)+
        0.25*np.mean(zcr_arr)*8+
        0.25*np.std(centroid)/(np.mean(centroid)+1e-6)+
        0.20*np.mean(onset)/(np.max(onset)+1e-6),0,1
    ))
    stress_markers={
        "stress_marker_score":round(stress,3),
        "level":"high" if stress>.68 else "medium" if stress>.38 else "low",
        "confidence":0.55,"method":"shared_acoustic_features",
        "note":"Acoustic marker only; not a clinical or psychological diagnosis.",
    }
    return {"quality":quality,"stress_markers":stress_markers}
