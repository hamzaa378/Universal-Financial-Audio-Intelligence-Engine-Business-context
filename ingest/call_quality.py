import numpy as np
import librosa

def call_quality(audio, sr=16000):
    x=np.asarray(audio, dtype=np.float32)
    rms=librosa.feature.rms(y=x, frame_length=1024, hop_length=512)[0]
    silence=float(np.mean(rms < max(1e-5, np.percentile(rms, 25)*0.35)))
    peak=float(np.max(np.abs(x))) if len(x) else 0.0
    clipping=float(np.mean(np.abs(x)>=0.995))
    zcr=float(np.mean(librosa.feature.zero_crossing_rate(x)[0]))
    # Robust relative SNR proxy: high-energy speech frames vs lower-energy background frames.
    hi=float(np.percentile(rms,75)+1e-9); lo=float(np.percentile(rms,20)+1e-9)
    snr_db=float(20*np.log10(hi/lo))
    snr_score=float(np.clip((snr_db-3)/22,0,1))
    silence_score=1-min(1.0,silence/0.65)
    clipping_score=1-min(1.0,clipping/0.02)
    volume_score=float(np.clip(peak/0.25,0,1))
    score=0.45*snr_score+0.25*silence_score+0.2*clipping_score+0.1*volume_score
    issues=[]
    if snr_db<8: issues.append("low_snr")
    if silence>0.45: issues.append("high_silence")
    if clipping>0.002: issues.append("clipping")
    if peak<0.04: issues.append("low_volume")
    return {"score":round(score,3),"snr_db_proxy":round(snr_db,2),"silence_ratio":round(silence,3),"clipping_ratio":round(clipping,5),"issues":issues,"confidence":0.82}
