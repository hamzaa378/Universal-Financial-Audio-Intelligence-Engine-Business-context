import numpy as np
import librosa

def detect_stress_markers(audio, sr=16000):
    # Acoustic stress markers, deliberately reported as markers rather than a definitive emotion label.
    y=np.asarray(audio,dtype=np.float32)
    rms=librosa.feature.rms(y=y)[0]
    zcr=librosa.feature.zero_crossing_rate(y)[0]
    centroid=librosa.feature.spectral_centroid(y=y,sr=sr)[0]
    onset=librosa.onset.onset_strength(y=y,sr=sr)
    score=float(np.clip(0.30*np.std(rms)/(np.mean(rms)+1e-6)+0.25*np.mean(zcr)*8+0.25*np.std(centroid)/(np.mean(centroid)+1e-6)+0.20*np.mean(onset)/(np.max(onset)+1e-6),0,1))
    return {"stress_marker_score":round(score,3),"level":"high" if score>.68 else "medium" if score>.38 else "low","confidence":0.55,"method":"acoustic_markers","note":"Not a clinical or psychological diagnosis."}
