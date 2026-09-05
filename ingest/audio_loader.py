from pathlib import Path
import numpy as np
import librosa

SUPPORTED_EXTENSIONS={".wav",".mp3",".flac",".m4a",".ogg",".aac",".mp4",".webm",".mov"}

def load_audio(path: str, sr: int=16000):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Audio/video file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS: raise ValueError(f"Unsupported format: {p.suffix}")
    audio, rate=librosa.load(str(p), sr=sr, mono=True)
    if audio.size==0: raise ValueError("Decoded audio is empty")
    audio=np.nan_to_num(audio.astype(np.float32))
    return audio, rate
