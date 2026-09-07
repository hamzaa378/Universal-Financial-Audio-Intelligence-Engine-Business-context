from pathlib import Path
import numpy as np

SUPPORTED_EXTENSIONS={".wav",".mp3",".flac",".m4a",".ogg",".aac",".mp4",".webm",".mov"}


def _fast_soundfile_decode(path: str, target_sr: int):
    """Fast libsndfile + libsoxr path.

    librosa.load ultimately performs the same decode/mono/resample operations, but its
    higher-level setup is needlessly expensive for long files.  We keep the same
    high-quality SOXR resampling used by modern librosa, so ASR input quality is not
    reduced. Unsupported containers transparently fall back to librosa below.
    """
    import soundfile as sf
    import soxr
    data, rate=sf.read(path,dtype="float32",always_2d=True)
    if data.size==0:
        raise ValueError("Decoded audio is empty")
    audio=np.mean(data,axis=1,dtype=np.float32) if data.shape[1]>1 else data[:,0]
    if int(rate)!=int(target_sr):
        audio=soxr.resample(audio,int(rate),int(target_sr),quality="HQ")
    return np.asarray(audio,dtype=np.float32), int(target_sr)


def load_audio(path: str, sr: int=16000):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Audio/video file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS: raise ValueError(f"Unsupported format: {p.suffix}")
    try:
        audio,rate=_fast_soundfile_decode(str(p),int(sr))
    except Exception:
        # FFmpeg/audioread-backed formats such as some MP4/M4A files may not be
        # supported by the local libsndfile build. Preserve the proven fallback.
        import librosa
        audio,rate=librosa.load(str(p),sr=sr,mono=True,res_type="soxr_hq")
        audio=np.asarray(audio,dtype=np.float32)
    if audio.size==0: raise ValueError("Decoded audio is empty")
    audio=np.nan_to_num(audio.astype(np.float32,copy=False))
    return audio,int(rate)
