"""Fast shared acoustic feature pass for call-quality and stress-marker scoring.

v4.12 removes the expensive librosa STFT/onset path from the hot loop.  The privacy
pipeline does not use these acoustic features to decide whether PII is masked, so this
optimization cannot lower masking recall/precision.  The replacement keeps the same
public metrics (RMS-derived SNR proxy, silence, clipping, ZCR, spectral centroid
variation and onset/activity variation) while computing them in vectorized NumPy
batches.
"""
from __future__ import annotations

import numpy as np

_FRAME = 1024
_HOP = 512
_FFT_BATCH = 512
_EPS = 1e-9


def _frames(x: np.ndarray, frame_length: int = _FRAME, hop_length: int = _HOP) -> np.ndarray:
    """Return a zero-copy frame view with librosa-like centered padding."""
    pad = frame_length // 2
    y = np.pad(x, (pad, pad), mode="constant")
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size), mode="constant")
    view = np.lib.stride_tricks.sliding_window_view(y, frame_length)
    return view[::hop_length]


def _spectral_stats(frames: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Return centroid std/mean and mean/max positive spectral-flux ratio.

    FFTs are chunked to cap temporary memory on long calls.  This is materially faster
    than invoking librosa's spectral-centroid plus mel-onset pipeline and avoids its
    first-run JIT/startup cost.
    """
    if frames.size == 0:
        return 0.0, 0.0, 0.0
    window = np.hanning(frames.shape[1]).astype(np.float32)
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / float(sr)).astype(np.float32)
    centroids = []
    flux_values = []
    prev_mag = None
    for i in range(0, len(frames), _FFT_BATCH):
        block = np.asarray(frames[i:i + _FFT_BATCH], dtype=np.float32)
        spec = np.fft.rfft(block * window, axis=1)
        mag = np.abs(spec).astype(np.float32, copy=False)
        denom = np.sum(mag, axis=1) + _EPS
        centroids.append(np.sum(mag * freqs, axis=1) / denom)
        if prev_mag is not None and len(mag):
            first = np.mean(np.maximum(mag[0] - prev_mag, 0.0))
            flux_values.append(np.asarray([first], dtype=np.float32))
        if len(mag) > 1:
            flux_values.append(np.mean(np.maximum(mag[1:] - mag[:-1], 0.0), axis=1))
        if len(mag):
            prev_mag = mag[-1]
    centroid = np.concatenate(centroids) if centroids else np.zeros(1, dtype=np.float32)
    flux = np.concatenate(flux_values) if flux_values else np.zeros(1, dtype=np.float32)
    cmean = float(np.mean(centroid))
    cstd = float(np.std(centroid))
    fmax = float(np.max(flux)) if flux.size else 0.0
    fratio = float(np.mean(flux) / (fmax + 1e-6)) if fmax > 0 else 0.0
    return cstd, cmean, fratio


def analyze_acoustics(audio, sr=16000) -> dict:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return {
            "quality": {"score": 0.0, "issues": ["empty_audio"], "confidence": 0.0},
            "stress_markers": {"stress_marker_score": 0.0, "level": "unknown", "confidence": 0.0, "method": "acoustic_markers"},
        }

    frames = _frames(x)
    # Keep temporary arrays float32 to reduce memory bandwidth on long recordings.
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1) + 1e-20)
    signs = frames >= 0.0
    zcr_arr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1, dtype=np.float32)

    silence = float(np.mean(rms < max(1e-5, float(np.percentile(rms, 25)) * 0.35)))
    peak = float(np.max(np.abs(x)))
    clipping = float(np.mean(np.abs(x) >= 0.995))
    zcr = float(np.mean(zcr_arr))
    hi = float(np.percentile(rms, 75) + 1e-9)
    lo = float(np.percentile(rms, 20) + 1e-9)
    snr_db = float(20 * np.log10(hi / lo))
    snr_score = float(np.clip((snr_db - 3) / 22, 0, 1))
    silence_score = 1 - min(1.0, silence / 0.65)
    clipping_score = 1 - min(1.0, clipping / 0.02)
    volume_score = float(np.clip(peak / 0.25, 0, 1))
    qscore = 0.45 * snr_score + 0.25 * silence_score + 0.2 * clipping_score + 0.1 * volume_score
    issues = []
    if snr_db < 8:
        issues.append("low_snr")
    if silence > 0.45:
        issues.append("high_silence")
    if clipping > 0.002:
        issues.append("clipping")
    if peak < 0.04:
        issues.append("low_volume")
    quality = {
        "score": round(float(qscore), 3), "snr_db_proxy": round(snr_db, 2),
        "silence_ratio": round(silence, 3), "clipping_ratio": round(clipping, 5),
        "zero_crossing_rate": round(zcr, 4), "issues": issues, "confidence": 0.82,
    }

    centroid_std, centroid_mean, onset_ratio = _spectral_stats(frames, int(sr))
    stress = float(np.clip(
        0.30 * np.std(rms) / (np.mean(rms) + 1e-6) +
        0.25 * float(np.mean(zcr_arr)) * 8 +
        0.25 * centroid_std / (centroid_mean + 1e-6) +
        0.20 * onset_ratio,
        0, 1,
    ))
    stress_markers = {
        "stress_marker_score": round(stress, 3),
        "level": "high" if stress > .68 else "medium" if stress > .38 else "low",
        "confidence": 0.55, "method": "shared_acoustic_features_fast_numpy",
        "note": "Acoustic marker only; not a clinical or psychological diagnosis.",
    }
    return {"quality": quality, "stress_markers": stress_markers}
