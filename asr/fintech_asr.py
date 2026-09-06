from __future__ import annotations

from functools import lru_cache
import math
import os
import shutil
from typing import Any

import numpy as np

from config import SETTINGS

_LAST_BACKEND = {
    "device": None,
    "compute_type": None,
    "cuda_ready": None,
    "strict_gpu": None,
    "error": None,
}


def cuda_runtime_status() -> dict:
    """Report CTranslate2 GPU visibility and Windows DLL visibility.

    DLL visibility is diagnostic only. When CUDA is explicitly requested, the actual
    CTranslate2 inference attempt is authoritative; v4.4 does not reject a working GPU
    merely because an auxiliary DLL probe is imperfect.
    """
    if os.getenv("FINAI_FORCE_CPU", "0") == "1":
        return {"ready": False, "reason": "FINAI_FORCE_CPU=1", "device_count": 0}
    try:
        import ctranslate2
        count = int(ctranslate2.get_cuda_device_count())
        version = getattr(ctranslate2, "__version__", "unknown")
    except Exception as exc:
        return {
            "ready": False,
            "reason": f"CTranslate2 CUDA probe failed: {type(exc).__name__}: {exc}",
            "device_count": 0,
        }
    dlls = {}
    if os.name == "nt":
        for name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
            dlls[name] = shutil.which(name)
    return {
        "ready": count > 0,
        "reason": None if count > 0 else "No CUDA device visible to CTranslate2",
        "device_count": count,
        "ctranslate2": version,
        "dlls": dlls,
    }


def _requested_device() -> str:
    if os.getenv("FINAI_FORCE_CPU", "0") == "1":
        return "cpu"
    return (os.getenv("FINAI_DEVICE", SETTINGS.device) or "auto").strip().lower()


def _strict_gpu() -> bool:
    raw = os.getenv("FINAI_STRICT_GPU")
    if raw is None:
        return bool(SETTINGS.strict_gpu)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_backend() -> tuple[str, str, dict]:
    requested = _requested_device()
    compute_requested = os.getenv("FINAI_COMPUTE_TYPE", SETTINGS.compute_type)
    status = cuda_runtime_status()

    if requested == "cpu":
        compute = "int8" if compute_requested == "auto" else compute_requested
        return "cpu", compute, status

    if requested == "cuda":
        if not status.get("ready"):
            raise RuntimeError(
                "CUDA was explicitly requested but CTranslate2 cannot see an NVIDIA GPU. "
                f"Diagnostic: {status.get('reason')}"
            )
        compute = "float16" if compute_requested == "auto" else compute_requested
        return "cuda", compute, status

    # auto mode may fall back to CPU. The normal v4.4 launcher does not use auto.
    if status.get("ready"):
        compute = "float16" if compute_requested == "auto" else compute_requested
        return "cuda", compute, status
    compute = "int8" if compute_requested == "auto" else compute_requested
    return "cpu", compute, status


@lru_cache(maxsize=6)
def _load_model(model_name: str, device: str, compute_type: str, device_index: int):
    from faster_whisper import WhisperModel
    kwargs: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
    }
    if device == "cuda":
        kwargs["device_index"] = device_index
    return WhisperModel(model_name, **kwargs)


def _domain_prompt() -> str:
    vocab = ", ".join(SETTINGS.financial_vocab)
    return (
        "Indian financial-services customer call. Preserve spoken digits, dates, money, "
        "interest rates, identifiers, acronyms and code-switched English/Hindi faithfully. "
        "Do not paraphrase. Financial vocabulary: " + vocab + "."
    )


def _profile(mode: str) -> dict:
    mode = (mode or "balanced").lower()
    if mode == "fast":
        return {
            "beam_size": 1,
            "best_of": 1,
            "condition_on_previous_text": False,
            "language_detection_segments": 1,
            "patience": 1.0,
        }
    if mode == "accuracy":
        return {
            "beam_size": max(5, SETTINGS.beam_size),
            "best_of": max(5, SETTINGS.best_of),
            "condition_on_previous_text": True,
            "language_detection_segments": 3,
            "patience": 1.2,
        }
    return {
        "beam_size": max(2, SETTINGS.beam_size),
        "best_of": 1,
        "condition_on_previous_text": True,
        "language_detection_segments": 2,
        "patience": 1.0,
    }


def backend_status() -> dict:
    try:
        device, compute, status = _resolve_backend()
        error = None
    except Exception as exc:
        device, compute, status, error = "error", "-", cuda_runtime_status(), str(exc)
    return {
        "requested_device": _requested_device(),
        "strict_gpu": _strict_gpu(),
        "planned_device": device,
        "planned_compute_type": compute,
        "cuda_status": status,
        "last_backend": dict(_LAST_BACKEND),
        "error": error,
    }


def _transcribe_materialized(model, audio_input, kwargs: dict):
    segments, info = model.transcribe(audio_input, **kwargs)
    return list(segments), info


def _gpu_error_message(exc: Exception, status: dict) -> str:
    dlls = status.get("dlls") or {}
    dll_text = ", ".join(f"{k}={'FOUND' if v else 'NOT_FOUND'}" for k, v in dlls.items())
    return (
        f"GPU ASR failed: {type(exc).__name__}: {exc}. "
        f"CTranslate2 devices={status.get('device_count')}; {dll_text}. "
        "v4.4 strict-GPU mode will not silently run this request on CPU."
    )


def warmup_model(model_name: str | None = None, *, smoke_test: bool = True) -> dict:
    """Load the configured model and optionally execute a tiny encoder inference.

    The smoke test makes the warm-up meaningful: CUDA/cuDNN errors that appear only at
    inference time are surfaced before the user processes a real call.
    """
    name = model_name or SETTINGS.whisper_model
    device, compute, status = _resolve_backend()
    index = int(os.getenv("FINAI_CUDA_INDEX", str(SETTINGS.cuda_index)))
    model = _load_model(name, device, compute, index)
    if smoke_test:
        silence = np.zeros(16000, dtype=np.float32)
        kwargs = dict(
            beam_size=1,
            best_of=1,
            word_timestamps=False,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
            language="en",
        )
        try:
            _transcribe_materialized(model, silence, kwargs)
        except Exception as exc:
            if device == "cuda":
                raise RuntimeError(_gpu_error_message(exc, status)) from exc
            raise
    return {
        "model": name,
        "device": device,
        "compute_type": compute,
        "device_index": index if device == "cuda" else None,
        "cuda_status": status,
        "smoke_test": bool(smoke_test),
    }


def transcribe(audio_input, model_name: str | None = None, speed_mode: str = "balanced"):
    """Transcribe a path or a pre-decoded 16 kHz float32 waveform.

    Passing the already-loaded waveform avoids decoding the same call twice.
    """
    name = model_name or SETTINGS.whisper_model
    profile = _profile(speed_mode)
    hotwords = ", ".join(SETTINGS.financial_vocab)
    kwargs = dict(
        beam_size=profile["beam_size"],
        best_of=profile["best_of"],
        patience=profile["patience"],
        word_timestamps=True,
        vad_filter=SETTINGS.vad_filter,
        vad_parameters={"min_silence_duration_ms": 260, "speech_pad_ms": 120},
        condition_on_previous_text=profile["condition_on_previous_text"],
        initial_prompt=_domain_prompt(),
        hotwords=hotwords,
        temperature=0.0,
        repetition_penalty=SETTINGS.repetition_penalty,
        no_repeat_ngram_size=SETTINGS.no_repeat_ngram_size,
        multilingual=True,
        language_detection_segments=profile["language_detection_segments"],
        hallucination_silence_threshold=SETTINGS.hallucination_silence_threshold,
    )
    device, compute, status = _resolve_backend()
    strict = _strict_gpu()
    index = int(os.getenv("FINAI_CUDA_INDEX", str(SETTINGS.cuda_index)))
    try:
        model = _load_model(name, device, compute, index)
        segments, info = _transcribe_materialized(model, audio_input, kwargs)
    except Exception as exc:
        if device == "cuda" and (strict or _requested_device() == "cuda"):
            _LAST_BACKEND.update({
                "device": "cuda",
                "compute_type": compute,
                "cuda_ready": bool(status.get("ready")),
                "strict_gpu": strict,
                "error": str(exc),
            })
            raise RuntimeError(_gpu_error_message(exc, status)) from exc
        # Only auto mode is allowed to recover on CPU.
        if device == "cuda":
            device, compute = "cpu", "int8"
            model = _load_model(name, device, compute, 0)
            segments, info = _transcribe_materialized(model, audio_input, kwargs)
        else:
            raise

    _LAST_BACKEND.update({
        "device": device,
        "compute_type": compute,
        "cuda_ready": bool(status.get("ready")),
        "strict_gpu": strict,
        "error": None,
    })

    words = []
    segs = []
    for s in segments:
        sw = []
        for w in (s.words or []):
            item = {
                "word": w.word.strip(),
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "confidence": round(float(w.probability), 4),
            }
            words.append(item)
            sw.append(item)
        conf = (
            sum(x["confidence"] for x in sw) / len(sw)
            if sw
            else max(0.0, min(1.0, float(math.exp(s.avg_logprob))))
        )
        segs.append({
            "id": s.id,
            "start": round(float(s.start), 3),
            "end": round(float(s.end), 3),
            "text": s.text.strip(),
            "confidence": round(conf, 4),
            "words": sw,
        })

    text = " ".join(s["text"] for s in segs).strip()
    overall = sum(w["confidence"] for w in words) / len(words) if words else 0.0
    return {
        "text": text,
        "segments": segs,
        "words": words,
        "language": getattr(info, "language", None),
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
        "confidence": round(overall, 4),
        "model": name,
        "backend": {
            "device": device,
            "compute_type": compute,
            "speed_mode": speed_mode,
            "strict_gpu": strict,
            "device_index": index if device == "cuda" else None,
        },
    }
