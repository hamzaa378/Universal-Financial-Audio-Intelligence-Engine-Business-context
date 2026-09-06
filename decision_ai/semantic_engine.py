"""Batched/cached semantic decision engine.

FastEmbed is optional. If fastembed-gpu is installed and CUDAExecutionProvider is
available, v4.4 can run the semantic layer on GPU too. Otherwise it uses CPU ONNX.
The engine batches all missing prototype embeddings so the first AI decision does not
re-encode the same example banks many times.
"""
from __future__ import annotations

import math
import os
from typing import Iterable

import numpy as np

_MODEL = None
_MODEL_NAME = None
_MODEL_ERROR = None
_MODEL_PROVIDER = None
_VECTOR_CACHE: dict[str, np.ndarray] = {}
_MAX_CACHE = 4096


def _pick_model(TextEmbedding) -> str:
    requested = os.getenv("FINAI_SEMANTIC_MODEL", "").strip()
    supported = TextEmbedding.list_supported_models()
    names = [x.get("model", "") for x in supported]
    if requested:
        if requested in names:
            return requested
        raise ValueError(f"FINAI_SEMANTIC_MODEL={requested!r} is not supported by this FastEmbed version")
    preferences = (
        "paraphrase-multilingual-MiniLM-L12-v2",
        "multilingual-e5-small",
        "all-MiniLM-L6-v2",
        "bge-small-en-v1.5",
    )
    for token in preferences:
        for name in names:
            if token.lower() in name.lower():
                return name
    if names:
        return names[0]
    raise RuntimeError("FastEmbed reported no supported text embedding models")


def _gpu_provider_available() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _load_model():
    global _MODEL, _MODEL_NAME, _MODEL_ERROR, _MODEL_PROVIDER
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERROR is not None:
        return None
    try:
        from fastembed import TextEmbedding
        _MODEL_NAME = _pick_model(TextEmbedding)
        requested = os.getenv("FINAI_SEMANTIC_DEVICE", "auto").strip().lower()
        providers = None
        if requested in {"cuda", "gpu"} or (requested == "auto" and _gpu_provider_available()):
            if _gpu_provider_available():
                providers = ["CUDAExecutionProvider"]
                _MODEL_PROVIDER = "CUDAExecutionProvider"
            elif requested in {"cuda", "gpu"}:
                raise RuntimeError(
                    "Semantic GPU was requested but ONNX Runtime CUDAExecutionProvider is unavailable. "
                    "Install fastembed-gpu or switch FINAI_SEMANTIC_DEVICE=cpu."
                )
        if providers is None:
            _MODEL_PROVIDER = "CPUExecutionProvider"
        kwargs = {"model_name": _MODEL_NAME}
        if providers is not None:
            kwargs["providers"] = providers
        try:
            _MODEL = TextEmbedding(**kwargs, lazy_load=True)
        except TypeError:
            _MODEL = TextEmbedding(**kwargs)
        return _MODEL
    except Exception as exc:
        _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def status() -> dict:
    model = _load_model()
    return {
        "available": model is not None,
        "model": _MODEL_NAME,
        "backend": "FastEmbed / ONNX Runtime" if model is not None else "rules fallback",
        "provider": _MODEL_PROVIDER,
        "cached_vectors": len(_VECTOR_CACHE),
        "error": _MODEL_ERROR,
    }


def _normalize(v) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x if n == 0 else x / n


def _ensure_vectors(texts: Iterable[str]) -> list[np.ndarray] | None:
    model = _load_model()
    if model is None:
        return None
    seq = [str(x) for x in texts]
    missing = []
    seen = set()
    for text in seq:
        if text not in _VECTOR_CACHE and text not in seen:
            missing.append(text)
            seen.add(text)
    if missing:
        try:
            vecs = list(model.embed(missing))
            if len(vecs) != len(missing):
                raise RuntimeError("Semantic model returned an unexpected embedding count")
            # Keep the cache bounded. Prototype banks are re-created cheaply if the cap
            # is ever reached during a very long-running server session.
            if len(_VECTOR_CACHE) + len(missing) > _MAX_CACHE:
                _VECTOR_CACHE.clear()
            for text, vec in zip(missing, vecs):
                _VECTOR_CACHE[text] = _normalize(vec)
        except Exception as exc:
            global _MODEL_ERROR
            _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    return [_VECTOR_CACHE[x] for x in seq]


def warmup(texts: Iterable[str]) -> dict:
    seq = list(dict.fromkeys(str(x) for x in texts if str(x).strip()))
    vecs = _ensure_vectors(seq)
    out = status()
    out["warmed_vectors"] = len(vecs) if vecs is not None else 0
    return out


def _score_vectors(q: np.ndarray, pos_vecs: list[np.ndarray], neg_vecs: list[np.ndarray]) -> dict:
    pos = max(float(np.dot(q, v)) for v in pos_vecs) if pos_vecs else 0.0
    neg = max(float(np.dot(q, v)) for v in neg_vecs) if neg_vecs else 0.0
    margin = pos - neg
    score = 1.0 / (1.0 + math.exp(-10.0 * margin))
    return {
        "available": True,
        "score": round(float(score), 4),
        "positive_similarity": round(pos, 4),
        "negative_similarity": round(neg, 4),
        "margin": round(margin, 4),
    }


def semantic_scores_many(cases: list[dict]) -> list[dict]:
    """Evaluate many semantic decisions in one embedding batch.

    Each case requires: text, positive_examples, negative_examples. Duplicate query or
    prototype strings are embedded only once.
    """
    if not cases:
        return []
    all_texts = []
    for c in cases:
        all_texts.append(str(c["text"]))
        all_texts.extend(c.get("positive_examples", ()))
        all_texts.extend(c.get("negative_examples", ()))
    unique = list(dict.fromkeys(all_texts))
    vecs = _ensure_vectors(unique)
    if vecs is None:
        return [{"available": False, "score": 0.5, "positive_similarity": 0.0, "negative_similarity": 0.0} for _ in cases]
    vmap = dict(zip(unique, vecs))
    out = []
    for c in cases:
        q = vmap[str(c["text"])]
        pos = [vmap[x] for x in c.get("positive_examples", ())]
        neg = [vmap[x] for x in c.get("negative_examples", ())]
        out.append(_score_vectors(q, pos, neg))
    return out


def semantic_score(text: str, positive_examples: tuple[str, ...], negative_examples: tuple[str, ...]) -> dict:
    return semantic_scores_many([{
        "text": text,
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
    }])[0]
