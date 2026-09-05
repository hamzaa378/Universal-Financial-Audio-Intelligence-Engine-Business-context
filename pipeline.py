from copy import deepcopy

from ingest.audio_loader import load_audio
from ingest.call_quality import call_quality
from ingest.tamper_detection import detect_tamper
from asr.fintech_asr import transcribe
from nlp.language import annotate_segment_languages
from nlp.pii import detect_pii, mask_pii, public_pii_metadata
from nlp.profanity import detect_profanity, reduce_profanity, public_profanity_metadata
from nlp.privacy import protect_text
from nlp.intent import classify_intent
from nlp.entities import extract_entities
from nlp.obligation import detect_obligations
from nlp.regulatory import check_regulatory
from analysis_ai.emotion import detect_stress_markers
from analysis_ai.diarization import diarize, assign_speakers


def _trust(asr_conf, quality, intent_conf):
    return round(max(0, min(1, 0.55*asr_conf + 0.25*quality + 0.20*intent_conf)), 3)


def _safe_transcription(asr: dict) -> dict:
    """Keep timing/confidence while preventing raw PII/profanity from normal output."""
    out = {
        "text": protect_text(asr.get("text", ""))["text"],
        "language": asr.get("language"),
        "language_probability": asr.get("language_probability"),
        "confidence": asr.get("confidence"),
        "model": asr.get("model"),
        "segments": [],
        "words": [],
    }
    for seg in asr.get("segments", []):
        item = {k: deepcopy(v) for k, v in seg.items() if k != "words"}
        item["text"] = protect_text(seg.get("text", ""))["text"]
        # Preserve word timing/confidence but not raw lexical values in privacy-safe mode.
        item["words"] = [
            {"token_index": i, "start": w.get("start"), "end": w.get("end"), "confidence": w.get("confidence")}
            for i, w in enumerate(seg.get("words", []))
        ]
        out["segments"].append(item)
    out["words"] = [
        {"token_index": i, "start": w.get("start"), "end": w.get("end"), "confidence": w.get("confidence")}
        for i, w in enumerate(asr.get("words", []))
    ]
    return out


def run_pipeline(audio_path, model_name=None, include_raw=False):
    audio, sr = load_audio(audio_path)
    quality = call_quality(audio, sr)
    asr = transcribe(audio_path, model_name)
    asr["segments"] = annotate_segment_languages(asr["segments"])
    dia = diarize(audio_path)
    if dia.get("turns"):
        asr["segments"] = assign_speakers(asr["segments"], dia["turns"])

    text = asr["text"]
    pii = detect_pii(text)
    profanity = detect_profanity(text)
    intent = classify_intent(text)
    entities = extract_entities(text)
    obligations = detect_obligations(text)

    pii_masked = mask_pii(text, pii)
    profanity_reduced = reduce_profanity(text, profanity)
    safe = protect_text(text)["text"]

    result = {
        "audio": {
            "quality": quality,
            "tamper_screen": detect_tamper(audio, sr),
            "diarization": dia,
        },
        "transcription": asr if include_raw else _safe_transcription(asr),
        "understanding": {
            "intent": intent,
            "financial_entities": entities,
            "obligations": obligations,
            "pii": public_pii_metadata(pii),
            "profanity": public_profanity_metadata(profanity),
            "regulatory": check_regulatory(text),
            "stress_markers": detect_stress_markers(audio, sr),
        },
        "privacy": {
            "pii_masked_transcript": pii_masked,
            "profanity_reduced_transcript": profanity_reduced,
            "safe_transcript": safe,
            "pii_count": len(pii),
            "profanity_count": len(profanity),
            "raw_transcript_included": bool(include_raw),
        },
        "confidence": {
            "overall_trust": _trust(asr["confidence"], quality["score"], intent["confidence"]),
            "interpretation": "Composite triage score; retain component confidences for audit.",
        },
    }
    if include_raw:
        result["privacy"]["warning"] = "Raw transcript requested: output may contain PII/profanity. Do not use this mode for normal UI/logging."
    return result
