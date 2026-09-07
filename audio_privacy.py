"""Create a privacy-protected audio copy from ASR word timings.

The text privacy engine decides *what* is sensitive. This module only converts those
character spans into time intervals and applies a tone or mute to the corresponding
audio. Raw PII values are never returned in the interval metadata.
"""
from __future__ import annotations

from pathlib import Path
import math
import re
import tempfile
from typing import Iterable

from nlp.token_alignment import global_word_map

import librosa
import numpy as np
import soundfile as sf


def _segment_offsets(segments: list[dict]) -> list[tuple[int, int, dict]]:
    """Offsets matching the ASR top-level text: ' '.join(segment['text'])."""
    out=[]
    cursor=0
    for i, seg in enumerate(segments):
        text=str(seg.get("text", ""))
        start=cursor
        end=start+len(text)
        out.append((start,end,seg))
        cursor=end + (1 if i < len(segments)-1 else 0)
    return out


def _word_char_spans(segment_text: str, words: list[dict]) -> list[tuple[int,int,dict]]:
    """Best-effort alignment of Whisper words to character positions in a segment."""
    spans=[]
    cursor=0
    lower=segment_text.lower()
    for w in words:
        token=str(w.get("word", "")).strip()
        if not token:
            continue
        # Faster-Whisper tokens usually occur verbatim in segment text.
        idx=lower.find(token.lower(), cursor)
        if idx < 0:
            # Retry without surrounding punctuation.
            clean=re.sub(r"^[^\w@+]+|[^\w@.%+-]+$", "", token, flags=re.UNICODE)
            idx=lower.find(clean.lower(), cursor) if clean else -1
            used=clean
        else:
            used=token
        if idx < 0:
            # Fall back to the next whitespace token. This keeps time masking safe even
            # when punctuation normalization differs between segment and word output.
            m=re.search(r"\S+", segment_text[cursor:])
            if not m:
                continue
            idx=cursor+m.start()
            end=cursor+m.end()
        else:
            end=idx+len(used)
        spans.append((idx,end,w))
        cursor=max(cursor,end)
    return spans


def _merge_intervals(intervals: Iterable[dict], gap_s: float = 0.06) -> list[dict]:
    items=sorted(intervals, key=lambda x:(x["start"],x["end"]))
    merged=[]
    for item in items:
        if not merged or item["start"] > merged[-1]["end"] + gap_s:
            merged.append(dict(item))
            continue
        prev=merged[-1]
        prev["end"]=max(prev["end"], item["end"])
        prev["types"]=sorted(set(prev.get("types",[])) | set(item.get("types",[])))
        prev["confidence"]=round(max(float(prev.get("confidence",0)), float(item.get("confidence",0))),4)
        pm=prev.get("alignment_method"); im=item.get("alignment_method")
        if pm and im and pm != im:
            prev["alignment_method"]="mixed"
        elif not pm:
            prev["alignment_method"]=im
        pcs=[x for x in (prev.get("alignment_confidence"),item.get("alignment_confidence")) if isinstance(x,(int,float))]
        if pcs:
            prev["alignment_confidence"]=round(min(float(x) for x in pcs),4)
    return merged


def spans_to_audio_intervals(asr: dict, pii: list[dict], profanity: list[dict], *, pad_s: float = 0.09) -> list[dict]:
    """Map sensitive entities to audio intervals, preferring owned Whisper tokens.

    v4.5 entities are enriched with token_ids/time_start/time_end by the pipeline.  The
    older character-span mapper remains only as a guarded fallback for unresolved cases.
    """
    sensitive=[]
    for x in pii:
        sensitive.append({**x,"type":x.get("type","PII")})
    for x in profanity:
        sensitive.append({**x,"type":"PROFANITY"})
    if not sensitive:
        return []

    word_map=global_word_map(asr)
    token_lookup={int(w["token_id"]):w for w in word_map}
    intervals=[]

    # Fast path: direct token/time ownership.
    unresolved=[]
    for entity in sensitive:
        start=entity.get("time_start"); end=entity.get("time_end")
        method=entity.get("alignment_method")
        align_conf=float(entity.get("alignment_confidence",0.0) or 0.0)
        if start is None or end is None:
            tids=[int(x) for x in (entity.get("token_ids") or []) if int(x) in token_lookup]
            if tids:
                hits=[token_lookup[x] for x in tids]
                start=min(x["time_start"] for x in hits); end=max(x["time_end"] for x in hits)
                align_conf=sum(x["confidence"] for x in hits)/len(hits)
                method="whisper_word_tokens"
        if start is not None and end is not None and float(end)>=float(start):
            # Low-confidence token boundaries receive a slightly larger privacy pad.
            extra=0.06 if align_conf and align_conf < 0.72 else 0.0
            pad=pad_s+extra
            intervals.append({
                "start":max(0.0,float(start)-pad),
                "end":max(float(start),float(end)+pad),
                "types":[entity["type"]],
                "confidence":round(float(entity.get("confidence",0.0)),4),
                "alignment_method":method or "whisper_word_tokens",
                "alignment_confidence":round(align_conf,4),
            })
        else:
            unresolved.append(entity)

    # Guarded compatibility fallback. Character ratio is used only if token ownership
    # genuinely failed, with extra padding to reduce leakage risk.
    if unresolved:
        for seg_global_start, seg_global_end, seg in _segment_offsets(asr.get("segments", [])):
            seg_text=str(seg.get("text", ""))
            words=seg.get("words", []) or []
            word_spans=_word_char_spans(seg_text,words)
            for entity in unresolved:
                es=int(entity.get("start",0)); ee=int(entity.get("end",es))
                if es >= seg_global_end or ee <= seg_global_start:
                    continue
                local_start=max(0,es-seg_global_start); local_end=min(len(seg_text),ee-seg_global_start)
                hits=[w for cs,ce,w in word_spans if cs < local_end and local_start < ce]
                if hits:
                    start=min(float(w.get("start",seg.get("start",0.0))) for w in hits)
                    end=max(float(w.get("end",seg.get("end",start))) for w in hits)
                    method="late_word_alignment"
                    align_conf=sum(float(w.get("confidence",0.0) or 0.0) for w in hits)/len(hits)
                    pad=pad_s+0.04
                else:
                    seg_start=float(seg.get("start",0.0)); seg_end=float(seg.get("end",seg_start))
                    duration=max(0.0,seg_end-seg_start); denom=max(1,len(seg_text))
                    start=seg_start+duration*(local_start/denom)
                    end=seg_start+duration*(local_end/denom)
                    method="character_ratio_fallback"
                    align_conf=0.0
                    pad=pad_s+0.12
                intervals.append({
                    "start":max(0.0,start-pad),"end":max(start,end+pad),
                    "types":[entity.get("type","PII")],
                    "confidence":round(float(entity.get("confidence",0.0)),4),
                    "alignment_method":method,
                    "alignment_confidence":round(float(align_conf),4),
                })
    return _merge_intervals(intervals)

def _apply_fade(mask: np.ndarray, sr: int, fade_ms: float = 8.0) -> np.ndarray:
    n=min(len(mask)//2, max(1,int(sr*fade_ms/1000.0)))
    if n <= 1:
        return mask
    ramp=np.linspace(0.0,1.0,n,dtype=np.float32)
    mask[:n]*=ramp
    mask[-n:]*=ramp[::-1]
    return mask


def create_protected_audio(
    audio_path: str,
    asr: dict,
    pii: list[dict],
    profanity: list[dict],
    *,
    method: str = "beep",
    tone_hz: float = 980.0,
    pad_s: float = 0.09,
    output_path: str | None = None,
    preloaded_audio: tuple[np.ndarray, int] | None = None,
    extra_intervals: list[dict] | None = None,
) -> dict:
    """Create a WAV with sensitive intervals replaced by a tone or silence."""
    if preloaded_audio is None:
        audio, sr=librosa.load(audio_path, sr=None, mono=True)
        audio=np.nan_to_num(audio.astype(np.float32))
    else:
        audio, sr=preloaded_audio
        audio=np.nan_to_num(np.asarray(audio,dtype=np.float32))
    intervals=spans_to_audio_intervals(asr, pii, profanity, pad_s=pad_s)
    # v4.10 may add audio-only intervals recovered from a targeted second ASR decode.
    # They intentionally have no primary-transcript character span because the first
    # ASR pass may have deleted the sensitive value entirely.
    if extra_intervals:
        recovered=[]
        for x in extra_intervals:
            if x.get("start") is None or x.get("end") is None:
                continue
            recovered.append({
                "start":max(0.0,float(x["start"])),"end":max(float(x["start"]),float(x["end"])),
                "types":[str(x.get("type","PII"))],
                "confidence":round(float(x.get("confidence",0.0)),4),
                "alignment_method":x.get("alignment_method","privacy_recovery"),
                "alignment_confidence":round(float(x.get("alignment_confidence",0.0) or 0.0),4),
            })
        intervals=_merge_intervals(list(intervals)+recovered)
    safe=audio.copy()
    method=(method or "beep").lower()
    if method not in {"beep","mute"}:
        raise ValueError("audio redaction method must be 'beep' or 'mute'")

    for item in intervals:
        s=max(0,min(len(safe),int(math.floor(item["start"]*sr))))
        e=max(s,min(len(safe),int(math.ceil(item["end"]*sr))))
        if e <= s:
            continue
        if method == "mute":
            safe[s:e]=0.0
        else:
            n=e-s
            t=np.arange(n,dtype=np.float32)/float(sr)
            # Keep tone audible but below clipping regardless of source loudness.
            source_peak=float(np.max(np.abs(safe[s:e]))) if n else 0.0
            amp=min(0.22,max(0.10,source_peak*0.85))
            tone=(amp*np.sin(2*np.pi*float(tone_hz)*t)).astype(np.float32)
            safe[s:e]=_apply_fade(tone,sr)

    if output_path is None:
        fd,path=tempfile.mkstemp(prefix="financial_audio_protected_",suffix=".wav")
        import os
        os.close(fd)
        output_path=path
    sf.write(output_path,safe,sr,subtype="PCM_16")
    return {
        "path":str(Path(output_path)),
        "method":method,
        "interval_count":len(intervals),
        "intervals":[{
            "start":round(float(x["start"]),3),
            "end":round(float(x["end"]),3),
            "types":x.get("types",[]),
            "confidence":x.get("confidence"),
            "alignment_method":x.get("alignment_method"),
            "alignment_confidence":x.get("alignment_confidence"),
        } for x in intervals],
        "duration_s":round(len(safe)/float(sr),3) if sr else 0.0,
        "sample_rate":int(sr),
    }
