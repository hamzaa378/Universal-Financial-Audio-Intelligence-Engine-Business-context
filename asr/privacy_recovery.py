"""Expectation-aware ASR privacy recovery (v4.10).

This module is deliberately audio-privacy oriented.  It never invents or publishes a
missing identifier.  When the primary transcript strongly implies that the next short
response contains a sensitive field but the text detector found no value, it can:

1. re-decode only that small audio window with a field-specific prompt; and
2. if recovery still fails, return a bounded conservative audio interval to mute/beep.

The normal transcript remains the primary transcript.  Alternate hypotheses are kept
internal and only safe metadata is returned to callers.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np

from config import SETTINGS
from nlp.pii import detect_pii, infer_expected_type
from nlp.token_alignment import global_word_map, segment_offsets

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
# Recovery is slightly more tolerant than the normal discourse detector because ASR can
# corrupt "Mine is" to "Mine ID".  It is safe only because the immediately preceding
# sentence must establish one sensitive field.
_RECOVERY_RESPONSE = re.compile(
    r"(?i)^\s*(?:mine\b|it\s+(?:is|was)\b|i\s+(?:received|got|have|use|said|gave)\b|"
    r"the\s+(?:number|code|value|id)\s+(?:is|was)\b)"
)

_FIELD_LABEL = {
    "AADHAAR":"Aadhaar number", "PAN":"PAN number", "EMAIL":"email address",
    "PHONE":"mobile number", "UPI":"UPI ID", "IFSC":"IFSC code", "OTP":"OTP",
    "CVV":"CVV", "ACCOUNT_NUMBER":"bank account number", "PINCODE":"PIN code",
    "DOB":"date of birth", "PASSPORT":"passport number",
    "DRIVING_LICENSE":"driving licence number", "VOTER_ID":"voter ID",
    "CARD":"credit card number", "ADDRESS":"residential address", "NAME":"full name",
}
_FIELD_HOTWORDS = {
    "AADHAAR":"Aadhaar Aadhar UID digits", "PAN":"PAN alphanumeric",
    "EMAIL":"email at dot underscore domain", "PHONE":"phone mobile plus nine one digits",
    "UPI":"UPI VPA at oksbi okaxis okhdfcbank okicici",
    "IFSC":"IFSC HDFC SBI ICICI AXIS zero letters digits", "OTP":"OTP one time password digits",
    "CVV":"CVV CVC three digits", "ACCOUNT_NUMBER":"account number digits",
    "PINCODE":"PIN code postal code digits", "DOB":"date of birth day month year",
    "PASSPORT":"passport phonetic alphabet letters digits",
    "DRIVING_LICENSE":"driving licence Karnataka Maharashtra Delhi state code letters digits",
    "VOTER_ID":"voter ID EPIC letters digits", "CARD":"credit card number digits",
    "ADDRESS":"flat house road street city PIN code", "NAME":"full name registered name",
}
_MAX_UNCERTAIN_SECONDS = {
    "CVV":1.4,"OTP":1.8,"PINCODE":1.8,"PAN":2.4,"PASSPORT":2.6,"IFSC":2.8,
    "PHONE":2.8,"ACCOUNT_NUMBER":3.0,"CARD":3.2,"AADHAAR":3.2,"UPI":3.2,
    "EMAIL":3.5,"DOB":3.0,"DRIVING_LICENSE":3.8,"NAME":2.5,"ADDRESS":4.5,"VOTER_ID":2.8,
}


def _sentence_spans(text: str) -> list[tuple[int,int]]:
    return [(m.start(),m.end()) for m in _SENTENCE_RE.finditer(text or "") if m.group(0).strip()]


def _overlap(s1:int,e1:int,s2:int,e2:int) -> bool:
    return s1 < e2 and s2 < e1


def _span_time(asr: dict, start: int, end: int) -> tuple[float,float,float] | None:
    words=global_word_map(asr)
    hits=[w for w in words if _overlap(start,end,int(w["char_start"]),int(w["char_end"]))]
    if hits:
        conf=sum(float(w.get("confidence",0.0) or 0.0) for w in hits)/len(hits)
        return min(float(w["time_start"]) for w in hits), max(float(w["time_end"]) for w in hits), conf
    # Segment-level fallback when word alignment is unavailable.
    for cs,ce,seg in segment_offsets(asr.get("segments",[])):
        if _overlap(start,end,cs,ce):
            return float(seg.get("start",0.0)), float(seg.get("end",seg.get("start",0.0))), float(seg.get("confidence",0.0) or 0.0)
    return None


def plan_privacy_recovery(asr: dict, accepted_pii: Iterable[dict], *, max_windows: int=4) -> list[dict]:
    """Find high-risk expected-field responses with no accepted value in primary text."""
    text=str(asr.get("text","") or "")
    spans=_sentence_spans(text)
    pii=list(accepted_pii or [])
    plans=[]
    for i,(s0,s1) in enumerate(spans[:-1]):
        expected=infer_expected_type(text[s0:s1])
        if not expected:
            continue
        n0,n1=spans[i+1]
        response=text[n0:n1]
        if not _RECOVERY_RESPONSE.search(response):
            continue
        if any(x.get("type")==expected and _overlap(int(x.get("start",0)),int(x.get("end",0)),n0,n1) for x in pii):
            continue
        timing=_span_time(asr,n0,n1)
        if not timing:
            continue
        ts,te,conf=timing
        # Do not re-decode a huge ASR sentence.  The conservative fallback is also
        # capped by field type below.
        max_s=_MAX_UNCERTAIN_SECONDS.get(expected,3.0)
        te=min(te,ts+max_s+0.7)
        plans.append({
            "type":expected,"char_start":n0,"char_end":n1,
            "time_start":max(0.0,ts-0.20),"time_end":max(ts,min(te+0.20,te+0.20)),
            "primary_alignment_confidence":round(conf,4),
            "reason":"expected_field_missing_in_primary_asr",
        })
        if len(plans)>=max_windows:
            break
    return plans


def _backend_model(model_name: str | None):
    # Local imports avoid creating a second model cache and keep the regular text-only
    # path free of Faster-Whisper imports.
    from asr.fintech_asr import _load_model, _resolve_backend, _strict_gpu, _requested_device, _gpu_error_message
    import os
    name=model_name or SETTINGS.whisper_model
    device,compute,status=_resolve_backend()
    index=int(os.getenv("FINAI_CUDA_INDEX",str(SETTINGS.cuda_index)))
    model=_load_model(name,device,compute,index)
    return model,name,device,compute,index,status,_strict_gpu(),_requested_device()


def _decode_window(audio: np.ndarray, sr: int, plan: dict, model_name: str | None) -> dict:
    from asr.fintech_asr import _transcribe_materialized
    s=max(0,int(math.floor(float(plan["time_start"])*sr)))
    e=min(len(audio),int(math.ceil(float(plan["time_end"])*sr)))
    clip=np.asarray(audio[s:e],dtype=np.float32)
    if clip.size < max(800,int(0.05*sr)):
        return {"text":"","segments":[],"words":[],"error":"recovery_window_too_short"}
    model,name,device,compute,index,status,strict,requested=_backend_model(model_name)
    kind=plan["type"]
    label=_FIELD_LABEL.get(kind,kind)
    prompt=(
        f"Privacy recovery. The speaker is answering with their {label}. "
        "Transcribe exactly what is spoken. Preserve every digit, letter, word 'at', "
        "word 'dot', phonetic alphabet word and separator. Do not paraphrase or infer missing characters."
    )
    kwargs=dict(
        beam_size=max(5,int(getattr(SETTINGS,"beam_size",2))), best_of=max(5,int(getattr(SETTINGS,"best_of",1))),
        patience=1.15, word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
        initial_prompt=prompt, hotwords=_FIELD_HOTWORDS.get(kind,label), temperature=0.0,
        repetition_penalty=1.0, no_repeat_ngram_size=0, multilingual=True,
        language_detection_segments=1, hallucination_silence_threshold=2.0,
    )
    try:
        segments,info=_transcribe_materialized(model,clip,kwargs)
    except Exception as exc:
        # Recovery must never crash the main privacy pipeline.  The caller will use the
        # bounded conservative mute interval instead.
        return {"text":"","segments":[],"words":[],"error":f"{type(exc).__name__}: {exc}"}
    segs=[]; words=[]; offset=float(plan["time_start"])
    for seg in segments:
        sw=[]
        for w in (seg.words or []):
            item={
                "word":w.word.strip(),"start":round(offset+float(w.start),3),
                "end":round(offset+float(w.end),3),"confidence":round(float(w.probability),4),
            }
            sw.append(item); words.append(item)
        segs.append({
            "id":getattr(seg,"id",len(segs)),"start":round(offset+float(seg.start),3),
            "end":round(offset+float(seg.end),3),"text":seg.text.strip(),
            "confidence":round(sum(x["confidence"] for x in sw)/len(sw),4) if sw else 0.0,
            "words":sw,
        })
    return {"text":" ".join(x["text"] for x in segs).strip(),"segments":segs,"words":words,"error":None}


def _confirm_expected(kind: str, alt: dict) -> dict | None:
    text=str(alt.get("text","") or "").strip()
    if not text:
        return None
    prefix=f"My {_FIELD_LABEL.get(kind,kind)} is "
    synthetic=prefix+text
    matches=[x for x in detect_pii(synthetic,min_confidence=0.72,use_semantic=False,use_ner=False) if x.get("type")==kind]
    if not matches:
        return None
    item=max(matches,key=lambda x:(int(x.get("ownership_strength",0)),float(x.get("confidence",0)),x.get("end",0)-x.get("start",0)))
    rs=max(0,int(item["start"])-len(prefix)); re=max(rs,int(item["end"])-len(prefix))
    # Map recovered character span to recovered word times.
    cursor=0; hits=[]
    lower=text.casefold()
    for w in alt.get("words",[]):
        token=str(w.get("word","") or "").strip()
        if not token: continue
        idx=lower.find(token.casefold(),cursor)
        if idx<0:
            m=re.search(r"\S+",text[cursor:])
            if not m: continue
            idx=cursor+m.start(); end=cursor+m.end()
        else:
            end=idx+len(token)
        if _overlap(rs,re,idx,end): hits.append(w)
        cursor=max(cursor,end)
    if not hits:
        return None
    return {
        "type":kind,"start":max(0.0,min(float(w["start"]) for w in hits)-0.10),
        "end":max(float(w["end"]) for w in hits)+0.12,
        "confidence":round(float(item.get("confidence",0.0)),4),
        "alignment_confidence":round(sum(float(w.get("confidence",0.0)) for w in hits)/len(hits),4),
        "alignment_method":"privacy_targeted_redecode",
        "recovery_status":"confirmed_alternate_decode",
    }


def recover_privacy_audio_intervals(
    audio: np.ndarray, sr: int, asr: dict, accepted_pii: Iterable[dict],
    *, model_name: str | None=None, max_windows: int=4, conservative_on_failure: bool=True,
) -> dict:
    """Return audio-only redaction intervals for PII lost by the primary transcript."""
    plans=plan_privacy_recovery(asr,accepted_pii,max_windows=max_windows)
    intervals=[]; audit=[]
    for plan in plans:
        alt=_decode_window(audio,sr,plan,model_name)
        confirmed=_confirm_expected(plan["type"],alt)
        if confirmed:
            intervals.append(confirmed)
            audit.append({
                "type":plan["type"],"status":"confirmed_alternate_decode",
                "primary_alignment_confidence":plan["primary_alignment_confidence"],
                "alternate_asr_confidence":confirmed.get("alignment_confidence"),
            })
            continue
        if conservative_on_failure:
            max_s=_MAX_UNCERTAIN_SECONDS.get(plan["type"],3.0)
            start=float(plan["time_start"])
            end=min(float(plan["time_end"]),start+max_s)
            intervals.append({
                "type":plan["type"],"start":start,"end":max(start,end),"confidence":0.72,
                "alignment_method":"expected_field_conservative_window",
                "alignment_confidence":float(plan["primary_alignment_confidence"]),
                "recovery_status":"conservative_audio_guard",
            })
            status="conservative_audio_guard"
        else:
            status="unresolved"
        audit.append({
            "type":plan["type"],"status":status,
            "primary_alignment_confidence":plan["primary_alignment_confidence"],
            "alternate_decode_error":bool(alt.get("error")),
        })
    return {"plans":len(plans),"intervals":intervals,"audit":audit}
