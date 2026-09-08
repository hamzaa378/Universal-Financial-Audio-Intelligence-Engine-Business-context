"""Expectation-aware ASR privacy recovery (v4.12).

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
import time
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

_NO_VALUE_RESPONSE = re.compile(
    r"(?i)^\s*(?:not\s+(?:available|provided|known)|unavailable|unknown|none|n/?a|"
    r"prefer\s+not\s+to\s+(?:say|provide|share)|do\s+not\s+have)\b"
)
# Explicit personal assignment starts used only to *plan* recovery when primary ASR did
# not yield a credible entity. They do not themselves create a text redaction.
_OWNED_RECOVERY_STARTS = (
    ("NAME",re.compile(r"(?i)\b(?:my\s+(?:(?:full|registered|legal|official|preferred)\s+)?name|name\s+on\s+my\s+(?:account|card|passport))\s+(?:is|:|=)\s*")),
    ("PHONE",re.compile(r"(?i)\b(?:my\s+(?:(?:registered|alternate|secondary|backup|personal|work)\s+)?(?:phone|mobile|contact)(?:\s+number)?\s+(?:is|:|=)|(?:call|reach|contact|text|message|sms)\s+me\s+(?:at|on))\s*")),
    ("EMAIL",re.compile(r"(?i)\b(?:my\s+(?:(?:registered|alternate|secondary|personal|work)\s+)?(?:email|e-mail)(?:\s+address)?|mail\s+me|email\s+me)\s+(?:is|at|:|=)?\s*")),
    ("PAN",re.compile(r"(?i)\bmy\s+(?:pan|permanent\s+account\s+number)(?:\s+number)?\s+(?:is|:|=)\s*")),
    ("AADHAAR",re.compile(r"(?i)\bmy\s+(?:aadhaar|aadhar)(?:\s+number)?\s+(?:is|:|=)\s*")),
    ("IFSC",re.compile(r"(?i)\bmy\s+(?:bank\s+)?ifsc(?:\s+code)?\s+(?:is|:|=)\s*")),
    ("UPI",re.compile(r"(?i)\bmy\s+(?:upi(?:\s+(?:id|address))?|vpa)\s+(?:is|:|=)\s*")),
    ("CARD",re.compile(r"(?i)\bmy\s+(?:(?:credit|debit)\s+)?card(?:\s+number)?\s+(?:is|:|=)\s*")),
    ("CVV",re.compile(r"(?i)\b(?:my\s+cvv|the\s+cvv\s+on\s+my\s+card)\s+(?:is|:|=)\s*")),
    ("OTP",re.compile(r"(?i)\bmy\s+(?:otp|verification\s+code|one[- ]time\s+(?:password|passcode))\s+(?:is|:|=)\s*")),
    ("ACCOUNT_NUMBER",re.compile(r"(?i)\bmy\s+(?:bank\s+)?account(?:\s+number)?\s+(?:is|:|=)\s*")),
    ("PINCODE",re.compile(r"(?i)\bmy\s+(?:pin\s*code|pincode|postal\s+code|zip\s+code)\s+(?:is|:|=)\s*")),
    ("DOB",re.compile(r"(?i)\bmy\s+(?:dob|date\s+of\s+birth|birth\s+date)\s+(?:is|:|=)\s*")),
    ("PASSPORT",re.compile(r"(?i)\bmy\s+passport(?:\s+number)?\s+(?:is|:|=)\s*")),
    ("DRIVING_LICENSE",re.compile(r"(?i)\bmy\s+(?:driving\s+licen[cs]e|driver'?s\s+licen[cs]e|dl)(?:\s+number)?\s+(?:is|:|=)\s*")),
    ("VOTER_ID",re.compile(r"(?i)\bmy\s+(?:voter\s*(?:id|card)|epic(?:\s+(?:id|number))?)\s+(?:is|:|=)\s*")),
    ("ADDRESS",re.compile(r"(?i)\bmy\s+(?:(?:residential|permanent|current|registered|mailing|billing|delivery)\s+)?address\s+(?:is|:|=)\s*")),
)
_RECOVERY_NEXT_FIELD = re.compile(
    r"(?i)(?:,\s*|\s+)(?:and\s+)?(?:(?:my|the|your)\s+)?(?:name|phone|mobile|email|account|ifsc|pan|upi|aadhaar|aadhar|otp|cvv|address|dob|date\s+of\s+birth|passport|driving\s+licen[cs]e|card|transaction|complaint|order|reference|product\s+batch)\b"
)
# v4.12: documentation/example material is a *hard* ownership boundary.  This is
# narrower than the semantic example veto: it is used only to stop raw/recovery spans
# from swallowing the next non-personal clause when ASR omitted punctuation.
_RECOVERY_HARD_BOUNDARY = re.compile(
    r"(?i)(?:[;!?]\s*|,\s*(?:and\s+|but\s+)?|\s+(?:and|but|while|whereas)\s+)"
    r"(?:(?:the|a|an)\s+)?(?:software\s+manual|documentation|docs?|tutorial|training\s+(?:document|video|material|guide)|"
    r"test\s+(?:value|number|card|data|case)|sample|example|demonstration|product\s+batch|shipment\s+reference|invoice\s+reference|reference\s+(?:number|document))\b"
)
_PHONETIC_CONTINUATION = re.compile(
    r"(?i)^\s*(?:alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|india|juliett?|kilo|lima|mike|november|oscar|papa|quebec|romeo|sierra|tango|uniform|victor|whiskey|x-?ray|yankee|zulu|zero|oh|one|two|three|four|five|six|seven|eight|nine|[A-Z]|\d)\b"
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


def _accepted_in_span(pii: list[dict], kind: str, start: int, end: int) -> bool:
    return any(x.get("type")==kind and _overlap(int(x.get("start",0)),int(x.get("end",0)),start,end) for x in pii)


def _owned_assignments(text: str, start: int=0, end: int | None=None) -> list[dict]:
    end=len(text) if end is None else end
    out=[]
    window=text[start:end]
    for kind,pat in _OWNED_RECOVERY_STARTS:
        for m in pat.finditer(window):
            abs_start=start+m.start(); abs_end=start+m.end()
            # Do not reinterpret the word "card" inside a CVV/CVC assignment as a
            # separate card-number ownership field ("CVV on my card is 391").
            if kind=="CARD" and re.search(r"(?i)\b(?:cvv|cvc|card\s+security\s+code|verification\s+value)\b",text[max(start,abs_start-36):abs_start]):
                continue
            out.append({"type":kind,"label_start":abs_start,"value_start":abs_end})
    return sorted(out,key=lambda x:x["label_start"])


def _bounded_owned_char_end(text: str, value_start: int, sentence_end: int) -> int:
    end=sentence_end
    for pat in (_RECOVERY_NEXT_FIELD,_RECOVERY_HARD_BOUNDARY):
        m=pat.search(text,value_start,end)
        if m:
            end=min(end,m.start())
    return max(value_start,end)


def _append_plan(plans: list[dict], asr: dict, *, kind: str, start: int, end: int, reason: str, source: str) -> None:
    if end<=start:
        return
    timing=_span_time(asr,start,end)
    if not timing:
        return
    ts,te,conf=timing
    max_s=_MAX_UNCERTAIN_SECONDS.get(kind,3.0)
    te=min(te,ts+max_s+0.7)
    candidate={
        "type":kind,"char_start":start,"char_end":end,
        "time_start":max(0.0,ts-0.20),"time_end":max(ts,te+0.20),
        "primary_alignment_confidence":round(conf,4),
        "reason":reason,"source":source,
    }
    # De-duplicate same-type windows that substantially overlap.
    for p in plans:
        if p["type"]==kind and _overlap(int(p["char_start"]),int(p["char_end"]),start,end):
            if (end-start)>(int(p["char_end"])-int(p["char_start"])):
                p.update(candidate)
            return
    plans.append(candidate)


def plan_privacy_recovery(asr: dict, accepted_pii: Iterable[dict], *, max_windows: int=4) -> list[dict]:
    """Plan bounded recovery windows for unresolved sensitive ownership.

    v4.12 retains two recovery lanes:
    1) a one-shot follow-up expectation ("Please enter your CVV. Mine is ..."); and
    2) an explicit owned field whose primary ASR value is malformed ("my name is 2210").

    Plans never expose or invent the value; they only select a small audio window for a
    second decode or, on failure, a conservative audio guard.
    """
    text=str(asr.get("text","") or "")
    spans=_sentence_spans(text)
    pii=list(accepted_pii or [])
    plans=[]

    # Lane A: one-shot previous-sentence expectation. If the introducing sentence
    # already contains an accepted value, or is itself a personal assignment, the
    # expectation is satisfied/owned there and is never propagated.
    for i,(s0,s1) in enumerate(spans[:-1]):
        previous=text[s0:s1]
        expected=infer_expected_type(previous)
        if not expected:
            continue
        if _accepted_in_span(pii,expected,s0,s1):
            continue
        if any(x["type"]==expected for x in _owned_assignments(text,s0,s1)):
            continue
        n0,n1=spans[i+1]
        response=text[n0:n1]
        if not _RECOVERY_RESPONSE.search(response) or _NO_VALUE_RESPONSE.search(response):
            continue
        response_end=_bounded_owned_char_end(text,n0,n1)
        if _accepted_in_span(pii,expected,n0,response_end):
            continue
        _append_plan(plans,asr,kind=expected,start=n0,end=response_end,
                     reason="one_shot_expected_field_missing",source="followup_expectation")

    # Lane B: same-sentence explicit ownership with no accepted entity. This is the
    # main protection for ASR corruption such as "my registered name is 2210".
    for assignment in _owned_assignments(text):
        kind=assignment["type"]
        value_start=int(assignment["value_start"])
        sent=None; sent_i=None
        for i,(a,b) in enumerate(spans):
            if a<=value_start<=b:
                sent=(a,b); sent_i=i; break
        if not sent:
            continue
        s0,s1=sent
        end=_bounded_owned_char_end(text,value_start,s1)
        tail=text[value_start:end]
        if _NO_VALUE_RESPONSE.search(tail):
            continue
        if _accepted_in_span(pii,kind,value_start,end):
            continue

        reason="owned_field_unresolved_primary_asr"
        # IFSC/PAN/passport/DL dictation is especially prone to Whisper inserting a
        # period mid-value. Permit exactly one short phonetic/alphanumeric continuation
        # when no new field begins.
        if kind in {"IFSC","PAN","PASSPORT","DRIVING_LICENSE"} and sent_i is not None and sent_i+1<len(spans):
            n0,n1=spans[sent_i+1]
            next_text=text[n0:n1]
            if _PHONETIC_CONTINUATION.search(next_text) and not _RECOVERY_NEXT_FIELD.match(next_text):
                extended=_bounded_owned_char_end(text,n0,n1)
                end=max(end,extended)
                reason="owned_field_cross_punctuation_unresolved"
        _append_plan(plans,asr,kind=kind,start=value_start,end=end,reason=reason,source="owned_field")

    # Prioritize smaller/high-risk windows and respect the configured inference cap.
    risk={"CVV":0,"OTP":1,"PAN":2,"IFSC":2,"AADHAAR":2,"CARD":2,"ACCOUNT_NUMBER":2,
          "PHONE":3,"EMAIL":3,"UPI":3,"PASSPORT":3,"DRIVING_LICENSE":3,"NAME":4,
          "DOB":4,"ADDRESS":5,"PINCODE":4,"VOTER_ID":4}
    plans.sort(key=lambda p:(risk.get(p["type"],9),float(p["time_end"])-float(p["time_start"]),p["time_start"]))
    return plans[:max(0,max_windows)]


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



# v5.2: when a sensitive value was already conservatively audio-guarded, a nearby
# explicit self-repair can carry the field type forward without another ASR decode.
# The new fragment must independently pass the normal type validator under a synthetic
# ownership prefix, so this is not a generic short-number detector.
_POST_GUARD_REPAIR=re.compile(
    r"(?ix)\b(?:i\s+(?:made\s+)?a?\s*mistake|that(?:'s|\s+is)\s+(?:wrong|incorrect)|"
    r"sorry|correction|i\s+mean|actually|make\s+that|let\s+me\s+correct(?:\s+that)?)\b"
)
_POST_GUARD_BLOCK=re.compile(
    r"(?i)\b(?:reference|ticket|order|transaction|complaint|experiment|invoice|page|"
    r"example|sample|documentation|manual|test\s+(?:case|value|data))\b"
)
_POST_GUARD_PARTIAL_TYPES={"PHONE","CARD","ACCOUNT_NUMBER","OTP","CVV","PINCODE","AADHAAR"}
_POST_GUARD_PARTIAL=re.compile(
    r"(?ix)\b(?:"
    r"(?:(?:change|replace|correct|make)\s+(?:the\s+)?)?"
    r"(?:last|final)\s+(?:(one|two|three|four|[1-4])\s+)?digit(?:s)?\s+"
    r"(?:is|are|to|with|as|should\s+be|must\s+be)\s+"
    r")"
    r"((?:[0-9](?:[\s-]*[0-9]){0,3})|"
    r"(?:(?:zero|oh|o|one|two|three|four|five|six|seven|eight|nine)"
    r"(?:[\s-]+(?:zero|oh|o|one|two|three|four|five|six|seven|eight|nine)){0,3}))\b"
)
_POST_GUARD_COUNT={"one":1,"two":2,"three":3,"four":4}
_POST_GUARD_DIGIT={"zero":"0","oh":"0","o":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9"}


def _post_guard_replacement_digits(raw: str) -> str:
    d=re.sub(r"\D","",str(raw or ""))
    if d: return d
    out=[]
    for tok in re.findall(r"[A-Za-z]+",str(raw or "").casefold()):
        if tok not in _POST_GUARD_DIGIT: return ""
        out.append(_POST_GUARD_DIGIT[tok])
    return "".join(out)

_OWNERSHIP_PREFIX={
    "NAME":"My name is ","PHONE":"My phone number is ","EMAIL":"My email is ",
    "PAN":"My PAN is ","AADHAAR":"My Aadhaar number is ","IFSC":"My IFSC is ",
    "UPI":"My UPI ID is ","CARD":"My card number is ","ACCOUNT_NUMBER":"My account number is ",
    "OTP":"My OTP is ","CVV":"My CVV is ","PINCODE":"My PIN code is ",
    "DOB":"My date of birth is ","PASSPORT":"My passport number is ",
    "VOTER_ID":"My voter ID is ","DRIVING_LICENSE":"My driving licence number is ",
    "ADDRESS":"My address is ","NAME":"My full name is ",
}


def _validated_primary_correction(text: str, kind: str, start: int, limit: int=64) -> dict | None:
    prefix=_OWNERSHIP_PREFIX.get(kind)
    if not prefix:
        return None
    end=min(len(text),start+max(16,int(limit)))
    tail=text[start:end]
    # Stop before another field or public/operational role.
    hard=_RECOVERY_NEXT_FIELD.search(tail)
    if hard: tail=tail[:hard.start()]
    if _POST_GUARD_BLOCK.search(tail):
        return None
    lead=re.match(r"[\s,.;:…\-–—]*",tail)
    lead_n=lead.end() if lead else 0
    body=tail[lead_n:]
    if not body:
        return None
    synthetic=prefix+body
    matches=[x for x in detect_pii(synthetic,min_confidence=0.72,use_semantic=False,use_ner=False) if x.get("type")==kind]
    if not matches:
        return None
    item=min(matches,key=lambda x:int(x.get("start",0)))
    rs=start+lead_n+max(0,int(item.get("start",0))-len(prefix))
    re_=start+lead_n+max(0,int(item.get("end",0))-len(prefix))
    if re_<=rs or rs<start or re_>end:
        return None
    return {"type":kind,"char_start":rs,"char_end":re_}


def _post_guard_correction_guards(asr: dict, accepted_pii: list[dict], audit: list[dict]) -> list[dict]:
    text=str(asr.get("text","") or "")
    out=[]
    for row in list(audit):
        if row.get("status")!="conservative_audio_guard" or row.get("source")=="correction_continuation":
            continue
        kind=str(row.get("type",""))
        base_end=int(row.get("char_end",0) or 0)
        if base_end<=0 or base_end>=len(text):
            continue
        window=text[base_end:min(len(text),base_end+180)]
        # A deterministic partial suffix edit is itself sensitive even when the
        # original guarded value could not be reconstructed. Protect only the spoken
        # replacement fragment; never synthesize a full identifier.
        if kind in _POST_GUARD_PARTIAL_TYPES and not _POST_GUARD_BLOCK.search(window):
            pm=_POST_GUARD_PARTIAL.search(window)
            if pm:
                repl=_post_guard_replacement_digits(pm.group(2))
                raw_count=pm.group(1)
                count=(int(raw_count) if raw_count and raw_count.isdigit() else _POST_GUARD_COUNT.get(str(raw_count or "").casefold()))
                if count is None and len(repl)==1: count=1
                if repl and count==len(repl) and 1<=count<=4:
                    cs=base_end+pm.start(2); ce=base_end+pm.end(2)
                    timing=_span_time(asr,cs,ce)
                    if timing and timing[2]>=0.55:
                        ts,te,conf=timing
                        out.append({
                            "type":kind,"start":max(0.0,ts-0.10),"end":max(ts,te+0.12),"confidence":0.90,
                            "alignment_method":"post_guard_partial_correction_alignment",
                            "alignment_confidence":round(float(conf),4),"recovery_status":"conservative_audio_guard",
                            "char_start":cs,"char_end":ce,"source":"correction_continuation",
                            "reason":"partial_edit_after_guarded_value",
                        })
                        continue
        repair=_POST_GUARD_REPAIR.search(window)
        if not repair:
            continue
        # Do not carry ownership through a new field or operational/example context.
        before=window[:repair.start()]
        if _RECOVERY_NEXT_FIELD.search(before) or _POST_GUARD_BLOCK.search(before):
            continue
        cand_start=base_end+repair.end()
        candidate=_validated_primary_correction(text,kind,cand_start,limit=72)
        if not candidate:
            continue
        cs,ce=int(candidate["char_start"]),int(candidate["char_end"])
        if _accepted_in_span(accepted_pii,kind,cs,ce):
            continue
        # The repair must be close. More than a few lexical words after the repair cue
        # indicates a new statement rather than a direct correction.
        bridge=text[base_end+repair.end():cs]
        if len(re.findall(r"[A-Za-z]+",bridge))>3 or _POST_GUARD_BLOCK.search(bridge) or _RECOVERY_NEXT_FIELD.search(bridge):
            continue
        timing=_span_time(asr,cs,ce)
        if not timing:
            continue
        ts,te,conf=timing
        if conf < 0.55:
            # Low confidence is still protected by the original field guard, but we do
            # not create a second possibly unrelated transcript replacement.
            continue
        out.append({
            "type":kind,"start":max(0.0,ts-0.10),"end":max(ts,te+0.12),"confidence":0.90,
            "alignment_method":"post_guard_correction_primary_alignment",
            "alignment_confidence":round(float(conf),4),"recovery_status":"conservative_audio_guard",
            "char_start":cs,"char_end":ce,"source":"correction_continuation",
            "reason":"validated_value_after_guarded_self_repair",
        })
    return out


def recover_privacy_audio_intervals(
    audio: np.ndarray, sr: int, asr: dict, accepted_pii: Iterable[dict],
    *, model_name: str | None=None, max_windows: int=4, conservative_on_failure: bool=True,
) -> dict:
    """Return audio-only redaction intervals plus privacy-safe recovery telemetry."""
    total0=time.perf_counter()
    accepted=list(accepted_pii or [])
    plans=plan_privacy_recovery(asr,accepted,max_windows=max_windows)
    intervals=[]; audit=[]
    telemetry={
        "windows_planned":len(plans),"windows_attempted":0,"recovered":0,
        "guarded":0,"unresolved":0,"direct_correction_guards":0,
        "decode_inference_ms":0.0,"total_ms":0.0,"by_type":{},
    }
    def _type_stats(kind: str) -> dict:
        return telemetry["by_type"].setdefault(kind,{
            "planned":0,"attempted":0,"recovered":0,"guarded":0,"unresolved":0,"decode_inference_ms":0.0
        })
    for p in plans:
        _type_stats(str(p.get("type","PII")))["planned"]+=1
    for plan in plans:
        kind=str(plan.get("type","PII"))
        tstats=_type_stats(kind)
        telemetry["windows_attempted"]+=1
        tstats["attempted"]+=1
        dt=time.perf_counter()
        alt=_decode_window(audio,sr,plan,model_name)
        decode_ms=(time.perf_counter()-dt)*1000.0
        telemetry["decode_inference_ms"]+=decode_ms
        tstats["decode_inference_ms"]+=decode_ms
        confirmed=_confirm_expected(plan["type"],alt)
        if confirmed:
            intervals.append(confirmed)
            telemetry["recovered"]+=1
            tstats["recovered"]+=1
            audit.append({
                "type":plan["type"],"status":"confirmed_alternate_decode",
                "source":plan.get("source"),"reason":plan.get("reason"),
                "time_start":round(float(plan["time_start"]),3),"time_end":round(float(plan["time_end"]),3),
                "primary_alignment_confidence":plan["primary_alignment_confidence"],
                "char_start":int(plan.get("char_start",0)),"char_end":int(plan.get("char_end",0)),
                "alternate_asr_confidence":confirmed.get("alignment_confidence"),
                "decode_ms":round(decode_ms,3),"alternate_text_present":bool(alt.get("text")),
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
                "char_start":int(plan.get("char_start",0)),"char_end":int(plan.get("char_end",0)),
                "source":plan.get("source"),"reason":plan.get("reason"),
            })
            status="conservative_audio_guard"
            telemetry["guarded"]+=1
            tstats["guarded"]+=1
        else:
            status="unresolved"
            telemetry["unresolved"]+=1
            tstats["unresolved"]+=1
        audit.append({
            "type":plan["type"],"status":status,"source":plan.get("source"),"reason":plan.get("reason"),
            "time_start":round(float(plan["time_start"]),3),"time_end":round(float(plan["time_end"]),3),
            "primary_alignment_confidence":plan["primary_alignment_confidence"],
            "char_start":int(plan.get("char_start",0)),"char_end":int(plan.get("char_end",0)),
            "alternate_decode_error":bool(alt.get("error")),"decode_ms":round(decode_ms,3),
            "alternate_text_present":bool(alt.get("text")),
        })
    # v5.2: if a guarded field is explicitly corrected in the next short utterance,
    # protect the corrected fragment directly from primary-ASR timing. This costs no
    # additional Whisper inference and produces a second [TYPE AUDIO PROTECTED] marker.
    direct=_post_guard_correction_guards(asr,accepted,audit) if conservative_on_failure else []
    for g in direct:
        intervals.append(g)
        kind=str(g.get("type","PII")); tstats=_type_stats(kind)
        telemetry["guarded"]+=1; telemetry["direct_correction_guards"]+=1; tstats["guarded"]+=1
        audit.append({
            "type":kind,"status":"conservative_audio_guard","source":"correction_continuation",
            "reason":"validated_value_after_guarded_self_repair",
            "time_start":round(float(g["start"]),3),"time_end":round(float(g["end"]),3),
            "primary_alignment_confidence":g.get("alignment_confidence"),
            "char_start":int(g.get("char_start",0)),"char_end":int(g.get("char_end",0)),
            "decode_ms":0.0,"alternate_text_present":False,
        })

    telemetry["decode_inference_ms"]=round(float(telemetry["decode_inference_ms"]),3)
    for stats in telemetry.get("by_type",{}).values():
        stats["decode_inference_ms"]=round(float(stats.get("decode_inference_ms",0.0)),3)
    telemetry["total_ms"]=round((time.perf_counter()-total0)*1000.0,3)
    return {"plans":len(plans),"intervals":intervals,"audit":audit,"telemetry":telemetry}

