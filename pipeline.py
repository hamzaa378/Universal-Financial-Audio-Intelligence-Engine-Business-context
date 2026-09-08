from __future__ import annotations

from copy import deepcopy
import re
import time

from ingest.audio_loader import load_audio
from ingest.tamper_detection import detect_tamper
from asr.fintech_asr import transcribe
from asr.privacy_recovery import recover_privacy_audio_intervals
from nlp.language import annotate_segment_languages
from nlp.pii import public_pii_metadata, semantic_status, ner_status, mask_pii, detect_pii
from nlp.profanity import public_profanity_metadata
from nlp.sensitive_financial import public_sensitive_id_metadata
from nlp.privacy import protect_text
from nlp.intent import classify_intent
from nlp.entities import extract_entities
from nlp.obligation import detect_obligations
from nlp.regulatory import check_regulatory
from analysis_ai.acoustic import analyze_acoustics
from analysis_ai.diarization import diarize, assign_speakers
from decision_ai.utterance_ai import analyze_utterance
from audio_privacy import create_protected_audio
from nlp.token_alignment import attach_entity_tokens
from nlp.corrections import resolve_timed_corrections, resolve_partial_corrections
from config import SETTINGS
from privacy_debug import write_privacy_debug_bundle


def _trust(asr_conf, quality, intent_conf, pii_conf=1.0):
    return round(max(0,min(1,0.48*asr_conf+0.22*quality+0.18*intent_conf+0.12*pii_conf)),3)




def _correction_candidate_validator(kind: str, tail: str) -> list[dict]:
    """Validate a possible restarted value with the normal v5 privacy validators.

    The synthetic ownership prefix is never exposed and never becomes part of the
    returned span. This helper is used only after an already-accepted owned PII value
    and only for the timing-based self-correction resolver.
    """
    prefixes={
        "NAME":"My name is ","PHONE":"My phone number is ","EMAIL":"My email is ",
        "PAN":"My PAN is ","AADHAAR":"My Aadhaar number is ","IFSC":"My IFSC is ",
        "UPI":"My UPI ID is ","CARD":"My card number is ","ACCOUNT_NUMBER":"My account number is ",
        "OTP":"My OTP is ","CVV":"My CVV is ","PINCODE":"My PIN code is ",
        "DOB":"My date of birth is ","PASSPORT":"My passport number is ",
        "VOTER_ID":"My voter ID is ","DRIVING_LICENSE":"My driving licence number is ",
        "ADDRESS":"My address is ","PASSWORD":"My password is ","USERNAME":"My username is ",
        "API_KEY":"My API key is ","AUTH_TOKEN":"My auth token is ",
    }
    prefix=prefixes.get(kind)
    if not prefix:
        return []
    lead_match=re.match(r"[\s,.;:…\-–—]*",tail or "")
    lead=lead_match.end() if lead_match else 0
    body=(tail or "")[lead:]
    if not body:
        return []
    synthetic=prefix+body
    out=[]
    for e in detect_pii(synthetic,min_confidence=0.72,use_semantic=False,use_ner=False):
        if e.get("type")!=kind:
            continue
        rs=int(e.get("start",0))-len(prefix)+lead
        re_=int(e.get("end",0))-len(prefix)+lead
        if rs<lead or re_<=rs or re_>len(tail):
            continue
        x=dict(e); x["start"]=rs; x["end"]=re_; x["value"]=tail[rs:re_]
        out.append(x)
    return out

def _safe_financial_entities(entities: list[dict], include_raw: bool=False) -> list[dict]:
    if include_raw:
        return entities
    out=[]
    for e in entities:
        x=dict(e)
        if e.get("type") in {"FINANCIAL_ACCOUNT_REF","ACCOUNT_NUMBER","CARD","UPI","PAN","AADHAAR"}:
            x["value"]=f"[{e.get('type','ENTITY')} REDACTED]"
        out.append(x)
    return out


def _policy_kwargs(privacy_profile,use_semantic_ai,use_ner_ai,mask_mode,mask_types,profanity_enabled,financial_ids_enabled):
    return dict(
        privacy_profile=privacy_profile,
        use_semantic=use_semantic_ai,
        use_ner=use_ner_ai,
        pii_mode=mask_mode,
        mask_types=set(mask_types) if mask_types else None,
        profanity_enabled=profanity_enabled,
        financial_ids_enabled=financial_ids_enabled,
    )


def _audio_guard_replacements(text: str, recovery: dict) -> list[dict]:
    """Build privacy-safe transcript markers for conservative audio guards.

    The marker never contains alternate ASR text or a guessed identifier. It only
    communicates that the corresponding audio interval was protected.
    """
    out=[]
    for row in (recovery or {}).get("audit",[]) or []:
        if row.get("status")!="conservative_audio_guard":
            continue
        try:
            start=max(0,int(row.get("char_start",0)))
            end=min(len(text),int(row.get("char_end",start)))
        except Exception:
            continue
        if end<=start:
            continue
        kind=str(row.get("type","PII"))
        raw=text[start:end]
        stripped=raw.strip()
        # Preserve a natural, non-sensitive ownership phrase when possible.
        if row.get("source")=="followup_expectation" and re.match(r"(?i)^mine\b",stripped):
            repl=f"Mine is [{kind} AUDIO PROTECTED]."
            # When the bounded response ends directly before a comma introducing the
            # next field, consume only that comma so we do not emit '. ,'.
            if end < len(text) and text[end]==',':
                end+=1
        elif re.match(r"(?i)^i\s+(?:received|got)\b",stripped):
            verb="received" if re.match(r"(?i)^i\s+received\b",stripped) else "got"
            repl=f"I {verb} [{kind} AUDIO PROTECTED]."
            if end < len(text) and text[end]==',':
                end+=1
        else:
            repl=f"[{kind} AUDIO PROTECTED]"
        out.append({"start":start,"end":end,"replacement":repl,"type":kind})
    return out


def _render_safe_text(text: str, internal_privacy: dict, *, mask_mode: str="full", audio_guards: list[dict] | None=None) -> str:
    """Render one authoritative safe transcript from already-decided spans.

    This prevents a second detector pass and lets audio-only recovery markers coexist
    with normal PII/financial/profanity masking without exposing alternate ASR text.
    """
    replacements=[]
    for e in internal_privacy.get("pii",[]):
        s=max(0,int(e.get("start",0))); en=min(len(text),int(e.get("end",s)))
        if en<=s: continue
        local={**e,"start":0,"end":en-s,"value":text[s:en]}
        repl=mask_pii(text[s:en],[local],mode=mask_mode)
        replacements.append((s,en,repl,30))
    for e in internal_privacy.get("financial_ids",[]):
        s=max(0,int(e.get("start",0))); en=min(len(text),int(e.get("end",s)))
        if en>s: replacements.append((s,en,f"[{e.get('type','SENSITIVE_ID')} REDACTED]",20))
    for e in internal_privacy.get("profanity",[]):
        s=max(0,int(e.get("start",0))); en=min(len(text),int(e.get("end",s)))
        if en>s: replacements.append((s,en,"[BLEEP]",10))
    for g in audio_guards or []:
        s=max(0,int(g.get("start",0))); en=min(len(text),int(g.get("end",s)))
        if en>s: replacements.append((s,en,str(g.get("replacement","[AUDIO PROTECTED]")),40))
    replacements.sort(key=lambda x:(x[3],x[1]-x[0]),reverse=True)
    kept=[]
    for r in replacements:
        if not any(r[0] < k[1] and k[0] < r[1] for k in kept):
            kept.append(r)
    safe=text
    for st,en,repl,_ in sorted(kept,key=lambda x:x[0],reverse=True):
        safe=safe[:st]+repl+safe[en:]
    return safe


def _safe_transcription(asr: dict, safe_text: str, internal_privacy: dict, *, mask_mode: str="full", audio_guards: list[dict] | None=None) -> dict:
    """Build safe transcript views by reusing already-computed privacy spans.

    v4.3 re-ran PII/semantic detection for every ASR segment. v4.4 instead maps the
    single full-transcript decision set back onto each segment, eliminating redundant
    neural inference and guaranteeing that accepted full-text decisions are not lost.
    """
    pii=list(internal_privacy.get("pii",[]))
    financial=list(internal_privacy.get("financial_ids",[]))
    profanity=list(internal_privacy.get("profanity",[]))

    out={
        "text":safe_text,
        "language":asr.get("language"),
        "language_probability":asr.get("language_probability"),
        "confidence":asr.get("confidence"),
        "model":asr.get("model"),
        "backend":deepcopy(asr.get("backend",{})),
        "segments":[],"words":[],
    }
    cursor=0
    global_token=0
    segments=asr.get("segments",[])
    for seg_i,seg in enumerate(segments):
        seg_text=str(seg.get("text",""))
        gs=cursor; ge=gs+len(seg_text)
        replacements=[]
        for e in pii:
            if e["start"] >= ge or e["end"] <= gs: continue
            ls=max(0,e["start"]-gs); le=min(len(seg_text),e["end"]-gs)
            local={**e,"start":ls,"end":le,"value":seg_text[ls:le]}
            repl=mask_pii(seg_text[ls:le],[{**local,"start":0,"end":le-ls}],mode=mask_mode)
            replacements.append((ls,le,repl,3))
        for e in financial:
            if e["start"] >= ge or e["end"] <= gs: continue
            ls=max(0,e["start"]-gs); le=min(len(seg_text),e["end"]-gs)
            replacements.append((ls,le,f"[{e['type']} REDACTED]",2))
        for e in profanity:
            if e["start"] >= ge or e["end"] <= gs: continue
            ls=max(0,e["start"]-gs); le=min(len(seg_text),e["end"]-gs)
            replacements.append((ls,le,"[BLEEP]",1))
        for g in audio_guards or []:
            if g["start"] >= ge or g["end"] <= gs: continue
            ls=max(0,g["start"]-gs); le=min(len(seg_text),g["end"]-gs)
            # A guard can rarely straddle Whisper segments. Emit the marker only in
            # the segment that owns its start and blank any continuation portion.
            repl=g["replacement"] if gs <= g["start"] < ge else ""
            replacements.append((ls,le,repl,4))
        replacements.sort(key=lambda x:(x[3],x[1]-x[0]),reverse=True)
        kept=[]
        for r in replacements:
            if not any(r[0] < k[1] and k[0] < r[1] for k in kept):
                kept.append(r)
        safe_seg=seg_text
        for ls,le,repl,_ in sorted(kept,key=lambda x:x[0],reverse=True):
            safe_seg=safe_seg[:ls]+repl+safe_seg[le:]

        item={k:deepcopy(v) for k,v in seg.items() if k!="words"}
        item["text"]=safe_seg
        item["words"]=[]
        for local_i,w in enumerate(seg.get("words",[])):
            item["words"].append({
                "token_id":global_token,"segment_token_index":local_i,
                "start":w.get("start"),"end":w.get("end"),"confidence":w.get("confidence")
            })
            global_token+=1
        out["segments"].append(item)
        cursor=ge+(1 if seg_i < len(segments)-1 else 0)
    out["words"]=[
        {"token_id":i,"start":w.get("start"),"end":w.get("end"),"confidence":w.get("confidence")}
        for i,w in enumerate(asr.get("words",[]))
    ]
    return out

def analyze_text(
    text: str,
    *,
    privacy_profile: str="balanced",
    use_semantic_ai: bool=False,
    use_ner_ai: bool=False,
    mask_mode: str="full",
    mask_types=None,
    profanity_enabled: bool=True,
    financial_ids_enabled: bool=False,
    include_internal: bool=False,
) -> dict:
    """Optimized text-only path.

    Semantic intent and abusive-tone decisions share one batched embedding pass. PII
    candidates that need AI review are also evaluated in a batch inside detect_pii().
    """
    t0=time.perf_counter()
    policy=_policy_kwargs(
        privacy_profile,use_semantic_ai,use_ner_ai,mask_mode,mask_types,profanity_enabled,financial_ids_enabled
    )
    protected=protect_text(text,**policy)
    t_priv=time.perf_counter()

    semantic_bundle=None
    if use_semantic_ai:
        try:
            semantic_bundle=analyze_utterance(text)
        except Exception:
            semantic_bundle=None
    intent=classify_intent(text,use_semantic=use_semantic_ai,semantic_result=semantic_bundle)
    abuse=(semantic_bundle or {}).get("abusive_tone") or {
        "available":False,"score":0.0,"level":"not_run" if not use_semantic_ai else "unavailable","method":"disabled_or_fallback"
    }
    entities=extract_entities(text)
    obligations=detect_obligations(text)
    regulatory=check_regulatory(text)
    t_end=time.perf_counter()

    pii_conf=sum(x["confidence"] for x in protected["pii"])/len(protected["pii"]) if protected["pii"] else 1.0
    result={
        "safe_text":protected["text"],
        "pii":public_pii_metadata(protected["pii"]),
        "sensitive_financial_ids":public_sensitive_id_metadata(protected.get("financial_ids",[])),
        "profanity":public_profanity_metadata(protected["profanity"]),
        "intent":intent,
        "financial_entities":entities,
        "obligations":obligations,
        "regulatory":regulatory,
        "abusive_tone":abuse,
        "pii_count":len(protected["pii"]),
        "sensitive_financial_id_count":len(protected.get("financial_ids",[])),
        "profanity_count":len(protected["profanity"]),
        "pii_mean_confidence":round(pii_conf,4),
        "privacy_profile":privacy_profile,
        "semantic_ai":semantic_status() if use_semantic_ai else {"available":False,"backend":"disabled","model":None,"provider":None,"error":None},
        "ner_ai":ner_status() if use_ner_ai else {"available":False,"backend":"disabled","provider":None,"error":None},
        "timing_ms":{
            "privacy":round((t_priv-t0)*1000,4),
            "understanding":round((t_end-t_priv)*1000,4),
            "total":round((t_end-t0)*1000,4),
        },
    }
    if include_internal:
        result["_internal_privacy"]=protected
    return result


def run_pipeline(
    audio_path,
    model_name=None,
    include_raw=False,
    *,
    privacy_profile="balanced",
    use_semantic_ai=False,
    use_ner_ai=False,
    mask_mode="full",
    mask_types=None,
    profanity_enabled=True,
    financial_ids_enabled=False,
    create_audio_output=False,
    audio_redaction_method="beep",
    asr_speed_mode="balanced",
    enable_diarization=False,
    asr_privacy_recovery: bool | None = None,
):
    timing={}; total0=time.perf_counter()

    t=time.perf_counter()
    audio,sr=load_audio(audio_path)
    timing["audio_load"]=(time.perf_counter()-t)*1000

    t=time.perf_counter()
    ta=time.perf_counter()
    acoustic=analyze_acoustics(audio,sr)
    timing["acoustic_analysis"]=(time.perf_counter()-ta)*1000
    tt=time.perf_counter()
    tamper=detect_tamper(audio,sr)
    timing["tamper_analysis"]=(time.perf_counter()-tt)*1000
    timing["signal_analysis"]=(time.perf_counter()-t)*1000

    # v4.4 passes the already-decoded 16 kHz waveform directly to Faster-Whisper.
    t=time.perf_counter()
    asr=transcribe(audio,model_name,speed_mode=asr_speed_mode)
    timing["asr"]=(time.perf_counter()-t)*1000
    asr["segments"]=annotate_segment_languages(asr["segments"])

    t=time.perf_counter()
    dia=diarize(audio_path,enabled=enable_diarization)
    timing["diarization"]=(time.perf_counter()-t)*1000
    if dia.get("turns"):
        asr["segments"]=assign_speakers(asr["segments"],dia["turns"])

    text=asr["text"]
    t=time.perf_counter()
    text_analysis=analyze_text(
        text,
        privacy_profile=privacy_profile,
        use_semantic_ai=use_semantic_ai,
        use_ner_ai=use_ner_ai,
        mask_mode=mask_mode,
        mask_types=mask_types,
        profanity_enabled=profanity_enabled,
        financial_ids_enabled=financial_ids_enabled,
        include_internal=True,
    )
    internal_privacy=text_analysis.pop("_internal_privacy")
    # v4.5: bind accepted entities to Whisper token IDs/timestamps before audio masking.
    internal_privacy["pii"]=attach_entity_tokens(asr,list(internal_privacy.get("pii",[])))
    internal_privacy["financial_ids"]=attach_entity_tokens(asr,list(internal_privacy.get("financial_ids",[])))
    internal_privacy["profanity"]=attach_entity_tokens(asr,list(internal_privacy.get("profanity",[])))

    # v5.0 correction resolver: after Whisper token/timing attachment, a later
    # same-type value can supersede an earlier mistaken value even when the speaker
    # never says "sorry", "correction" or "I mean".  The resolver can only choose
    # between PII candidates already accepted by the normal detector, so it does not
    # create new false-positive candidates.  Parallel fields (alternate/secondary/
    # primary, lists joined by and/or, etc.) are explicitly excluded.
    internal_privacy["pii"],correction_audit=resolve_timed_corrections(
        text,internal_privacy["pii"],asr,candidate_validator=_correction_candidate_validator
    )
    # v5.2: deterministic partial self-repairs (for example "last digit is one")
    # are resolved only after an already-accepted owned numeric PII value. The
    # corrected full identifier is never reconstructed or stored; only the spoken
    # replacement fragment is protected.
    internal_privacy["pii"],partial_correction_audit=resolve_partial_corrections(
        text,internal_privacy["pii"],asr
    )
    if partial_correction_audit:
        correction_audit.extend(partial_correction_audit)
    if correction_audit:
        # The earlier mistaken value must become visible again in the safe transcript
        # while the final committed value remains masked. Re-render from the same
        # already-decided spans; no detector or threshold is re-run.
        text_analysis["safe_text"]=_render_safe_text(text,internal_privacy,mask_mode=mask_mode)
        text_analysis["pii_count"]=len(internal_privacy["pii"])
        vals=[float(x.get("confidence",0.0)) for x in internal_privacy["pii"]]
        text_analysis["pii_mean_confidence"]=round(sum(vals)/len(vals),4) if vals else 1.0
    text_analysis["pii"]=public_pii_metadata(internal_privacy["pii"])
    timing["privacy_and_nlp"]=(time.perf_counter()-t)*1000

    protected_audio=None
    audio_guards=[]
    privacy_recovery={"enabled":False,"plans":0,"audit":[],"audio_interval_count":0,"telemetry":{"windows_planned":0,"windows_attempted":0,"recovered":0,"guarded":0,"unresolved":0,"decode_inference_ms":0.0,"total_ms":0.0,"by_type":{}}}
    if create_audio_output:
        t=time.perf_counter()
        recovery_enabled=SETTINGS.privacy_redecode if asr_privacy_recovery is None else bool(asr_privacy_recovery)
        recovery_intervals=[]
        if recovery_enabled:
            rt=time.perf_counter()
            try:
                rr=recover_privacy_audio_intervals(
                    audio,sr,asr,internal_privacy.get("pii",[]),model_name=model_name,
                    max_windows=max(0,int(SETTINGS.privacy_redecode_max_windows)),
                    conservative_on_failure=bool(SETTINGS.privacy_conservative_audio_guard),
                )
                recovery_intervals=list(rr.get("intervals",[]))
                privacy_recovery={
                    "enabled":True,"plans":int(rr.get("plans",0)),
                    "audit":list(rr.get("audit",[])),
                    "audio_interval_count":len(recovery_intervals),
                    "telemetry":dict(rr.get("telemetry",{})),
                }
            except Exception as exc:
                # Privacy recovery is an additional guard. A failure must never discard
                # the normal detector/audio-redaction result.
                privacy_recovery={
                    "enabled":True,"plans":0,"audit":[],"audio_interval_count":0,
                    "telemetry":{"windows_planned":0,"windows_attempted":0,"recovered":0,"guarded":0,"unresolved":0,"decode_inference_ms":0.0,"total_ms":0.0,"by_type":{}},
                    "error":f"{type(exc).__name__}: {exc}",
                }
            timing["privacy_redecode"]=(time.perf_counter()-rt)*1000
            audio_guards=_audio_guard_replacements(text,privacy_recovery)
            if audio_guards:
                text_analysis["safe_text"]=_render_safe_text(text,internal_privacy,mask_mode=mask_mode,audio_guards=audio_guards)
        all_sensitive=list(internal_privacy.get("pii",[]))+list(internal_privacy.get("financial_ids",[]))
        protected_audio=create_protected_audio(
            audio_path,asr,all_sensitive,internal_privacy.get("profanity",[]),
            method=audio_redaction_method,preloaded_audio=(audio,sr),extra_intervals=recovery_intervals,
        )
        timing["audio_redaction"]=(time.perf_counter()-t)*1000

    result={
        "audio":{
            "quality":acoustic["quality"],
            "tamper_screen":tamper,
            "diarization":dia,
        },
        "transcription":asr if include_raw else _safe_transcription(asr,text_analysis["safe_text"],internal_privacy,mask_mode=mask_mode,audio_guards=audio_guards),
        "understanding":{
            "intent":text_analysis["intent"],
            "financial_entities":_safe_financial_entities(text_analysis["financial_entities"],include_raw=include_raw),
            "obligations":text_analysis["obligations"],
            "pii":text_analysis["pii"],
            "sensitive_financial_ids":text_analysis["sensitive_financial_ids"],
            "profanity":text_analysis["profanity"],
            "regulatory":text_analysis["regulatory"],
            "stress_markers":acoustic["stress_markers"],
            "abusive_tone":text_analysis["abusive_tone"],
        },
        "privacy":{
            "safe_transcript":text_analysis["safe_text"],
            "protected_audio":protected_audio,
            "pii_count":text_analysis["pii_count"],
            "sensitive_financial_id_count":text_analysis["sensitive_financial_id_count"],
            "profanity_count":text_analysis["profanity_count"],
            "profile":privacy_profile,
            "mask_mode":mask_mode,
            "financial_ids_enabled":financial_ids_enabled,
            "semantic_ai":text_analysis["semantic_ai"],
            "ner_ai":text_analysis.get("ner_ai",{}),
            "raw_transcript_included":bool(include_raw),
            "asr_privacy_recovery":privacy_recovery,
            "audio_guard_markers":[{"type":g["type"],"start":g["start"],"end":g["end"],"replacement":g["replacement"]} for g in audio_guards],
            "correction_resolution":{
                "count":len(correction_audit),
                "events":correction_audit,
                "policy":"later committed value masked; superseded mistaken value left unmasked only when same-field repair evidence is strong",
            },
        },
        "confidence":{
            "overall_trust":_trust(asr["confidence"],acoustic["quality"]["score"],text_analysis["intent"]["confidence"],text_analysis["pii_mean_confidence"]),
            "asr":asr["confidence"],
            "pii_mean":text_analysis["pii_mean_confidence"],
            "interpretation":"Composite triage score; retain component confidences for audit.",
        },
    }
    timing["total"]=(time.perf_counter()-total0)*1000
    result["performance"]={k:round(v,3) for k,v in timing.items()}
    result["performance"]["audio_duration_s"]=round(len(audio)/float(sr),3)
    result["performance"]["real_time_factor"]=round((timing["total"]/1000)/(len(audio)/float(sr)),4) if len(audio) else None
    if SETTINGS.privacy_debug_artifacts:
        try:
            debug_entities=(list(internal_privacy.get("pii",[]))+list(internal_privacy.get("financial_ids",[]))+list(internal_privacy.get("profanity",[])))
            result["privacy"]["debug_bundle"]=write_privacy_debug_bundle(
                str(asr.get("text","") or ""),
                str(text_analysis.get("safe_text","") or ""),
                debug_entities,
                privacy_recovery,
                base_dir=SETTINGS.privacy_debug_dir,
                ttl_hours=SETTINGS.privacy_debug_ttl_hours,
                include_raw=bool(SETTINGS.privacy_debug_raw),
            )
        except Exception as exc:
            result["privacy"]["debug_bundle"]={"enabled":True,"error":f"{type(exc).__name__}: {exc}"}
    else:
        result["privacy"]["debug_bundle"]={"enabled":False}
    if include_raw:
        result["privacy"]["warning"]="Raw transcript requested: output may contain PII/profanity. Do not use this mode for normal UI/logging."
    return result
