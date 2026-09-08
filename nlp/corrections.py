"""Correction-aware privacy resolution for spoken self-repairs.

This module is intentionally conservative. It does not infer a correction merely
because two values of the same type occur near each other. A later value replaces an
earlier value only when all of the following hold:

* both values belong to the same privacy type and same local field episode;
* the first value is explicitly/strongly owned by the speaker;
* no new field label or parallel-role cue (alternate/secondary/primary/etc.) appears;
* the values are different; and
* speech timing or a repair-style textual boundary supports a restart.

The resolver returns privacy-safe audit metadata only; it never exposes either value.
"""
from __future__ import annotations

import re
from typing import Iterable, Callable

from nlp.token_alignment import global_word_map

_CORRECTABLE_TYPES={
    "NAME","PHONE","EMAIL","PAN","AADHAAR","IFSC","UPI","CARD","ACCOUNT_NUMBER",
    "OTP","CVV","PINCODE","DOB","PASSPORT","VOTER_ID","DRIVING_LICENSE",
    "ADDRESS","PASSWORD","USERNAME","API_KEY","AUTH_TOKEN",
}

# A later explicitly distinct field is not a self-correction.
_NEW_FIELD=re.compile(
    r"(?i)\b(?:(?:my|your|the)\s+)?(?:(?:alternate|secondary|backup|other|primary|registered|work|personal)\s+)?"
    r"(?:name|phone|mobile|contact|email|e-mail|account|ifsc|pan|upi|aadhaar|aadhar|otp|cvv|cvc|"
    r"pin\s*code|pincode|postal\s+code|dob|date\s+of\s+birth|passport|voter\s+id|driving\s+licen[cs]e|"
    r"card|address|password|username|api\s*key|token|transaction|complaint|order|reference)\b"
)
_PARALLEL_ROLE=re.compile(
    r"(?i)\b(?:alternate|secondary|backup|other|primary|work|personal|old|previous|former|first|second)\b"
)
_LIST_JOIN=re.compile(r"(?i)\b(?:and|or|plus|also)\b")
_REPAIR_FILLER=re.compile(
    r"(?ix)^\s*[,.…;:\-–—]*\s*(?:(?:uh+|um+|erm+|er+|hmm+|mm+|ah+|wait|no|rather|actually|"
    r"i\s+mean|make\s+that|let\s+me\s+restart|start\s+again)\s*[,.…;:\-–—]*\s*)?$"
)
_PUNCT_ONLY=re.compile(r"^[\s,.…;:\-–—]+$")

# v5.1 hardening: a correction decision is allowed to *unmask* the earlier value,
# therefore timing evidence must be stronger than ordinary masking evidence. If
# alignment or speaker attribution is weak, keep both values protected.
_MIN_CORRECTION_ALIGNMENT_CONF=0.68


def _speaker_set(x: dict) -> set[str]:
    return {str(v) for v in (x.get("speakers") or []) if v not in (None,"","UNKNOWN")}


def _timing_evidence_safe(a: dict, b: dict) -> bool:
    """Return True only when timing is reliable enough to expose a superseded value.

    Text masking itself never depends on this gate; failing it simply leaves both
    accepted PII values masked. When diarization is available, cross-speaker pairs are
    never collapsed into a self-correction.
    """
    for x in (a,b):
        try:
            conf=float(x.get("alignment_confidence",0.0) or 0.0)
        except Exception:
            return False
        if conf < _MIN_CORRECTION_ALIGNMENT_CONF:
            return False
        if not x.get("token_ids"):
            return False
        if str(x.get("alignment_method","")) not in {"whisper_word_tokens","late_word_alignment","mixed"}:
            return False
    sa,sb=_speaker_set(a),_speaker_set(b)
    if sa and sb and sa.isdisjoint(sb):
        return False
    return True


def _superseded_value_reused(items: list[dict], ia: int, ib: int, a: dict) -> bool:
    """Protect a superseded value if it is independently owned elsewhere.

    A mistaken value can itself be a real old/secondary identifier. If the same
    normalized value appears in another strongly-owned occurrence, v5.1 refuses to
    expose it just because one local episode looked like a correction.
    """
    kind=str(a.get("type","")); nv=_norm(kind,a.get("value",""))
    if not nv:
        return False
    for j,x in enumerate(items):
        if j in {ia,ib} or x.get("type")!=kind:
            continue
        if int(x.get("ownership_strength",0) or 0) < 3:
            continue
        if _norm(kind,x.get("value",""))==nv:
            return True
    return False


def _norm(kind: str, value: str) -> str:
    v=str(value or "").casefold().strip()
    if kind in {"PHONE","AADHAAR","CARD","ACCOUNT_NUMBER","OTP","CVV","PINCODE"}:
        d=re.sub(r"\D","",v)
        return d or v
    if kind in {"PAN","IFSC","PASSPORT","VOTER_ID","DRIVING_LICENSE"}:
        return re.sub(r"[^a-z0-9]","",v)
    if kind in {"EMAIL","UPI","USERNAME","API_KEY","AUTH_TOKEN","PASSWORD"}:
        return re.sub(r"\s+","",v)
    return re.sub(r"\s+"," ",re.sub(r"[^\w@.+#'\- ]"," ",v)).strip()


def _char_bridge(text: str, a: dict, b: dict) -> str:
    return text[int(a.get("end",0)):int(b.get("start",0))]


def _episode_ok(text: str, a: dict, b: dict) -> bool:
    if a.get("type")!=b.get("type") or a.get("type") not in _CORRECTABLE_TYPES:
        return False
    if int(b.get("start",0)) <= int(a.get("end",0)):
        return False
    if int(b.get("start",0))-int(a.get("end",0)) > 72:
        return False
    if int(a.get("ownership_strength",0)) < 3:
        return False
    if _norm(a.get("type",""),a.get("value","")) == _norm(b.get("type",""),b.get("value","")):
        return False
    bridge=_char_bridge(text,a,b)
    if _NEW_FIELD.search(bridge) or _PARALLEL_ROLE.search(bridge) or _LIST_JOIN.search(bridge):
        return False
    # Long lexical material means this is almost certainly a new statement, not a restart.
    words=re.findall(r"[A-Za-z]+",bridge)
    if len(words)>3 and not _REPAIR_FILLER.match(bridge):
        return False
    return True


def _text_repair_signal(bridge: str) -> bool:
    # Text-only mode is intentionally much stricter than audio mode. A bare comma or
    # whitespace can be a legitimate list of two private values, so it must NOT make
    # us expose the first one. Keyword-free comma/pause repairs are resolved later
    # from Whisper timing. Text alone resolves only an unmistakable restart/filler or
    # an ellipsis-style break.
    if _REPAIR_FILLER.match(bridge) and re.search(r"\b(?:uh|um|erm|wait|no|rather|actually|mean|restart)\b",bridge,re.I):
        return True
    return bool(re.fullmatch(r"\s*(?:\.\.\.|…)+\s*",bridge))


def resolve_text_corrections(text: str, entities: Iterable[dict]) -> tuple[list[dict],list[dict]]:
    """Conservatively resolve explicit punctuation/filler self-repairs.

    Returns (kept_entities, safe_audit). The earlier superseded value is removed from
    masking; only the later committed value remains protected.
    """
    items=sorted((dict(x) for x in entities),key=lambda x:(int(x.get("start",0)),int(x.get("end",0))))
    removed=set(); audit=[]
    by_type={}
    for idx,x in enumerate(items): by_type.setdefault(x.get("type"),[]).append((idx,x))
    for kind,group in by_type.items():
        for (ia,a),(ib,b) in zip(group,group[1:]):
            if ia in removed or not _episode_ok(text,a,b):
                continue
            # Human names are too semantically open-ended for punctuation alone: a
            # comma can introduce a location/title. Spoken NAME correction therefore
            # requires Whisper timing and is resolved only in resolve_timed_corrections.
            if kind=="NAME":
                continue
            bridge=_char_bridge(text,a,b)
            if not _text_repair_signal(bridge):
                continue
            removed.add(ia)
            b["correction_authoritative"]=True
            b["evidence"]=str(b.get("evidence",""))+"+self_correction_commit"
            audit.append({
                "type":kind,"superseded_start":int(a.get("start",0)),"superseded_end":int(a.get("end",0)),
                "final_start":int(b.get("start",0)),"final_end":int(b.get("end",0)),
                "method":"text_repair_boundary","wrong_value_masked":False,"final_value_masked":True,
            })
    return [x for i,x in enumerate(items) if i not in removed],audit


def _timing_gap(a: dict, b: dict) -> float | None:
    try:
        ae=float(a.get("time_end")); bs=float(b.get("time_start"))
    except Exception:
        return None
    if ae<0 or bs<0: return None
    return bs-ae


def _timed_signal(text: str, a: dict, b: dict) -> tuple[bool,str,float | None]:
    bridge=_char_bridge(text,a,b)
    gap=_timing_gap(a,b)
    if gap is None:
        return False,"no_timing",None
    # Negative overlap can occur from loose word alignment; never infer correction then.
    if gap < -0.08 or gap > 2.2:
        return False,"gap_out_of_range",gap
    if _text_repair_signal(bridge):
        return (gap>=0.10),"timed_punctuation_restart",gap
    # This is the no-keyword path requested for natural speech. A plain break is only
    # accepted when there is a meaningful acoustic pause and no second-field/list cue.
    if not bridge.strip() and gap>=0.42:
        return True,"silent_restart",gap
    if re.fullmatch(r"\s*,?\s*",bridge) and gap>=0.32:
        return True,"pause_restart",gap
    return False,"insufficient_restart_evidence",gap


def _timing_for_span(word_map: list[dict], start: int, end: int) -> dict:
    hits=[w for w in word_map if int(w.get("char_start",0)) < end and start < int(w.get("char_end",0))]
    if not hits:
        return {"token_ids":[],"alignment_method":"unresolved","alignment_confidence":0.0}
    return {
        "token_ids":[w.get("token_id") for w in hits],
        "time_start":min(float(w.get("time_start",0.0)) for w in hits),
        "time_end":max(float(w.get("time_end",0.0)) for w in hits),
        "alignment_method":"whisper_word_tokens",
        "alignment_confidence":round(sum(float(w.get("confidence",0.0)) for w in hits)/len(hits),4),
        "speakers":sorted({str(w.get("speaker")) for w in hits if w.get("speaker") not in (None,"","UNKNOWN")}),
    }


def _discover_timed_replacement(
    text: str, a: dict, word_map: list[dict],
    candidate_validator: Callable[[str,str], list[dict]] | None,
) -> dict | None:
    """Find a strict same-type value immediately after an owned value.

    This is how v5 catches a correction that the normal detector did not accept due to
    punctuation/context reset (for example `My OTP is 638291 ... 638292`). The
    validator is supplied by the caller and must use the normal privacy validators.
    Nothing is promoted unless timing later proves a restart.
    """
    if candidate_validator is None or int(a.get("ownership_strength",0))<3:
        return None
    start=int(a.get("end",0)); end=min(len(text),start+78)
    tail=text[start:end]
    # Never search through a clearly separate field/list role.
    hard=_NEW_FIELD.search(tail)
    if hard: tail=tail[:hard.start()]
    if _LIST_JOIN.search(tail):
        # We still allow a candidate before the conjunction, but never beyond it.
        m=_LIST_JOIN.search(tail); tail=tail[:m.start()]
    if not tail.strip():
        return None
    try:
        candidates=candidate_validator(str(a.get("type","")),tail) or []
    except Exception:
        return None
    for c in sorted(candidates,key=lambda x:(int(x.get("start",0)),int(x.get("end",0)))):
        gs=start+int(c.get("start",0)); ge=start+int(c.get("end",0))
        if ge<=gs or gs<start: continue
        b=dict(c); b["start"]=gs; b["end"]=ge; b["value"]=text[gs:ge]
        b.setdefault("ownership_strength",int(a.get("ownership_strength",0)))
        b["correction_discovered"]=True
        b["evidence"]=str(b.get("evidence",""))+"+correction_continuation_validation"
        b.update(_timing_for_span(word_map,gs,ge))
        if _episode_ok(text,a,b):
            return b
    return None


def resolve_timed_corrections(
    text: str, entities: Iterable[dict], asr: dict,
    candidate_validator: Callable[[str,str], list[dict]] | None=None,
) -> tuple[list[dict],list[dict]]:
    """Use Whisper timing to resolve keyword-free spoken corrections.

    The normal path chooses between two already-accepted values. When the second value
    was lost only because ASR punctuation reset its ownership context, an optional
    validator may rediscover it using the *same normal type validators*. It is promoted
    only after the same timing/episode checks pass, so this does not become a generic
    high-recall detector.
    """
    items=sorted((dict(x) for x in entities),key=lambda x:(int(x.get("start",0)),int(x.get("end",0))))
    removed=set(); audit=[]
    groups={}
    for idx,x in enumerate(items): groups.setdefault(x.get("type"),[]).append((idx,x))
    for kind,group in groups.items():
        # Collapse correction chains left-to-right, so X -> Y -> Z leaves only Z.
        for pos in range(len(group)-1):
            ia,a=group[pos]; ib,b=group[pos+1]
            if ia in removed or not _episode_ok(text,a,b):
                continue
            if not _timing_evidence_safe(a,b):
                continue
            if _superseded_value_reused(items,ia,ib,a):
                continue
            ok,method,gap=_timed_signal(text,a,b)
            if not ok:
                continue
            removed.add(ia)
            b["correction_authoritative"]=True
            b["evidence"]=str(b.get("evidence",""))+"+timed_self_correction_commit"
            audit.append({
                "type":kind,"superseded_start":int(a.get("start",0)),"superseded_end":int(a.get("end",0)),
                "final_start":int(b.get("start",0)),"final_end":int(b.get("end",0)),
                "method":method,"pause_s":None if gap is None else round(float(gap),3),
                "wrong_value_masked":False,"final_value_masked":True,
            })
    # A second strict value may have been missed by ordinary context after ASR inserted
    # punctuation. Discover only immediately after strong owned values, then apply the
    # exact same restart/timing decision before replacing anything.
    word_map=global_word_map(asr)
    additions=[]
    for ia,a in enumerate(items):
        if ia in removed or a.get("type") not in _CORRECTABLE_TYPES:
            continue
        # If a later same-type accepted value already exists nearby, the normal pass
        # above handled it (or deliberately rejected it).
        later=[b for b in items if b.get("type")==a.get("type") and int(a.get("end",0)) < int(b.get("start",0)) <= int(a.get("end",0))+72]
        if later:
            continue
        b=_discover_timed_replacement(text,a,word_map,candidate_validator)
        if not b:
            continue
        if not _timing_evidence_safe(a,b):
            continue
        # There is no accepted b index yet, so only exclude the source occurrence.
        if _superseded_value_reused(items,ia,-1,a):
            continue
        ok,method,gap=_timed_signal(text,a,b)
        if not ok:
            continue
        removed.add(ia)
        b["correction_authoritative"]=True
        b["evidence"]=str(b.get("evidence",""))+"+timed_self_correction_commit"
        additions.append(b)
        audit.append({
            "type":a.get("type"),"superseded_start":int(a.get("start",0)),"superseded_end":int(a.get("end",0)),
            "final_start":int(b.get("start",0)),"final_end":int(b.get("end",0)),
            "method":method+"_rediscovered","pause_s":None if gap is None else round(float(gap),3),
            "wrong_value_masked":False,"final_value_masked":True,
        })
    kept=[x for i,x in enumerate(items) if i not in removed]+additions
    kept.sort(key=lambda x:(int(x.get("start",0)),int(x.get("end",0))))
    return kept,audit


# v5.2 partial self-correction.  A partial edit never becomes a generic short-number
# detector: it is only considered after an already accepted, strongly-owned numeric
# PII value and only for an explicit suffix edit with an exact replacement arity.
_PARTIAL_EDIT_TYPES={"PHONE","CARD","ACCOUNT_NUMBER","OTP","CVV","PINCODE","AADHAAR"}
_PARTIAL_CONTEXT_VETO=re.compile(
    r"(?i)\b(?:example|sample|documentation|docs?|manual|tutorial|training|test\s+(?:case|value|data)|"
    r"reference|ticket|order|transaction|complaint|experiment|invoice|page)\b"
)
_COUNT_WORD={"one":1,"two":2,"three":3,"four":4}
_DIGIT_WORD={"zero":"0","oh":"0","o":"0","one":"1","two":"2","three":"3","four":"4",
             "five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
_PARTIAL_EDIT_RE=re.compile(
    r"(?ix)\b(?:"
    r"(?:(?:change|replace|correct|make)\s+(?:the\s+)?)?"
    r"(?:last|final)\s+(?:(one|two|three|four|[1-4])\s+)?digit(?:s)?\s+"
    r"(?:is|are|to|with|as|should\s+be|must\s+be)\s+"
    r")"
    r"((?:[0-9](?:[\s-]*[0-9]){0,3})|"
    r"(?:(?:zero|oh|o|one|two|three|four|five|six|seven|eight|nine)"
    r"(?:[\s-]+(?:zero|oh|o|one|two|three|four|five|six|seven|eight|nine)){0,3}))\b"
)


def _replacement_digits(raw: str) -> str:
    raw=str(raw or "").strip().casefold()
    direct=re.sub(r"\D","",raw)
    if direct:
        return direct
    out=[]
    for tok in re.findall(r"[a-z]+",raw):
        if tok not in _DIGIT_WORD:
            return ""
        out.append(_DIGIT_WORD[tok])
    return "".join(out)


def _declared_digit_count(raw_count: str | None, replacement: str) -> int | None:
    if raw_count:
        key=str(raw_count).casefold()
        return int(key) if key.isdigit() else _COUNT_WORD.get(key)
    # Singular `digit` without a number is captured as an omitted count.  The regex
    # does not expose singular/plural directly, so only infer one when replacement is
    # exactly one digit. Multi-digit edits must state their arity explicitly.
    return 1 if len(replacement)==1 else None


def resolve_partial_corrections(
    text: str, entities: Iterable[dict], asr: dict,
) -> tuple[list[dict],list[dict]]:
    """Mask only the spoken replacement fragment for deterministic suffix edits.

    Example: `My phone is 9876543210. Last digit is 1.` can leave the superseded
    number visible while protecting only `1`, provided Whisper timing is strong and
    same-speaker. If timing is weak, both old value and replacement fragment stay
    masked. The function never reconstructs or stores the corrected full identifier.
    """
    items=sorted((dict(x) for x in entities),key=lambda x:(int(x.get("start",0)),int(x.get("end",0))))
    additions=[]; removed=set(); audit=[]
    word_map=global_word_map(asr)
    for ia,a in enumerate(items):
        kind=str(a.get("type",""))
        if kind not in _PARTIAL_EDIT_TYPES or int(a.get("ownership_strength",0) or 0)<3:
            continue
        base=_norm(kind,a.get("value",""))
        if not base.isdigit():
            continue
        search_start=int(a.get("end",0)); search_end=min(len(text),search_start+150)
        tail=text[search_start:search_end]
        # A new PII field or a public/example role ends the edit episode.
        hard=_NEW_FIELD.search(tail)
        if hard:
            tail=tail[:hard.start()]
        if _PARTIAL_CONTEXT_VETO.search(tail):
            continue
        m=_PARTIAL_EDIT_RE.search(tail)
        if not m:
            continue
        # Do not jump through a long unrelated sentence to reach the edit.
        pre=tail[:m.start()]
        if len(re.findall(r"[A-Za-z]+",pre))>6 or _PARALLEL_ROLE.search(pre) or _LIST_JOIN.search(pre):
            continue
        repl=_replacement_digits(m.group(2))
        count=_declared_digit_count(m.group(1),repl)
        if not repl or count is None or count!=len(repl) or not (1<=count<=4) or len(base)<count:
            continue
        rs=search_start+m.start(2); re_=search_start+m.end(2)
        b={
            "type":kind,"start":rs,"end":re_,"value":text[rs:re_],"confidence":0.995,
            "ownership_strength":4,"partial_correction":True,
            "evidence":"deterministic_suffix_partial_correction",
        }
        b.update(_timing_for_span(word_map,rs,re_))
        # The replacement itself is always protected once the explicit edit is proven.
        additions.append(b)
        safe_to_expose_old=_timing_evidence_safe(a,b) and not _superseded_value_reused(items,ia,-1,a)
        if safe_to_expose_old:
            removed.add(ia)
        audit.append({
            "type":kind,"superseded_start":int(a.get("start",0)),"superseded_end":int(a.get("end",0)),
            "final_start":rs,"final_end":re_,"method":"deterministic_suffix_partial_edit",
            "digit_count":count,"wrong_value_masked":not safe_to_expose_old,
            "final_value_masked":True,"partial_fragment_only":True,"reconstructed_value_stored":False,
        })
    kept=[x for i,x in enumerate(items) if i not in removed]+additions
    kept.sort(key=lambda x:(int(x.get("start",0)),int(x.get("end",0))))
    return kept,audit
